"""
run_multiseed_ablation.py — 独立跑多 seed × 多架构对比

通过 importlib 加载其组件。
前提：原文件的 ImprovedLongTermModel.forward 已加 prior_only 参数。

支持断点续跑：若 multiseed_<arch>_s<seed>.pt 已存在，跳过训练直接加载。
"""

import importlib.util
import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# =============================================================================
# 配置
# =============================================================================
ORIGINAL_PY = r'D:\cdomproject\transcode\long_term_cdom_model-all.py'
EXP_NAME    = 'd600_800'
SPLIT_MODE  = 'buoy'
ARCHS       = ['full', 'no_patch']
SEEDS       = [42, 2024, 7]
SEQ_LEN, FORECAST_DAYS = 730, 365
BATCH_SIZE  = 32
DROPOUT     = 0.3
MAX_GAP_DAYS = 30
STEP        = 30
EPOCHS      = 200
PATIENCE    = 50
USE_AMP     = False

# =============================================================================
# 加载原模块
# =============================================================================
print(f"加载原模块: {ORIGINAL_PY}")
spec = importlib.util.spec_from_file_location("orig_module", ORIGINAL_PY)
orig = importlib.util.module_from_spec(spec)
sys.modules["orig_module"] = orig
spec.loader.exec_module(orig)

EXPERIMENTS             = orig.EXPERIMENTS
LongTermDataProcessor   = orig.LongTermDataProcessor
LongTermTrainer         = orig.LongTermTrainer
build_model_for_arch    = orig.build_model_for_arch
_build_local_datasets   = orig._build_local_datasets
_do_split               = orig._do_split
_tc_scalar              = orig._tc_scalar
_safe_savefig           = orig._safe_savefig
set_seed                = orig.set_seed
CACHE_ROOT              = orig.CACHE_ROOT

orig.SPLIT_MODE = SPLIT_MODE

LR           = orig.LR
WEIGHT_DECAY = orig.WEIGHT_DECAY
WARMUP       = orig.WARMUP
GRAD_CLIP    = orig.GRAD_CLIP


# =============================================================================
# 单次训练/加载 + 预测
# =============================================================================
def train_and_predict(arch, seed, tr_l, va_l, te_l,
                      mu_va, sg_va, mu_te, sg_te,
                      n_features, device, save_dir):
    print(f"\n--- arch={arch}, seed={seed} ---")
    set_seed(seed)

    model = build_model_for_arch(
        arch, input_dim=n_features, seq_len=SEQ_LEN,
        forecast_days=FORECAST_DAYS, dropout=DROPOUT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {n_params:,}")

    trainer = LongTermTrainer(model)
    model_path = os.path.join(save_dir, f'multiseed_{arch}_s{seed}.pt')

    t0 = time.time()
    if os.path.exists(model_path):
        # ★ 断点续跑：加载已训练权重
        print(f"  [断点续跑] 加载已训练模型: {model_path}")
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        n_epochs = -1
    else:
        print(f"  [训练] 无已存在模型，开始训练")
        hist = trainer.train(
            tr_l, va_l, mu_va, sg_va,
            epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
            patience=PATIENCE, warmup=WARMUP,
            use_amp=USE_AMP,
            model_path=model_path,
            verbose=False)
        n_epochs = len(hist['train_loss'])

    preds, targets = trainer.predict(te_l, mu_te, sg_te)
    rmse = float(np.sqrt(mean_squared_error(targets.flatten(),
                                            preds.flatten())))
    mae  = float(mean_absolute_error(targets.flatten(), preds.flatten()))
    tc   = _tc_scalar(preds, targets)
    elapsed = time.time() - t0

    print(f"  完成: RMSE={rmse:.4f}  MAE={mae:.4f}  TC={tc:+.4f}  "
          f"epochs={n_epochs}  time={elapsed:.1f}s")

    try:
        trainer.model.to('cpu')
    except Exception:
        pass
    del model, trainer
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'arch': arch, 'seed': seed, 'n_params': n_params,
        'rmse': rmse, 'mae': mae, 'tc': tc,
        'n_epochs': n_epochs, 'time_s': round(elapsed, 1),
        'preds': preds, 'targets': targets,
    }


# =============================================================================
# 主流程
# =============================================================================
def main():
    exp = EXPERIMENTS[EXP_NAME]
    save_dir = str(CACHE_ROOT)
    print(f"\n===== Multi-seed ablation: {EXP_NAME} (split={SPLIT_MODE}) =====")
    print(f"架构: {ARCHS}")
    print(f"种子: {SEEDS}")

    # ---------- 1. 数据 ----------
    proc = LongTermDataProcessor(exp, seq_len=SEQ_LEN,
                                 forecast_days=FORECAST_DAYS, step=STEP)
    if not proc.load_data(use_cache=True):
        print("数据加载失败"); return
    seqs, tars, meta = proc.prepare_sequences(
        min_sequence_length=SEQ_LEN + FORECAST_DAYS,
        apply_smoothing=False, max_gap_days=MAX_GAP_DAYS,
        interpolate_limit_days=MAX_GAP_DAYS)
    if seqs is None:
        return

    n_samples, _, n_features = seqs.shape
    order = sorted(range(n_samples), key=lambda i: meta[i]['start_date'])
    seqs, tars, meta = seqs[order], tars[order], [meta[i] for i in order]

    gap_samples = max(0, (SEQ_LEN // max(STEP, 1)) - 1)
    (Xtr, ytr, _, Xva, yva, _, Xte, yte, te_meta) = _do_split(
        seqs, tars, meta, gap=gap_samples)
    print(f"样本: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}")

    # ---------- 2. 标准化 ----------
    tr_l, va_l, te_l, mu_va, sg_va, mu_te, sg_te = _build_local_datasets(
        Xtr, ytr, Xva, yva, Xte, yte, BATCH_SIZE)

    # ---------- 3. 遍历 ----------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = defaultdict(list)
    t_start = time.time()

    for arch in ARCHS:
        for seed in SEEDS:
            r = train_and_predict(
                arch, seed, tr_l, va_l, te_l,
                mu_va, sg_va, mu_te, sg_te,
                n_features, device, save_dir)
            all_results[arch].append(r)

    total_time = time.time() - t_start
    print(f"\n总耗时: {total_time/60:.2f} 分钟")

    # ---------- 4. 汇总（保守写法） ----------
    print(f"\n===== 汇总 =====")
    summary = {}
    for arch in ARCHS:
        rs = all_results.get(arch, [])
        print(f"  [debug] arch={arch}, n_results={len(rs)}")
        if len(rs) == 0:
            print(f"  [警告] {arch} 无结果，跳过")
            continue

        tc_list    = [r['tc']   for r in rs]
        rmse_list  = [r['rmse'] for r in rs]
        mae_list   = [r['mae']  for r in rs]
        tcs   = np.array(tc_list,   dtype=float)
        rmses = np.array(rmse_list, dtype=float)
        maes  = np.array(mae_list,  dtype=float)

        ens_preds = np.mean([r['preds'] for r in rs], axis=0)
        targets   = rs[0]['targets']
        ens_rmse  = float(np.sqrt(mean_squared_error(
            targets.flatten(), ens_preds.flatten())))
        ens_mae   = float(mean_absolute_error(
            targets.flatten(), ens_preds.flatten()))
        ens_tc    = _tc_scalar(ens_preds, targets)

        per_seed = []
        for r in rs:
            per_seed.append({
                'seed': r['seed'], 'rmse': r['rmse'],
                'mae': r['mae'], 'tc': r['tc'],
                'epochs': r['n_epochs'],
            })

        summary[arch] = {
            'seeds': list(SEEDS),
            'per_seed': per_seed,
            'tc_mean':   float(tcs.mean()),
            'tc_std':    float(tcs.std()),
            'tc_min':    float(tcs.min()),
            'tc_max':    float(tcs.max()),
            'rmse_mean': float(rmses.mean()),
            'rmse_std':  float(rmses.std()),
            'mae_mean':  float(maes.mean()),
            'ens_rmse':  ens_rmse,
            'ens_mae':   ens_mae,
            'ens_tc':    ens_tc,
            'n_params':  rs[0]['n_params'],
        }

        s = summary[arch]
        seed_str = ' | '.join(
            f"seed={p['seed']}:TC={p['tc']:+.3f}" for p in per_seed)
        print(f"\n[{arch}]  params={s['n_params']:,}")
        print(f"  {seed_str}")
        print(f"  TC   = {s['tc_mean']:+.4f} ± {s['tc_std']:.4f}  "
              f"(min={s['tc_min']:+.3f}, max={s['tc_max']:+.3f})")
        print(f"  RMSE = {s['rmse_mean']:.4f} ± {s['rmse_std']:.4f}")
        print(f"  3-seed 集成: RMSE={s['ens_rmse']:.4f}  "
              f"MAE={s['ens_mae']:.4f}  TC={s['ens_tc']:+.4f}")

    print(f"\n[debug] summary keys = {list(summary.keys())}")

    # ---------- 5. Welch t-test ----------
    t_stat = p_val = None
    if len(ARCHS) == 2 and all(a in summary for a in ARCHS):
        a, b = ARCHS
        tcs_a = np.array([r['tc'] for r in all_results[a]], dtype=float)
        tcs_b = np.array([r['tc'] for r in all_results[b]], dtype=float)
        t_stat, p_val = scipy_stats.ttest_ind(tcs_a, tcs_b, equal_var=False)
        print(f"\n[{a} vs {b}] Welch t-test on TC:")
        print(f"  {a}: mean={tcs_a.mean():+.4f}  n={len(tcs_a)}")
        print(f"  {b}: mean={tcs_b.mean():+.4f}  n={len(tcs_b)}")
        print(f"  t={t_stat:+.3f}  p={p_val:.4f}  "
              f"{'显著 (p<0.05)' if p_val < 0.05 else '不显著'}")

    # ---------- 6. 可视化 ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) 逐 seed TC 散点
    ax = axes[0]
    for i, arch in enumerate(ARCHS):
        if arch not in summary:
            continue
        tcs_i = [r['tc'] for r in all_results[arch]]
        ax.scatter([i]*len(tcs_i), tcs_i, s=120,
                   label=f'{arch} (mean={np.mean(tcs_i):+.3f})')
        ax.axhline(np.mean(tcs_i), color=f'C{i}', linestyle='--', alpha=0.6)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(range(len(ARCHS)))
    ax.set_xticklabels(ARCHS)
    ax.set_ylabel('TrendCorr')
    ax.set_title('(a) Per-seed TC')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # (b) 测试集第 1 个样本预测
    ax = axes[1]
    targets = all_results[ARCHS[0]][0]['targets']
    ax.plot(targets[0], 'k-', label='True', linewidth=2, alpha=0.8)
    for i, arch in enumerate(ARCHS):
        for j, r in enumerate(all_results[arch]):
            ax.plot(r['preds'][0], color=f'C{i}', alpha=0.4,
                    label=f'{arch} s{r["seed"]}' if j == 0 else None)
    ax.set_xlabel('Forecast Day'); ax.set_ylabel('CDOM')
    ax.set_title('(b) Sample 1 predictions')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (c) 均值 ± std
    ax = axes[2]
    means = [summary[a]['tc_mean'] for a in ARCHS if a in summary]
    stds  = [summary[a]['tc_std']  for a in ARCHS if a in summary]
    labels = [a for a in ARCHS if a in summary]
    bars = ax.bar(labels, means, yerr=stds, capsize=8,
                  color=['C0', 'C1'][:len(labels)], alpha=0.8)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + s + 0.02 if s > 0 else m + 0.02,
                f'{m:+.3f}\n±{s:.3f}', ha='center', fontsize=9)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('TrendCorr')
    ax.set_title('(c) Mean ± std across seeds')
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(
        f"{EXP_NAME} ({SPLIT_MODE}): {' vs '.join(ARCHS)} over "
        f"{len(SEEDS)} seeds", fontsize=13)
    plt.tight_layout()
    save_path = CACHE_ROOT / f'multiseed_{EXP_NAME}_{SPLIT_MODE}.png'
    actual = _safe_savefig(fig, save_path, dpi=150)
    plt.close(fig)
    print(f"\n对比图已保存: {actual}")

    # ---------- 7. 保存 JSON ----------
    out = {
        'exp': EXP_NAME, 'split': SPLIT_MODE,
        'archs': ARCHS, 'seeds': SEEDS,
        'summary': summary,
        'total_time_min': round(total_time/60, 2),
    }
    if t_stat is not None:
        out['stat_test'] = {
            'arch_a': ARCHS[0], 'arch_b': ARCHS[1],
            't_stat': float(t_stat), 'p_value': float(p_val),
        }
    json_path = CACHE_ROOT / f'multiseed_{EXP_NAME}_{SPLIT_MODE}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {json_path}")

    return out


if __name__ == '__main__':
    main()