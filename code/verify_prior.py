"""
verify_prior.py — 独立验证 DTRN 内部先验 vs 独立谐波基线

通过 importlib 按路径加载其组件。
运行前提：原文件已运行过 experiments，生成 exp_<exp>_<split>_full.pt。

用法（改顶部常量即可）：
    EXP_NAME = 'd600_800'
    SPLIT_MODE = 'buoy'
"""

import importlib.util
import sys
import os
import json
import numpy as np
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# 配置
# =============================================================================
ORIGINAL_PY = r'D:\cdomproject\transcode\long_term_cdom_model-all.py'
EXP_NAME    = 'd600_800'      # 要验证的深度
SPLIT_MODE  = 'buoy'          # 'buoy' 或 'chrono'
SEQ_LEN, FORECAST_DAYS = 730, 365
BATCH_SIZE = 32
DROPOUT = 0.3
MAX_GAP_DAYS = 30
STEP = 30

# =============================================================================
# 加载原模块
# =============================================================================
print(f"加载原模块: {ORIGINAL_PY}")
spec = importlib.util.spec_from_file_location("orig_module", ORIGINAL_PY)
orig = importlib.util.module_from_spec(spec)
sys.modules["orig_module"] = orig   # dataclass 需要
spec.loader.exec_module(orig)

# 提取需要复用的符号
ImprovedLongTermModel  = orig.ImprovedLongTermModel
LongTermDataProcessor  = orig.LongTermDataProcessor
LongTermTrainer        = orig.LongTermTrainer
EXPERIMENTS            = orig.EXPERIMENTS
_baseline_harmonic     = orig._baseline_harmonic
_build_local_datasets  = orig._build_local_datasets
_do_split              = orig._do_split
_tc_scalar             = orig._tc_scalar
_safe_savefig          = orig._safe_savefig
CACHE_ROOT             = orig.CACHE_ROOT

# 覆盖原模块的全局 SPLIT_MODE，让 _do_split 走 buoy
orig.SPLIT_MODE = SPLIT_MODE

from sklearn.metrics import mean_squared_error, mean_absolute_error


# =============================================================================
# 门控参数诊断
# =============================================================================
def inspect_gate(model):
    w_p = float(torch.sigmoid(model.prior_logit).item())
    w_d = float(torch.sigmoid(model.deep_logit).item())
    s = w_p + w_d + 1e-8
    print(f"\n[门控参数]")
    print(f"  prior_logit = {model.prior_logit.item():+.4f}  "
          f"-> w_prior = {w_p:.4f}")
    print(f"  deep_logit  = {model.deep_logit.item():+.4f}  "
          f"-> w_deep  = {w_d:.4f}")
    print(f"  归一化：w_prior_n = {w_p/s:.4f},  w_deep_n = {w_d/s:.4f}")
    return {'w_prior': w_p, 'w_deep': w_d}


# =============================================================================
# 手动 forward 三种模式
# =============================================================================
def _forward_all(model, loader, mode, device):
    ps = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if mode == 'prior_only':
                pred = model(x, prior_only=True)
            else:
                pred = model(x)
            pred = pred.squeeze(-1) if pred.dim() == 3 else pred
            ps.append(pred.cpu().numpy())
    return np.vstack(ps)


# =============================================================================
# 主验证流程
# =============================================================================
def verify(exp_name, split_mode):
    exp = EXPERIMENTS[exp_name]
    print(f"\n===== 先验验证: {exp_name} (split={split_mode}) =====")

    # 1. 数据
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

    # 2. 标准化
    tr_l, va_l, te_l, mu_va, sg_va, mu_te, sg_te = _build_local_datasets(
        Xtr, ytr, Xva, yva, Xte, yte, BATCH_SIZE)

    # 3. 找模型文件
    candidates = [
        f'exp_{exp_name}_{split_mode}_full.pt',
        f'exp_{exp_name}_{split_mode}_no_patch.pt',
        f'{exp.out_prefix}.pt',
    ]
    model_path = None
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            break
    if model_path is None:
        print(f"[错误] 找不到模型文件，尝试过:")
        for c in candidates:
            print(f"    {c}")
        print(f"[提示] 请先跑 RUN_MODE='experiments', ONLY_EXP='full'")
        return
    print(f"加载模型: {model_path}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ImprovedLongTermModel(input_dim=n_features, seq_len=SEQ_LEN,
                                  forecast_days=FORECAST_DAYS, dropout=DROPOUT)
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    gate = inspect_gate(model)

    # 4. 三种预测
    preds_full_norm  = _forward_all(model, te_l, 'full', device)
    preds_prior_norm = _forward_all(model, te_l, 'prior_only', device)

    preds_full_orig  = np.clip(preds_full_norm  * sg_te + mu_te, 0.0, 20.0)
    preds_prior_orig = np.clip(preds_prior_norm * sg_te + mu_te, 0.0, 20.0)
    preds_harm_orig  = _baseline_harmonic(Xte, FORECAST_DAYS)
    targets_orig     = yte

    # 5. 评估
    def ev(p, t, name):
        rmse = float(np.sqrt(mean_squared_error(t.flatten(), p.flatten())))
        mae  = float(mean_absolute_error(t.flatten(), p.flatten()))
        tc   = _tc_scalar(p, t)
        print(f"  {name:24s} RMSE={rmse:.4f}  MAE={mae:.4f}  TC={tc:+.4f}")
        return {'rmse': rmse, 'mae': mae, 'tc': tc}

    print(f"\n[三种预测对比]")
    res_full  = ev(preds_full_orig,  targets_orig, 'DTRN 完整 (full)')
    res_prior = ev(preds_prior_orig, targets_orig, 'DTRN 仅先验')
    res_harm  = ev(preds_harm_orig,  targets_orig, '独立谐波基线')

    # 6. 关键判断
    gap_pvsh = abs(res_prior['tc'] - res_harm['tc'])
    gap_fvsp = res_full['tc']  - res_prior['tc']

    print(f"\n[关键判断]")
    if gap_pvsh < 0.05:
        verdict = "先验实现正确"
        print(f"  ✅ |TC(prior_only) - TC(harmonic)| = {gap_pvsh:.4f} < 0.05")
        print(f"     -> 先验分支实现正确；full 模型变差源于深度分支污染融合。")
    elif gap_pvsh < 0.15:
        verdict = "先验实现部分偏离"
        print(f"  ⚠️  |TC(prior_only) - TC(harmonic)| = {gap_pvsh:.4f} 在 [0.05, 0.15)")
        print(f"     -> 先验实现有差异，但未到严重 bug 程度。需逐项排查差异来源。")
    else:
        verdict = "先验实现有 bug"
        print(f"  ❌ |TC(prior_only) - TC(harmonic)| = {gap_pvsh:.4f} >= 0.15")
        print(f"     -> 先验分支实现与独立谐波不一致，需排查代码。")

    print(f"  full - prior_only = {gap_fvsp:+.4f} "
          f"({'深度分支有害' if gap_fvsp < 0 else '深度分支有益'})")

    # 7. 逐样本差异
    diff = preds_prior_orig - preds_harm_orig
    print(f"\n[prior_only vs harmonic 逐样本差异]")
    print(f"  bias     = {diff.mean():+.6f}")
    print(f"  std      = {diff.std():.6f}")
    print(f"  prior_only 整体: mean={preds_prior_orig.mean():.4f}, "
          f"std={preds_prior_orig.std():.4f}")
    print(f"  harmonic 整体:   mean={preds_harm_orig.mean():.4f}, "
          f"std={preds_harm_orig.std():.4f}")
    corrs = []
    for i in range(preds_prior_orig.shape[0]):
        if (np.std(preds_prior_orig[i]) > 1e-6 and
                np.std(preds_harm_orig[i]) > 1e-6):
            r = np.corrcoef(preds_prior_orig[i], preds_harm_orig[i])[0, 1]
            if np.isfinite(r):
                corrs.append(r)
    if corrs:
        print(f"  逐样本曲线相关: mean={np.mean(corrs):.4f}, "
              f"median={np.median(corrs):.4f}, n={len(corrs)}")

    # 8. 诊断图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i in range(min(3, preds_full_orig.shape[0])):
        ax = axes[i]
        ax.plot(targets_orig[i], 'k-', label='True', linewidth=2, alpha=0.8)
        ax.plot(preds_full_orig[i],  'r--', label='DTRN full',  alpha=0.8)
        ax.plot(preds_prior_orig[i], 'b--', label='DTRN prior only', alpha=0.8)
        ax.plot(preds_harm_orig[i],  'g:',  label='Harmonic baseline',
                linewidth=2, alpha=0.8)
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel('Forecast Day'); ax.set_ylabel('CDOM')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(
        f"Prior verification: {exp_name} ({split_mode})\n"
        f"TC: full={res_full['tc']:+.3f}  "
        f"prior_only={res_prior['tc']:+.3f}  harmonic={res_harm['tc']:+.3f}",
        fontsize=12)
    plt.tight_layout()
    save_path = CACHE_ROOT / f'verify_prior_{exp_name}_{split_mode}.png'
    actual = _safe_savefig(fig, save_path, dpi=150)
    plt.close(fig)
    print(f"\n诊断图已保存: {actual}")

    # 9. 保存 JSON
    out = {
        'exp': exp_name, 'split': split_mode,
        'gate': gate,
        'full':  res_full,
        'prior': res_prior,
        'harmonic': res_harm,
        'gap_prior_vs_harm': gap_pvsh,
        'gap_full_vs_prior': gap_fvsp,
        'verdict': verdict,
    }
    json_path = CACHE_ROOT / f'verify_prior_{exp_name}_{split_mode}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {json_path}")

    return out


if __name__ == '__main__':
    verify(EXP_NAME, SPLIT_MODE)