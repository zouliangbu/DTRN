"""
long_term_cdom_model-all.py

多深度 CDOM 长期预测：对比 + 消融 + Bootstrap CI + 纬度带分层。

本次修复：
  1. train() 中 backward() 重复调用导致的 RuntimeError
  2. _tc_scalar / _bootstrap_metric 在常数样本上产生 nan，污染 CI
  3. metrics_from 的纬度带部分加调试信息，便于定位 lat 缺失
"""

import os
import re
import csv
import json
import math
import time
import random
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats, signal
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ.setdefault('MKL_THREADING_LAYER', 'GNU')


# =============================================================================
# ★★★ 用户配置区 ★★★
# =============================================================================

# ---------- 路径 ----------
DATA_ROOT  = Path(r'D:\cdomproject\temporal_depth_data')
CACHE_ROOT = Path(r'D:\cdomproject\transcode')

# ---------- 运行模式 ----------
# 'train'       : 单深度训练（画图 + npz）
# 'experiments' : 对比 + 消融（CSV + 报告）
# 'summary'     : 只做跨深度汇总
# 'seasonal'    : 只画季节诊断图 + 重生成报告（不重训）
# 'debug_train' : 单深度两模型 verbose 训练诊断
RUN_MODE = 'train'

# 目标实验（key 必须出现在 build_experiments() 里）

RUN_EXPS = ['d600_800']

# ---------- 切分模式 ----------
# 'chrono' : 按起始日期时间序切分（时序外推）
# 'buoy'   : 按浮标随机切分（空间泛化，训练/测试浮标不重叠）
SPLIT_MODE = 'buoy'

# ---------- 数据 & 特征 ----------
CACHE_VERSION = 'v3_2025-01'
SEQ_LEN       = 730
FORECAST_DAYS = 365
MAX_GAP_DAYS  = 30
TOP_K_LAYERS  = 10
MAX_LAYERS    = 10

# ---------- 训练超参 ----------
STEP          = 30
BATCH_SIZE    = 32
EPOCHS_TRAIN  = 200
EPOCHS_EXP    = 100
LR            = 5e-4
DROPOUT       = 0.3
PATIENCE_TRAIN = 40
PATIENCE_EXP   = 20

SEEDS   = [42, 2024, 7]

WEIGHT_DECAY   = 5e-2
WARMUP         = 5
GRAD_CLIP      = 1.0
USE_AMP        = False    # no_trendloss 崩溃（0xC0000005），关闭 AMP

# ---------- 消融/训练开关 ----------
FORCE_RERUN = True
ONLY_EXP = ''

SHORT_W         = 1.0
SHORT_HORIZON   = 30
WRITE_REPORT    = True

# ---------- per-sample R² 方差门槛 ----------
R2_ABS_FLOOR = 0.02
R2_REL_FLOOR = 0.15

# ---------- Bootstrap ----------
BOOT_N    = 1000
BOOT_SEED = 42


# =============================================================================
# 实验配置
# =============================================================================

@dataclass
class DepthExperiment:
    name: str
    data_dir: Path
    depth_range: tuple | None = None
    top_k_layers: int = TOP_K_LAYERS
    max_layers: int = MAX_LAYERS
    patience_override: int | None = None
    epochs_override: int | None = None
    use_prior_default: bool | None = None

    def _tag(self, kind: str, split_mode: str | None = None) -> Path:
        if kind == 'ltfeat':
            return CACHE_ROOT / f'ltfeat_{self.name}_{CACHE_VERSION}'
        sm = split_mode if split_mode is not None else SPLIT_MODE
        return CACHE_ROOT / f'{kind}_{self.name}_{sm}_{CACHE_VERSION}'

    @property
    def cache_path(self) -> Path:
        return self._tag('ltfeat').with_suffix('.pkl')

    @property
    def out_prefix(self) -> str:
        return str(self._tag('ltres'))

    @property
    def csv_path(self) -> str:
        return str(self._tag('exp').with_suffix('.csv'))

    @property
    def report_path(self) -> str:
        return str(self._tag('report').with_suffix('.md'))

    def csv_path_for(self, split_mode: str) -> str:
        return str(self._tag('exp', split_mode=split_mode).with_suffix('.csv'))


def build_experiments() -> dict:
    return {
        'd0_200':   DepthExperiment('d0_200',   DATA_ROOT / 'depth_0_200',   (0, 200)),
        'd200_400': DepthExperiment('d200_400', DATA_ROOT / 'depth_200_400', (200, 400)),
        'd400_600': DepthExperiment('d400_600', DATA_ROOT / 'depth_400_600', (400, 600)),
        'd600_800': DepthExperiment('d600_800', DATA_ROOT / 'depth_600_800', (600, 800),
                                    patience_override=50, epochs_override=200),
    }


EXPERIMENTS = build_experiments()


# =============================================================================
# 通用工具
# =============================================================================

def smooth_ma(x: np.ndarray, w: int = 30) -> np.ndarray:
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    return np.convolve(xp, np.ones(w) / w, mode='valid')


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pinv_f64(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.pinv(A.double()).float()


def _fmt_num(v, fmt='.4f') -> str:
    if v is None:
        return '—'
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return '—'
    if math.isnan(fv):
        return '—'
    return f"{fv:{fmt}}"


def _fmt_int(v) -> str:
    if v is None:
        return '—'
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return '—'
    if math.isnan(fv):
        return '—'
    return str(int(fv))


# =============================================================================
# 1. 数据处理器
# =============================================================================

class LongTermDataProcessor:
    def __init__(self, exp: DepthExperiment, seq_len=SEQ_LEN,
                 forecast_days=FORECAST_DAYS, step=STEP, smooth_window=3):
        self.exp = exp
        self.data_dir = str(exp.data_dir)
        self.depth_range = exp.depth_range
        self.top_k_layers = exp.top_k_layers
        self.max_layers = exp.max_layers
        self.seq_len = seq_len
        self.forecast_days = forecast_days
        self.step = step
        self.smooth_window = smooth_window
        self.buoy_data: dict = {}
        self.feature_names: list = []

    def load_data(self, use_cache: bool = True) -> bool:
        cache_path = self.exp.cache_path
        if use_cache and cache_path.exists():
            print(f"发现缓存，正在加载: {cache_path}")
            with open(cache_path, 'rb') as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and 'buoy_data' in payload:
                if payload.get('version') != CACHE_VERSION:
                    print(f"缓存版本不匹配，重新提取")
                    self.buoy_data = {}
                else:
                    self.buoy_data = payload['buoy_data']
            else:
                self.buoy_data = payload
            total = sum(len(v) for v in self.buoy_data.values())
            if total > 0:
                print(f"缓存加载完成: {total} 个数据点，{len(self.buoy_data)} 个浮标")
                return True
            print("缓存为空，重新提取")
            self.buoy_data = {}

        if not self.exp.data_dir.exists():
            print(f"[错误] 数据目录不存在: {self.exp.data_dir}")
            return False

        excel_files = [f for f in os.listdir(self.data_dir) if f.endswith('.xlsx')]
        print(f"找到 {len(excel_files)} 个数据文件，开始提取特征（首次运行较慢）")

        err_counter: Counter = Counter()
        nrows = None if self.max_layers <= 0 else (self.max_layers + 1)
        for file_path in tqdm(excel_files, desc="加载数据"):
            try:
                full_path = os.path.join(self.data_dir, file_path)
                buoy_id, timestamp = self._parse_filename(file_path)
                if not buoy_id or not timestamp:
                    err_counter['bad_filename'] += 1
                    continue
                df = pd.read_excel(full_path, header=None, nrows=nrows)
                if len(df) < 2:
                    err_counter['empty_file'] += 1
                    continue
                lon = float(df.iloc[0, 1])
                lat = float(df.iloc[0, 2])
                features = self._extract_features(df, timestamp, lon, lat)
                if features is None:
                    err_counter['no_valid_profile'] += 1
                    continue
                self.buoy_data.setdefault(buoy_id, []).append({
                    'timestamp': timestamp, 'lon': lon, 'lat': lat,
                    'features': features, 'raw_cdom': features[3],
                })
            except Exception as e:
                key = type(e).__name__
                err_counter[key] += 1
                if err_counter[key] <= 5:
                    print(f"[skip] {file_path}: {e}")
                continue

        for bid in self.buoy_data:
            self.buoy_data[bid].sort(key=lambda x: x['timestamp'])

        total = sum(len(v) for v in self.buoy_data.values())
        print(f"成功加载 {total} 个数据点，{len(self.buoy_data)} 个浮标")
        if err_counter:
            print(f"跳过原因统计: {dict(err_counter)}")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({'version': CACHE_VERSION, 'buoy_data': self.buoy_data}, f)
            print(f"特征缓存已保存至: {cache_path}")
        except Exception as e:
            print(f"缓存保存失败（不影响训练）: {e}")
        return total > 0

    @staticmethod
    def _parse_filename(filename: str):
        base = os.path.splitext(filename)[0]
        m = re.match(r'(\d{6,7})_cdom_data_(\d{8})(?:_\d+)?$', base)
        if m:
            try:
                return m.group(1), datetime.strptime(m.group(2), "%Y%m%d")
            except ValueError:
                pass
        digits = re.findall(r'\d+', base)
        buoy_id = date_str = None
        for d in digits:
            if 6 <= len(d) <= 7 and buoy_id is None:
                buoy_id = d
            elif len(d) == 8 and date_str is None:
                date_str = d
        if buoy_id and date_str:
            try:
                return buoy_id, datetime.strptime(date_str, "%Y%m%d")
            except ValueError:
                pass
        return None, None

    def _extract_features(self, df, timestamp, lon, lat):
        try:
            data_rows = df.iloc[1:]
            if len(data_rows) == 0:
                return None
            depth_col, cdom_col, temp_col, sal_col = 5, 8, 10, 11
            sample_size = min(self.top_k_layers, len(data_rows))
            data_rows = data_rows.iloc[:sample_size]

            depths = pd.to_numeric(data_rows[depth_col], errors='coerce').values
            cdom_v = pd.to_numeric(data_rows[cdom_col], errors='coerce').values
            temp_v = pd.to_numeric(data_rows[temp_col], errors='coerce').values
            sal_v  = pd.to_numeric(data_rows[sal_col],  errors='coerce').values

            dmin, dmax = self.depth_range if self.depth_range else (0.0, 1e9)
            mask = (~np.isnan(depths)) & (~np.isnan(cdom_v)) & \
                   (~np.isnan(temp_v)) & (~np.isnan(sal_v)) & \
                   (depths >= dmin) & (depths <= dmax)
            if mask.sum() < 3:
                return None

            depth = float(np.nanmean(depths[mask]))
            temp  = float(np.nanmean(temp_v[mask]))
            sal   = float(np.nanmean(sal_v[mask]))
            cdom  = float(np.nanmean(cdom_v[mask]))
            if not (0 <= depth <= 2000) or not (-2 <= temp <= 40) or \
                    not (0 <= sal <= 40) or not (0 <= cdom <= 20):
                return None

            feats = [depth, temp, sal, cdom]

            dc, dt, ds = cdom_v[mask], temp_v[mask], sal_v[mask]
            if len(dc) > 2:
                idx = np.arange(len(dc))
                c_slope = np.polyfit(idx, dc, 1)[0]
                t_slope = np.polyfit(idx, dt, 1)[0]
                s_slope = np.polyfit(idx, ds, 1)[0]
                c_curv  = np.polyfit(idx, dc, 2)[0]
                t_curv  = np.polyfit(idx, dt, 2)[0]
            else:
                c_slope = t_slope = s_slope = c_curv = t_curv = 0.0
            feats += [c_slope, t_slope, s_slope, c_curv, t_curv]

            doy   = timestamp.timetuple().tm_yday
            month = timestamp.month
            woy   = doy // 7
            feats += [
                doy / 365.25,
                math.sin(2*math.pi*doy/365.25), math.cos(2*math.pi*doy/365.25),
                math.sin(4*math.pi*doy/365.25), math.cos(4*math.pi*doy/365.25),
                math.sin(2*math.pi*month/12),   math.cos(2*math.pi*month/12),
                woy / 52.0,
                ]
            feats += [
                (lon - 180) / 180, (lat - 90) / 90, abs(lat) / 90,
                lon * lat / 10000, lon**2 / 10000, lat**2 / 10000,
                ]

            ts_ratio = temp / (sal + 1e-6) if sal > 0 else 0
            cd_ratio = cdom / (depth + 1) if depth > 0 else 0
            feats += [
                ts_ratio, cd_ratio,
                temp * depth, temp * sal, cdom * temp, cdom * sal,
                cdom * depth, ts_ratio * depth,
                ]
            feats += [
                math.log(cdom + 1), math.log(depth + 1),
                math.log(temp + 3),  math.log(sal + 1),
            ]

            if len(dc) > 1:
                c_std  = float(np.std(dc))
                c_cv   = c_std / (cdom + 1e-6)
                c_skew = float(stats.skew(dc))
                c_kurt = float(stats.kurtosis(dc))
            else:
                c_std = c_cv = c_skew = c_kurt = 0.0
            feats += [c_std, c_cv, c_skew, c_kurt]

            feats += [
                cdom * temp * sal / 100,
                doy * lat / 10000,
                doy * temp / 100,
                depth * temp / 100,
                depth * sal / 100,
                ]

            if not self.feature_names:
                self.feature_names = [
                    'depth', 'temp', 'sal', 'cdom',
                    'cdom_slope', 'temp_slope', 'sal_slope', 'cdom_curv', 'temp_curv',
                    'day_norm', 'day_sin', 'day_cos', 'day_sin2', 'day_cos2',
                    'month_sin', 'month_cos', 'week_norm',
                    'norm_lon', 'norm_lat', 'distance_from_equator',
                    'lon_lat_product', 'lon_sq', 'lat_sq',
                    'temp_sal_ratio', 'cdom_depth_ratio', 'temp_depth_product',
                    'temp_sal_product', 'cdom_temp', 'cdom_sal', 'cdom_depth',
                    'temp_sal_depth',
                    'log_cdom', 'log_depth', 'log_temp', 'log_sal',
                    'cdom_std', 'cdom_cv', 'cdom_skew', 'cdom_kurt',
                    'cdom_temp_sal', 'day_lat', 'day_temp', 'depth_temp', 'depth_sal',
                ]

            if any(not np.isfinite(f) for f in feats):
                return None
            return feats
        except Exception:
            return None

    def prepare_sequences(self, min_sequence_length=None, apply_smoothing=False,
                          max_gap_days=MAX_GAP_DAYS, interpolate_limit_days=MAX_GAP_DAYS):
        if min_sequence_length is None:
            min_sequence_length = self.seq_len + self.forecast_days

        all_seq, all_tar, all_meta = [], [], []
        for buoy_id, points in tqdm(self.buoy_data.items(), desc="按浮标生成序列"):
            points = sorted(points, key=lambda x: x['timestamp'])
            if len(points) < 2:
                continue

            segments, cur = [], [points[0]]
            for p in points[1:]:
                if (p['timestamp'] - cur[-1]['timestamp']).days > max_gap_days:
                    segments.append(cur); cur = [p]
                else:
                    cur.append(p)
            segments.append(cur)

            for seg in segments:
                if len(seg) < 2:
                    continue
                start_dt = seg[0]['timestamp'].replace(hour=0, minute=0, second=0, microsecond=0)
                end_dt   = seg[-1]['timestamp'].replace(hour=0, minute=0, second=0, microsecond=0)
                if (end_dt - start_dt).days + 1 < min_sequence_length:
                    continue

                dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
                seg_times = pd.DatetimeIndex([
                    p['timestamp'].replace(hour=0, minute=0, second=0, microsecond=0)
                    for p in seg])
                feat_df = pd.DataFrame([p['features'] for p in seg],
                                       index=seg_times, dtype=np.float32)
                cdom_s = pd.Series([p['raw_cdom'] for p in seg],
                                   index=seg_times, dtype=np.float32)

                feat_df = feat_df.reindex(dates).interpolate(
                    method='linear', limit=interpolate_limit_days, limit_area='inside')
                cdom_s = cdom_s.reindex(dates).interpolate(
                    method='linear', limit=interpolate_limit_days, limit_area='inside')

                valid = feat_df.notna().all(axis=1) & cdom_s.notna()
                feat_df, cdom_s = feat_df[valid], cdom_s[valid]
                if len(feat_df) < min_sequence_length:
                    continue

                if apply_smoothing and len(feat_df) > self.smooth_window:
                    fa = feat_df.values.astype(np.float32)
                    ca = cdom_s.values.astype(np.float32)
                    for i in range(fa.shape[1]):
                        try:
                            fa[:, i] = signal.savgol_filter(fa[:, i], self.smooth_window, 2)
                        except Exception:
                            pass
                    try:
                        ca = signal.savgol_filter(ca, self.smooth_window, 2)
                    except Exception:
                        pass
                    feat_df = pd.DataFrame(fa, index=feat_df.index)
                    cdom_s = pd.Series(ca, index=cdom_s.index)

                # 该浮标段代表位置（用段首点）
                seg_lat = float(seg[0]['lat'])
                seg_lon = float(seg[0]['lon'])

                for st in range(0, len(feat_df) - self.seq_len - self.forecast_days + 1, self.step):
                    end_seq = st + self.seq_len
                    end_tar = end_seq + self.forecast_days
                    x = feat_df.iloc[st:end_seq].values.astype(np.float32)
                    y = cdom_s.iloc[end_seq:end_tar].values.astype(np.float32)
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        continue
                    all_seq.append(x); all_tar.append(y)
                    all_meta.append({
                        'buoy_id': buoy_id,
                        'start_date': feat_df.index[st],
                        'end_date': feat_df.index[end_tar - 1],
                        'lat': seg_lat,
                        'lon': seg_lon,
                    })

        if not all_seq:
            print(f"[{self.exp.name}] 未生成任何序列，请检查数据是否满足 "
                  f"{self.seq_len}+{self.forecast_days} 天观测")
            return None, None, None

        all_seq = np.array(all_seq, dtype=np.float32)
        all_tar = np.array(all_tar, dtype=np.float32)
        print(f"[{self.exp.name}] 生成样本 {len(all_seq)}，"
              f"X={all_seq.shape}，y={all_tar.shape}")
        return all_seq, all_tar, all_meta


# =============================================================================
# 2. 模型
# =============================================================================

class PatchTSTBranch(nn.Module):
    def __init__(self, input_dim, seq_len=730, forecast_days=365,
                 patch_len=15, stride=7, d_model=32, n_heads=4,
                 e_layers=1, n_pool=6, dropout=0.3):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.n_pool = n_pool
        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.query = nn.Parameter(torch.randn(n_pool, d_model) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(n_pool * d_model), nn.Dropout(0.2),
            nn.Linear(n_pool * d_model, forecast_days))

    def forward(self, x):
        B = x.shape[0]
        patches = x.unfold(1, self.patch_len, self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(B, self.n_patches, -1)
        h = self.dropout(self.patch_embed(patches) + self.pos_embed)
        h = self.norm(self.encoder(h))
        scores = torch.matmul(h, self.query.t()) / math.sqrt(h.size(-1))
        attn = torch.softmax(scores, dim=1)
        pooled = torch.matmul(attn.transpose(1, 2), h)
        return self.head(pooled.reshape(B, -1))


class VariableAttentionBranch(nn.Module):
    def __init__(self, input_dim, seq_len=730, forecast_days=365,
                 d_model=24, n_heads=4, e_layers=1, dropout=0.3):
        super().__init__()
        self.var_proj = nn.Linear(seq_len, d_model)
        self.var_embed = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=e_layers)
        self.norm = nn.LayerNorm(d_model)
        self.temporal_proj = nn.Linear(d_model, forecast_days)
        self.out_proj = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = self.var_proj(x.permute(0, 2, 1)) + self.var_embed
        h = self.norm(self.encoder(h))
        h = self.temporal_proj(h)
        return self.out_proj(h.permute(0, 2, 1)).squeeze(-1)


class MultiScaleConvBranch(nn.Module):
    def __init__(self, input_dim, forecast_days=365, d_model=16, dropout=0.3):
        super().__init__()
        kernels = [3, 7, 15, 31, 61, 91]
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, d_model, kernel_size=k, padding=k // 2),
                nn.GELU(), nn.AdaptiveAvgPool1d(1))
            for k in kernels])
        self.head = nn.Sequential(
            nn.Linear(d_model * len(kernels), d_model * 4), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 4, forecast_days))

    def forward(self, x):
        x_t = x.transpose(1, 2)
        feats = [c(x_t).squeeze(-1) for c in self.convs]
        return self.head(torch.cat(feats, dim=1))


class FreqBranch(nn.Module):
    def __init__(self, seq_len=730, forecast_days=365, keep_frac=0.3):
        super().__init__()
        self.forecast_days = forecast_days
        K = seq_len // 2 + 1
        self.Kf = max(8, int(round(K * keep_frac)))
        self.Tf = forecast_days // 2 + 1
        self.gain = nn.Parameter(torch.randn(self.Kf, 2) * 0.05)

    def forward(self, cdom):
        xf = torch.fft.rfft(cdom, dim=1)
        low = xf[:, :self.Kf]
        g = torch.view_as_complex(self.gain.contiguous())
        y = low * g
        yr = F.interpolate(y.real.unsqueeze(1), size=self.Tf,
                           mode='linear', align_corners=False).squeeze(1)
        yi = F.interpolate(y.imag.unsqueeze(1), size=self.Tf,
                           mode='linear', align_corners=False).squeeze(1)
        spec = torch.complex(yr, yi)
        return torch.fft.irfft(spec, n=self.forecast_days, dim=1)


class ImprovedLongTermModel(nn.Module):
    def __init__(self, input_dim=44, seq_len=730, forecast_days=365,
                 cdom_idx=3, hidden=48, dropout=0.3,
                 patch_len=15, patch_stride=7, patch_d_model=24,
                 n_heads=4, e_layers=1, n_pool=4,
                 var_d_model=16, conv_d_model=12,
                 trend_win=365, slope_win=180, anom_slope_win=120,
                 freq_keep_frac=0.3,
                 use_patch=True, use_var=True, use_conv=True, use_freq=True,
                 use_prior=True, use_coef_adj=True):
        super().__init__()
        self.cdom_idx = cdom_idx
        self.seq_len = seq_len
        self.forecast_days = forecast_days
        self.trend_win = trend_win
        self.slope_win = slope_win
        self.anom_slope_win = anom_slope_win
        self.use_patch = use_patch
        self.use_var = use_var
        self.use_conv = use_conv
        self.use_freq = use_freq
        self.use_prior = use_prior
        self.use_coef_adj = use_coef_adj
        self.prior_logit = nn.Parameter(torch.tensor(2.0))
        self.deep_logit  = nn.Parameter(torch.tensor(-2.0))

        t  = torch.arange(seq_len, dtype=torch.float32)
        ft = torch.arange(seq_len, seq_len + forecast_days, dtype=torch.float32)
        w1 = 2.0 * math.pi / 365.25
        w2 = 4.0 * math.pi / 365.25
        w3 = 6.0 * math.pi / 365.25

        def design(tt):
            return torch.stack([
                torch.ones_like(tt),
                torch.sin(w1 * tt), torch.cos(w1 * tt),
                torch.sin(w2 * tt), torch.cos(w2 * tt),
                torch.sin(w3 * tt), torch.cos(w3 * tt)], dim=1)

        self.register_buffer('design_hist', design(t))
        self.register_buffer('design_fut',  design(ft))
        self.register_buffer('pinv_hist',   pinv_f64(design(t)))
        tr = torch.arange(365, dtype=torch.float32)
        self.register_buffer('pinv_recent', pinv_f64(design(tr)))

        def slope_design(W):
            tt = torch.arange(W, dtype=torch.float32)
            return torch.stack([torch.ones(W), tt], dim=1)
        self.register_buffer('pinv_slope',      pinv_f64(slope_design(slope_win)))
        self.register_buffer('pinv_anom_slope', pinv_f64(slope_design(anom_slope_win)))

        self.summary_net = nn.Sequential(
            nn.Linear(11, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 4))

        self.coef_adj_net = nn.Sequential(
            nn.Linear(32, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 2))
        with torch.no_grad():
            self.coef_adj_net[-1].weight.zero_()
            self.coef_adj_net[-1].bias.zero_()

        self.patch_branch = self.var_branch = None
        self.conv_branch  = self.freq_branch = None
        if use_patch:
            self.patch_branch = PatchTSTBranch(
                input_dim, seq_len, forecast_days, patch_len, patch_stride,
                patch_d_model, n_heads, e_layers, n_pool, dropout)
        if use_var:
            self.var_branch = VariableAttentionBranch(
                input_dim, seq_len, forecast_days, var_d_model,
                n_heads, e_layers, dropout)
        if use_conv:
            self.conv_branch = MultiScaleConvBranch(
                input_dim, forecast_days, conv_d_model, dropout)
        if use_freq:
            self.freq_branch = FreqBranch(seq_len, forecast_days, freq_keep_frac)

        self.n_branch = int(use_patch) + int(use_var) + int(use_conv) + int(use_freq)
        self.fuse = nn.Sequential(
            nn.Conv1d(self.n_branch, 16, 1), nn.GELU(),
            nn.Dropout(dropout * 0.5), nn.Conv1d(16, 1, 1))
        self.aux_heads = nn.ModuleList([nn.Conv1d(self.n_branch, 1, 1) for _ in range(3)])

    @staticmethod
    def _moving_avg(x, win):
        pad = win // 2
        xp = F.pad(x.unsqueeze(1), (pad, pad), mode='replicate')
        return F.avg_pool1d(xp, kernel_size=win, stride=1).squeeze(1)

    def forward(self, x, return_aux=False , prior_only=False):
        B = x.shape[0]
        cdom = x[:, :, self.cdom_idx]

        trend = self._moving_avg(cdom, self.trend_win)
        seas = cdom - trend
        coef = seas @ self.pinv_hist.t()
        seas_fit = (self.design_hist @ coef.t()).transpose(0, 1)

        tail = trend[:, -self.slope_win:]
        slope_coef = tail @ self.pinv_slope.t()
        level = trend[:, -1]
        slope = slope_coef[:, 1]
        anom = cdom - trend - seas_fit
        atail = anom[:, -self.anom_slope_win:]
        anom_coef = atail @ self.pinv_anom_slope.t()
        anom_slope = anom_coef[:, 1]
        anom_last = anom[:, -1]

        amp1 = torch.sqrt(coef[:, 1]**2 + coef[:, 2]**2 + 1e-8)
        amp2 = torch.sqrt(coef[:, 3]**2 + coef[:, 4]**2 + 1e-8)
        amp3 = torch.sqrt(coef[:, 5]**2 + coef[:, 6]**2 + 1e-8)

        summary = torch.stack([
            cdom[:, -1], cdom[:, -30:].mean(dim=1), cdom[:, -90:].mean(dim=1),
            cdom[:, -365:].std(dim=1),
            slope, level, anom_last, anom_slope, amp1, amp2, amp3], dim=1)

        if self.use_coef_adj:
            coef_recent = seas[:, -365:] @ self.pinv_recent.t()
            adj_in = torch.cat([summary, coef, coef_recent,
                                coef - coef_recent], dim=-1)
            pa = self.coef_adj_net(adj_in)
            dphase = torch.tanh(pa[:, 0]) * 45.0
            alpha = 1.0 + torch.tanh(pa[:, 1]) * 0.8
            w1 = 2.0 * math.pi / 365.25
            coef_hat = coef.clone()
            for k, mult in [(1, 1), (3, 2), (5, 3)]:
                amp = torch.sqrt(coef[:, k]**2 + coef[:, k+1]**2 + 1e-8)
                ang = torch.atan2(coef[:, k], coef[:, k+1]) + mult * w1 * dphase
                scl = alpha if k == 1 else 1.0
                coef_hat[:, k]   = amp * scl * torch.sin(ang)
                coef_hat[:, k+1] = amp * scl * torch.cos(ang)
        else:
            coef_hat = coef

        seas_fut = (self.design_fut @ coef_hat.t()).transpose(0, 1)
        g = self.summary_net(summary)
        t_idx = torch.arange(self.forecast_days, device=x.device, dtype=x.dtype)
        t_norm = t_idx / self.forecast_days

        if self.use_prior:
            damp = torch.sigmoid(g[:, 2:3])
            trend_gate = torch.sigmoid(g[:, 3:4]) * 1.5
            seasonal_prior = seas_fut * (1.0 - damp * t_norm.unsqueeze(0))
            trend_prior = level.unsqueeze(1) + slope.unsqueeze(1) * t_idx.unsqueeze(0)
            prior = seasonal_prior + trend_gate * trend_prior
        else:
            prior = torch.zeros_like(t_norm).unsqueeze(0).expand(B, -1)

        outs = []
        if self.use_patch: outs.append(self.patch_branch(x))
        if self.use_var:   outs.append(self.var_branch(x))
        if self.use_conv:  outs.append(self.conv_branch(x))
        if self.use_freq:  outs.append(self.freq_branch(cdom))
        rep = torch.stack(outs, dim=-1)
        deep = self.fuse(rep.transpose(1, 2)).squeeze(1)

        # if self.use_prior:
        #     w_res = g[:, 0:1] + g[:, 1:2] * t_norm.unsqueeze(0)
        #     w_prior = torch.sigmoid(self.prior_logit)
        #     w_deep  = torch.sigmoid(self.deep_logit)
        #     w_sum   = w_prior + w_deep + 1e-8
        #     pred = (w_prior / w_sum) * prior + (w_deep / w_sum) * (w_res * deep)
        # else:
        #     pred = deep

        if self.use_prior:
            if prior_only:
                pred = prior                       # ★ 新增
            else:
                w_res = g[:, 0:1] + g[:, 1:2] * t_norm.unsqueeze(0)
                w_prior = torch.sigmoid(self.prior_logit)
                w_deep  = torch.sigmoid(self.deep_logit)
                w_sum   = w_prior + w_deep + 1e-8
                pred = (w_prior / w_sum) * prior + (w_deep / w_sum) * (w_res * deep)
        else:
            pred = prior if prior_only else deep     # ★ 新增

        if return_aux:
            aux = {}
            for h, head in zip([30, 90, 180], self.aux_heads):
                if h <= self.forecast_days:
                    aux[h] = head(rep[:, :h].transpose(1, 2)).squeeze(1)
            return pred, aux
        return pred


class V1HybridModel(nn.Module):
    def __init__(self, input_dim, seq_len=730, forecast_days=365,
                 cdom_idx=3, hidden=64):
        super().__init__()
        self.cdom_idx = cdom_idx
        self.forecast_days = forecast_days
        t  = torch.arange(seq_len, dtype=torch.float32)
        ft = torch.arange(seq_len, seq_len + forecast_days, dtype=torch.float32)
        f1, f2 = 2.0*math.pi/365.25, 4.0*math.pi/365.25
        design = torch.stack([torch.ones(seq_len), t,
                              torch.sin(f1*t), torch.cos(f1*t),
                              torch.sin(f2*t), torch.cos(f2*t)], dim=1)
        self.register_buffer('pinv', pinv_f64(design))
        self.register_buffer('fd', torch.stack([
            torch.ones(forecast_days), ft,
            torch.sin(f1*ft), torch.cos(f1*ft),
            torch.sin(f2*ft), torch.cos(f2*ft)], dim=1))
        self.summary_net = nn.Sequential(
            nn.Linear(7, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.res_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, 7, padding=3), nn.GELU(),
            nn.Conv1d(32, 32, 7, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1))
        self.res_head = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(32, forecast_days))

    def forward(self, x):
        cdom = x[:, :, self.cdom_idx]
        coef = cdom @ self.pinv.t()
        base = (self.fd @ coef.t()).transpose(0, 1)
        last, l30, l90 = cdom[:, -1], cdom[:, -30:].mean(dim=1), cdom[:, -90:].mean(dim=1)
        slope  = coef[:, 1]
        annual = torch.sqrt(coef[:, 2]**2 + coef[:, 3]**2 + 1e-8)
        semi   = torch.sqrt(coef[:, 4]**2 + coef[:, 5]**2 + 1e-8)
        summary = torch.stack([last, l30, l90, slope, annual, semi,
                               cdom.std(dim=1)], dim=1)
        g = self.summary_net(summary)
        trend = torch.arange(self.forecast_days, device=x.device,
                             dtype=x.dtype).unsqueeze(0) / self.forecast_days
        pred = base * torch.sigmoid(g[:, 0:1]) + g[:, 1:2] + trend * g[:, 2:3]
        res = self.res_encoder(x.transpose(1, 2)).squeeze(-1)
        return pred + self.res_head(res)


class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, seq_len=730, forecast_days=365,
                 hidden=64, layers=2, dropout=0.3, block=10):
        super().__init__()
        self.block = block
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(0.2),
            nn.Linear(hidden, forecast_days))

    def forward(self, x):
        B, L, C = x.shape
        nb = L // self.block
        if nb >= 1:
            x = x[:, :nb * self.block].reshape(B, nb, self.block, C).mean(dim=2)
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


# =============================================================================
# 3. 损失函数
# =============================================================================

class CombinedLoss(nn.Module):
    def __init__(self, lambda_mae=0.5, lambda_anchor=0.15,
                 lambda_trend15=0.5, lambda_trend30=1.2, lambda_corr=0.5,
                 lambda_trenddir=0.2, lambda_shape=0.1, lambda_peak=0.1,
                 lambda_horizon=0.15, lambda_aux=0.3, use_trend_terms=True,
                 short_w=1.0, short_horizon=30, peak_ema=0.9):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_anchor = lambda_anchor
        self.lambda_trend15 = lambda_trend15
        self.lambda_trend30 = lambda_trend30
        self.lambda_corr = lambda_corr
        self.lambda_trenddir = lambda_trenddir
        self.lambda_shape = lambda_shape
        self.lambda_peak = lambda_peak
        self.lambda_horizon = lambda_horizon
        self.lambda_aux = lambda_aux
        self.use_trend_terms = use_trend_terms
        self.short_w = short_w
        self.short_horizon = short_horizon
        self.peak_ema = peak_ema
        self.register_buffer('peak_thresh', torch.tensor(0.0))

    @staticmethod
    def _smooth(x, w):
        return F.avg_pool1d(x.unsqueeze(1), kernel_size=w, stride=1,
                            padding=w // 2, count_include_pad=False).squeeze(1)

    def forward(self, pred, target, aux=None, last_cdom=None):
        huber = F.smooth_l1_loss(pred, target, beta=0.3)
        mae = F.l1_loss(pred, target)
        loss = huber + self.lambda_mae * mae

        if last_cdom is not None:
            loss = loss + self.lambda_anchor * F.mse_loss(pred[:, 0], last_cdom)

        if self.use_trend_terms:
            s15p, s15t = self._smooth(pred, 15), self._smooth(target, 15)
            s30p, s30t = self._smooth(pred, 30), self._smooth(target, 30)
            loss = loss + self.lambda_trend15 * F.mse_loss(s15p, s15t)
            loss = loss + self.lambda_trend30 * F.mse_loss(s30p, s30t)

            pc = s30p - s30p.mean(dim=1, keepdim=True)
            tc = s30t - s30t.mean(dim=1, keepdim=True)
            denom = pc.norm(dim=1) * tc.norm(dim=1) + 1e-6
            corr = (pc * tc).sum(dim=1) / denom
            loss = loss + self.lambda_corr * (1.0 - corr).mean()

            if self.lambda_trenddir > 0 and s30p.size(1) > 31:
                dp = s30p[:, 30:] - s30p[:, :-30]
                dt = s30t[:, 30:] - s30t[:, :-30]
                loss = loss + self.lambda_trenddir * F.relu(-(dp * dt)).mean()

            pdiff = pred[:, 1:] - pred[:, :-1]
            tdiff = target[:, 1:] - target[:, :-1]
            loss = loss + self.lambda_shape * F.relu(-(pdiff * tdiff)).mean()

        with torch.no_grad():
            batch_q = target.quantile(0.9)
            if self.peak_thresh.item() == 0.0:
                self.peak_thresh.fill_(batch_q.item())
            else:
                self.peak_thresh.mul_(self.peak_ema).add_(
                    batch_q, alpha=1.0 - self.peak_ema)
        mask = (target > self.peak_thresh).float()
        if mask.sum() > 0:
            loss = loss + self.lambda_peak * ((pred - target)**2 * mask).sum() / mask.sum()

        hw = torch.ones(pred.size(1), device=pred.device)
        if self.short_w > 1.0 and self.short_horizon > 0:
            h = min(self.short_horizon, pred.size(1))
            hw[:h] = self.short_w
        ramp = torch.linspace(0.9, 1.15, pred.size(1), device=pred.device)
        hw = hw * ramp
        loss = loss + self.lambda_horizon * (((pred - target)**2) * hw).mean()

        if aux is not None:
            for h in aux:
                loss = loss + (self.lambda_aux / len(aux)) * F.mse_loss(
                    aux[h], target[:, :h])
        return loss


# =============================================================================
# 4. 数据集 & Bootstrap 工具
# =============================================================================

class AugDataset(Dataset):
    def __init__(self, X, y, noise=0.03, noise_prob=0.5, cdom_idx=3):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.noise = noise
        self.noise_prob = noise_prob
        self.cdom_idx = cdom_idx

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.y[i]
        if self.noise_prob > 0 and torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(x) * self.noise
            noise[:, self.cdom_idx] = 0.0
            x = x + noise
        return x, y


def _bootstrap_metric(preds, targets, metric_fn, n_boot=BOOT_N, seed=BOOT_SEED,
                      alpha=0.05):
    """按样本为单位做 bootstrap，估计 metric 的 (1-alpha) 置信区间。
    ★ 修复：显式过滤 NaN，避免 percentile 返回 nan。"""
    n = preds.shape[0]
    if n < 5:
        return float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, n)
        v = metric_fn(preds[idx], targets[idx])
        boots[b] = v if np.isfinite(v) else np.nan
    boots = boots[np.isfinite(boots)]
    if len(boots) < max(100, n_boot // 10):
        return float('nan'), float('nan')
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def _tc_scalar(preds, targets):
    """TrendCorr 标量版本，用于 bootstrap。
    ★ 修复：显式过滤常数/NaN 样本。"""
    tcs = []
    for i in range(preds.shape[0]):
        y_t, p_t = smooth_ma(targets[i]), smooth_ma(preds[i])
        if np.std(y_t) > 1e-6 and np.std(p_t) > 1e-6:
            r = np.corrcoef(y_t, p_t)[0, 1]
            if np.isfinite(r):
                tcs.append(r)
    return float(np.mean(tcs)) if tcs else 0.0


def _lat_band_metrics(preds, targets, lats):
    """纬度带分层评估：S (<-30) / T (-30~30) / N (>30)。"""
    out = {}
    bands = [('S', -90.0, -30.0), ('T', -30.0, 30.0), ('N', 30.0, 90.0)]
    lats = np.asarray(lats, dtype=np.float64)
    for name, lo, hi in bands:
        if name == 'T':
            mask = (lats >= lo) & (lats <= hi)
        else:
            mask = (lats >= lo) & (lats < hi)
        n = int(mask.sum())
        out[f'n_{name}'] = n
        if n < 3:
            out[f'TC_{name}'] = float('nan')
            out[f'RMSE_{name}'] = float('nan')
            continue
        p = preds[mask]; t = targets[mask]
        out[f'TC_{name}'] = _tc_scalar(p, t)
        out[f'RMSE_{name}'] = float(np.sqrt(
            mean_squared_error(t.flatten(), p.flatten())))
    return out


# =============================================================================
# 5. 训练器
# =============================================================================

class LongTermTrainer:
    def __init__(self, model, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.history = defaultdict(list)
        self.target_scaler = None

    def train(self, train_loader, val_loader, val_mu, val_sigma,
              epochs=200, lr=LR, weight_decay=WEIGHT_DECAY, grad_clip=GRAD_CLIP,
              patience=PATIENCE_TRAIN, warmup=WARMUP, use_amp=False,
              model_path='best_long_term_model_v3.pt',
              use_trend_terms=True, verbose=True):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                      weight_decay=weight_decay)

        def lr_lambda(ep):
            if ep < warmup:
                return max(1e-4, ep / max(1, warmup))
            p = (ep - warmup) / max(1, epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * p))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = CombinedLoss(use_trend_terms=use_trend_terms,
                                 short_w=SHORT_W, short_horizon=SHORT_HORIZON)
        amp_enabled = use_amp and self.device == 'cuda'
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        best_score, wait = float('inf'), 0

        for epoch in range(epochs):
            self.model.train()
            tr_loss, n_tr = 0.0, 0
            for x, y in tqdm(train_loader,
                             desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    try:
                        pred, aux = self.model(x, return_aux=True)
                    except TypeError:
                        pred, aux = self.model(x), None
                    pred = pred.squeeze(-1) if pred.dim() == 3 else pred
                    loss = criterion(pred, y, aux=aux, last_cdom=x[:, -1, 3])

                # ★ NaN/Inf 检测：只做一次，跳过坏 batch
                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)

                # ★ 梯度 NaN/Inf 检测
                has_bad_grad = any(
                    not torch.isfinite(p.grad).all()
                    for p in self.model.parameters() if p.grad is not None
                )
                if has_bad_grad:
                    optimizer.zero_grad()
                    continue

                scaler.step(optimizer); scaler.update()
                tr_loss += loss.item() * x.size(0)
                n_tr += x.size(0)
            tr_loss /= max(1, n_tr)

            self.model.eval()
            va_loss, n_va = 0.0, 0
            vp_list, vt_list = [], []
            with torch.no_grad():
                for x, y in tqdm(val_loader,
                                 desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False):
                    x, y = x.to(self.device), y.to(self.device)
                    try:
                        pred, aux = self.model(x, return_aux=True)
                    except TypeError:
                        pred, aux = self.model(x), None
                    pred = pred.squeeze(-1) if pred.dim() == 3 else pred
                    loss = criterion(pred, y, aux=aux, last_cdom=x[:, -1, 3])
                    va_loss += loss.item() * x.size(0)
                    n_va += x.size(0)
                    vp_list.append(pred.cpu().numpy())
                    vt_list.append(y.cpu().numpy())
            va_loss /= max(1, n_va)
            vp = np.vstack(vp_list); vt = np.vstack(vt_list)

            if val_mu is not None:
                vp_orig = np.clip(vp * val_sigma + val_mu, 0.0, 20.0)
                vt_orig = vt * val_sigma + val_mu
            else:
                assert self.target_scaler is not None, \
                    "target_scaler 未设置：全局标准化路径下需先赋值"
                vp_orig = np.clip(self.target_scaler.inverse_transform(
                    vp.reshape(-1, 1)).reshape(vp.shape), 0.0, 20.0)
                vt_orig = self.target_scaler.inverse_transform(
                    vt.reshape(-1, 1)).reshape(vt.shape)

            rmse = float(np.sqrt(mean_squared_error(vt_orig.flatten(), vp_orig.flatten())))
            mae  = float(mean_absolute_error(vt_orig.flatten(), vp_orig.flatten()))
            r2   = float(r2_score(vt_orig.flatten(), vp_orig.flatten()))

            tc = []
            for i in range(vp_orig.shape[0]):
                y_t, p_t = smooth_ma(vt_orig[i]), smooth_ma(vp_orig[i])
                if np.std(y_t) > 1e-6 and np.std(p_t) > 1e-6:
                    tc.append(np.corrcoef(y_t, p_t)[0, 1])
            avg_tc = float(np.mean(tc)) if tc else 0.0

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(va_loss)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            self.history['val_rmse'].append(rmse)
            self.history['val_trend_corr'].append(avg_tc)

            if verbose:
                print(f"Epoch {epoch+1:3d}: Train={tr_loss:.6f} Val={va_loss:.6f} "
                      f"RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} TC={avg_tc:.4f}")

            beta = 0.4
            val_score = rmse * (1.0 + beta * (1.0 - avg_tc))

            if val_score < best_score:
                best_score, wait = val_score, 0
                torch.save(self.model.state_dict(), model_path)
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            scheduler.step()

        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        return self.history

    def predict(self, loader, mu, sigma):
        self.model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                pred = self.model(x).squeeze(-1)
                ps.append(pred.cpu().numpy())
                ts.append(y.numpy())
        ps = np.vstack(ps); ts = np.vstack(ts)
        if mu is not None:
            return (np.clip(ps * sigma + mu, 0.0, 20.0), ts * sigma + mu)
        assert self.target_scaler is not None, \
            "target_scaler 未设置：全局标准化路径下需先赋值"
        return (np.clip(self.target_scaler.inverse_transform(
            ps.reshape(-1, 1)).reshape(ps.shape), 0.0, 20.0),
                self.target_scaler.inverse_transform(
                    ts.reshape(-1, 1)).reshape(ts.shape))

    @staticmethod
    def metrics_from(preds, targets, test_meta=None):
        m = {}
        m['RMSE'] = float(np.sqrt(mean_squared_error(targets.flatten(),
                                                     preds.flatten())))
        m['MAE']  = float(mean_absolute_error(targets.flatten(), preds.flatten()))
        m['R2']   = float(r2_score(targets.flatten(), preds.flatten()))

        denom = max(np.mean(np.abs(targets)), 1e-6)
        m['nRMSE'] = m['RMSE'] / denom

        r2_list, tc_list = [], []
        n_skipped_r2 = 0
        for i in range(preds.shape[0]):
            y_i, p_i = targets[i], preds[i]
            std_i  = float(np.std(y_i))
            mean_i = float(np.mean(y_i))
            floor  = max(R2_ABS_FLOOR, R2_REL_FLOOR * abs(mean_i))
            if std_i > floor:
                r2_list.append(1 - np.sum((p_i - y_i)**2) /
                               (np.sum((y_i - y_i.mean())**2) + 1e-9))
            else:
                n_skipped_r2 += 1
            y_t, p_t = smooth_ma(y_i), smooth_ma(p_i)
            if np.std(y_t) > 1e-6 and np.std(p_t) > 1e-6:
                tc_list.append(np.corrcoef(y_t, p_t)[0, 1])

        m['R2_sample_mean'] = float(np.mean(r2_list)) if r2_list else float('nan')
        m['R2_sample_n']    = int(len(r2_list))
        m['R2_sample_skip'] = int(n_skipped_r2)
        m['TrendCorr'] = float(np.mean(tc_list)) if tc_list else 0.0

        thr = np.percentile(targets, 90)
        peak = targets > thr
        if peak.sum() > 0:
            m['PeakError'] = float(np.mean(
                np.abs(preds[peak] - targets[peak]) / (targets[peak] + 1e-6)))
        else:
            m['PeakError'] = float('nan')

        for d in [30, 90, 180, 365]:
            if d <= preds.shape[1]:
                m[f'RMSE_{d}d'] = float(np.sqrt(mean_squared_error(
                    targets[:, :d].flatten(), preds[:, :d].flatten())))

        # ★ 新增：per-sample RMSE 分布（用于诊断少数极端样本）
        per_sample_rmse = np.sqrt(np.mean((preds - targets) ** 2, axis=1))
        m['RMSE_p50'] = float(np.median(per_sample_rmse))
        m['RMSE_p90'] = float(np.percentile(per_sample_rmse, 90))
        m['RMSE_max'] = float(np.max(per_sample_rmse))

        # ---------------- Bootstrap CI ----------------
        rmse_fn = lambda p, t: float(np.sqrt(
            mean_squared_error(t.flatten(), p.flatten())))
        mae_fn  = lambda p, t: float(
            mean_absolute_error(t.flatten(), p.flatten()))
        r2_fn   = lambda p, t: float(r2_score(t.flatten(), p.flatten()))

        lo, hi = _bootstrap_metric(preds, targets, rmse_fn)
        m['RMSE_CI_low'], m['RMSE_CI_high'] = lo, hi
        lo, hi = _bootstrap_metric(preds, targets, mae_fn)
        m['MAE_CI_low'], m['MAE_CI_high'] = lo, hi
        lo, hi = _bootstrap_metric(preds, targets, r2_fn)
        m['R2_CI_low'], m['R2_CI_high'] = lo, hi
        lo, hi = _bootstrap_metric(preds, targets, _tc_scalar)
        m['TC_CI_low'], m['TC_CI_high'] = lo, hi

        # ---------------- 纬度带分层 ----------------
        _empty_lat_keys = ['n_S', 'n_T', 'n_N', 'TC_S', 'TC_T', 'TC_N',
                           'RMSE_S', 'RMSE_T', 'RMSE_N']
        if test_meta is not None:
            lats = np.array([tm.get('lat', 0.0) for tm in test_meta])
            if len(lats) != preds.shape[0]:
                print(f"  [警告] lats 长度 {len(lats)} != preds 长度 "
                      f"{preds.shape[0]}，跳过纬度带")
                for k in _empty_lat_keys:
                    m[k] = float('nan')
            else:
                lat_out = _lat_band_metrics(preds, targets, lats)
                print(f"  [纬度带] n_S={lat_out.get('n_S', 0)}, "
                      f"n_T={lat_out.get('n_T', 0)}, "
                      f"n_N={lat_out.get('n_N', 0)}")
                m.update(lat_out)
        else:
            for k in _empty_lat_keys:
                m[k] = float('nan')

        return m


# =============================================================================
# 6. 绘图
# =============================================================================

def _safe_savefig(fig, path, dpi=150):
    path = Path(path)
    try:
        fig.savefig(path, dpi=dpi)
        return path
    except (OSError, PermissionError) as e:
        alt = path.with_name(
            f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}")
        fig.savefig(alt, dpi=dpi)
        print(f"[警告] 原路径保存失败（{e}），已改存为: {alt}")
        return alt


def plot_results(history, metrics, preds, targets, test_meta=None,
                 save_prefix='long_term_results_v3'):
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    axes = axes.flatten()
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Train / Val Loss'); axes[0].legend(); axes[0].grid(True)

    ax2 = axes[1]
    ax2.plot(epochs, history['lr'], color='tab:blue')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('LR', color='tab:blue')
    ax2.set_yscale('log'); ax2.grid(True)
    if 'val_rmse' in history:
        ax2b = ax2.twinx()
        ax2b.plot(epochs, history['val_rmse'], 'r-', alpha=0.7, label='Val RMSE')
        ax2b.plot(epochs, history['val_trend_corr'], 'g-', alpha=0.7,
                  label='Val TrendCorr')
        ax2b.set_ylabel('RMSE / TrendCorr', color='tab:red')
        ax2b.legend(loc='upper right', fontsize=8)

    for i in range(min(3, preds.shape[0])):
        axes[2].plot(targets[i], 'b-', alpha=0.6, label='True' if i == 0 else '')
        axes[2].plot(preds[i], 'r--', alpha=0.6, label='Pred' if i == 0 else '')
    axes[2].set_xlabel('Forecast Day'); axes[2].set_ylabel('CDOM')
    axes[2].set_title('Sample Forecasts'); axes[2].legend(); axes[2].grid(True)

    ft = targets.flatten(); fp = preds.flatten()
    axes[3].scatter(ft, fp, s=1, alpha=0.3)
    lo = min(ft.min(), fp.min()); hi = max(ft.max(), fp.max())
    axes[3].plot([lo, hi], [lo, hi], 'r--', linewidth=1)
    axes[3].set_xlabel('Observed'); axes[3].set_ylabel('Predicted')
    axes[3].set_title(f"Scatter (R²={metrics.get('R2', np.nan):.3f})")
    axes[3].grid(True)

    axes[4].hist(fp - ft, bins=50)
    axes[4].set_xlabel('Pred - True'); axes[4].set_ylabel('Count')
    axes[4].set_title('Error Distribution'); axes[4].grid(True)

    hb = list(range(0, min(preds.shape[1], 365) + 1, 30))
    h_rmse, h_lbl = [], []
    for i in range(len(hb) - 1):
        s, e = hb[i], min(hb[i + 1], preds.shape[1])
        if s >= preds.shape[1]:
            break
        h_lbl.append(f'{s+1}-{e}d')
        h_rmse.append(np.sqrt(mean_squared_error(
            targets[:, s:e].flatten(), preds[:, s:e].flatten())))
    axes[5].bar(range(len(h_rmse)), h_rmse, tick_label=h_lbl)
    axes[5].set_xlabel('Horizon'); axes[5].set_ylabel('RMSE')
    axes[5].set_title('RMSE by Horizon')
    axes[5].tick_params(axis='x', rotation=45); axes[5].grid(True, axis='y')

    if test_meta is not None:
        mt, mp = defaultdict(list), defaultdict(list)
        for i in range(preds.shape[0]):
            start = test_meta[i]['start_date']
            if hasattr(start, 'to_pydatetime'):
                start = start.to_pydatetime()
            for d in range(preds.shape[1]):
                mon = (start + timedelta(days=d)).month
                mt[mon].append(targets[i, d]); mp[mon].append(preds[i, d])
        months = sorted(mt.keys())
        axes[6].plot(months, [np.mean(mt[m]) for m in months], 'o-', label='True')
        axes[6].plot(months, [np.mean(mp[m]) for m in months], 's--', label='Pred')
        axes[6].set_xticks(months)
        axes[6].set_xlabel('Month'); axes[6].set_ylabel('Mean CDOM')
        axes[6].set_title('Seasonal Cycle'); axes[6].legend(); axes[6].grid(True)
    else:
        axes[6].plot(targets.mean(axis=0), label='Mean True')
        axes[6].plot(preds.mean(axis=0), label='Mean Pred')
        axes[6].set_xlabel('Day'); axes[6].set_ylabel('Mean CDOM')
        axes[6].set_title('Average Profile'); axes[6].legend(); axes[6].grid(True)

    tc_list = []
    for i in range(preds.shape[0]):
        y_t, p_t = smooth_ma(targets[i]), smooth_ma(preds[i])
        if np.std(y_t) > 1e-6 and np.std(p_t) > 1e-6:
            tc_list.append(np.corrcoef(y_t, p_t)[0, 1])
    axes[7].hist(tc_list, bins=20)
    axes[7].set_xlabel('TrendCorr per Sample'); axes[7].set_ylabel('Count')
    axes[7].set_title(f"Per-sample TrendCorr "
                      f"(mean={metrics.get('TrendCorr', 0):.3f})")
    axes[7].grid(True)

    err = preds - targets
    n_show = min(50, err.shape[0])
    ds = max(1, err.shape[1] // 73)
    im = axes[8].imshow(err[:n_show, ::ds].T, aspect='auto',
                        cmap='RdBu_r', origin='lower')
    axes[8].set_xlabel('Sample'); axes[8].set_ylabel('Forecast Day (downsampled)')
    axes[8].set_title('Error Heatmap'); fig.colorbar(im, ax=axes[8])

    fig.suptitle(
        f"RMSE={metrics.get('RMSE', 0):.4f}  MAE={metrics.get('MAE', 0):.4f}  "
        f"R²={metrics.get('R2', 0):.3f}  nRMSE={metrics.get('nRMSE', 0):.3f}  "
        f"TrendCorr={metrics.get('TrendCorr', 0):.3f}", fontsize=14)
    plt.tight_layout()
    _safe_savefig(fig, f'{save_prefix}.png', dpi=150)
    _safe_savefig(fig, f'{save_prefix}_detailed.png', dpi=300)
    plt.close(fig)
    print(f"图已保存: {save_prefix}.png / {save_prefix}_detailed.png")


# =============================================================================
# 6b. 季节性检查图 / 相位一致性图
# =============================================================================

def plot_seasonal_by_depth(exp_names, save_path=None):
    if save_path is None:
        save_path = CACHE_ROOT / 'seasonal_by_depth.png'

    MIN_OBS_PER_BUOY   = 100
    MIN_MONTHS_COVERED = 6

    n = len(exp_names)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    summary = {}
    for i, name in enumerate(exp_names):
        exp = EXPERIMENTS.get(name)
        if exp is None:
            print(f"[{name}] 未知实验，跳过")
            continue
        proc = LongTermDataProcessor(exp)
        if not proc.load_data(use_cache=True):
            print(f"[{name}] 缓存缺失，跳过")
            continue

        buoy_rel_amps, buoy_strengths, buoy_amps, buoy_within_std = [], [], [], []
        monthly_anom    = defaultdict(list)
        monthly_raw     = defaultdict(list)
        all_cdom        = []

        for bid, pts in proc.buoy_data.items():
            if len(pts) < MIN_OBS_PER_BUOY:
                continue
            m_dict = defaultdict(list)
            vals   = []
            for p in pts:
                m_dict[p['timestamp'].month].append(p['raw_cdom'])
                vals.append(p['raw_cdom'])
            if len(m_dict) < MIN_MONTHS_COVERED:
                continue

            vals = np.asarray(vals, dtype=float)
            mean_i = float(vals.mean()); std_i = float(vals.std())
            all_cdom.extend(vals.tolist())

            monthly_means = np.array([np.mean(m_dict[m]) for m in sorted(m_dict.keys())])
            amp_i = float(monthly_means.max() - monthly_means.min())

            buoy_amps.append(amp_i)
            if abs(mean_i) > 1e-6:
                buoy_rel_amps.append(amp_i / abs(mean_i))
            if std_i > 1e-6:
                buoy_strengths.append(amp_i / std_i)
                buoy_within_std.append(std_i)

            for m in sorted(m_dict.keys()):
                monthly_anom[m].append(float(np.mean(m_dict[m])) - mean_i)
                monthly_raw[m].append(float(np.mean(m_dict[m])))

        if not buoy_strengths:
            print(f"[{name}] 无浮标满足数据覆盖要求，跳过")
            continue

        months = sorted(monthly_anom.keys())
        anom_means = np.array([np.mean(monthly_anom[m]) for m in months])
        anom_stds  = np.array([np.std(monthly_anom[m])  for m in months])
        ns         = np.array([len(monthly_anom[m])     for m in months])

        seasonal_amp  = float(np.mean(buoy_amps))
        rel_amp       = float(np.mean(buoy_rel_amps)) if buoy_rel_amps else float('nan')
        within_std    = float(np.mean(buoy_within_std))
        strength      = float(np.median(buoy_strengths))

        summary[name] = {
            'n_buoy_valid':   int(len(buoy_strengths)),
            'n_cdom_total':   int(len(all_cdom)),
            'seasonal_amp':   seasonal_amp,
            'rel_amp':        rel_amp,
            'within_std':     within_std,
            'strength':       strength,
            'monthly_anom':   anom_means.tolist(),
            'monthly_anom_std': anom_stds.tolist(),
            'monthly_n':      ns.tolist(),
            'months':         months,
        }

        ax = axes[i]
        ax.errorbar(months, anom_means, yerr=anom_stds, fmt='o-', capsize=4,
                    label='Monthly mean anomaly ± std')
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.6)
        ax.set_xticks(range(1, 13))
        ax.set_xlabel('Month'); ax.set_ylabel('CDOM anomaly')
        ax.set_title(f"{name}  strength={strength:.3f}  rel_amp={rel_amp:.3f}  "
                     f"n_buoy={len(buoy_strengths)}")
        ax.grid(True, alpha=0.4); ax.legend(fontsize=8)

    for j in range(len(exp_names), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Monthly CDOM anomaly by depth (buoy-mean removed)', fontsize=14)
    plt.tight_layout()
    actual = _safe_savefig(fig, save_path, dpi=150)
    plt.close(fig)
    print(f"\n季节性检查图已保存: {actual}")

    print(f"\n{'exp':12s} {'strength':>10s} {'rel_amp':>10s} "
          f"{'within_std':>12s} {'season_amp':>12s} "
          f"{'n_buoy':>8s} {'n_cdom':>10s}  verdict")
    for name, s in summary.items():
        if s['strength'] > 1.0:   verdict = 'Strong'
        elif s['strength'] > 0.5: verdict = 'Moderate'
        elif s['strength'] > 0.3: verdict = 'Weak'
        else:                     verdict = 'None'
        print(f"{name:12s} {s['strength']:10.3f} {s['rel_amp']:10.3f} "
              f"{s['within_std']:12.4f} {s['seasonal_amp']:12.4f} "
              f"{s['n_buoy_valid']:8d} {s['n_cdom_total']:10d}  {verdict}")

    out_json = CACHE_ROOT / 'seasonal_strength.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n季节性强度已保存: {out_json}")
    return summary


def diagnose_phase_consistency(exp_names, save_path=None):
    if save_path is None:
        save_path = CACHE_ROOT / 'phase_consistency.png'

    n = len(exp_names)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows),
                             squeeze=False, subplot_kw={'projection': 'polar'})
    axes = axes.flatten()

    summary = {}
    for i, name in enumerate(exp_names):
        exp = EXPERIMENTS.get(name)
        if exp is None:
            continue
        proc = LongTermDataProcessor(exp)
        if not proc.load_data(use_cache=True):
            continue

        w1 = 2 * np.pi / 365.25
        peaks, amps = [], []
        for bid, pts in proc.buoy_data.items():
            if len(pts) < 200:
                continue
            t = np.array([p['timestamp'].timetuple().tm_yday for p in pts], dtype=float)
            v = np.array([p['raw_cdom'] for p in pts], dtype=float)
            D = np.stack([np.ones_like(t), np.sin(w1*t), np.cos(w1*t)], 1)
            coef, *_ = np.linalg.lstsq(D, v, rcond=None)
            a, b = coef[1], coef[2]
            A = float(np.sqrt(a**2 + b**2))
            if A < 1e-4:
                continue
            phi = float(np.arctan2(a, b))
            doy_peak = (phi / w1) % 365.25
            peaks.append(doy_peak)
            amps.append(A)

        if not peaks:
            axes[i].set_title(f"{name}\n(no valid buoy)")
            continue

        peaks = np.array(peaks); amps = np.array(amps)
        theta = 2 * np.pi * peaks / 365.25
        R = float(np.abs(np.mean(np.exp(1j * theta))))

        ax = axes[i]
        weights = amps / amps.max()
        ax.scatter(theta, weights, s=30, alpha=0.6)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                        'Jul','Aug','Sep','Oct','Nov','Dec']
        ax.set_xticklabels(month_labels, fontsize=8)
        ax.set_title(f"{name}\nR={R:.3f}  n={len(peaks)}", pad=15)

        if R > 0.5:   verdict = 'Coherent'
        elif R > 0.3: verdict = 'Moderate'
        else:         verdict = 'Dispersed'
        summary[name] = {
            'phase_R': R, 'n_buoy': len(peaks),
            'mean_amp': float(amps.mean()), 'verdict': verdict,
        }

    for j in range(len(exp_names), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Per-buoy annual harmonic phase (polar; radius = amplitude)\n'
                 'R closer to 1 = phases more concentrated', fontsize=12)
    plt.tight_layout()
    actual = _safe_savefig(fig, save_path, dpi=150)
    plt.close(fig)
    print(f"\n相位一致性图已保存: {actual}")

    print(f"\n{'exp':12s} {'phase_R':>10s} {'n_buoy':>8s} "
          f"{'mean_amp':>10s}  verdict")
    for name, s in summary.items():
        print(f"{name:12s} {s['phase_R']:10.3f} {s['n_buoy']:8d} "
              f"{s['mean_amp']:10.4f}  {s['verdict']}")
    return summary


# =============================================================================
# 7. 数据管线辅助
# =============================================================================

def per_sample_stats(X: np.ndarray):
    hist = X[:, -365:, 3]
    mu = hist.mean(axis=1, keepdims=True).astype(np.float32)
    rel_floor = 0.05 * np.abs(mu) + 1e-4
    sigma = np.maximum(hist.std(axis=1, keepdims=True), rel_floor).astype(np.float32)
    return mu, sigma


def _split_chrono(seqs, tars, meta, gap: int = 0):
    n = len(seqs)
    order = sorted(range(n), key=lambda i: meta[i]['start_date'])
    seqs, tars = seqs[order], tars[order]
    meta = [meta[i] for i in order]
    tr = int(0.7 * n); va = int(0.15 * n)
    if gap > 0:
        tr_eff = max(0, tr - gap // 2)
        va_eff_s = tr + gap // 2
        va_eff_e = max(va_eff_s, tr + va - gap // 2)
        te_eff_s = tr + va + gap // 2
        return (seqs[:tr_eff], tars[:tr_eff], meta[:tr_eff],
                seqs[va_eff_s:va_eff_e], tars[va_eff_s:va_eff_e], meta[va_eff_s:va_eff_e],
                seqs[te_eff_s:], tars[te_eff_s:], meta[te_eff_s:])
    return (seqs[:tr], tars[:tr], meta[:tr],
            seqs[tr:tr + va], tars[tr:tr + va], meta[tr:tr + va],
            seqs[tr + va:], tars[tr + va:], meta[tr + va:])


def _split_by_buoy(seqs, tars, meta, seed=42, ratios=(0.7, 0.15, 0.15), gap=0):
    _ = gap
    rng = np.random.RandomState(seed)
    buoy_set = sorted(set(m['buoy_id'] for m in meta))
    rng.shuffle(buoy_set)
    n = len(buoy_set)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    set_tr = set(buoy_set[:n_tr])
    set_va = set(buoy_set[n_tr:n_tr + n_va])
    set_te = set(buoy_set[n_tr + n_va:])
    idx_tr = [i for i, m in enumerate(meta) if m['buoy_id'] in set_tr]
    idx_va = [i for i, m in enumerate(meta) if m['buoy_id'] in set_va]
    idx_te = [i for i, m in enumerate(meta) if m['buoy_id'] in set_te]
    print(f"  [按浮标切分] 训练浮标={len(set_tr)}，"
          f"验证浮标={len(set_va)}，测试浮标={len(set_te)}")
    return (seqs[idx_tr], tars[idx_tr], [meta[i] for i in idx_tr],
            seqs[idx_va], tars[idx_va], [meta[i] for i in idx_va],
            seqs[idx_te], tars[idx_te], [meta[i] for i in idx_te])


def _do_split(seqs, tars, meta, gap: int = 0):
    if SPLIT_MODE == 'buoy':
        return _split_by_buoy(seqs, tars, meta, gap=gap)
    return _split_chrono(seqs, tars, meta, gap=gap)


def _build_local_datasets(Xtr, ytr, Xva, yva, Xte, yte, batch_size):
    fs = StandardScaler()
    fs.fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr_s = fs.transform(Xtr.reshape(-1, Xtr.shape[2])).reshape(Xtr.shape)
    Xva_s = fs.transform(Xva.reshape(-1, Xva.shape[2])).reshape(Xva.shape)
    Xte_s = fs.transform(Xte.reshape(-1, Xte.shape[2])).reshape(Xte.shape)

    mu_tr, sg_tr = per_sample_stats(Xtr)
    mu_va, sg_va = per_sample_stats(Xva)
    mu_te, sg_te = per_sample_stats(Xte)
    CLIP_VAL = 10.0
    Xtr_s[:, :, 3] = np.clip((Xtr[:, :, 3] - mu_tr) / sg_tr, -CLIP_VAL, CLIP_VAL)
    Xva_s[:, :, 3] = np.clip((Xva[:, :, 3] - mu_va) / sg_va, -CLIP_VAL, CLIP_VAL)
    Xte_s[:, :, 3] = np.clip((Xte[:, :, 3] - mu_te) / sg_te, -CLIP_VAL, CLIP_VAL)
    ytr_n = np.clip((ytr - mu_tr) / sg_tr, -CLIP_VAL, CLIP_VAL)
    yva_n = np.clip((yva - mu_va) / sg_va, -CLIP_VAL, CLIP_VAL)
    yte_n = np.clip((yte - mu_te) / sg_te, -CLIP_VAL, CLIP_VAL)

    tr_ds = AugDataset(Xtr_s, ytr_n, noise=0.03, noise_prob=0.5)
    va_ds = torch.utils.data.TensorDataset(torch.FloatTensor(Xva_s),
                                           torch.FloatTensor(yva_n))
    te_ds = torch.utils.data.TensorDataset(torch.FloatTensor(Xte_s),
                                           torch.FloatTensor(yte_n))
    return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True),
            DataLoader(va_ds, batch_size=batch_size, shuffle=False),
            DataLoader(te_ds, batch_size=batch_size, shuffle=False),
            mu_va, sg_va, mu_te, sg_te)


def _build_global_datasets(Xtr, ytr, Xva, yva, Xte, yte, batch_size):
    fs = StandardScaler()
    fs.fit(Xtr.reshape(-1, Xtr.shape[2]))
    ts = StandardScaler()
    ts.fit(ytr.reshape(-1, 1))
    Xtr_s = fs.transform(Xtr.reshape(-1, Xtr.shape[2])).reshape(Xtr.shape)
    Xva_s = fs.transform(Xva.reshape(-1, Xva.shape[2])).reshape(Xva.shape)
    Xte_s = fs.transform(Xte.reshape(-1, Xte.shape[2])).reshape(Xte.shape)
    for Xr, Xs in [(Xtr, Xtr_s), (Xva, Xva_s), (Xte, Xte_s)]:
        Xs[:, :, 3] = ts.transform(
            Xr[:, :, 3].reshape(-1, 1)).reshape(Xr.shape[0], Xr.shape[1])
    ytr_n = ts.transform(ytr.reshape(-1, 1)).reshape(ytr.shape)
    yva_n = ts.transform(yva.reshape(-1, 1)).reshape(yva.shape)
    yte_n = ts.transform(yte.reshape(-1, 1)).reshape(yte.shape)
    tr_ds = AugDataset(Xtr_s, ytr_n, noise=0.03, noise_prob=0.5)
    va_ds = torch.utils.data.TensorDataset(torch.FloatTensor(Xva_s),
                                           torch.FloatTensor(yva_n))
    te_ds = torch.utils.data.TensorDataset(torch.FloatTensor(Xte_s),
                                           torch.FloatTensor(yte_n))
    return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True),
            DataLoader(va_ds, batch_size=batch_size, shuffle=False),
            DataLoader(te_ds, batch_size=batch_size, shuffle=False), ts)


# =============================================================================
# 8. 无训练基线
# =============================================================================

def _baseline_persist(X, forecast_days=FORECAST_DAYS):
    return np.repeat(X[:, -1:, 3], forecast_days, axis=1)


def _baseline_seas_persist(X, forecast_days=FORECAST_DAYS):
    L = X.shape[1]
    start = max(0, L - forecast_days)
    return X[:, start:L, 3].copy()


def _baseline_harmonic(X, forecast_days=FORECAST_DAYS):
    w1, w2, w3 = (2*np.pi/365.25, 4*np.pi/365.25, 6*np.pi/365.25)
    L = X.shape[1]
    t = np.arange(L, dtype=float)
    tt = np.arange(L, L + forecast_days, dtype=float)
    D = np.stack([np.ones(L), np.sin(w1*t), np.cos(w1*t),
                  np.sin(w2*t), np.cos(w2*t),
                  np.sin(w3*t), np.cos(w3*t)], 1)
    Df = np.stack([np.ones(forecast_days), np.sin(w1*tt), np.cos(w1*tt),
                   np.sin(w2*tt), np.cos(w2*tt),
                   np.sin(w3*tt), np.cos(w3*tt)], 1)
    inv = np.linalg.pinv(D)
    return np.array([Df @ (inv @ X[i, :, 3]) for i in range(X.shape[0])],
                    dtype=np.float32)


def _baseline_dlinear(X, forecast_days=FORECAST_DAYS, trend_win=365, slope_win=180):
    L = X.shape[1]
    out = []
    for i in range(X.shape[0]):
        c = X[i, :, 3]
        pad = trend_win // 2
        trend = np.convolve(np.pad(c, (pad, pad), mode='edge'),
                            np.ones(trend_win) / trend_win, mode='valid')[:L]
        seasonal = c - trend
        tail = trend[-slope_win:]
        sl = np.polyfit(np.arange(slope_win), tail, 1)[0]
        tt = np.arange(L, L + forecast_days, dtype=float)
        trend_fut = trend[-1] + sl * (tt - (L - 1))
        seas_fut = seasonal[L - forecast_days:].copy()
        out.append(trend_fut + seas_fut)
    return np.array(out, dtype=np.float32)


def _baseline_clim(proc, X, meta_list, forecast_days=FORECAST_DAYS, min_points=30):
    out = _baseline_harmonic(X, forecast_days)
    for i in range(X.shape[0]):
        tm = meta_list[i]; bid = tm['buoy_id']; start = tm['start_date']
        if hasattr(start, 'to_pydatetime'):
            start = start.to_pydatetime()
        pts = proc.buoy_data.get(bid, [])
        pre = [p for p in pts if p['timestamp'] < start]
        if len(pre) < min_points:
            continue
        doys = np.array([p['timestamp'].timetuple().tm_yday for p in pre])
        cvals = np.array([p['raw_cdom'] for p in pre])
        prof = np.full(367, np.nan)
        for d in range(1, 367):
            diff = np.abs(((doys - d + 183) % 366) - 183)
            sel = diff <= 15
            if sel.sum() > 0:
                prof[d] = np.mean(cvals[sel])
        idx = np.where(~np.isnan(prof))[0]
        if len(idx) > 1:
            prof = np.interp(np.arange(367), idx, prof[idx])
        fc = np.array([prof[(start + timedelta(days=k)).timetuple().tm_yday]
                       for k in range(forecast_days)])
        if np.all(np.isfinite(fc)) and np.std(fc) > 1e-9:
            out[i] = fc
    return out


# =============================================================================
# 9. 架构工厂
# =============================================================================

_ABLATION_KW = {
    'no_patch':   {'use_patch':    False},
    'no_var':     {'use_var':      False},
    'no_conv':    {'use_conv':     False},
    'no_freq':    {'use_freq':     False},
    'no_prior':   {'use_prior':    False},
    'no_coefadj': {'use_coef_adj': False},
}


def build_model_for_arch(arch, input_dim, seq_len=SEQ_LEN,
                         forecast_days=FORECAST_DAYS, dropout=DROPOUT):
    if arch == 'full':
        return ImprovedLongTermModel(input_dim, seq_len, forecast_days,
                                     dropout=dropout)
    if arch in _ABLATION_KW:
        return ImprovedLongTermModel(input_dim, seq_len, forecast_days,
                                     dropout=dropout, **_ABLATION_KW[arch])
    if arch == 'lstm':
        return LSTMBaseline(input_dim, seq_len, forecast_days)
    if arch == 'patchtst':
        return PatchTSTBranch(input_dim, seq_len, forecast_days)
    if arch == 'itransformer':
        return VariableAttentionBranch(input_dim, seq_len, forecast_days)
    if arch == 'v1_hybrid':
        return V1HybridModel(input_dim, seq_len, forecast_days)
    raise ValueError(f'未知架构: {arch}')


# =============================================================================
# 10. 训练流程
# =============================================================================

METRIC_KEYS = [
    'RMSE', 'MAE', 'R2', 'nRMSE',
    'R2_sample_mean', 'R2_sample_n', 'R2_sample_skip',
    'TrendCorr', 'PeakError',
    'RMSE_30d', 'RMSE_90d', 'RMSE_180d', 'RMSE_365d',
    'RMSE_p50', 'RMSE_p90', 'RMSE_max',
    'RMSE_CI_low', 'RMSE_CI_high',
    'MAE_CI_low',  'MAE_CI_high',
    'R2_CI_low',   'R2_CI_high',
    'TC_CI_low',   'TC_CI_high',
    'n_S', 'n_T', 'n_N',
    'TC_S', 'TC_T', 'TC_N',
    'RMSE_S', 'RMSE_T', 'RMSE_N',
]


def _append_csv(path, row):
    fieldnames = ['model', 'type', 'params', 'epochs', 'time_s'] + METRIC_KEYS

    # ★ 新增：若现有 CSV 的 header 与新 fieldnames 不一致，重写整个文件
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header != fieldnames:
                # 表头不匹配 → 读旧数据，清洗后重写
                print(f"  [CSV] 表头不匹配，重建文件: {os.path.basename(path)}")
                old_rows = []
                with open(path, 'r', encoding='utf-8') as f:
                    for r in csv.DictReader(f):
                        old_rows.append(r)
                # 用新表头重写（旧行缺失字段自动填 ''）
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames,
                                       extrasaction='ignore')
                    w.writeheader()
                    for r in old_rows:
                        w.writerow({k: r.get(k, '') for k in fieldnames})
        except Exception:
            pass

    new_file = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if new_file:
            w.writeheader()
        w.writerow(row)


def _cleanup(model, trainer):
    try:
        if trainer is not None and hasattr(trainer, 'model'):
            trainer.model.to('cpu')
    except Exception:
        pass
    try:
        if model is not None:
            model.cpu()
    except Exception:
        pass
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_training(exp: DepthExperiment):
    print(f"\n===== [{exp.name}] 训练 =====")
    print(f"数据目录: {exp.data_dir}")
    print(f"深度范围: {exp.depth_range}")
    print(f"输出前缀: {exp.out_prefix}")

    proc = LongTermDataProcessor(exp, seq_len=SEQ_LEN,
                                 forecast_days=FORECAST_DAYS, step=STEP)
    if not proc.load_data(use_cache=True):
        print(f"[{exp.name}] 数据加载失败"); return

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

    print(f"样本数: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}"
          f"（gap={gap_samples}）")
    if len(Xtr) == 0 or len(Xte) == 0:
        print(f"[{exp.name}] 样本不足，跳过"); return

    tr_l, va_l, te_l, mu_va, sg_va, mu_te, sg_te = _build_local_datasets(
        Xtr, ytr, Xva, yva, Xte, yte, BATCH_SIZE)

    all_preds = []
    targets_orig = None
    best_history = None
    use_amp = USE_AMP and torch.cuda.is_available()
    eff_epochs   = exp.epochs_override   if exp.epochs_override   else EPOCHS_TRAIN
    eff_patience = exp.patience_override if exp.patience_override else PATIENCE_TRAIN

    for si, seed in enumerate(SEEDS):
        print(f"\n--- 模型 {si+1}/{len(SEEDS)} (seed={seed}) ---")
        set_seed(seed)
        model = ImprovedLongTermModel(input_dim=n_features, seq_len=SEQ_LEN,
                                      forecast_days=FORECAST_DAYS, dropout=DROPOUT)
        print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
        trainer = LongTermTrainer(model)
        model_path = (f'{exp.out_prefix}_s{seed}.pt'
                      if len(SEEDS) > 1 else f'{exp.out_prefix}.pt')

        hist = trainer.train(tr_l, va_l, mu_va, sg_va,
                             epochs=eff_epochs, lr=LR,
                             weight_decay=WEIGHT_DECAY,
                             patience=eff_patience, warmup=WARMUP,
                             use_amp=use_amp, model_path=model_path)
        best_history = hist

        preds_i, targets_i = trainer.predict(te_l, mu_te, sg_te)
        all_preds.append(preds_i)
        if targets_orig is None:
            targets_orig = targets_i
        if len(SEEDS) > 1:
            m_i = LongTermTrainer.metrics_from(preds_i, targets_i,
                                               test_meta=te_meta)
            print(f"    seed={seed}: RMSE={m_i['RMSE']:.4f} R2={m_i['R2']:.4f} "
                  f"nRMSE={m_i['nRMSE']:.4f} TC={m_i['TrendCorr']:.4f}")

        _cleanup(model, trainer)
        model = None; trainer = None

    preds_orig = np.mean(all_preds, axis=0) if len(all_preds) > 1 else all_preds[0]
    metrics = LongTermTrainer.metrics_from(preds_orig, targets_orig,
                                           test_meta=te_meta)

    print(f"\n===== [{exp.name}] 测试集指标 =====")
    for k, v in metrics.items():
        try:
            print(f"  {k:16s}: {float(v):.4f}")
        except (TypeError, ValueError):
            print(f"  {k:16s}: {v}")

    plot_results(best_history, metrics, preds_orig, targets_orig, te_meta,
                 save_prefix=exp.out_prefix)
    np.savez(f'{exp.out_prefix}.npz', preds=preds_orig, targets=targets_orig,
             metrics=metrics, exp_name=exp.name)
    print(f"结果已保存: {exp.out_prefix}.npz")


# =============================================================================
# 11. 消融实验
# =============================================================================

def run_experiments(exp: DepthExperiment):
    print(f"\n===== [{exp.name}] 对比与消融实验 =====")
    eff_epochs   = exp.epochs_override   if exp.epochs_override   else EPOCHS_EXP
    eff_patience = exp.patience_override if exp.patience_override else PATIENCE_EXP
    print(f"协议: STEP={STEP}, epochs≤{eff_epochs}, patience={eff_patience}, "
          f"切分={SPLIT_MODE} 70/15/15")

    proc = LongTermDataProcessor(exp, seq_len=SEQ_LEN,
                                 forecast_days=FORECAST_DAYS, step=STEP)
    if not proc.load_data(use_cache=True):
        print(f"[{exp.name}] 数据加载失败"); return
    seqs, tars, meta = proc.prepare_sequences(
        min_sequence_length=SEQ_LEN + FORECAST_DAYS, apply_smoothing=False,
        max_gap_days=MAX_GAP_DAYS, interpolate_limit_days=MAX_GAP_DAYS)
    if seqs is None:
        return
    gap_samples = max(0, (SEQ_LEN // max(STEP, 1)) - 1)
    (Xtr, ytr, _, Xva, yva, _, Xte, yte, te_meta) = _do_split(
        seqs, tars, meta, gap=gap_samples)

    print(f"样本: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}"
          f"（gap={gap_samples}）")
    if len(Xtr) == 0 or len(Xte) == 0:
        print(f"[{exp.name}] 样本不足，跳过"); return

    done = set()
    if os.path.exists(exp.csv_path) and not FORCE_RERUN:
        try:
            with open(exp.csv_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    done.add(row['model'])
        except Exception:
            done = set()
    print(f"已完成: {sorted(done) if done else '无'}")

    results = []
    use_amp = USE_AMP and torch.cuda.is_available()

    # ---------------- 无训练基线 ----------------
    analytic = [
        ('persist',      lambda: _baseline_persist(Xte, FORECAST_DAYS)),
        ('seas_persist', lambda: _baseline_seas_persist(Xte, FORECAST_DAYS)),
        ('harmonic',     lambda: _baseline_harmonic(Xte, FORECAST_DAYS)),
        ('dlinear',      lambda: _baseline_dlinear(Xte, FORECAST_DAYS)),
        ('clim',         lambda: _baseline_clim(proc, Xte, te_meta, FORECAST_DAYS)),
    ]
    for name, fn in analytic:
        if ONLY_EXP and name != ONLY_EXP:
            continue
        if name in done and not FORCE_RERUN:
            continue
        t0 = time.time()
        preds = fn()
        m = LongTermTrainer.metrics_from(preds, yte, test_meta=te_meta)
        row = {'model': name, 'type': 'baseline', 'params': 0, 'epochs': 0,
               'time_s': round(time.time() - t0, 1), **m}
        results.append(row)
        print(f"[baseline] {name:14s} RMSE={m['RMSE']:.4f} nRMSE={m['nRMSE']:.4f} "
              f"TC={m['TrendCorr']:.4f}")
        _append_csv(exp.csv_path, row)

    # ---------------- 学习模型 ----------------
    tr_l, va_l, te_l, mu_va, sg_va, mu_te, sg_te = _build_local_datasets(
        Xtr, ytr, Xva, yva, Xte, yte, BATCH_SIZE)

    learned = [
        ('lstm',         'learned',  {}),
        ('patchtst',     'learned',  {}),
        ('itransformer', 'learned',  {}),
        ('v1_hybrid',    'learned',  {}),
        ('full',         'learned',  {}),
        ('no_patch',     'learned',  {}),
        ('no_var',       'learned',  {}),
        ('no_conv',      'learned',  {}),
        ('no_freq',      'learned',  {}),
        ('no_prior',     'learned',  {}),
        ('no_coefadj',   'learned',  {}),
        ('no_trendloss', 'learned',  {'use_trend_terms': False}),
        ('no_aug',       'learned',  {}),
    ]
    for name, typ, kw in learned:
        if ONLY_EXP and name != ONLY_EXP:
            continue
        if name in done and not FORCE_RERUN:
            continue
        print(f"\n>>> 训练 {name} ...")
        set_seed(42)
        arch_name = 'full' if name in ('no_trendloss', 'no_aug') else name
        try:
            model = build_model_for_arch(arch_name, Xtr.shape[2], SEQ_LEN,
                                         FORECAST_DAYS, dropout=DROPOUT)
        except ValueError as e:
            print(f"[skip] {name}: {e}")
            continue
        n_params = sum(p.numel() for p in model.parameters())
        trainer = LongTermTrainer(model)
        t0 = time.time()

        tr_loader = tr_l
        if name == 'no_aug' and hasattr(tr_l.dataset, 'X'):
            Xa = tr_l.dataset.X.numpy()
            ya = tr_l.dataset.y.numpy()
            tr_loader = DataLoader(AugDataset(Xa, ya, noise=0.0, noise_prob=0.0),
                                   batch_size=BATCH_SIZE, shuffle=True)

        try:
            hist = trainer.train(
                tr_loader, va_l, mu_va, sg_va,
                epochs=eff_epochs, lr=LR, weight_decay=WEIGHT_DECAY,
                patience=eff_patience, warmup=WARMUP, use_amp=use_amp,
                model_path=f'exp_{exp.name}_{SPLIT_MODE}_{name}.pt',
                use_trend_terms=kw.get('use_trend_terms', True), verbose=False)
        except Exception as e:
            print(f"[fail] {name}: {e}")
            _cleanup(model, trainer)
            model = None; trainer = None
            continue

        preds, targets = trainer.predict(te_l, mu_te, sg_te)
        m = LongTermTrainer.metrics_from(preds, targets, test_meta=te_meta)
        row = {'model': name, 'type': typ, 'params': n_params,
               'epochs': len(hist['train_loss']),
               'time_s': round(time.time() - t0, 1), **m}
        results.append(row)
        print(f"[learned] {name:14s} RMSE={m['RMSE']:.4f} nRMSE={m['nRMSE']:.4f} "
              f"R2={m['R2']:.4f} TC={m['TrendCorr']:.4f} "
              f"TC_CI=[{m.get('TC_CI_low', float('nan')):.3f}, "
              f"{m.get('TC_CI_high', float('nan')):.3f}] "
              f"RMSE_30d={m['RMSE_30d']:.4f} RMSE_365d={m['RMSE_365d']:.4f} "
              f"epochs={len(hist['train_loss'])}")
        _append_csv(exp.csv_path, row)

        _cleanup(model, trainer)
        model = None; trainer = None

    # ---------------- global_norm 消融 ----------------
    if (not ONLY_EXP or ONLY_EXP == 'global_norm') and \
            ('global_norm' not in done or FORCE_RERUN):
        print(f"\n>>> 训练 global_norm ...")
        set_seed(42)
        model = None; trainer = None
        try:
            tr_g, va_g, te_g, ts_g = _build_global_datasets(
                Xtr, ytr, Xva, yva, Xte, yte, BATCH_SIZE)
            model = ImprovedLongTermModel(Xtr.shape[2], SEQ_LEN,
                                          FORECAST_DAYS, dropout=DROPOUT)
            n_params = sum(p.numel() for p in model.parameters())
            trainer = LongTermTrainer(model)
            trainer.target_scaler = ts_g
            t0 = time.time()
            hist = trainer.train(
                tr_g, va_g, None, None,
                epochs=eff_epochs, lr=LR, weight_decay=WEIGHT_DECAY,
                patience=eff_patience, warmup=WARMUP, use_amp=use_amp,
                model_path=f'exp_{exp.name}_{SPLIT_MODE}_global_norm.pt',
                verbose=False)
            preds, targets = trainer.predict(te_g, None, None)
            m = LongTermTrainer.metrics_from(preds, targets, test_meta=te_meta)
            row = {'model': 'global_norm', 'type': 'ablation', 'params': n_params,
                   'epochs': len(hist['train_loss']),
                   'time_s': round(time.time() - t0, 1), **m}
            results.append(row)
            print(f"[ablation] global_norm    RMSE={m['RMSE']:.4f} "
                  f"nRMSE={m['nRMSE']:.4f} R2={m['R2']:.4f} TC={m['TrendCorr']:.4f}")
            _append_csv(exp.csv_path, row)
        except Exception as e:
            print(f"[fail] global_norm: {e}")
        finally:
            _cleanup(model, trainer)
            model = None; trainer = None

    # ---------------- 合并 + 报告 ----------------
    merged = {}
    if os.path.exists(exp.csv_path):
        try:
            with open(exp.csv_path, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    merged[row['model']] = row
        except Exception:
            pass
    for r in results:
        merged[r['model']] = {**merged.get(r['model'], {}), **r}
    for r in merged.values():
        for k in METRIC_KEYS + ['params', 'epochs', 'time_s']:
            if k in r and r[k] not in ('', None):
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    pass
    results = list(merged.values())

    if WRITE_REPORT:
        _write_report(exp.report_path, results, exp, split_mode=SPLIT_MODE)
    print(f"\n报告: {exp.csv_path} / {exp.report_path}")


# =============================================================================
# 12. 报告
# =============================================================================

def _fmt_lat_line(label, n, tc):
    if n is None or (isinstance(n, float) and math.isnan(n)) or n == 0:
        return f"- {label}: n=0"
    if tc is None or (isinstance(tc, float) and math.isnan(tc)):
        return f"- {label}: n={int(n)}, TC=—"
    return f"- {label}: n={int(n)}, TC={tc:.4f}"


def _write_report(path, results, exp: DepthExperiment, split_mode: str = 'chrono'):
    split_desc = {
        'chrono': '按起始日期时间序切分 70/15/15（时序外推）',
        'buoy':   '按浮标随机切分 70/15/15（空间泛化，训练/测试浮标不重叠）',
    }.get(split_mode, split_mode)

    lines = [f"# 长期 CDOM 预测：{exp.name} 对比与消融实验\n",
             "## 实验协议\n",
             f"- 深度范围：{exp.depth_range if exp.depth_range else '全部'} 米",
             f"- 数据目录：`{exp.data_dir}`",
             f"- 730 天输入 → 365 天预测；**{split_desc}**；STEP={STEP}",
             "- 指标：RMSE / MAE / R² / nRMSE / per-sample R² / TrendCorr / "
             "PeakError / 分时域 RMSE",
             f"- Bootstrap CI：n_boot={BOOT_N}, seed={BOOT_SEED}, 95% 置信区间",
             f"- per-sample R² 门槛：std > max({R2_ABS_FLOOR}, "
             f"{R2_REL_FLOOR} * |mean|)",
             "- 纬度带：S (lat < -30)、T (-30 ~ 30)、N (lat > 30)",
             "- 学习模型：AdamW + warmup-cosine，early stopping，seed=42\n"]

    order = ['persist', 'seas_persist', 'harmonic', 'clim', 'dlinear',
             'v1_hybrid', 'lstm', 'patchtst', 'itransformer',
             'full', 'no_patch', 'no_var', 'no_conv', 'no_freq',
             'no_prior', 'no_coefadj', 'no_trendloss', 'no_aug', 'global_norm']
    res_map = {r['model']: r for r in results}

    header = ("| 模型 | 类型 | RMSE | RMSE 95%CI | MAE | R² | "
              "R²(sample) | TrendCorr | TC 95%CI | "
              "TC(S) | TC(T) | TC(N) | n_S | n_T | n_N | "
              "RMSE_30d | RMSE_365d | Params | Epochs |")
    lines += ["## 结果\n", header, "|" + "---|" * 19]
    for name in order:
        if name not in res_map:
            continue
        r = res_map[name]
        rmse_ci = f"[{_fmt_num(r.get('RMSE_CI_low'))}, " \
                  f"{_fmt_num(r.get('RMSE_CI_high'))}]"
        tc_ci   = f"[{_fmt_num(r.get('TC_CI_low'))}, " \
                  f"{_fmt_num(r.get('TC_CI_high'))}]"
        lines.append(
            f"| {name} | {r.get('type','')} | {_fmt_num(r.get('RMSE'))} | "
            f"{rmse_ci} | {_fmt_num(r.get('MAE'))} | {_fmt_num(r.get('R2'))} | "
            f"{_fmt_num(r.get('R2_sample_mean'))} | "
            f"{_fmt_num(r.get('TrendCorr'))} | {tc_ci} | "
            f"{_fmt_num(r.get('TC_S'))} | {_fmt_num(r.get('TC_T'))} | "
            f"{_fmt_num(r.get('TC_N'))} | "
            f"{_fmt_int(r.get('n_S'))} | {_fmt_int(r.get('n_T'))} | "
            f"{_fmt_int(r.get('n_N'))} | "
            f"{_fmt_num(r.get('RMSE_30d'))} | {_fmt_num(r.get('RMSE_365d'))} | "
            f"{int(r['params']):,} | {int(r['epochs'])} |")

    lines.append("")

    # ---- persist 基线增益 ----
    if 'persist' in res_map:
        p_rmse  = float(res_map['persist']['RMSE'])
        p_nrmse = float(res_map['persist'].get('nRMSE', float('nan')))
        lines += ["## Persist 基线对比（零参数基线）\n",
                  f"- persist 基线：RMSE={p_rmse:.4f}，nRMSE={p_nrmse:.4f}",
                  "",
                  "增益 = (persist_RMSE - model_RMSE) / persist_RMSE",
                  "",
                  "| 模型 | RMSE | **persist 增益** | 判定 |",
                  "|---|---|---|---|"]
        for name in order:
            if name not in res_map:
                continue
            r = res_map[name]
            try:
                rmse_v = float(r['RMSE'])
            except (TypeError, ValueError):
                continue
            gain = (p_rmse - rmse_v) / p_rmse if p_rmse > 1e-9 else 0.0
            if name == 'persist':
                verdict = '—'
            elif gain > 0.10:   verdict = '**显著优于** persist'
            elif gain > 0.03:   verdict = '略优于 persist'
            elif gain > -0.03:  verdict = '与 persist 持平'
            else:               verdict = '**差于 persist**'
            lines.append(f"| {name} | {rmse_v:.4f} | "
                         f"{gain*100:+.2f}% | {verdict} |")
        lines.append("")

    # ---- Per-sample RMSE 分布 ----
    if 'full' in res_map:
        r = res_map['full']
        if 'RMSE_p50' in r:
            lines += [
                "## Per-sample RMSE 分布（full 模型）\n",
                f"- p50 (中位样本): {_fmt_num(r.get('RMSE_p50'))}",
                f"- p90: {_fmt_num(r.get('RMSE_p90'))}",
                f"- max (最差样本): {_fmt_num(r.get('RMSE_max'))}",
                "",
                "若 p50 远小于全局 RMSE，说明少数极端样本拉高了平均值。",
                "",
            ]

    # ---- full vs no_prior ----
    if 'full' in res_map and 'no_prior' in res_map:
        full = res_map['full']; nop = res_map['no_prior']
        lines += [
            "## 关键对比（自动生成）\n",
            f"- full vs no_prior：RMSE {_fmt_num(full.get('RMSE'))} vs "
            f"{_fmt_num(nop.get('RMSE'))}，"
            f"TrendCorr {_fmt_num(full.get('TrendCorr'))} vs "
            f"{_fmt_num(nop.get('TrendCorr'))}；",
        ]
        if 'global_norm' in res_map:
            gn = res_map['global_norm']
            lines.append(f"- full vs global_norm：RMSE {_fmt_num(full.get('RMSE'))} "
                         f"vs {_fmt_num(gn.get('RMSE'))}；")
        lines.append("")

    # ---- 纬度带分层（full 模型） ----
    if 'full' in res_map:
        r = res_map['full']
        if any(r.get(f'n_{k}', 0) and not (
                isinstance(r.get(f'n_{k}'), float) and math.isnan(r.get(f'n_{k}')))
               for k in ['S', 'T', 'N']):
            lines += [
                "## 纬度带分层（full 模型）\n",
                _fmt_lat_line("S 半球高纬 (lat < -30)", r.get('n_S'), r.get('TC_S')),
                _fmt_lat_line("T 热带 (-30 ~ 30)",      r.get('n_T'), r.get('TC_T')),
                _fmt_lat_line("N 半球高纬 (lat > 30)",  r.get('n_N'), r.get('TC_N')),
                "",
            ]

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"报告已生成: {path}（{len(res_map)} 个模型）")


# =============================================================================
# 13. 跨深度汇总
# =============================================================================

def summarize_across_depths(exp_names, model='full', split_mode=None):
    sm = split_mode if split_mode is not None else SPLIT_MODE
    rows = []
    for name in exp_names:
        exp = EXPERIMENTS.get(name)
        if exp is None:
            continue
        csv_path = exp.csv_path_for(sm)
        if not os.path.exists(csv_path):
            print(f"[{name}] CSV 不存在（split={sm}）: {csv_path}")
            continue
        with open(csv_path, 'r', encoding='utf-8') as f:
            last_row = None
            for r in csv.DictReader(f):
                if r['model'] == model:
                    last_row = r        # ★ 只保留最后一条
            if last_row is not None:
                row = {'exp': name, 'depth_range': exp.depth_range}
                for k in METRIC_KEYS:
                    if last_row.get(k) not in ('', None):
                        try:
                            row[k] = float(last_row[k])
                        except (TypeError, ValueError):
                            pass
                rows.append(row)


    print(f"\n===== 跨深度汇总 (model={model}, split={sm}) =====")
    if not rows:
        print("（无可汇总结果）"); return rows
    print(f"{'exp':12s} {'RMSE':>8s} {'TC':>8s} "
          f"{'TC_CI':>18s} {'TC(S)':>8s} {'TC(T)':>8s} {'TC(N)':>8s}")
    for r in rows:
        tc_ci = (f"[{r.get('TC_CI_low', float('nan')):.3f}, "
                 f"{r.get('TC_CI_high', float('nan')):.3f}]")
        print(f"{r['exp']:12s} {r.get('RMSE', float('nan')):8.4f} "
              f"{r.get('TrendCorr', float('nan')):8.4f} "
              f"{tc_ci:>18s} "
              f"{r.get('TC_S', float('nan')):8.4f} "
              f"{r.get('TC_T', float('nan')):8.4f} "
              f"{r.get('TC_N', float('nan')):8.4f}")
    out_path = CACHE_ROOT / f'cross_depth_summary_{sm}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"已保存: {out_path}")
    return rows


def _regen_report_from_csv(exp: DepthExperiment, split_mode=None):
    sm = split_mode if split_mode is not None else SPLIT_MODE
    csv_path = exp.csv_path_for(sm)
    if not os.path.exists(csv_path):
        print(f"[{exp.name}] CSV 不存在（split={sm}）: {csv_path}")
        return
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            for k in METRIC_KEYS + ['params', 'epochs', 'time_s']:
                if k in row and row[k] not in ('', None):
                    try:
                        row[k] = float(row[k])
                    except (TypeError, ValueError):
                        pass
            rows.append(row)
    merged = {}
    for r in rows:
        merged[r['model']] = r
    report_path = str(exp._tag('report', split_mode=sm).with_suffix('.md'))
    _write_report(report_path, list(merged.values()), exp, split_mode=sm)


# =============================================================================
# 14. 训练诊断
# =============================================================================

def run_debug_train(exp_name):
    exp = EXPERIMENTS[exp_name]
    print(f"\n===== 训练诊断: {exp_name} =====")

    proc = LongTermDataProcessor(exp, seq_len=SEQ_LEN,
                                 forecast_days=FORECAST_DAYS, step=STEP)
    if not proc.load_data(use_cache=True):
        return
    seqs, tars, meta = proc.prepare_sequences(
        min_sequence_length=SEQ_LEN + FORECAST_DAYS, apply_smoothing=False,
        max_gap_days=MAX_GAP_DAYS, interpolate_limit_days=MAX_GAP_DAYS)
    if seqs is None:
        return
    gap_samples = max(0, (SEQ_LEN // max(STEP, 1)) - 1)
    (Xtr, ytr, _, Xva, yva, _, Xte, yte, te_meta) = _do_split(
        seqs, tars, meta, gap=gap_samples)
    print(f"样本: train={len(Xtr)}, val={len(Xva)}, test={len(Xte)}")

    tr_l, va_l, te_l, mu_va, sg_va, mu_te, sg_te = _build_local_datasets(
        Xtr, ytr, Xva, yva, Xte, yte, BATCH_SIZE)

    print(f"\n标准化后训练目标 y: 均值={ytr.mean():.4f} "
          f"std={((ytr - ytr.mean()) / ytr.std()).std():.4f}")
    y_var = ytr.reshape(len(ytr), -1).std(axis=1)
    print(f"每个训练样本 y 的 std 分布: "
          f"min={y_var.min():.4f} p25={np.percentile(y_var, 25):.4f} "
          f"median={np.median(y_var):.4f} p75={np.percentile(y_var, 75):.4f} "
          f"max={y_var.max():.4f}")
    print(f"y 方差 < 0.05 的样本占比: "
          f"{(y_var < 0.05).sum() / len(y_var) * 100:.1f}%")

    for model_name, kw in [('full', {}), ('no_prior', {'use_prior': False})]:
        print(f"\n----- 训练 {model_name} -----")
        set_seed(42)
        model = ImprovedLongTermModel(Xtr.shape[2], SEQ_LEN, FORECAST_DAYS,
                                      dropout=DROPOUT, **kw)
        trainer = LongTermTrainer(model)
        hist = trainer.train(
            tr_l, va_l, mu_va, sg_va,
            epochs=100, lr=LR, weight_decay=WEIGHT_DECAY,
            patience=80, warmup=5,
            use_amp=False,
            model_path=f'debug_{exp_name}_{model_name}.pt',
            verbose=True)

        print(f"\n{model_name} 前 30 epoch 的 val_rmse:")
        print("  " + " ".join(f"{v:.4f}" for v in hist['val_rmse'][:30]))
        print(f"{model_name} 前 30 epoch 的 val_trend_corr:")
        print("  " + " ".join(f"{v:+.3f}" for v in hist['val_trend_corr'][:30]))

        preds, targets = trainer.predict(te_l, mu_te, sg_te)
        m = LongTermTrainer.metrics_from(preds, targets, test_meta=te_meta)
        print(f"[{model_name}] 测试: RMSE={m['RMSE']:.4f} R2={m['R2']:.4f} "
              f"TC={m['TrendCorr']:.4f} "
              f"TC_CI=[{m['TC_CI_low']:.3f}, {m['TC_CI_high']:.3f}] "
              f"TC(S)={m.get('TC_S', float('nan')):.3f} "
              f"TC(T)={m.get('TC_T', float('nan')):.3f} "
              f"TC(N)={m.get('TC_N', float('nan')):.3f}")

        _cleanup(model, trainer)
        model = None; trainer = None


# =============================================================================
# 15. 入口
# =============================================================================

def main():
    print(f"运行模式: {RUN_MODE}")
    print(f"目标实验: {RUN_EXPS}")

    if RUN_MODE == 'debug_train':
        for t in RUN_EXPS:
            if t in EXPERIMENTS:
                run_debug_train(t)
        return


    if RUN_MODE == 'seasonal':
        plot_seasonal_by_depth(RUN_EXPS)
        diagnose_phase_consistency(RUN_EXPS)
        for name in RUN_EXPS:
            exp = EXPERIMENTS.get(name)
            if exp is None:
                continue
            _regen_report_from_csv(exp)
        summarize_across_depths(RUN_EXPS, model='full')

    elif RUN_MODE == 'experiments':
        for t in RUN_EXPS:
            if t not in EXPERIMENTS:
                print(f"[skip] 未知实验: {t}")
                continue
            run_experiments(EXPERIMENTS[t])
        summarize_across_depths(RUN_EXPS, model='full')

    elif RUN_MODE == 'summary':
        summarize_across_depths(RUN_EXPS, model='full')

    else:  # 'train'
        for t in RUN_EXPS:
            if t not in EXPERIMENTS:
                print(f"[skip] 未知实验: {t}")
                continue
            run_training(EXPERIMENTS[t])


if __name__ == '__main__':
    main()