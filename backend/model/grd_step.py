# =========================================================
# K-step Hazard GRU Survival (Leak-free) - Local Full Script
# =========================================================
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# -----------------------------
# 0) Imports
# -----------------------------
import os, sys, time, copy, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, average_precision_score, recall_score

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())

# -----------------------------
# 1) Load CSV (train/valid/test) from local path
# -----------------------------
# CSV 폴더 경로: 환경변수 DATA_DIR 우선, 없으면 ./data
BASE_DIR = Path(os.environ.get("DATA_DIR", "./data")).expanduser().resolve()

# 3) 파일 경로
train_path = BASE_DIR / "train_df.csv"
valid_path = BASE_DIR / "valid_df.csv"
test_path  = BASE_DIR / "test_df.csv"

# 4) 존재 확인
missing = [str(p) for p in [train_path, valid_path, test_path] if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Drive는 마운트됐는데 CSV 경로가 틀림. 아래 경로를 확인해서 BASE_DIR를 고쳐야 함:\n"
        + "\n".join(missing)
    )

# 5) 로드
train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)
test_df  = pd.read_csv(test_path)

print("train:", train_df.shape, "valid:", valid_df.shape, "test:", test_df.shape)


# -----------------------------
# 2) Schema / Hyperparams
# -----------------------------
ID_COL   = "stay_id"
TIME_COL = "t"
LABEL_COL = "event" if "event" in train_df.columns else "delta"

FEATURE_COLS = [
    "HeartRate_std_6h","RespRate_std_6h","Temp_std_6h","GCS_Total_mean_6h",
    "DiasBP_mean_6h","SysBP","MeanBP","SpO2_measured","FiO2","pH","GCS_Verbal","GCS_Motor",
]

K  = 48      # future horizon
L  = 24       # lookback
POS_WINDOW = 48  # positive window size (steps before event)
DT = 1.0

# 샘플 수 줄이고 싶으면 stride 키워라 (예: 2~6 추천)
STRIDE = 2    # 1이면 전체 사용, 3이면 3시간마다 샘플 생성

# -----------------------------
# 3) Sort + Mean Impute (train stats)
# -----------------------------
def _sort(df):
    return df.sort_values([ID_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

train_df = _sort(train_df)
valid_df = _sort(valid_df)
test_df  = _sort(test_df)

need = [ID_COL, TIME_COL, LABEL_COL] + FEATURE_COLS
for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise KeyError(f"{name}_df missing columns: {miss}")

impute_stats = {c: float(train_df[c].astype(float).mean()) for c in FEATURE_COLS}

def impute(df):
    out = df.copy()
    for c in FEATURE_COLS:
        out[c] = out[c].astype(float).fillna(impute_stats[c])
    return out

train_i, valid_i, test_i = impute(train_df), impute(valid_df), impute(test_df)
stay_label_valid = valid_i.groupby(ID_COL)[LABEL_COL].max()
stay_label_test = test_i.groupby(ID_COL)[LABEL_COL].max()

# -----------------------------
# 4) Row-level Balance (optional)
# -----------------------------
def balance_ratio(df, target_col, ratio_pos=0.10, random_state=42):
    """
    Row-level balance to reach a target positive ratio (ratio_pos).
    ratio_pos=0.10 -> neg:pos = 9:1
    - Keep all positives; if negatives are insufficient, keep all data.
    """
    if not (0 < ratio_pos < 1):
        raise ValueError("ratio_pos must be between 0 and 1")

    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]

    if len(pos) == 0 or len(neg) == 0:
        return df.copy()

    desired_neg = int(len(pos) * (1 - ratio_pos) / ratio_pos)

    if desired_neg < len(neg):
        neg = neg.sample(n=desired_neg, random_state=random_state)

    return (
        pd.concat([pos, neg])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

def _print_ratio(df, target_col, tag):
    pos = int((df[target_col] == 1).sum())
    neg = int((df[target_col] == 0).sum())
    total = pos + neg
    ratio = (pos / total) if total > 0 else 0.0
    print(f"[{tag}] pos={pos} neg={neg} pos_ratio={ratio:.4f}")

TARGET_ROW_POS_RATIO = 0.10  # neg:pos = 9:1
USE_ROW_BALANCE = False  # keep False to avoid row loss
if USE_ROW_BALANCE:
    _print_ratio(train_i, LABEL_COL, "train before row balance")
    train_bal = balance_ratio(train_i, target_col=LABEL_COL, ratio_pos=TARGET_ROW_POS_RATIO, random_state=42)
    _print_ratio(train_bal, LABEL_COL, "train after row balance")
else:
    train_bal = train_i
    _print_ratio(train_bal, LABEL_COL, "train (no row balance)")

# -----------------------------
# 5) Build Samples (Leak-free, safer mapping)
# -----------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm

def _choose_indices_per_stay(
    t_arr: np.ndarray,
    event_time: int | None,
    n_per_stay: int,
    seed: int,
    pre_event_window: int = 24,
    pre_event_take: int = 8,
    keep_pre_event_all: bool = True,
) -> np.ndarray:
    """
    stay 내에서 샘플을 만들 'row index'들을 선택한다.

    - Negative stay (event_time=None): 전체에서 분산(균일+랜덤)으로 n_per_stay개
    - Positive stay: 이벤트 직전 구간은 전부 유지 + 나머지 분산 샘플링
    """
    rng = np.random.default_rng(seed)
    n = len(t_arr)
    idx_all = np.arange(n)

    if n <= n_per_stay:
        return idx_all

    # negative stay
    if event_time is None:
        k1 = n_per_stay // 2
        uni = np.linspace(0, n - 1, k1).round().astype(int)
        k2 = n_per_stay - k1
        rnd = rng.choice(idx_all, size=k2, replace=False)
        idx = np.unique(np.concatenate([uni, rnd]))
        if idx.size > n_per_stay:
            idx = rng.choice(idx, size=n_per_stay, replace=False)
        return np.sort(idx)

    # positive stay: 이벤트 직전 구간 유지
    pre_mask = (t_arr >= event_time - pre_event_window) & (t_arr < event_time)
    pre_idx = idx_all[pre_mask]

    picked = []

    if pre_idx.size > 0:
        if keep_pre_event_all:
            picked.append(pre_idx)
        else:
            k_pre = min(pre_event_take, pre_idx.size, n_per_stay)
            picked_pre = rng.choice(pre_idx, size=k_pre, replace=False)
            picked.append(picked_pre)

    remain = max(0, n_per_stay - (picked[0].size if picked else 0))

    # 나머지: event 이전 구간에서 분산+랜덤
    if remain > 0:
        base_idx = idx_all[t_arr < event_time]
        if base_idx.size == 0:
            base_idx = idx_all

        if picked:
            base_idx = np.setdiff1d(base_idx, picked[0], assume_unique=False)

        if base_idx.size <= remain:
            picked.append(base_idx)
        else:
            k1 = remain // 2
            uni = np.linspace(0, base_idx.size - 1, k1).round().astype(int)
            uni = base_idx[uni]
            k2 = remain - k1
            rnd = rng.choice(base_idx, size=k2, replace=False)
            picked.append(np.unique(np.concatenate([uni, rnd])))

    idx = np.unique(np.concatenate(picked)) if picked else rng.choice(idx_all, size=n_per_stay, replace=False)
    if idx.size > n_per_stay and (event_time is None or not keep_pre_event_all):
        idx = rng.choice(idx, size=n_per_stay, replace=False)
    return np.sort(idx)


def build_samples(
    df: pd.DataFrame,
    L: int,
    K: int,
    stride: int = 1,
    n_per_stay: int | None = None,
    n_per_stay_pos: int | None = None,
    n_per_stay_neg: int | None = None,
    return_ids: bool = False,
    return_times: bool = False,
    pos_window: int = POS_WINDOW,
    seed: int = 42,
    pre_event_window: int = 24,
    pre_event_take: int = 8,
):
    """
    For each stay_id, pick sample times and build:
      X_seq: (L, F) from [t-L+1..t], missing -> zeros
      y:     (K,)  1 if within [event_time - pos_window + 1, event_time]
      m:     (K,)  at-risk mask (censoring / end / after event)

    n_per_stay:
      - None: 기존 방식(모든 시점 사용, stride 적용)
      - int : stay_id별로 n_per_stay개 시점만 샘플링(중복/불균형 완화 목적)

    n_per_stay_pos / n_per_stay_neg:
      - 제공되면 stay 라벨에 따라 각기 다른 개수로 샘플링

    return_ids:
      - True면 샘플별 stay_id 배열도 반환

    pos_window:
      - event 시점 직전 pos_window 범위를 positive로 설정
    """
    X_list, y_list, m_list = [], [], []
    sid_list = [] if return_ids else None
    t_list = [] if return_times else None

    for sid, g in tqdm(df.groupby(ID_COL), desc="build_samples"):
        g = g.sort_values(TIME_COL)

        t_arr = g[TIME_COL].to_numpy(dtype=int)
        X = g[FEATURE_COLS].to_numpy(dtype=np.float32)
        e = g[LABEL_COL].to_numpy(dtype=np.int64)

        # mapping: time -> row index (missing t 안전)
        t_to_row = {int(t_arr[r]): r for r in range(len(t_arr))}
        end_time = int(t_arr[-1])

        ev_rows = np.where(e == 1)[0]
        event_time = int(t_arr[ev_rows[0]]) if len(ev_rows) > 0 else None

        # ✅ 여기서 "어떤 i들을 돌지"를 결정
        if (n_per_stay_pos is not None) or (n_per_stay_neg is not None):
            nps = n_per_stay_pos if event_time is not None else n_per_stay_neg
            if nps is None:
                idx_list = range(0, len(t_arr), stride)
            else:
                idx = _choose_indices_per_stay(
                    t_arr=t_arr,
                    event_time=event_time,
                    n_per_stay=int(nps),
                    seed=(seed + int(sid) * 1000003) % (2**32 - 1),  # stay별로 seed 다르게
                    pre_event_window=pre_event_window,
                    pre_event_take=min(pre_event_take, int(nps)),
                    keep_pre_event_all=True,
                )
                idx_list = idx  # row index 배열
        elif n_per_stay is None:
            idx_list = range(0, len(t_arr), stride)
        else:
            idx = _choose_indices_per_stay(
                t_arr=t_arr,
                event_time=event_time,
                n_per_stay=int(n_per_stay),
                seed=(seed + int(sid) * 1000003) % (2**32 - 1),  # stay별로 seed 다르게
                pre_event_window=pre_event_window,
                pre_event_take=min(pre_event_take, int(n_per_stay)),
                keep_pre_event_all=True,
            )
            idx_list = idx  # row index 배열

        for i in idx_list:
            t = int(t_arr[i])

            # past sequence: [t-L+1 .. t]
            start_t = t - L + 1
            seq = np.zeros((L, X.shape[1]), dtype=np.float32)
            for j in range(L):
                tt = start_t + j
                r = t_to_row.get(tt, None)
                if r is not None:
                    seq[j] = X[r]

            # future labels/mask (leak-free)
            yk = np.zeros((K,), dtype=np.float32)
            mk = np.zeros((K,), dtype=np.float32)
            win_start = (event_time - pos_window + 1) if event_time is not None else None
            for k in range(1, K + 1):
                ft = t + k
                if ft > end_time:
                    break
                if event_time is not None and ft > event_time:
                    break
                mk[k - 1] = 1.0
                # positive window: [event_time - pos_window + 1, event_time]
                if event_time is not None and win_start is not None:
                    if win_start <= ft <= event_time:
                        yk[k - 1] = 1.0
                if event_time is not None and ft == event_time:
                    break

            X_list.append(seq)
            y_list.append(yk)
            m_list.append(mk)
            if return_ids:
                sid_list.append(sid)
            if return_times:
                t_list.append(t)

    X_out = np.stack(X_list)
    y_out = np.stack(y_list)
    m_out = np.stack(m_list)

    if return_ids or return_times:
        out = [X_out, y_out, m_out]
        if return_ids:
            out.append(np.array(sid_list))
        if return_times:
            out.append(np.array(t_list))
        return tuple(out)

    return X_out, y_out, m_out


# ✅ train만 샘플링 적용 (pos/neg stay별로 다르게)
def _print_stay_ratio(df, id_col, label_col, tag):
    stay_label = df.groupby(id_col)[label_col].max()
    pos = int((stay_label == 1).sum())
    neg = int((stay_label == 0).sum())
    total = pos + neg
    ratio = (pos / total) if total > 0 else 0.0
    print(f"[{tag}] pos_stay={pos} neg_stay={neg} pos_ratio={ratio:.4f}")

def undersample_neg_stays(df, id_col, label_col, ratio_pos=0.20, random_state=42):
    """
    Stay-level undersampling: keep all positive stays, downsample negative stays
    to reach target positive ratio (ratio_pos).
    """
    if not (0 < ratio_pos < 1):
        raise ValueError("ratio_pos must be between 0 and 1")

    stay_label = df.groupby(id_col)[label_col].max()
    pos_ids = stay_label[stay_label == 1].index.to_numpy()
    neg_ids = stay_label[stay_label == 0].index.to_numpy()

    if pos_ids.size == 0 or neg_ids.size == 0:
        return df

    desired_neg = int(pos_ids.size * (1 - ratio_pos) / ratio_pos)
    if desired_neg >= neg_ids.size:
        return df

    rng = np.random.default_rng(random_state)
    keep_neg = rng.choice(neg_ids, size=desired_neg, replace=False)
    keep_ids = np.concatenate([pos_ids, keep_neg])
    return df[df[id_col].isin(keep_ids)].copy()

# ✅ stay_id 기준으로 음성 stay만 언더샘플링
USE_STAY_NEG_SAMPLING = True
TARGET_STAY_POS_RATIO = 0.30  # pos:neg ~= 3:7
_print_stay_ratio(train_bal, ID_COL, LABEL_COL, "stay ratio (before neg sampling)")
if USE_STAY_NEG_SAMPLING:
    train_bal = undersample_neg_stays(
        train_bal, id_col=ID_COL, label_col=LABEL_COL, ratio_pos=TARGET_STAY_POS_RATIO, random_state=42
    )
_print_stay_ratio(train_bal, ID_COL, LABEL_COL, "stay ratio (after neg sampling)")

USE_STAY_SAMPLING = True
POS_N_PER_STAY = 80
NEG_N_PER_STAY = 2
_print_stay_ratio(train_bal, ID_COL, LABEL_COL, "stay ratio (train)")

if USE_STAY_SAMPLING:
    Xtr, ytr, mtr = build_samples(
        train_bal,
        L=L,
        K=K,
        stride=STRIDE,
        n_per_stay_pos=POS_N_PER_STAY,
        n_per_stay_neg=NEG_N_PER_STAY,
        seed=42,
    )
else:
    Xtr, ytr, mtr = build_samples(
        train_bal,
        L=L,
        K=K,
        stride=STRIDE,
        n_per_stay=None,
        seed=42,
    )

# ✅ valid/test는 원본 그대로 평가(절대 샘플링하지 마)
Xva, yva, mva, sid_va, t_va = build_samples(
    valid_i, L=L, K=K, stride=STRIDE, n_per_stay=None, return_ids=True, return_times=True
)
Xte, yte, mte, sid_te, t_te = build_samples(
    test_i, L=L, K=K, stride=STRIDE, n_per_stay=None, return_ids=True, return_times=True
)

# -----------------------------
# 6) Sample-level Balance (train only, keep positives)
# -----------------------------
def _sample_labels_any(y, m):
    return ((y * m).max(axis=1) > 0).astype(int)

def balance_samples_keep_pos(X, y, m, ratio_pos=0.10, random_state=42):
    """
    Balance at sample level using y/m (any event within horizon).
    Keep all positive samples and downsample negatives to target ratio.
    """
    if not (0 < ratio_pos < 1):
        raise ValueError("ratio_pos must be between 0 and 1")

    y_any = _sample_labels_any(y, m)
    pos_idx = np.where(y_any == 1)[0]
    neg_idx = np.where(y_any == 0)[0]

    if pos_idx.size == 0 or neg_idx.size == 0:
        return X, y, m

    desired_neg = int(pos_idx.size * (1 - ratio_pos) / ratio_pos)
    if desired_neg < neg_idx.size:
        rng = np.random.default_rng(random_state)
        neg_idx = rng.choice(neg_idx, size=desired_neg, replace=False)

    idx = np.concatenate([pos_idx, neg_idx])
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(idx)
    return X[idx], y[idx], m[idx]

def _print_sample_ratio(y, m, tag):
    y_any = _sample_labels_any(y, m)
    pos = int(y_any.sum())
    neg = int(y_any.size - pos)
    total = pos + neg
    ratio = (pos / total) if total > 0 else 0.0
    print(f"[{tag}] pos={pos} neg={neg} pos_ratio={ratio:.4f}")

TARGET_SAMPLE_POS_RATIO = 0.40  # neg:pos = 1.5:1 (very aggressive downsample)
_print_sample_ratio(ytr, mtr, "samples before balance")
USE_SAMPLE_BALANCE = True  # stay 언더샘플링 + 샘플 언더샘플링 병행
if USE_SAMPLE_BALANCE:
    Xtr, ytr, mtr = balance_samples_keep_pos(
        Xtr, ytr, mtr, ratio_pos=TARGET_SAMPLE_POS_RATIO, random_state=42
    )
    _print_sample_ratio(ytr, mtr, "samples after balance")
else:
    _print_sample_ratio(ytr, mtr, "samples balance skipped")

print("train:", Xtr.shape, ytr.shape, mtr.shape)

# -----------------------------
# 7) Imbalance (mask-aware)
# -----------------------------
pos = float((ytr * mtr).sum())
tot = float(mtr.sum())
neg = tot - pos
spw_ratio = (neg / pos) if pos > 0 else 1.0


print(f"[IMBALANCE] pos={pos:.0f}, neg={neg:.0f}, pos_weight={spw_ratio:.4f}")

# -----------------------------
# 8) Model
# -----------------------------
class KStepHazardGRU(nn.Module):
    def __init__(self, n_feat: int, hidden: int, K: int, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(hidden, K)

    def forward(self, x):
        _, h = self.gru(x)
        last = h[-1]
        last = self.dropout(last)
        return self.head(last)

# -----------------------------
# 9) DataLoader (GPU-friendly)
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, X, y, m):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.m = torch.from_numpy(m)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i], self.m[i]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

BATCH_TRAIN = 256 if device == "cuda" else 128
BATCH_EVAL  = 512 if device == "cuda" else 256

num_workers = 2 if device == "cuda" else 0
pin_memory = (device == "cuda")

train_loader = DataLoader(
    SeqDataset(Xtr, ytr, mtr),
    batch_size=BATCH_TRAIN, shuffle=True,
    num_workers=num_workers, pin_memory=pin_memory,
    persistent_workers=(num_workers > 0)
)

def make_row_subset(X, y, m, ids=None, times=None, n=200_000, seed=42):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    n = min(n, N)
    idx = rng.choice(N, size=n, replace=False)
    out = [X[idx], y[idx], m[idx]]
    if ids is not None:
        out.append(ids[idx])
    if times is not None:
        out.append(times[idx])
    return tuple(out)

def make_stay_subset(X, y, m, ids, times=None, n_stays=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_ids = np.unique(ids)
    n_stays = min(n_stays, unique_ids.size)
    pick = rng.choice(unique_ids, size=n_stays, replace=False)
    mask = np.isin(ids, pick)
    out = [X[mask], y[mask], m[mask], ids[mask]]
    if times is not None:
        out.append(times[mask])
    return tuple(out)

# row-level subset (fast), stay-level subset (clinical)
Xva_sub, yva_sub, mva_sub, sid_va_sub, t_va_sub = make_row_subset(
    Xva, yva, mva, ids=sid_va, times=t_va, n=200_000, seed=42
)
Xva_stay, yva_stay, mva_stay, sid_va_stay, t_va_stay = make_stay_subset(
    Xva, yva, mva, ids=sid_va, times=t_va, n_stays=2000, seed=42
)

valid_loader_sub = DataLoader(
    SeqDataset(Xva_sub, yva_sub, mva_sub),
    batch_size=BATCH_EVAL, shuffle=False,
    num_workers=num_workers, pin_memory=pin_memory,
    persistent_workers=(num_workers > 0)
)

# -----------------------------
# 10) Loss / Optim
# -----------------------------
HIDDEN = 64
DROPOUT = 0.2  # 과적합 방지로 추천
model = KStepHazardGRU(n_feat=len(FEATURE_COLS), hidden=HIDDEN, K=K, dropout=DROPOUT).to(device)

pos_weight = torch.tensor([spw_ratio], dtype=torch.float32, device=device)
def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=3.0, pos_weight=None):
    """
    logits: (B,K)
    targets: (B,K) float {0,1}
    pos_weight: Tensor([w]) or None
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)  # p_t
    focal = (1 - pt).pow(gamma)

    # alpha weighting: positive/negative balance
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    return alpha_t * focal * bce

FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0


opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# mixed precision (CUDA일 때만)
use_amp = (device == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

def run_epoch(loader, train: bool, log_mask_stats: bool = False, tag: str = "", log_every: int = 2000):
    model.train(train)
    total_loss, total_mask, total_elems = 0.0, 0.0, 0.0

    for b_idx, (Xb, yb, mb) in enumerate(loader, start=1):
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        
        amp_device = "cuda" if device == "cuda" else "cpu"
        scaler = torch.amp.GradScaler(amp_device, enabled=(amp_device == "cuda"))

        with torch.amp.autocast(device_type=amp_device, enabled=(amp_device == "cuda")):
            logits = model(Xb)
            loss_mat = focal_loss_with_logits(
                logits, yb, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, pos_weight=pos_weight
            ) * mb

            loss = loss_mat.sum() / (mb.sum() + 1e-8)

        if train:
            opt.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

        total_loss += float(loss_mat.sum().detach().cpu())
        total_mask += float(mb.sum().detach().cpu())
        total_elems += float(mb.numel())

        if log_mask_stats and (b_idx % max(1, log_every) == 0):
            valid = float(mb.sum().detach().cpu())
            total = float(mb.numel())
            ratio = (valid / total) if total > 0 else 0.0
            tag_txt = f"{tag} " if tag else ""
            print(f"[MASK {tag_txt}batch {b_idx}] valid={valid:.0f} total={total:.0f} ratio={ratio:.4f}")

    return total_loss / (total_mask + 1e-8), total_mask, total_elems

# -----------------------------
# 11) Metrics (mask-aware)
# -----------------------------
def safe_auc_ap(y, s):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y, s), average_precision_score(y, s)

USE_PROD_RISK = True  # 권장: 1 - Π(1-h)

def predict_hazard_seq(model, X, batch=1024):
    model.eval()
    out = []
    loader = DataLoader(torch.from_numpy(X), batch_size=batch, shuffle=False, num_workers=0)
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            logits = model(xb)
            hz = torch.sigmoid(logits).cpu().numpy()
            out.append(hz)
    return np.vstack(out)

from sklearn.metrics import precision_score, f1_score

def compute_risk_window(hz: np.ndarray, m: np.ndarray, wh: int, dt: float = 1.0, use_prod: bool = True):
    mwin = m[:, :wh].astype(np.float32)
    hzwin = hz[:, :wh] * mwin

    if use_prod:
        p = np.clip(hzwin, 1e-8, 1 - 1e-8)
        p = np.where(mwin > 0, p, 0.0)  # mask=0이면 곱셈 영향 없게
        risk = 1.0 - np.prod(1.0 - p, axis=1)
    else:
        H = hzwin.sum(axis=1) * float(dt)
        risk = 1.0 - np.exp(-H)

    return risk

def make_y_any(y: np.ndarray, m: np.ndarray, wh: int):
    mwin = m[:, :wh].astype(np.float32)
    return ((y[:, :wh] * mwin).max(axis=1) > 0).astype(int)

def aggregate_stay_risk(
    risk: np.ndarray,
    stay_ids: np.ndarray,
    times: np.ndarray | None = None,
    method: str = "max",
    topk: int = 5,
    recent_hours: int = 24,
):
    df = pd.DataFrame({"stay_id": stay_ids, "risk": risk})
    if times is not None:
        df["t"] = times
    if method == "max":
        return df.groupby("stay_id")["risk"].max()
    if method == "mean_topk":
        def _mean_topk(x):
            if x.size <= topk:
                return float(np.mean(x))
            return float(np.mean(np.sort(x)[-topk:]))
        return df.groupby("stay_id")["risk"].apply(_mean_topk)
    if method == "recent_mean":
        if "t" not in df.columns:
            raise ValueError("recent_mean requires times")
        def _recent_mean(g):
            t_max = g["t"].max()
            cutoff = t_max - recent_hours + 1
            recent = g[g["t"] >= cutoff]["risk"]
            if recent.empty:
                return float(g["risk"].mean())
            return float(recent.mean())
        return df.groupby("stay_id").apply(_recent_mean)
    if method == "cumprod":
        def _cum_risk(x):
            p = np.clip(x.to_numpy(), 1e-8, 1 - 1e-8)
            return float(1.0 - np.prod(1.0 - p))
        return df.groupby("stay_id")["risk"].apply(_cum_risk)
    raise ValueError(f"unknown method: {method}")

def stay_level_metrics(
    risk: np.ndarray,
    stay_ids: np.ndarray,
    stay_label_map: pd.Series,
    times: np.ndarray | None = None,
    method: str = "max",
    topk: int = 5,
    recent_hours: int = 24,
    recall_lo: float = 0.70,
    recall_hi: float = 0.85,
    calc_threshold: bool = True,
    threshold_mode: str = "recall_band",
):
    risk_by_stay = aggregate_stay_risk(
        risk, stay_ids, times=times, method=method, topk=topk, recent_hours=recent_hours
    )
    common = risk_by_stay.index.intersection(stay_label_map.index)
    if len(common) == 0:
        return {"auc": np.nan, "ap": np.nan, "precision": np.nan, "recall": np.nan, "thr": np.nan, "n": 0}

    y_true = stay_label_map.loc[common].to_numpy(dtype=int)
    y_pred = risk_by_stay.loc[common].to_numpy(dtype=float)
    auc, ap = safe_auc_ap(y_true, y_pred)

    prec = rec = thr = np.nan
    if calc_threshold:
        if threshold_mode == "recall_band":
            best_thr = choose_threshold_max_precision_in_recall_band(
                risk=y_pred,
                y_true=y_true,
                recall_lo=recall_lo,
                recall_hi=recall_hi,
                max_alert_rate=None,
            )
        elif threshold_mode == "f1":
            best_thr = choose_threshold_max_f1(y_pred, y_true)
        else:
            raise ValueError(f"unknown threshold_mode: {threshold_mode}")

        if best_thr is None and threshold_mode != "f1":
            # fallback: maximize F1 across all thresholds
            best_thr = choose_threshold_max_f1(y_pred, y_true)

        if best_thr is not None:
            thr = best_thr["thr"]
            prec = best_thr["precision"]
            rec = best_thr["recall"]

    f1 = np.nan
    if prec == prec and rec == rec and (prec + rec) > 0:
        f1 = 2 * prec * rec / (prec + rec)

    return {
        "auc": auc,
        "ap": ap,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "thr": thr,
        "n": int(len(common)),
    }

def plot_training_history(history: dict, out_dir: Path):
    if not history["epoch"]:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], label="train_loss")
    ax.plot(epochs, history["valid_loss"], label="valid_loss")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["auc24"], label="valid_24h_auc")
    ax.plot(epochs, history["ap24"], label="valid_24h_ap")
    ax.set_title("Valid 24h AUC/AP")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "valid_24h_auc_ap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["stay_auc"], label="stay_auc")
    ax.plot(epochs, history["stay_ap"], label="stay_ap")
    ax.set_title("Stay-level AUC/AP (recent mean)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stay_auc_ap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["stay_precision"], label="stay_precision")
    ax.plot(epochs, history["stay_recall"], label="stay_recall")
    ax.plot(epochs, history["stay_f1"], label="stay_f1")
    ax.set_title("Stay-level Precision/Recall/F1 (recent mean)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "stay_precision_recall.png", dpi=150)
    plt.close(fig)

def choose_threshold_max_precision_in_recall_band(
    risk: np.ndarray,
    y_true: np.ndarray,
    recall_lo: float,
    recall_hi: float,
    max_alert_rate: float | None = None,
    max_candidates: int = 5000,
):
    # 후보 threshold: risk 값 기반 (중복 제거)
    cand = np.unique(risk)
    # 너무 많으면 분위수 기반으로 줄이기
    if cand.size > 5000:
        cand = np.quantile(risk, np.linspace(0, 1, 5000))

    best = None
    for thr in cand:
        pred = (risk >= thr).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        if rec < recall_lo or rec > recall_hi:
            continue

        alert_rate = pred.mean()
        if (max_alert_rate is not None) and (alert_rate > max_alert_rate):
            continue

        prec = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        # 1순위: precision 최대
        # 2순위: recall 높을수록(범위 안에서)
        # 3순위: f1
        key = (prec, rec, f1)

        if (best is None) or (key > best["key"]):
            best = {
                "thr": float(thr),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "alert_rate": float(alert_rate),
                "key": key
            }

    return best

def choose_threshold_max_f1(risk: np.ndarray, y_true: np.ndarray, max_candidates: int = 5000):
    cand = np.unique(risk)
    if cand.size > max_candidates:
        cand = np.quantile(risk, np.linspace(0, 1, max_candidates))

    best = None
    for thr in cand:
        pred = (risk >= thr).astype(int)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        if (prec + rec) == 0:
            f1 = 0.0
        else:
            f1 = 2 * prec * rec / (prec + rec)
        key = (f1, prec, rec)
        if (best is None) or (key > best["key"]):
            best = {
                "thr": float(thr),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "key": key,
            }

    return best

def eval_window_metrics(
    model, X_eval, y_eval, m_eval, windows=(6,24,72), dt=1.0, hz_eval=None
):
    hz = hz_eval if hz_eval is not None else predict_hazard_seq(model, X_eval, batch=1024)
    metrics = {}

    for wh in windows:
        mwin = m_eval[:, :wh].astype(np.float32)
        hzwin = hz[:, :wh] * mwin

        if USE_PROD_RISK:
            p = np.clip(hzwin, 1e-8, 1 - 1e-8)
            # mask=0이면 곱셈에 영향 없게 p=0(=> 1-p=1)
            p = np.where(mwin > 0, p, 0.0)
            risk = 1.0 - np.prod(1.0 - p, axis=1)
        else:
            H = hzwin.sum(axis=1) * dt
            risk = 1.0 - np.exp(-H)

        y_any = ((y_eval[:, :wh] * mwin).max(axis=1) > 0).astype(int)
        auc, ap = safe_auc_ap(y_any, risk)
        metrics[f"{wh}h_auc"] = auc
        metrics[f"{wh}h_ap"]  = ap

    return metrics

# -----------------------------
# 12) Train with Early Stopping + metrics on improvement
# -----------------------------
MAX_EPOCHS = 50
PATIENCE   = 4
MIN_DELTA  = 1e-4
STAY_TOPK = 10
STAY_RECENT_HOURS = 24
STAY_THRESHOLD_MODE = "f1"
STAY_RECALL_LO = 0.65
STAY_RECALL_HI = 0.75
LOG_MASK_BATCH = False
LOG_MASK_EVERY = 1

best_val = float("inf")
best_state = None
bad = 0
history = {
    "epoch": [],
    "train_loss": [],
    "valid_loss": [],
    "stay_auc": [],
    "stay_ap": [],
    "stay_precision": [],
    "stay_recall": [],
    "stay_f1": [],
    "stay_n": [],
    "auc24": [],
    "ap24": [],
}

for epoch in range(1, MAX_EPOCHS + 1):
    # epoch 1~MAX_EPOCHS 동안 pos_weight를 50 -> 10으로 서서히 감소
    w0, w1 = 50.0, 10.0
    w = w0 + (w1 - w0) * (epoch - 1) / max(1, (MAX_EPOCHS - 1))
    pos_weight = torch.tensor([w], dtype=torch.float32, device=device)

    t0 = time.time()

    tr_loss, tr_mask, tr_total = run_epoch(
        train_loader,
        train=True,
        log_mask_stats=LOG_MASK_BATCH,
        tag="train",
        log_every=LOG_MASK_EVERY,
    )
    va_loss, va_mask, va_total = run_epoch(
        valid_loader_sub,
        train=False,
        log_mask_stats=LOG_MASK_BATCH,
        tag="valid",
        log_every=LOG_MASK_EVERY,
    )
    if tr_total > 0 and va_total > 0:
        print(
            f"[MASK epoch] train_valid={tr_mask:.0f}/{tr_total:.0f}({tr_mask/tr_total:.4f}) "
            f"valid_valid={va_mask:.0f}/{va_total:.0f}({va_mask/va_total:.4f})"
        )

    # stay-level eval on valid subset (24h risk)
    # row-level subset (fast)
    hz_sub = predict_hazard_seq(model, Xva_sub, batch=1024)
    risk24_sub = compute_risk_window(hz_sub, mva_sub, wh=24, dt=DT, use_prod=USE_PROD_RISK)
    y24_sub = make_y_any(yva_sub, mva_sub, wh=24)
    auc24_sub, ap24_sub = safe_auc_ap(y24_sub, risk24_sub)

    # stay-level subset (clinical)
    hz_stay = predict_hazard_seq(model, Xva_stay, batch=1024)
    risk24_stay = compute_risk_window(hz_stay, mva_stay, wh=24, dt=DT, use_prod=USE_PROD_RISK)

    stay_recent = stay_level_metrics(
        risk24_stay,
        sid_va_stay,
        stay_label_valid,
        times=t_va_stay,
        method="recent_mean",
        recent_hours=STAY_RECENT_HOURS,
        topk=STAY_TOPK,
        recall_lo=STAY_RECALL_LO,
        recall_hi=STAY_RECALL_HI,
        calc_threshold=True,
        threshold_mode=STAY_THRESHOLD_MODE,
    )
    stay_topk = stay_level_metrics(
        risk24_stay,
        sid_va_stay,
        stay_label_valid,
        method="mean_topk",
        topk=STAY_TOPK,
        calc_threshold=False,
    )
    stay_max = stay_level_metrics(
        risk24_stay,
        sid_va_stay,
        stay_label_valid,
        method="max",
        topk=STAY_TOPK,
        calc_threshold=False,
    )

    history["epoch"].append(epoch)
    history["train_loss"].append(tr_loss)
    history["valid_loss"].append(va_loss)
    history["stay_auc"].append(stay_recent["auc"])
    history["stay_ap"].append(stay_recent["ap"])
    history["stay_precision"].append(stay_recent["precision"])
    history["stay_recall"].append(stay_recent["recall"])
    history["stay_f1"].append(stay_recent["f1"])
    history["stay_n"].append(stay_recent["n"])
    history["auc24"].append(auc24_sub)
    history["ap24"].append(ap24_sub)

    improved = (best_val - va_loss) > MIN_DELTA

    if improved:
        best_val = va_loss
        best_state = copy.deepcopy(model.state_dict())
        bad = 0

        mets = eval_window_metrics(
            model,
            Xva_sub, yva_sub, mva_sub,
            windows=(6,24,72),
            dt=DT,
            hz_eval=hz_sub,
        )

        print(
          f"[BEST] epoch {epoch:02d} | "
          f"train_loss={tr_loss:.6f} | valid_loss={va_loss:.6f} | "
          f"6h AUC={mets['6h_auc']:.4f} AP={mets['6h_ap']:.4f} | "
          f"24h AUC={mets['24h_auc']:.4f} AP={mets['24h_ap']:.4f} | "
          f"72h AUC={mets['72h_auc']:.4f} AP={mets['72h_ap']:.4f} | "
          f"stayRecent AUC={stay_recent['auc']:.4f} AP={stay_recent['ap']:.4f} "
          f"P={stay_recent['precision']:.4f} R={stay_recent['recall']:.4f} "
          f"F1={stay_recent['f1']:.4f} N={stay_recent['n']} | "
          f"stayTopK AUC={stay_topk['auc']:.4f} AP={stay_topk['ap']:.4f} | "
          f"stayMax AUC={stay_max['auc']:.4f} AP={stay_max['ap']:.4f} | "
          f"time={time.time()-t0:.1f}s"
      )

    else:
        bad += 1
        print(
            f"epoch {epoch:02d} | train_loss={tr_loss:.6f} | valid_loss={va_loss:.6f} | "
            f"stayRecent AUC={stay_recent['auc']:.4f} AP={stay_recent['ap']:.4f} "
            f"P={stay_recent['precision']:.4f} R={stay_recent['recall']:.4f} "
            f"F1={stay_recent['f1']:.4f} N={stay_recent['n']} | "
            f"stayTopK AUC={stay_topk['auc']:.4f} AP={stay_topk['ap']:.4f} | "
            f"stayMax AUC={stay_max['auc']:.4f} AP={stay_max['ap']:.4f} | "
            f"bad={bad}/{PATIENCE} | time={time.time()-t0:.1f}s"
        )
        if bad >= PATIENCE:
            print(f"[EARLY STOP] best_valid_loss={best_val:.6f}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    print("[OK] loaded best model state")

# -----------------------------
# 13) Final Eval (full valid/test 한번만)
#     - 여기서 full valid는 너무 크면 오래 걸릴 수 있음
# -----------------------------
print("\n[FINAL EVAL] valid(full) + test(full) metrics (may take time)")

mets_va = eval_window_metrics(model, Xva, yva, mva, windows=(6,24,72), dt=DT)
mets_te = eval_window_metrics(model, Xte, yte, mte, windows=(6,24,72), dt=DT)

hz_va = predict_hazard_seq(model, Xva, batch=1024)
hz_te = predict_hazard_seq(model, Xte, batch=1024)
risk24_va = compute_risk_window(hz_va, mva, wh=24, dt=DT, use_prod=USE_PROD_RISK)
risk24_te = compute_risk_window(hz_te, mte, wh=24, dt=DT, use_prod=USE_PROD_RISK)

stay_va_recent = stay_level_metrics(
    risk24_va,
    sid_va,
    stay_label_valid,
    times=t_va,
    method="recent_mean",
    recent_hours=STAY_RECENT_HOURS,
    topk=STAY_TOPK,
    recall_lo=STAY_RECALL_LO,
    recall_hi=STAY_RECALL_HI,
    calc_threshold=True,
    threshold_mode=STAY_THRESHOLD_MODE,
)
stay_te_recent = stay_level_metrics(
    risk24_te,
    sid_te,
    stay_label_test,
    times=t_te,
    method="recent_mean",
    recent_hours=STAY_RECENT_HOURS,
    topk=STAY_TOPK,
    recall_lo=STAY_RECALL_LO,
    recall_hi=STAY_RECALL_HI,
    calc_threshold=True,
    threshold_mode=STAY_THRESHOLD_MODE,
)
stay_va_topk = stay_level_metrics(
    risk24_va,
    sid_va,
    stay_label_valid,
    method="mean_topk",
    topk=STAY_TOPK,
    calc_threshold=False,
)
stay_te_topk = stay_level_metrics(
    risk24_te,
    sid_te,
    stay_label_test,
    method="mean_topk",
    topk=STAY_TOPK,
    calc_threshold=False,
)
stay_va = stay_level_metrics(risk24_va, sid_va, stay_label_valid, method="max", topk=STAY_TOPK, calc_threshold=False)
stay_te = stay_level_metrics(risk24_te, sid_te, stay_label_test, method="max", topk=STAY_TOPK, calc_threshold=False)

print("[VALID row]", mets_va)
print(
    "[VALID stay] "
    f"recent AUC={stay_va_recent['auc']:.4f} AP={stay_va_recent['ap']:.4f} "
    f"P={stay_va_recent['precision']:.4f} R={stay_va_recent['recall']:.4f} "
    f"F1={stay_va_recent['f1']:.4f} N={stay_va_recent['n']} | "
    f"topk AUC={stay_va_topk['auc']:.4f} AP={stay_va_topk['ap']:.4f} | "
    f"max AUC={stay_va['auc']:.4f} AP={stay_va['ap']:.4f}"
)
print("[TEST row ]", mets_te)
print(
    "[TEST stay] "
    f"recent AUC={stay_te_recent['auc']:.4f} AP={stay_te_recent['ap']:.4f} "
    f"P={stay_te_recent['precision']:.4f} R={stay_te_recent['recall']:.4f} "
    f"F1={stay_te_recent['f1']:.4f} N={stay_te_recent['n']} | "
    f"topk AUC={stay_te_topk['auc']:.4f} AP={stay_te_topk['ap']:.4f} | "
    f"max AUC={stay_te['auc']:.4f} AP={stay_te['ap']:.4f}"
)

# -----------------------------
# 14) Export (state_dict + meta)
# -----------------------------
SAVE_DIR = Path("./artifacts")
SAVE_DIR.mkdir(exist_ok=True)

plot_training_history(history, SAVE_DIR)
print("saved plots to:", SAVE_DIR.resolve())

torch.save(model.state_dict(), SAVE_DIR / "kstep_gru_state.pt")

meta = {
    "FEATURE_COLS": FEATURE_COLS,
    "impute_stats": impute_stats,
    "K": K,
    "L": L,
    "DT": DT,
    "hidden": HIDDEN,
    "dropout": DROPOUT,
    "stride": STRIDE,
    "risk_formula": "prod" if USE_PROD_RISK else "exp",
    "pos_window": POS_WINDOW,
    "stay_topk": STAY_TOPK,
    "stay_recent_hours": STAY_RECENT_HOURS,
    "stay_threshold_mode": STAY_THRESHOLD_MODE,
    "stay_recall_lo": STAY_RECALL_LO,
    "stay_recall_hi": STAY_RECALL_HI,
    "stay_neg_sampling": USE_STAY_NEG_SAMPLING,
    "stay_pos_ratio": TARGET_STAY_POS_RATIO,
    "stay_sampling": USE_STAY_SAMPLING,
    "pos_n_per_stay": POS_N_PER_STAY,
    "neg_n_per_stay": NEG_N_PER_STAY,
    "sample_balance": USE_SAMPLE_BALANCE,
    "sample_pos_ratio": TARGET_SAMPLE_POS_RATIO,
}

with open(SAVE_DIR / "kstep_gru_meta.pkl", "wb") as f:
    pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

print("saved to:", SAVE_DIR.resolve())
