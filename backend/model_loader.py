from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


FEATURE_COLS = [
    "HeartRate_std_6h",
    "RespRate_std_6h",
    "Temp_std_6h",
    "GCS_Total_mean_6h",
    "DiasBP_mean_6h",
    "SysBP",
    "MeanBP",
    "SpO2_measured",
    "FiO2",
    "pH",
    "GCS_Verbal",
    "GCS_Motor",
]

DEFAULT_META = {
    "FEATURE_COLS": FEATURE_COLS,
    "impute_stats": {col: 0.0 for col in FEATURE_COLS},
    "standardize_stats": {},
    "K": 120,
    "L": 24,
    "DT": 1,
    "hidden": 64,
    "num_layers": 1,
}

FALLBACK_RANGES: Dict[str, Tuple[float, float, str]] = {
    "HeartRate_std_6h": (0.02, 0.35, "high"),
    "RespRate_std_6h": (0.02, 0.30, "high"),
    "Temp_std_6h": (0.01, 0.20, "high"),
    "GCS_Total_mean_6h": (3.0, 15.0, "low"),
    "DiasBP_mean_6h": (45.0, 90.0, "low"),
    "SysBP": (80.0, 140.0, "low"),
    "MeanBP": (55.0, 100.0, "low"),
    "SpO2_measured": (88.0, 100.0, "low"),
    "FiO2": (0.21, 0.90, "high"),
    "pH": (7.20, 7.45, "low"),
    "GCS_Verbal": (1.0, 5.0, "low"),
    "GCS_Motor": (1.0, 6.0, "low"),
}


@dataclass
class ModelBundle:
    model: Optional["nn.Module"]
    device: str
    meta: Dict
    use_fallback: bool
    mode: str


if nn is not None:
    class KStepGRU(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int,
            k_steps: int,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.head = nn.Linear(hidden_dim, k_steps)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            _, h = self.gru(x)
            last_hidden = h[-1]
            last_hidden = self.dropout(last_hidden)
            return self.head(last_hidden)
else:
    class KStepGRU:
        pass


def load_model_bundle(artifacts_dir: Optional[Path] = None) -> ModelBundle:
    artifacts = artifacts_dir or Path(__file__).resolve().parent / "artifacts"
    state_path = artifacts / "kstep_gru_state.pt"

    meta = _load_meta(artifacts)
    feature_cols = meta.get("FEATURE_COLS") or meta.get("feature_cols") or FEATURE_COLS
    meta["FEATURE_COLS"] = feature_cols
    meta["feature_cols"] = feature_cols
    if "impute_stats" not in meta or not isinstance(meta["impute_stats"], dict):
        meta["impute_stats"] = {col: 0.0 for col in feature_cols}
    else:
        meta["impute_stats"] = {
            col: float(meta["impute_stats"].get(col, 0.0)) for col in feature_cols
        }
    if "standardize_stats" not in meta or not isinstance(meta["standardize_stats"], dict):
        meta["standardize_stats"] = {}

    mode = _infer_mode(meta)
    if torch is None or nn is None:
        return ModelBundle(model=None, device="cpu", meta=meta, use_fallback=True, mode=mode)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not state_path.exists():
        return ModelBundle(model=None, device=device, meta=meta, use_fallback=True, mode=mode)

    num_layers = int(
        meta.get("num_layers", meta.get("n_layers", meta.get("config", {}).get("model", {}).get("n_layers", 1)))
    )
    dropout = float(meta.get("dropout", meta.get("config", {}).get("model", {}).get("dropout", 0.0)))
    hidden_dim = int(meta.get("hidden", meta.get("config", {}).get("model", {}).get("hidden", 64)))

    if mode == "timewise":
        from .model.model import TimewiseGRU

        model = TimewiseGRU(
            n_feat=len(feature_cols),
            hidden=hidden_dim,
            n_layers=num_layers,
            dropout=dropout,
        )
    else:
        model = KStepGRU(
            input_dim=len(feature_cols),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            k_steps=int(meta.get("K", meta.get("k_steps", 120))),
            dropout=dropout,
        )
    state = torch.load(state_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return ModelBundle(model=model, device=device, meta=meta, use_fallback=False, mode=mode)


def infer_hazard_sequence(
    bundle: ModelBundle,
    sequence: Sequence[Sequence[float]],
    length: Optional[int] = None,
) -> List[float]:
    k_steps = _target_k_steps(bundle)
    if bundle.use_fallback or bundle.model is None or torch is None:
        return fallback_hazard_sequence(sequence, k_steps, bundle.meta.get("FEATURE_COLS"))

    feature_cols = bundle.meta.get("FEATURE_COLS") or FEATURE_COLS
    standardize_stats = bundle.meta.get("standardize_stats")
    sequence_for_model = (
        _standardize_sequence(sequence, feature_cols, standardize_stats)
        if standardize_stats
        else sequence
    )

    tensor = torch.tensor(sequence_for_model, dtype=torch.float32, device=bundle.device).unsqueeze(0)
    with torch.no_grad():
        if bundle.mode == "timewise":
            seq_len = len(sequence)
            if length is None:
                length = seq_len
            length = max(0, min(int(length), seq_len))
            if length == 0:
                return fallback_hazard_sequence(sequence, k_steps, feature_cols)
            lengths = torch.tensor([length], dtype=torch.long, device=bundle.device)
            logits = bundle.model(tensor, lengths)
            last_logit = logits[0, length - 1]
            risk = float(torch.sigmoid(last_logit).item())
            hazard_value = _constant_hazard_from_risk(
                risk,
                int(bundle.meta.get("horizon_hours", 24)),
            )
            return [hazard_value for _ in range(k_steps)]
        raw = bundle.model(tensor).squeeze(0).flatten()

    if raw.numel() == 0:
        return fallback_hazard_sequence(sequence, k_steps, feature_cols)

    raw = torch.sigmoid(raw)
    hazard = raw.clamp(0.0, 1.0).tolist()

    if len(hazard) < k_steps:
        hazard.extend([hazard[-1]] * (k_steps - len(hazard)))
    elif len(hazard) > k_steps:
        hazard = hazard[:k_steps]

    return hazard


def fallback_hazard_sequence(
    sequence: Sequence[Sequence[float]],
    k_steps: int,
    feature_cols: Optional[Iterable[str]] = None,
) -> List[float]:
    cols = list(feature_cols) if feature_cols else FEATURE_COLS
    last = sequence[-1] if sequence else [0.0 for _ in cols]
    severity = _severity_score(cols, last)

    base = 0.003 + 0.008 * severity
    drift = (severity - 0.45) * 0.004
    hazard_seq = []
    for idx in range(k_steps):
        wave = 0.003 * math.sin(idx / 6.5 + severity * 2.4)
        value = base + drift * (idx / max(k_steps - 1, 1)) + wave
        hazard_seq.append(_clamp(value, 0.001, 0.03))
    return hazard_seq


def _load_meta(artifacts: Path) -> Dict:
    meta = dict(DEFAULT_META)
    json_path = artifacts / "kstep_gru_meta.json"
    pkl_path = artifacts / "kstep_gru_meta.pkl"

    loaded = None
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    elif pkl_path.exists():
        with pkl_path.open("rb") as handle:
            loaded = pickle.load(handle)

    if isinstance(loaded, dict):
        meta.update(loaded)
        if (
            "K" not in loaded
            and "k_steps" not in loaded
            and isinstance(loaded.get("config"), dict)
            and "sequence" in loaded["config"]
        ):
            meta.pop("K", None)

    feature_cols = meta.get("FEATURE_COLS") or meta.get("feature_cols") or FEATURE_COLS
    meta["FEATURE_COLS"] = feature_cols
    meta["feature_cols"] = feature_cols

    sequence_len = meta.get("config", {}).get("sequence", {}).get("max_len")
    if sequence_len is None:
        sequence_len = meta.get("L") or meta.get("sequence_len") or meta.get("sequence_max_len")
    if sequence_len is None:
        sequence_len = 24
    meta["sequence_len"] = int(sequence_len or 24)

    pad_value = meta.get("pad_value")
    if pad_value is None:
        pad_value = meta.get("config", {}).get("sequence", {}).get("pad_value", 0.0)
    meta["pad_value"] = float(pad_value)

    horizon = meta.get("horizon_hours")
    if horizon is None:
        horizon = meta.get("config", {}).get("data", {}).get("horizon_hours", 24)
    meta["horizon_hours"] = int(horizon or 24)

    if "K" not in meta and "k_steps" in meta:
        meta["K"] = meta["k_steps"]

    return meta


def _infer_mode(meta: Dict) -> str:
    if "K" in meta or "k_steps" in meta:
        return "kstep"
    if isinstance(meta.get("config"), dict) and "sequence" in meta["config"]:
        return "timewise"
    return "kstep"


def _target_k_steps(bundle: ModelBundle) -> int:
    k_steps = bundle.meta.get("K") or bundle.meta.get("k_steps")
    if k_steps:
        return int(k_steps)
    sequence_len = bundle.meta.get("sequence_len") or bundle.meta.get("sequence_max_len")
    return int(max(72, sequence_len or 0, 120))


def _standardize_sequence(
    sequence: Sequence[Sequence[float]],
    feature_cols: Sequence[str],
    stats: Dict,
) -> List[List[float]]:
    out: List[List[float]] = []
    for row in sequence:
        scaled: List[float] = []
        for value, col in zip(row, feature_cols):
            col_stats = stats.get(col) if isinstance(stats, dict) else None
            if not isinstance(col_stats, dict):
                scaled.append(float(value))
                continue
            mean = float(col_stats.get("mean", 0.0))
            std = float(col_stats.get("std", 1.0))
            if std == 0.0:
                std = 1.0
            scaled.append((float(value) - mean) / std)
        scaled.extend(float(value) for value in row[len(scaled):])
        out.append(scaled)
    return out


def _constant_hazard_from_risk(risk: float, horizon_hours: int) -> float:
    if horizon_hours <= 0:
        return _clamp(risk, 0.0, 1.0)
    risk = _clamp(risk, 0.0, 0.999)
    hazard = 1.0 - math.pow(1.0 - risk, 1.0 / float(horizon_hours))
    return _clamp(hazard, 0.0, 1.0)


def _severity_score(feature_cols: Sequence[str], values: Sequence[float]) -> float:
    if not values:
        return 0.3
    scores: List[float] = []
    for col, raw in zip(feature_cols, values):
        if col not in FALLBACK_RANGES:
            continue
        low, high, direction = FALLBACK_RANGES[col]
        if high <= low:
            continue
        if direction == "high":
            score = (raw - low) / (high - low)
        else:
            score = (high - raw) / (high - low)
        scores.append(_clamp(score, 0.0, 1.0))
    if not scores:
        return 0.3
    return sum(scores) / len(scores)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))
