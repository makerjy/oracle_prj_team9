from __future__ import annotations

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
    meta_path = artifacts / "kstep_gru_meta.pkl"
    state_path = artifacts / "kstep_gru_state.pt"

    meta = dict(DEFAULT_META)
    if meta_path.exists():
        with meta_path.open("rb") as handle:
            loaded = pickle.load(handle)
        if isinstance(loaded, dict):
            meta.update(loaded)

    feature_cols = meta.get("FEATURE_COLS") or FEATURE_COLS
    meta["FEATURE_COLS"] = feature_cols
    if "impute_stats" not in meta or not isinstance(meta["impute_stats"], dict):
        meta["impute_stats"] = {col: 0.0 for col in feature_cols}

    if torch is None or nn is None:
        return ModelBundle(model=None, device="cpu", meta=meta, use_fallback=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not state_path.exists():
        return ModelBundle(model=None, device=device, meta=meta, use_fallback=True)

    num_layers = int(meta.get("num_layers", meta.get("n_layers", 1)))
    dropout = float(meta.get("dropout", 0.0))
    model = KStepGRU(
        input_dim=len(feature_cols),
        hidden_dim=int(meta.get("hidden", 64)),
        num_layers=num_layers,
        k_steps=int(meta.get("K", 120)),
        dropout=dropout,
    )
    state = torch.load(state_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return ModelBundle(model=model, device=device, meta=meta, use_fallback=False)


def infer_hazard_sequence(bundle: ModelBundle, sequence: Sequence[Sequence[float]]) -> List[float]:
    k_steps = int(bundle.meta.get("K", 120))
    if bundle.use_fallback or bundle.model is None or torch is None:
        return fallback_hazard_sequence(sequence, k_steps, bundle.meta.get("FEATURE_COLS"))

    tensor = torch.tensor(sequence, dtype=torch.float32, device=bundle.device).unsqueeze(0)
    with torch.no_grad():
        raw = bundle.model(tensor).squeeze(0).flatten()

    if raw.numel() == 0:
        return fallback_hazard_sequence(sequence, k_steps, bundle.meta.get("FEATURE_COLS"))

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
