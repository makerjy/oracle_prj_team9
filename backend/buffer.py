from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Deque, Dict, List, Mapping, Sequence


class SequenceBuffer:
    def __init__(
        self,
        feature_cols: Sequence[str],
        seq_len: int,
        impute_stats: Mapping[str, float] | None = None,
        pad_value: float = 0.0,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.seq_len = int(seq_len)
        self.impute_stats = {
            col: float((impute_stats or {}).get(col, 0.0)) for col in self.feature_cols
        }
        self.pad_value = float(pad_value)
        self.buffers: Dict[str, Deque[List[float]]] = defaultdict(
            lambda: deque(maxlen=self.seq_len)
        )

    def add(self, patient_id: str, features: Mapping[str, float]) -> tuple[List[List[float]], int]:
        row = [self._coerce(features.get(col), self.impute_stats[col]) for col in self.feature_cols]
        self.buffers[patient_id].append(row)
        length = len(self.buffers[patient_id])
        return self.get_sequence(patient_id), length

    def get_sequence(self, patient_id: str) -> List[List[float]]:
        series = list(self.buffers[patient_id])
        if len(series) < self.seq_len:
            pad = [
                [self.pad_value for _ in self.feature_cols]
                for _ in range(self.seq_len - len(series))
            ]
            series = pad + series
        return series

    def _coerce(self, value: object, default: float) -> float:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(num) or math.isinf(num):
            return default
        return num
