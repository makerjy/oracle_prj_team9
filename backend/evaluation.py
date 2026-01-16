from __future__ import annotations

from typing import Iterable, List, Optional


def window_label(event_time: Optional[int], t: int, horizon_hours: int) -> int:
    """
    Binary label: 1 if event occurs in (t, t + horizon_hours], else 0.
    event_time is hours from ICU admission.
    """
    if event_time is None:
        return 0
    return int(t < event_time <= t + horizon_hours)


def make_window_labels(
    times: Iterable[int], event_time: Optional[int], horizon_hours: int
) -> List[int]:
    return [window_label(event_time, t, horizon_hours) for t in times]
