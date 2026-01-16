from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Sequence

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .buffer import SequenceBuffer
from .model_loader import FEATURE_COLS, ModelBundle, infer_hazard_sequence, load_model_bundle

router = APIRouter()

ALERT_THRESHOLD = 0.20
ALERT_RISE = 0.05
MAX_TIMELINE = 720
MAX_ALERTS = 120

MODEL_BUNDLE: ModelBundle = load_model_bundle()
FEATURE_ORDER = MODEL_BUNDLE.meta.get("FEATURE_COLS", FEATURE_COLS)
SEQ_LEN = int(MODEL_BUNDLE.meta.get("L", 24))
BUFFER = SequenceBuffer(FEATURE_ORDER, SEQ_LEN, MODEL_BUNDLE.meta.get("impute_stats", {}))

STATE_LOCK = threading.Lock()
BUFFER_LOCK = threading.Lock()


@dataclass
class PatientState:
    origin_ts: datetime
    last_ts: datetime
    risk_history: List[Dict]
    alert_history: List[Dict]


PATIENT_STATE: Dict[str, PatientState] = {}


class InferRequest(BaseModel):
    patient_id: str = Field(..., examples=["P123"])
    timestamp: datetime
    features: Dict[str, float]


class InferResponse(BaseModel):
    patient_id: str
    timestamp: datetime
    hazard_seq: List[float]
    risk_6h: float
    risk_24h: float
    risk_72h: float
    alert_24h: bool
    alert_reason: List[str]


class PatientSummary(BaseModel):
    patient_id: str
    last_timestamp: datetime
    risk_6h: float
    risk_24h: float
    risk_72h: float
    alert_24h: bool
    alert_reason: List[str]


class TimelinePoint(BaseModel):
    t: int
    timestamp: datetime
    risk_6h: float
    risk_24h: float
    risk_72h: float
    alert_24h: bool


class TimelineResponse(BaseModel):
    patient_id: str
    timeline: List[TimelinePoint]
    alerts: List[Dict]


@router.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    if not request.patient_id:
        raise HTTPException(status_code=400, detail="patient_id is required")

    with BUFFER_LOCK:
        sequence = BUFFER.add(request.patient_id, request.features)
    hazard_seq = infer_hazard_sequence(MODEL_BUNDLE, sequence)

    risk_6h = compute_window_risk(hazard_seq, 6)
    risk_24h = compute_window_risk(hazard_seq, 24)
    risk_72h = compute_window_risk(hazard_seq, 72)

    effective_ts = request.timestamp

    with STATE_LOCK:
        state = PATIENT_STATE.get(request.patient_id)
        if state is None:
            state = PatientState(
                origin_ts=effective_ts,
                last_ts=effective_ts,
                risk_history=[],
                alert_history=[],
            )
            PATIENT_STATE[request.patient_id] = state
        else:
            if effective_ts <= state.last_ts:
                effective_ts = state.last_ts + timedelta(hours=1)

        elapsed_hours = _elapsed_hours(state.origin_ts, effective_ts)
        alert_24h, alert_reason = evaluate_alert(state, risk_24h)

        state.last_ts = effective_ts
        state.risk_history.append(
            {
                "t": elapsed_hours,
                "timestamp": effective_ts,
                "risk_6h": risk_6h,
                "risk_24h": risk_24h,
                "risk_72h": risk_72h,
                "alert_24h": alert_24h,
                "alert_reason": alert_reason,
            }
        )
        if len(state.risk_history) > MAX_TIMELINE:
            state.risk_history = state.risk_history[-MAX_TIMELINE:]

        if alert_24h:
            state.alert_history.append(
                {
                    "t": elapsed_hours,
                    "timestamp": effective_ts,
                    "reason": alert_reason,
                }
            )
            if len(state.alert_history) > MAX_ALERTS:
                state.alert_history = state.alert_history[-MAX_ALERTS:]

    return InferResponse(
        patient_id=request.patient_id,
        timestamp=effective_ts,
        hazard_seq=hazard_seq,
        risk_6h=risk_6h,
        risk_24h=risk_24h,
        risk_72h=risk_72h,
        alert_24h=alert_24h,
        alert_reason=alert_reason,
    )


@router.get("/patients", response_model=List[PatientSummary])
def list_patients() -> List[PatientSummary]:
    with STATE_LOCK:
        summaries = []
        for patient_id, state in PATIENT_STATE.items():
            if not state.risk_history:
                continue
            latest = state.risk_history[-1]
            summaries.append(
                PatientSummary(
                    patient_id=patient_id,
                    last_timestamp=state.last_ts,
                    risk_6h=latest["risk_6h"],
                    risk_24h=latest["risk_24h"],
                    risk_72h=latest["risk_72h"],
                    alert_24h=latest["alert_24h"],
                    alert_reason=latest["alert_reason"] if latest["alert_24h"] else [],
                )
            )
    summaries.sort(key=lambda item: item.patient_id)
    return summaries


@router.get("/patients/{patient_id}/timeline", response_model=TimelineResponse)
def patient_timeline(patient_id: str) -> TimelineResponse:
    with STATE_LOCK:
        state = PATIENT_STATE.get(patient_id)
        if state is None or not state.risk_history:
            raise HTTPException(status_code=404, detail="patient not found")
        timeline = [
            TimelinePoint(
                t=point["t"],
                timestamp=point["timestamp"],
                risk_6h=point["risk_6h"],
                risk_24h=point["risk_24h"],
                risk_72h=point["risk_72h"],
                alert_24h=point["alert_24h"],
            )
            for point in state.risk_history
        ]
        alerts = list(state.alert_history)
    return TimelineResponse(patient_id=patient_id, timeline=timeline, alerts=alerts)


def compute_window_risk(hazard_seq: Sequence[float], window: int) -> float:
    horizon = min(int(window), len(hazard_seq))
    survival = 1.0
    for value in hazard_seq[:horizon]:
        survival *= max(0.0, 1.0 - float(value))
    return 1.0 - survival


def evaluate_alert(state: PatientState, risk_24h: float) -> tuple[bool, List[str]]:
    if risk_24h < ALERT_THRESHOLD:
        return False, []

    reasons = ["risk_24h>=0.20"]
    consecutive = False
    rise = False

    if state.risk_history:
        prev = state.risk_history[-1]["risk_24h"]
        if prev >= ALERT_THRESHOLD:
            consecutive = True

    if len(state.risk_history) >= 6:
        past = state.risk_history[-6]["risk_24h"]
        if risk_24h - past >= ALERT_RISE:
            rise = True

    if not (consecutive or rise):
        return False, []

    if consecutive:
        reasons.append("2 consecutive breaches")
    if rise:
        reasons.append("6h rise>=0.05")

    return True, reasons


def _elapsed_hours(origin: datetime, current: datetime) -> int:
    delta = current - origin
    return max(0, int(delta.total_seconds() // 3600))
