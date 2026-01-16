from __future__ import annotations

import asyncio
import math
import random
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Deque, Dict, List, Sequence, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import router as api_router

app = FastAPI(title="ICU Risk Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")


async def demo_event_loop() -> None:
    while True:
        ts = now_utc()
        for patient in PATIENTS:
            stay_id = patient["stay_id"]
            signals = generate_signals(patient, ts)
            ingest_event_internal(stay_id, ts, signals)
        await asyncio.sleep(5)


@app.on_event("startup")
async def start_demo_seed() -> None:
    global demo_task
    if demo_task is None:
        for patient in PATIENTS:
            hours = 10 if patient["stay_id"] == "pt-0999" else 4
            seed_history_for_patient(patient, hours)
        demo_task = asyncio.create_task(demo_event_loop())

START_TIME = time.time()
MAX_TIME = 120
WINDOWS = [6, 24, 72]
WINDOW_HOURS = 6
MIN_COUNT = 3

FEATURE_ORDER = [
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

WARDS = [
    "Intensive Care Unit (ICU)",
    "Medical Intensive Care Unit (MICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Trauma SICU (TSICU)",
    "Coronary Care Unit (CCU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
]

PATIENTS = [
    {
        "stay_id": "pt-0142",
        "name": "김OO",
        "ward": WARDS[0],
        "room": "3-2",
        "guardian_name": "김OO 보호자",
        "gender": "남",
        "age": 76,
        "weight": 62,
        "admit_date": "2026-01-14",
        "diagnosis": "패혈증 및 호흡부전",
        "offset": 42,
        "trend": "worsen",
        "bases": {
            "heart_rate": 94,
            "sys_bp": 110,
            "mean_bp": 74,
            "resp_rate": 24,
            "spo2": 94,
            "wbc": 15.2,
            "creatinine": 1.8,
            "lactate": 3.2,
            "ph": 7.32,
            "glucose": 168,
            "gcs_total": 9,
            "gcs_eye": 3,
            "gcs_verbal": 2,
            "gcs_motor": 4,
            "pupil": 3,
        },
    },
    {
        "stay_id": "pt-0177",
        "name": "박OO",
        "ward": WARDS[1],
        "room": "3-5",
        "guardian_name": "박OO 보호자",
        "gender": "여",
        "age": 68,
        "weight": 55,
        "admit_date": "2026-01-13",
        "diagnosis": "폐렴 및 급성호흡곤란",
        "offset": 55,
        "trend": "worsen",
        "bases": {
            "heart_rate": 88,
            "sys_bp": 116,
            "mean_bp": 82,
            "resp_rate": 22,
            "spo2": 93,
            "wbc": 12.8,
            "creatinine": 1.3,
            "lactate": 2.4,
            "ph": 7.36,
            "glucose": 154,
            "gcs_total": 11,
            "gcs_eye": 4,
            "gcs_verbal": 3,
            "gcs_motor": 4,
            "pupil": 2.8,
        },
    },
    {
        "stay_id": "pt-0093",
        "name": "이OO",
        "ward": WARDS[2],
        "room": "4-1",
        "guardian_name": "이OO 보호자",
        "gender": "남",
        "age": 59,
        "weight": 70,
        "admit_date": "2026-01-12",
        "diagnosis": "심부전 및 신기능 저하",
        "offset": 67,
        "trend": "stable",
        "bases": {
            "heart_rate": 104,
            "sys_bp": 104,
            "mean_bp": 68,
            "resp_rate": 20,
            "spo2": 96,
            "wbc": 10.2,
            "creatinine": 2.3,
            "lactate": 2.8,
            "ph": 7.34,
            "glucose": 142,
            "gcs_total": 12,
            "gcs_eye": 4,
            "gcs_verbal": 4,
            "gcs_motor": 4,
            "pupil": 3.2,
        },
    },
    {
        "stay_id": "pt-0201",
        "name": "최OO",
        "ward": WARDS[3],
        "room": "4-3",
        "guardian_name": "최OO 보호자",
        "gender": "여",
        "age": 73,
        "weight": 58,
        "admit_date": "2026-01-11",
        "diagnosis": "다발성 장기부전",
        "offset": 81,
        "trend": "worsen",
        "bases": {
            "heart_rate": 98,
            "sys_bp": 92,
            "mean_bp": 64,
            "resp_rate": 26,
            "spo2": 91,
            "wbc": 17.1,
            "creatinine": 2.6,
            "lactate": 4.1,
            "ph": 7.28,
            "glucose": 182,
            "gcs_total": 8,
            "gcs_eye": 2,
            "gcs_verbal": 2,
            "gcs_motor": 4,
            "pupil": 3.6,
        },
    },
    {
        "stay_id": "pt-0228",
        "name": "정OO",
        "ward": WARDS[4],
        "room": "5-1",
        "guardian_name": "정OO 보호자",
        "gender": "남",
        "age": 64,
        "weight": 66,
        "admit_date": "2026-01-10",
        "diagnosis": "복부 수술 후 패혈증 의심",
        "offset": 36,
        "trend": "improve",
        "bases": {
            "heart_rate": 86,
            "sys_bp": 118,
            "mean_bp": 83,
            "resp_rate": 18,
            "spo2": 97,
            "wbc": 11.4,
            "creatinine": 1.1,
            "lactate": 1.9,
            "ph": 7.38,
            "glucose": 132,
            "gcs_total": 13,
            "gcs_eye": 4,
            "gcs_verbal": 4,
            "gcs_motor": 5,
            "pupil": 2.6,
        },
    },
    {
        "stay_id": "pt-0304",
        "name": "윤OO",
        "ward": WARDS[5],
        "room": "6-4",
        "guardian_name": "윤OO 보호자",
        "gender": "여",
        "age": 52,
        "weight": 60,
        "admit_date": "2026-01-09",
        "diagnosis": "외상성 출혈 후 회복 중",
        "offset": 24,
        "trend": "improve",
        "bases": {
            "heart_rate": 102,
            "sys_bp": 98,
            "mean_bp": 72,
            "resp_rate": 20,
            "spo2": 95,
            "wbc": 9.2,
            "creatinine": 0.9,
            "lactate": 2.2,
            "ph": 7.36,
            "glucose": 140,
            "gcs_total": 12,
            "gcs_eye": 4,
            "gcs_verbal": 3,
            "gcs_motor": 5,
            "pupil": 3.1,
        },
    },
    {
        "stay_id": "pt-0419",
        "name": "서OO",
        "ward": WARDS[6],
        "room": "7-2",
        "guardian_name": "서OO 보호자",
        "gender": "남",
        "age": 71,
        "weight": 72,
        "admit_date": "2026-01-08",
        "diagnosis": "급성 관상동맥 증후군",
        "offset": 48,
        "trend": "stable",
        "bases": {
            "heart_rate": 90,
            "sys_bp": 122,
            "mean_bp": 85,
            "resp_rate": 16,
            "spo2": 98,
            "wbc": 8.7,
            "creatinine": 1.0,
            "lactate": 1.6,
            "ph": 7.4,
            "glucose": 128,
            "gcs_total": 14,
            "gcs_eye": 4,
            "gcs_verbal": 5,
            "gcs_motor": 5,
            "pupil": 2.4,
        },
    },
    {
        "stay_id": "pt-0520",
        "name": "한OO",
        "ward": WARDS[7],
        "room": "8-3",
        "guardian_name": "한OO 보호자",
        "gender": "여",
        "age": 47,
        "weight": 54,
        "admit_date": "2026-01-07",
        "diagnosis": "신경외과 수술 후 모니터링",
        "offset": 30,
        "trend": "stable",
        "bases": {
            "heart_rate": 78,
            "sys_bp": 114,
            "mean_bp": 80,
            "resp_rate": 17,
            "spo2": 99,
            "wbc": 7.9,
            "creatinine": 0.8,
            "lactate": 1.3,
            "ph": 7.42,
            "glucose": 118,
            "gcs_total": 15,
            "gcs_eye": 4,
            "gcs_verbal": 5,
            "gcs_motor": 6,
            "pupil": 2.2,
        },
    },
    {
        "stay_id": "pt-0601",
        "name": "강OO",
        "ward": WARDS[0],
        "room": "3-7",
        "guardian_name": "강OO 보호자",
        "gender": "남",
        "age": 58,
        "weight": 68,
        "admit_date": "2026-01-16",
        "diagnosis": "응급 수술 후 즉시 입실",
        "offset": 1,
        "trend": "stable",
        "bases": {
            "heart_rate": 88,
            "sys_bp": 124,
            "mean_bp": 86,
            "resp_rate": 18,
            "spo2": 98,
            "wbc": 9.5,
            "creatinine": 1.0,
            "lactate": 1.6,
            "ph": 7.4,
            "glucose": 126,
            "gcs_total": 14,
            "gcs_eye": 4,
            "gcs_verbal": 5,
            "gcs_motor": 5,
            "pupil": 2.6,
        },
    },
    {
        "stay_id": "pt-0712",
        "name": "오OO",
        "ward": WARDS[6],
        "room": "7-5",
        "guardian_name": "오OO 보호자",
        "gender": "여",
        "age": 62,
        "weight": 57,
        "admit_date": "2026-01-15",
        "diagnosis": "관상동맥 중재술 후 경과 관찰",
        "offset": 8,
        "trend": "improve",
        "bases": {
            "heart_rate": 72,
            "sys_bp": 128,
            "mean_bp": 90,
            "resp_rate": 16,
            "spo2": 99,
            "wbc": 6.2,
            "creatinine": 0.7,
            "lactate": 1.1,
            "ph": 7.41,
            "glucose": 110,
            "gcs_total": 15,
            "gcs_eye": 4,
            "gcs_verbal": 5,
            "gcs_motor": 6,
            "pupil": 2.1,
        },
    },
    {
        "stay_id": "pt-0818",
        "name": "장OO",
        "ward": WARDS[1],
        "room": "2-4",
        "guardian_name": "장OO 보호자",
        "gender": "남",
        "age": 49,
        "weight": 75,
        "admit_date": "2026-01-15",
        "diagnosis": "폐렴 호전 후 관찰",
        "offset": 12,
        "trend": "improve",
        "bases": {
            "heart_rate": 76,
            "sys_bp": 120,
            "mean_bp": 88,
            "resp_rate": 17,
            "spo2": 98,
            "wbc": 7.0,
            "creatinine": 0.8,
            "lactate": 1.2,
            "ph": 7.39,
            "glucose": 114,
            "gcs_total": 15,
            "gcs_eye": 4,
            "gcs_verbal": 5,
            "gcs_motor": 6,
            "pupil": 2.3,
        },
    },
    {
        "stay_id": "pt-0999",
        "name": "류OO",
        "ward": WARDS[0],
        "room": "1-1",
        "guardian_name": "류OO 보호자",
        "gender": "남",
        "age": 81,
        "weight": 59,
        "admit_date": "2026-01-16",
        "diagnosis": "패혈성 쇼크 의심",
        "offset": 6,
        "trend": "worsen",
        "bases": {
            "heart_rate": 122,
            "sys_bp": 86,
            "mean_bp": 55,
            "resp_rate": 30,
            "spo2": 88,
            "wbc": 19.8,
            "creatinine": 3.2,
            "lactate": 5.4,
            "ph": 7.18,
            "glucose": 210,
            "gcs_total": 7,
            "gcs_eye": 2,
            "gcs_verbal": 2,
            "gcs_motor": 3,
            "pupil": 4.0,
        },
    },
]

PATIENT_LOOKUP = {patient["stay_id"]: patient for patient in PATIENTS}
RISK_BOOST = {
    "pt-0999": 2.8,
}


def elapsed_hours(offset: int) -> int:
    # 1 hour per 30 seconds for demo
    hours = int((time.time() - START_TIME) / 30)
    return max(0, min(MAX_TIME, hours + offset))


def seeded_rng(stay_id: str, salt: str) -> random.Random:
    seed = hash(f"{stay_id}:{salt}") & 0xFFFFFFFF
    return random.Random(seed)


def generate_series(base: float, variance: float, current_time: int, rng: random.Random) -> List[dict]:
    data = []
    for t in range(0, current_time + 1, 6):
        wave = math.sin(t / 14) * variance * 0.6
        noise = (rng.random() - 0.5) * variance
        data.append({"time": t, "value": round(base + wave + noise, 2)})
    return data


def cumulative_risk_from_hazards(hazards: Sequence[float]) -> float:
    survival = 1.0
    for hazard in hazards:
        survival *= max(0.0, 1 - hazard)
    return 1 - survival


def build_hourly_trajectory(history: List[Dict[str, float]]) -> List[Dict[str, float]]:
    if not history:
        return []

    bins: Dict[int, float] = {}
    for point in history:
        hour = int(point["t"])
        bins[hour] = point["hazard"]

    max_hour = max(bins.keys())
    default_hazard = next(iter(bins.values()))
    last_hazard = default_hazard
    survival = 1.0
    trajectory = []
    for hour in range(0, max_hour + 1):
        hazard = bins.get(hour, last_hazard)
        last_hazard = hazard
        survival *= max(0.0, 1 - hazard)
        trajectory.append(
            {
                "t": hour,
                "hazard": round(hazard, 4),
                "cum_risk": round(1 - survival, 4),
            }
        )
    return trajectory


def generate_trajectory_from_features(current_time: int, feature_builder) -> List[dict]:
    data = []
    hazards = []
    for t in range(0, current_time + 1, 1):
        features = feature_builder(t)
        hazard = model_predict_hazard(features, MODEL)
        hazards.append(hazard)
        data.append(
            {
                "t": t,
                "hazard": round(hazard, 4),
                "cum_risk": round(cumulative_risk_from_hazards(hazards), 4),
            }
        )
    return data


def window_risk_from_hazard(hazard_seq: Sequence[float], hours: int, dt: float = 1.0) -> float:
    horizon = min(hours, len(hazard_seq))
    cumulative_hazard = sum(hazard_seq[:horizon]) * dt
    return 1 - math.exp(-cumulative_hazard)


def model_predict_hazard(features: dict, model=None) -> float:
    # Placeholder for BoxHED integration: replace with model inference.
    # features is current-time feature row only (no future data).
    if model is None:
        base = features.get("lactate", 2.0) * 0.022 + features.get("wbc", 8) * 0.0015
        return max(0.005, min(0.08, base))
    return float(model.predict(features))


def model_predict_hazard_seq(features: dict, horizon_hours: int, model=None) -> List[float]:
    # Placeholder: hazard sequence derived from current-time features only.
    base = model_predict_hazard(features, model)
    seq = []
    for k in range(1, horizon_hours + 1):
        drift = math.sin(k / 10) * 0.01
        seq.append(max(0.002, min(0.1, base + drift)))
    return seq


@app.get("/demo/patients")
def get_demo_patients():
    response = []
    for patient in PATIENTS:
        current_time = elapsed_hours(patient["offset"])
        response.append(
            {
                "stay_id": patient["stay_id"],
                "name": patient["name"],
                "ward": patient["ward"],
                "room": patient["room"],
                "gender": patient["gender"],
                "age": patient["age"],
                "weight": patient["weight"],
                "admit_date": patient["admit_date"],
                "diagnosis": patient["diagnosis"],
                "elapsed_hours": current_time,
            }
        )
    return response


@app.get("/demo/patients/{stay_id}/risk-summary")
def get_demo_risk_summary(stay_id: str):
    patient = next((item for item in PATIENTS if item["stay_id"] == stay_id), None)
    if not patient:
        raise HTTPException(status_code=404, detail="stay_id not found")

    current_time = elapsed_hours(patient["offset"])
    time_factor = math.sin(time.time() / 40) * 0.6

    def feature_builder(_: int) -> dict:
        return {
            "time": current_time,
            "lactate": patient["bases"]["lactate"] + time_factor * 0.15,
            "wbc": patient["bases"]["wbc"] + time_factor * 0.1,
        }

    hazard_seq = model_predict_hazard_seq(feature_builder(current_time), max(WINDOWS), MODEL)
    trajectory = generate_trajectory_from_features(current_time, feature_builder)
    recent_window = trajectory[-6:] if len(trajectory) >= 6 else trajectory
    recent_hazards = [point["hazard"] for point in recent_window] if recent_window else []
    current_hazard = recent_hazards[-1] if recent_hazards else 0.0
    recent_6h_avg = sum(recent_hazards) / len(recent_hazards) if recent_hazards else 0.0
    recent_6h_slope = (
        (recent_hazards[-1] - recent_hazards[0]) / max(len(recent_hazards) - 1, 1)
        if recent_hazards
        else 0.0
    )
    cum_risk_120h_est = cumulative_risk_from_hazards(hazard_seq)
    return {
        "current_time": current_time,
        "current_hazard": round(current_hazard, 4),
        "recent_6h_avg": round(recent_6h_avg, 4),
        "recent_6h_slope": round(recent_6h_slope, 4),
        "cum_risk_120h_est": round(cum_risk_120h_est, 4),
    }


@app.get("/demo/patients/{stay_id}/risk-trajectory")
def get_demo_risk_trajectory(stay_id: str):
    patient = next((item for item in PATIENTS if item["stay_id"] == stay_id), None)
    if not patient:
        raise HTTPException(status_code=404, detail="stay_id not found")

    current_time = elapsed_hours(patient["offset"])
    time_factor = math.sin(time.time() / 40) * 0.6

    def feature_builder(t: int) -> dict:
        progress = t / max(current_time, 1)
        trend = patient.get("trend", "stable")
        if trend == "worsen":
            trend_factor = 0.6 * progress
        elif trend == "improve":
            trend_factor = -0.35 * progress
        else:
            trend_factor = 0.05 * progress
        return {
            "time": t,
            "lactate": patient["bases"]["lactate"] + time_factor * 0.15 + trend_factor,
            "wbc": patient["bases"]["wbc"] + time_factor * 0.1 + trend_factor * 2.0,
        }

    risk_series = generate_trajectory_from_features(current_time, feature_builder)

    vital = [
        ("심박수", "bpm", [60, 100], patient["bases"]["heart_rate"], 10),
        ("수축기혈압", "mmHg", [90, 140], patient["bases"]["sys_bp"], 14),
        ("평균혈압", "mmHg", [70, 100], patient["bases"]["mean_bp"], 10),
        ("호흡수", "/분", [12, 20], patient["bases"]["resp_rate"], 4),
        ("산소포화도", "%", [95, 100], patient["bases"]["spo2"], 3),
    ]
    lab = [
        ("백혈구(WBC)", "K/μL", [4, 11], patient["bases"]["wbc"], 1.6),
        ("크레아티닌", "mg/dL", [0.6, 1.2], patient["bases"]["creatinine"], 0.25),
        ("젖산", "mmol/L", [0.5, 2.0], patient["bases"]["lactate"], 0.5),
        ("pH", "", [7.35, 7.45], patient["bases"]["ph"], 0.06),
        ("혈당", "mg/dL", [70, 140], patient["bases"]["glucose"], 18),
    ]
    neuro = [
        ("GCS 총점", "점", [13, 15], patient["bases"]["gcs_total"], 1),
        ("GCS 눈", "점", [4, 4], patient["bases"]["gcs_eye"], 0.5),
        ("GCS 언어", "점", [5, 5], patient["bases"]["gcs_verbal"], 0.6),
        ("GCS 운동", "점", [6, 6], patient["bases"]["gcs_motor"], 0.7),
        ("동공 반응", "mm", [2, 4], patient["bases"]["pupil"], 0.4),
    ]

    def build_items(items, salt):
        results = []
        for name, unit, normal, base, variance in items:
            data = generate_series(base + time_factor, variance, current_time, seeded_rng(stay_id, f"{salt}-{name}"))
            current_value = data[-1]["value"] if data else base
            results.append(
                {
                    "name": name,
                    "unit": unit,
                    "normal": normal,
                    "current": round(current_value, 2),
                    "data": data,
                }
            )
        return results

    vital_items = build_items(vital, "vital")
    lab_items = build_items(lab, "lab")
    neuro_items = build_items(neuro, "neuro")

    return {
        "stay_id": stay_id,
        "current_time": current_time,
        "elapsed_hours": current_time,
        "trajectory": risk_series,
        "vitals": {
            "vital": vital_items,
            "lab": lab_items,
            "neuro": neuro_items,
        },
        "status": {
            "circulation": {
                "meanBP": vital_items[2]["current"],
                "vasopressor": "사용 중",
                "drug": "Norepinephrine 0.15 μg/kg/min",
            },
            "respiration": {
                "spo2": vital_items[4]["current"],
                "fio2": 60,
                "peep": 8,
            },
            "neurologic": {
                "gcs": neuro_items[0]["current"],
                "trend": "↓ 감소",
            },
            "infection": {
                "lactate": lab_items[2]["current"],
                "wbc": lab_items[0]["current"],
            },
        },
    }


@app.post("/events")
def ingest_event(payload: dict):
    stay_id = payload.get("stay_id")
    timestamp = payload.get("timestamp")
    signals = payload.get("signals", {})
    if not stay_id or not timestamp:
        raise HTTPException(status_code=400, detail="stay_id and timestamp are required")

    ts = parse_timestamp(timestamp)
    result = ingest_event_internal(stay_id, ts, signals)

    return {
        "stay_id": stay_id,
        "timestamp": ts.isoformat(),
        "hazard": round(result["hazard"], 4),
        "cum_risk": round(result["cum_risk"], 4),
        "features": result["features"],
    }


@app.get("/patients")
def list_patients():
    response = []
    for stay_id in PATIENT_LOOKUP.keys():
        ensure_history(stay_id)
        history = hazard_history.get(stay_id, [])
        trajectory = build_hourly_trajectory(history)
        latest = trajectory[-1] if trajectory else {"hazard": 0.0, "cum_risk": 0.0}
        response.append(
            {
                "stay_id": stay_id,
                "last_timestamp": last_timestamp.get(stay_id).isoformat()
                if stay_id in last_timestamp
                else None,
                "current_hazard": latest["hazard"],
                "cum_risk": latest["cum_risk"],
                "name": PATIENT_LOOKUP.get(stay_id, {}).get("name"),
                "ward": PATIENT_LOOKUP.get(stay_id, {}).get("ward"),
                "room": PATIENT_LOOKUP.get(stay_id, {}).get("room"),
                "guardian_name": PATIENT_LOOKUP.get(stay_id, {}).get("guardian_name"),
                "gender": PATIENT_LOOKUP.get(stay_id, {}).get("gender"),
                "age": PATIENT_LOOKUP.get(stay_id, {}).get("age"),
                "weight": PATIENT_LOOKUP.get(stay_id, {}).get("weight"),
                "diagnosis": PATIENT_LOOKUP.get(stay_id, {}).get("diagnosis"),
            }
        )
    return response


@app.get("/patients/{stay_id}/risk-trajectory")
def get_stream_trajectory(stay_id: str):
    ensure_history(stay_id)
    history = hazard_history.get(stay_id)
    if not history:
        raise HTTPException(status_code=404, detail="stay_id not found")
    trajectory = build_hourly_trajectory(history)
    current_time = trajectory[-1]["t"] if trajectory else 0
    return {
        "stay_id": stay_id,
        "current_time": current_time,
        "elapsed_hours": current_time,
        "trajectory": trajectory,
    }


@app.get("/patients/{stay_id}/risk-summary")
def get_stream_summary(stay_id: str):
    ensure_history(stay_id)
    history = hazard_history.get(stay_id)
    if not history:
        raise HTTPException(status_code=404, detail="stay_id not found")
    trajectory = build_hourly_trajectory(history)
    recent = trajectory[-6:] if len(trajectory) >= 6 else trajectory
    hazards = [point["hazard"] for point in recent] if recent else []
    current_hazard = hazards[-1] if hazards else 0.0
    recent_6h_avg = sum(hazards) / len(hazards) if hazards else 0.0
    recent_6h_slope = (
        (hazards[-1] - hazards[0]) / max(len(hazards) - 1, 1) if hazards else 0.0
    )
    return {
        "current_time": trajectory[-1]["t"] if trajectory else 0,
        "current_hazard": round(current_hazard, 4),
        "recent_6h_avg": round(recent_6h_avg, 4),
        "recent_6h_slope": round(recent_6h_slope, 4),
        "cum_risk_120h_est": round(trajectory[-1]["cum_risk"] if trajectory else 0.0, 4),
    }
def load_model():
    # Placeholder for BoxHED model loading.
    # Replace with actual PyTorch loading and device placement.
    return None


MODEL = load_model()

buffers: Dict[str, Dict[str, Deque[Tuple[datetime, float]]]] = defaultdict(
    lambda: defaultdict(deque)
)
hazard_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
survival_state: Dict[str, float] = defaultdict(lambda: 1.0)
stay_start: Dict[str, datetime] = {}
last_timestamp: Dict[str, datetime] = {}
demo_task: asyncio.Task | None = None


def parse_timestamp(timestamp: str) -> datetime:
    if timestamp.endswith("Z"):
        timestamp = timestamp.replace("Z", "+00:00")
    return datetime.fromisoformat(timestamp).astimezone(timezone.utc)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def prune_buffer(stay_buffer: Dict[str, Deque[Tuple[datetime, float]]], now: datetime) -> None:
    cutoff = now - timedelta(hours=WINDOW_HOURS)
    for series in stay_buffer.values():
        while series and series[0][0] < cutoff:
            series.popleft()


def latest_value(series: Deque[Tuple[datetime, float]]) -> float:
    return series[-1][1] if series else float("nan")


def stats_or_nan(values: List[float]) -> Tuple[float, float]:
    if len(values) < MIN_COUNT:
        return float("nan"), float("nan")
    return mean(values), pstdev(values)


def compute_features(stay_buffer: Dict[str, Deque[Tuple[datetime, float]]]) -> Dict[str, float]:
    hr_values = [value for _, value in stay_buffer.get("HeartRate", [])]
    rr_values = [value for _, value in stay_buffer.get("RespRate", [])]
    temp_values = [value for _, value in stay_buffer.get("Temp", [])]
    gcs_values = [value for _, value in stay_buffer.get("GCS_Total", [])]
    dias_values = [value for _, value in stay_buffer.get("DiasBP", [])]

    gcs_mean, _ = stats_or_nan(gcs_values)
    dias_mean, _ = stats_or_nan(dias_values)
    _, hr_std = stats_or_nan(hr_values)
    _, rr_std = stats_or_nan(rr_values)
    _, temp_std = stats_or_nan(temp_values)

    return {
        "HeartRate_std_6h": hr_std,
        "RespRate_std_6h": rr_std,
        "Temp_std_6h": temp_std,
        "GCS_Total_mean_6h": gcs_mean,
        "DiasBP_mean_6h": dias_mean,
        "SysBP": latest_value(stay_buffer.get("SysBP", deque())),
        "MeanBP": latest_value(stay_buffer.get("MeanBP", deque())),
        "SpO2_measured": latest_value(stay_buffer.get("SpO2", deque())),
        "FiO2": latest_value(stay_buffer.get("FiO2", deque())),
        "pH": latest_value(stay_buffer.get("pH", deque())),
        "GCS_Verbal": latest_value(stay_buffer.get("GCS_Verbal", deque())),
        "GCS_Motor": latest_value(stay_buffer.get("GCS_Motor", deque())),
    }


def impute_feature(value: float, default: float) -> float:
    if value != value:
        return default
    return value


def feature_vector(features: Dict[str, float]) -> List[float]:
    defaults = {
        "HeartRate_std_6h": 5.0,
        "RespRate_std_6h": 3.0,
        "Temp_std_6h": 0.3,
        "GCS_Total_mean_6h": 12.0,
        "DiasBP_mean_6h": 70.0,
        "SysBP": 110.0,
        "MeanBP": 75.0,
        "SpO2_measured": 96.0,
        "FiO2": 0.3,
        "pH": 7.4,
        "GCS_Verbal": 4.0,
        "GCS_Motor": 5.0,
    }
    return [impute_feature(features[name], defaults[name]) for name in FEATURE_ORDER]


def predict_hazard(x: List[float]) -> float:
    if MODEL is None:
        weighted = (
            0.0015 * x[0]
            + 0.0012 * x[1]
            + 0.01 * x[2]
            - 0.002 * x[3]
            + 0.001 * (100 - x[7])
        )
        return max(0.005, min(0.12, 0.02 + weighted))
    return float(MODEL.predict(x))


def ingest_event_internal(stay_id: str, ts: datetime, signals: Dict[str, float]) -> Dict[str, float]:
    stay_buffer = buffers[stay_id]
    if stay_id not in stay_start:
        stay_start[stay_id] = ts

    for signal, value in signals.items():
        stay_buffer[signal].append((ts, float(value)))

    prune_buffer(stay_buffer, ts)
    last_timestamp[stay_id] = ts

    features = compute_features(stay_buffer)
    x = feature_vector(features)
    hazard = predict_hazard(x)
    if stay_id in RISK_BOOST:
        hazard = min(0.2, hazard * RISK_BOOST[stay_id])
    survival_state[stay_id] *= max(0.0, 1 - hazard)
    cum_risk = 1 - survival_state[stay_id]

    hazard_history[stay_id].append(
        {
            "t": (ts - stay_start[stay_id]).total_seconds() / 3600,
            "hazard": hazard,
            "cum_risk": cum_risk,
        }
    )

    return {
        "hazard": hazard,
        "cum_risk": cum_risk,
        "features": features,
    }


def generate_signals(patient: dict, ts: datetime) -> Dict[str, float]:
    bases = patient["bases"]
    trend = patient.get("trend", "stable")
    elapsed_hours = (ts - stay_start.get(patient["stay_id"], ts)).total_seconds() / 3600
    progress = min(max(elapsed_hours / 72, 0), 1)
    if trend == "worsen":
        trend_factor = 0.6 * progress
    elif trend == "improve":
        trend_factor = -0.35 * progress
    else:
        trend_factor = 0.05 * progress

    noise = lambda scale: (random.random() - 0.5) * scale

    return {
        "HeartRate": bases["heart_rate"] + trend_factor * 4 + noise(6),
        "RespRate": bases["resp_rate"] + trend_factor * 2 + noise(3),
        "Temp": 36.8 + trend_factor * 0.6 + noise(0.4),
        "SysBP": bases["sys_bp"] - trend_factor * 8 + noise(6),
        "DiasBP": bases["mean_bp"] - trend_factor * 6 + noise(4),
        "MeanBP": bases["mean_bp"] - trend_factor * 7 + noise(4),
        "SpO2": bases["spo2"] - trend_factor * 2 + noise(1.5),
        "FiO2": 0.3 + trend_factor * 0.1 + noise(0.05),
        "pH": bases["ph"] - trend_factor * 0.03 + noise(0.02),
        "GCS_Verbal": bases["gcs_verbal"] - trend_factor * 0.5 + noise(0.3),
        "GCS_Motor": bases["gcs_motor"] - trend_factor * 0.4 + noise(0.3),
        "GCS_Total": bases["gcs_total"] - trend_factor * 1.2 + noise(0.6),
    }


def seed_history_for_patient(patient: dict, hours: int) -> None:
    end_ts = now_utc()
    start_ts = end_ts - timedelta(hours=hours)
    current = start_ts
    while current <= end_ts:
        signals = generate_signals(patient, current)
        ingest_event_internal(patient["stay_id"], current, signals)
        current += timedelta(minutes=20)


def ensure_history(stay_id: str) -> None:
    if stay_id in hazard_history and hazard_history[stay_id]:
        return
    patient = PATIENT_LOOKUP.get(stay_id)
    if not patient:
        return
    hours = 10 if stay_id == "pt-0999" else 4
    seed_history_for_patient(patient, hours)
