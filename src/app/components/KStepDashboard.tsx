import { useEffect, useMemo, useRef, useState } from 'react';

import {
  fetchPatients,
  fetchTimeline,
  inferRisk,
  PatientSummary,
  TimelineResponse,
} from '@/app/api';
import PatientDetail from '@/app/components/PatientDetail';
import PatientList, { PatientListItem } from '@/app/components/PatientList';

const WARDS = [
  'Intensive Care Unit (ICU)',
  'Medical Intensive Care Unit (MICU)',
  'Cardiac Vascular Intensive Care Unit (CVICU)',
  'Medical/Surgical Intensive Care Unit (MICU/SICU)',
  'Surgical Intensive Care Unit (SICU)',
  'Trauma SICU (TSICU)',
  'Coronary Care Unit (CCU)',
  'Neuro Surgical Intensive Care Unit (Neuro SICU)',
];

const DEMO_PATIENTS = [
  { id: 'P1001', name: 'Patient A', ward: WARDS[0], severity: 0.2, drift: -0.01 },
  { id: 'P1002', name: 'Patient B', ward: WARDS[2], severity: 0.35, drift: 0.01 },
  { id: 'P1003', name: 'Patient C', ward: WARDS[4], severity: 0.6, drift: 0.02 },
  { id: 'P1004', name: 'Patient D', ward: WARDS[1], severity: 0.15, drift: -0.015 },
  { id: 'P1005', name: 'Patient E', ward: WARDS[6], severity: 0.45, drift: 0.005 },
];

interface SimState {
  severity: number;
  drift: number;
  timestamp: Date;
}

const clamp = (value: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, value));
const jitter = (value: number, range: number) => value + (Math.random() - 0.5) * range;

const buildFeatures = (severity: number) => ({
  HeartRate_std_6h: clamp(jitter(0.05 + severity * 0.2, 0.04), 0.02, 0.4),
  RespRate_std_6h: clamp(jitter(0.04 + severity * 0.18, 0.03), 0.02, 0.35),
  Temp_std_6h: clamp(jitter(0.02 + severity * 0.12, 0.02), 0.01, 0.25),
  GCS_Total_mean_6h: clamp(jitter(15 - severity * 7, 0.4), 3, 15),
  DiasBP_mean_6h: clamp(jitter(85 - severity * 30, 1.5), 45, 90),
  SysBP: clamp(jitter(130 - severity * 40, 2), 80, 140),
  MeanBP: clamp(jitter(95 - severity * 28, 1.5), 55, 100),
  SpO2_measured: clamp(jitter(98 - severity * 10, 0.4), 88, 100),
  FiO2: clamp(jitter(0.3 + severity * 0.5, 0.03), 0.21, 0.9),
  pH: clamp(jitter(7.42 - severity * 0.2, 0.02), 7.2, 7.45),
  GCS_Verbal: clamp(Math.round(5 - severity * 3), 1, 5),
  GCS_Motor: clamp(Math.round(6 - severity * 3), 1, 6),
});

export default function KStepDashboard() {
  const [patientSummaries, setPatientSummaries] = useState<PatientSummary[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState(DEMO_PATIENTS[0].id);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [isApiAvailable, setIsApiAvailable] = useState(true);
  const simStateRef = useRef<Record<string, SimState>>({});

  const patientList: PatientListItem[] = useMemo(
    () =>
      DEMO_PATIENTS.map((patient) => ({
        id: patient.id,
        name: patient.name,
        ward: patient.ward,
        summary: patientSummaries.find((item) => item.patient_id === patient.id),
      })),
    [patientSummaries]
  );

  useEffect(() => {
    if (Object.keys(simStateRef.current).length > 0) return;
    const now = new Date();
    DEMO_PATIENTS.forEach((patient, index) => {
      simStateRef.current[patient.id] = {
        severity: patient.severity,
        drift: patient.drift,
        timestamp: new Date(now.getTime() - (index + 1) * 3600 * 1000),
      };
    });
  }, []);

  useEffect(() => {
    let active = true;
    const tick = async () => {
      try {
        await Promise.all(
          DEMO_PATIENTS.map(async (patient) => {
            const state = simStateRef.current[patient.id];
            if (!state) return;
            const nextSeverity = clamp(
              state.severity + state.drift + (Math.random() - 0.5) * 0.02,
              0.05,
              0.9
            );
            const nextTimestamp = new Date(state.timestamp.getTime() + 3600 * 1000);
            simStateRef.current[patient.id] = {
              severity: nextSeverity,
              drift: state.drift,
              timestamp: nextTimestamp,
            };

            await inferRisk({
              patient_id: patient.id,
              timestamp: nextTimestamp.toISOString(),
              features: buildFeatures(nextSeverity),
            });
          })
        );

        const summaries = await fetchPatients();
        if (active) {
          setPatientSummaries(summaries);
          setIsApiAvailable(true);
        }
      } catch (error) {
        if (active) {
          setIsApiAvailable(false);
        }
      }
    };

    tick();
    const interval = window.setInterval(tick, 5000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!selectedPatientId || !isApiAvailable) return;
    let active = true;
    const refresh = async () => {
      try {
        const data = await fetchTimeline(selectedPatientId);
        if (active) {
          setTimeline(data);
        }
      } catch (error) {
        if (active) {
          setTimeline(null);
        }
      }
    };

    refresh();
    const interval = window.setInterval(refresh, 5000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, [selectedPatientId, isApiAvailable]);

  const selectedSummary = patientSummaries.find(
    (item) => item.patient_id === selectedPatientId
  );

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-xs text-slate-500">K-step hazard service</p>
            <h1 className="text-2xl font-semibold text-slate-900">
              ICU Risk Monitoring (K-step)
            </h1>
            <p className="text-sm text-slate-500">
              Model uses past sequence only. No future trajectory is rendered.
            </p>
          </div>
          <div
            className={`rounded-full px-3 py-1 text-xs font-semibold ${
              isApiAvailable ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
            }`}
          >
            {isApiAvailable ? 'API Connected' : 'API Offline'}
          </div>
        </div>

        <div className="mt-8 grid gap-6 lg:grid-cols-[280px_1fr]">
          <PatientList
            patients={patientList}
            selectedId={selectedPatientId}
            onSelect={setSelectedPatientId}
          />
          <PatientDetail
            patient={patientList.find((item) => item.id === selectedPatientId) ?? null}
            summary={selectedSummary}
            timeline={timeline?.timeline ?? []}
            alerts={timeline?.alerts ?? []}
          />
        </div>
      </div>
    </div>
  );
}
