import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import {
  fetchPatients as fetchKStepPatients,
  fetchTimeline as fetchKStepTimeline,
  inferRisk as inferKStepRisk,
  PatientSummary as KStepPatientSummary,
  TimelineResponse as KStepTimelineResponse,
} from '@/app/api';

interface MedicalStaffDashboardProps {
  isDarkMode: boolean;
  selectedPatientId: string;
  onSelectPatient: (patientId: string) => void;
}

type TabType = 'vital' | 'lab' | 'neuro';

type VitalSeriesPoint = {
  time: number;
  value: number;
};

type VitalItem = {
  name: string;
  unit: string;
  current: number;
  normal: [number, number];
  data: VitalSeriesPoint[];
};

type Patient = {
  id: string;
  wardId: string;
  wardName: string;
  name: string;
  gender: string;
  age: number;
  weight: number;
  admitDate: string;
  room: string;
  elapsedHours: number;
  diagnosis: string;
  riskSummary: {
    currentHazard: number;
    recent6hAvg: number;
    recent6hSlope: number;
    cumRisk120hEst: number;
  };
  riskSeries: Array<{ t: number; hazard: number; cumRisk: number }>;
  vitals: {
    vital: VitalItem[];
    lab: VitalItem[];
    neuro: VitalItem[];
  };
  status: {
    circulation: { meanBP: number; vasopressor: string; drug: string };
    respiration: { spo2: number; fio2: number; peep: number };
    neurologic: { gcs: number; trend: string };
    infection: { lactate: number; wbc: number };
  };
};

type Ward = {
  id: string;
  name: string;
  patients: Patient[];
};

type ApiPatient = {
  stay_id: string;
  name?: string;
  ward?: string;
  room?: string;
  current_hazard?: number;
  cum_risk?: number;
  gender?: string;
  age?: number;
  weight?: number;
  admit_date?: string;
  diagnosis?: string;
  elapsed_hours?: number;
};

type ApiRiskResponse = {
  stay_id: string;
  current_time: number;
  elapsed_hours: number;
  trajectory: Array<{ t: number; hazard: number; cum_risk: number }>;
  vitals?: {
    vital: VitalItem[];
    lab: VitalItem[];
    neuro: VitalItem[];
  };
  status?: Patient['status'];
};

type ApiRiskSummary = {
  current_time: number;
  current_hazard: number;
  recent_6h_avg: number;
  recent_6h_slope: number;
  cum_risk_120h_est: number;
};

type KStepSimState = {
  severity: number;
  drift: number;
  timestamp: Date;
  minSeverity: number;
  maxSeverity: number;
};

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

const MAX_TIME = 120;

const getWardLabel = (wardName: string) => {
  if (wardName.includes('CVICU')) return 'CVICU';
  if (wardName.includes('CCU')) return 'CCU';
  if (wardName.includes('Neuro SICU')) return 'Neuro SICU';
  if (wardName.includes('MICU/SICU')) return 'MICU/SICU';
  if (wardName.includes('MICU')) return 'MICU';
  if (wardName.includes('TSICU')) return 'TSICU';
  if (wardName.includes('SICU')) return 'SICU';
  if (wardName.includes('ICU')) return 'ICU';
  return wardName;
};

const createRng = (seed: number) => {
  let value = seed;
  return () => {
    value |= 0;
    value = (value + 0x6d2b79f5) | 0;
    let t = Math.imul(value ^ (value >>> 15), 1 | value);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

const hashString = (value: string) => {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
};

const jitter = (value: number, range: number) => value + (Math.random() - 0.5) * range;

const buildKStepFeatures = (severity: number) => ({
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

const generateVitalSeries = (base: number, variance: number, rng: () => number) => {
  const data: VitalSeriesPoint[] = [];
  for (let i = 0; i <= 120; i += 6) {
    const noise = (rng() - 0.5) * variance;
    const wave = Math.sin(i / 15) * (variance * 0.6);
    data.push({ time: i, value: base + wave + noise });
  }
  return data;
};

const generateRiskSeries = (base: number, rng: () => number, currentTime: number) => {
  const data: Array<{ t: number; hazard: number; cumRisk: number }> = [];
  let survival = 1;
  for (let i = 0; i <= currentTime; i += 1) {
    const drift = Math.sin(i / 10) * 0.01 + (rng() - 0.5) * 0.008;
    const hazard = clamp(base + drift, 0.005, 0.12);
    survival *= Math.max(0, 1 - hazard);
    data.push({
      t: i,
      hazard,
      cumRisk: 1 - survival,
    });
  }
  return data;
};

const getLatestValue = (data: VitalSeriesPoint[]) => data[data.length - 1]?.value ?? 0;

const buildSyntheticBases = (seed: number) => {
  const rng = createRng(seed);
  return {
    heartRate: 78 + rng() * 28,
    sysBP: 108 + rng() * 24,
    meanBP: 70 + rng() * 18,
    respRate: 16 + rng() * 8,
    spo2: 92 + rng() * 6,
    wbc: 7 + rng() * 6,
    creatinine: 0.9 + rng() * 1.0,
    lactate: 1.2 + rng() * 1.8,
    ph: 7.32 + rng() * 0.12,
    glucose: 110 + rng() * 50,
    gcsTotal: 9 + rng() * 6,
    gcsEye: 3 + rng() * 1,
    gcsVerbal: 3 + rng() * 2,
    gcsMotor: 4 + rng() * 2,
    pupil: 2.6 + rng() * 1.0,
  };
};

const getSyntheticVitals = (base: Patient, seed: number) => {
  const profile = {
    id: base.id,
    wardId: base.wardId,
    wardName: base.wardName,
    name: base.name,
    gender: base.gender || '-',
    age: base.age || 60,
    weight: base.weight || 65,
    admitDate: base.admitDate && base.admitDate !== '-' ? base.admitDate : '2026-01-01',
    room: base.room || '1-1',
    elapsedHours: Math.max(6, Math.round(base.elapsedHours || 12)),
    diagnosis: base.diagnosis || '상태 관찰',
    bases: buildSyntheticBases(seed),
  };
  const synthetic = createPatient(seed, profile);
  return { vitals: synthetic.vitals, status: synthetic.status };
};

const getRiskLabel = (score: number, mode: 'hazard' | 'risk24h' = 'hazard') => {
  const thresholds = mode === 'risk24h' ? { high: 0.2, mid: 0.1 } : { high: 0.08, mid: 0.04 };
  if (score >= thresholds.high) return { label: '위험', color: 'bg-red-600 text-white' };
  if (score >= thresholds.mid) return { label: '주의', color: 'bg-amber-500 text-black' };
  return { label: '안정', color: 'bg-emerald-500 text-white' };
};

const formatPercent = (value?: number, digits = 1) =>
  typeof value === 'number' && !Number.isNaN(value)
    ? `${(value * 100).toFixed(digits)}%`
    : '--';

const createPatient = (seed: number, profile: Omit<Patient, 'riskSummary' | 'riskSeries' | 'vitals' | 'status'> & {
  bases: {
    heartRate: number;
    sysBP: number;
    meanBP: number;
    respRate: number;
    spo2: number;
    wbc: number;
    creatinine: number;
    lactate: number;
    ph: number;
    glucose: number;
    gcsTotal: number;
    gcsEye: number;
    gcsVerbal: number;
    gcsMotor: number;
    pupil: number;
  };
}) => {
  const rng = createRng(seed);

  const vital = [
    { name: '심박수', unit: 'bpm', normal: [60, 100], base: profile.bases.heartRate, variance: 10 },
    { name: '수축기혈압', unit: 'mmHg', normal: [90, 140], base: profile.bases.sysBP, variance: 14 },
    { name: '평균혈압', unit: 'mmHg', normal: [70, 100], base: profile.bases.meanBP, variance: 10 },
    { name: '호흡수', unit: '/분', normal: [12, 20], base: profile.bases.respRate, variance: 4 },
    { name: '산소포화도', unit: '%', normal: [95, 100], base: profile.bases.spo2, variance: 3 },
  ].map((item) => {
    const data = generateVitalSeries(item.base, item.variance, rng);
    return {
      name: item.name,
      unit: item.unit,
      normal: item.normal as [number, number],
      data,
      current: parseFloat(getLatestValue(data).toFixed(1)),
    };
  });

  const lab = [
    { name: '백혈구(WBC)', unit: 'K/μL', normal: [4, 11], base: profile.bases.wbc, variance: 1.6 },
    { name: '크레아티닌', unit: 'mg/dL', normal: [0.6, 1.2], base: profile.bases.creatinine, variance: 0.25 },
    { name: '젖산', unit: 'mmol/L', normal: [0.5, 2.0], base: profile.bases.lactate, variance: 0.5 },
    { name: 'pH', unit: '', normal: [7.35, 7.45], base: profile.bases.ph, variance: 0.06 },
    { name: '혈당', unit: 'mg/dL', normal: [70, 140], base: profile.bases.glucose, variance: 18 },
  ].map((item) => {
    const data = generateVitalSeries(item.base, item.variance, rng);
    return {
      name: item.name,
      unit: item.unit,
      normal: item.normal as [number, number],
      data,
      current: parseFloat(getLatestValue(data).toFixed(1)),
    };
  });

  const neuro = [
    { name: 'GCS 총점', unit: '점', normal: [13, 15], base: profile.bases.gcsTotal, variance: 1 },
    { name: 'GCS 눈', unit: '점', normal: [4, 4], base: profile.bases.gcsEye, variance: 0.5 },
    { name: 'GCS 언어', unit: '점', normal: [5, 5], base: profile.bases.gcsVerbal, variance: 0.6 },
    { name: 'GCS 운동', unit: '점', normal: [6, 6], base: profile.bases.gcsMotor, variance: 0.7 },
    { name: '동공 반응', unit: 'mm', normal: [2, 4], base: profile.bases.pupil, variance: 0.4 },
  ].map((item) => {
    const data = generateVitalSeries(item.base, item.variance, rng);
    return {
      name: item.name,
      unit: item.unit,
      normal: item.normal as [number, number],
      data,
      current: parseFloat(getLatestValue(data).toFixed(1)),
    };
  });

  const currentTime = Math.min(profile.elapsedHours, MAX_TIME);
  const riskSeries = generateRiskSeries(profile.bases.lactate * 0.02, rng, currentTime);
  const recentWindow = riskSeries.slice(-6);
  const recentHazards = recentWindow.map((point) => point.hazard);
  const currentHazard = recentHazards[recentHazards.length - 1] ?? 0;
  const recent6hAvg =
    recentHazards.length > 0
      ? recentHazards.reduce((sum, value) => sum + value, 0) / recentHazards.length
      : 0;
  const recent6hSlope =
    recentHazards.length > 1
      ? (recentHazards[recentHazards.length - 1] - recentHazards[0]) / (recentHazards.length - 1)
      : 0;
  const latestRisk = {
    currentHazard,
    recent6hAvg,
    recent6hSlope,
    cumRisk120hEst: riskSeries[riskSeries.length - 1]?.cumRisk ?? 0,
  };

  return {
    ...profile,
    riskSummary: latestRisk,
    riskSeries,
    vitals: { vital, lab, neuro },
    status: {
      circulation: {
        meanBP: parseFloat(getLatestValue(vital[2].data).toFixed(0)),
        vasopressor: '사용 중',
        drug: 'Norepinephrine 0.15 μg/kg/min',
      },
      respiration: {
        spo2: parseFloat(getLatestValue(vital[4].data).toFixed(0)),
        fio2: 60,
        peep: 8,
      },
      neurologic: {
        gcs: parseFloat(getLatestValue(neuro[0].data).toFixed(0)),
        trend: '↓ 감소',
      },
      infection: {
        lactate: parseFloat(getLatestValue(lab[2].data).toFixed(1)),
        wbc: parseFloat(getLatestValue(lab[0].data).toFixed(1)),
      },
    },
  };
};

const buildDemoData = () => {
  const patients: Patient[] = [
    createPatient(42, {
      id: 'pt-0142',
      wardId: 'ward-icu-1',
      wardName: WARDS[0],
      name: '김OO',
      gender: '남',
      age: 76,
      weight: 62,
      admitDate: '2026-01-14',
      room: '3-2',
      elapsedHours: 42,
      diagnosis: '패혈증 및 호흡부전',
      bases: {
        heartRate: 94,
        sysBP: 110,
        meanBP: 74,
        respRate: 24,
        spo2: 94,
        wbc: 15.2,
        creatinine: 1.8,
        lactate: 3.2,
        ph: 7.32,
        glucose: 168,
        gcsTotal: 9,
        gcsEye: 3,
        gcsVerbal: 2,
        gcsMotor: 4,
        pupil: 3,
      },
    }),
    createPatient(77, {
      id: 'pt-0177',
      wardId: 'ward-icu-2',
      wardName: WARDS[1],
      name: '박OO',
      gender: '여',
      age: 68,
      weight: 55,
      admitDate: '2026-01-13',
      room: '3-5',
      elapsedHours: 55,
      diagnosis: '폐렴 및 급성호흡곤란',
      bases: {
        heartRate: 88,
        sysBP: 116,
        meanBP: 82,
        respRate: 22,
        spo2: 93,
        wbc: 12.8,
        creatinine: 1.3,
        lactate: 2.4,
        ph: 7.36,
        glucose: 154,
        gcsTotal: 11,
        gcsEye: 4,
        gcsVerbal: 3,
        gcsMotor: 4,
        pupil: 2.8,
      },
    }),
    createPatient(93, {
      id: 'pt-0093',
      wardId: 'ward-icu-3',
      wardName: WARDS[2],
      name: '이OO',
      gender: '남',
      age: 59,
      weight: 70,
      admitDate: '2026-01-12',
      room: '4-1',
      elapsedHours: 67,
      diagnosis: '심부전 및 신기능 저하',
      bases: {
        heartRate: 104,
        sysBP: 104,
        meanBP: 68,
        respRate: 20,
        spo2: 96,
        wbc: 10.2,
        creatinine: 2.3,
        lactate: 2.8,
        ph: 7.34,
        glucose: 142,
        gcsTotal: 12,
        gcsEye: 4,
        gcsVerbal: 4,
        gcsMotor: 4,
        pupil: 3.2,
      },
    }),
    createPatient(120, {
      id: 'pt-0201',
      wardId: 'ward-icu-4',
      wardName: WARDS[3],
      name: '최OO',
      gender: '여',
      age: 73,
      weight: 58,
      admitDate: '2026-01-11',
      room: '4-3',
      elapsedHours: 81,
      diagnosis: '다발성 장기부전',
      bases: {
        heartRate: 98,
        sysBP: 92,
        meanBP: 64,
        respRate: 26,
        spo2: 91,
        wbc: 17.1,
        creatinine: 2.6,
        lactate: 4.1,
        ph: 7.28,
        glucose: 182,
        gcsTotal: 8,
        gcsEye: 2,
        gcsVerbal: 2,
        gcsMotor: 4,
        pupil: 3.6,
      },
    }),
  ];

  const wards: Ward[] = [
    {
      id: 'ward-icu-1',
      name: WARDS[0],
      patients: patients.filter((patient) => patient.wardId === 'ward-icu-1'),
    },
    {
      id: 'ward-icu-2',
      name: WARDS[1],
      patients: patients.filter((patient) => patient.wardId === 'ward-icu-2'),
    },
    {
      id: 'ward-icu-3',
      name: WARDS[2],
      patients: patients.filter((patient) => patient.wardId === 'ward-icu-3'),
    },
    {
      id: 'ward-icu-4',
      name: WARDS[3],
      patients: patients.filter((patient) => patient.wardId === 'ward-icu-4'),
    },
  ];

  return wards;
};

const groupPatientsByWard = (patients: Patient[]): Ward[] => {
  const grouped = new Map<string, Patient[]>();
  patients.forEach((patient) => {
    const list = grouped.get(patient.wardName) ?? [];
    list.push(patient);
    grouped.set(patient.wardName, list);
  });
  return Array.from(grouped.entries()).map(([wardName, wardPatients]) => ({
    id: wardName.toLowerCase().replace(/[^a-z0-9]+/g, '-'),
    name: wardName,
    patients: wardPatients,
  }));
};

const toPatientFromApi = (patient: ApiPatient): Patient => ({
  id: patient.stay_id,
  wardId: patient.ward ?? 'Unknown',
  wardName: patient.ward ?? 'Unknown',
  name: patient.name ?? '미상',
  gender: patient.gender ?? '-',
  age: patient.age ?? 0,
  weight: patient.weight ?? 0,
  admitDate: patient.admit_date ?? '-',
  room: patient.room ?? '-',
  elapsedHours: patient.elapsed_hours ?? 0,
  diagnosis: patient.diagnosis ?? '-',
  riskSummary: {
    currentHazard: patient.current_hazard ?? 0,
    recent6hAvg: 0,
    recent6hSlope: 0,
    cumRisk120hEst: patient.cum_risk ?? 0,
  },
  riskSeries: [],
  vitals: { vital: [], lab: [], neuro: [] },
  status: {
    circulation: { meanBP: 0, vasopressor: '-', drug: '-' },
    respiration: { spo2: 0, fio2: 0, peep: 0 },
    neurologic: { gcs: 0, trend: '-' },
    infection: { lactate: 0, wbc: 0 },
  },
});

const mergeRiskSeries = (
  prev: Array<{ t: number; hazard: number; cumRisk: number }>,
  next: Array<{ t: number; hazard: number; cumRisk: number }>
) => {
  const merged = new Map<number, { t: number; hazard: number; cumRisk: number }>();
  prev.forEach((point) => merged.set(point.t, point));
  next.forEach((point) => merged.set(point.t, point));
  return Array.from(merged.values()).sort((a, b) => a.t - b.t);
};

const KStepTooltip = ({ active, payload, isDarkMode }: any) => {
  if (!active || !payload || !payload.length) return null;
  const time = payload[0]?.payload?.t;
  const risk = payload[0]?.value;

  return (
    <div
      className={`rounded border px-3 py-2 text-xs shadow-sm ${
        isDarkMode
          ? 'bg-slate-900 border-slate-700 text-slate-100'
          : 'bg-white border-slate-200 text-slate-800'
      }`}
    >
      <div className="mb-1">t = {Math.round(time)}h</div>
      <div>현재 기준 24시간 위험도: {typeof risk === 'number' ? risk.toFixed(1) : '--'}%</div>
    </div>
  );
};

const MiniTooltip = ({ active, payload, isDarkMode, unit, label }: any) => {
  if (!active || !payload || !payload.length) return null;
  const time = payload[0]?.payload?.time;
  const value = payload[0]?.value;

  return (
    <div
      className={`rounded border px-2 py-1 text-[11px] shadow-sm ${
        isDarkMode
          ? 'bg-slate-900 border-slate-700 text-slate-100'
          : 'bg-white border-slate-200 text-slate-800'
      }`}
    >
      <div className="font-medium">{label}</div>
      <div>
        {time}시간: {value?.toFixed(1)}
        {unit}
      </div>
    </div>
  );
};

export default function MedicalStaffDashboard({
  isDarkMode,
  selectedPatientId,
  onSelectPatient,
}: MedicalStaffDashboardProps) {
  const [activeTab, setActiveTab] = useState<TabType>('vital');
  const demoWards = useMemo(() => buildDemoData(), []);
  const demoPatientMap = useMemo(() => {
    const map = new Map<string, Patient>();
    demoWards.flatMap((ward) => ward.patients).forEach((patient) => {
      map.set(patient.id, patient);
    });
    return map;
  }, [demoWards]);
  const demoPatients = useMemo(() => demoWards.flatMap((ward) => ward.patients), [demoWards]);
  const [apiPatients, setApiPatients] = useState<Patient[] | null>(null);
  const [isApiAvailable, setIsApiAvailable] = useState<boolean>(false);
  const [selectedWard, setSelectedWard] = useState<string>('ALL');
  const [kStepSummaries, setKStepSummaries] = useState<Record<string, KStepPatientSummary>>({});
  const [kStepTimeline, setKStepTimeline] = useState<KStepTimelineResponse | null>(null);
  const [isKStepAvailable, setIsKStepAvailable] = useState<boolean>(false);
  const kStepSimRef = useRef<Record<string, KStepSimState>>({});
  const patientScrollRef = useRef(false);
  const syntheticCacheRef = useRef<Map<string, { vitals: Patient['vitals']; status: Patient['status'] }>>(
    new Map()
  );
  const patientsForKStep =
    apiPatients && apiPatients.length > 0 ? apiPatients : demoPatients;

  useEffect(() => {
    let isActive = true;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);

    fetch('http://localhost:8000/patients', { signal: controller.signal })
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data: ApiPatient[]) => {
        if (!isActive) return;
        const mapped = data.map((raw) => {
          const base = toPatientFromApi(raw);
          const demo = demoPatientMap.get(base.id);
          if (demo) {
            return {
              ...base,
              vitals: demo.vitals,
              status: demo.status,
              riskSummary: {
                ...demo.riskSummary,
                currentHazard: base.riskSummary.currentHazard,
                cumRisk120hEst: base.riskSummary.cumRisk120hEst,
              },
            };
          }
          const cached = syntheticCacheRef.current.get(base.id);
          if (cached) {
            return {
              ...base,
              vitals: cached.vitals,
              status: cached.status,
            };
          }
          const seed = hashString(base.id);
          const synthetic = getSyntheticVitals(base, seed);
          syntheticCacheRef.current.set(base.id, synthetic);
          return {
            ...base,
            vitals: synthetic.vitals,
            status: synthetic.status,
          };
        });
        setApiPatients((prev) =>
          mapped.map((patient) => {
            const existing = prev?.find((item) => item.id === patient.id);
            return existing
              ? {
                  ...patient,
                  riskSeries: existing.riskSeries,
                  riskSummary: existing.riskSummary,
                }
              : patient;
          })
        );
        setIsApiAvailable(true);
        if (mapped.length > 0 && !mapped.some((patient) => patient.id === selectedPatientId)) {
          onSelectPatient(mapped[0].id);
        }
      })
      .catch(() => {
        if (!isActive) return;
        setIsApiAvailable(false);
      })
      .finally(() => clearTimeout(timeout));

    return () => {
      isActive = false;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    if (patientsForKStep.length === 0) return;
    const now = new Date();
    patientsForKStep.forEach((patient, index) => {
      if (kStepSimRef.current[patient.id]) return;
      const rng = createRng(hashString(patient.id));
      const isHighRisk = patient.id.endsWith('0999') || patient.id.endsWith('0201');
      const minSeverity = isHighRisk ? 0.55 : 0.08;
      const maxSeverity = isHighRisk ? 0.9 : 0.55;
      const severity = clamp(minSeverity + rng() * (maxSeverity - minSeverity), minSeverity, maxSeverity);
      const drift = isHighRisk ? 0.008 + rng() * 0.004 : (rng() - 0.5) * 0.01;
      kStepSimRef.current[patient.id] = {
        severity,
        drift,
        timestamp: new Date(now.getTime() - (index + 1) * 3600 * 1000),
        minSeverity,
        maxSeverity,
      };
    });
  }, [patientsForKStep]);

  useEffect(() => {
    if (patientsForKStep.length === 0) return;
    let isActive = true;
    const tick = async () => {
      try {
        await Promise.all(
          patientsForKStep.map(async (patient) => {
            const state = kStepSimRef.current[patient.id];
            if (!state) return;
            const nextSeverity = clamp(
              state.severity + state.drift + (Math.random() - 0.5) * 0.015,
              state.minSeverity,
              state.maxSeverity
            );
            const nextTimestamp = new Date(state.timestamp.getTime() + 3600 * 1000);
            kStepSimRef.current[patient.id] = {
              severity: nextSeverity,
              drift: state.drift,
              timestamp: nextTimestamp,
              minSeverity: state.minSeverity,
              maxSeverity: state.maxSeverity,
            };

            await inferKStepRisk({
              patient_id: patient.id,
              timestamp: nextTimestamp.toISOString(),
              features: buildKStepFeatures(nextSeverity),
            });
          })
        );
        const summaries = await fetchKStepPatients();
        if (!isActive) return;
        summaries.forEach((summary) => {
          const state = kStepSimRef.current[summary.patient_id];
          if (!state) return;
          const summaryTs = new Date(summary.last_timestamp);
          if (!Number.isNaN(summaryTs.getTime()) && summaryTs > state.timestamp) {
            kStepSimRef.current[summary.patient_id] = {
              ...state,
              timestamp: summaryTs,
            };
          }
        });
        const summaryMap = summaries.reduce<Record<string, KStepPatientSummary>>((acc, item) => {
          acc[item.patient_id] = item;
          return acc;
        }, {});
        setKStepSummaries(summaryMap);
        setIsKStepAvailable(true);
      } catch (error) {
        if (!isActive) return;
        setIsKStepAvailable(false);
      }
    };

    tick();
    const interval = setInterval(tick, 5000);
    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [patientsForKStep]);

  useEffect(() => {
    if (!selectedPatientId) return;
    let isActive = true;
    const refresh = async () => {
      try {
        const data = await fetchKStepTimeline(selectedPatientId);
        if (isActive) {
          setKStepTimeline(data);
        }
      } catch (error) {
        if (isActive) {
          setKStepTimeline(null);
        }
      }
    };

    refresh();
    const interval = setInterval(refresh, 5000);
    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [selectedPatientId]);

  useEffect(() => {
    if (!selectedPatientId) return;
    if (!patientScrollRef.current) {
      patientScrollRef.current = true;
      return;
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [selectedPatientId]);

  useEffect(() => {
    if (!isApiAvailable || !selectedPatientId) return;

    let isActive = true;
    const fetchTrajectory = () => {
      fetch(`http://localhost:8000/patients/${selectedPatientId}/risk-trajectory`)
        .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
        .then((data: ApiRiskResponse) => {
          if (!isActive) return;
          setApiPatients((prev) =>
            prev
              ? prev.map((patient) => {
                  if (patient.id !== data.stay_id) return patient;
                  const incomingRisk = data.trajectory.map((item) => ({
                    t: item.t,
                    hazard: item.hazard,
                    cumRisk: item.cum_risk,
                  }));
                  return {
                    ...patient,
                    elapsedHours: data.elapsed_hours ?? patient.elapsedHours,
                    riskSeries: mergeRiskSeries(patient.riskSeries, incomingRisk),
                    vitals: data.vitals ?? patient.vitals,
                    status: data.status ?? patient.status,
                  };
                })
              : prev
          );
        })
        .catch(() => undefined);
    };

    fetchTrajectory();
    const interval = setInterval(fetchTrajectory, 5000);
    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [isApiAvailable, selectedPatientId]);

  useEffect(() => {
    if (!isApiAvailable || !selectedPatientId) return;
    let isActive = true;
    const fetchSummary = () => {
      fetch(`http://localhost:8000/patients/${selectedPatientId}/risk-summary`)
        .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
        .then((data: ApiRiskSummary) => {
          if (!isActive) return;
          setApiPatients((prev) =>
            prev
              ? prev.map((patient) =>
                  patient.id === selectedPatientId
                    ? {
                        ...patient,
                        riskSummary: {
                          currentHazard: data.current_hazard,
                          recent6hAvg: data.recent_6h_avg,
                          recent6hSlope: data.recent_6h_slope,
                          cumRisk120hEst: data.cum_risk_120h_est,
                        },
                      }
                    : patient
                )
              : prev
          );
        })
        .catch(() => undefined);
    };

    fetchSummary();
    const interval = setInterval(fetchSummary, 5000);
    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [isApiAvailable, selectedPatientId]);

  const wards = isApiAvailable && apiPatients ? groupPatientsByWard(apiPatients) : demoWards;
  const filteredWards =
    selectedWard === 'ALL' ? wards : wards.filter((ward) => ward.name === selectedWard);
  const allPatients = filteredWards.flatMap((ward) => ward.patients);
  const selectedPatient =
    allPatients.find((patient) => patient.id === selectedPatientId) ?? allPatients[0] ?? null;
  const activeKStepSummary = isKStepAvailable ? kStepSummaries[selectedPatientId] : undefined;
  const currentRiskValue = selectedPatient
    ? activeKStepSummary?.risk_24h ?? selectedPatient.riskSummary.currentHazard
    : 0;
  const riskLabel = selectedPatient
    ? getRiskLabel(currentRiskValue, activeKStepSummary ? 'risk24h' : 'hazard')
    : null;
  const riskTimes = selectedPatient?.riskSeries.map((item) => item.t) ?? [];
  const currentTime = selectedPatient
    ? riskTimes.length > 0
      ? Math.round(Math.max(...riskTimes))
      : Math.round(Math.min(selectedPatient.elapsedHours, MAX_TIME))
    : 0;
  const kStepTimelinePoints = useMemo(() => {
    if (!kStepTimeline || kStepTimeline.patient_id !== selectedPatientId) {
      return [];
    }
    return [...kStepTimeline.timeline].sort((a, b) => a.t - b.t);
  }, [kStepTimeline, selectedPatientId]);
  const kStepCurrentTime =
    kStepTimelinePoints.length > 0
      ? kStepTimelinePoints[kStepTimelinePoints.length - 1].t
      : null;
  const isHighRisk = selectedPatient
    ? activeKStepSummary?.alert_24h ??
      (selectedPatient.riskSummary.currentHazard > 0.08 ||
        selectedPatient.riskSummary.recent6hSlope > 0.01)
    : false;
  const stayDays = selectedPatient
    ? Math.max(1, Math.floor(((kStepCurrentTime ?? currentTime) || 0) / 24) + 1)
    : 0;
  const kStepChartData = useMemo(
    () =>
      kStepTimelinePoints.map((point) => ({
        t: point.t,
        risk24: point.risk_24h * 100,
      })),
    [kStepTimelinePoints]
  );
  const kStepAlertLog =
    kStepTimeline && kStepTimeline.patient_id === selectedPatientId ? kStepTimeline.alerts : [];
  const sortedAlertLog = useMemo(
    () => [...kStepAlertLog].sort((a, b) => a.t - b.t),
    [kStepAlertLog]
  );
  const kStepStartTime = kStepChartData[0]?.t ?? 0;
  const kStepEndTime = kStepChartData[kStepChartData.length - 1]?.t ?? 0;
  const alertRuleText = activeKStepSummary
    ? 'Rule: risk_24h >= 0.20 + (2 consecutive or 6h rise)'
    : 'Rule: current hazard > 0.08 or 6h slope > 0.01';
  const safeRuleText = activeKStepSummary
    ? 'Rule: risk_24h < 0.20 or no sustained rise'
    : 'Rule: current hazard <= 0.08 and 6h slope <= 0.01';

  useEffect(() => {
    if (allPatients.length === 0) return;
    if (!allPatients.find((patient) => patient.id === selectedPatientId)) {
      onSelectPatient(allPatients[0].id);
    }
  }, [allPatients, onSelectPatient, selectedPatientId]);

  const bgColor = isDarkMode ? 'bg-slate-950' : 'bg-slate-50';
  const cardBg = isDarkMode ? 'bg-slate-900' : 'bg-white';
  const borderColor = isDarkMode ? 'border-slate-800' : 'border-slate-200';
  const textPrimary = isDarkMode ? 'text-slate-100' : 'text-slate-900';
  const textSecondary = isDarkMode ? 'text-slate-300' : 'text-slate-600';
  const textMuted = isDarkMode ? 'text-slate-400' : 'text-slate-500';
  const gridColor = isDarkMode ? '#1E293B' : '#E2E8F0';
  const axisColor = isDarkMode ? '#94A3B8' : '#64748B';
  const lineColor = isDarkMode ? '#F8FAFC' : '#0F172A';

  return (
    <div className={`min-h-screen ${bgColor} ${textPrimary}`}>
      <div className="flex">
        {/* Ward + Patient Sidebar */}
        <aside className={`w-72 border-r ${borderColor} ${cardBg} p-4`}> 
          <div className="flex items-center gap-3 mb-4">
            <div className={`text-xs ${textMuted}`}>병동</div>
            <div className="relative w-full">
              <select
                value={selectedWard}
                onChange={(event) => setSelectedWard(event.target.value)}
                className={`w-full appearance-none rounded-xl border px-4 py-2 text-sm transition focus:outline-none ${
                  isDarkMode
                    ? 'border-slate-700 bg-slate-900 text-slate-100 focus:border-slate-500'
                    : 'border-slate-300 bg-white text-slate-900 focus:border-slate-500'
                }`}
              >
                <option value="ALL">전체</option>
                {WARDS.map((wardName) => (
                  <option key={wardName} value={wardName}>
                    {getWardLabel(wardName)}
                  </option>
                ))}
              </select>
              <span className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400">
                ▼
              </span>
            </div>
          </div>
          <div className={`text-xs ${textMuted} mb-3`}>병동별 환자 대시보드</div>
          <div className="space-y-4">
            {filteredWards.map((ward) => (
              <div key={ward.id}>
                <div className={`text-sm font-semibold ${textPrimary} mb-2`}>
                  {getWardLabel(ward.name)}
                  <span className={`ml-2 text-xs ${textMuted}`}>{ward.patients.length}명</span>
                </div>
                <div className="space-y-2">
                  {ward.patients.map((patient) => {
                    const kStepSummary = kStepSummaries[patient.id];
                    const riskValue = kStepSummary?.risk_24h ?? patient.riskSummary.currentHazard;
                    const badge = getRiskLabel(riskValue, kStepSummary ? 'risk24h' : 'hazard');
                    const isActive = patient.id === selectedPatientId;
                    return (
                      <button
                        key={patient.id}
                        onClick={() => onSelectPatient(patient.id)}
                        className={`w-full rounded border px-3 py-2 text-left transition ${
                          isActive
                            ? isDarkMode
                              ? 'border-amber-400 bg-slate-800'
                              : 'border-amber-400 bg-amber-50'
                            : `${borderColor} hover:border-amber-300`
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div>
                            <div className="text-sm font-medium">{patient.name}</div>
                            <div className={`text-xs ${textMuted}`}>
                              {getWardLabel(patient.wardName)} · {patient.gender} · {patient.age}세
                            </div>
                          </div>
                          <span className={`text-[11px] px-2 py-0.5 rounded ${badge.color}`}>
                            {badge.label}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
            {filteredWards.length === 0 && (
              <div className={`text-xs ${textMuted}`}>선택된 병동이 없습니다.</div>
            )}
          </div>
        </aside>

        {/* Main Content */}
        <div className="flex-1 p-6">
          {!selectedPatient && (
            <div className={`${cardBg} border ${borderColor} p-6 shadow-sm`}>
              <div className={`text-sm ${textPrimary}`}>선택된 병동에 환자가 없습니다.</div>
              <div className={`text-xs ${textMuted} mt-2`}>병동 필터를 조정해 주세요.</div>
            </div>
          )}
          {selectedPatient && (
            <>
          {/* Patient Header */}
          <div className={`${cardBg} border ${borderColor} px-6 py-5 shadow-sm mb-6`}>
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div>
                <div className={`text-xs ${textMuted} mb-2`}>환자별 상세 모니터링</div>
                <div className="text-2xl font-semibold">
                  {selectedPatient.name} 환자
                </div>
                <div className={`text-sm ${textSecondary}`}>
                  주요 진단: {selectedPatient.diagnosis}
                </div>
              </div>
              <div className="flex items-center gap-4">
                {riskLabel && (
                  <span className={`text-sm px-3 py-1 rounded ${riskLabel.color}`}>
                    {riskLabel.label} · {(currentRiskValue * 100).toFixed(1)}%
                  </span>
                )}
                <span className={`text-xs ${textMuted}`}>
                  최근 업데이트 {(kStepCurrentTime ?? currentTime) || 0}시간
                </span>
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-4">
              <div>
                <div className={`text-xs ${textMuted}`}>환자번호</div>
                <div className={`text-sm ${textPrimary}`}>{selectedPatient.id.toUpperCase()}</div>
              </div>
              <div>
                <div className={`text-xs ${textMuted}`}>이름</div>
                <div className={`text-sm ${textPrimary}`}>{selectedPatient.name}</div>
              </div>
              <div>
                <div className={`text-xs ${textMuted}`}>성별</div>
                <div className={`text-sm ${textPrimary}`}>{selectedPatient.gender}</div>
              </div>
              <div>
                <div className={`text-xs ${textMuted}`}>나이</div>
                <div className={`text-sm ${textPrimary}`}>{selectedPatient.age}세</div>
              </div>
              <div>
                <div className={`text-xs ${textMuted}`}>체중</div>
                <div className={`text-sm ${textPrimary}`}>{selectedPatient.weight}kg</div>
              </div>
              <div>
                <div className={`text-xs ${textMuted}`}>입원일</div>
                <div className={`text-sm ${textPrimary}`}>{selectedPatient.admitDate}</div>
              </div>
              <div>
                <div className={`text-xs ${textMuted}`}>입원실</div>
                <div className={`text-sm ${textPrimary}`}>{selectedPatient.wardName}</div>
              </div>
              <div>
                <div className={`text-xs ${textMuted}`}>입원경과</div>
                <div className={`text-sm ${textPrimary}`}>{stayDays}일차</div>
              </div>
            </div>
            <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className={`${borderColor} border rounded-lg px-4 py-3`}>
                <div className={`text-xs ${textMuted}`}>Next 6h Risk</div>
                <div className="text-lg font-semibold">{formatPercent(activeKStepSummary?.risk_6h)}</div>
              </div>
              <div className={`${borderColor} border rounded-lg px-4 py-3`}>
                <div className={`text-xs ${textMuted}`}>Next 24h Risk</div>
                <div className="text-lg font-semibold">{formatPercent(activeKStepSummary?.risk_24h)}</div>
              </div>
              <div className={`${borderColor} border rounded-lg px-4 py-3`}>
                <div className={`text-xs ${textMuted}`}>Next 72h Risk</div>
                <div className="text-lg font-semibold">{formatPercent(activeKStepSummary?.risk_72h)}</div>
              </div>
            </div>
            <div className="mt-4">
              {isHighRisk ? (
                <div className="flex items-center justify-between rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-700">
                  <span className="font-semibold">High Risk Detected</span>
                  <span className="text-xs">{alertRuleText}</span>
                </div>
              ) : (
                <div className="flex items-center justify-between rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                  <span className="font-semibold">No Critical Alert</span>
                  <span className="text-xs">{safeRuleText}</span>
                </div>
              )}
            </div>
          </div>

          {/* Risk Trend */}
          <div className={`${cardBg} border ${borderColor} p-6 shadow-sm mb-6`}>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <div className={`text-sm font-medium ${textPrimary}`}>24시간 내 위험도 추이</div>
                <div className={`text-xs ${textMuted}`}>
                  각 시점 t에서 “향후 24시간 내 사건 발생 확률”을 표시합니다
                </div>
              </div>
              <div className={`text-xs ${textMuted}`}>
                시간축: 입실 후 경과시간 (t={kStepStartTime}~{kStepEndTime}시간)
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-4 text-xs">
              <div className={`flex items-center gap-2 ${textMuted}`}>
                <span
                  className={`h-0.5 w-6 rounded-full ${isDarkMode ? 'bg-sky-300' : 'bg-blue-600'}`}
                />
                <span>24시간 위험도</span>
              </div>
              <div className={`flex items-center gap-2 ${textMuted}`}>
                <span className="h-0.5 w-6 rounded-full border-t-2 border-dashed border-orange-500" />
                <span>알림 기준 20%</span>
              </div>
              <div className={`flex items-center gap-2 ${textMuted}`}>
                <span className="h-0.5 w-6 rounded-full border-t-2 border-dashed border-emerald-500" />
                <span>현재 시점</span>
              </div>
              <div className={`text-xs ${textMuted}`}>미래 trajectory 미표시</div>
            </div>
            <div className="mt-4 h-[260px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={kStepChartData}>
                  <CartesianGrid strokeDasharray="4 4" stroke={gridColor} />
                  <XAxis
                    dataKey="t"
                    type="number"
                    domain={['dataMin', 'dataMax']}
                    tick={{ fill: axisColor, fontSize: 11 }}
                    axisLine={{ stroke: axisColor }}
                    tickLine={false}
                    tickFormatter={(value) => `${Math.round(value)}시간`}
                  />
                  <YAxis
                    tick={{ fill: axisColor, fontSize: 11 }}
                    axisLine={{ stroke: axisColor }}
                    tickLine={false}
                    domain={[0, 100]}
                    tickFormatter={(value) => `${Math.round(value)}%`}
                    label={{
                      value: '24h 위험도(%)',
                      angle: -90,
                      position: 'insideLeft',
                      fill: axisColor,
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    content={<KStepTooltip isDarkMode={isDarkMode} />}
                    cursor={{ stroke: axisColor, strokeDasharray: '3 3' }}
                  />
                  <ReferenceLine
                    y={20}
                    stroke={isDarkMode ? '#F97316' : '#EA580C'}
                    strokeDasharray="4 4"
                    label={{
                      value: '알림 기준 20%',
                      fill: axisColor,
                      fontSize: 10,
                      position: 'insideTopRight',
                    }}
                  />
                  <Line
                    type="linear"
                    dataKey="risk24"
                    stroke={isDarkMode ? '#93C5FD' : '#1D4ED8'}
                    strokeWidth={2.2}
                    dot={false}
                    connectNulls={false}
                    isAnimationActive={false}
                  />
                  {typeof kStepCurrentTime === 'number' && (
                    <ReferenceLine
                      x={kStepCurrentTime}
                      stroke="#10B981"
                      strokeDasharray="4 4"
                      strokeWidth={1.5}
                      label={{
                        value: `현재 ${kStepCurrentTime}h`,
                        fill: '#10B981',
                        fontSize: 10,
                        position: 'insideTop',
                      }}
                    />
                  )}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Alert Log */}
          <div className={`${cardBg} border ${borderColor} p-6 shadow-sm mb-6`}>
            <div className="flex items-center justify-between">
              <div className={`text-sm font-medium ${textPrimary}`}>Alert Log</div>
              <div className={`text-xs ${textMuted}`}>최근 알림 기준</div>
            </div>
            <div className="mt-4 space-y-2 text-sm">
              {kStepAlertLog.length === 0 && (
                <div className={`text-xs ${textMuted}`}>알림 기록이 없습니다.</div>
              )}
              {sortedAlertLog
                .slice(-5)
                .reverse()
                .map((alert) => (
                  <div
                    key={`${alert.timestamp}-${alert.t}`}
                    className="flex items-center justify-between gap-3"
                  >
                    <span className={`text-xs ${textSecondary}`}>t={alert.t}h</span>
                    <span className={`text-xs ${textMuted}`}>
                      {new Date(alert.timestamp).toLocaleString('ko-KR', { hour12: false })}
                    </span>
                    <span className="text-xs font-semibold text-red-600">
                      {alert.reason.join(', ')}
                    </span>
                  </div>
                ))}
            </div>
          </div>

          {/* Patient Status Cards */}
          <div className="grid grid-cols-1 gap-4 mb-6 lg:grid-cols-2 xl:grid-cols-4">
            <div className={`${cardBg} border ${borderColor} p-4 shadow-sm`}>
              <div className={`text-xs ${textMuted} mb-3`}>순환</div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className={`text-xs ${textSecondary}`}>평균 혈압</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-red-600 text-white">낮음</span>
                </div>
                <div className="text-2xl text-red-500">{selectedPatient.status.circulation.meanBP} mmHg</div>
                <div className={`flex justify-between items-center pt-2 border-t ${borderColor}`}>
                  <span className={`text-xs ${textSecondary}`}>승압제</span>
                  <span className={`text-sm ${textPrimary}`}>{selectedPatient.status.circulation.vasopressor}</span>
                </div>
                <div className={`text-xs ${textMuted}`}>{selectedPatient.status.circulation.drug}</div>
              </div>
            </div>

            <div className={`${cardBg} border ${borderColor} p-4 shadow-sm`}>
              <div className={`text-xs ${textMuted} mb-3`}>호흡</div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className={`text-xs ${textSecondary}`}>SpO2</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-amber-500 text-black">주의</span>
                </div>
                <div className="text-2xl text-amber-500">{selectedPatient.status.respiration.spo2}%</div>
                <div className={`pt-2 border-t ${borderColor} space-y-1`}>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>FiO2</span>
                    <span className={textPrimary}>{selectedPatient.status.respiration.fio2}%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className={textSecondary}>PEEP</span>
                    <span className={textPrimary}>{selectedPatient.status.respiration.peep} cmH2O</span>
                  </div>
                </div>
              </div>
            </div>

            <div className={`${cardBg} border ${borderColor} p-4 shadow-sm`}>
              <div className={`text-xs ${textMuted} mb-3`}>신경학</div>
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className={`text-xs ${textSecondary}`}>GCS 총점</span>
                  <span className="text-xs px-2 py-0.5 rounded bg-red-600 text-white">위험</span>
                </div>
                <div className="text-2xl text-red-500">{selectedPatient.status.neurologic.gcs}점</div>
                <div className={`text-xs ${textMuted} flex items-center gap-1`}>
                  <span>추이:</span>
                  <span className="text-red-500">{selectedPatient.status.neurologic.trend}</span>
                </div>
              </div>
            </div>

            <div className={`${cardBg} border ${borderColor} p-4 shadow-sm`}>
              <div className={`text-xs ${textMuted} mb-3`}>감염/대사</div>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className={`text-xs ${textSecondary}`}>젖산</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-amber-500 text-black">높음</span>
                  </div>
                  <div className="text-lg text-amber-500">{selectedPatient.status.infection.lactate} mmol/L</div>
                </div>
                <div className={`pt-2 border-t ${borderColor}`}>
                  <div className="flex justify-between items-center mb-1">
                    <span className={`text-xs ${textSecondary}`}>백혈구</span>
                    <span className="text-xs px-2 py-0.5 rounded bg-amber-500 text-black">높음</span>
                  </div>
                  <div className="text-lg text-amber-500">{selectedPatient.status.infection.wbc} K/μL</div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom: Mini Trend Cards with Tabs */}
          <div className={`${cardBg} border ${borderColor} shadow-sm`}>
            <div className={`flex border-b ${borderColor}`}>
              {(['vital', 'lab', 'neuro'] as TabType[]).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-6 py-3 text-sm transition-colors ${
                    activeTab === tab
                      ? isDarkMode
                        ? 'bg-slate-800 text-white border-b-2 border-white'
                        : 'bg-slate-100 text-slate-900 border-b-2 border-slate-900'
                      : isDarkMode
                      ? 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                      : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
                  }`}
                >
                  {tab === 'vital' ? '바이탈' : tab === 'lab' ? '검사' : '신경계'}
                </button>
              ))}
            </div>

            <div className="p-6">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3 xl:grid-cols-5">
                {selectedPatient.vitals[activeTab].map((vital, index) => {
                  const isNormal = vital.current >= vital.normal[0] && vital.current <= vital.normal[1];
                  const isDanger = vital.current < vital.normal[0] * 0.8 || vital.current > vital.normal[1] * 1.2;
                  const color = isNormal ? '#10B981' : isDanger ? '#EF4444' : '#F59E0B';

                  return (
                    <div
                      key={`${vital.name}-${index}`}
                      className={`${cardBg} border ${borderColor} p-4 transition-all duration-200 hover:shadow-md`}
                    >
                      <div className={`text-xs ${textMuted} mb-2`}>{vital.name}</div>
                      <div className="flex items-baseline justify-between mb-2">
                        <div className="text-xl" style={{ color }}>
                          {vital.current}
                        </div>
                        <div className={`text-xs ${textMuted}`}>{vital.unit}</div>
                      </div>
                      <div className={`text-xs ${textMuted} mb-2`}>
                        정상범위: {vital.normal[0]}-{vital.normal[1]}
                      </div>
                      <ResponsiveContainer width="100%" height={60}>
                        <LineChart data={vital.data}>
                          <Line type="monotone" dataKey="value" stroke={color} strokeWidth={1.6} dot={false} />
                          <Tooltip
                            content={<MiniTooltip isDarkMode={isDarkMode} unit={vital.unit} label={vital.name} />}
                            cursor={{ stroke: axisColor, strokeDasharray: '3 3' }}
                          />
                          <XAxis dataKey="time" hide />
                          <YAxis hide domain={['dataMin - 5', 'dataMax + 5']} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
