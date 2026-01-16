import { PatientSummary } from '@/app/api';

export interface PatientListItem {
  id: string;
  name: string;
  ward: string;
  summary?: PatientSummary;
}

interface PatientListProps {
  patients: PatientListItem[];
  selectedId: string;
  onSelect: (patientId: string) => void;
}

const formatPercent = (value?: number) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${(value * 100).toFixed(1)}%`;
};

const getRiskTone = (value?: number) => {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return 'text-slate-500';
  }
  if (value >= 0.2) return 'text-red-600';
  if (value >= 0.1) return 'text-amber-600';
  return 'text-emerald-600';
};

export default function PatientList({ patients, selectedId, onSelect }: PatientListProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 px-4 py-3">
        <p className="text-sm font-semibold text-slate-700">Patient List</p>
        <p className="text-xs text-slate-500">K-step risk summary</p>
      </div>
      <div className="divide-y divide-slate-100">
        {patients.map((patient) => {
          const risk = patient.summary?.risk_24h;
          return (
            <button
              key={patient.id}
              onClick={() => onSelect(patient.id)}
              className={`flex w-full items-center justify-between px-4 py-3 text-left transition ${
                selectedId === patient.id
                  ? 'bg-slate-900 text-white'
                  : 'bg-white text-slate-700 hover:bg-slate-50'
              }`}
            >
              <div>
                <p className="text-sm font-semibold">{patient.name}</p>
                <p
                  className={`text-xs ${
                    selectedId === patient.id ? 'text-slate-300' : 'text-slate-500'
                  }`}
                >
                  {patient.ward}
                </p>
              </div>
              <div className="text-right">
                <p
                  className={`text-sm font-semibold ${
                    selectedId === patient.id ? 'text-white' : getRiskTone(risk)
                  }`}
                >
                  {formatPercent(risk)}
                </p>
                <p
                  className={`text-[11px] ${
                    selectedId === patient.id
                      ? 'text-slate-300'
                      : patient.summary?.alert_24h
                      ? 'text-red-600'
                      : 'text-slate-400'
                  }`}
                >
                  {patient.summary?.alert_24h ? 'Alert detected' : 'Monitoring'}
                </p>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
