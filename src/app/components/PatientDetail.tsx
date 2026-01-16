import { useMemo } from 'react';
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

import { PatientSummary, TimelinePoint } from '@/app/api';
import { PatientListItem } from '@/app/components/PatientList';

interface PatientDetailProps {
  patient: PatientListItem | null;
  summary?: PatientSummary;
  timeline: TimelinePoint[];
  alerts: Array<{ t: number; timestamp: string; reason: string[] }>;
}

const formatPercent = (value?: number) => {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '--';
  }
  return `${(value * 100).toFixed(1)}%`;
};

const formatTimestamp = (value?: string) => {
  if (!value) return '-';
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? '-'
    : date.toLocaleString('en-US', { hour12: false });
};

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-700 shadow-md">
      <p className="font-semibold">t = {point.t}h</p>
      <p>
        Risk 24h: {typeof point.risk_24h === 'number' ? point.risk_24h.toFixed(1) : '--'}%
      </p>
    </div>
  );
};

export default function PatientDetail({ patient, summary, timeline, alerts }: PatientDetailProps) {
  const chartData = useMemo(() => {
    const recent = timeline.slice(-72);
    return recent.map((point) => ({
      t: point.t,
      risk_24h: point.risk_24h * 100,
    }));
  }, [timeline]);

  if (!patient) {
  return (
    <div className="rounded-2xl border border-dashed border-slate-200 bg-white p-8 text-center text-slate-500">
      Select a patient to view the risk trajectory.
    </div>
  );
  }

  return (
    <div className="space-y-6">
      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs text-slate-500">Patient Monitoring</p>
            <h2 className="text-2xl font-semibold text-slate-900">{patient.name}</h2>
            <p className="text-sm text-slate-500">{patient.id}</p>
            <p className="mt-2 text-xs text-slate-500">{patient.ward}</p>
          </div>
          <div className="text-right">
            <p className="text-xs text-slate-500">Last update</p>
            <p className="text-sm font-semibold text-slate-800">
              {formatTimestamp(summary?.last_timestamp)}
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {[
          { label: 'Next 6h Risk', value: summary?.risk_6h },
          { label: 'Next 24h Risk', value: summary?.risk_24h },
          { label: 'Next 72h Risk', value: summary?.risk_72h },
        ].map((card) => (
          <div key={card.label} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <p className="text-xs text-slate-500">{card.label}</p>
            <p className="mt-3 text-2xl font-semibold text-slate-900">
              {formatPercent(card.value)}
            </p>
          </div>
        ))}
      </div>

      <div
        className={`rounded-2xl border px-5 py-4 text-sm font-semibold ${
          summary?.alert_24h
            ? 'border-red-200 bg-red-50 text-red-700'
            : 'border-emerald-200 bg-emerald-50 text-emerald-700'
        }`}
      >
        {summary?.alert_24h ? 'High Risk Detected' : 'No Critical Alert'}
        <span className="ml-2 text-xs font-normal text-slate-500">
          Rule: risk_24h &gt;= 0.20 + (2 consecutive or 6h rise)
        </span>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs text-slate-500">Risk 24h Trend</p>
            <p className="text-lg font-semibold text-slate-900">Last 72 hours</p>
          </div>
          <div className="text-xs text-slate-500">No future trajectory</div>
        </div>
        <div className="mt-6 h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis dataKey="t" tick={{ fontSize: 11 }} />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 11 }}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={20} stroke="#F97316" strokeDasharray="4 4" />
              <Line
                type="linear"
                dataKey="risk_24h"
                stroke="#0F172A"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-slate-800">Alert log</p>
          <p className="text-xs text-slate-400">Recent alerts</p>
        </div>
        <div className="mt-4 space-y-2 text-sm text-slate-600">
          {alerts.length === 0 && <p className="text-xs text-slate-400">No alerts yet.</p>}
          {alerts.slice(-5).reverse().map((alert) => (
            <div key={`${alert.timestamp}-${alert.t}`} className="flex items-center justify-between">
              <span className="text-xs text-slate-500">t={alert.t}h</span>
              <span className="text-xs text-slate-400">{formatTimestamp(alert.timestamp)}</span>
              <span className="text-xs font-semibold text-red-600">
                {alert.reason.join(', ')}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
