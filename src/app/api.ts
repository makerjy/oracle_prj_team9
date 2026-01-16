export const API_BASE = 'http://localhost:8000/api';

export interface InferPayload {
  patient_id: string;
  timestamp: string;
  features: Record<string, number>;
}

export interface InferResponse {
  patient_id: string;
  timestamp: string;
  hazard_seq: number[];
  risk_6h: number;
  risk_24h: number;
  risk_72h: number;
  alert_24h: boolean;
  alert_reason: string[];
}

export interface PatientSummary {
  patient_id: string;
  last_timestamp: string;
  risk_6h: number;
  risk_24h: number;
  risk_72h: number;
  alert_24h: boolean;
  alert_reason: string[];
}

export interface TimelinePoint {
  t: number;
  timestamp: string;
  risk_6h: number;
  risk_24h: number;
  risk_72h: number;
  alert_24h: boolean;
}

export interface TimelineResponse {
  patient_id: string;
  timeline: TimelinePoint[];
  alerts: Array<{ t: number; timestamp: string; reason: string[] }>;
}

async function requestJson<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, options);
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return (await res.json()) as T;
}

export function inferRisk(payload: InferPayload): Promise<InferResponse> {
  return requestJson<InferResponse>(`${API_BASE}/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export function fetchPatients(): Promise<PatientSummary[]> {
  return requestJson<PatientSummary[]>(`${API_BASE}/patients`);
}

export function fetchTimeline(patientId: string): Promise<TimelineResponse> {
  return requestJson<TimelineResponse>(`${API_BASE}/patients/${patientId}/timeline`);
}
