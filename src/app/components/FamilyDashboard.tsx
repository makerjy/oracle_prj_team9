import { useEffect, useRef, useState } from 'react';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';

// 숫자 없는 트렌드 데이터 생성
const generateTrendData = (pattern: 'improving' | 'stable' | 'declining') => {
  const data = [];
  for (let i = 0; i <= 20; i++) {
    let value = 50;
    if (pattern === 'improving') {
      value = 30 + i * 2 + Math.random() * 5;
    } else if (pattern === 'declining') {
      value = 70 - i * 1.5 + Math.random() * 5;
    } else {
      value = 50 + Math.sin(i / 3) * 5 + Math.random() * 3;
    }
    data.push({ index: i, value });
  }
  return data;
};

const timelineEvents = [
  { 
    time: '0h', 
    label: '중환자실 입실', 
    description: '응급실에서 중환자실로 이송되었습니다',
    icon: '🏥',
    type: 'info'
  },
  { 
    time: '2h', 
    label: '호흡 보조 시작', 
    description: '호흡을 돕기 위한 장치를 사용하기 시작했습니다',
    icon: '🫁',
    type: 'info'
  },
  { 
    time: '6h', 
    label: '항생제 치료', 
    description: '감염 치료를 위한 약물 투여를 시작했습니다',
    icon: '💊',
    type: 'info'
  },
  { 
    time: '18h', 
    label: '혈압 저하 발생', 
    description: '일시적으로 혈압이 낮아졌으나 약물로 관리 중입니다',
    icon: '⚠️',
    type: 'warning'
  },
  { 
    time: '28h', 
    label: '혈압 약 증량', 
    description: '혈압 유지를 위해 약물을 조정했습니다',
    icon: '💉',
    type: 'warning'
  },
  { 
    time: '42h', 
    label: '현재', 
    description: '현재 상태를 면밀히 관찰하고 있습니다',
    icon: '⏰',
    type: 'current'
  },
];

interface FamilyDashboardProps {
  selectedPatientId: string;
  onSelectPatient: (patientId: string) => void;
}

export default function FamilyDashboard({
  selectedPatientId,
  onSelectPatient,
}: FamilyDashboardProps) {
  type ApiPatient = {
    stay_id: string;
    name?: string;
    ward?: string;
    guardian_name?: string;
    gender?: string;
    age?: number;
    diagnosis?: string;
    current_hazard?: number;
  };

  type ApiRiskSummary = {
    current_hazard: number;
    recent_6h_avg: number;
    recent_6h_slope: number;
    cum_risk_120h_est: number;
  };

  type ApiRiskTrajectory = {
    elapsed_hours?: number;
  };

  const [patientList, setPatientList] = useState<ApiPatient[]>([]);
  const [demoStatus, setDemoStatus] = useState<Record<string, number>>({});
  const [demoDays, setDemoDays] = useState<Record<string, number>>({});
  const [summary, setSummary] = useState<ApiRiskSummary | null>(null);
  const [elapsedHours, setElapsedHours] = useState(0);
  const [hoveredEvent, setHoveredEvent] = useState<number | null>(null);
  const [hoveredTrend, setHoveredTrend] = useState<string | null>(null);
  const patientScrollRef = useRef(false);

  useEffect(() => {
    let isActive = true;
    const fetchList = () => {
      fetch('http://localhost:8000/patients')
        .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
        .then((data: ApiPatient[]) => {
          if (!isActive) return;
          setPatientList(data);
          if (data.length > 0 && !data.find((item) => item.stay_id === selectedPatientId)) {
            onSelectPatient(data[0].stay_id);
          }
        })
        .catch(() => undefined);
    };

    fetchList();
    const interval = setInterval(fetchList, 5000);
    return () => {
      isActive = false;
      clearInterval(interval);
    };
  }, [onSelectPatient, selectedPatientId]);

  useEffect(() => {
    if (patientList.length === 0) return;
    setDemoDays((prev) => {
      const next = { ...prev };
      patientList.forEach((patient, index) => {
        if (!next[patient.stay_id]) {
          next[patient.stay_id] = 1 + (index % 4);
        }
      });
      return next;
    });
    setDemoStatus((prev) => {
      const next = { ...prev };
      patientList.forEach((patient, index) => {
        if (next[patient.stay_id] === undefined) {
          next[patient.stay_id] = (index % 3) * 0.02;
        }
      });
      return next;
    });
  }, [patientList]);

  useEffect(() => {
    if (patientList.length === 0) return;
    const interval = setInterval(() => {
      setDemoStatus((prev) => {
        const next = { ...prev };
        patientList.forEach((patient) => {
          const current = next[patient.stay_id] ?? 0.01;
          const delta = (Math.random() - 0.4) * 0.01;
          next[patient.stay_id] = Math.max(0, Math.min(0.12, current + delta));
        });
        return next;
      });
      setDemoDays((prev) => {
        const next = { ...prev };
        patientList.forEach((patient) => {
          const current = next[patient.stay_id] ?? 1;
          next[patient.stay_id] = Math.min(14, current + (Math.random() > 0.7 ? 1 : 0));
        });
        return next;
      });
    }, 5000);
    return () => clearInterval(interval);
  }, [patientList]);

  useEffect(() => {
    if (!selectedPatientId) return;
    let isActive = true;

    const fetchSummary = () => {
      fetch(`http://localhost:8000/patients/${selectedPatientId}/risk-summary`)
        .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
        .then((data: ApiRiskSummary) => {
          if (!isActive) return;
          setSummary(data);
        })
        .catch(() => undefined);
    };

    const fetchTrajectory = () => {
      fetch(`http://localhost:8000/patients/${selectedPatientId}/risk-trajectory`)
        .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
        .then((data: ApiRiskTrajectory) => {
          if (!isActive) return;
          setElapsedHours(Math.round(data.elapsed_hours ?? 0));
        })
        .catch(() => undefined);
    };

    fetchSummary();
    fetchTrajectory();
    const interval = setInterval(() => {
      fetchSummary();
      fetchTrajectory();
    }, 5000);
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

  const selectedPatient =
    patientList.find((patient) => patient.stay_id === selectedPatientId) ?? patientList[0];
  const days = demoDays[selectedPatient?.stay_id ?? ''] ?? 1;

  const slope = summary?.recent_6h_slope ?? demoStatus[selectedPatient?.stay_id ?? ''] ?? 0;
  const pattern = slope > 0.01 ? 'declining' : slope < -0.01 ? 'improving' : 'stable';
  const consciousnessTrend = generateTrendData(pattern);
  const respirationTrend = generateTrendData(pattern);
  const overallTrend = generateTrendData(pattern);

  const currentHazard =
    summary?.current_hazard ?? demoStatus[selectedPatient?.stay_id ?? ''] ?? 0;
  const statusTone =
    currentHazard > 0.08
      ? 'warning'
      : currentHazard > 0.04
      ? 'info'
      : 'neutral';
  const statusText =
    statusTone === 'warning'
      ? '현재 집중 관찰이 필요한 상태입니다'
      : statusTone === 'info'
      ? '상태 변화를 면밀히 관찰 중입니다'
      : '상태가 안정적으로 유지되고 있습니다';
  const statusStyles =
    statusTone === 'warning'
      ? 'bg-orange-100 border-2 border-orange-400 text-orange-900'
      : statusTone === 'info'
      ? 'bg-blue-50 border-2 border-blue-300 text-blue-900'
      : 'bg-gray-100 border-2 border-gray-300 text-gray-800';

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header Card */}
        <div className="bg-white border border-gray-300 p-8 mb-6">
          <div className="flex items-start justify-between">
            <div>
              <div className="text-3xl text-black mb-3">
                환자성함: {selectedPatient?.name ?? '미상'} 님
              </div>
              <div className="text-lg text-gray-600">
                입원 {days}일차
              </div>
            </div>
            <div className="flex items-start gap-4">
              <div className="text-right">
                <div className="text-xs text-gray-500 mb-2">환자 선택</div>
                <select
                  value={selectedPatientId}
                  onChange={(event) => onSelectPatient(event.target.value)}
                  className="border border-gray-300 px-3 py-2 text-sm bg-white"
                >
                  {patientList.map((patient) => (
                    <option key={patient.stay_id} value={patient.stay_id}>
                      {patient.name ?? '미상'} · {patient.ward ?? '-'}
                    </option>
                  ))}
                </select>
              </div>
              <div className={`${statusStyles} px-6 py-3`}>
                <div className="text-sm">{statusText}</div>
              </div>
            </div>
          </div>
          <div className="mt-6 grid grid-cols-2 gap-4 text-sm text-gray-700">
            <div>
              <div className="text-xs text-gray-500 mb-1">나이</div>
              <div>{selectedPatient?.age ?? '-'}세</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">체중</div>
              <div>{selectedPatient?.weight ?? '-'}kg</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">병동</div>
              <div>{selectedPatient?.ward ?? '-'}</div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">보호자명</div>
              <div>{selectedPatient?.guardian_name ?? '-'}</div>
            </div>
          </div>
        </div>

        {/* Main Status Cards with Trend Graphs */}
        <div className="grid grid-cols-3 gap-6 mb-6">
          {/* Consciousness Card */}
          <div 
            className="bg-white border border-gray-300 p-6 transition-all duration-200 hover:shadow-lg"
            onMouseEnter={() => setHoveredTrend('consciousness')}
            onMouseLeave={() => setHoveredTrend(null)}
          >
            <div className="text-sm text-gray-500 mb-4 pb-3 border-b border-gray-200">의식 수준</div>
            <div className="space-y-3">
              <div className="text-base text-black leading-relaxed">
                외부 자극에 대한 반응이 줄어든 상태입니다.
              </div>
              
              {/* Trend Graph - No Numbers */}
              <div className="py-3">
                <ResponsiveContainer width="100%" height={60}>
                  <LineChart data={consciousnessTrend}>
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#F59E0B" 
                      strokeWidth={2.5} 
                      dot={false} 
                    />
                    <YAxis hide domain={[0, 100]} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>과거</span>
                  <span>현재</span>
                </div>
              </div>

              <div className="text-sm text-gray-600 leading-relaxed pt-3 border-t border-gray-200">
                의료진이 지속적으로 신경 상태를 확인하고 있습니다.
              </div>
              
              {hoveredTrend === 'consciousness' && (
                <div className="text-xs text-orange-600 bg-orange-50 p-2 animate-fade-in">
                  ⚠️ 이 시기에 주의가 필요한 상태
                </div>
              )}
            </div>
          </div>

          {/* Respiratory Card */}
          <div 
            className="bg-white border border-gray-300 p-6 transition-all duration-200 hover:shadow-lg"
            onMouseEnter={() => setHoveredTrend('respiration')}
            onMouseLeave={() => setHoveredTrend(null)}
          >
            <div className="text-sm text-gray-500 mb-4 pb-3 border-b border-gray-200">호흡 상태</div>
            <div className="space-y-3">
              <div className="text-base text-black leading-relaxed">
                산소 보조 장치를 사용 중입니다.
              </div>
              
              {/* Trend Graph - No Numbers */}
              <div className="py-3">
                <ResponsiveContainer width="100%" height={60}>
                  <LineChart data={respirationTrend}>
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#3B82F6" 
                      strokeWidth={2.5} 
                      dot={false} 
                    />
                    <YAxis hide domain={[0, 100]} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>과거</span>
                  <span>현재</span>
                </div>
              </div>

              <div className="text-sm text-gray-600 leading-relaxed pt-3 border-t border-gray-200">
                현재 산소 수치는 안정적이나 면밀히 관찰 중입니다.
              </div>
              
              {hoveredTrend === 'respiration' && (
                <div className="text-xs text-blue-600 bg-blue-50 p-2 animate-fade-in">
                  ℹ️ 안정적으로 유지되고 있습니다
                </div>
              )}
            </div>
          </div>

          {/* General Status Card */}
          <div 
            className="bg-white border border-gray-300 p-6 transition-all duration-200 hover:shadow-lg"
            onMouseEnter={() => setHoveredTrend('overall')}
            onMouseLeave={() => setHoveredTrend(null)}
          >
            <div className="text-sm text-gray-500 mb-4 pb-3 border-b border-gray-200">전신 상태</div>
            <div className="space-y-3">
              <div className="text-base text-black leading-relaxed">
                감염으로 인해 몸이 많이 약해진 상태입니다.
              </div>
              
              {/* Trend Graph - No Numbers */}
              <div className="py-3">
                <ResponsiveContainer width="100%" height={60}>
                  <LineChart data={overallTrend}>
                    <Line 
                      type="monotone" 
                      dataKey="value" 
                      stroke="#8B5CF6" 
                      strokeWidth={2.5} 
                      dot={false} 
                    />
                    <YAxis hide domain={[0, 100]} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>과거</span>
                  <span>현재</span>
                </div>
              </div>

              <div className="text-sm text-gray-600 leading-relaxed pt-3 border-t border-gray-200">
                회복을 위해 집중 치료가 진행 중입니다.
              </div>
              
              {hoveredTrend === 'overall' && (
                <div className="text-xs text-purple-600 bg-purple-50 p-2 animate-fade-in">
                  ℹ️ 회복을 위한 치료가 진행 중
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Timeline Card */}
        <div className="bg-white border border-gray-300 p-8 mb-6">
          <div className="text-lg text-black mb-6 pb-4 border-b border-gray-300">상태 변화 타임라인</div>
          <div className="relative">
            {/* Timeline Line */}
            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-300"></div>
            
            {/* Timeline Events */}
            <div className="space-y-6">
              {timelineEvents.map((event, index) => (
                <div
                  key={index}
                  className={`relative pl-20 transition-all duration-200 ${
                    hoveredEvent === index ? 'transform translate-x-2' : ''
                  }`}
                  onMouseEnter={() => setHoveredEvent(index)}
                  onMouseLeave={() => setHoveredEvent(null)}
                >
                  {/* Icon Circle */}
                  <div
                    className={`absolute left-0 w-16 h-16 flex items-center justify-center text-2xl border-2 ${
                      event.type === 'current'
                        ? 'bg-green-100 border-green-500'
                        : event.type === 'warning'
                        ? 'bg-orange-100 border-orange-400'
                        : 'bg-blue-100 border-blue-400'
                    }`}
                  >
                    {event.icon}
                  </div>
                  
                  {/* Event Content */}
                  <div className={`border-l-4 pl-4 ${
                    event.type === 'current'
                      ? 'border-green-500'
                      : event.type === 'warning'
                      ? 'border-orange-400'
                      : 'border-blue-400'
                  }`}>
                    <div className="flex items-start justify-between mb-1">
                      <div className="text-base text-black">{event.label}</div>
                      <div className="text-xs text-gray-500">{event.time}</div>
                    </div>
                    <div className="text-sm text-gray-600">{event.description}</div>
                    
                    {hoveredEvent === index && (
                      <div className={`text-xs mt-2 p-2 animate-fade-in ${
                        event.type === 'current'
                          ? 'bg-green-50 text-green-700'
                          : event.type === 'warning'
                          ? 'bg-orange-50 text-orange-700'
                          : 'bg-blue-50 text-blue-700'
                      }`}>
                        {event.type === 'current' 
                          ? '지금 이 시점입니다' 
                          : event.type === 'warning'
                          ? '주의가 필요했던 시점'
                          : '치료 진행 중'}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Changes Card */}
        <div className="bg-white border border-gray-300 p-8 mb-6">
          <div className="text-lg text-black mb-6 pb-4 border-b border-gray-300">최근 24시간 변화</div>
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-orange-100 border border-orange-300 flex items-center justify-center text-xl">
                ⚠️
              </div>
              <div className="flex-1">
                <div className="text-base text-black mb-1">혈압이 일시적으로 낮아짐</div>
                <div className="text-sm text-gray-600">약물 치료를 통해 관리하고 있습니다</div>
              </div>
              <div className="flex-shrink-0 px-3 py-1 bg-orange-100 text-xs text-orange-900">주의</div>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-orange-100 border border-orange-300 flex items-center justify-center text-xl">
                ⚠️
              </div>
              <div className="flex-1">
                <div className="text-base text-black mb-1">호흡 보조 단계 증가</div>
                <div className="text-sm text-gray-600">더 많은 산소 공급이 필요한 상태입니다</div>
              </div>
              <div className="flex-shrink-0 px-3 py-1 bg-orange-100 text-xs text-orange-900">주의</div>
            </div>

            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-12 h-12 bg-gray-100 border border-gray-300 flex items-center justify-center text-xl">
                👁️
              </div>
              <div className="flex-1">
                <div className="text-base text-black mb-1">감염 수치 아직 높음</div>
                <div className="text-sm text-gray-600">항생제 치료가 계속 진행되고 있습니다</div>
              </div>
              <div className="flex-shrink-0 px-3 py-1 bg-gray-100 text-xs text-gray-700">관찰</div>
            </div>
          </div>
        </div>

        {/* Treatment Plan Card */}
        <div className="bg-gray-100 border border-gray-400 p-8 mb-6">
          <div className="text-lg text-black mb-4 pb-4 border-b border-gray-400">향후 치료 계획</div>
          <div className="text-base text-black leading-relaxed">
            앞으로 며칠간은 호흡과 감염 관리가 가장 중요한 시기입니다.
            의료진이 상태를 지속적으로 관찰하며 필요한 치료를 진행하고 있습니다.
          </div>
        </div>

        {/* Additional Information */}
        <div className="grid grid-cols-3 gap-6 mb-6">
          <div className="bg-white border border-gray-300 p-6">
            <div className="text-xs text-gray-500 mb-3 pb-2 border-b border-gray-200">담당 의료진</div>
            <div className="text-sm text-black">중환자의학과</div>
            <div className="text-sm text-black">이OO 전문의</div>
          </div>

          <div className="bg-white border border-gray-300 p-6">
            <div className="text-xs text-gray-500 mb-3 pb-2 border-b border-gray-200">문의 시간</div>
            <div className="text-sm text-black">평일 오전 10시 ~ 11시</div>
            <div className="text-sm text-black">오후 4시 ~ 5시</div>
          </div>

          <div className="bg-white border border-gray-300 p-6">
            <div className="text-xs text-gray-500 mb-3 pb-2 border-b border-gray-200">다음 상태 업데이트</div>
            <div className="text-sm text-black">12시간 후</div>
            <div className="text-sm text-gray-600">(2026년 1월 16일 오전 8시)</div>
          </div>
        </div>

        {/* Notice */}
        <div className="bg-blue-50 border border-blue-300 p-6">
          <div className="text-sm text-blue-900 leading-relaxed">
            <span className="block mb-2">💡 <strong>안내사항</strong></span>
            환자분의 상태는 매 시간 의료진이 확인하고 있으며, 중요한 변화가 있을 경우 즉시 보호자님께 연락드립니다. 
            궁금하신 사항은 중환자실 간호사실(내선 3301)로 언제든지 문의해 주세요.
          </div>
        </div>
      </div>
    </div>
  );
}
