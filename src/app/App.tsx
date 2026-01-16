import { useState } from 'react';
import MedicalStaffDashboard from '@/app/components/MedicalStaffDashboard';
import FamilyDashboard from '@/app/components/FamilyDashboard';

export default function App() {
  const [currentView, setCurrentView] = useState<'medical' | 'family'>('medical');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [selectedPatientId, setSelectedPatientId] = useState('pt-0999');

  return (
    <div className="min-h-screen">
      {/* Navigation Toggle */}
      <div
        className={
          currentView === 'medical'
            ? isDarkMode
              ? 'border-b border-slate-800 bg-slate-900'
              : 'border-b border-slate-200 bg-white'
            : 'border-b border-slate-200 bg-white'
        }
      >
        <div className="flex items-center justify-between">
          <div className="flex">
            <button
              onClick={() => setCurrentView('medical')}
              className={`px-6 py-4 text-sm transition-colors ${
                currentView === 'medical'
                  ? isDarkMode
                    ? 'bg-white text-slate-900'
                    : 'bg-slate-900 text-white'
                  : isDarkMode
                  ? 'bg-slate-900 text-slate-400 hover:bg-slate-800'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              의료진용 대시보드
            </button>
            <button
              onClick={() => setCurrentView('family')}
              className={`px-6 py-4 text-sm transition-colors ${
                currentView === 'family'
                  ? 'bg-slate-900 text-white'
                  : 'bg-white text-slate-600 hover:bg-slate-100'
              }`}
            >
              가족/보호자용 화면
            </button>
          </div>
          
          {/* Dark/Light Mode Toggle - Only show for medical view */}
          {currentView === 'medical' && (
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className={`mr-6 px-4 py-2 text-sm border transition-colors ${
                isDarkMode 
                  ? 'bg-slate-800 text-slate-200 border-slate-600 hover:bg-slate-700' 
                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'
              }`}
            >
              {isDarkMode ? '라이트 모드' : '다크 모드'}
            </button>
          )}
        </div>
      </div>

      {/* Dashboard Content */}
      {currentView === 'medical' ? (
        <MedicalStaffDashboard
          isDarkMode={isDarkMode}
          selectedPatientId={selectedPatientId}
          onSelectPatient={setSelectedPatientId}
        />
      ) : (
        <FamilyDashboard
          selectedPatientId={selectedPatientId}
          onSelectPatient={setSelectedPatientId}
        />
      )}
    </div>
  );
}
