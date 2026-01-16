# ICU Risk Monitor: 모델 원리 및 프론트/백엔드 연동 설명

## 1) 모델 개요
이 서비스는 ICU 환자의 **현재 시점 기준 위험도**를 실시간으로 모니터링합니다.  
모델은 **미래 상태를 직접 예측하지 않고**, 과거 L시간의 관측된 시계열만을 사용해 **향후 일정 기간 내 사건 발생 확률**을 계산합니다.

### 핵심 아이디어
- 입력: 과거 L시간의 feature 시퀀스 (예: L=24, 시간 단위)
- 모델 출력: K-step hazard sequence (예: K=120)
- 서비스 표시: 다음 6/24/72시간 내 위험도(확률)

즉, **각 시점 t에서 "향후 24시간 내 사건 발생 확률"을 계산**해 화면에 보여줍니다.

---

## 2) 입력/출력 구조

### 2-1. 입력 Feature (12개, 순서 고정)
```
1. HeartRate_std_6h
2. RespRate_std_6h
3. Temp_std_6h
4. GCS_Total_mean_6h
5. DiasBP_mean_6h
6. SysBP
7. MeanBP
8. SpO2_measured
9. FiO2
10. pH
11. GCS_Verbal
12. GCS_Motor
```

### 2-2. 모델 입력 형태
- shape: (1, L, 12)
- L은 과거 시퀀스 길이 (기본 24시간)

### 2-3. 모델 출력 형태
- hazard_seq: 길이 K (예: 120)
- 형태 예시:
```
[h(t+1), h(t+2), ..., h(t+K)]
```

---

## 3) 위험도 계산 방식 (서비스 표준)

모델 출력은 time-step별 hazard 값입니다.  
이를 **향후 특정 시간 window 내 사건 발생 확률**로 변환합니다.

### Window risk 계산
```
risk(window) = 1 - Π_{k=1..window} (1 - hazard_seq[k])
```

서비스에서 제공하는 위험도:
- Next 6h risk
- Next 24h risk
- Next 72h risk

이 값만 UI의 “위험도”로 사용합니다.

---

## 4) 미래 누출 방지 원칙
- 입력은 항상 **현재 시점 t까지 관측된 데이터**만 사용
- t 이후 feature는 모델에 들어가지 않음
- 그래프도 **현재 시점까지의 위험도만 표시**

---

## 5) 백엔드 처리 흐름 (FastAPI)

### 5-1. 요청 흐름
1. 프론트가 `/api/infer`로 현재 시점 feature 전송  
2. 서버는 환자별 최근 L시간 시퀀스를 유지 (슬라이딩 buffer)
3. 결측치 대체(impute) 후 모델 입력 생성
4. 모델이 hazard_seq 생성
5. risk_6h / 24h / 72h 계산
6. 결과 저장 후 응답

### 5-2. 주요 엔드포인트
```
POST /api/infer
GET  /api/patients
GET  /api/patients/{id}/timeline
```

---

## 6) 프론트 표시 규칙

### 6-1. 카드
- Next 6h Risk
- Next 24h Risk
- Next 72h Risk

### 6-2. 그래프
- x축: 입실 후 경과시간 (t = 0 → 현재)
- y축: “각 시점 t에서 향후 24시간 내 사건 발생 확률”
- 미래 trajectory는 표시하지 않음

---

## 7) 알림 규칙 (예시)

- risk_24h >= 0.20
- 연속 2회 이상 임계값 초과 또는 최근 6시간 상승폭 >= 0.05

이 기준은 검증 데이터에서 조정 가능.

---

## 8) 평가 지표 (AUROC/AUPRC)

실시간 위험도 예측 품질은 **window-based binary label**로 평가합니다.

### 8-1. 라벨 정의 (예: 24시간 window)
```
y = 1 if event occurs in (t, t + 24h]
y = 0 otherwise
```

### 8-2. 지표
- **AUROC**: 전체 분류 성능 요약
- **AUPRC**: 희귀 이벤트(사망 등)에서 더 민감한 지표
- **Recall@Threshold**: 임상적 threshold 기반 감지율

### 8-3. 주의
- **stay-level survival 분석(KM curve)**과 **row-level 예측 성능**은 분리해서 해석합니다.
- 실시간 UI 점수는 window risk만 사용합니다.

---

## 9) 모델 학습 파이프라인 (요약)

### 9-1. 데이터 준비
1. Raw 이벤트 수집 → 시간 단위로 정렬  
2. **최근 L시간 window** 기준 feature 계산  
3. 결측치 처리 (학습 시 사용한 impute 통계 적용)

### 9-2. 학습 입력/출력
- 입력: (batch, L, 12) 과거 시퀀스
- 출력: K-step hazard_seq (K=120)
- 학습 목적: survival likelihood 기반 손실

### 9-3. 서빙과 정합성
- 학습/서빙 모두 **동일한 FEATURE_COLS 순서**
- L, K, DT는 meta.pkl에 저장하여 서버가 동일하게 사용
- 서빙에서는 **risk window 계산만 수행**, 추가 스케일링 금지

---

## 8) 모델 파일 위치
모델 파일은 아래 경로에 두면 됩니다:
```
backend/artifacts/kstep_gru_state.pt
backend/artifacts/kstep_gru_meta.pkl
```

`backend/model_loader.py`에서 자동 로드합니다.
