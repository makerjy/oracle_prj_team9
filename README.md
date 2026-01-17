# ICU Risk Monitor (grd_kstep 기반)

ICU 환자의 현재 시점 위험도를 실시간으로 모니터링하기 위한 모델/서빙 파이프라인 요약입니다.  
모델은 미래를 직접 관측하지 않고, 과거 관측 시계열만 사용해 **각 시점 t에서 “향후 H시간 내 이벤트 발생 여부”**를 예측합니다.

---

## 1) 프로젝트 개요
- 레포: `grd_kstep`
- 목적: 각 시점 t에서 **(t, t+H] 구간 내 이벤트 발생 여부**를 예측하는 row-level 위험도 모델
- 서비스 가정: ICU 환자 데이터의 **현재 시점 위험도**를 실시간 업데이트
- 핵심 원칙: **미래 데이터 누출 금지** (t 이후 관측값은 입력에 포함하지 않음)

---

## 2) 서비스 기능 및 구현 범위
- 실시간 위험도 업데이트: 환자별 시퀀스 버퍼 유지 → 모델 추론 → 6/24/72h 위험도 산출
- 환자 목록/타임라인 API: `GET /api/patients`, `GET /api/patients/{id}/timeline`
- 알림 규칙 계산: 24h 위험도 임계치/상승폭 기반 alert flag + reason 제공
- 데모 시뮬레이션: `backend/main.py`에서 synthetic signals 생성 후 `/patients`, `/patients/{id}/risk-summary`, `/patients/{id}/risk-trajectory`, `/demo/*` 제공
- 프론트 화면: 의료진 대시보드(차트/상태), 보호자 대시보드(요약/트렌드), K-step API 테스트 화면

---

## 3) 데이터 스키마
### 3-1. 주요 컬럼
- `stay_id`, `t`, `event`
- `feature_cols` (고정 순서)
  1. `HeartRate_std_6h`
  2. `GCS_Verbal`
  3. `SpO2_measured`
  4. `RespRate_std_6h`
  5. `SysBP`
  6. `GCS_Motor`
  7. `GCS_Total_mean_6h`
  8. `Temp_std_6h`
  9. `pH`
  10. `DiasBP_mean_6h`
  11. `MeanBP`
  12. `FiO2`

### 3-2. 라벨 정의 (핵심)
- `_future_label`:
  - 현재 시점 t 이후 **(t, t+H]** 구간에 `event==1`이 한 번이라도 있으면 1, 없으면 0
- `_label_observable`:
  - 미래 H 구간을 끝까지 관측했으면 1  
  - `_future_label == 1`인 경우는 항상 observable

### 3-3. 학습/평가 사용 범위
- **`_label_observable == 1`인 행만 학습/평가에 사용**
- 서비스 추론에서는 **현재 시점 입력만 사용**하며 observable 필터는 필요 없음

---

## 4) 소스 구조 (grd_kstep 기준)
- `main.py`: 실행 엔트리, `src.grd_step.main` 호출
- `src/grd_step.py`: `RunConfig` 생성 후 `train_and_evaluate` 실행
- `src/config.py`: 데이터/모델/학습/출력 설정
- `src/data.py`: 데이터 로딩, 라벨 생성/필터, impute/standardize, 시퀀스 생성
- `src/model.py`: `TimewiseGRU` 정의
- `src/metrics.py`: row/stay 평가, threshold 선택, metric 계산
- `src/viz.py`: PPT용 시각화 및 테이블 이미지 생성
- `src/train.py`: 전처리 → 학습 → 평가 → 저장 파이프라인

---

## 5) 설정 기본값 (src/config.py)
- `data.horizon_hours = 24` (H)
- `data.label_col = "_future_label"`
- `data.label_observable_col = "_label_observable"`
- `data.drop_after_event = True`
- `data.cutoff_hours = 24` (stay-level 평가 cutoff)
- `data.agg_mode = "max"`
- `data.target_recall = 0.80`
- `data.use_precomputed_labels = True`
- `sequence.max_len = 120` (stay 시퀀스 최대 길이, 초과 시 앞부분 유지)
- `model.hidden = 64`, `model.dropout = 0.2`, `model.n_layers = 1`
- `train.lr = 1e-3`, `weight_decay = 1e-4`, `patience = 5`, `min_delta = 1e-4`

---

## 6) 데이터 로딩 & 전처리 흐름 (src/train.py + src/data.py)
1. train/valid/test parquet 로드
2. 필수 컬럼 체크 (`id/time/event + feature_cols`)
3. `id/time` 기준 정렬
4. 라벨 준비
   - `use_precomputed_labels=True`이면 기존 `_future_label/_label_observable` 사용
   - 없거나 `recompute=True`면 event로부터 새로 생성
5. `_label_observable == 1` 필터
6. `drop_after_event=True`면 첫 event 이후 행 제거
7. (옵션) `cutoff_hours`로 train 데이터 제한
8. 결측치 처리: **train 평균으로 impute**
9. 표준화: **train 평균/표준편차로 standardize**
10. 시퀀스 구성
    - `stay_id`별로 시간순 (X, y, t, idx) 생성
    - `max_len` 초과 시 앞부분 유지

---

## 7) 모델 (src/model.py)
### TimewiseGRU
- GRU (`batch_first=True`) → Dropout → Linear(hidden → 1)
- `pack_padded_sequence`로 길이 다른 시퀀스 처리
- 출력: 시퀀스 길이만큼의 **timewise logits**
  - 각 row에 대한 위험도(logit)를 의미

---

## 8) 학습 (src/train.py)
- Loss: `BCEWithLogitsLoss` + `pos_weight` (train 내 클래스 불균형 기반)
- Padding mask로 pad 구간 loss 제외
- Optimizer: `AdamW`
- AMP: CUDA일 때 mixed precision
- Early stopping: valid loss 기준
- Validation row score: row index로 복원하여 row-level AUC/AP 계산

---

## 9) 평가 (src/metrics.py)
### Row-level
- valid/test 전체 row_score vs `_future_label`
- AUC / AP 계산

### Stay-level
- `time <= cutoff_hours` 구간만 사용
- agg_mode에 따라 stay score 계산 (`max`/`mean`/`last`)
- stay label = `_future_label`의 stay max

### Threshold 선택
- valid에서 **target_recall 이상** 만족하는 threshold 중 precision 최대
- 해당 threshold로 test precision/recall/f1 산출

---

## 10) 산출물
### artifacts/
- `kstep_gru_state.pt` (모델 가중치)
- `kstep_gru_meta.json` (전처리/설정/metrics 메타)
- `training_history.json`

### output/
- `row_metrics.csv`, `stay_metrics.csv`
- `row_metrics_table.png`, `stay_metrics_table.png`
- `training_loss.png`, `valid_row_auc_ap.png`, `valid_stay_auc_ap.png`, `stay_metrics.png`

---

## 11) 서비스 추론 체크리스트
- **train 기준 `impute_stats + standardize_stats` 적용**
- `feature_cols` 순서를 고정 유지
- stay별 시퀀스 입력 후, 현재 시점 위험도는 **해당 시점 row_score** 사용
- `_label_observable` 필터는 학습/평가에만 적용 (실서비스에서는 불필요)

---

## 12) 서비스 연동 (이 레포)
### 백엔드
- FastAPI 기반 (`backend/main.py`, `backend/routes.py`)
- 요청 흐름
  1. `/api/infer`로 현재 시점 feature 전송
  2. 서버는 환자별 시퀀스 버퍼 유지
  3. impute/standardize → 모델 추론
  4. 6/24/72시간 위험도 계산
  5. 환자별 타임라인/알림 상태 업데이트

### 주요 API
```
POST /api/infer
GET  /api/patients
GET  /api/patients/{id}/timeline
```

---

## 13) 재현 실행
### 학습 (grd_kstep)
```
python main.py
```

### 서비스 실행 (이 레포)
```
uvicorn backend.main:app --reload --port 8000
npm run dev
```

---

## 14) 주의 사항
- `_future_label`이 **H 시간 기준**으로 생성되었는지 확인
- stay-level 평가는 `cutoff_hours` 이후 구간 제외
- `max_len` 자르기 전략은 “앞부분 유지” (필요 시 “마지막 N개 유지”로 변경 가능)
