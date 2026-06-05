# CHANGELOG

이 문서는 캡스톤 디자인 프로젝트에서 수행된 코드 수정 및 개선 내역을 기록합니다.  
수정이 발생할 때마다 이 파일에 업데이트됩니다.

---

## [2026-05-20] 확장 RF 파이프라인 코드 리뷰 및 버그 수정

### 1. 평가 지표 불일치 수정

**배경**  
팀원이 작성한 `train_rf_extended.py`의 자체 `calculate_metrics()`는 RMSE, MAE, R², MAPE만 계산하여,  
기존 ARIMA·ANN 파이프라인이 사용하는 `evaluate.py` 기준(MBE, 방향성 정확도 포함)과 달랐음.

**수정 파일 및 내용**

#### `src/evaluate.py`
- `r2_score` import 추가
- `regression_metrics()` 함수에 **R²** 계산·출력 추가
- 반환 딕셔너리 키 정렬: `rmse → mae → mape → mbe → r2`

#### `src/train_rf_extended.py`
- 자체 `calculate_metrics(y_true, y_pred)` **삭제**
- `src/evaluate.py`에서 `regression_metrics()`, `direction_accuracy()` import
- 새 `calculate_metrics(y_true, y_pred, model_name)` 래퍼 함수로 교체
  - 계산 지표: **RMSE, MAE, MAPE, MBE, R², 방향성 정확도** (6개 통합)
- 중복 정의된 `predict_ai_model()` 함수 **삭제**
  - `ai_model.py`에 동일 함수가 이미 존재했음
  - `rf_model.predict()` 직접 호출로 통일
- 결과 DataFrame 컬럼 변경:  
  `["Model", "RMSE", "MAE", "R2", "MAPE"]`  
  → `["Model", "rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]`

#### `run_rf_extended.py`
- 전체 종목 평균 요약 테이블 `groupby` 컬럼 변경:  
  `["RMSE", "MAE", "R2", "MAPE"]`  
  → `["rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]`

---

### 2. 확장 피처 품질 개선 (검토 항목 3개)

#### (1) 비율 기반 변수 추가 (절대 가격 변수와 병행)

**배경**  
`EXTENDED_FEATURES`에 있던 `BB_upper`, `BB_lower`, `MA_3~20`, `Volume_MA5/20` 등은  
주가 절대값에 의존하므로 종목 간 스케일 차이가 발생할 수 있음.  
→ 절대 가격 변수를 제거하는 대신, **비율 기반 변수를 추가**하여 모델이 두 정보를 모두 활용할 수 있도록 함.

**수정 파일: `src/train_rf_extended.py` — `EXTENDED_FEATURES`**

| 유지된 변수 (절대 가격) | 새로 추가된 변수 (비율 기반) |
|---|---|
| `BB_upper`, `BB_lower` | `BB_width`, `BB_percent` |
| `MA_3`, `MA_5`, `MA_10`, `MA_20` | `Close_MA5_ratio`, `Close_MA20_ratio`, `MA5_MA20_gap` |
| `Volume_MA5`, `Volume_MA20` | `Volume_ratio`, `Volume_change` |

---

## [2026-05-20] 추천 항목 구현 + 추가 기술적 지표 확장

### 1. 추가 기술적 지표 4종 신규 구현

**수정 파일: `src/feature_engineering_extended.py`**

| 변수명 | 설명 | 추가 이유 |
|---|---|---|
| `ATR_14` | Average True Range (14일) | 고가/저가/전일종가를 모두 활용하는 변동성 지표. RSI·볼린저밴드와 달리 갭상승/갭하락 반영 |
| `OBV` | On-Balance Volume | 가격 상승일 거래량 누적 - 하락일 거래량 누적. `Volume_ratio`보다 추세 방향 정보 풍부 |
| `Stoch_K`, `Stoch_D` | 스토캐스틱 오실레이터 (14일 %K, 3일 %D) | RSI와 달리 14일 고가/저가 범위 내 현재 종가 위치 반영. 과매수·과매도 보완 |
| `lag_1/2/3_return` | 1~3일 전 일별 수익률 | 과거 수익률을 피처로 직접 제공 → 단기 추세 기억 효과, RF 모델에 특히 효과적 |

**수정 파일: `src/train_rf_extended.py` — `EXTENDED_FEATURES`**
- 위 4종(`ATR_14`, `OBV`, `Stoch_K`, `Stoch_D`, `lag_1_return`, `lag_2_return`, `lag_3_return`) 목록에 추가

---

### 2. 피처 중요도 시각화 추가

**수정 파일: `src/train_rf_extended.py`**
- Extended RF 학습 완료 후 `model.feature_importances_`로 상위 20개 변수 중요도 bar chart 생성
- 저장 경로: `results/figures/{data_name}_feature_importance.png`

---

### 3. 기본 RF vs 확장 RF 성능 개선 요약 출력 추가

**수정 파일: `run_rf_extended.py`**
- 전체 종목 실행 완료 후, 종목별 `Existing RF vs Extended RF` RMSE 개선율(%) 테이블 출력
- ✅ 개선 / ❌ 악화 여부 표시
- 전체 종목 중 개선된 비율 요약 출력

---

### 4. `main.py`에 확장 RF 파이프라인 통합

**수정 파일: `main.py`**
- `run_rf_extended()` 함수 정의 (14개 종목 딕셔너리 포함)
- `main()` 내 `# 2-1. 확장 RF 모델` 단계로 호출 추가
- 이제 `python main.py` 하나로 ARIMA → RF+ANN → 확장 RF 전체 실험이 순차 실행됨

---

### 5. 그래프/결과 저장 경로 통일

**수정 파일: `src/train_rf_extended.py`, `run_rf_extended.py`**
- 예측 그래프: `figures/` → **`results/figures/`**
- 종목별 결과 CSV: `results/` → **`results/metrics/`**

---

### 6. `MA3_return` (3일 수익률 이동평균) 변수 추가

팀원 요청 사항 반영.

**수정 파일: `src/feature_engineering_extended.py`**
- `MA3_return = daily_return.rolling(window=3).mean()` 추가
- `lag_1/2/3_return`(특정 과거 시점 수익률 레그)과 달리 최근 3일 수익률의 **평균 방향성** 포착

**수정 파일: `src/train_rf_extended.py` — `EXTENDED_FEATURES`**
- `"MA3_return"` 목록에 추가

---

## [2026-05-21] 확장 ANN 모델 구현

기존 ANN(`train_ai_pipeline.py`)은 원본 7개 변수만 사용하고 있었음.  
확장 RF와 동일한 30개+ 피처를 ANN에도 적용하여 공정한 모델 비교 환경 구성.

### 변경 파일: `src/train_rf_extended.py`

**추가 내용 (8-F 단계):**
- `src/ai_model`에서 `train_ann_model`, `predict_ai_model` import
- Extended RF와 **동일한 `scaler_ext` 스케일러, 동일한 `EXTENDED_FEATURES`** 재사용
- `train_ann_model(X_train_ext, y_train.values)` 호출 → 64→32→1 구조 ANN 학습 (100 epoch)
- `results["Extended ANN"]` 에 예측값 저장

**결과 변경:**
- 비교 모델: Benchmark / ARIMA / Existing RF / Extended RF → **+ Extended ANN** (5개 모델)
- 저장 파일명: `{data_name}_rf_extended_results.csv` → **`{data_name}_ai_extended_results.csv`**
- 예측 그래프: `{data_name}_rf_extended_prediction.png` → **`{data_name}_ai_extended_prediction.png`**
  - Extended ANN 선: `forestgreen` 색상으로 추가

### 변경 파일: `run_rf_extended.py`

- 종합 결과 파일명: `rf_extended_models_summary.csv` → **`ai_extended_models_summary.csv`**
- 기본 RF vs 확장 RF/ANN 개선율 비교 테이블 업데이트:
  - `RF_improvement_%` 와 `ANN_improvement_%` 두 컬럼으로 확장
  - 확장 RF / 확장 ANN 각각 몇 개 종목에서 개선됐는지 최종 요약 출력



---

#### (2) `volatility` 컬럼명 충돌 해결

**배경**  
- `feature_engineering.py`의 `add_volatility()`는 **log_return** 기반 `volatility_7` 생성  
- `feature_engineering_extended.py`의 `add_extended_features()`는 **pct_change(daily_return)** 기반 `volatility_5`, `volatility_10` 생성  
- 이름만 보면 같은 종류의 변수처럼 보이나 계산 기준이 달라 혼동 가능

**수정 파일: `src/feature_engineering_extended.py`**
- `volatility_5` → **`pct_volatility_5`**
- `volatility_10` → **`pct_volatility_10`**
- 주석 추가: pct_change 기반임을 명시, `volatility_7`과의 차이 설명

**수정 파일: `src/train_rf_extended.py` — `EXTENDED_FEATURES`**
- `"volatility_5"`, `"volatility_10"` → `"pct_volatility_5"`, `"pct_volatility_10"` 으로 변경

---

#### (3) `Trading_value` 로그 변환

**배경**  
`Trading_value = Close × Volume` 은 종목에 따라 수십억~수조 단위의 값을 가짐.  
MinMaxScaler 정규화 후에도 Train/Test 구간 스케일 차이가 커 정보 비대칭 발생 가능.

**수정 파일: `src/feature_engineering_extended.py`**
- 변경 전: `res["Trading_value"] = res["Close"] * res["Volume"]`
- 변경 후: `res["log_trading_value"] = np.log1p(res["Close"] * res["Volume"])`

**수정 파일: `src/train_rf_extended.py` — `EXTENDED_FEATURES`**
- `"Trading_value"` → `"log_trading_value"` 로 변경
