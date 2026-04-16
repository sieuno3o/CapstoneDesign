from src.data_loader import download_yahoo_data
from src.split import split_time_series
from src.modeling import find_best_arima_model
from src.diagnostics import evaluate_residuals
from src.evaluate import evaluate_and_plot
from configs.settings import TARGET_TICKER, PERIOD, TARGET_COLUMN

def train_classical_pipeline():
    """
    고전적 모델(ARIMA)을 위한 데이터 로드, 분할, 최적 파라미터 훈련 및 잔차 진단까지의 통합 파이프라인
    """
    print("=" * 50)
    print(f" [고전적 모델] {TARGET_TICKER} - ARIMA 베이스라인 구축 시작")
    print("=" * 50)
    
    # 1. 데이터 로드
    df = download_yahoo_data(TARGET_TICKER, PERIOD)
    
    # 2. 데이터 시계열 분리 (Train / Validation / Test)
    train_df, val_df, test_df = split_time_series(df, train_ratio=0.7, val_ratio=0.15)
    print(f"\n[데이터 분할 완료]")
    print(f"  - Train Shape: {train_df.shape}")
    print(f"  - Valid Shape: {val_df.shape}")
    print(f"  - Test Shape : {test_df.shape}")
    
    # 3. 모델 타깃 지정
    series_train = train_df[TARGET_COLUMN]
    
    # 4. 모델 파라미터 탐색 및 학습
    print("\n[모델 최적 파라미터 자동 탐색 및 학습 시작 (Auto ARIMA)]")
    # auto_arima는 내부적으로 fit까지 완료된 모델을 반환합니다.
    best_model = find_best_arima_model(series_train)
    
    # 5. 모형 진단
    evaluate_residuals(best_model)
    
    # 6. 테스트 데이터 대상 Out-Of-Sample 예측
    print("\n[테스트 셋 기반 예측 및 평가 진행]")
    n_periods = len(test_df)
    # auto_arima로 생성된 모델은 .predict(n_periods=...) 로 미래 예측 가능
    forecasts = best_model.predict(n_periods=n_periods)
    
    series_test = test_df[TARGET_COLUMN]
    
    # 7. 평가 수행 및 그래프 저장
    evaluate_and_plot(series_test, forecasts, title=f"ARIMA Forecast vs Actual ({TARGET_TICKER})")
    
    print("\n[고전적 모델 파이프라인 구축 완료]")
    return best_model, train_df, val_df, test_df

def train_all_models():
    """
    4개 모델을 순서대로 학습시키는 통합 함수
    """
    # 현재는 고전적 모델만 완성된 상태입니다.
    train_classical_pipeline()
    pass