from src.data_loader import load_price_data
from src.split import split_time_series
from src.modeling import find_best_arima_model
from src.diagnostics import evaluate_residuals
from src.evaluate import evaluate_and_plot
from configs.settings import RAW_DATA_DIR, TARGET_COLUMN
import pandas as pd
import os

def train_classical_pipeline(data_name: str, file_path: str):
    """
    고전적 모델(ARIMA)을 위한 데이터 로드, 분할, 최적 파라미터 훈련 및 잔차 진단까지의 통합 파이프라인
    """
    print("=" * 50)
    print(f" [고전적 모델] {data_name} - ARIMA 베이스라인 구축 시작")
    print("=" * 50)
    
    # 1. 데이터 로드 (로컬 CSV 파일 읽기)
    df = load_price_data(file_path)
    
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
    evaluate_residuals(best_model, data_name=data_name)
    
    # 6. 테스트 데이터 대상 Out-Of-Sample 예측
    print("\n[테스트 셋 기반 예측 및 평가 진행]")
    n_periods = len(test_df)
    # auto_arima로 생성된 모델은 .predict(n_periods=...) 로 미래 예측 가능
    forecasts = best_model.predict(n_periods=n_periods)
    
    series_test = test_df[TARGET_COLUMN]
    
    # 7. 평가 수행 및 그래프 저장
    metrics = evaluate_and_plot(series_test, forecasts, title=f"ARIMA Forecast vs Actual ({data_name})", data_name=data_name)
    
    print(f"\n[{data_name} 고전적 모델 파이프라인 구축 완료]")
    return best_model, metrics

def train_all_models():
    """
    지정된 대상 종목 데이터를 순회하며 ARIMA 모델을 학습하고, 결과를 종합합니다.
    """
    target_companies = [
        "hanwha_aerospace", "lig_nex1", "snt_dynamics", "firstec",
        "samsung_electronics", "sk_hynix", "wonik_ips", "ia",
        "rtx", "aerovironment", "draganfly", "nvidia", "axt", "maxlinear"
    ]

    csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".csv")]
    
    filtered_files = []
    for f in csv_files:
        data_name = f.replace("_5y.csv", "").replace(".csv", "")
        if data_name in target_companies:
            filtered_files.append(f)

    all_metrics = []
    
    print(f"총 {len(filtered_files)}개의 종목에 대한 고전적 모델 파이프라인(ARIMA)을 일괄 실행합니다.")
    
    for f in filtered_files:
        data_name = f.replace("_5y.csv", "").replace(".csv", "")
        file_path = os.path.join(RAW_DATA_DIR, f)
        
        try:
            _, metrics = train_classical_pipeline(data_name, file_path)
            metrics["data_name"] = data_name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[{data_name}] ❌ 학습 중 오류 발생: {e}")
            
    # 전체 결과를 DataFrame으로 변환하여 저장
    df_metrics = pd.DataFrame(all_metrics)
    os.makedirs("results/metrics", exist_ok=True)
    summary_path = "results/metrics/classical_arima_summary.csv"
    df_metrics.to_csv(summary_path, index=False, encoding="utf-8-sig")
    
    print("=" * 50)
    print(f"🚀 전체 {len(filtered_files)}개 종목 분석 완료! 결과 요약 저장됨: {summary_path}")
    print("=" * 50)
    return df_metrics