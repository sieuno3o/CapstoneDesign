from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from src.data_loader import load_price_data
from src.preprocess import (
    add_return_features,
    add_target_next_close,
    drop_missing_rows,
)
from src.feature_engineering import (
    add_moving_averages,
    add_volatility,
    add_price_structure_features,
)
from src.feature_engineering_extended import add_extended_features
from src.split import split_time_series
from src.modeling import find_best_arima_model
from sklearn.ensemble import RandomForestRegressor

def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model

def predict_ai_model(model, X):
    pred = model.predict(X)
    # Keras returns 2D array, RF returns 1D array
    if len(pred.shape) > 1 and pred.shape[1] == 1:
        pred = pred.flatten()
    return pred


ORIGINAL_FEATURES = [
    "Volume",
    "ma_7",
    "ma_14",
    "ma_21",
    "volatility_7",
    "hl_diff",
    "oc_diff"
]

EXTENDED_FEATURES = ORIGINAL_FEATURES + [
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width", "BB_percent",
    "MA_3", "MA_5", "MA_10", "MA_20",
    "Close_MA5_ratio", "Close_MA20_ratio", "MA5_MA20_gap",
    "daily_return", "abs_return", "high_low_ratio", "open_close_ratio",
    "volatility_5", "volatility_10",
    "Volume_MA5", "Volume_MA20", "Volume_change", "Volume_ratio",
    "Trading_value"
]

TARGET_COL = "target_next_close"


def calculate_metrics(y_true, y_pred):
    """
    Computes RMSE, MAE, R^2, and MAPE metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def train_rf_extended_pipeline(data_name: str, file_path: str):
    print("=" * 80)
    print(f"[확장 RF 파이프라인] {data_name} 실행 시작")
    print("=" * 80)

    # 1. 데이터 불러오기
    df = load_price_data(file_path)

    # 2. 기존 파생변수 생성
    df = add_return_features(df, price_col="Close")
    df = add_moving_averages(df, price_col="Close")
    df = add_volatility(df, return_col="log_return")
    df = add_price_structure_features(df)

    # 3. 추가 입력변수 (RSI, MACD, 볼린저밴드 등) 생성
    df = add_extended_features(df)

    # 4. 타깃 생성 (다음날 종가)
    df = add_target_next_close(df, price_col="Close")

    # 5. 전처리 완료된 데이터 저장 (결측 제거 전 전체 구조 보존용)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    processed_save_path = f"data/processed/{data_name}_features_extended.csv"
    df.to_csv(processed_save_path, index=False, encoding="utf-8-sig")
    print(f"[전처리 데이터 저장 완료] -> {processed_save_path}")

    # 6. 결측 및 무한대 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    df = drop_missing_rows(df)

    # 7. 데이터 분할 (시간 순서대로 Train 70%, Val 15%, Test 15%)
    train_df, val_df, test_df = split_time_series(df, train_ratio=0.7, val_ratio=0.15)
    print(f"[데이터 분할 완료] Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # dates for plotting
    test_dates = pd.to_datetime(test_df["Date"])
    y_train = train_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    
    # 8. 4가지 모델 예측 수행
    results = {}

    # (A) Benchmark 모델 (Naive: 내일 종가 = 오늘 종가)
    # y_test는 shift(-1)한 값이므로, Naive 예측은 shift(-1) 전인 오늘 종가인 Close입니다.
    naive_pred = test_df["Close"]
    results["Benchmark"] = naive_pred.values

    # (B) 기본 시계열 모델 (ARIMA)
    print(f"\n[{data_name}] ARIMA 베이스라인 학습 중...")
    try:
        # ARIMA는 과거 종가(Close)의 패턴만 학습
        arima_model = find_best_arima_model(train_df["Close"])
        arima_pred = arima_model.predict(n_periods=len(test_df))
        # 만약 arima_pred가 Series 형식이거나 인덱스가 밀리는 것을 방지하기 위해 array 변환
        results["ARIMA"] = arima_pred if isinstance(arima_pred, np.ndarray) else arima_pred.values
    except Exception as e:
        print(f"[오류] Auto ARIMA 탐색 실패, 기본 ARIMA(1,1,1)로 폴백합니다. ({e})")
        from statsmodels.tsa.arima.model import ARIMA
        try:
            basic_model = ARIMA(train_df["Close"], order=(1,1,1))
            fitted = basic_model.fit()
            arima_pred = fitted.forecast(steps=len(test_df))
            results["ARIMA"] = arima_pred.values
        except Exception as e_fallback:
            print(f"[오류] 폴백 ARIMA마저 실패하여 Naive 예측으로 대체합니다: {e_fallback}")
            results["ARIMA"] = naive_pred.values

    # (C) 기존 Random Forest 모델 (기존 입력변수 적용)
    print(f"\n[{data_name}] 기존 Random Forest 학습 중...")
    scaler_orig = MinMaxScaler()
    X_train_orig = scaler_orig.fit_transform(train_df[ORIGINAL_FEATURES])
    X_test_orig = scaler_orig.transform(test_df[ORIGINAL_FEATURES])
    
    rf_model_orig = train_rf_model(X_train_orig, y_train)
    rf_pred_orig = predict_ai_model(rf_model_orig, X_test_orig)
    results["Existing RF"] = rf_pred_orig

    # (D) 추가 입력변수를 적용한 Random Forest 모델
    print(f"\n[{data_name}] 추가 입력변수 적용 Random Forest 학습 중...")
    scaler_ext = MinMaxScaler()
    X_train_ext = scaler_ext.fit_transform(train_df[EXTENDED_FEATURES])
    X_test_ext = scaler_ext.transform(test_df[EXTENDED_FEATURES])
    
    rf_model_ext = train_rf_model(X_train_ext, y_train)
    rf_pred_ext = predict_ai_model(rf_model_ext, X_test_ext)
    results["Extended RF"] = rf_pred_ext

    # 9. 모델 평가
    metrics_list = []
    for model_name, y_pred in results.items():
        metrics = calculate_metrics(y_test.values, y_pred)
        metrics["Model"] = model_name
        metrics_list.append(metrics)
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics[["Model", "RMSE", "MAE", "R2", "MAPE"]]
    
    # 평가 결과 저장
    Path("results").mkdir(parents=True, exist_ok=True)
    results_save_path = f"results/{data_name}_rf_extended_results.csv"
    df_metrics.to_csv(results_save_path, index=False, encoding="utf-8-sig")
    print(f"[평가 결과 저장 완료] -> {results_save_path}")
    print(df_metrics.to_string(index=False))

    # 10. 시각화 (예측 그래프)
    plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(test_dates, y_test.values, label="Actual Close", color="black", linewidth=2.0)
    plt.plot(test_dates, results["Benchmark"], label="Benchmark (Naive)", color="gray", linestyle=":", alpha=0.8)
    plt.plot(test_dates, results["ARIMA"], label="ARIMA Forecast", color="red", linestyle="-.", alpha=0.8)
    plt.plot(test_dates, results["Existing RF"], label="Existing RF (7 features)", color="orange", linestyle="--", alpha=0.8)
    plt.plot(test_dates, results["Extended RF"], label="Extended RF (26+ features)", color="royalblue", linestyle="-", alpha=0.9)
    
    plt.title(f"Model Predictions Comparison - {data_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    
    # Save graph
    Path("figures").mkdir(parents=True, exist_ok=True)
    figure_save_path = f"figures/{data_name}_rf_extended_prediction.png"
    plt.savefig(figure_save_path, bbox_inches="tight")
    plt.close()
    print(f"[예측 그래프 저장 완료] -> {figure_save_path}")
    
    return df_metrics
