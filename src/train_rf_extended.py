from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

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
from src.evaluate import regression_metrics, direction_accuracy
from src.ai_model import train_ann_model, predict_ai_model
from sklearn.ensemble import RandomForestRegressor

def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


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
    # 모멘텀/추세 지표
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    # 볼린저밴드 (절대 가격 + 비율 변수 함께 포함)
    "BB_upper", "BB_lower", "BB_width", "BB_percent",
    # 이동평균 (절대 가격 + 비율 변수 함께 포함)
    "MA_3", "MA_5", "MA_10", "MA_20",
    "Close_MA5_ratio", "Close_MA20_ratio", "MA5_MA20_gap",
    # 수익률 및 가격 구조
    "daily_return", "abs_return", "high_low_ratio", "open_close_ratio",
    # 변동성 (pct 기반, volatility_7은 ORIGINAL_FEATURES에 이미 포함됨)
    "pct_volatility_5", "pct_volatility_10",
    # 거래량 (절대값 + 비율 변수 함께 포함)
    "Volume_MA5", "Volume_MA20", "Volume_change", "Volume_ratio",
    # 거래대금 원본 + 로그 변환 함께 포함
    "log_trading_value",
    # 추가 기술적 지표
    "ATR_14",           # 변동성: 고가/저가/전일종가 모두 활용
    "OBV",              # 거래량-가격 방향 결합
    "Stoch_K", "Stoch_D",  # 스토캐스틱 오실레이터
    "lag_1_return", "lag_2_return", "lag_3_return",  # 단기 수익률 래그
    "MA3_return"        # 3일 수익률 이동평균 (단기 모멘텀 스무딩)
]

TARGET_COL = "target_next_close"


def calculate_metrics(y_true, y_pred, model_name: str):
    """
    공통 evaluate.py 함수를 사용해 RMSE, MAE, MAPE, MBE, R², 방향성 정확도를 계산합니다.
    """
    metrics = regression_metrics(y_true, y_pred)
    dir_acc = direction_accuracy(y_true, y_pred)
    metrics["direction_accuracy"] = dir_acc
    metrics["Model"] = model_name
    return metrics


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
    rf_pred_orig = rf_model_orig.predict(X_test_orig)
    results["Existing RF"] = rf_pred_orig

    # 8-D. 추가 입력변수를 적용한 Random Forest 모델
    print(f"\n[{data_name}] 추가 입력변수 적용 Random Forest 학습 중...")
    scaler_ext = MinMaxScaler()
    X_train_ext = scaler_ext.fit_transform(train_df[EXTENDED_FEATURES])
    X_test_ext = scaler_ext.transform(test_df[EXTENDED_FEATURES])
    
    rf_model_ext = train_rf_model(X_train_ext, y_train)
    rf_pred_ext = rf_model_ext.predict(X_test_ext)
    results["Extended RF"] = rf_pred_ext

    # 8-E. 피처 중요도 시각화 (Extended RF)
    importances = rf_model_ext.feature_importances_
    feat_series = pd.Series(importances, index=EXTENDED_FEATURES).sort_values(ascending=True)
    top_n = min(20, len(feat_series))
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    feat_series.tail(top_n).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance (Extended RF) - {data_name.replace('_', ' ').title()}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=11)
    ax.grid(True, alpha=0.3)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fi_path = f"results/figures/{data_name}_feature_importance.png"
    plt.tight_layout()
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"[피처 중요도 그래프 저장 완료] -> {fi_path}")

    # 8-F. 확장 피처를 적용한 ANN 모델
    print(f"\n[{data_name}] 확장 피처 ANN 학습 중...")
    # 확장 RF와 동일한 스케일러를 공유하여 스케일링 합니다.
    ann_model_ext = train_ann_model(X_train_ext, y_train.values)
    ann_pred_ext = predict_ai_model(ann_model_ext, X_test_ext)
    results["Extended ANN"] = ann_pred_ext

    # 9. 모델 평가 (공통 evaluate.py 함수로 통일)
    print(f"\n{'='*60}")
    print(f"[{data_name}] 모델별 평가 결과")
    metrics_list = []
    for model_name, y_pred in results.items():
        print(f"\n--- {model_name} ---")
        metrics = calculate_metrics(y_test.values, y_pred, model_name)
        metrics_list.append(metrics)
    
    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics[["Model", "rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]]
    
    # 평가 결과 저장 (results/metrics/ 로 경로 통일)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    results_save_path = f"results/metrics/{data_name}_ai_extended_results.csv"
    df_metrics.to_csv(results_save_path, index=False, encoding="utf-8-sig")
    print(f"[평가 결과 저장 완료] -> {results_save_path}")
    print(df_metrics.to_string(index=False))

    # 10. 시각화 (예측 그래프)
    plt.figure(figsize=(14, 7), dpi=100)
    plt.plot(test_dates, y_test.values, label="Actual Close", color="black", linewidth=2.0)
    plt.plot(test_dates, results["Benchmark"], label="Benchmark (Naive)", color="gray", linestyle=":", alpha=0.8)
    plt.plot(test_dates, results["ARIMA"], label="ARIMA Forecast", color="red", linestyle="-.", alpha=0.8)
    plt.plot(test_dates, results["Existing RF"], label="Existing RF (7 features)", color="orange", linestyle="--", alpha=0.8)
    plt.plot(test_dates, results["Extended RF"], label="Extended RF (30+ features)", color="royalblue", linestyle="-", alpha=0.9)
    plt.plot(test_dates, results["Extended ANN"], label="Extended ANN (30+ features)", color="forestgreen", linestyle="-", alpha=0.9)
    
    plt.title(f"Model Predictions Comparison - {data_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)

    # 예측 그래프 저장 경로 results/figures/ 로 통일
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    figure_save_path = f"results/figures/{data_name}_ai_extended_prediction.png"
    plt.savefig(figure_save_path, bbox_inches="tight")
    plt.close()
    print(f"[예측 그래프 저장 완료] -> {figure_save_path}")
    
    return df_metrics
