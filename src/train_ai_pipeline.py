from pathlib import Path
import os
import pandas as pd

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
from src.split import split_time_series
from src.ai_model import train_ai_model, predict_ai_model
from src.evaluate import evaluate_and_plot


FEATURE_COLS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "ma_5",
    "ma_20",
    "ma_60",
    "simple_return",
    "log_return",
    "volatility_5",
    "volatility_20",
    "hl_diff",
    "oc_diff",
    "close_ma5_gap",
    "close_ma20_gap",
]

TARGET_COL = "target_next_close"


def train_ai_pipeline(data_name: str, file_path: str):
    print("=" * 60)
    print(f"[AI 모델] {data_name} - Random Forest 파이프라인 시작")
    print("=" * 60)

    # 1. 데이터 불러오기
    df = load_price_data(file_path)

    # 2. 전처리 및 파생변수 생성
    df = add_return_features(df, price_col="Close")
    df = add_moving_averages(df, price_col="Close")
    df = add_volatility(df, return_col="log_return")
    df = add_price_structure_features(df)

    # 3. 타깃 생성 (내일 종가)
    df = add_target_next_close(df, price_col="Close")

    # 4. 결측 제거
    df = drop_missing_rows(df)

    # 5. 데이터 분할
    train_df, val_df, test_df = split_time_series(df, train_ratio=0.7, val_ratio=0.15)

    print(f"[데이터 분할 완료]")
    print(f"  Train: {train_df.shape}")
    print(f"  Valid: {val_df.shape}")
    print(f"  Test : {test_df.shape}")

    # 6. 입력/정답 분리
    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # 7. 모델 학습
    model = train_ai_model(X_train, y_train)

    # 8. 예측
    y_pred = predict_ai_model(model, X_test)

    # 9. 평가용 시리즈 변환
    y_test_series = pd.Series(y_test.values, index=test_df["Date"], name="Actual")
    y_pred_series = pd.Series(y_pred, index=test_df["Date"], name="Predicted")

    # 10. 평가 및 그래프 저장
    metrics = evaluate_and_plot(
        y_true=y_test_series,
        y_pred=y_pred_series,
        title=f"Random Forest Forecast vs Actual ({data_name})",
        save_dir="results/figures",
        data_name=f"rf_{data_name}",
    )

    print(f"[완료] {data_name} Random Forest 파이프라인 완료")
    return model, metrics


def train_all_ai_models(raw_data_dir="data/rawdata"):
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv")]
    all_metrics = []

    print(f"총 {len(csv_files)}개 종목 RF 파이프라인 실행 시작")

    for f in csv_files:
        data_name = f.replace("_5y.csv", "").replace(".csv", "")
        file_path = os.path.join(raw_data_dir, f)

        try:
            _, metrics = train_ai_pipeline(data_name, file_path)
            metrics["data_name"] = data_name
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[{data_name}] 오류 발생: {e}")

    df_metrics = pd.DataFrame(all_metrics)

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    save_path = "results/metrics/random_forest_summary.csv"
    df_metrics.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"전체 RF 결과 저장 완료: {save_path}")
    print("=" * 60)

    return df_metrics
