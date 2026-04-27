from pathlib import Path
import os
import pandas as pd
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
from src.split import split_time_series
from src.ai_model import train_rf_model, train_ann_model, predict_ai_model
from src.evaluate import evaluate_and_plot


FEATURE_COLS = [
    "Volume",
    "ma_7",
    "ma_14",
    "ma_21",
    "volatility_7",
    "hl_diff",
    "oc_diff"
]

TARGET_COL = "target_next_close"


def train_ai_pipeline(data_name: str, file_path: str):
    print("=" * 60)
    print(f"[AI 모델] {data_name} - RF & ANN 파이프라인 시작")
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

    # 6. 입력/정답 분리 및 스케일링
    X_train_raw = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_test_raw = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    y_test_series = pd.Series(y_test.values, index=test_df["Date"], name="Actual")

    # 7. Random Forest 모델 학습 및 평가
    print(f"\n[{data_name}] Random Forest 학습 중...")
    rf_model = train_rf_model(X_train, y_train)
    rf_pred = predict_ai_model(rf_model, X_test)
    rf_pred_series = pd.Series(rf_pred, index=test_df["Date"], name="Predicted")
    
    rf_metrics = evaluate_and_plot(
        y_true=y_test_series,
        y_pred=rf_pred_series,
        title=f"RF Forecast vs Actual ({data_name})",
        save_dir="results/figures",
        data_name=f"rf_{data_name}",
    )
    rf_metrics["model"] = "Random Forest"

    # 8. ANN 모델 학습 및 평가
    print(f"\n[{data_name}] ANN 학습 중...")
    ann_model = train_ann_model(X_train, y_train)
    ann_pred = predict_ai_model(ann_model, X_test)
    ann_pred_series = pd.Series(ann_pred, index=test_df["Date"], name="Predicted")

    ann_metrics = evaluate_and_plot(
        y_true=y_test_series,
        y_pred=ann_pred_series,
        title=f"ANN Forecast vs Actual ({data_name})",
        save_dir="results/figures",
        data_name=f"ann_{data_name}",
    )
    ann_metrics["model"] = "ANN"

    print(f"[완료] {data_name} 파이프라인 완료")
    return rf_metrics, ann_metrics


def train_all_ai_models(raw_data_dir="data/rawdata"):
    target_companies = [
        "hanwha_aerospace", "lig_nex1", "snt_dynamics", "firstec",
        "samsung_electronics", "sk_hynix", "wonik_ips", "ia",
        "rtx", "aerovironment", "draganfly", "nvidia", "axt", "maxlinear"
    ]
    
    csv_files = [f for f in os.listdir(raw_data_dir) if f.endswith(".csv")]
    
    filtered_files = []
    for f in csv_files:
        data_name = f.replace("_5y.csv", "").replace(".csv", "")
        if data_name in target_companies:
            filtered_files.append(f)

    all_metrics = []

    print(f"총 {len(filtered_files)}개 종목 RF & ANN 파이프라인 실행 시작")

    for f in filtered_files:
        data_name = f.replace("_5y.csv", "").replace(".csv", "")
        file_path = os.path.join(raw_data_dir, f)

        try:
            rf_metrics, ann_metrics = train_ai_pipeline(data_name, file_path)
            rf_metrics["data_name"] = data_name
            ann_metrics["data_name"] = data_name
            all_metrics.append(rf_metrics)
            all_metrics.append(ann_metrics)
        except Exception as e:
            print(f"[{data_name}] 오류 발생: {e}")

    df_metrics = pd.DataFrame(all_metrics)

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    save_path = "results/metrics/ai_models_summary.csv"
    df_metrics.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print(f"전체 AI 모델 결과 저장 완료: {save_path}")
    print("=" * 60)

    return df_metrics
