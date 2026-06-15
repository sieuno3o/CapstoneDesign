from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE_DIR = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, accuracy_score

from src.data_loader import load_price_data
from src.preprocess import (
    add_return_features,
    add_target_next_close,
    add_target_next_return,
    drop_missing_rows,
)
from src.feature_engineering import (
    add_moving_averages,
    add_volatility,
    add_price_structure_features,
)
from src.feature_engineering_extended import add_extended_features
from src.ai_model import train_ann_model, predict_ai_model
from src.classical_model import train_classical_model, predict_classical_model

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ORIGINAL_FEATURES = [
    "Volume",
    "ma_7",
    "ma_14",
    "ma_21",
    "volatility_7",
    "hl_diff",
    "oc_diff"
]

EXTENDED_FEATURES = [
    "MA_3",
    "MA_5",
    "MA_10",
    "MA_20",
    "BB_upper",
    "OBV",
    "ATR_14",
    "Volume",
    "hl_diff",
    "volatility_7",
]

TARGET_COL = "target_next_return"
TARGET_PRICE_COL = "target_next_close"

SELECTED_STOCKS = {
    "samsung_electronics": str(BASE_DIR / "data/raw/samsung_electronics_5y.csv"),
    "sk_hynix":            str(BASE_DIR / "data/raw/sk_hynix_5y.csv"),
    "wonik_ips":           str(BASE_DIR / "data/raw/wonik_ips_5y.csv"),
    "dongjin_semichem":    str(BASE_DIR / "data/raw/dongjin_semichem_5y.csv"),
    "hanwha_aerospace":    str(BASE_DIR / "data/raw/hanwha_aerospace_5y.csv"),
    "lig_nex1":            str(BASE_DIR / "data/raw/lig_nex1_5y.csv"),
    "snt_dynamics":        str(BASE_DIR / "data/raw/snt_dynamics_5y.csv"),
    "firstec":             str(BASE_DIR / "data/raw/firstec_5y.csv"),
}

FINAL_INPUT_PATH = str(BASE_DIR / "data/sentiment/final_input.csv")
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
FINAL_MODEL_DIR = FIGURES_DIR / "final_model"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_regression(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mbe = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1:
        true_dir = (y_true[1:] > y_true[:-1]).astype(int)
        pred_dir = (y_pred[1:] > y_true[:-1]).astype(int)
        direction_accuracy = accuracy_score(true_dir, pred_dir)
    else:
        direction_accuracy = np.nan

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "mbe": float(mbe),
        "r2": float(r2),
        "direction_accuracy": float(direction_accuracy),
    }


def prepare_stock_dataframe(file_path: str) -> pd.DataFrame:
    df = load_price_data(file_path)
    df = add_return_features(df, price_col="Close")
    df = add_moving_averages(df, price_col="Close")
    df = add_volatility(df, return_col="log_return")
    df = add_price_structure_features(df)
    df = add_extended_features(df)
    df = add_target_next_return(df, price_col="Close")
    df = add_target_next_close(df, price_col="Close")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = drop_missing_rows(df)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_final_input(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns or "K_NSI_Short" not in df.columns or "K_NSI_Long" not in df.columns:
        raise ValueError("final_input.csv는 date, K_NSI_Short, K_NSI_Long 컬럼을 포함해야 합니다.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).reset_index(drop=True)


def train_baseline_models(train_df: pd.DataFrame):
    y_train = train_df[TARGET_COL].values

    # Existing RF (7 features)
    X_train_rf_orig = train_df[ORIGINAL_FEATURES].values
    scaler_rf_orig = MinMaxScaler()
    X_train_rf_orig_scaled = scaler_rf_orig.fit_transform(X_train_rf_orig)
    rf_orig = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_orig.fit(X_train_rf_orig_scaled, y_train)

    # Existing ANN (7 features)
    val_split_orig = max(int(len(X_train_rf_orig_scaled) * 0.85), 1)
    ann_orig = train_ann_model(
        X_train_rf_orig_scaled[:val_split_orig],
        y_train[:val_split_orig],
        X_val=X_train_rf_orig_scaled[val_split_orig:] if len(X_train_rf_orig_scaled[val_split_orig:]) > 0 else None,
        y_val=y_train[val_split_orig:] if len(y_train[val_split_orig:]) > 0 else None,
    )

    # Extended RF (10 features)
    X_train_rf_ext = train_df[EXTENDED_FEATURES].values
    scaler_rf_ext = MinMaxScaler()
    X_train_rf_ext_scaled = scaler_rf_ext.fit_transform(X_train_rf_ext)
    rf_ext = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_ext.fit(X_train_rf_ext_scaled, y_train)

    # Extended ANN (10 features)
    val_split_ext = max(int(len(X_train_rf_ext_scaled) * 0.85), 1)
    ann_ext = train_ann_model(
        X_train_rf_ext_scaled[:val_split_ext],
        y_train[:val_split_ext],
        X_val=X_train_rf_ext_scaled[val_split_ext:] if len(X_train_rf_ext_scaled[val_split_ext:]) > 0 else None,
        y_val=y_train[val_split_ext:] if len(y_train[val_split_ext:]) > 0 else None,
    )

    return rf_orig, scaler_rf_orig, ann_orig, rf_ext, scaler_rf_ext, ann_ext


def predict_baseline_price(model, scaler, df: pd.DataFrame, features_list):
    X = scaler.transform(df[features_list].values)
    y_pred_return = model.predict(X)
    y_pred_return = np.asarray(y_pred_return)
    if y_pred_return.ndim > 1:
        y_pred_return = y_pred_return.ravel()
    return df["Close"].values * np.exp(y_pred_return)


def train_correction_model(pred_price: np.ndarray, k_short: np.ndarray, k_long: np.ndarray, y_true: np.ndarray):
    X = np.column_stack([pred_price, k_short, k_long])
    corr_model = LinearRegression()
    corr_model.fit(X, y_true)
    return corr_model


def run_two_step_for_stock(stock_name: str, stock_path: str, final_input: pd.DataFrame):
    print("=" * 80)
    print(f"[2-Step Correction] {stock_name} 처리 시작")
    print("=" * 80)

    df = prepare_stock_dataframe(stock_path)
    final_input_dates = final_input["date"].min(), final_input["date"].max()
    df_test = df[(df["Date"] >= final_input_dates[0]) & (df["Date"] <= final_input_dates[1])].copy()
    if df_test.empty:
        raise ValueError(f"{stock_name}의 주가 데이터에서 final_input 날짜 범위와 겹치는 구간이 없습니다.")

    df_test = df_test.merge(final_input, left_on="Date", right_on="date", how="inner")
    if len(df_test) < 10:
        raise ValueError(f"{stock_name}의 final_input 결합 후 데이터가 충분하지 않습니다. 행 개수: {len(df_test)}")

    train_df = df[df["Date"] < df_test["Date"].min()].copy()
    if len(train_df) < 50:
        raise ValueError(f"{stock_name}의 baseline 학습 데이터가 부족합니다. 행 개수: {len(train_df)}")

    rf_orig, scaler_rf_orig, ann_orig, rf_ext, scaler_rf_ext, ann_ext = train_baseline_models(train_df)

    df_test = df_test.sort_values("Date").reset_index(drop=True)
    df_test["rf_orig_price"] = predict_baseline_price(rf_orig, scaler_rf_orig, df_test, ORIGINAL_FEATURES)
    df_test["rf_ext_price"] = predict_baseline_price(rf_ext, scaler_rf_ext, df_test, EXTENDED_FEATURES)
    df_test["ann_orig_price"] = predict_baseline_price(ann_orig, scaler_rf_orig, df_test, ORIGINAL_FEATURES)
    df_test["ann_ext_price"] = predict_baseline_price(ann_ext, scaler_rf_ext, df_test, EXTENDED_FEATURES)

    # ARIMA Baseline 예측
    try:
        arima_fitted = train_classical_model(train_df["Close"])
        arima_pred_prices = predict_classical_model(arima_fitted, steps=len(df_test))
        df_test["arima_baseline_price"] = np.asarray(arima_pred_prices)
    except Exception as e:
        print(f"[경고] {stock_name} ARIMA 학습 실패: {e}")
        df_test["arima_baseline_price"] = df_test["Close"].values

    # 최근 2달 검증 구간 내부에서 correction train / eval split
    split_index = max(int(len(df_test) * 0.6), 1)
    correction_train = df_test.iloc[:split_index].copy()
    correction_eval = df_test.iloc[split_index:].copy()

    if len(correction_eval) < 5:
        raise ValueError(f"{stock_name} 검증 구간이 너무 작습니다. 현재 행 수: {len(correction_eval)}")

    y_eval = correction_eval[TARGET_PRICE_COL].values

    # --- 8번 모델: Benchmark + 심리지수 결합모델 (선형 회귀 보정) ---
    bench_corr_model = train_correction_model(
        correction_train["Close"].values,
        correction_train["K_NSI_Short"].values,
        correction_train["K_NSI_Long"].values,
        correction_train[TARGET_PRICE_COL].values,
    )
    bench_eval_pred = bench_corr_model.predict(
        np.column_stack([
            correction_eval["Close"].values,
            correction_eval["K_NSI_Short"].values,
            correction_eval["K_NSI_Long"].values,
        ])
    )
    bench_baseline_eval = correction_eval["Close"].values

    rf_orig_eval = correction_eval["rf_orig_price"].values
    rf_ext_eval = correction_eval["rf_ext_price"].values
    ann_orig_eval = correction_eval["ann_orig_price"].values
    ann_ext_eval = correction_eval["ann_ext_price"].values
    arima_baseline_eval = correction_eval["arima_baseline_price"].values

    # --- ARIMA + 심리지수 결합모델 (선형 회귀 보정) ---
    try:
        arima_corr_model = train_correction_model(
            correction_train["arima_baseline_price"].values,
            correction_train["K_NSI_Short"].values,
            correction_train["K_NSI_Long"].values,
            correction_train[TARGET_PRICE_COL].values,
        )
        arima_eval_pred = arima_corr_model.predict(
            np.column_stack([
                correction_eval["arima_baseline_price"].values,
                correction_eval["K_NSI_Short"].values,
                correction_eval["K_NSI_Long"].values,
            ])
        )
    except Exception as e:
        print(f"[경고] {stock_name} ARIMA 보정 모델 학습 실패: {e}")
        arima_eval_pred = arima_baseline_eval

    # --- Existing RF + 심리지수 결합모델 (선형 회귀 보정) ---
    try:
        rf_orig_corr_model = train_correction_model(
            correction_train["rf_orig_price"].values,
            correction_train["K_NSI_Short"].values,
            correction_train["K_NSI_Long"].values,
            correction_train[TARGET_PRICE_COL].values,
        )
        rf_orig_eval_pred = rf_orig_corr_model.predict(
            np.column_stack([
                correction_eval["rf_orig_price"].values,
                correction_eval["K_NSI_Short"].values,
                correction_eval["K_NSI_Long"].values,
            ])
        )
    except Exception as e:
        print(f"[경고] {stock_name} Existing RF 보정 모델 학습 실패: {e}")
        rf_orig_eval_pred = rf_orig_eval

    # --- Existing ANN + 심리지수 결합모델 (선형 회귀 보정) ---
    try:
        ann_orig_corr_model = train_correction_model(
            correction_train["ann_orig_price"].values,
            correction_train["K_NSI_Short"].values,
            correction_train["K_NSI_Long"].values,
            correction_train[TARGET_PRICE_COL].values,
        )
        ann_orig_eval_pred = ann_orig_corr_model.predict(
            np.column_stack([
                correction_eval["ann_orig_price"].values,
                correction_eval["K_NSI_Short"].values,
                correction_eval["K_NSI_Long"].values,
            ])
        )
    except Exception as e:
        print(f"[경고] {stock_name} Existing ANN 보정 모델 학습 실패: {e}")
        ann_orig_eval_pred = ann_orig_eval

    eval_date_index = correction_eval["Date"].dt.strftime("%Y-%m-%d")

    metrics = []
    metrics.append({"Company": stock_name, "Model": "Benchmark", **evaluate_regression(y_eval, bench_baseline_eval)})
    metrics.append({"Company": stock_name, "Model": "Benchmark + 심리지수 결합모델", **evaluate_regression(y_eval, bench_eval_pred)})
    metrics.append({"Company": stock_name, "Model": "ARIMA", **evaluate_regression(y_eval, arima_baseline_eval)})
    metrics.append({"Company": stock_name, "Model": "ARIMA + 심리지수 결합모델", **evaluate_regression(y_eval, arima_eval_pred)})
    metrics.append({"Company": stock_name, "Model": "Existing RF", **evaluate_regression(y_eval, rf_orig_eval)})
    metrics.append({"Company": stock_name, "Model": "Existing RF + 심리지수 결합모델", **evaluate_regression(y_eval, rf_orig_eval_pred)})
    metrics.append({"Company": stock_name, "Model": "Extended RF", **evaluate_regression(y_eval, rf_ext_eval)})
    metrics.append({"Company": stock_name, "Model": "Existing ANN", **evaluate_regression(y_eval, ann_orig_eval)})
    metrics.append({"Company": stock_name, "Model": "Existing ANN + 심리지수 결합모델", **evaluate_regression(y_eval, ann_orig_eval_pred)})
    metrics.append({"Company": stock_name, "Model": "Extended ANN", **evaluate_regression(y_eval, ann_ext_eval)})

    df_metrics = pd.DataFrame(metrics)
    metrics_path = METRICS_DIR / f"{stock_name}_2step_correction_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"[저장] {metrics_path}")

    # 시각화 (5단 구성으로 변경)
    fig, axes = plt.subplots(5, 1, figsize=(14, 28), dpi=120)
    
    # 0. Benchmark vs Corrected
    axes[0].plot(eval_date_index, y_eval, label="Actual Close", color="black", linewidth=2.5)
    axes[0].plot(eval_date_index, bench_baseline_eval, label="Benchmark", color="dimgray", linestyle="--")
    axes[0].plot(eval_date_index, bench_eval_pred, label="Benchmark + 심리지수 결합모델", color="purple", linestyle="-")
    axes[0].set_title(f"{stock_name} - Benchmark vs Benchmark + 심리지수 결합모델")
    axes[0].set_ylabel("Price (KRW)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 1. ARIMA vs Corrected
    axes[1].plot(eval_date_index, y_eval, label="Actual Close", color="black", linewidth=2.5)
    axes[1].plot(eval_date_index, arima_baseline_eval, label="ARIMA", color="teal", linestyle="--")
    axes[1].plot(eval_date_index, arima_eval_pred, label="ARIMA + 심리지수 결합모델", color="orchid", linestyle="-")
    axes[1].set_title(f"{stock_name} - ARIMA vs ARIMA + 심리지수 결합모델")
    axes[1].set_ylabel("Price (KRW)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 2. RF Models
    axes[2].plot(eval_date_index, y_eval, label="Actual Close", color="black", linewidth=2.5)
    axes[2].plot(eval_date_index, rf_orig_eval, label="Existing RF", color="royalblue", linestyle="--")
    axes[2].plot(eval_date_index, rf_orig_eval_pred, label="Existing RF + 심리지수 결합모델", color="forestgreen", linestyle="-")
    axes[2].set_title(f"{stock_name} - Existing RF vs Existing RF + 심리지수 결합모델")
    axes[2].set_ylabel("Price (KRW)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # 3. ANN Models
    axes[3].plot(eval_date_index, y_eval, label="Actual Close", color="black", linewidth=2.5)
    axes[3].plot(eval_date_index, ann_orig_eval, label="Existing ANN", color="darkorange", linestyle="--")
    axes[3].plot(eval_date_index, ann_orig_eval_pred, label="Existing ANN + 심리지수 결합모델", color="chocolate", linestyle="-")
    axes[3].set_title(f"{stock_name} - Existing ANN vs Existing ANN + 심리지수 결합모델")
    axes[3].set_ylabel("Price (KRW)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    # 4. RMSE Comparison
    bar_names = [
        "Benchmark", "Benchmark + 심리지수 결합모델",
        "ARIMA", "ARIMA + 심리지수 결합모델",
        "Existing RF", "Existing RF + 심리지수 결합모델",
        "Extended RF", "Existing ANN", "Existing ANN + 심리지수 결합모델", "Extended ANN"
    ]
    bar_values = [df_metrics.loc[df_metrics["Model"] == name, "rmse"].values[0] for name in bar_names]
    axes[4].bar(bar_names, bar_values, color=["dimgray", "purple", "teal", "orchid", "royalblue", "forestgreen", "darkgreen", "darkorange", "chocolate", "crimson"])
    axes[4].set_title(f"{stock_name} - RMSE Comparison")
    axes[4].set_ylabel("RMSE (KRW)")
    for i, val in enumerate(bar_values):
        axes[4].text(i, val + max(bar_values) * 0.01, f"{val:,.0f}", ha="center", va="bottom", fontsize=10)
    axes[4].grid(True, axis="y", alpha=0.3)

    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    figure_path = FINAL_MODEL_DIR / f"{stock_name}_2step_correction.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {figure_path}")

    return df_metrics


def plot_rmse_summary(df_summary: pd.DataFrame):
    pivot = df_summary.pivot(index="Company", columns="Model", values="rmse")
    pivot = pivot[[
        "Benchmark", "Benchmark + 심리지수 결합모델",
        "ARIMA", "ARIMA + 심리지수 결합모델",
        "Existing RF", "Existing RF + 심리지수 결합모델",
        "Extended RF", "Existing ANN", "Existing ANN + 심리지수 결합모델", "Extended ANN"
    ]]

    fig, ax = plt.subplots(figsize=(14, 8), dpi=120)
    pivot.plot(kind="bar", ax=ax, color=["dimgray", "purple", "teal", "orchid", "royalblue", "forestgreen", "darkgreen", "darkorange", "chocolate", "crimson"], width=0.85)
    ax.set_title("2-Step Correction RMSE Comparison Across Stocks")
    ax.set_ylabel("RMSE (KRW)")
    ax.set_xlabel("Company")
    ax.legend(title="Model")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    summary_path = FINAL_MODEL_DIR / "2step_correction_rmse_summary.png"
    fig.savefig(summary_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[저장] {summary_path}")


def main():
    final_input = load_final_input(FINAL_INPUT_PATH)
    all_metrics = []

    for stock_name, stock_path in SELECTED_STOCKS.items():
        try:
            df_metrics = run_two_step_for_stock(stock_name, stock_path, final_input)
            all_metrics.append(df_metrics)
        except Exception as exc:
            print(f"[오류] {stock_name} 처리 중 오류 발생: {exc}")

    if not all_metrics:
        print("[종료] 처리 가능한 종목이 없습니다.")
        return

    df_summary = pd.concat(all_metrics, ignore_index=True)
    summary_path = METRICS_DIR / "2step_psychological_correction_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[저장] {summary_path}")

    plot_rmse_summary(df_summary)

    print("\n=== 최종 요약 ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
