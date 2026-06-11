from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE_DIR = Path(__file__).resolve().parents[2]
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
_korean_fonts = [f.name for f in fm.fontManager.ttflist if "Gothic" in f.name or "Nanum" in f.name or "Apple" in f.name]
if _korean_fonts:
    matplotlib.rc("font", family=_korean_fonts[0])
matplotlib.rcParams["axes.unicode_minus"] = False

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
from src.split import split_time_series
from src.modeling import find_best_arima_model
from src.evaluate import regression_metrics, direction_accuracy
from src.ai_model import train_ann_model, predict_ai_model

# ── Extended Features (10개) ────────────────────────────────────────────────
EXTENDED_FEATURES = [
    "MA_3",         # 3일 이동평균
    "MA_5",         # 5일 이동평균
    "MA_10",        # 10일 이동평균
    "MA_20",        # 20일 이동평균
    "BB_upper",     # 볼린저밴드 상단
    "OBV",          # 온밸런스 볼륨
    "ATR_14",       # Average True Range
    "Volume",       # 거래량
    "hl_diff",      # 고가 - 저가
    "volatility_7", # 7일 수익률 표준편차
]

TARGET_COL = "target_next_return"           # 수익률 타깃
TARGET_PRICE_COL = "target_next_close"      # 종가 타깃 (역변환 검증용)

# ── 선택한 8개 기업 ─────────────────────────────────────────────────────────
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


def train_rf_model(X_train, y_train):
    """Random Forest 모델 학습"""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def calculate_metrics(y_true, y_pred, model_name: str):
    """평가 지표 계산"""
    metrics = regression_metrics(y_true, y_pred)
    dir_acc = direction_accuracy(y_true, y_pred)
    metrics["direction_accuracy"] = dir_acc
    metrics["Model"] = model_name
    return metrics


def train_rf_extended_single_stock(stock_name: str, file_path: str):
    """
    단일 종목에 대해 Extended RF 파이프라인 실행
    """
    print("=" * 80)
    print(f"[선택 기업 Extended RF 파이프라인] {stock_name} 실행 시작")
    print("=" * 80)

    # 1. 데이터 불러오기
    df = load_price_data(file_path)

    # 2. 기존 파생변수 생성
    df = add_return_features(df, price_col="Close")
    df = add_moving_averages(df, price_col="Close")
    df = add_volatility(df, return_col="log_return")
    df = add_price_structure_features(df)

    # 3. 추가 입력변수 생성
    df = add_extended_features(df)

    # 4. 타깃 생성 (다음날 수익률)
    df = add_target_next_return(df, price_col="Close")
    df = add_target_next_close(df, price_col="Close")

    # 5. 전처리 완료된 데이터 저장
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    processed_save_path = f"data/processed/{stock_name}_features_extended.csv"
    df.to_csv(processed_save_path, index=False, encoding="utf-8-sig")
    print(f"[전처리 데이터 저장 완료] -> {processed_save_path}")

    # 6. 결측 및 무한대 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    df = drop_missing_rows(df)

    # 7. 데이터 분할 (70% Train, 15% Val, 15% Test)
    train_df, val_df, test_df = split_time_series(df, train_ratio=0.7, val_ratio=0.15)
    print(f"[데이터 분할 완료] Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    # 테스트 데이터 준비
    test_dates  = pd.to_datetime(test_df["Date"])
    today_close = test_df["Close"].values
    y_train_r   = train_df[TARGET_COL]
    y_val_r     = val_df[TARGET_COL]
    y_test_r    = test_df[TARGET_COL]
    y_test_price = test_df[TARGET_PRICE_COL].values

    def ret_to_price(ret_pred):
        """예측 수익률 → 예측 종가 변환"""
        return today_close * (1 + np.array(ret_pred))

    # 8. 모델 예측
    results_price = {}
    results_return = {}

    # (A) Naive 벤치마크
    naive_ret  = np.zeros(len(test_df))
    naive_price = today_close.copy()
    results_return["Benchmark"] = naive_ret
    results_price["Benchmark"]  = naive_price

    # (B) ARIMA
    print(f"\n[{stock_name}] ARIMA 베이스라인 학습 중...")
    try:
        arima_model = find_best_arima_model(train_df["Close"])
        arima_price_pred = arima_model.predict(n_periods=len(test_df))
        arima_price_pred = arima_price_pred if isinstance(arima_price_pred, np.ndarray) else arima_price_pred.values
    except Exception as e:
        print(f"[오류] Auto ARIMA 탐색 실패, ARIMA(1,1,1)로 폴백합니다. ({e})")
        from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
        try:
            basic_model = StatsARIMA(train_df["Close"], order=(1,1,1))
            fitted = basic_model.fit()
            arima_price_pred = fitted.forecast(steps=len(test_df)).values
        except Exception as e2:
            print(f"[오류] 폴백 ARIMA 실패, Naive로 대체합니다: {e2}")
            arima_price_pred = naive_price.copy()
    
    arima_ret = (arima_price_pred - today_close) / today_close
    results_return["ARIMA"] = arima_ret
    results_price["ARIMA"]  = arima_price_pred

    # (C) Extended RF (10개 변수, 수익률 예측)
    print(f"\n[{stock_name}] Extended Random Forest 학습 중...")
    scaler_ext  = MinMaxScaler()
    X_train_ext = scaler_ext.fit_transform(train_df[EXTENDED_FEATURES])
    X_val_ext   = scaler_ext.transform(val_df[EXTENDED_FEATURES])
    X_test_ext  = scaler_ext.transform(test_df[EXTENDED_FEATURES])
    rf_ext = train_rf_model(X_train_ext, y_train_r)
    rf_ext_ret  = rf_ext.predict(X_test_ext)
    results_return["Extended RF"] = rf_ext_ret
    results_price["Extended RF"]  = ret_to_price(rf_ext_ret)

    # (D) Feature Importance 시각화 (Extended RF)
    importances = rf_ext.feature_importances_
    feat_series = pd.Series(importances, index=EXTENDED_FEATURES).sort_values(ascending=True)
    top_n = min(20, len(feat_series))
    fig_fi, ax_fi = plt.subplots(figsize=(10, 8), dpi=100)
    feat_series.tail(top_n).plot(kind="barh", ax=ax_fi, color="steelblue")
    ax_fi.set_title(f"Feature Importance (Extended RF) - {stock_name.replace('_', ' ').title()}",
                    fontsize=13, fontweight="bold")
    ax_fi.set_xlabel("Importance", fontsize=11)
    ax_fi.grid(True, alpha=0.3)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fi_path = f"results/figures/{stock_name}_feature_importance.png"
    plt.tight_layout()
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"[피처 중요도 그래프 저장 완료] -> {fi_path}")

    # (E) Extended ANN (수익률 예측)
    print(f"\n[{stock_name}] Extended ANN 학습 중...")
    ann_ext = train_ann_model(
        X_train_ext, y_train_r.values,
        X_val=X_val_ext, y_val=y_val_r.values
    )
    ann_ext_ret  = predict_ai_model(ann_ext, X_test_ext)
    results_return["Extended ANN"] = ann_ext_ret
    results_price["Extended ANN"]  = ret_to_price(ann_ext_ret)

    # 9. 모델 평가 — 종가 단위로 통일
    print(f"\n{'='*60}")
    print(f"[{stock_name}] 모델별 평가 결과 (종가 단위 RMSE)")
    metrics_list = []
    for model_name, price_pred in results_price.items():
        print(f"\n--- {model_name} ---")
        metrics = calculate_metrics(y_test_price, price_pred, model_name)
        metrics_list.append(metrics)

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics[["Model", "rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]]

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    results_save_path = f"results/metrics/{stock_name}_extended_results.csv"
    df_metrics.to_csv(results_save_path, index=False, encoding="utf-8-sig")
    print(f"[평가 결과 저장 완료] -> {results_save_path}")
    print(df_metrics.to_string(index=False))

    results = results_price

    # 10. 시각화
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # --- 모델별 RMSE 계산 ---
    rmse_dict = {}
    for model_name, y_pred in results.items():
        rmse_val = np.sqrt(np.mean((y_test_price - y_pred) ** 2))
        rmse_dict[model_name] = rmse_val

    # ── 그래프 1: 예측값 비교 ──────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7), dpi=120)

    # 실제 종가
    ax.plot(test_dates, y_test_price,
            label="Actual Close (실제 종가)",
            color="black", linewidth=2.5, zorder=5)

    # Naive 벤치마크
    ax.plot(test_dates, results["Benchmark"],
            label=f"Naive Benchmark (전일 종가)  RMSE={rmse_dict['Benchmark']:,.0f}",
            color="dimgray", linestyle=(0, (1, 1)), linewidth=1.8, alpha=0.85)

    # ARIMA
    ax.plot(test_dates, results["ARIMA"],
            label=f"ARIMA Forecast              RMSE={rmse_dict['ARIMA']:,.0f}",
            color="crimson", linestyle="-.", linewidth=1.6, alpha=0.85)

    # Extended RF
    ax.plot(test_dates, results["Extended RF"],
            label=f"Extended RF  (10 features)  RMSE={rmse_dict['Extended RF']:,.0f}",
            color="royalblue", linestyle="-", linewidth=2.0, alpha=0.9)

    # Extended ANN
    ax.plot(test_dates, results["Extended ANN"],
            label=f"Extended ANN (10 features)  RMSE={rmse_dict['Extended ANN']:,.0f}",
            color="forestgreen", linestyle="-", linewidth=2.0, alpha=0.9)

    ax.set_title(
        f"Model Predictions Comparison — {stock_name.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold", pad=14
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (KRW)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left",
              framealpha=0.85, edgecolor="gray",
              prop={"family": "monospace"})
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # 개선율 텍스트 박스
    note_text = (
        "[RMSE 개선율 기준]\n"
        f"Naive 대비 Extended RF  : "
        f"{(rmse_dict['Benchmark']-rmse_dict['Extended RF'])/rmse_dict['Benchmark']*100:+.2f}%\n"
        f"Naive 대비 Extended ANN : "
        f"{(rmse_dict['Benchmark']-rmse_dict['Extended ANN'])/rmse_dict['Benchmark']*100:+.2f}%"
    )
    ax.text(
        0.99, 0.02, note_text,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9)
    )

    fig.tight_layout()
    pred_graph_path = f"results/figures/{stock_name}_extended_prediction.png"
    fig.savefig(pred_graph_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[예측 비교 그래프 저장 완료] -> {pred_graph_path}")

    # ── 그래프 2: RMSE 막대 비교 ─────────────────────
    model_names = list(rmse_dict.keys())
    rmse_values = list(rmse_dict.values())
    bar_colors = ["dimgray", "crimson", "royalblue", "forestgreen"]
    hatches     = ["", "", "", ""]

    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=120)
    bars = ax2.bar(model_names, rmse_values,
                   color=bar_colors, hatch=hatches,
                   edgecolor="white", linewidth=0.8, alpha=0.88)

    # 막대 위에 수치 표시
    for bar, val in zip(bars, rmse_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_values) * 0.01,
            f"{val:,.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    # Naive 기준선
    naive_rmse = rmse_dict["Benchmark"]
    ax2.axhline(naive_rmse, color="dimgray", linestyle=":", linewidth=1.5,
                label=f"Naive RMSE = {naive_rmse:,.0f}")

    ax2.set_title(
        f"RMSE Comparison by Model — {stock_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold"
    )
    ax2.set_ylabel("RMSE (KRW)", fontsize=11)
    ax2.set_xlabel("Model", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    fig2.tight_layout()

    rmse_bar_path = f"results/figures/{stock_name}_rmse_comparison.png"
    fig2.savefig(rmse_bar_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"[RMSE 막대 비교 그래프 저장 완료] -> {rmse_bar_path}")

    return df_metrics


def main():
    """
    8개 선택 기업 모두에 대해 Extended RF 파이프라인 실행
    """
    print("\n" + "=" * 80)
    print("[ 선택 기업 Extended RF 파이프라인 ] 전체 실행 시작")
    print("=" * 80 + "\n")

    all_metrics = []

    for stock_name, file_path in SELECTED_STOCKS.items():
        try:
            df_metrics = train_rf_extended_single_stock(stock_name, file_path)
            df_metrics["Stock"] = stock_name
            all_metrics.append(df_metrics)
        except Exception as e:
            print(f"\n[오류] {stock_name} 처리 중 오류 발생: {e}")
            continue

    # 전체 결과 통합
    if all_metrics:
        df_all_metrics = pd.concat(all_metrics, ignore_index=True)
        Path("results").mkdir(parents=True, exist_ok=True)
        summary_path = "results/selected_stocks_extended_summary.csv"
        df_all_metrics.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[전체 결과 요약 저장 완료] -> {summary_path}")
        print("\n" + "=" * 80)
        print("[ 전체 결과 요약 ]")
        print("=" * 80)
        print(df_all_metrics.to_string(index=False))

    print("\n" + "=" * 80)
    print("[ 모든 기업 처리 완료 ]")
    print("=" * 80)


if __name__ == "__main__":
    main()
