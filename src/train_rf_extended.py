from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler

# ── 한글 폰트 설정 (macOS: AppleGothic, 없으면 기본 유지) ──────────────────
_korean_fonts = [f.name for f in fm.fontManager.ttflist if "Gothic" in f.name or "Nanum" in f.name or "Apple" in f.name]
if _korean_fonts:
    matplotlib.rc("font", family=_korean_fonts[0])
matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

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
from sklearn.ensemble import RandomForestRegressor

def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model




# ── 기존 논문 변수 (비교 기준용, 7개) ───────────────────────────────────────
ORIGINAL_FEATURES = [
    "Volume",       # 당일 거래량
    "ma_7",         # 7일 이동평균 (기존 논문)
    "ma_14",        # 14일 이동평균 (기존 논문)
    "ma_21",        # 21일 이동평균 (기존 논문)
    "volatility_7", # 7일 수익률 표준편차
    "hl_diff",      # 고가 - 저가
    "oc_diff"       # 시가 - 종가
]

# ── Ablation Study 기반 최종 확정 변수 (총 10개, 독립 정의) ──────────────────
EXTENDED_FEATURES = [
    "MA_3",         # 3일 단순이동평균
    "MA_5",         # 5일 단순이동평균
    "MA_10",        # 10일 단순이동평균
    "MA_20",        # 20일 단순이동평균
    "BB_upper",     # 볼린저밴드 상단
    "OBV",          # OBV 수급지표
    "ATR_14",       # ATR 변동성지표
    "Volume",       # 당일 총 거래량
    "hl_diff",      # 고가 - 저가
    "volatility_7", # 7일 수익률 표준편차
]



TARGET_COL = "target_next_return"   # 수익률 예측
TARGET_PRICE_COL = "target_next_close"  # 역변환 검증용 종가 타깃


def calculate_metrics(y_true, y_pred, model_name: str):
    metrics = regression_metrics(y_true, y_pred)
    dir_acc = direction_accuracy(y_true, y_pred)
    metrics["direction_accuracy"] = dir_acc
    metrics["Model"] = model_name
    return metrics


def train_rf_extended_pipeline(data_name: str, file_path: str):
    print("=" * 80)
    print(f"[확장 RF 파이프라인] {data_name} 실행 시작")
    print("=" * 80)

    # 1. 주가 데이터 불러오기
    df = load_price_data(file_path)

    # 2. 파생변수 생성
    df = add_return_features(df, price_col="Close")
    df = add_moving_averages(df, price_col="Close")
    df = add_volatility(df, return_col="log_return")
    df = add_price_structure_features(df)

    # 3. 추가 입력변수 (RSI, MACD, 볼린저밴드 등) 생성
    df = add_extended_features(df)

    # 4. 타깃 생성
    df = add_target_next_return(df, price_col="Close")
    df = add_target_next_close(df, price_col="Close")

    # 5. 뉴스 심리지수 로드 및 계산
    news_path = "data/raw/macro_news_counts_1st.csv"
    if os.path.exists(news_path):
        news_df = pd.read_csv(news_path)
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")
        news_indexed = news_df.set_index("date")

        short_w = {
            'N1': -0.38235294117647056,
            'N2': -0.4117647058823529,
            'N3': 0.35294117647058826,
            'N4': -0.35294117647058826,
            'N5': 1.088235294117647
        }
        long_w = {
            'N1': 0.25,
            'N2': 0.05,
            'N3': 0.5,
            'N4': -0.05,
            'N5': 1.2
        }

        def calc_nsi(news_indexed, weights):
            p_sum = pd.Series(0.0, index=news_indexed.index)
            n_sum = pd.Series(0.0, index=news_indexed.index)
            for col, w in weights.items():
                if col in news_indexed.columns:
                    if w > 0: p_sum += w * news_indexed[col]
                    elif w < 0: n_sum += abs(w) * news_indexed[col]
            nsi = (((p_sum - n_sum) / (p_sum + n_sum + 1.0)) * 100.0 + 100.0)
            return nsi.rolling(window=7, min_periods=1).mean()

        nsi_s = calc_nsi(news_indexed, short_w)
        nsi_l = calc_nsi(news_indexed, long_w)

        df["date_key"] = df["Date"].dt.strftime("%Y-%m-%d")
        df["nsi_short"] = df["date_key"].map(nsi_s).fillna(100.0)
        df["nsi_long"]  = df["date_key"].map(nsi_l).fillna(100.0)
    else:
        print("[오류] 뉴스 데이터 파일이 없습니다. 기본값 100으로 대체합니다.")
        df["nsi_short"] = 100.0
        df["nsi_long"] = 100.0
        news_df = pd.DataFrame(columns=["date"])

    # 결측 및 무한대 제거
    df = df.replace([np.inf, -np.inf], np.nan)
    df = drop_missing_rows(df)

    # 6. 뉴스 매칭 기간을 기준으로 기차역사적 시계열 분할
    #    과거 뉴스 데이터 시작 이전 기간을 학습 및 검증용(Train/Val)으로 쓰고, 뉴스 기간 전체를 평가(Test)용으로 설정
    valid_news_dates = news_df["date"].values
    if len(valid_news_dates) > 0:
        df["date_key"] = df["Date"].dt.strftime("%Y-%m-%d")
        news_indices = df[df["date_key"].isin(valid_news_dates)].index
        if len(news_indices) > 10:
            start_idx = news_indices.min()
            end_idx   = news_indices.max()
            
            # 학습과 검증용
            train_val_df = df.iloc[:start_idx]
            test_df      = df.iloc[start_idx:end_idx+1]
            
            # Train/Val 85:15 분할
            val_split = int(len(train_val_df) * 0.85)
            train_df  = train_val_df.iloc[:val_split]
            val_df    = train_val_df.iloc[val_split:]
        else:
            train_df, val_df, test_df = split_time_series(df, train_ratio=0.7, val_ratio=0.15)
    else:
        train_df, val_df, test_df = split_time_series(df, train_ratio=0.7, val_ratio=0.15)

    print(f"[데이터 분할 완료] Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

    test_dates  = pd.to_datetime(test_df["Date"])
    today_close = test_df["Close"].values          # 역변환에 사용할 오늘 종가
    y_train_r   = train_df[TARGET_COL]             # 수익률 타깃 (학습용)
    y_val_r     = val_df[TARGET_COL]               # 수익률 타깃 (검증용)
    y_test_r    = test_df[TARGET_COL]              # 수익률 타깃 (평가용)
    y_test_price = test_df[TARGET_PRICE_COL].values  # 실제 내일 종가

    def ret_to_price(ret_pred):
        return today_close * (1 + np.array(ret_pred))

    results_price = {}
    results_return = {}

    # (A) Naive 벤치마크
    naive_ret  = np.zeros(len(test_df))
    naive_price = today_close.copy()
    results_return["Benchmark"] = naive_ret
    results_price["Benchmark"]  = naive_price

    # (B) ARIMA
    print(f"\n[{data_name}] ARIMA 베이스라인 학습 중...")
    try:
        arima_model = find_best_arima_model(train_df["Close"])
        arima_price_pred = arima_model.predict(n_periods=len(test_df))
        arima_price_pred = arima_price_pred if isinstance(arima_price_pred, np.ndarray) else arima_price_pred.values
    except Exception as e:
        print(f"[오류] Auto ARIMA 탐색 실패, Naive로 대체합니다: {e}")
        arima_price_pred = naive_price.copy()
    arima_ret = (arima_price_pred - today_close) / today_close
    results_return["ARIMA"] = arima_ret
    results_price["ARIMA"]  = arima_price_pred

    # (C) Existing RF (Price Only - 7 features)
    print(f"\n[{data_name}] 기존 Random Forest 학습 중...")
    scaler_orig  = MinMaxScaler()
    X_train_orig = scaler_orig.fit_transform(train_df[ORIGINAL_FEATURES])
    X_test_orig  = scaler_orig.transform(test_df[ORIGINAL_FEATURES])
    rf_orig = train_rf_model(X_train_orig, y_train_r)
    rf_orig_ret   = rf_orig.predict(X_test_orig)
    results_return["Existing RF"] = rf_orig_ret
    results_price["Existing RF"]  = ret_to_price(rf_orig_ret)

    # (D) Extended RF (Price Only - 10 features)
    print(f"\n[{data_name}] Extended Random Forest 학습 중...")
    scaler_ext  = MinMaxScaler()
    X_train_ext = scaler_ext.fit_transform(train_df[EXTENDED_FEATURES])
    X_val_ext   = scaler_ext.transform(val_df[EXTENDED_FEATURES])
    X_test_ext  = scaler_ext.transform(test_df[EXTENDED_FEATURES])
    rf_ext = train_rf_model(X_train_ext, y_train_r)
    rf_ext_ret  = rf_ext.predict(X_test_ext)
    results_return["Extended RF"] = rf_ext_ret
    results_price["Extended RF"]  = ret_to_price(rf_ext_ret)

    # (E) Extended ANN (Price Only - 10 features)
    print(f"\n[{data_name}] Extended ANN 학습 중...")
    try:
        ann_ext = train_ann_model(
            X_train_ext, y_train_r.values,
            X_val=X_val_ext, y_val=y_val_r.values
        )
        ann_ext_ret  = predict_ai_model(ann_ext, X_test_ext)
    except Exception as e:
        print(f"[오류] ANN 학습 실패, Naive로 폴백: {e}")
        ann_ext_ret = naive_ret.copy()
    results_return["Extended ANN"] = ann_ext_ret
    results_price["Extended ANN"]  = ret_to_price(ann_ext_ret)

    # 7. 피처 중요도 시각화 (Extended RF)
    importances = rf_ext.feature_importances_
    feat_series = pd.Series(importances, index=EXTENDED_FEATURES).sort_values(ascending=True)
    fig_fi, ax_fi = plt.subplots(figsize=(10, 8), dpi=100)
    feat_series.plot(kind="barh", ax=ax_fi, color="royalblue")
    ax_fi.set_title(f"Feature Importance (Extended RF) - {data_name.replace('_', ' ').title()}",
                    fontsize=13, fontweight="bold")
    ax_fi.set_xlabel("Importance", fontsize=11)
    ax_fi.grid(True, alpha=0.3)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    fi_path = f"results/figures/{data_name}_feature_importance.png"
    plt.tight_layout()
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"[피처 중요도 그래프 저장 완료] -> {fi_path}")

    # 8. 모델 평가 — 종가 단위로 통일
    print(f"\n{'='*60}")
    print(f"[{data_name}] 모델별 평가 결과 (종가 단위 RMSE)")
    metrics_list = []
    for model_name, price_pred in results_price.items():
        print(f"\n--- {model_name} ---")
        metrics = calculate_metrics(y_test_price, price_pred, model_name)
        metrics_list.append(metrics)

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics = df_metrics[["Model", "rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]]

    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    results_save_path = f"results/metrics/{data_name}_extended_results.csv"
    df_metrics.to_csv(results_save_path, index=False, encoding="utf-8-sig")
    print(f"[평가 결과 저장 완료] -> {results_save_path}")
    print(df_metrics.to_string(index=False))

    # 9. 시각화
    rmse_dict = {}
    for model_name, y_pred in results_price.items():
        rmse_val = np.sqrt(np.mean((y_test_price - y_pred) ** 2))
        rmse_dict[model_name] = rmse_val

    # ── 그래프 1: 예측값 비교 (6개 모델 종합 비교) ──────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7), dpi=120)

    # 실제 종가
    ax.plot(test_dates, y_test_price, label="Actual Close (실제 종가)", color="black", linewidth=2.5, zorder=5)

    # ARIMA
    ax.plot(test_dates, results_price["ARIMA"], label=f"ARIMA Forecast              RMSE={rmse_dict['ARIMA']:,.0f}", color="crimson", linestyle="-.", linewidth=1.5)

    # Existing RF
    ax.plot(test_dates, results_price["Existing RF"], label=f"Existing RF  (7 features)   RMSE={rmse_dict['Existing RF']:,.0f}", color="darkorange", linestyle="--", linewidth=1.6)

    # Extended RF
    ax.plot(test_dates, results_price["Extended RF"], label=f"Extended RF  (10 features)  RMSE={rmse_dict['Extended RF']:,.0f}", color="royalblue", linestyle="-", linewidth=1.8)

    # Extended ANN
    ax.plot(test_dates, results_price["Extended ANN"], label=f"Extended ANN (10 features)  RMSE={rmse_dict['Extended ANN']:,.0f}", color="forestgreen", linestyle="--", linewidth=1.6)

    ax.set_title(f"Model Predictions Comparison — {data_name.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (KRW)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85, edgecolor="gray", prop={"family": "monospace"})
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    # 기준 설명 텍스트 박스
    improve_existing = (rmse_dict["Existing RF"] - rmse_dict["Extended RF"]) / rmse_dict["Existing RF"] * 100
    improve_naive = (rmse_dict["Benchmark"] - rmse_dict["Extended RF"]) / rmse_dict["Benchmark"] * 100
    
    note_text = (
        "[최종 Extended 개선율]\n"
        f"Existing RF 대비 개선 : {improve_existing:+.2f}%\n"
        f"Naive 대비 개선        : {improve_naive:+.2f}%"
    )
    ax.text(0.99, 0.02, note_text, transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    pred_graph_path = f"results/figures/{data_name}_extended_prediction.png"
    fig.savefig(pred_graph_path, bbox_inches="tight")
    plt.close(fig)

    # ── 그래프 1-2: 최근 60일 예측값 줌인 비교 ──────────────────────────
    fig_zoom, ax_zoom = plt.subplots(figsize=(14, 6), dpi=120)
    zoom_days = min(60, len(test_dates))
    
    z_dates = test_dates[-zoom_days:]
    z_actual = y_test_price[-zoom_days:]
    
    ax_zoom.plot(z_dates, z_actual, label="Actual Close (실제 종가)", color="black", linewidth=2.5, zorder=5)
    ax_zoom.plot(z_dates, results_price["ARIMA"][-zoom_days:], label="ARIMA", color="crimson", linestyle="-.", linewidth=1.5)
    ax_zoom.plot(z_dates, results_price["Existing RF"][-zoom_days:], label="Existing RF", color="darkorange", linestyle="--", linewidth=1.5)
    ax_zoom.plot(z_dates, results_price["Extended RF"][-zoom_days:], label="Extended RF", color="royalblue", linestyle="-", linewidth=1.8)
    ax_zoom.plot(z_dates, results_price["Extended ANN"][-zoom_days:], label="Extended ANN", color="forestgreen", linestyle="--", linewidth=1.5)
    
    ax_zoom.set_title(f"Model Predictions (Recent {zoom_days} Days Zoom-in) — {data_name.replace('_', ' ').title()}", fontsize=12, fontweight="bold", pad=12)
    ax_zoom.set_xlabel("Date", fontsize=11)
    ax_zoom.set_ylabel("Price (KRW)", fontsize=11)
    ax_zoom.legend(fontsize=8, loc="upper left", framealpha=0.85, edgecolor="gray", prop={"family": "monospace"})
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.tick_params(axis="x", rotation=15)
    fig_zoom.tight_layout()
    zoom_graph_path = f"results/figures/{data_name}_extended_prediction_zoom.png"
    fig_zoom.savefig(zoom_graph_path, bbox_inches="tight")
    plt.close(fig_zoom)
    print(f"[줌인 비교 그래프 저장 완료] -> {zoom_graph_path}")
    print(f"[예측 비교 그래프 저장 완료] -> {pred_graph_path}")

    # ── 그래프 2: RMSE 막대 비교 ─────────────────────
    model_names = list(rmse_dict.keys())
    rmse_values = list(rmse_dict.values())
    bar_colors = ["dimgray", "dimgray", "darkorange", "royalblue", "forestgreen", "orchid"]
    
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=120)
    bars = ax2.bar(model_names, rmse_values, color=bar_colors, edgecolor="white", linewidth=0.8, alpha=0.88)

    for bar, val in zip(bars, rmse_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(rmse_values) * 0.01,
                 f"{val:,.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    naive_rmse = rmse_dict["Benchmark"]
    ax2.axhline(naive_rmse, color="dimgray", linestyle=":", linewidth=1.5, label=f"Naive RMSE = {naive_rmse:,.0f}")

    ax2.set_title(f"RMSE Comparison by Model — {data_name.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    ax2.set_ylabel("RMSE (KRW)", fontsize=11)
    ax2.set_xlabel("Model", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.35)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    fig2.tight_layout()

    rmse_bar_path = f"results/figures/{data_name}_rmse_comparison.png"
    fig2.savefig(rmse_bar_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"[RMSE 막대 비교 그래프 저장 완료] -> {rmse_bar_path}")

    return df_metrics

