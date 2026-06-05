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
# 선정 기준: 5개 국내 종목 Ablation Study 전 종목 Top-10 출현 빈도 + Feature Importance
# ※ ma_7/14/21(ORIGINAL) → MA_3~MA_20 계열로 통일하여 이동평균 중복 제거
# ※ MACD 3개: 제거 실험에서 RMSE 차이 0.017% → 영향 없음 확인 (교수님 지적 일치)
EXTENDED_FEATURES = [
    # ① 이동평균 4개 (단기~중장기 추세 — 전 종목 Top-3 압도적)
    "MA_3",         # 3일: 삼성 94.1%, SK 59.2%, LIG 47.3% — 전 종목 5/5 1~3위
    "MA_5",         # 5일: 전 종목 5/5 Top-5 이내 안정적 등장
    "MA_10",        # 10일: 4/5 종목 Top-10 (중기 추세 보완)
    "MA_20",        # 20일: 4/5 종목 Top-10 (볼린저밴드 기준선과 동일)
    # ② 볼린저밴드 상단 (가격 위치·변동성 레벨 — 4/5 종목 등장)
    "BB_upper",     # SK 2위, SNT 2위, LIG 5위, 한화 8위
    # ③ OBV — 거래량·가격 방향 결합 (수급 흐름 — 4/5 종목 등장)
    "OBV",          # SNT 1위, LIG 2위, 삼성 4위, 한화 9위
    # ④ ATR_14 — 고가·저가·전일종가 기반 실제 변동폭 (변동성 대표)
    "ATR_14",       # SK 9위, 한화 10위: MACD·래그 대비 유일하게 유효한 모멘텀 보조
    # ⑤ 거래량·가격 구조 기본값 (해석 가능성 확보)
    "Volume",       # 당일 총 거래량 (수급 규모)
    "hl_diff",      # 고가 - 저가 (일중 변동폭)
    "volatility_7", # 7일 수익률 표준편차 (단기 변동성 연속성)
]
# ※ 총 10개 | 제거: ma_7/14/21(MA_3~20으로 통일), MACD 3개, lag 3개,
#   Stoch_K/D, pct_volatility, log_trading_value, Close_MA_ratio 등 31개

TARGET_COL = "target_next_return"   # 수익률 예측 (종가→수익률로 변경)
TARGET_PRICE_COL = "target_next_close"  # 역변환 검증용 종가 타깃 (참고용)


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

    # 4. 타깃 생성 (다음날 수익률 — 절대 가격 대신 수익률 예측으로 변경)
    #    target_next_return = (Close_{t+1} - Close_t) / Close_t
    #    Naive 예측: 수익률 0% → 예측 종가 = 오늘 종가 (기존 Naive와 동일)
    #    예측 종가 역변환: predicted_close = today_close × (1 + predicted_return)
    df = add_target_next_return(df, price_col="Close")
    df = add_target_next_close(df, price_col="Close")   # 역변환 검증용 유지

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
    test_dates  = pd.to_datetime(test_df["Date"])
    today_close = test_df["Close"].values          # 역변환에 사용할 오늘 종가
    y_train_r   = train_df[TARGET_COL]             # 수익률 타깃 (학습용)
    y_val_r     = val_df[TARGET_COL]               # 수익률 타깃 (검증용)
    y_test_r    = test_df[TARGET_COL]              # 수익률 타깃 (평가용)
    y_test_price = test_df[TARGET_PRICE_COL].values  # 실제 내일 종가 (그래프/RMSE 표시용)

    def ret_to_price(ret_pred):
        """예측 수익률 → 예측 종가 변환: today_close × (1 + ret)"""
        return today_close * (1 + np.array(ret_pred))

    # 8. 모델 예측 수행 (수익률 예측 → 종가 변환)
    results_price = {}   # 종가 단위 (그래프·RMSE 표시용)
    results_return = {}  # 수익률 단위 (모델 평가 원본)

    # (A) Naive 벤치마크: 수익률 0% 예측 → 예측 종가 = 오늘 종가
    naive_ret  = np.zeros(len(test_df))            # 수익률 0%
    naive_price = today_close.copy()               # 예측 종가 = 오늘 종가
    results_return["Benchmark"] = naive_ret
    results_price["Benchmark"]  = naive_price

    # (B) ARIMA: 종가 예측 후 수익률로 변환
    print(f"\n[{data_name}] ARIMA 베이스라인 학습 중...")
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
    # 종가 예측 → 수익률 변환
    arima_ret = (arima_price_pred - today_close) / today_close
    results_return["ARIMA"] = arima_ret
    results_price["ARIMA"]  = arima_price_pred

    # (C) Existing RF (기존 7개 변수, 수익률 예측)
    print(f"\n[{data_name}] 기존 Random Forest 학습 중...")
    scaler_orig  = MinMaxScaler()
    X_train_orig = scaler_orig.fit_transform(train_df[ORIGINAL_FEATURES])
    X_test_orig  = scaler_orig.transform(test_df[ORIGINAL_FEATURES])
    rf_orig = train_rf_model(X_train_orig, y_train_r)
    rf_orig_ret   = rf_orig.predict(X_test_orig)
    results_return["Existing RF"] = rf_orig_ret
    results_price["Existing RF"]  = ret_to_price(rf_orig_ret)

    # (D) Extended RF (10개 변수, 수익률 예측)
    print(f"\n[{data_name}] Extended Random Forest 학습 중...")
    scaler_ext  = MinMaxScaler()
    X_train_ext = scaler_ext.fit_transform(train_df[EXTENDED_FEATURES])
    X_val_ext   = scaler_ext.transform(val_df[EXTENDED_FEATURES])
    X_test_ext  = scaler_ext.transform(test_df[EXTENDED_FEATURES])
    rf_ext = train_rf_model(X_train_ext, y_train_r)
    rf_ext_ret  = rf_ext.predict(X_test_ext)
    results_return["Extended RF"] = rf_ext_ret
    results_price["Extended RF"]  = ret_to_price(rf_ext_ret)

    # (E) Feature Importance 시각화 (Extended RF)
    importances = rf_ext.feature_importances_
    feat_series = pd.Series(importances, index=EXTENDED_FEATURES).sort_values(ascending=True)
    top_n = min(20, len(feat_series))
    fig_fi, ax_fi = plt.subplots(figsize=(10, 8), dpi=100)
    feat_series.tail(top_n).plot(kind="barh", ax=ax_fi, color="steelblue")
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

    # (F) Extended ANN (수익률 예측)
    print(f"\n[{data_name}] Extended ANN 학습 중...")
    ann_ext = train_ann_model(
        X_train_ext, y_train_r.values,
        X_val=X_val_ext, y_val=y_val_r.values
    )
    ann_ext_ret  = predict_ai_model(ann_ext, X_test_ext)
    results_return["Extended ANN"] = ann_ext_ret
    results_price["Extended ANN"]  = ret_to_price(ann_ext_ret)

    # 9. 모델 평가 — 종가 단위로 통일 (수익률 역변환 후 비교)
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
    results_save_path = f"results/metrics/{data_name}_ai_extended_results.csv"
    df_metrics.to_csv(results_save_path, index=False, encoding="utf-8-sig")
    print(f"[평가 결과 저장 완료] -> {results_save_path}")
    print(df_metrics.to_string(index=False))

    # results 변수명을 이후 시각화 코드와 호환되도록 통일
    results = results_price

    # 10. 시각화
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    # --- 모델별 RMSE 계산 (범례 표시용) ---
    rmse_dict = {}
    for model_name, y_pred in results.items():
        rmse_val = np.sqrt(np.mean((y_test_price - y_pred) ** 2))
        rmse_dict[model_name] = rmse_val

    # ── 그래프 1: 예측값 비교 (5개 모델 + 실제값) ──────────────────────────
    fig, ax = plt.subplots(figsize=(16, 7), dpi=120)

    # 실제 종가 (굵은 검정)
    ax.plot(test_dates, y_test_price,
            label="Actual Close (실제 종가)",
            color="black", linewidth=2.5, zorder=5)

    # Naive 벤치마크: 전일 종가를 그대로 예측
    ax.plot(test_dates, results["Benchmark"],
            label=f"Naive Benchmark (전일 종가)  RMSE={rmse_dict['Benchmark']:,.0f}",
            color="dimgray", linestyle=(0, (1, 1)), linewidth=1.8, alpha=0.85)

    # ARIMA
    ax.plot(test_dates, results["ARIMA"],
            label=f"ARIMA Forecast              RMSE={rmse_dict['ARIMA']:,.0f}",
            color="crimson", linestyle="-.", linewidth=1.6, alpha=0.85)

    # Existing RF (7개 피처)
    ax.plot(test_dates, results["Existing RF"],
            label=f"Existing RF  (7 features)   RMSE={rmse_dict['Existing RF']:,.0f}",
            color="darkorange", linestyle="--", linewidth=1.8, alpha=0.9)

    # Extended RF (30+ 피처)
    ax.plot(test_dates, results["Extended RF"],
            label=f"Extended RF  (30+ features) RMSE={rmse_dict['Extended RF']:,.0f}",
            color="royalblue", linestyle="-", linewidth=2.0, alpha=0.9)

    # Extended ANN (30+ 피처)
    ax.plot(test_dates, results["Extended ANN"],
            label=f"Extended ANN (30+ features) RMSE={rmse_dict['Extended ANN']:,.0f}",
            color="forestgreen", linestyle="-", linewidth=2.0, alpha=0.9)

    ax.set_title(
        f"Model Predictions Comparison — {data_name.replace('_', ' ').title()}",
        fontsize=14, fontweight="bold", pad=14
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (KRW)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left",
              framealpha=0.85, edgecolor="gray",
              prop={"family": "monospace"})   # 고정폭 폰트 → 수치 정렬
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # 기준 설명 텍스트 박스
    note_text = (
        "[RMSE 개선율 기준]\n"
        f"Existing RF 대비 Extended RF : "
        f"{(rmse_dict['Existing RF']-rmse_dict['Extended RF'])/rmse_dict['Existing RF']*100:+.2f}%\n"
        f"Naive 대비 Extended RF        : "
        f"{(rmse_dict['Benchmark']-rmse_dict['Extended RF'])/rmse_dict['Benchmark']*100:+.2f}%"
    )
    ax.text(
        0.99, 0.02, note_text,
        transform=ax.transAxes,
        fontsize=8, verticalalignment="bottom", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9)
    )

    fig.tight_layout()
    pred_graph_path = f"results/figures/{data_name}_ai_extended_prediction.png"
    fig.savefig(pred_graph_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[예측 비교 그래프 저장 완료] -> {pred_graph_path}")

    # ── 그래프 2: RMSE 막대 비교 (모델별 한눈에 비교) ─────────────────────
    model_names = list(rmse_dict.keys())
    rmse_values = list(rmse_dict.values())
    bar_colors = ["dimgray", "dimgray", "darkorange", "royalblue", "forestgreen"]
    hatches     = ["\\\\", "", "", "", ""]

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
        f"RMSE Comparison by Model — {data_name.replace('_', ' ').title()}",
        fontsize=13, fontweight="bold"
    )
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
