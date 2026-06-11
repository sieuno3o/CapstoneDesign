"""
Ablation Study: 변수 선택 실험
- 목적: 어떤 변수가 실제로 성능에 기여하는지 실험으로 확인
- 방법:
    1. Extended RF 전체 변수(30+)로 학습 후 Feature Importance 추출
    2. Top-N 변수만 사용해 재학습 → N별 RMSE 비교
    3. MACD 포함/제외 실험 (교수님 지적 사항)
- 출력:
    - results/ablation/ablation_results.csv  : 실험별 RMSE 비교 테이블
    - results/ablation/ablation_rmse_plot.png: RMSE vs 변수 개수 그래프
    - results/ablation/feature_importance_avg.png: 전체 종목 평균 중요도
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE_DIR = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from src.data_loader import load_price_data
from src.preprocess import add_return_features, add_target_next_close, drop_missing_rows
from src.feature_engineering import add_moving_averages, add_volatility, add_price_structure_features
from src.feature_engineering_extended import add_extended_features
from src.split import split_time_series
from src.evaluate import regression_metrics

# ── 한글 폰트 설정 ────────────────────────────────────────────────────────────
_korean_fonts = [f.name for f in fm.fontManager.ttflist
                 if "Gothic" in f.name or "Nanum" in f.name or "Apple" in f.name]
if _korean_fonts:
    matplotlib.rc("font", family=_korean_fonts[0])
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 전체 확장 피처 목록 ────────────────────────────────────────────────────────
ALL_FEATURES = [
    "Volume", "ma_7", "ma_14", "ma_21", "volatility_7", "hl_diff", "oc_diff",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width", "BB_percent",
    "MA_3", "MA_5", "MA_10", "MA_20",
    "Close_MA5_ratio", "Close_MA20_ratio", "MA5_MA20_gap",
    "daily_return", "abs_return", "high_low_ratio", "open_close_ratio",
    "pct_volatility_5", "pct_volatility_10",
    "Volume_MA5", "Volume_MA20", "Volume_change", "Volume_ratio",
    "log_trading_value",
    "ATR_14", "OBV", "Stoch_K", "Stoch_D",
    "lag_1_return", "lag_2_return", "lag_3_return", "MA3_return"
]

TARGET_COL = "target_next_close"

# ── 분석 대상 종목 ─────────────────────────────────────────────────────────────
# 연산 대상 종목 (교수님 피드백: 국내 종목 위주로, 너무 많이 할 필요 없음)
STOCKS = {
    "samsung_electronics": str(BASE_DIR / "data/raw/samsung_electronics_5y.csv"),
    "sk_hynix":            str(BASE_DIR / "data/raw/sk_hynix_5y.csv"),
    "hanwha_aerospace":    str(BASE_DIR / "data/raw/hanwha_aerospace_5y.csv"),
    "lig_nex1":            str(BASE_DIR / "data/raw/lig_nex1_5y.csv"),
    "snt_dynamics":        str(BASE_DIR / "data/raw/snt_dynamics_5y.csv"),
}

# ── Top-N 실험 구간 ────────────────────────────────────────────────────────────
TOP_N_LIST = [5, 10, 15, 20, 25, 30, len(ALL_FEATURES)]


def load_and_prepare(file_path: str):
    """데이터 로드 + 피처 생성 + 분할까지 공통 처리"""
    df = load_price_data(file_path)
    df = add_return_features(df, price_col="Close")
    df = add_moving_averages(df, price_col="Close")
    df = add_volatility(df, return_col="log_return")
    df = add_price_structure_features(df)
    df = add_extended_features(df)
    df = add_target_next_close(df, price_col="Close")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = drop_missing_rows(df)
    train_df, val_df, test_df = split_time_series(df, train_ratio=0.7, val_ratio=0.15)
    return train_df, val_df, test_df


def train_rf(X_train, y_train):
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def get_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def run_ablation():
    Path(str(BASE_DIR / "results/ablation")).mkdir(parents=True, exist_ok=True)

    all_rows = []          # 실험 결과 누적
    importance_accum = {}  # 종목별 피처 중요도 누적

    for stock_name, file_path in STOCKS.items():
        if not os.path.exists(file_path):
            print(f"[경고] {file_path} 없음, 스킵")
            continue

        print("\n" + "=" * 70)
        print(f"  [{stock_name}] Ablation Study 시작")
        print("=" * 70)

        train_df, val_df, test_df = load_and_prepare(file_path)
        y_train = train_df[TARGET_COL]
        y_test  = test_df[TARGET_COL]

        # Naive RMSE (기준선)
        naive_rmse = get_rmse(y_test.values, test_df["Close"].values)
        print(f"  Naive RMSE: {naive_rmse:,.2f}")

        # ── Step 1: 전체 변수로 학습 → Importance 순위 추출 ──────────────────
        scaler_all = MinMaxScaler()
        X_train_all = scaler_all.fit_transform(train_df[ALL_FEATURES])
        X_test_all  = scaler_all.transform(test_df[ALL_FEATURES])

        rf_all = train_rf(X_train_all, y_train)
        importances = pd.Series(rf_all.feature_importances_, index=ALL_FEATURES)
        importance_accum[stock_name] = importances

        # 중요도 내림차순 정렬
        sorted_features = importances.sort_values(ascending=False)
        print(f"\n  [Top 10 중요 변수]")
        for feat, imp in sorted_features.head(10).items():
            print(f"    {feat:<25} {imp:.4f}")

        # ── Step 2: Top-N 실험 ────────────────────────────────────────────────
        for top_n in TOP_N_LIST:
            top_features = sorted_features.index[:top_n].tolist()
            scaler_n = MinMaxScaler()
            X_train_n = scaler_n.fit_transform(train_df[top_features])
            X_test_n  = scaler_n.transform(test_df[top_features])

            rf_n  = train_rf(X_train_n, y_train)
            pred_n = rf_n.predict(X_test_n)
            rmse_n = get_rmse(y_test.values, pred_n)

            row = {
                "Company":    stock_name,
                "Experiment": f"Top-{top_n}",
                "N_features": top_n,
                "RMSE":       rmse_n,
                "vs_Naive_%": round((naive_rmse - rmse_n) / naive_rmse * 100, 2),
                "Features":   ", ".join(top_features[:5]) + ("..." if top_n > 5 else ""),
            }
            all_rows.append(row)
            print(f"  Top-{top_n:2d} 변수: RMSE={rmse_n:>12,.2f}  "
                  f"vs Naive {row['vs_Naive_%']:+.1f}%")

        # ── Step 3: MACD 포함/제외 실험 (교수님 지적) ─────────────────────────
        macd_cols   = ["MACD", "MACD_signal", "MACD_hist"]
        no_macd_features = [f for f in ALL_FEATURES if f not in macd_cols]

        scaler_nm = MinMaxScaler()
        X_train_nm = scaler_nm.fit_transform(train_df[no_macd_features])
        X_test_nm  = scaler_nm.transform(test_df[no_macd_features])
        rf_nm  = train_rf(X_train_nm, y_train)
        pred_nm = rf_nm.predict(X_test_nm)
        rmse_nm = get_rmse(y_test.values, pred_nm)

        row_nm = {
            "Company":    stock_name,
            "Experiment": "No-MACD (전체-MACD)",
            "N_features": len(no_macd_features),
            "RMSE":       rmse_nm,
            "vs_Naive_%": round((naive_rmse - rmse_nm) / naive_rmse * 100, 2),
            "Features":   "MACD 3개 제거",
        }
        all_rows.append(row_nm)
        print(f"  No-MACD   : RMSE={rmse_nm:>12,.2f}  "
              f"vs Naive {row_nm['vs_Naive_%']:+.1f}%")

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    df_result = pd.DataFrame(all_rows)
    save_path = str(BASE_DIR / "results/ablation/ablation_results.csv")
    df_result.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n[결과 저장] {save_path}")

    # ── 그래프 1: RMSE vs Top-N (종목별 선 그래프) ────────────────────────────
    df_topn = df_result[df_result["Experiment"].str.startswith("Top-")]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=120)

    colors = ["royalblue", "forestgreen", "darkorange", "crimson", "purple"]
    for i, (stock, grp) in enumerate(df_topn.groupby("Company")):
        grp_sorted = grp.sort_values("N_features")
        ax.plot(grp_sorted["N_features"], grp_sorted["RMSE"],
                marker="o", label=stock, color=colors[i % len(colors)], linewidth=1.8)

    ax.set_title("Ablation Study — RMSE vs 변수 개수 (Top-N)", fontsize=13, fontweight="bold")
    ax.set_xlabel("사용 변수 개수 (Top-N by Importance)", fontsize=11)
    ax.set_ylabel("RMSE", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(BASE_DIR / "results/ablation/ablation_rmse_plot.png"), bbox_inches="tight")
    plt.close(fig)
    print("[그래프 저장] results/ablation/ablation_rmse_plot.png")

    # ── 그래프 2: 전체 종목 평균 Feature Importance (Top 20) ─────────────────
    if importance_accum:
        avg_importance = pd.DataFrame(importance_accum).mean(axis=1).sort_values(ascending=True)
        top20 = avg_importance.tail(20)

        fig2, ax2 = plt.subplots(figsize=(10, 8), dpi=120)
        bars = ax2.barh(top20.index, top20.values, color="steelblue", alpha=0.85)
        for bar, val in zip(bars, top20.values):
            ax2.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=8)
        ax2.set_title("전체 종목 평균 Feature Importance (Extended RF, Top 20)",
                      fontsize=12, fontweight="bold")
        ax2.set_xlabel("평균 Importance (상대적 기여도)", fontsize=10)
        ax2.grid(True, axis="x", alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(str(BASE_DIR / "results/ablation/feature_importance_avg.png"), bbox_inches="tight")
        plt.close(fig2)
        print("[그래프 저장] results/ablation/feature_importance_avg.png")

    # ── 최종 요약 출력 ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [Ablation Study 최종 요약]")
    print("=" * 70)
    summary = df_topn.groupby("N_features")["RMSE"].mean().reset_index()
    summary.columns = ["N_features", "평균 RMSE"]
    summary["vs_Naive 평균"] = df_topn.groupby("N_features")["vs_Naive_%"].mean().values
    print(summary.to_string(index=False))

    print("\n  [MACD 포함 vs 제외]")
    macd_rows = df_result[df_result["Experiment"].str.contains("MACD")]
    all_rows_macd = df_result[df_result["Experiment"] == f"Top-{len(ALL_FEATURES)}"]
    print(f"  MACD 포함 (전체 변수) 평균 RMSE: "
          f"{all_rows_macd['RMSE'].mean():,.2f}")
    print(f"  MACD 제거 후 평균 RMSE:          "
          f"{macd_rows['RMSE'].mean():,.2f}")

    print("\n[완료] results/ablation/ 폴더를 확인하세요.")


if __name__ == "__main__":
    run_ablation()
