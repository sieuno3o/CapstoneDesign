# -*- coding: utf-8 -*-
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE_DIR = Path(__file__).resolve().parents[2]
import base64
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score, accuracy_score

from src.data_loader import load_price_data
from src.preprocess import add_return_features, add_target_next_close, add_target_next_return, drop_missing_rows
from src.feature_engineering import add_moving_averages, add_volatility, add_price_structure_features
from src.feature_engineering_extended import add_extended_features
from src.ai_model import train_ann_model, predict_ai_model
from src.classical_model import train_classical_model, predict_classical_model

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────
_korean_fonts = [
    "AppleGothic",
    "Apple SD Gothic Neo",
    "Nanum Gothic",
    "Noto Sans CJK KR",
    "Malgun Gothic",
]
_available_fonts = {f.name for f in fm.fontManager.ttflist}
for _font in _korean_fonts:
    if _font in _available_fonts:
        matplotlib.rc("font", family=_font)
        break
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 피처 정의 ───────────────────────────────────────────────────────────────
EXTENDED_FEATURES = [
    "MA_3", "MA_5", "MA_10", "MA_20", "BB_upper", "OBV", "ATR_14", "Volume", "hl_diff", "volatility_7"
]

# 모델 14-15에 사용될 피처 (10개 가격변수 + 2개 심리변수 = 총 12개)
HYBRID_FEATURES = EXTENDED_FEATURES + ["nsi_short", "nsi_long"]

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

FINAL_INPUT_PATH = str(BASE_DIR / "data/sentiment/macro_news_counts_1st.csv")
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── 평가 메트릭 함수 ────────────────────────────────────────────────────────
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

# ── 데이터 준비 함수 ────────────────────────────────────────────────────────
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

# ── K-NSI 계산 ──────────────────────────────────────────────────────────────
short_w = {'N1': -0.38235294117647056, 'N2': -0.4117647058823529, 'N3': 0.35294117647058826, 'N4': -0.35294117647058826, 'N5': 1.088235294117647}
long_w = {'N1': 0.25, 'N2': 0.05, 'N3': 0.5, 'N4': -0.05, 'N5': 1.2}

def calc_nsi(news_indexed, weights):
    p_sum = pd.Series(0.0, index=news_indexed.index)
    n_sum = pd.Series(0.0, index=news_indexed.index)
    for col, w in weights.items():
        if col in news_indexed.columns:
            if w > 0: p_sum += w * news_indexed[col]
            elif w < 0: n_sum += abs(w) * news_indexed[col]
    nsi = (((p_sum - n_sum) / (p_sum + n_sum + 1.0)) * 100.0 + 100.0)
    return nsi.rolling(window=7, min_periods=1).mean()


def split_train_validation(X, y, val_ratio=0.15):
    split_idx = int(len(X) * (1 - val_ratio))
    split_idx = min(max(split_idx, 1), len(X) - 1)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def image_to_data_uri(path: Path) -> str:
    if not path.exists():
        print(f"[경고] 이미지 파일이 없습니다: {path}")
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# ── 종목별 직접 결합 모델 학습 및 평가 ─────────────────────────────────────────
def run_direct_hybrid_for_stock(stock_name: str, stock_path: str, news_df: pd.DataFrame):
    print("=" * 80)
    print(f"[Direct Hybrid] {stock_name} 처리 시작")
    print("=" * 80)

    df = prepare_stock_dataframe(stock_path)
    
    # 뉴스 데이터 결합
    news_indexed = news_df.set_index("date")
    nsi_s = calc_nsi(news_indexed, short_w)
    nsi_l = calc_nsi(news_indexed, long_w)
    
    df["date_key"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["nsi_short"] = df["date_key"].map(nsi_s).fillna(100.0)
    df["nsi_long"] = df["date_key"].map(nsi_l).fillna(100.0)
    
    # 뉴스 데이터가 있는 기간으로 한정
    valid_news_dates = news_df["date"].values
    matched_df = df[df["date_key"].isin(valid_news_dates)].copy()
    
    if len(matched_df) < 20:
        raise ValueError(f"{stock_name}의 뉴스 결합 데이터가 부족합니다. 행 개수: {len(matched_df)}")
        
    matched_df = matched_df.sort_values("Date").reset_index(drop=True)
    
    # 시계열 70/30 분할 (Train/Test)
    split_idx = int(len(matched_df) * 0.7)
    train_df = matched_df.iloc[:split_idx].copy()
    test_df = matched_df.iloc[split_idx:].copy()
    
    y_train_r = train_df[TARGET_COL].values
    today_close = test_df["Close"].values
    y_test_price = test_df[TARGET_PRICE_COL].values
    
    # 스케일러 설정 (Hybrid: 12개 피처)
    scaler_hyb = MinMaxScaler()
    X_train_hyb = scaler_hyb.fit_transform(train_df[HYBRID_FEATURES])
    X_test_hyb = scaler_hyb.transform(test_df[HYBRID_FEATURES])
    
    # 비교를 위한 Extended (10개 피처)
    scaler_ext = MinMaxScaler()
    X_train_ext = scaler_ext.fit_transform(train_df[EXTENDED_FEATURES])
    X_test_ext = scaler_ext.transform(test_df[EXTENDED_FEATURES])
    
    # ── 베이스라인 예측 ──
    # 1. Benchmark (Naive)
    bench_price = today_close.copy()

    # 2. ARIMA
    try:
        arima_fitted = train_classical_model(train_df["Close"])
        arima_price = predict_classical_model(arima_fitted, steps=len(test_df))
        arima_price = np.asarray(arima_price, dtype=float)
        if len(arima_price) != len(test_df) or np.isnan(arima_price).any():
            raise ValueError("ARIMA forecast produced invalid values.")
    except Exception as e:
        print(f"[경고] {stock_name} ARIMA 학습 실패: {e}")
        arima_price = bench_price.copy()
    
    # 3. Extended RF
    rf_ext = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_ext.fit(X_train_ext, y_train_r)
    rf_ext_ret = rf_ext.predict(X_test_ext)
    rf_ext_price = today_close * (1 + rf_ext_ret)
    
    # 4. Extended ANN
    X_train_ext_model, y_train_ext_model, X_val_ext, y_val_ext = split_train_validation(X_train_ext, y_train_r)
    ann_ext = train_ann_model(
        X_train_ext_model,
        y_train_ext_model,
        X_val=X_val_ext,
        y_val=y_val_ext,
    )
    ann_ext_ret = predict_ai_model(ann_ext, X_test_ext)
    ann_ext_price = today_close * (1 + ann_ext_ret)
    
    # ── 직접 결합 모델 (Models 14-15) ──
    # 5. Direct Hybrid RF (결정변수 + 심리변수 12개 RF)
    rf_hyb = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_hyb.fit(X_train_hyb, y_train_r)
    rf_hyb_ret = rf_hyb.predict(X_test_hyb)
    rf_hyb_price = today_close * (1 + rf_hyb_ret)
    
    # 6. Direct Hybrid ANN (결정변수 + 심리변수 12개 ANN)
    X_train_hyb_model, y_train_hyb_model, X_val_hyb, y_val_hyb = split_train_validation(X_train_hyb, y_train_r)
    ann_hyb = train_ann_model(
        X_train_hyb_model,
        y_train_hyb_model,
        X_val=X_val_hyb,
        y_val=y_val_hyb,
    )
    ann_hyb_ret = predict_ai_model(ann_hyb, X_test_hyb)
    ann_hyb_price = today_close * (1 + ann_hyb_ret)
    
    eval_date_index = test_df["Date"].dt.strftime("%Y-%m-%d")
    
    # 평가 메트릭 취합
    metrics = []
    metrics.append({"Company": stock_name, "Model": "Benchmark", **evaluate_regression(y_test_price, bench_price)})
    metrics.append({"Company": stock_name, "Model": "ARIMA", **evaluate_regression(y_test_price, arima_price)})
    metrics.append({"Company": stock_name, "Model": "Extended RF", **evaluate_regression(y_test_price, rf_ext_price)})
    metrics.append({"Company": stock_name, "Model": "Direct Hybrid RF", **evaluate_regression(y_test_price, rf_hyb_price)})
    metrics.append({"Company": stock_name, "Model": "Extended ANN", **evaluate_regression(y_test_price, ann_ext_price)})
    metrics.append({"Company": stock_name, "Model": "Direct Hybrid ANN", **evaluate_regression(y_test_price, ann_hyb_price)})
    
    df_metrics = pd.DataFrame(metrics)
    metrics_path = METRICS_DIR / f"{stock_name}_direct_hybrid_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    
    # ── 피처 중요도 시각화 (Direct Hybrid RF) ──
    importances = rf_hyb.feature_importances_
    feat_series = pd.Series(importances, index=HYBRID_FEATURES).sort_values(ascending=True)
    fig_fi, ax_fi = plt.subplots(figsize=(10, 6), dpi=100)
    feat_series.plot(kind="barh", ax=ax_fi, color="indigo")
    ax_fi.set_title(f"Feature Importance (Direct Hybrid RF) - {stock_name}", fontsize=12, fontweight="bold")
    ax_fi.set_xlabel("Importance", fontsize=10)
    ax_fi.grid(True, alpha=0.3)
    fi_path = FIGURES_DIR / f"{stock_name}_direct_hybrid_rf_fi.png"
    plt.tight_layout()
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    
    # ── 예측 결과 시각화 ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), dpi=120)
    
    # 1. RF 비교
    axes[0].plot(eval_date_index, y_test_price, label="Actual Close", color="black", linewidth=2.5)
    axes[0].plot(eval_date_index, rf_ext_price, label="Extended RF (10 features)", color="royalblue", linestyle="--")
    axes[0].plot(eval_date_index, rf_hyb_price, label="Direct Hybrid RF (12 features)", color="darkviolet", linestyle="-")
    axes[0].set_title(f"{stock_name} - Extended RF vs Direct Hybrid RF")
    axes[0].set_ylabel("Price (KRW)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. ANN 비교
    axes[1].plot(eval_date_index, y_test_price, label="Actual Close", color="black", linewidth=2.5)
    axes[1].plot(eval_date_index, ann_ext_price, label="Extended ANN (10 features)", color="darkorange", linestyle="--")
    axes[1].plot(eval_date_index, ann_hyb_price, label="Direct Hybrid ANN (12 features)", color="chocolate", linestyle="-")
    axes[1].set_title(f"{stock_name} - Extended ANN vs Direct Hybrid ANN")
    axes[1].set_ylabel("Price (KRW)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. RMSE 비교
    bar_names = ["Benchmark", "ARIMA", "Extended RF", "Direct Hybrid RF", "Extended ANN", "Direct Hybrid ANN"]
    bar_values = [df_metrics.loc[df_metrics["Model"] == name, "rmse"].values[0] for name in bar_names]
    axes[2].bar(bar_names, bar_values, color=["dimgray", "teal", "royalblue", "darkviolet", "darkorange", "chocolate"])
    axes[2].set_title(f"{stock_name} - RMSE Comparison")
    axes[2].set_ylabel("RMSE (KRW)")
    for i, val in enumerate(bar_values):
        axes[2].text(i, val + max(bar_values) * 0.01, f"{val:,.0f}", ha="center", va="bottom", fontsize=10)
    axes[2].grid(True, axis="y", alpha=0.3)
    
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    figure_path = FIGURES_DIR / f"{stock_name}_direct_hybrid_prediction.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    
    return df_metrics


def generate_html_report(df_summary: pd.DataFrame):
    report_path = RESULTS_DIR / "direct_hybrid_report.html"
    
    # 평균 메트릭스
    avg_metrics = df_summary.groupby("Model").mean(numeric_only=True).reset_index()
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>결정변수 + 심리변수(12개) 직접 결합 모델 결과 리포트</title>
        <style>
            body {{ font-family: 'Malgun Gothic', 'AppleGothic', sans-serif; margin: 20px; background-color: #f9f9fc; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: center; }}
            th {{ background-color: #34495e; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #e8f4f8; }}
            .highlight {{ color: #e74c3c; font-weight: bold; }}
            .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>결정변수 + 심리변수(12개) 직접 결합 모델 결과 리포트 (Models 14-15)</h1>
        <p>기존 가격변수 10개(EXTENDED_FEATURES)와 K_NSI(Short/Long) 2개를 하나의 데이터 프레임으로 구성하여 <strong>직접 결합(Feature Concatenation)</strong>한 예측 모델의 결과입니다.</p>
        
        <h2>전체 평균 성과 (8개 종목)</h2>
        {avg_metrics.to_html(index=False, classes='table table-striped')}
        
        <h2>개별 종목 성과</h2>
        {df_summary.to_html(index=False, classes='table table-striped')}
        
        <h2>종목별 상세 시각화</h2>
    """
    
    for stock in SELECTED_STOCKS.keys():
        pred_img = image_to_data_uri(FIGURES_DIR / f"{stock}_direct_hybrid_prediction.png")
        fi_img = image_to_data_uri(FIGURES_DIR / f"{stock}_direct_hybrid_rf_fi.png")
        
        html_content += f"""
        <div class="card" style="margin-bottom: 30px;">
            <h3>{stock.replace('_', ' ').title()}</h3>
            <div class="grid-container">
                <div>
                    <h4>Prediction & RMSE</h4>
                    <img src="{pred_img}" alt="{stock} Predictions">
                </div>
                <div>
                    <h4>Feature Importance (Direct Hybrid RF)</h4>
                    <img src="{fi_img}" alt="{stock} Feature Importance">
                </div>
            </div>
        </div>
        """
        
    html_content += """
    </body>
    </html>
    """
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[저장] HTML 리포트 생성 완료: {report_path}")


def main():
    if not Path(FINAL_INPUT_PATH).exists():
        print(f"[오류] 뉴스 데이터 파일이 없습니다: {FINAL_INPUT_PATH}")
        return

    news_df = pd.read_csv(FINAL_INPUT_PATH)
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")

    all_metrics = []

    for stock_name, stock_path in SELECTED_STOCKS.items():
        try:
            df_metrics = run_direct_hybrid_for_stock(stock_name, stock_path, news_df)
            all_metrics.append(df_metrics)
        except Exception as exc:
            print(f"[오류] {stock_name} 처리 중 오류 발생: {exc}")

    if not all_metrics:
        print("[종료] 처리 가능한 종목이 없습니다.")
        return

    df_summary = pd.concat(all_metrics, ignore_index=True)
    summary_path = METRICS_DIR / "direct_hybrid_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[저장] {summary_path}")

    # 리포트 생성
    generate_html_report(df_summary)

if __name__ == "__main__":
    main()
