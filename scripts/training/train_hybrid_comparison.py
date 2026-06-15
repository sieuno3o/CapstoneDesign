# -*- coding: utf-8 -*-
"""
train_hybrid_comparison.py
------------------------------
오리지널 5개 모델(Benchmark, ARIMA, Existing RF, Extended RF, Extended ANN)에
뉴스 심리지수(K-NSI)를 결합한 [Hybrid Sentiment RF] 모델을 신규 추가하여,
총 6대 모델의 최종 수익률 예측 및 종가 복원 성능을 종합 비교하고 시각화합니다.

[비교 대상 6개 모델]
  1. Naive Benchmark (수익률 0%)
  2. ARIMA
  3. Existing RF (기존 논문 변수 7개)
  4. Extended RF (보조지표 추가 10개)
  5. Extended ANN (인공신경망 10개)
  6. Hybrid Sentiment RF (Extended RF 10개 + NSI 단기 + NSI 장기)

[출력물]
  - results/metrics/hybrid_final_comparison_summary.csv   (전체 성능 요약표)
  - results/figures/{ticker}_final_comparison.png         (6대 모델 종합 비교 차트)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE_DIR = Path(__file__).resolve().parents[2]

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

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

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

# ─── 한글 폰트 설정 ───
plt.rcParams["font.family"]       = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════
# 0. 경로 설정
# ═══════════════════════════════════════════
RAWDATA_DIR  = BASE_DIR / "data" / "raw"
NEWS_PATH    = BASE_DIR / "data" / "sentiment" / "macro_news_counts_90d.csv"
EXCEL_PATH   = BASE_DIR / "주가 데이터와 인간의 투자심리를 활용한 주가 예측 모델 개발 설문조사(응답).xlsx"
METRICS_DIR  = BASE_DIR / "results" / "metrics"
FIGURES_DIR  = BASE_DIR / "results" / "figures"
REPORT_PATH  = BASE_DIR / "results" / "hybrid_final_comparison_report.html"

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── 국내 8개 타겟 종목 ───
TARGET_COMPANIES = {
    "samsung_electronics": str(BASE_DIR / "data/raw/samsung_electronics_5y.csv"),
    "sk_hynix":            str(BASE_DIR / "data/raw/sk_hynix_5y.csv"),
    "wonik_ips":           str(BASE_DIR / "data/raw/wonik_ips_5y.csv"),
    "dongjin_semichem":    str(BASE_DIR / "data/raw/dongjin_semichem_5y.csv"),
    "hanwha_aerospace":    str(BASE_DIR / "data/raw/hanwha_aerospace_5y.csv"),
    "lig_nex1":            str(BASE_DIR / "data/raw/lig_nex1_5y.csv"),
    "snt_dynamics":        str(BASE_DIR / "data/raw/snt_dynamics_5y.csv"),
    "firstec":             str(BASE_DIR / "data/raw/firstec_5y.csv"),
}

# ─── 변수 세트 정의 ───
ORIGINAL_FEATURES = ["Volume", "ma_7", "ma_14", "ma_21", "volatility_7", "hl_diff", "oc_diff"]
EXTENDED_FEATURES = ["MA_3", "MA_5", "MA_10", "MA_20", "BB_upper", "OBV", "ATR_14", "Volume", "hl_diff", "volatility_7"]

TARGET_COL = "target_next_return"

# ═══════════════════════════════════════════
# 1. 설문조사 가중치 및 K-NSI 연산
# ═══════════════════════════════════════════

def parse_score(val):
    if pd.isna(val): return 0.0
    try: return float(str(val).strip().split()[0])
    except: return 0.0

def load_weights(excel_path=None):
    # 설문조사 응답 데이터 가중치 평균값 백업 복구 (엑셀 대체)
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
    return short_w, long_w

def calc_nsi(news_df, weights):
    df = news_df.copy()
    p_sum = pd.Series(0.0, index=df.index)
    n_sum = pd.Series(0.0, index=df.index)
    for col, w in weights.items():
        if col not in df.columns: continue
        if w > 0: p_sum += w * df[col]
        elif w < 0: n_sum += abs(w) * df[col]
    daily_nsi = ((p_sum - n_sum) / (p_sum + n_sum + 1.0)) * 100.0 + 100.0
    return daily_nsi.rolling(window=7, min_periods=1).mean()

# ═══════════════════════════════════════════
# 2. 종합 비교 파이프라인 실행 함수
# ═══════════════════════════════════════════

def run_comparison_pipeline(name: str, path: str, short_w: dict, long_w: dict, news_df: pd.DataFrame):
    print("\n" + "="*70)
    print(f"  [{name}] 6대 모델 종합 비교 시작")
    print("="*70)

    # A. 데이터 준비
    df = load_price_data(path)
    df = add_return_features(df, price_col="Close")
    df = add_moving_averages(df, price_col="Close")
    df = add_volatility(df, return_col="log_return")
    df = add_price_structure_features(df)
    df = add_extended_features(df)

    df = add_target_next_return(df, price_col="Close")
    df = add_target_next_close(df, price_col="Close")

    # 1. 가격/보조지표 결측치 먼저 정제 (유실 방지)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = drop_missing_rows(df)

    # 2. NSI 생성 및 병합 (결측치 제거 후 매핑)
    news_indexed = news_df.set_index("date")
    nsi_s = calc_nsi(news_indexed, short_w)
    nsi_l = calc_nsi(news_indexed, long_w)

    df["date_key"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["nsi_short"] = df["date_key"].map(nsi_s).fillna(100.0)
    df["nsi_long"]  = df["date_key"].map(nsi_l).fillna(100.0)

    # 뉴스 매칭 범위 필터 (NSI가 실제 뉴스 크롤링 날짜 범위에 존재하는 인덱스 추출)
    valid_news_dates = news_df["date"].values
    news_indices = df[df["date_key"].isin(valid_news_dates)].index

    if len(news_indices) < 10:
        print(f"  [스킵] 뉴스 데이터와 겹치는 기간 부족")
        return []

    start_idx = news_indices.min()
    end_idx   = news_indices.max()

    # 데이터 분할 (train_df에는 과거 5년치 전체가 보존됨)
    train_df = df.iloc[:start_idx]
    test_df  = df.iloc[start_idx:end_idx+1]
    
    val_split = len(train_df) - int(len(train_df) * 0.15)
    val_df   = train_df.iloc[val_split:]
    train_df = train_df.iloc[:val_split]

    test_dates = pd.to_datetime(test_df["Date"])
    today_close = test_df["Close"].values
    y_train_r = train_df[TARGET_COL]
    y_val_r   = val_df[TARGET_COL]
    y_test_r  = test_df[TARGET_COL]
    y_test_price = test_df["target_next_close"].values

    def ret_to_price(ret_pred):
        return today_close * (1 + np.array(ret_pred))

    results_price = {}
    results_return = {}

    # 1. Benchmark (Naive)
    results_return["Benchmark"] = np.zeros(len(test_df))
    results_price["Benchmark"]  = today_close.copy()

    # 2. ARIMA
    try:
        arima_model = find_best_arima_model(train_df["Close"])
        arima_price_pred = arima_model.predict(n_periods=len(test_df))
        arima_price_pred = arima_price_pred if isinstance(arima_price_pred, np.ndarray) else arima_price_pred.values
    except:
        arima_price_pred = today_close.copy()
    results_return["ARIMA"] = (arima_price_pred - today_close) / today_close
    results_price["ARIMA"]  = arima_price_pred

    # 3. Existing RF (7 features)
    scaler_orig  = MinMaxScaler()
    X_train_orig = scaler_orig.fit_transform(train_df[ORIGINAL_FEATURES])
    X_test_orig  = scaler_orig.transform(test_df[ORIGINAL_FEATURES])
    rf_orig = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_orig, y_train_r)
    rf_orig_ret = rf_orig.predict(X_test_orig)
    results_return["Existing RF"] = rf_orig_ret
    results_price["Existing RF"]  = ret_to_price(rf_orig_ret)

    # 4. Existing ANN (7 features - 기존 논문 변수)
    X_val_orig   = scaler_orig.transform(val_df[ORIGINAL_FEATURES])
    ann_orig = train_ann_model(X_train_orig, y_train_r.values, X_val=X_val_orig, y_val=y_val_r.values)
    ann_orig_ret = predict_ai_model(ann_orig, X_test_orig)
    results_return["Existing ANN"] = ann_orig_ret
    results_price["Existing ANN"]  = ret_to_price(ann_orig_ret)

    # 4. Extended RF (10 features)
    scaler_ext  = MinMaxScaler()
    X_train_ext = scaler_ext.fit_transform(train_df[EXTENDED_FEATURES])
    X_test_ext  = scaler_ext.transform(test_df[EXTENDED_FEATURES])
    rf_ext = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_ext, y_train_r)
    rf_ext_ret = rf_ext.predict(X_test_ext)
    results_return["Extended RF"] = rf_ext_ret
    results_price["Extended RF"]  = ret_to_price(rf_ext_ret)

    # 5. Extended ANN (10 features)
    X_val_ext = scaler_ext.transform(val_df[EXTENDED_FEATURES])
    ann_ext = train_ann_model(X_train_ext, y_train_r.values, X_val=X_val_ext, y_val=y_val_r.values)
    ann_ext_ret = predict_ai_model(ann_ext, X_test_ext)
    results_return["Extended ANN"] = ann_ext_ret
    results_price["Extended ANN"]  = ret_to_price(ann_ext_ret)

    # 6. Ensemble: Extended RF + Extended ANN (3가지 방식 자동 비교 → 최적 선택)
    # --- 방식 A: 단순 평균 (Simple Average) ---
    ens_simple_ret  = (rf_ext_ret + ann_ext_ret) / 2.0

    # --- 방식 B: RMSE 역수 기반 가중 평균 (Weighted Average) ---
    #   val 세트에서 각 모델 성능 측정 후 RMSE가 낮은 모델에 높은 가중치 부여
    y_val_np = y_val_r.values
    rf_val_ret  = rf_ext.predict(X_val_ext)
    ann_val_ret = predict_ai_model(ann_ext, X_val_ext)
    rmse_rf_val  = np.sqrt(np.mean((y_val_np - rf_val_ret) ** 2))
    rmse_ann_val = np.sqrt(np.mean((y_val_np - ann_val_ret) ** 2))
    w_rf  = 1.0 / (rmse_rf_val  + 1e-9)
    w_ann = 1.0 / (rmse_ann_val + 1e-9)
    w_total = w_rf + w_ann
    ens_weighted_ret = (w_rf * rf_ext_ret + w_ann * ann_ext_ret) / w_total

    # --- 방식 C: 스태킹 (Stacking) ---
    #   Train: RF + ANN val 예측값을 meta-feature로 LinearRegression 학습
    #   Test:  RF + ANN test 예측값을 meta-feature로 최종 예측
    from sklearn.linear_model import Ridge
    meta_X_train = np.column_stack([rf_val_ret, ann_val_ret])
    meta_y_train = y_val_np
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X_train, meta_y_train)
    meta_X_test    = np.column_stack([rf_ext_ret, ann_ext_ret])
    ens_stacking_ret = meta_model.predict(meta_X_test)

    # --- 3가지 방식 val RMSE 비교 → 가장 낮은 방식 자동 선택 ---
    ens_candidates = {
        "Ensemble (Simple Avg)"   : ens_simple_ret,
        "Ensemble (Weighted Avg)" : ens_weighted_ret,
        "Ensemble (Stacking)"     : ens_stacking_ret,
    }
    # val 세트에서 비교 (수익률 기준)
    val_rf_pred  = rf_ext.predict(X_val_ext)
    val_ann_pred = predict_ai_model(ann_ext, X_val_ext)
    ens_val_candidates = {
        "Ensemble (Simple Avg)"   : (val_rf_pred + val_ann_pred) / 2.0,
        "Ensemble (Weighted Avg)" : (w_rf * val_rf_pred + w_ann * val_ann_pred) / w_total,
        "Ensemble (Stacking)"     : meta_model.predict(np.column_stack([val_rf_pred, val_ann_pred])),
    }
    best_ens_name = min(
        ens_val_candidates,
        key=lambda k: np.sqrt(np.mean((y_val_np - ens_val_candidates[k]) ** 2))
    )
    best_ens_ret = ens_candidates[best_ens_name]
    print(f"  [앙상블 자동 선택] 최적 방식 = {best_ens_name}")
    print(f"    Simple Avg   val RMSE: {np.sqrt(np.mean((y_val_np - ens_val_candidates['Ensemble (Simple Avg)'])**2)):.6f}")
    print(f"    Weighted Avg val RMSE: {np.sqrt(np.mean((y_val_np - ens_val_candidates['Ensemble (Weighted Avg)'])**2)):.6f}")
    print(f"    Stacking     val RMSE: {np.sqrt(np.mean((y_val_np - ens_val_candidates['Ensemble (Stacking)'])**2)):.6f}")

    results_return["Ensemble RF+ANN"] = best_ens_ret
    results_price["Ensemble RF+ANN"]  = ret_to_price(best_ens_ret)

    # 7. Hybrid Sentiment RF (10 features + NSI_short + NSI_long)
    # NSI 결합 피처 정의
    HYBRID_FEATURES = EXTENDED_FEATURES + ["nsi_short", "nsi_long"]
    # 누락 NSI 100 보정
    train_df["nsi_short"] = train_df["nsi_short"].fillna(100.0)
    train_df["nsi_long"]  = train_df["nsi_long"].fillna(100.0)
    test_df["nsi_short"]  = test_df["nsi_short"].ffill().bfill().fillna(100.0)
    test_df["nsi_long"]   = test_df["nsi_long"].ffill().bfill().fillna(100.0)

    scaler_hyb  = MinMaxScaler()
    X_train_hyb = scaler_hyb.fit_transform(train_df[HYBRID_FEATURES])
    X_test_hyb  = scaler_hyb.transform(test_df[HYBRID_FEATURES])
    rf_hyb = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_hyb, y_train_r)
    rf_hyb_ret = rf_hyb.predict(X_test_hyb)
    results_return["Hybrid RF+NSI"] = rf_hyb_ret
    results_price["Hybrid RF+NSI"]  = ret_to_price(rf_hyb_ret)

    # ─── 평가 메트릭 계산 ───
    metrics_list = []
    rmse_dict = {}
    for model_name, price_pred in results_price.items():
        rmse_val = np.sqrt(mean_squared_error(y_test_price, price_pred))
        rmse_dict[model_name] = rmse_val
        
        # 회귀 지표
        metrics = regression_metrics(y_test_price, price_pred)
        # 방향 정확도 (가격 기준: 내일 종가가 오늘보다 올랐는지/내렸는지 판정)
        # ※ 수익률을 넣으면 "수익률 크기 비교"가 되어 방향성과 무관한 값이 나옴
        dir_acc = direction_accuracy(y_test_price, price_pred)
        metrics["direction_accuracy"] = dir_acc
        metrics["Model"] = model_name
        metrics["Company"] = name
        metrics_list.append(metrics)

    # ─── 시각화 (사진 포맷 일치) ───
    fig, ax = plt.subplots(figsize=(16, 7), dpi=120)
    
    # 1. 실제 종가 (굵은 검정)
    ax.plot(test_dates, y_test_price, label="Actual Close (실제 종가)", color="black", linewidth=2.5, zorder=5)
    # 2. Naive
    ax.plot(test_dates, results_price["Benchmark"], label=f"Naive Benchmark (전일 종가)  RMSE={rmse_dict['Benchmark']:,.0f}", color="dimgray", linestyle=":", linewidth=1.8)
    # 3. ARIMA
    ax.plot(test_dates, results_price["ARIMA"], label=f"ARIMA Forecast              RMSE={rmse_dict['ARIMA']:,.0f}", color="crimson", linestyle="-.", linewidth=1.6)
    # 4. Existing RF
    ax.plot(test_dates, results_price["Existing RF"], label=f"Existing RF  (7 features)   RMSE={rmse_dict['Existing RF']:,.0f}", color="darkorange", linestyle="--", linewidth=1.8)
    # 5. Existing ANN
    ax.plot(test_dates, results_price["Existing ANN"], label=f"Existing ANN (7 features)   RMSE={rmse_dict['Existing ANN']:,.0f}", color="coral", linestyle="--", linewidth=1.8)
    # 6. Extended RF
    ax.plot(test_dates, results_price["Extended RF"], label=f"Extended RF  (10 features)  RMSE={rmse_dict['Extended RF']:,.0f}", color="royalblue", linestyle="-", linewidth=2.0)
    # 7. Extended ANN
    ax.plot(test_dates, results_price["Extended ANN"], label=f"Extended ANN (10 features)  RMSE={rmse_dict['Extended ANN']:,.0f}", color="forestgreen", linestyle="-", linewidth=2.0)
    # 8. Ensemble RF+ANN (자동 선택된 최적 앙상블)
    ax.plot(test_dates, results_price["Ensemble RF+ANN"], label=f"Ensemble RF+ANN (best)      RMSE={rmse_dict['Ensemble RF+ANN']:,.0f}", color="deeppink", linestyle="-", linewidth=2.2, zorder=4)
    # 9. 최종 Hybrid RF+NSI (두껍고 눈에 띄는 퍼플 실선)
    ax.plot(test_dates, results_price["Hybrid RF+NSI"], label=f"Hybrid RF+NSI (12 features) RMSE={rmse_dict['Hybrid RF+NSI']:,.0f}", color="darkviolet", linestyle="-", linewidth=2.5, zorder=4)

    ax.set_title(f"Model Predictions Comparison — {name.replace('_', ' ').title()}", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (KRW)", fontsize=12)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85, edgecolor="gray", prop={"family": "monospace"})
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    # 성능 개선 상자
    improve_existing = (rmse_dict["Existing RF"] - rmse_dict["Hybrid RF+NSI"]) / rmse_dict["Existing RF"] * 100
    improve_naive = (rmse_dict["Benchmark"] - rmse_dict["Hybrid RF+NSI"]) / rmse_dict["Benchmark"] * 100
    note_text = (
        "[최종 Hybrid 개선율]\n"
        f"Existing RF 대비 개선 : {improve_existing:+.2f}%\n"
        f"Naive 대비 개선        : {improve_naive:+.2f}%"
    )
    ax.text(0.99, 0.02, note_text, transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    pred_graph_path = os.path.join(FIGURES_DIR, f"{name}_final_comparison.png")
    fig.savefig(pred_graph_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [차트 저장 완료] -> {pred_graph_path}")

    return metrics_list

# ═══════════════════════════════════════════
# 3. HTML 보고서 생성 함수
# ═══════════════════════════════════════════

def generate_html_report(df_summary, figures_dir, report_path):
    overall_metrics = df_summary.groupby("Model").mean().reset_index()
    overall_metrics = overall_metrics[["Model", "rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]]
    overall_metrics = overall_metrics.round({"rmse": 2, "mae": 2, "mape": 2, "mbe": 2, "r2": 4, "direction_accuracy": 4})

    company_table = df_summary.copy()
    company_table = company_table[["Company", "Model", "rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]]
    company_table = company_table.round({"rmse": 2, "mae": 2, "mape": 2, "mbe": 2, "r2": 4, "direction_accuracy": 4})

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>Hybrid Final Comparison Report</title>
  <style>
    body {{ font-family: 'Malgun Gothic', Arial, sans-serif; margin: 24px; background: #f8fafc; color: #111827; }}
    h1, h2 {{ color: #1f2937; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 10px; text-align: center; }}
    th {{ background: #111827; color: #f8fafc; }}
    tr:nth-child(even) {{ background: #f3f4f6; }}
    .section {{ margin-bottom: 40px; }}
    .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 24px; }}
    .figure-card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08); padding: 16px; }}
    .figure-card img {{ width: 100%; height: auto; border-radius: 6px; }}
    .figure-card h3 {{ margin: 0 0 10px; font-size: 1rem; color: #111827; }}
    .footer {{ font-size: 0.95rem; color: #475569; margin-top: 32px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #2563eb; color: white; font-size: 0.85rem; margin-right: 8px; }}
  </style>
</head>
<body>
  <h1>Hybrid Final Comparison Report</h1>
  <p>이 보고서는 <strong>6개 모델</strong>의 최종 예측 성능을 종합 비교하며, 각 모델의 평균 지표와 회사별 지표를 모두 제공합니다.</p>

  <div class="section">
    <h2>1. 모델별 평균 성능 지표</h2>
    {overall_metrics.to_html(index=False, classes='summary-table', border=0, justify='center')}
  </div>

  <div class="section">
    <h2>2. 회사별 모델 성능 지표</h2>
    {company_table.to_html(index=False, classes='company-table', border=0, justify='center')}
  </div>

  <div class="section">
    <h2>3. 대상 종목별 비교 차트</h2>
    <div class="figure-grid">
"""

    for company in sorted(df_summary["Company"].unique()):
        image_path = os.path.join(figures_dir, f"{company}_final_comparison.png")
        if os.path.exists(image_path):
            html += f"""
      <div class="figure-card">
        <h3>{company.replace('_', ' ').title()}</h3>
        <img src="figures/{company}_final_comparison.png" alt="{company} final comparison" />
      </div>
"""

    html += f"""
    </div>
    <div class="footer">
      <p>생성 경로: {report_path}</p>
      <p>데이터 출처: data/raw/*.csv 및 data/sentiment/macro_news_counts_90d.csv</p>
    </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [HTML 보고서 생성 완료] -> {report_path}")

# ═══════════════════════════════════════════
# 4. 메인 실행
# ═══════════════════════════════════════════

def main():
    print("="*80)
    print("   [6대 주가 예측 모델 종합 성능 비교 파이프라인 (심리지수 통합)]   ")
    print("="*80)

    if not os.path.exists(NEWS_PATH):
        print("[오류] 뉴스 기사 CSV 파일이 누락되었습니다. 크롤러를 먼저 가동해 주세요.")
        return

    # 설문 가중치 및 뉴스 로드
    short_w, long_w = load_weights(EXCEL_PATH)
    news_df = pd.read_csv(NEWS_PATH)
    news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")

    all_metrics = []

    for name, path in TARGET_COMPANIES.items():
        if not os.path.exists(path):
            print(f"[경고] {path} 파일 없음, 스킵")
            continue
        try:
            metrics_list = run_comparison_pipeline(name, path, short_w, long_w, news_df)
            all_metrics.extend(metrics_list)
        except Exception as e:
            print(f"[{name}] 오류 발생: {e}")

    # 결과 취합 저장
    if all_metrics:
        df_summary = pd.DataFrame(all_metrics)
        df_summary = df_summary[["Company", "Model", "rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]]
        summary_path = os.path.join(METRICS_DIR, "hybrid_final_comparison_summary.csv")
        df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        generate_html_report(df_summary, FIGURES_DIR, REPORT_PATH)

        print("\n" + "="*80)
        print("   [6대 모델 종합 비교 완료 및 성능 지표 테이블]")
        print("="*80)
        print(f"CSV 결과 저장 경로: {summary_path}")
        print(f"HTML 보고서 저장 경로: {REPORT_PATH}\n")
        print(df_summary.to_string(index=False))
        print("="*80)
    else:
        print("[오류] 수집된 결과 지표가 없습니다.")


if __name__ == "__main__":
    main()
