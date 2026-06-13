# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
BASE_DIR = Path(__file__).resolve().parents[2]
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# 인코딩 설정
sys.stdout.reconfigure(encoding="utf-8")

# 폰트 설정 (Windows 기본 한글 폰트 적용)
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. K-NSI 데이터 로드 및 지수 계산
news_path = str(BASE_DIR / "data/sentiment/macro_news_counts_1st.csv")
news_df = pd.read_csv(news_path)
news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")
news_indexed = news_df.set_index("date")

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

nsi_s = calc_nsi(news_indexed, short_w)
nsi_l = calc_nsi(news_indexed, long_w)

# 대상 종목 정의
COMPANIES = {
    "samsung_electronics": str(BASE_DIR / "data/raw/samsung_electronics_5y.csv"),
    "sk_hynix":            str(BASE_DIR / "data/raw/sk_hynix_5y.csv"),
    "wonik_ips":           str(BASE_DIR / "data/raw/wonik_ips_5y.csv"),
    "dongjin_semichem":    str(BASE_DIR / "data/raw/dongjin_semichem_5y.csv"),
    "hanwha_aerospace":    str(BASE_DIR / "data/raw/hanwha_aerospace_5y.csv"),
    "lig_nex1":            str(BASE_DIR / "data/raw/lig_nex1_5y.csv"),
    "snt_dynamics":        str(BASE_DIR / "data/raw/snt_dynamics_5y.csv"),
    "firstec":             str(BASE_DIR / "data/raw/firstec_5y.csv"),
}

# 기존 Price Only 모델들의 학습 피처들 가져오기
from src.preprocess import add_return_features, add_target_next_return, add_target_next_close
from src.feature_engineering import add_moving_averages, add_volatility, add_price_structure_features
from src.feature_engineering_extended import add_extended_features
from src.classical_model import train_classical_model, predict_classical_model
from src.ai_model import train_ann_model

ORIGINAL_FEATURES = ["Volume", "ma_7", "ma_14", "ma_21", "volatility_7", "hl_diff", "oc_diff"]
EXTENDED_FEATURES = ["MA_3", "MA_5", "MA_10", "MA_20", "BB_upper", "OBV", "ATR_14", "Volume", "hl_diff", "volatility_7"]

def direction_accuracy(y_true, y_pred, y_today):
    true_dir = (y_true > y_today).astype(int)
    pred_dir = (y_pred > y_today).astype(int)
    return np.mean(true_dir == pred_dir)

all_results = []
plot_data = {}

print("=========================================================================")
print("      [K-NSI 뉴스 심리 선형 모델 vs 기존 가격 모델 4개년 통합 평가]")
print("=========================================================================")

for name, path in COMPANIES.items():
    if not os.path.exists(path):
        print(f"[경고] {name} 주가 파일이 존재하지 않습니다.")
        continue
        
    # 데이터 로드
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # 지표들 미리 생성 (기존 Price Only 피처용)
    df = add_return_features(df)
    df = add_target_next_return(df)
    df = add_target_next_close(df)
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_price_structure_features(df)
    df = add_extended_features(df)
    df = df.dropna().reset_index(drop=True)
    
    df["date_key"] = df["Date"].dt.strftime("%Y-%m-%d")
    df["nsi_short"] = df["date_key"].map(nsi_s)
    df["nsi_long"]  = df["date_key"].map(nsi_l)
    
    valid_news_dates = news_df["date"].values
    matched_df = df[df["date_key"].isin(valid_news_dates)].copy().reset_index(drop=True)
    
    # 7:3 시계열 분할
    split_idx = int(len(matched_df) * 0.7)
    train_df = matched_df.iloc[:split_idx]
    test_df  = matched_df.iloc[split_idx:]
    
    y_train_r = train_df["target_next_return"].values
    y_test_price = test_df["target_next_close"].values
    today_close = test_df["Close"].values
    
    # 1. Benchmark (Naive)
    bench_rmse = np.sqrt(np.mean((y_test_price - today_close)**2))
    bench_mae = np.mean(np.abs(y_test_price - today_close))
    bench_mape = np.mean(np.abs((y_test_price - today_close) / y_test_price)) * 100
    bench_acc = 0.50
    all_results.append({"Company": name, "Model": "Benchmark", "RMSE": bench_rmse, "MAE": bench_mae, "MAPE": bench_mape, "Acc": bench_acc})
    
    # 2. ARIMA
    try:
        arima_fitted = train_classical_model(train_df["Close"])
        arima_pred = predict_classical_model(arima_fitted, steps=len(test_df))
        arima_rmse = np.sqrt(np.mean((y_test_price - arima_pred)**2))
        arima_mae = np.mean(np.abs(y_test_price - arima_pred))
        arima_mape = np.mean(np.abs((y_test_price - arima_pred) / y_test_price)) * 100
        arima_acc = direction_accuracy(y_test_price, arima_pred, today_close)
    except Exception as e:
        arima_rmse, arima_mae, arima_mape, arima_acc = np.nan, np.nan, np.nan, np.nan
    all_results.append({"Company": name, "Model": "ARIMA", "RMSE": arima_rmse, "MAE": arima_mae, "MAPE": arima_mape, "Acc": arima_acc})
    
    # 3. Existing RF (Price Only - 7 features)
    scaler_orig = MinMaxScaler()
    X_train_orig = scaler_orig.fit_transform(train_df[ORIGINAL_FEATURES])
    X_test_orig  = scaler_orig.transform(test_df[ORIGINAL_FEATURES])
    rf_orig = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_orig, y_train_r)
    rf_orig_price = today_close * np.exp(rf_orig.predict(X_test_orig))
    rf_orig_rmse = np.sqrt(np.mean((y_test_price - rf_orig_price)**2))
    rf_orig_mae = np.mean(np.abs(y_test_price - rf_orig_price))
    rf_orig_mape = np.mean(np.abs((y_test_price - rf_orig_price) / y_test_price)) * 100
    rf_orig_acc = direction_accuracy(y_test_price, rf_orig_price, today_close)
    all_results.append({"Company": name, "Model": "Existing RF", "RMSE": rf_orig_rmse, "MAE": rf_orig_mae, "MAPE": rf_orig_mape, "Acc": rf_orig_acc})
    
    # 3.5 Existing ANN (Price Only - 7 features)
    try:
        ann_orig_model = train_ann_model(X_train_orig, y_train_r, X_train_orig, y_train_r)
        ann_orig_pred_price = today_close * np.exp(ann_orig_model.predict(X_test_orig).flatten())
        ann_orig_rmse = np.sqrt(np.mean((y_test_price - ann_orig_pred_price)**2))
        ann_orig_mae = np.mean(np.abs(y_test_price - ann_orig_pred_price))
        ann_orig_mape = np.mean(np.abs((y_test_price - ann_orig_pred_price) / y_test_price)) * 100
        ann_orig_acc = direction_accuracy(y_test_price, ann_orig_pred_price, today_close)
    except Exception as e:
        ann_orig_rmse, ann_orig_mae, ann_orig_mape, ann_orig_acc = np.nan, np.nan, np.nan, np.nan
    all_results.append({"Company": name, "Model": "Existing ANN", "RMSE": ann_orig_rmse, "MAE": ann_orig_mae, "MAPE": ann_orig_mape, "Acc": ann_orig_acc})

    # 4. Extended RF (Price Only - 10 features)
    scaler_ext = MinMaxScaler()
    X_train_ext = scaler_ext.fit_transform(train_df[EXTENDED_FEATURES])
    X_test_ext  = scaler_ext.transform(test_df[EXTENDED_FEATURES])
    rf_ext = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train_ext, y_train_r)
    rf_ext_price = today_close * np.exp(rf_ext.predict(X_test_ext))
    rf_ext_rmse = np.sqrt(np.mean((y_test_price - rf_ext_price)**2))
    rf_ext_mae = np.mean(np.abs(y_test_price - rf_ext_price))
    rf_ext_mape = np.mean(np.abs((y_test_price - rf_ext_price) / y_test_price)) * 100
    rf_ext_acc = direction_accuracy(y_test_price, rf_ext_price, today_close)
    all_results.append({"Company": name, "Model": "Extended RF", "RMSE": rf_ext_rmse, "MAE": rf_ext_mae, "MAPE": rf_ext_mape, "Acc": rf_ext_acc})
    
    # 5. Extended ANN (Price Only - 10 features)
    try:
        ann_model = train_ann_model(X_train_ext, y_train_r, X_train_ext, y_train_r)
        ann_pred_price = today_close * np.exp(ann_model.predict(X_test_ext).flatten())
        ann_rmse = np.sqrt(np.mean((y_test_price - ann_pred_price)**2))
        ann_mae = np.mean(np.abs(y_test_price - ann_pred_price))
        ann_mape = np.mean(np.abs((y_test_price - ann_pred_price) / y_test_price)) * 100
        ann_acc = direction_accuracy(y_test_price, ann_pred_price, today_close)
    except Exception as e:
        ann_rmse, ann_mae, ann_mape, ann_acc = np.nan, np.nan, np.nan, np.nan
    all_results.append({"Company": name, "Model": "Extended ANN", "RMSE": ann_rmse, "MAE": ann_mae, "MAPE": ann_mape, "Acc": ann_acc})
    
    # 6. Linear Regression (Sentiment Only - K-NSI 2 features)
    scaler_nsi = MinMaxScaler()
    X_train_nsi = scaler_nsi.fit_transform(train_df[["nsi_short", "nsi_long"]])
    X_test_nsi  = scaler_nsi.transform(test_df[["nsi_short", "nsi_long"]])
    lr_nsi = LinearRegression().fit(X_train_nsi, y_train_r)
    lr_nsi_price = today_close * np.exp(lr_nsi.predict(X_test_nsi))
    lr_nsi_rmse = np.sqrt(np.mean((y_test_price - lr_nsi_price)**2))
    lr_nsi_mae = np.mean(np.abs(y_test_price - lr_nsi_price))
    lr_nsi_mape = np.mean(np.abs((y_test_price - lr_nsi_price) / y_test_price)) * 100
    lr_nsi_acc = direction_accuracy(y_test_price, lr_nsi_price, today_close)
    all_results.append({"Company": name, "Model": "Sentiment Only (LR)", "RMSE": lr_nsi_rmse, "MAE": lr_nsi_mae, "MAPE": lr_nsi_mape, "Acc": lr_nsi_acc})
    
    print(f"[{name}] 완료. 데이터: {len(matched_df)}일")
    
    # SK하이닉스만 따로 결과 보관 (그래프용)
    if name == "sk_hynix":
        plot_data["Date"] = test_df["Date"]
        plot_data["Actual"] = y_test_price
        plot_data["Benchmark"] = today_close
        plot_data["ARIMA"] = arima_pred if 'arima_pred' in locals() else today_close
        plot_data["Extended_RF"] = rf_ext_price
        plot_data["Sentiment_LR"] = lr_nsi_price

# CSV 요약 저장
df_res = pd.DataFrame(all_results)
os.makedirs(str(BASE_DIR / "results/metrics"), exist_ok=True)
df_res.to_csv(str(BASE_DIR / "results/metrics/sentiment_comparison_summary.csv"), index=False, encoding="utf-8-sig")
print("\n- 성능 결과 저장 완료: results/metrics/sentiment_comparison_summary.csv")

# 요약 출력
print("\n" + "="*80)
print("      [K-NSI 뉴스 심리 선형 모델 vs 기존 모델 방향 정확도(Acc) 비교]")
print("" + "="*80)
for name in COMPANIES.keys():
    sub = df_res[df_res["Company"] == name]
    print(f"[{name}]")
    for idx, row in sub.iterrows():
        print(f"  - {row['Model']}: Acc={row['Acc']*100.0:.2f}% | RMSE={row['RMSE']:.2f}")

# 그래프 드로잉 (SK하이닉스 대표 시각화)
if plot_data:
    plt.figure(figsize=(14, 7))
    plt.plot(plot_data["Date"], plot_data["Actual"], label="Actual (실제 내일 종가)", color="#0f172a", linewidth=2.2)
    plt.plot(plot_data["Date"], plot_data["Sentiment_LR"], label="Sentiment Only (LR) 예측", color="#7c3aed", linestyle="--", linewidth=2.0)
    plt.plot(plot_data["Date"], plot_data["Extended_RF"], label="Extended RF (Price Only)", color="#3b82f6", linestyle=":", alpha=0.8, linewidth=1.5)
    plt.plot(plot_data["Date"], plot_data["ARIMA"], label="ARIMA 예측", color="#10b981", linestyle="-.", alpha=0.6, linewidth=1.2)
    plt.plot(plot_data["Date"], plot_data["Benchmark"], label="Naive Benchmark (오늘 종가)", color="#f59e0b", alpha=0.7, linewidth=1.2)
    
    plt.title("SK하이닉스 주가 예측 모델 비교 (Sentiment Only vs Others)", fontsize=15, fontweight='bold', pad=15, color="#0f172a")
    plt.xlabel("날짜 (Date)", fontsize=11, labelpad=10, color="#0f172a")
    plt.ylabel("주가 (Price, 원)", fontsize=11, labelpad=10, color="#0f172a")
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1", labelcolor="#0f172a")
    plt.grid(True, color="#e2e8f0", linestyle="-")
    
    os.makedirs(str(BASE_DIR / "results/figures"), exist_ok=True)
    fig_path = str(BASE_DIR / "results/figures/nsi_comparison_chart.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"- 시각화 차트 저장 완료: {fig_path}")
