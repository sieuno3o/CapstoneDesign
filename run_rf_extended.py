import os
import pandas as pd
from pathlib import Path
from src.train_rf_extended import train_rf_extended_pipeline

stocks = {
    "samsung_electronics": "data/rawdata/samsung_electronics_5y.csv",
    "sk_hynix": "data/rawdata/sk_hynix_5y.csv",
    "wonik_ips": "data/rawdata/wonik_ips_5y.csv",
    "ia": "data/rawdata/ia_5y.csv",
    "hanwha_aerospace": "data/rawdata/hanwha_aerospace_5y.csv",
    "lig_nex1": "data/rawdata/lig_nex1_5y.csv",
    "snt_dynamics": "data/rawdata/snt_dynamics_5y.csv",
    "firstec": "data/rawdata/firstec_5y.csv",
    "rtx": "data/rawdata/rtx_5y.csv",
    "aerovironment": "data/rawdata/aerovironment_5y.csv",
    "draganfly": "data/rawdata/draganfly_5y.csv",
    "nvidia": "data/rawdata/nvidia_5y.csv",
    "axt": "data/rawdata/axt_5y.csv",
    "maxlinear": "data/rawdata/maxlinear_5y.csv",
}

if __name__ == "__main__":
    print("=" * 80)
    print("      [머신러닝 입력변수 추가 및 비교 분석 프로젝트 실행]      ")
    print("=" * 80)

    # Ensure output directories exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)

    all_results = []

    for name, path in stocks.items():
        print("\n" + "#" * 80)
        print(f"## [종목 처리 중] {name.upper()} (데이터 경로: {path})")
        print("#" * 80)

        if not os.path.exists(path):
            print(f"[경고] {path} 파일이 존재하지 않아 스킵합니다.")
            continue

        try:
            # Run the training and evaluation pipeline
            df_metrics = train_rf_extended_pipeline(name, path)
            df_metrics["Company"] = name
            all_results.append(df_metrics)
        except Exception as e:
            print(f"[오류 발생] {name} 실행 중 예외 발생: {e}")

    if all_results:
        # Combine all metrics and save to results/metrics
        df_summary = pd.concat(all_results, ignore_index=True)
        summary_save_path = "results/metrics/ai_extended_models_summary.csv"
        df_summary.to_csv(summary_save_path, index=False, encoding="utf-8-sig")
        
        print("\n" + "=" * 80)
        print("      [종합 모델 예측 성능 비교 완료]      ")
        print("=" * 80)
        print(f"종합 결과 파일 저장 경로: {summary_save_path}\n")
        
        # 전체 종목 평균 성능 비교 테이블
        df_avg = df_summary.groupby("Model")[["rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]].mean().reset_index()
        print("### [전체 종목 평균 성능 비교 테이블]")
        print("-" * 80)
        print(df_avg.to_string(index=False))
        print("-" * 80)

        # 기본 RF vs 확장 RF 성능 개선 요약
        print("\n### [기본 RF → 확장 RF RMSE 개선율 (종목별)]")
        print("-" * 80)
        df_exist = df_summary[df_summary["Model"] == "Existing RF"][["Company", "rmse"]].rename(columns={"rmse": "rmse_existing"})
        df_ext   = df_summary[df_summary["Model"] == "Extended RF"][["Company", "rmse"]].rename(columns={"rmse": "rmse_extended_rf"})
        df_ann   = df_summary[df_summary["Model"] == "Extended ANN"][["Company", "rmse"]].rename(columns={"rmse": "rmse_extended_ann"})
        df_compare = pd.merge(df_exist, df_ext, on="Company")
        df_compare = pd.merge(df_compare, df_ann, on="Company")
        df_compare["RF_improvement_%"] = (
            (df_compare["rmse_existing"] - df_compare["rmse_extended_rf"]) / df_compare["rmse_existing"] * 100
        ).round(2)
        df_compare["ANN_improvement_%"] = (
            (df_compare["rmse_existing"] - df_compare["rmse_extended_ann"]) / df_compare["rmse_existing"] * 100
        ).round(2)
        df_compare["RF의과자"] = df_compare["RF_improvement_%"].apply(lambda x: "✅ 개선" if x > 0 else "❌ 악화")
        df_compare["ANN의과자"] = df_compare["ANN_improvement_%"].apply(lambda x: "✅ 개선" if x > 0 else "❌ 악화")
        print(df_compare.to_string(index=False))
        print("-" * 80)
        rf_improved  = (df_compare["RF_improvement_%"] > 0).sum()
        ann_improved = (df_compare["ANN_improvement_%"] > 0).sum()
        total = len(df_compare)
        print(f"\n확장 RF : {total}개 종목 중 {rf_improved}개 개선 / {total - rf_improved}개 악화")
        print(f"확장 ANN: {total}개 종목 중 {ann_improved}개 개선 / {total - ann_improved}개 악화")
    else:
        print("[오류] 결과가 전혀 수집되지 않았습니다.")

