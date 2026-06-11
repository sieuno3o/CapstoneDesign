import os
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.train_rf_extended import train_rf_extended_pipeline

BASE_DIR = Path(__file__).resolve().parents[2]


# ── 분석 대상 종목 (교수님 피드백: 해외 기업 제거, 국내 기업 집중) ─────────────
# "기업을 너무 많이 할 필요 없어. 초점이 흐트러지니까." (2026.05.22 면담)
stocks = {
    # 국내 반도체
    "samsung_electronics": str(BASE_DIR / "data/raw/samsung_electronics_5y.csv"),
    "sk_hynix":            str(BASE_DIR / "data/raw/sk_hynix_5y.csv"),
    "wonik_ips":           str(BASE_DIR / "data/raw/wonik_ips_5y.csv"),
    "dongjin_semichem":    str(BASE_DIR / "data/raw/dongjin_semichem_5y.csv"),
    # 국내 방산
    "hanwha_aerospace":    str(BASE_DIR / "data/raw/hanwha_aerospace_5y.csv"),
    "lig_nex1":            str(BASE_DIR / "data/raw/lig_nex1_5y.csv"),
    "snt_dynamics":        str(BASE_DIR / "data/raw/snt_dynamics_5y.csv"),
    "firstec":             str(BASE_DIR / "data/raw/firstec_5y.csv"),
}

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except:
        pass

    print("=" * 80)
    print("      [머신러닝 입력변수 추가 및 비교 분석 프로젝트 실행]      ")
    print("=" * 80)

    # Ensure output directories exist
    (BASE_DIR / "data/processed").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results/metrics").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "results/figures").mkdir(parents=True, exist_ok=True)

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
        summary_save_path = str(BASE_DIR / "results/metrics/hybrid_comparison_summary.csv")
        df_summary.to_csv(summary_save_path, index=False, encoding="utf-8-sig")
        
        print("\n" + "=" * 80)
        print("      [종합 하이브리드 모델 예측 성능 비교 완료]      ")
        print("=" * 80)
        print(f"종합 결과 파일 저장 경로: {summary_save_path}\n")
        
        # 전체 종목 평균 성능 비교 테이블
        df_avg = df_summary.groupby("Model")[["rmse", "mae", "mape", "mbe", "r2", "direction_accuracy"]].mean().reset_index()
        print("### [전체 종목 평균 성능 비교 테이블]")
        print("-" * 80)
        print(df_avg.to_string(index=False))
        print("-" * 80)

        # Existing RF 및 Extended RF 대비 성능 개선 요약
        print("\n### [RMSE 성능 비교 테이블 (Hybrid RF+NSI 대비 개선율)]")
        print("-" * 80)
        df_exist = df_summary[df_summary["Model"] == "Existing RF"][["Company", "rmse"]].rename(columns={"rmse": "rmse_existing"})
        df_ext   = df_summary[df_summary["Model"] == "Extended RF"][["Company", "rmse"]].rename(columns={"rmse": "rmse_extended"})
        df_hyb   = df_summary[df_summary["Model"] == "Hybrid RF+NSI"][["Company", "rmse"]].rename(columns={"rmse": "rmse_hybrid"})
        df_naive = df_summary[df_summary["Model"] == "Benchmark"][["Company", "rmse"]].rename(columns={"rmse": "rmse_naive"})
        
        df_compare = pd.merge(df_exist, df_ext, on="Company")
        df_compare = pd.merge(df_compare, df_hyb, on="Company")
        df_compare = pd.merge(df_compare, df_naive, on="Company")

        # 개선율 계산
        df_compare["Hybrid_vs_Existing_%"] = (
            (df_compare["rmse_existing"] - df_compare["rmse_hybrid"]) / df_compare["rmse_existing"] * 100
        ).round(2)
        df_compare["Hybrid_vs_Extended_%"] = (
            (df_compare["rmse_extended"] - df_compare["rmse_hybrid"]) / df_compare["rmse_extended"] * 100
        ).round(2)
        df_compare["Hybrid_vs_Naive_%"] = (
            (df_compare["rmse_naive"] - df_compare["rmse_hybrid"]) / df_compare["rmse_naive"] * 100
        ).round(2)

        df_compare["Result_vs_Existing"] = df_compare["Hybrid_vs_Existing_%"].apply(lambda x: "개선" if x > 0 else "악화")
        df_compare["Result_vs_Extended"] = df_compare["Hybrid_vs_Extended_%"].apply(lambda x: "개선" if x > 0 else "악화")
        
        print(df_compare.to_string(index=False))
        print("-" * 80)
        
        hyb_improved_exist  = (df_compare["Hybrid_vs_Existing_%"] > 0).sum()
        hyb_improved_ext    = (df_compare["Hybrid_vs_Extended_%"] > 0).sum()
        total = len(df_compare)
        print(f"\n[Hybrid RF+NSI 성능 최종 분석]")
        print(f"  Existing RF (7개 변수) 대비 : {total}개 종목 중 {hyb_improved_exist}개 개선 / {total - hyb_improved_exist}개 악화")
        print(f"  Extended RF (10개 변수) 대비 : {total}개 종목 중 {hyb_improved_ext}개 개선 / {total - hyb_improved_ext}개 악화")
    else:
        print("[오류] 결과가 전혀 수집되지 않았습니다.")

