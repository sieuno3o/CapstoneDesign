from src.train import train_all_models
from src.train_ai_pipeline import train_all_ai_models

def run_rf_extended():
    """확장 RF 파이프라인 (run_rf_extended.py 로직을 main에서 직접 호출)"""
    import os
    from pathlib import Path
    from src.train_rf_extended import train_rf_extended_pipeline
    import pandas as pd

    stocks = {
        "samsung_electronics": "data/raw/samsung_electronics_5y.csv",
        "sk_hynix": "data/raw/sk_hynix_5y.csv",
        "wonik_ips": "data/raw/wonik_ips_5y.csv",
        "ia": "data/raw/ia_5y.csv",
        "hanwha_aerospace": "data/raw/hanwha_aerospace_5y.csv",
        "lig_nex1": "data/raw/lig_nex1_5y.csv",
        "snt_dynamics": "data/raw/snt_dynamics_5y.csv",
        "firstec": "data/raw/firstec_5y.csv",
        "rtx": "data/raw/rtx_5y.csv",
        "aerovironment": "data/raw/aerovironment_5y.csv",
        "draganfly": "data/raw/draganfly_5y.csv",
        "nvidia": "data/raw/nvidia_5y.csv",
        "axt": "data/raw/axt_5y.csv",
        "maxlinear": "data/raw/maxlinear_5y.csv",
    }

    all_results = []
    for name, path in stocks.items():
        if not os.path.exists(path):
            print(f"[경고] {path} 파일 없음, 스킵")
            continue
        try:
            df_metrics = train_rf_extended_pipeline(name, path)
            df_metrics["Company"] = name
            all_results.append(df_metrics)
        except Exception as e:
            print(f"[{name}] 오류 발생: {e}")

    if all_results:
        df_summary = pd.concat(all_results, ignore_index=True)
        Path("results/metrics").mkdir(parents=True, exist_ok=True)
        df_summary.to_csv("results/metrics/rf_extended_models_summary.csv", index=False, encoding="utf-8-sig")


def main():
    print("4개 모델 비교 프로젝트 시작")
    print("================================")
    print("1. [실행됨] 고전적 모델 (ARIMA 파이프라인)")
    print("2. [실행됨] AI 기반 모델 (RF & ANN)")
    print("2-1. [실행됨] 확장 RF 모델 (기술적 지표 추가)")
    print("3. [대기중] 심리 지수만 이용한 모델")
    print("4. [대기중] AI + 심리 지수 결합 최종 모델")
    print("================================\n")

    # 1. 고전적 모델(ARIMA) 14개 종목 순회 파이프라인 실행
    train_all_models()

    # 2. AI 기반 모델(RF, ANN) 14개 종목 순회 파이프라인 실행
    train_all_ai_models()

    # 2-1. 확장 RF 모델 (기술적 지표 추가 버전)
    run_rf_extended()


if __name__ == "__main__":
    main()