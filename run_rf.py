from src.train_ai_pipeline import train_ai_pipeline

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
    for name, path in stocks.items():
        print("=" * 80)
        print(f"[실행 시작] {name}")
        print("=" * 80)

        try:
            train_ai_pipeline(name, path)
        except Exception as e:
            print(f"[오류] {name}: {e}")
