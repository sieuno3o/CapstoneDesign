import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.train_ai_pipeline import train_ai_pipeline

BASE_DIR = Path(__file__).resolve().parents[2]

stocks = {
    "samsung_electronics": str(BASE_DIR / "data/raw/samsung_electronics_5y.csv"),
    "sk_hynix":             str(BASE_DIR / "data/raw/sk_hynix_5y.csv"),
    "wonik_ips":            str(BASE_DIR / "data/raw/wonik_ips_5y.csv"),
    "ia":                   str(BASE_DIR / "data/raw/ia_5y.csv"),
    "hanwha_aerospace":     str(BASE_DIR / "data/raw/hanwha_aerospace_5y.csv"),
    "lig_nex1":             str(BASE_DIR / "data/raw/lig_nex1_5y.csv"),
    "snt_dynamics":         str(BASE_DIR / "data/raw/snt_dynamics_5y.csv"),
    "firstec":              str(BASE_DIR / "data/raw/firstec_5y.csv"),
    "rtx":                  str(BASE_DIR / "data/raw/rtx_5y.csv"),
    "aerovironment":        str(BASE_DIR / "data/raw/aerovironment_5y.csv"),
    "draganfly":            str(BASE_DIR / "data/raw/draganfly_5y.csv"),
    "nvidia":               str(BASE_DIR / "data/raw/nvidia_5y.csv"),
    "axt":                  str(BASE_DIR / "data/raw/axt_5y.csv"),
    "maxlinear":            str(BASE_DIR / "data/raw/maxlinear_5y.csv"),
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
