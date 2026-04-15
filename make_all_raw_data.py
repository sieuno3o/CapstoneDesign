import yfinance as yf
import pandas as pd
import os

stocks = {
    "hanwha_aerospace": {"ticker": "012450.KS", "country": "korea", "industry": "defense"},
    "lig_nex1": {"ticker": "079550.KS", "country": "korea", "industry": "defense"},
    "snt_dynamics": {"ticker": "003570.KS", "country": "korea", "industry": "defense"},
    "intellian_tech": {"ticker": "189300.KQ", "country": "korea", "industry": "defense"},
    "firstec": {"ticker": "010820.KS", "country": "korea", "industry": "defense"},
    "vigtec": {"ticker": "065450.KQ", "country": "korea", "industry": "defense"},

    "samsung_electronics": {"ticker": "005930.KS", "country": "korea", "industry": "semiconductor"},
    "sk_hynix": {"ticker": "000660.KS", "country": "korea", "industry": "semiconductor"},
    "wonik_ips": {"ticker": "240810.KQ", "country": "korea", "industry": "semiconductor"},
    "dongjin_semichem": {"ticker": "005290.KQ", "country": "korea", "industry": "semiconductor"},
    "ia": {"ticker": "038880.KQ", "country": "korea", "industry": "semiconductor"},

    "rtx": {"ticker": "RTX", "country": "us", "industry": "defense"},
    "lockheed_martin": {"ticker": "LMT", "country": "us", "industry": "defense"},
    "kratos": {"ticker": "KTOS", "country": "us", "industry": "defense"},
    "aerovironment": {"ticker": "AVAV", "country": "us", "industry": "defense"},
    "mercury_systems": {"ticker": "MRCY", "country": "us", "industry": "defense"},
    "draganfly": {"ticker": "DPRO", "country": "us", "industry": "defense"},

    "nvidia": {"ticker": "NVDA", "country": "us", "industry": "semiconductor"},
    "broadcom": {"ticker": "AVGO", "country": "us", "industry": "semiconductor"},
    "amd": {"ticker": "AMD", "country": "us", "industry": "semiconductor"},
    "axt": {"ticker": "AXTI", "country": "us", "industry": "semiconductor"},
    "onto_innovation": {"ticker": "ONTO", "country": "us", "industry": "semiconductor"},
    "ambarella": {"ticker": "AMBA", "country": "us", "industry": "semiconductor"},
    "maxlinear": {"ticker": "MXL", "country": "us", "industry": "semiconductor"},
    "indie_semiconductor": {"ticker": "INDI", "country": "us", "industry": "semiconductor"},
    "ceva": {"ticker": "CEVA", "country": "us", "industry": "semiconductor"},

    "huntington_ingalls": {"ticker": "HII", "country": "us", "industry": "marine"},
    "general_dynamics": {"ticker": "GD", "country": "us", "industry": "marine"},
    "kirby_corporation": {"ticker": "KEX", "country": "us", "industry": "marine"},
    "matson": {"ticker": "MATX", "country": "us", "industry": "marine"},
}

save_dir = "data/rawdata"
os.makedirs(save_dir, exist_ok=True)

for name, info in stocks.items():
    ticker = info["ticker"]
    print(f"\n===== {name} ({ticker}) 다운로드 시작 =====")

    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False)

    if df.empty:
        print(f"{name}: 데이터를 불러오지 못했습니다.")
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.reset_index(inplace=True)

    print(df.head())
    print(f"{name}: 데이터 확인 완료")

    #file_path = os.path.join(save_dir, f"{name}_5y.csv")
    #df.to_csv(file_path, index=False, encoding="utf-8-sig")
    #print(f"{name}: 저장 완료 -> {file_path}")