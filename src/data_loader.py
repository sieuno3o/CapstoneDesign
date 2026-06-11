from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime


def load_price_data(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    df = pd.read_csv(file_path)
    if "Date" not in df.columns:
        raise ValueError("Date 컬럼이 필요합니다.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def load_sentiment_data(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    df = pd.read_csv(file_path)
    if "Date" not in df.columns:
        raise ValueError("Date 컬럼이 필요합니다.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def download_yahoo_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    yfinance 라이브러리를 사용하여 시계열 주가(OHLCV) 데이터를 다운로드합니다.
    """
    print(f"[INFO] Downloading {ticker} data for {period}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    
    if df.empty:
        raise ValueError(f"{ticker}의 데이터를 불러오지 못했습니다. Ticker를 확인하세요.")
        
    df = df.reset_index()
    
    # 시간대(Timezone) 정보 제거
    if pd.api.types.is_datetime64tz_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.tz_localize(None)
        
    # 날짜 기준으로 정렬
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"[INFO] {ticker} 데이터 다운로드 완료 (Shape: {df.shape})")
    
    return df