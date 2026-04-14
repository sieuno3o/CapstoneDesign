from pathlib import Path
import pandas as pd


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