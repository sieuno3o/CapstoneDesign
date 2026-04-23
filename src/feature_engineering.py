import pandas as pd


def add_moving_averages(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    result = df.copy()
    result["ma_5"] = result[price_col].rolling(5).mean()
    result["ma_20"] = result[price_col].rolling(20).mean()
    result["ma_60"] = result[price_col].rolling(60).mean()
    return result


def add_volatility(df: pd.DataFrame, return_col: str = "log_return") -> pd.DataFrame:
    result = df.copy()
    result["volatility_5"] = result[return_col].rolling(5).std()
    result["volatility_20"] = result[return_col].rolling(20).std()
    return result


def add_price_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["hl_diff"] = result["High"] - result["Low"]
    result["oc_diff"] = result["Open"] - result["Close"]
    result["close_ma5_gap"] = result["Close"] - result["ma_5"]
    result["close_ma20_gap"] = result["Close"] - result["ma_20"]
    return result


def merge_price_and_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(price_df, sentiment_df, on="Date", how="left")
    return merged
