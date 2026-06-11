import pandas as pd


def add_moving_averages(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    result = df.copy()
    result["ma_7"] = result[price_col].rolling(7).mean()
    result["ma_14"] = result[price_col].rolling(14).mean()
    result["ma_21"] = result[price_col].rolling(21).mean()
    return result


def add_volatility(df: pd.DataFrame, return_col: str = "log_return") -> pd.DataFrame:
    result = df.copy()
    result["volatility_7"] = result[return_col].rolling(7).std()
    return result


def add_price_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["hl_diff"] = result["High"] - result["Low"]
    result["oc_diff"] = result["Open"] - result["Close"]
    return result


def merge_price_and_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.merge(price_df, sentiment_df, on="Date", how="left")
    return merged
