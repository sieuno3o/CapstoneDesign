import numpy as np
import pandas as pd


def add_return_features(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    result = df.copy()
    result["simple_return"] = result[price_col].pct_change()
    result["log_return"] = np.log(result[price_col] / result[price_col].shift(1))
    return result


def add_target_next_open(df: pd.DataFrame, open_col: str = "Open") -> pd.DataFrame:
    result = df.copy()
    result["target_next_open"] = result[open_col].shift(-1)
    return result


def add_target_direction(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    result = df.copy()
    result["target_direction"] = (result[price_col].shift(-1) > result[price_col]).astype(int)
    return result


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)