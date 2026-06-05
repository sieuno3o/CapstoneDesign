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

def add_target_next_close(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    result = df.copy()
    result["target_next_close"] = result[price_col].shift(-1)
    return result

def add_target_direction(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    result = df.copy()
    result["target_direction"] = (result[price_col].shift(-1) > result[price_col]).astype(int)
    return result


def add_target_next_return(df: pd.DataFrame, price_col: str = "Close") -> pd.DataFrame:
    """
    내일 수익률을 타겟으로 생성.
    target_next_return = (Close_{t+1} - Close_t) / Close_t
    - Naive 예측: 수익률 0% (변동 없음)
    - 예측 종가로 변환: today_close x (1 + predicted_return)
    """
    result = df.copy()
    result["target_next_return"] = (
        result[price_col].shift(-1) - result[price_col]
    ) / result[price_col]
    return result


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().reset_index(drop=True)
