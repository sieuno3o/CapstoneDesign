import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def perform_adf_test(series: pd.Series, significance_level: float = 0.05):
    """
    주어진 시계열 데이터(Series)에 대해 Augmented Dickey-Fuller(ADF) Test를 수행합니다.
    p-value가 유의수준(significance_level)보다 작으면 정상성(Stationary)을 갖는다고 판단합니다.
    """
    result = adfuller(series.dropna())
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]

    print("=== Augmented Dickey-Fuller Test Results ===")
    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("Critical Values:")
    for key, value in critical_values.items():
        print(f"  {key}: {value:.4f}")
        
    if p_value < significance_level:
        print(f"결론: p-value < {significance_level} 이므로 귀무가설을 기각합니다. -> '정상성(Stationary) 데이터'입니다.")
        is_stationary = True
    else:
        print(f"결론: p-value >= {significance_level} 이므로 귀무가설을 기각하지 못합니다. -> '비정상성(Non-Stationary) 데이터'입니다.")
        is_stationary = False
        
    return {
        "adf_statistic": adf_statistic,
        "is_stationary": is_stationary
    }

def make_stationary(series: pd.Series, significance_level: float = 0.05):
    """
    원본 시계열 데이터가 비정상성일 경우, 1차 차분과 로그 차분을 수행하여 정상성을 재확인합니다.
    """
    print("\n--- [Step 1] 원본 데이터 정상성 확인 ---")
    res_original = perform_adf_test(series, significance_level)
    
    if res_original["is_stationary"]:
        print("=> 원본 데이터가 이미 정상성을 만족합니다. 변환이 필요하지 않습니다.")
        return series, "Original"
        
    print("\n--- [Step 2] 1차 차분(Differencing) 데이터 정상성 확인 ---")
    diff_series = series.diff().dropna()
    res_diff = perform_adf_test(diff_series, significance_level)
    
    if res_diff["is_stationary"]:
        print("=> 1차 차분 후 정상성을 만족합니다.")
        return diff_series, "Differenced"
        
    print("\n--- [Step 3] 로그 차분(Log Differencing) 데이터 정상성 확인 ---")
    # 값이 양수일 때만 로그 변환 가능
    if (series <= 0).any():
        print("경고: 0 이하의 값이 포함되어 있어 로그 변환을 수행할 수 없습니다.")
        return diff_series, "Differenced (Log failed)"
        
    log_diff_series = np.log(series).diff().dropna()
    res_log_diff = perform_adf_test(log_diff_series, significance_level)
    
    if res_log_diff["is_stationary"]:
        print("=> 로그 1차 차분 후 정상성을 만족합니다.")
        return log_diff_series, "Log Differenced"
        
    print("=> 경고: 1차 차분 및 로그 차분 후에도 정상성을 확보하지 못했습니다. (d=2 이상 필요할 수 있음)")
    return diff_series, "Differenced (Non-Stationary)"
