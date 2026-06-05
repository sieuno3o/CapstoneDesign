import pandas as pd
import numpy as np

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's smoothed RSI (standard financial calculation equivalent to TA-Lib).
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's exponential smoothing
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical and volume-based features to the OHLCV dataframe.
    Calculates RSI, MACD, Bollinger Bands, Moving Averages, Volatility, and Volume indicators.
    """
    res = df.copy()
    
    # Ensure Date column is standard
    if "Date" in res.columns:
        res["Date"] = pd.to_datetime(res["Date"])
    
    # --- 1. RSI ---
    res["RSI_14"] = calculate_rsi(res["Close"], period=14)
    
    # --- 2. MACD ---
    ema_12 = res["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = res["Close"].ewm(span=26, adjust=False).mean()
    res["MACD"] = ema_12 - ema_26
    res["MACD_signal"] = res["MACD"].ewm(span=9, adjust=False).mean()
    res["MACD_hist"] = res["MACD"] - res["MACD_signal"]
    
    # --- 3. Bollinger Bands ---
    ma_20 = res["Close"].rolling(window=20).mean()
    std_20 = res["Close"].rolling(window=20).std()
    res["BB_upper"] = ma_20 + 2 * std_20
    res["BB_lower"] = ma_20 - 2 * std_20
    res["BB_width"] = (res["BB_upper"] - res["BB_lower"]) / ma_20
    res["BB_percent"] = (res["Close"] - res["BB_lower"]) / (res["BB_upper"] - res["BB_lower"])
    
    # --- 4. Moving Averages & Trend Ratios ---
    res["MA_3"] = res["Close"].rolling(window=3).mean()
    res["MA_5"] = res["Close"].rolling(window=5).mean()
    res["MA_10"] = res["Close"].rolling(window=10).mean()
    res["MA_20"] = ma_20  # Use already calculated 20-day moving average
    
    res["Close_MA5_ratio"] = res["Close"] / res["MA_5"]
    res["Close_MA20_ratio"] = res["Close"] / res["MA_20"]
    res["MA5_MA20_gap"] = res["MA_5"] - res["MA_20"]
    
    # --- 5. Volatility & Price Ratios ---
    res["daily_return"] = res["Close"].pct_change()
    res["abs_return"] = res["daily_return"].abs()
    res["high_low_ratio"] = (res["High"] - res["Low"]) / res["Close"]
    res["open_close_ratio"] = (res["Close"] - res["Open"]) / res["Open"]
    # pct_volatility_* : daily_return(pct_change) 기반 변동성
    # volatility_7(log_return 기반)과 구분하기 위해 pct_ 접두사 사용
    res["pct_volatility_5"] = res["daily_return"].rolling(window=5).std()
    res["pct_volatility_10"] = res["daily_return"].rolling(window=10).std()
    
    # --- 6. Volume & Liquidity ---
    res["Volume_MA5"] = res["Volume"].rolling(window=5).mean()
    res["Volume_MA20"] = res["Volume"].rolling(window=20).mean()
    res["Volume_change"] = res["Volume"].pct_change()
    res["Volume_ratio"] = res["Volume"] / res["Volume_MA20"]
    # Trading_value = Close * Volume 은 값이 매우 크므로 로그 변환 적용
    res["log_trading_value"] = np.log1p(res["Close"] * res["Volume"])

    # --- 7. ATR (Average True Range) ---
    # 고가/저가/전일종가를 모두 활용하는 변동성 지표
    prev_close = res["Close"].shift(1)
    tr = pd.concat([
        res["High"] - res["Low"],
        (res["High"] - prev_close).abs(),
        (res["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    res["ATR_14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # --- 8. OBV (On-Balance Volume) ---
    # 가격 상승일 거래량 누적 - 가격 하락일 거래량 누적
    direction = np.sign(res["Close"].diff())
    res["OBV"] = (direction * res["Volume"]).cumsum()

    # --- 9. Stochastic Oscillator (%K, %D) ---
    period_k = 14
    lowest_low = res["Low"].rolling(window=period_k).min()
    highest_high = res["High"].rolling(window=period_k).max()
    res["Stoch_K"] = 100 * (res["Close"] - lowest_low) / (highest_high - lowest_low)
    res["Stoch_D"] = res["Stoch_K"].rolling(window=3).mean()  # %D = %K의 3일 이동평균

    # --- 10. Lag Return Features & MA Return ---
    # 과거 수익률 정보를 직접 피처로 제공 (단기 추세 기억 효과)
    res["lag_1_return"] = res["daily_return"].shift(1)
    res["lag_2_return"] = res["daily_return"].shift(2)
    res["lag_3_return"] = res["daily_return"].shift(3)
    # MA3_return: 3일 수익률 이동평균 (단기 모멘텀 스무딩 효과)
    res["MA3_return"] = res["daily_return"].rolling(window=3).mean()

    return res
