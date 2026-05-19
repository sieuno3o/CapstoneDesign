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
    res["volatility_5"] = res["daily_return"].rolling(window=5).std()
    res["volatility_10"] = res["daily_return"].rolling(window=10).std()
    
    # --- 6. Volume & Liquidity ---
    res["Volume_MA5"] = res["Volume"].rolling(window=5).mean()
    res["Volume_MA20"] = res["Volume"].rolling(window=20).mean()
    res["Volume_change"] = res["Volume"].pct_change()
    res["Volume_ratio"] = res["Volume"] / res["Volume_MA20"]
    res["Trading_value"] = res["Close"] * res["Volume"]
    
    return res
