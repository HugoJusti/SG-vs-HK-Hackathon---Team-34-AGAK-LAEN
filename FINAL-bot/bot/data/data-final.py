"""
Data module: historical data fetching, feature engineering, and live ticker.
"""
import os
import time
import numpy as np
import pandas as pd
import requests
import logging
from datetime import datetime, timedelta
from config import (
    DATA_DIR, PAIRS, HMM_TRAINING_HOURS,
    ENTRY_MA_SHORT, EXIT_MA_LONG
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# BINANCE HISTORICAL DATA (for HMM training)
# ═══════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str = "5m",
                         limit_hours: int = HMM_TRAINING_HOURS) -> pd.DataFrame:
    """
    Fetch historical klines from Binance public API.
    symbol: e.g. "BTCUSDT", "ETHUSDT"
    interval: "5m", "1h", "4h", "1d", etc.
    Returns DataFrame with OHLCV + timestamp.
    """
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []

    # Binance returns max 1000 candles per request, so we paginate
    end_time = int(time.time() * 1000)
    start_time = end_time - (limit_hours * 3600 * 1000)
    current_start = start_time

    logger.info(f"Fetching {limit_hours} hours of {interval} klines for {symbol}...")

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_klines.extend(data)
            # Move start to after last candle
            current_start = data[-1][0] + 1

            # Be nice to Binance API
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API error: {e}")
            time.sleep(1)
            continue

    if not all_klines:
        logger.error(f"No kline data fetched for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)

    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    logger.info(f"Fetched {len(df)} candles for {symbol}")
    return df


def pair_to_binance_symbol(pair: str) -> str:
    """Convert 'BTC/USD' → 'BTCUSDT' for Binance API."""
    coin = pair.split("/")[0]
    return f"{coin}USDT"


def load_or_fetch_historical(pair: str, force_refresh: bool = False) -> pd.DataFrame:
    """Load cached data or fetch fresh from Binance."""
    os.makedirs(DATA_DIR, exist_ok=True)
    symbol = pair_to_binance_symbol(pair)
    cache_path = os.path.join(DATA_DIR, f"{symbol}_1h.csv")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["open_time"])
        age_hours = (datetime.utcnow() - df["open_time"].iloc[-1]).total_seconds() / 3600
        if age_hours < 24:
            logger.info(f"Using cached data for {symbol} ({len(df)} candles)")
            return df

    df = fetch_binance_klines(symbol, "5m", HMM_TRAINING_HOURS)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (for HMM observations)
# ═══════════════════════════════════════════════════════════════

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 4 HMM observation features + EMA indicators.

    Features:
      1. log_return:    ln(close_t / close_{t-1})
      2. rolling_vol:   20-period rolling std of log returns (annualised)
      3. momentum:      10-period rate of change
      4. volume_zscore: (volume - rolling_mean) / rolling_std

    Indicators (not HMM features, used as trade filters):
      - ma20: 20-period EMA
      - ma50: 50-period EMA
    """
    df = df.copy()

    # 1. Log returns
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 2. Rolling volatility (20-period, annualised for 5m data)
    #    Annualisation factor: sqrt(12 * 24 * 365) = sqrt(105120)
    df["rolling_vol"] = df["log_return"].rolling(window=20).std() * np.sqrt(105120)

    # 3. Momentum (10-period Rate of Change)
    df["momentum"] = df["close"].pct_change(periods=10)

    # 4. Volume z-score (20-period rolling)
    vol_mean = df["volume"].rolling(window=20).mean()
    vol_std = df["volume"].rolling(window=20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std

    # EMAs (exponential moving averages — trade filters, not HMM features)
    df["ma20"] = df["close"].ewm(span=ENTRY_MA_SHORT, adjust=False).mean()
    df["ma50"] = df["close"].ewm(span=EXIT_MA_LONG, adjust=False).mean()

    # Drop NaN rows from rolling calculations
    df = df.dropna().reset_index(drop=True)

    return df


def get_hmm_observations(df: pd.DataFrame) -> np.ndarray:
    """Extract the 4 HMM feature columns as a numpy array."""
    feature_cols = ["log_return", "rolling_vol", "momentum", "volume_zscore"]
    return df[feature_cols].values


# ═══════════════════════════════════════════════════════════════
# LIVE DATA HELPERS
# ═══════════════════════════════════════════════════════════════

def compute_live_features(historical_df: pd.DataFrame, live_price: float,
                          live_volume: float) -> dict:
    """
    Given the recent historical DataFrame and a new live price/volume,
    compute current feature values for signal checking.
    """
    closes = list(historical_df["close"].values[-50:]) + [live_price]
    volumes = list(historical_df["volume"].values[-50:]) + [live_volume]

    closes = np.array(closes)
    volumes = np.array(volumes)

    # Log return
    log_return = np.log(closes[-1] / closes[-2])

    # Rolling vol (last 20 log returns, annualised for 5m)
    log_returns = np.diff(np.log(closes[-21:]))
    rolling_vol = np.std(log_returns) * np.sqrt(105120)

    # Momentum (10-period ROC)
    momentum = (closes[-1] - closes[-11]) / closes[-11]

    # Volume z-score
    vol_window = volumes[-20:]
    volume_zscore = (volumes[-1] - np.mean(vol_window)) / (np.std(vol_window) + 1e-10)

    # EMAs
    alpha20 = 2 / (ENTRY_MA_SHORT + 1)
    weights20 = np.array([(1 - alpha20) ** i for i in range(20)])[::-1]
    ma20 = np.sum(weights20 * closes[-20:]) / np.sum(weights20)

    alpha50 = 2 / (EXIT_MA_LONG + 1)
    weights50 = np.array([(1 - alpha50) ** i for i in range(50)])[::-1]
    ma50 = np.sum(weights50 * closes[-50:]) / np.sum(weights50)

    return {
        "log_return": log_return,
        "rolling_vol": rolling_vol,
        "momentum": momentum,
        "volume_zscore": volume_zscore,
        "ma20": ma20,
        "ma50": ma50,
        "close": live_price,
    }
