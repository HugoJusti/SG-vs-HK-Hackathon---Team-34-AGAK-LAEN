"""
Data module: historical data fetching and SMC indicator computation.

Fetches OHLCV from Binance public API (no key required) and computes
all indicators needed by the SMC strategy:
  RSI, MACD, EMA50, Volume MA, ATR

Also fetches 1H higher-timeframe (HTF) data used for the macro trend gate.
"""
import os
import time
import pandas as pd
import requests
import logging
from datetime import datetime
from config import (
    DATA_DIR, PAIRS, DATA_FETCH_HOURS,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL_PERIOD,
    EMA_TREND_PERIOD, VOLUME_MA_PERIOD,
    HTF_INTERVAL, HTF_FETCH_DAYS,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# BINANCE HISTORICAL DATA
# ═══════════════════════════════════════════════════════════════

def fetch_binance_klines(symbol: str, interval: str = "5m",
                         limit_hours: int = DATA_FETCH_HOURS) -> pd.DataFrame:
    """
    Fetch historical OHLCV klines from Binance public API (no auth needed).
    symbol   : e.g. "TRXUSDT", "TAOUSDT"
    interval : "5m", "1h", "4h", "1d", etc.
    Paginates automatically since Binance caps at 1000 candles per request.
    """
    url      = "https://api.binance.com/api/v3/klines"
    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - (limit_hours * 3600 * 1000)
    current  = start_ms
    all_rows = []

    logger.info(f"Fetching {limit_hours}h of {interval} candles for {symbol}…")

    while current < end_ms:
        params = {
            "symbol":    symbol,
            "interval":  interval,
            "startTime": current,
            "endTime":   end_ms,
            "limit":     1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            all_rows.extend(data)
            current = data[-1][0] + 1
            time.sleep(0.2)
        except requests.exceptions.RequestException as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            time.sleep(1)
            continue

    if not all_rows:
        logger.error(f"No kline data returned for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = (df[["open_time", "open", "high", "low", "close", "volume"]]
          .drop_duplicates(subset=["open_time"])
          .sort_values("open_time")
          .reset_index(drop=True))

    logger.info(f"Fetched {len(df)} candles for {symbol}")
    return df


def pair_to_binance_symbol(pair: str) -> str:
    """'TRX/USD' → 'TRXUSDT'"""
    return pair.split("/")[0] + "USDT"


def load_or_fetch_historical(pair: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load 5m data from local CSV cache (if < 1h old), otherwise fetch from Binance.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    symbol     = pair_to_binance_symbol(pair)
    cache_path = os.path.join(DATA_DIR, f"{symbol}_5m.csv")

    if not force_refresh and os.path.exists(cache_path):
        df    = pd.read_csv(cache_path, parse_dates=["open_time"])
        age_h = (datetime.utcnow() - df["open_time"].iloc[-1]).total_seconds() / 3600
        if age_h < 1:
            logger.info(f"Using cached 5m data for {symbol} ({len(df)} candles, {age_h:.1f}h old)")
            return df

    df = fetch_binance_klines(symbol, "5m", DATA_FETCH_HOURS)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df


def append_new_candles(pair: str) -> pd.DataFrame:
    """
    Incrementally update the 5m cache by fetching only candles newer than the
    last cached timestamp.  Much faster than a full 14-day re-fetch.
    Falls back to a full fetch if no cache exists.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    symbol     = pair_to_binance_symbol(pair)
    cache_path = os.path.join(DATA_DIR, f"{symbol}_5m.csv")

    if not os.path.exists(cache_path):
        return load_or_fetch_historical(pair, force_refresh=True)

    df      = pd.read_csv(cache_path, parse_dates=["open_time"])
    last_ts = df["open_time"].iloc[-1]

    # Fetch only candles after the last cached bar (+ 1 ms to avoid overlap)
    start_ms = int(last_ts.timestamp() * 1000) + 1
    end_ms   = int(time.time() * 1000)

    if end_ms - start_ms < 60_000:          # less than 1 minute — nothing new
        return df

    url    = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol":    symbol,
        "interval":  "5m",
        "startTime": start_ms,
        "endTime":   end_ms,
        "limit":     1000,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
    except requests.exceptions.RequestException as e:
        logger.error("append_new_candles: Binance error for %s: %s", symbol, e)
        return df

    if not rows:
        return df

    new_df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ])
    new_df["open_time"] = pd.to_datetime(new_df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        new_df[col] = new_df[col].astype(float)
    new_df = new_df[["open_time", "open", "high", "low", "close", "volume"]]

    combined = (pd.concat([df, new_df])
                .drop_duplicates(subset=["open_time"])
                .sort_values("open_time")
                .reset_index(drop=True))

    combined.to_csv(cache_path, index=False)
    logger.info("append_new_candles: %s +%d new candles → %d total",
                symbol, len(new_df), len(combined))
    return combined


def load_or_fetch_htf(pair: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load 1H (HTF) data from local CSV cache (if < 4h old), otherwise fetch from Binance.
    Used for the macro trend gate: entry blocked when 1H close < 1H EMA50.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    symbol     = pair_to_binance_symbol(pair)
    cache_path = os.path.join(DATA_DIR, f"{symbol}_{HTF_INTERVAL}.csv")

    if not force_refresh and os.path.exists(cache_path):
        df    = pd.read_csv(cache_path, parse_dates=["open_time"])
        age_h = (datetime.utcnow() - df["open_time"].iloc[-1]).total_seconds() / 3600
        if age_h < 4:
            logger.info(f"Using cached {HTF_INTERVAL} data for {symbol} ({len(df)} candles)")
            return df

    df = fetch_binance_klines(symbol, HTF_INTERVAL, HTF_FETCH_DAYS * 24)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df


# ═══════════════════════════════════════════════════════════════
# SMC INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_smc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all SMC-relevant indicators to the OHLCV DataFrame.

    Added columns:
      rsi          — RSI (RSI_PERIOD bars)
      macd         — MACD line (EMA_FAST - EMA_SLOW)
      macd_signal  — MACD signal line
      macd_hist    — MACD histogram
      ema50        — EMA trend filter (EMA_TREND_PERIOD)
      vol_ma       — Volume moving average (VOLUME_MA_PERIOD)
      atr          — Average True Range (14 bars) — used for TP calculation

    Drops rows with NaN (warmup period) and resets the index.
    """
    df = df.copy()

    # ── RSI ──────────────────────────────────────────────────────
    delta  = df["close"].diff()
    gain   = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss   = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    rs     = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────
    ema_fast          = df["close"].ewm(span=MACD_FAST,   adjust=False).mean()
    ema_slow          = df["close"].ewm(span=MACD_SLOW,   adjust=False).mean()
    df["macd"]        = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL_PERIOD, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── EMA50 trend filter ────────────────────────────────────────
    df["ema50"] = df["close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()

    # ── Volume MA ─────────────────────────────────────────────────
    df["vol_ma"] = df["volume"].rolling(VOLUME_MA_PERIOD).mean()

    # ── ATR (14-bar) ──────────────────────────────────────────────
    tr        = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df = df.dropna().reset_index(drop=True)
    return df
