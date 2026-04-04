"""
Configuration for the SMC + Confluence Trading Bot.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Roostoo API ──────────────────────────────────────────────
API_KEY    = os.getenv("RST_API_KEY",    "YOUR_API_KEY_HERE")
SECRET_KEY = os.getenv("RST_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
BASE_URL   = "https://mock-api.roostoo.com"

# ── Trading Pairs ────────────────────────────────────────────
PAIRS = ["TRX/USD", "TAO/USD", "SOL/USD", "FET/USD", "ALGO/USD"]  # bullish-trend altcoins

# ── Historical Data ──────────────────────────────────────────
DATA_FETCH_HOURS              = 14 * 24   # 14 days of 5m candles (~4032 candles)
DATA_REFRESH_INTERVAL_HOURS  = 1         # kept for HTF (1H) refresh cadence
DATA_REFRESH_INTERVAL_MINUTES = 5        # 5m candles refreshed every new bar

# ── Swing Structure Detection ────────────────────────────────
SWING_LEFT_BARS  = 3    # fractal bars required to the left of swing point
SWING_RIGHT_BARS = 2    # fractal bars required to the right — reduced from 3 (10-min lag → 5-min)
SWING_LOOKBACK   = 200  # scan last N candles when detecting swings
SWING_SIGNIFICANCE_ATR_MULT = 0.2  # pivot must stand out by ≥ 0.2× ATR from neighbours to count as structural

# ── Support Zones ────────────────────────────────────────────
ZONE_MARGIN_PCT = 0.003  # cluster swing lows within 0.3% of each other

# ── Liquidity Sweep ──────────────────────────────────────────
SWEEP_TOLERANCE_PCT  = 0.001  # min 0.1% wick below zone to qualify
SWEEP_LOOKBACK       = 10     # check last N completed candles for a sweep
SWEEP_WICK_RATIO_MIN = 0.30   # lower wick must be ≥ 30% of total candle range
POI_TOUCH_LOOKBACK   = 50     # candles to look back for a POI touch (50 bars = 4.2 hours)

# ── Break of Structure ───────────────────────────────────────
BOS_LOOKBACK          = 10   # look back N candles for the most recent swing high
BOS_CLOSE_BUFFER_PCT  = 0.001  # close must be ≥ 0.1% above BOS level — filters marginal breaks
BOS_MAX_AGE_CANDLES   = 5    # skip entry if BOS fired more than N candles ago (stale setup)

# ── Indicators ───────────────────────────────────────────────
RSI_PERIOD         = 14
RSI_OVERBOUGHT     = 70   # block entry if RSI > this (already extended)
MACD_FAST          = 12
MACD_SLOW          = 26
MACD_SIGNAL_PERIOD = 9
EMA_TREND_PERIOD   = 50   # bullish bias when price is above this EMA
VOLUME_MA_PERIOD   = 20
VOLUME_SPIKE_MULT  = 1.5  # sweep volume must be ≥ this × rolling average

# ── Higher Timeframe Filter ──────────────────────────────────
HTF_INTERVAL  = "1h"   # timeframe for macro trend bias
HTF_FETCH_DAYS = 30    # how many days of 1H candles to fetch

# ── Confluences (weighted scoring) ───────────────────────────
# Sweep presence:     +2  (stop-hunt fingerprint — optional bonus)
# Vol spike on sweep: +1  (extra if sweep had volume)
# Equilibrium:        +2  (price in 40-60% retrace band)
# EMA50:              +1  (5m trend filter)
# MACD:               +1  (momentum)
# RSI:                +1  (not overbought)
# Max score = 8 (sweep+vol+equil+ema+macd+rsi), min without sweep = 5
CONFLUENCE_WEIGHTS = {
    "vol_spike": 2,   # used as sweep-presence weight
    "ema50":     1,
    "macd":      1,
    "rsi":       1,
}
MIN_CONFLUENCES = 3  # minimum score to enter (lowered since FVG +2 removed)

# ── Entry Filter ─────────────────────────────────────────────
ENTRY_SPREAD_MAX_PCT = 0.05  # skip pair if bid-ask spread exceeds this

# ── Position Sizing (risk-based) ─────────────────────────────
RISK_PER_TRADE_PCT = 0.02   # risk 2% of portfolio on each trade
MAX_POSITION_PCT   = 0.30   # hard cap: no more than 30% per trade
MIN_POSITION_PCT   = 0.05   # floor: at least 5% if SL is very wide

# ── Stop Loss / Take Profit ──────────────────────────────────
SL_BUFFER_PCT    = 0.002  # place SL 0.2% below the sweep wick low
TP_MIN_PCT       = 0.008  # minimum take profit of 0.8%
TP_MAX_PCT       = 0.04   # cap take profit at 4%
MIN_RR_RATIO     = 2.0    # minimum reward:risk ratio — skip trade if TP can't reach this
RECONCILE_SL_PCT = 0.03   # fallback SL for reconciled positions: 3% below current price

# ── Concurrent Positions ────────────────────────────────────
MAX_CONCURRENT_POSITIONS = 2   # at most 2 open trades across all pairs at once

# ── Daily Loss Circuit Breaker ───────────────────────────────
MAX_DAILY_LOSS_PCT = 0.06      # stop all new entries if portfolio drops >6% in one calendar day

# ── Cooldown ─────────────────────────────────────────────────
COOLDOWN_MIN_MINUTES        = 10    # minimum wait after closing a position
COOLDOWN_STABILITY_CHECKS   = 2     # kept for CooldownManager compatibility
COOLDOWN_STABILITY_THRESHOLD = 0.65

# ── Order Execution ──────────────────────────────────────────
USE_LIMIT_ORDERS            = True
LIMIT_ORDER_OFFSET_PCT      = 0.0003
LIMIT_ORDER_TIMEOUT_SEC     = 45
LIMIT_ORDER_FALLBACK_MARKET = True

# ── Polling & Rate Limits ────────────────────────────────────
POLL_INTERVAL_SEC     = 30
MAX_API_CALLS_PER_MIN = 30
RETRY_BASE_DELAY_SEC  = 1.0
RETRY_MAX_ATTEMPTS    = 3

# ── Logging ──────────────────────────────────────────────────
LOG_DIR          = "logs"
LOG_LEVEL        = "INFO"
LOG_TRADES_FILE  = "logs/trades.jsonl"
LOG_SIGNALS_FILE = "logs/signals.jsonl"
LOG_ERRORS_FILE  = "logs/errors.log"

# ── Data Paths ───────────────────────────────────────────────
DATA_DIR = "data"
