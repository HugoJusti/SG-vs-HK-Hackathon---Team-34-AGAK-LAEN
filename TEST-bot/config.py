"""
Configuration for the HMM + Monte Carlo Trading Bot
All tunable parameters in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Roostoo API ──────────────────────────────────────────────
API_KEY = os.getenv("RST_API_KEY", "YOUR_API_KEY_HERE")
SECRET_KEY = os.getenv("RST_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
BASE_URL = "https://mock-api.roostoo.com"

# ── Trading Pairs ────────────────────────────────────────────
PAIRS = ["BTC/USD", "ETH/USD", "ZEC/USD"]

# ── HMM Settings ─────────────────────────────────────────────
HMM_N_STATES = 3                # bullish, bearish, neutral
HMM_FEATURES = [                # observation features fed to HMM
    "log_return",
    "rolling_vol",
    "momentum",
    "volume_zscore",
]
# HMM_TRAINING_HOURS = 2 * 365 * 24   # ~2 years of hourly data
HMM_TRAINING_HOURS = 6 * 30 * 24    # 6 months
HMM_RETRAIN_INTERVAL_HOURS = 24     # retrain HMM every 24 hours

# ── Entry Thresholds ─────────────────────────────────────────
ENTRY_HMM_CONFIDENCE = 0.586    # P(bullish) must exceed this
ENTRY_MAX_VOLATILITY = 0.657    # annualised rolling vol cap
ENTRY_MOMENTUM_MIN = 0.0181     # 10-period ROC must be > 0
ENTRY_VOLUME_ZSCORE_MAX = 4.284 # |z| must be below this
ENTRY_MA_SHORT = 20             # close > MA20 required
ENTRY_SPREAD_MAX_PCT = 0.05     # bid-ask spread filter (Option B)

# ── Exit Thresholds ──────────────────────────────────────────
EXIT_HMM_CONFIDENCE = 0.60      # P(bearish) triggers exit
EXIT_MA_LONG = 20               # close < MA20 emergency exit
EXIT_STOP_LOSS_PCT = 0.02       # 2% drawdown hard stop
EXIT_TAKE_PROFIT_PCT = 0.02     # 2% gain take profit

# ── Monte Carlo Position Sizing ──────────────────────────────
MC_NUM_SIMULATIONS = 1000       # number of forward paths
MC_HORIZON_HOURS = 24           # how far forward to simulate
MC_MAX_POSITION_PCT = 0.30      # max 30% of portfolio per trade
MC_MIN_POSITION_PCT = 0.05      # min 5% if signal exists
MC_HIGH_CONFIDENCE_THRESHOLD = 0.85   # % of paths positive → size up
MC_LOW_CONFIDENCE_THRESHOLD = 0.60    # % of paths positive → size down
MC_TAIL_RISK_LIMIT = 0.02      # 5th percentile loss cap (2%)

# ── Cooldown ─────────────────────────────────────────────────
COOLDOWN_MIN_MINUTES = 5       # 5 minutes minimum after exit
COOLDOWN_STABILITY_CHECKS = 3   # consecutive stable HMM reads needed
COOLDOWN_STABILITY_THRESHOLD = 0.65  # regime confidence for stability

# ── Order Execution ──────────────────────────────────────────
USE_LIMIT_ORDERS = True
LIMIT_ORDER_OFFSET_PCT = 0.0003  # place limit 0.03% inside spread
LIMIT_ORDER_TIMEOUT_SEC = 45     # cancel unfilled limit after this
LIMIT_ORDER_FALLBACK_MARKET = True  # use market order if limit fails

# ── Polling & Rate Limits ────────────────────────────────────
# POLL_INTERVAL_SEC = 120        # check signals every 2 minutes
POLL_INTERVAL_SEC = 60        # check signals every 2 minutes
MAX_API_CALLS_PER_MIN = 30
RETRY_BASE_DELAY_SEC = 1.0
RETRY_MAX_ATTEMPTS = 3

# ── Logging ──────────────────────────────────────────────────
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
LOG_TRADES_FILE = "logs/trades.jsonl"
LOG_SIGNALS_FILE = "logs/signals.jsonl"
LOG_ERRORS_FILE = "logs/errors.log"

# ── Data Paths ───────────────────────────────────────────────
DATA_DIR = "data"
HMM_MODEL_DIR = "models"
