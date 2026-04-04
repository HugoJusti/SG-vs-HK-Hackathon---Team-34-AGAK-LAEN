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
PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD"]

# ── HMM Settings ─────────────────────────────────────────────
HMM_N_STATES = 3                # bullish, bearish, neutral
HMM_FEATURES = [                # observation features fed to HMM
    "log_return",
    "rolling_vol",
    "momentum",
    "volume_zscore",
]
HMM_TRAINING_HOURS = 1 * 60 * 24    # 2 month
HMM_RETRAIN_INTERVAL_HOURS = 24     # retrain HMM every 24 hours

# ── Entry Thresholds ─────────────────────────────────────────
ENTRY_HMM_CONFIDENCE = 0.494     # P(bullish) must exceed this
ENTRY_MAX_VOLATILITY = 0.1997     # annualised rolling vol cap
ENTRY_MOMENTUM_MIN = -0.0346     # 10-period ROC must be > this
ENTRY_VOLUME_ZSCORE_MAX = 4.5283  # |z| must be below this
ENTRY_MA_SHORT = 8             # close > EMA8 required
ENTRY_SPREAD_MAX_PCT = 0.0362     # bid-ask spread filter

# ── Exit Thresholds ──────────────────────────────────────────
EXIT_HMM_CONFIDENCE = 0.6087      # unused — kept for reference
EXIT_MA_LONG = 20               # unused — kept for reference
EXIT_STOP_LOSS_PCT = 0.007      # 0.7% drawdown hard stop
EXIT_TAKE_PROFIT_PCT = 0.007    # 0.7% gain take profit

# ── Monte Carlo Position Sizing ──────────────────────────────
MC_NUM_SIMULATIONS = 1000       # number of forward paths
MC_HORIZON_HOURS = 24           # how far forward to simulate
MC_MAX_POSITION_PCT = 0.30      # max 30% of portfolio per trade
MC_MIN_POSITION_PCT = 0.05      # min 5% if tail risk too high
MC_HIGH_CONFIDENCE_THRESHOLD = 0.85
MC_LOW_CONFIDENCE_THRESHOLD = 0.60
MC_TAIL_RISK_LIMIT = 0.02       # 5th percentile loss cap (2%)

# ── Cooldown ─────────────────────────────────────────────────
COOLDOWN_MIN_MINUTES = 5        # 5 minutes minimum after exit
COOLDOWN_STABILITY_CHECKS = 3   # consecutive stable HMM reads needed
COOLDOWN_STABILITY_THRESHOLD = 0.65

# ── Order Execution ──────────────────────────────────────────
USE_LIMIT_ORDERS = True
LIMIT_ORDER_OFFSET_PCT = 0.0003
LIMIT_ORDER_TIMEOUT_SEC = 45
LIMIT_ORDER_FALLBACK_MARKET = True

# ── Polling & Rate Limits ────────────────────────────────────
POLL_INTERVAL_SEC = 30
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
