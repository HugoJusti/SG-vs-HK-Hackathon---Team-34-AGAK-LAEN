"""
Main trading bot loop.

Ties together: data fetching, HMM training, signal generation,
Monte Carlo position sizing, order execution, and cooldown management.

Usage:
    python main-final.py
"""
import os
import sys
import time
import json
import logging
import argparse
import warnings
import importlib.util
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ═══════════════════════════════════════════════════════════════
# MODULE LOADER (handles hyphenated filenames)
# ═══════════════════════════════════════════════════════════════

def _load_module(name: str, rel_path: str):
    """Load a module from a file path and register it in sys.modules."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load in order — config first since all others depend on it
_load_module("config",    "bot/config/config-final.py")
_load_module("data",      "bot/data/data-final.py")
_load_module("strategy",  "bot/strategy/strategy-final.py")
_load_module("execution", "bot/execution/execution-final.py")

# Normal imports now work
from config import (
    PAIRS, POLL_INTERVAL_SEC, LOG_DIR, LOG_LEVEL,
    LOG_SIGNALS_FILE, LOG_ERRORS_FILE, HMM_RETRAIN_INTERVAL_HOURS
)
from data import (
    load_or_fetch_historical, compute_features, get_hmm_observations,
    compute_live_features
)
from strategy import RegimeHMM, monte_carlo_position_size, SignalGenerator
from execution import (
    RoostooClient, OrderExecutor, CooldownManager, rate_limiter
)


# =============================================================
# LOGGING SETUP
# =============================================================

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_ERRORS_FILE),
        ]
    )


# =============================================================
# POSITION TRACKER
# =============================================================

STATE_FILE = "logs/state.json"


class PositionTracker:
    """Track open positions per pair, persisted across restarts."""

    def __init__(self):
        self.positions = {}
        for pair in PAIRS:
            self.positions[pair] = {"side": "none", "entry_price": 0, "quantity": 0}
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                saved = json.load(f)
            self.positions.update(saved)
            logging.getLogger("PositionTracker").info("Restored positions: %s", self.positions)

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.positions, f, indent=2)

    def reconcile_positions(self, client):
        """On startup, verify saved positions match actual exchange balances."""
        logger = logging.getLogger("PositionTracker")
        balance = client.get_balance()
        wallet = balance.get("SpotWallet", balance.get("Wallet", {}))

        for pair in PAIRS:
            coin = pair.split("/")[0]
            holding = wallet.get(coin, {})
            qty = holding.get("Free", 0) + holding.get("Lock", 0)
            saved = self.positions.get(pair, {})

            if qty > 0 and saved.get("side") != "long":
                ticker = client.get_ticker_spread(pair)
                entry = ticker["last"] if ticker["last"] > 0 else 0
                self.positions[pair] = {"side": "long", "entry_price": entry, "quantity": qty}
                logger.warning(
                    "Reconciled %s: found qty=%.6f on exchange with no saved position. "
                    "Entry price set to current price $%.2f", pair, qty, entry
                )
            elif qty == 0 and saved.get("side") == "long":
                logger.warning(
                    "Reconciled %s: saved position found but no coins on exchange. Clearing.", pair
                )
                self.positions[pair] = {"side": "none", "entry_price": 0, "quantity": 0}

        self._save()

    def open_position(self, pair: str, entry_price: float, quantity: float):
        self.positions[pair] = {
            "side": "long",
            "entry_price": entry_price,
            "quantity": quantity,
        }
        self._save()

    def close_position(self, pair: str):
        self.positions[pair] = {"side": "none", "entry_price": 0, "quantity": 0}
        self._save()

    def get(self, pair: str) -> dict:
        return self.positions.get(pair, {"side": "none", "entry_price": 0, "quantity": 0})


# =============================================================
# MAIN BOT
# =============================================================

class TradingBot:
    def __init__(self):
        self.client = RoostooClient()
        self.executor = OrderExecutor(self.client)
        self.cooldown = CooldownManager()
        self.positions = PositionTracker()
        self.signal_gen = SignalGenerator()

        self.hmm_models = {}
        self.historical_data = {}
        self.feature_data = {}
        self.last_train_time = {}

        self.logger = logging.getLogger("Bot")

    def initialise(self):
        """Fetch historical data and train HMMs for all pairs."""
        self.logger.info("=" * 60)
        self.logger.info("INITIALISING TRADING BOT")
        self.logger.info("Pairs: %s", PAIRS)
        self.logger.info("=" * 60)

        server_time = self.client.get_server_time()
        self.logger.info("Server time: %s", server_time)

        exchange_info = self.client.get_exchange_info()
        if exchange_info:
            available = list(exchange_info.get("TradePairs", {}).keys())
            self.logger.info("Available pairs: %s", available)

        for pair in PAIRS:
            self.logger.info("\n--- Setting up %s ---", pair)

            df = load_or_fetch_historical(pair)
            if df.empty:
                self.logger.error("No data for %s, skipping", pair)
                continue

            df_features = compute_features(df)
            self.historical_data[pair] = df
            self.feature_data[pair] = df_features

            hmm = RegimeHMM()
            observations = get_hmm_observations(df_features)
            hmm.train(observations, pair)
            hmm.save(pair)
            self.hmm_models[pair] = hmm
            self.last_train_time[pair] = datetime.utcnow()

        balance = self.client.get_balance()
        self.logger.info("Starting balance: %s", json.dumps(balance, indent=2))

        self.positions.reconcile_positions(self.client)
        self.logger.info("Position state after reconciliation: %s", self.positions.positions)

    def run(self):
        """Main trading loop."""
        self.initialise()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STARTING LIVE TRADING LOOP")
        self.logger.info("Poll interval: %ss", POLL_INTERVAL_SEC)
        self.logger.info("=" * 60)

        cycle = 0
        while True:
            cycle += 1
            try:
                self.logger.info("\n%s Cycle %d %s", "-"*40, cycle, "-"*40)
                self._trading_cycle()

            except KeyboardInterrupt:
                self.logger.info("Shutdown requested. Exiting...")
                break
            except Exception as e:
                self.logger.error("Error in cycle %d: %s", cycle, e, exc_info=True)

            self._check_retrain()

            self.logger.info(
                "Rate limiter: %d calls remaining. Sleeping %ds...",
                rate_limiter.remaining_calls, POLL_INTERVAL_SEC
            )
            time.sleep(POLL_INTERVAL_SEC)

    def _trading_cycle(self):
        """Single iteration: check signals for each pair and act."""
        balance = self.client.get_balance()
        wallet = balance.get("SpotWallet", balance.get("Wallet", {}))

        # Pre-fetch all tickers once and reuse throughout the cycle
        ticker_cache = {pair: self.client.get_ticker_spread(pair) for pair in PAIRS}

        self.logger.info("--- Portfolio Snapshot ---")
        usd = wallet.get("USD", {})
        self.logger.info("  USD   | Free: $%-12s | Lock: $%s",
                         f"{usd.get('Free', 0):,.2f}", f"{usd.get('Lock', 0):,.2f}")

        for coin, amounts in wallet.items():
            if coin == "USD":
                continue
            pair = f"{coin}/USD"
            pos = self.positions.get(pair)

            pnl_str = "N/A"
            if pos['side'] == "long" and pos['entry_price'] > 0:
                ticker = ticker_cache.get(pair) or self.client.get_ticker_spread(pair)
                if ticker["last"] > 0:
                    current_price = ticker["last"]
                    pnl_pct = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                    pnl_usd = (current_price - pos['entry_price']) * pos['quantity']
                    pnl_str = f"{pnl_pct:+.2f}% (${pnl_usd:+,.2f})"

            self.logger.info(
                "  %-5s | Free: %-12s | Lock: %-12s | Side: %-5s | Entry: $%-12s | Qty: %-14s | PnL: %s",
                coin,
                f"{amounts.get('Free', 0):.6f}",
                f"{amounts.get('Lock', 0):.6f}",
                pos['side'],
                f"{pos['entry_price']:,.2f}",
                f"{pos['quantity']:.6f}",
                pnl_str
            )
        self.logger.info("-" * 26)

        # Calculate portfolio value from already-fetched wallet + ticker cache
        portfolio_value = usd.get("Free", 0) + usd.get("Lock", 0)
        for coin, amounts in wallet.items():
            if coin == "USD":
                continue
            holding = amounts.get("Free", 0) + amounts.get("Lock", 0)
            if holding > 0:
                t = ticker_cache.get(f"{coin}/USD") or self.client.get_ticker_spread(f"{coin}/USD")
                if t and t["last"] > 0:
                    portfolio_value += holding * t["last"]
        self.logger.info("Portfolio value: $%s", f"{portfolio_value:,.2f}")

        for pair in PAIRS:
            if pair not in self.hmm_models:
                continue

            self.logger.info("\n  -- %s --", pair)

            ticker = ticker_cache[pair]
            if ticker["spread_pct"] >= 999:
                self.logger.warning("  Ticker unavailable for %s, skipping", pair)
                continue

            self.logger.info(
                "  Price: $%s | Bid: $%s | Ask: $%s | Spread: %s%%",
                f"{ticker['last']:,.2f}", f"{ticker['bid']:,.2f}",
                f"{ticker['ask']:,.2f}", f"{ticker['spread_pct']:.4f}"
            )

            features = compute_live_features(
                self.feature_data[pair],
                ticker["last"],
                ticker["volume"]
            )

            hmm = self.hmm_models[pair]
            obs_row = np.array([[
                features["log_return"],
                features["rolling_vol"],
                features["momentum"],
                features["volume_zscore"]
            ]])
            recent_obs = get_hmm_observations(self.feature_data[pair])
            full_obs = np.vstack([recent_obs[-287:], obs_row])
            regime_probs = hmm.get_regime_probabilities(full_obs)

            self.logger.info(
                "  Regime: Bull=%s | Bear=%s | Neutral=%s",
                f"{regime_probs['bullish']:.3f}",
                f"{regime_probs['bearish']:.3f}",
                f"{regime_probs['neutral']:.3f}"
            )

            if self.cooldown.is_on_cooldown(pair, regime_probs):
                self.logger.info("  %s on cooldown, skipping", pair)
                continue

            position = self.positions.get(pair)
            signal = self.signal_gen.generate_signal(
                regime_probs, features, ticker["spread_pct"], position
            )

            self.logger.info("  Signal: %s", signal["action"])
            for reason in signal["reasons"]:
                self.logger.info("    %s", reason)

            if signal["action"] == "BUY":
                self._execute_buy(pair, ticker, regime_probs, portfolio_value, hmm)
            elif signal["action"] == "SELL":
                self._execute_sell(pair, ticker, signal)

            self._log_signal(pair, signal, regime_probs, features, ticker)

    def _execute_buy(self, pair, ticker, regime_probs, portfolio_value, hmm):
        """Run Monte Carlo, size position, execute buy."""
        regime_params = hmm.get_regime_params()
        current_regime = hmm.bull_idx

        mc_result = monte_carlo_position_size(
            regime_params, current_regime, portfolio_value
        )

        if mc_result["position_pct"] <= 0:
            self.logger.info("  MC says don't trade (position=0%%)")
            return

        position_usd = mc_result["position_usd"]
        price = ticker["ask"]

        quantity = position_usd / price

        exchange_info = self.client.get_exchange_info()
        pair_info = exchange_info.get("TradePairs", {}).get(pair, {})
        self.logger.info("  Exchange pair_info for %s: %s", pair, pair_info)

        precision = pair_info.get("AmountPrecision", 6)
        step = pair_info.get("AmountStep", None)

        if step and step > 0:
            quantity = int(quantity / step) * step
            quantity = round(quantity, precision)
        else:
            factor = 10 ** precision
            quantity = int(quantity * factor) / factor

        if quantity * price < 1.0:
            self.logger.info("  Order too small: $%s < $1.00", f"{quantity * price:.2f}")
            return

        self.logger.info(
            "  BUYING %s: qty=%s (~$%s, %s%% of portfolio)",
            pair, quantity, f"{position_usd:,.0f}",
            f"{mc_result['position_pct']*100:.1f}"
        )
        self.logger.info(
            "  MC: %s%% paths positive, tail risk=%s%%",
            f"{mc_result['paths_positive_pct']*100:.1f}",
            f"{mc_result['tail_risk_5pct']*100:.3f}"
        )

        result = self.executor.execute_buy(pair, quantity, ticker)

        if result.get("Success"):
            filled_price = result.get("OrderDetail", {}).get("FilledAverPrice", price)
            if filled_price == 0:
                filled_price = price
            self.positions.open_position(pair, filled_price, quantity)
            self.logger.info("  BUY filled at $%s", f"{filled_price:,.2f}")
        else:
            self.logger.error("  BUY failed: %s", result.get("ErrMsg", "unknown"))

    def _execute_sell(self, pair, ticker, signal):
        """Execute sell and start cooldown."""
        position = self.positions.get(pair)
        quantity = position.get("quantity", 0)

        if quantity <= 0:
            self.logger.warning("  No position to sell for %s", pair)
            return

        self.logger.info(
            "  SELLING %s: qty=%s | Reason: %s",
            pair, quantity,
            signal["reasons"][0] if signal["reasons"] else "unknown"
        )

        result = self.executor.execute_sell(pair, quantity, ticker)

        if result.get("Success"):
            filled_price = result.get("OrderDetail", {}).get("FilledAverPrice", ticker["bid"])
            entry_price = position.get("entry_price", 0)
            pnl_pct = ((filled_price - entry_price) / entry_price * 100) if entry_price else 0

            self.positions.close_position(pair)
            self.cooldown.start_cooldown(pair)

            self.logger.info(
                "  SELL filled at $%s | PnL: %s%%",
                f"{filled_price:,.2f}", f"{pnl_pct:+.2f}"
            )
        else:
            self.logger.error("  SELL failed: %s", result.get("ErrMsg", "unknown"))

    def _check_retrain(self):
        """Retrain HMM if enough time has passed."""
        for pair in PAIRS:
            if pair not in self.last_train_time:
                continue
            hours_since = (datetime.utcnow() - self.last_train_time[pair]).total_seconds() / 3600
            if hours_since >= HMM_RETRAIN_INTERVAL_HOURS:
                self.logger.info("Retraining HMM for %s (%.1fh since last)", pair, hours_since)
                try:
                    df = load_or_fetch_historical(pair, force_refresh=True)
                    df_features = compute_features(df)
                    self.historical_data[pair] = df
                    self.feature_data[pair] = df_features

                    obs = get_hmm_observations(df_features)
                    self.hmm_models[pair].train(obs, pair)
                    self.hmm_models[pair].save(pair)
                    self.last_train_time[pair] = datetime.utcnow()
                except Exception as e:
                    self.logger.error("Retrain failed for %s: %s", pair, e)

    def _log_signal(self, pair, signal, regime_probs, features, ticker):
        """Log signal details to JSONL file."""
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pair": pair,
            "action": signal["action"],
            "confidence": signal["confidence"],
            "reasons": signal["reasons"],
            "regime": regime_probs,
            "features": {k: round(v, 6) if isinstance(v, float) else v
                         for k, v in features.items()},
            "ticker": {
                "last": ticker["last"],
                "bid": ticker["bid"],
                "ask": ticker["ask"],
                "spread_pct": ticker["spread_pct"],
            },
            "position": self.positions.get(pair),
        }
        with open(LOG_SIGNALS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")


# =============================================================
# ENTRYPOINT
# =============================================================

def main():
    parser = argparse.ArgumentParser(description="HMM + Monte Carlo Crypto Trading Bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("Main")

    if args.backtest:
        logger.info("Backtest mode not yet implemented. Use live mode.")
        sys.exit(0)

    bot = TradingBot()
    bot.run()


if __name__ == "__main__":
    main()
