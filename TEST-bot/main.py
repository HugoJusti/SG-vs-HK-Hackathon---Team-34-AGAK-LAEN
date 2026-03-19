"""
Main trading bot loop.

Ties together: data fetching, HMM training, signal generation,
Monte Carlo position sizing, order execution, and cooldown management.

Usage:
    python main.py              # Run live trading
    python main.py --backtest   # Run backtest on historical data
"""
import os
import sys
import time
import json
import logging
import argparse
import warnings
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=DeprecationWarning)

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

class PositionTracker:
    """Track open positions per pair."""

    def __init__(self):
        self.positions = {}
        for pair in PAIRS:
            self.positions[pair] = {"side": "none", "entry_price": 0, "quantity": 0}

    def open_position(self, pair: str, entry_price: float, quantity: float):
        self.positions[pair] = {
            "side": "long",
            "entry_price": entry_price,
            "quantity": quantity,
        }

    def close_position(self, pair: str):
        self.positions[pair] = {"side": "none", "entry_price": 0, "quantity": 0}

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

        # Per-pair HMM models and data
        self.hmm_models = {}
        self.historical_data = {}
        self.feature_data = {}
        self.last_train_time = {}

        self.logger = logging.getLogger("Bot")

    # -- Initialisation --

    def initialise(self):
        """Fetch historical data and train HMMs for all pairs."""
        self.logger.info("=" * 60)
        self.logger.info("INITIALISING TRADING BOT")
        self.logger.info("Pairs: %s", PAIRS)
        self.logger.info("=" * 60)

        # Verify API connectivity
        server_time = self.client.get_server_time()
        self.logger.info("Server time: %s", server_time)

        exchange_info = self.client.get_exchange_info()
        if exchange_info:
            available = list(exchange_info.get("TradePairs", {}).keys())
            self.logger.info("Available pairs: %s", available)

        # Fetch data and train HMM for each pair
        for pair in PAIRS:
            self.logger.info("\n--- Setting up %s ---", pair)

            # Load historical data
            df = load_or_fetch_historical(pair)
            if df.empty:
                self.logger.error("No data for %s, skipping", pair)
                continue

            # Compute features
            df_features = compute_features(df)
            self.historical_data[pair] = df
            self.feature_data[pair] = df_features

            # Train HMM on raw (unscaled) features
            hmm = RegimeHMM()
            observations = get_hmm_observations(df_features)
            hmm.train(observations, pair)
            hmm.save(pair)
            self.hmm_models[pair] = hmm
            self.last_train_time[pair] = datetime.utcnow()

        # Check balance
        balance = self.client.get_balance()
        self.logger.info("Starting balance: %s", json.dumps(balance, indent=2))

    # -- Main Loop --

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

            # Check if HMM needs retraining
            self._check_retrain()

            # Wait for next cycle
            self.logger.info(
                "Rate limiter: %d calls remaining. Sleeping %ds...",
                rate_limiter.remaining_calls, POLL_INTERVAL_SEC
            )
            time.sleep(POLL_INTERVAL_SEC)

    def _trading_cycle(self):
        """Single iteration: check signals for each pair and act."""
        portfolio_value = self.client.get_portfolio_value()
        self.logger.info("Portfolio value: $%s", f"{portfolio_value:,.2f}")

        for pair in PAIRS:
            if pair not in self.hmm_models:
                continue

            self.logger.info("\n  -- %s --", pair)

            # 1. Get live ticker data
            ticker = self.client.get_ticker_spread(pair)
            if ticker["spread_pct"] >= 999:
                self.logger.warning("  Ticker unavailable for %s, skipping", pair)
                continue

            self.logger.info(
                "  Price: $%s | Bid: $%s | Ask: $%s | Spread: %s%%",
                f"{ticker['last']:,.2f}", f"{ticker['bid']:,.2f}",
                f"{ticker['ask']:,.2f}", f"{ticker['spread_pct']:.4f}"
            )

            # 2. Compute live features
            features = compute_live_features(
                self.feature_data[pair],
                ticker["last"],
                ticker["volume"]
            )

            # 3. Get HMM regime probabilities
            hmm = self.hmm_models[pair]
            obs_row = np.array([[
                features["log_return"],
                features["rolling_vol"],
                features["momentum"],
                features["volume_zscore"]
            ]])
            # Use raw (unscaled) observations for context
            recent_obs = get_hmm_observations(self.feature_data[pair])
            full_obs = np.vstack([recent_obs[-99:], obs_row])
            self.logger.info("  DEBUG last hist row: %s", recent_obs[-1])
            self.logger.info("  DEBUG live obs row:  %s", obs_row[0])
            regime_probs = hmm.get_regime_probabilities(full_obs)
            # Also check what happens with ONLY the last 10 rows
            short_obs = np.vstack([recent_obs[-9:], obs_row])
            short_probs = hmm.get_regime_probabilities(short_obs)
            self.logger.info("  DEBUG short context probs: Bull=%.3f Bear=%.3f Neutral=%.3f",
                short_probs["bullish"], short_probs["bearish"], short_probs["neutral"])
            self.logger.info(
                "  Regime: Bull=%s | Bear=%s | Neutral=%s",
                f"{regime_probs['bullish']:.3f}",
                f"{regime_probs['bearish']:.3f}",
                f"{regime_probs['neutral']:.3f}"
            )

            # 4. Check cooldown
            if self.cooldown.is_on_cooldown(pair, regime_probs):
                self.logger.info("  %s on cooldown, skipping", pair)
                continue

            # 5. Generate signal
            position = self.positions.get(pair)
            signal = self.signal_gen.generate_signal(
                regime_probs, features, ticker["spread_pct"], position
            )

            self.logger.info("  Signal: %s", signal["action"])
            for reason in signal["reasons"]:
                self.logger.info("    %s", reason)

            # 6. Act on signal
            if signal["action"] == "BUY":
                self._execute_buy(pair, ticker, regime_probs, portfolio_value, hmm)

            elif signal["action"] == "SELL":
                self._execute_sell(pair, ticker, signal)

            # Log signal
            self._log_signal(pair, signal, regime_probs, features, ticker)

    # -- Trade Execution --

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
        coin = pair.split("/")[0]

        quantity = position_usd / price

        exchange_info = self.client.get_exchange_info()
        pair_info = exchange_info.get("TradePairs", {}).get(pair, {})
        precision = pair_info.get("AmountPrecision", 6)

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

    # -- HMM Retraining --

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

    # -- Signal Logging --

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