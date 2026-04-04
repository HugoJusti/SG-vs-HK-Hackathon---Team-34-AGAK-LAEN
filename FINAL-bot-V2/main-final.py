"""
Main trading bot loop — SMC FINAL bot V2.

Loads all modules from bot_internal/ using importlib so that
hyphenated filenames (config-final.py, data-final.py,
execution-final.py, strategy-final.py) work correctly with standard Python imports.
"""
import os
import sys
import importlib.util

# ── Bootstrap: register bot_internal modules under their original names ──
# Must happen BEFORE any 'from config import ...' style imports.

_BOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_internal")


def _load_module(module_name: str, filename: str):
    """Load a module from bot_internal/ and register it in sys.modules."""
    path = os.path.join(_BOT_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod   # register BEFORE exec so cross-imports resolve
    spec.loader.exec_module(mod)
    return mod


_load_module("config",    "config-final.py")
_load_module("data",      "data-final.py")
_load_module("execution", "execution-final.py")
_load_module("strategy",  "strategy-final.py")

# ── Standard imports (now resolve correctly via sys.modules) ─────────────
import time
import json
import logging
import argparse
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from config import (
    PAIRS, POLL_INTERVAL_SEC, DATA_REFRESH_INTERVAL_HOURS,
    DATA_REFRESH_INTERVAL_MINUTES,
    LOG_DIR, LOG_LEVEL, LOG_SIGNALS_FILE, LOG_ERRORS_FILE,
    RECONCILE_SL_PCT, TP_MIN_PCT, MAX_CONCURRENT_POSITIONS,
    RISK_PER_TRADE_PCT, MAX_DAILY_LOSS_PCT,
)
from data import (
    load_or_fetch_historical, load_or_fetch_htf,
    append_new_candles, compute_smc_indicators,
)
from strategy import SMCSignalGenerator, calculate_position_size, get_trailing_sl
from execution import RoostooClient, OrderExecutor, CooldownManager, rate_limiter


# ═══════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    # Force UTF-8 on stdout so Unicode chars don't crash on Windows cp1252
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_ERRORS_FILE, encoding="utf-8"),
        ],
    )


# ═══════════════════════════════════════════════════════════════
# POSITION TRACKER
# ═══════════════════════════════════════════════════════════════

STATE_FILE = "logs/state.json"

_EMPTY_POS = lambda: {
    "side": "none", "entry_price": 0.0,
    "quantity": 0.0, "sl_price": 0.0, "tp_price": 0.0,
    "entry_time": "",   # ISO timestamp — used by trailing SL to find post-entry swing lows
}


class PositionTracker:
    """Track open positions per pair (persisted to JSON)."""

    def __init__(self):
        self.positions = {pair: _EMPTY_POS() for pair in PAIRS}
        self._load()

    def _load(self):
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                saved = json.load(f)
            for pair, pos in saved.items():
                self.positions[pair] = {**_EMPTY_POS(), **pos}
            logging.getLogger("Positions").info("Restored: %s", self.positions)

    def _save(self):
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(self.positions, f, indent=2)

    def reconcile(self, client: RoostooClient):
        """
        On startup: align saved state with actual exchange balances.
        Positions found on exchange but missing from state get a fallback
        SL (RECONCILE_SL_PCT below current price) and TP so the bot can exit.
        """
        log     = logging.getLogger("Positions")
        balance = client.get_balance()
        wallet  = balance.get("SpotWallet", balance.get("Wallet", {}))

        for pair in PAIRS:
            coin    = pair.split("/")[0]
            holding = wallet.get(coin, {})
            qty     = holding.get("Free", 0) + holding.get("Lock", 0)
            saved   = self.positions.get(pair, _EMPTY_POS())

            if qty > 0 and saved["side"] != "long":
                ticker   = client.get_ticker_spread(pair)
                price    = ticker["last"] if ticker["last"] > 0 else 0.0
                sl_price = price * (1 - RECONCILE_SL_PCT)
                tp_price = price * (1 + TP_MIN_PCT)
                self.positions[pair] = {
                    "side":        "long",
                    "entry_price": price,
                    "quantity":    qty,
                    "sl_price":    round(sl_price, 6),
                    "tp_price":    round(tp_price, 6),
                    "entry_time":  datetime.utcnow().isoformat(),
                }
                log.warning(
                    "Reconciled %s: qty=%.6f at $%.4f | fallback SL=%.4f TP=%.4f",
                    pair, qty, price, sl_price, tp_price,
                )

            elif qty == 0 and saved["side"] == "long":
                log.warning(
                    "Reconciled %s: state says long but no coins on exchange — clearing",
                    pair,
                )
                self.positions[pair] = _EMPTY_POS()

        self._save()

    def open_position(self, pair: str, entry_price: float, quantity: float,
                      sl_price: float, tp_price: float, entry_time: str = ""):
        self.positions[pair] = {
            "side":        "long",
            "entry_price": entry_price,
            "quantity":    quantity,
            "sl_price":    sl_price,
            "tp_price":    tp_price,
            "entry_time":  entry_time,
        }
        self._save()

    def update_sl(self, pair: str, new_sl: float):
        """Move SL up to new_sl. Only saves if pair has an open long."""
        if self.positions.get(pair, {}).get("side") == "long":
            self.positions[pair]["sl_price"] = round(new_sl, 6)
            self._save()

    def close_position(self, pair: str):
        self.positions[pair] = _EMPTY_POS()
        self._save()

    def get(self, pair: str) -> dict:
        return self.positions.get(pair, _EMPTY_POS())


# ═══════════════════════════════════════════════════════════════
# TRADING BOT
# ═══════════════════════════════════════════════════════════════

class TradingBot:

    def __init__(self):
        self.client     = RoostooClient()
        self.executor   = OrderExecutor(self.client)
        self.cooldown   = CooldownManager()
        self.positions  = PositionTracker()
        self.signal_gen = SMCSignalGenerator()

        self.smc_data      = {}   # pair -> 5m DataFrame with SMC indicators
        self.htf_data      = {}   # pair -> 1H DataFrame with SMC indicators
        self.last_refresh  = {}   # pair -> datetime of last 5m fetch
        self.exchange_info = {}   # cached once at startup

        self.logger = logging.getLogger("Bot")

        # Daily loss circuit breaker
        self.daily_start_value = 0.0
        self.daily_date        = None

    # ── Initialisation ──────────────────────────────────────────

    def initialise(self):
        self.logger.info("=" * 60)
        self.logger.info("SMC TRADING BOT — FINAL V2")
        self.logger.info("Pairs: %s", PAIRS)
        self.logger.info("=" * 60)

        self.logger.info("Server time: %s", self.client.get_server_time())

        self.exchange_info = self.client.get_exchange_info()
        if self.exchange_info:
            available = list(self.exchange_info.get("TradePairs", {}).keys())
            self.logger.info("Exchange pairs available: %s", available)

        for pair in PAIRS:
            self.logger.info("--- Loading data for %s ---", pair)
            self._refresh_data(pair)

        balance = self.client.get_balance()
        self.logger.info("Starting balance: %s", json.dumps(balance, indent=2))
        self.positions.reconcile(self.client)

    def _refresh_data(self, pair: str, force: bool = False):
        """Fetch 5m + 1H data and recompute SMC indicators for a pair."""
        df = load_or_fetch_historical(pair, force_refresh=force)
        if df.empty:
            self.logger.error("No 5m data for %s — skipping", pair)
            return
        self.smc_data[pair]     = compute_smc_indicators(df)
        self.last_refresh[pair] = datetime.utcnow()
        self.logger.info("%s: %d x 5m candles loaded", pair, len(self.smc_data[pair]))

        htf_df = load_or_fetch_htf(pair, force_refresh=force)
        if not htf_df.empty:
            self.htf_data[pair] = compute_smc_indicators(htf_df)
            self.logger.info("%s: %d x 1H candles loaded (HTF)", pair, len(self.htf_data[pair]))
        else:
            self.logger.warning("%s: no 1H data — HTF gate disabled for this pair", pair)

    # ── Main loop ───────────────────────────────────────────────

    def run(self):
        self.initialise()
        self.logger.info("\n" + "=" * 60)
        self.logger.info("LIVE TRADING — poll every %ds", POLL_INTERVAL_SEC)
        self.logger.info("=" * 60)

        cycle = 0
        while True:
            cycle += 1
            try:
                self.logger.info("\n%s Cycle %d %s", "-" * 40, cycle, "-" * 40)
                self._trading_cycle()
            except KeyboardInterrupt:
                self.logger.info("Shutdown requested. Exiting.")
                break
            except Exception as e:
                self.logger.error("Cycle %d error: %s", cycle, e, exc_info=True)

            self._maybe_refresh_data()

            self.logger.info(
                "Rate limit: %d calls remaining. Sleeping %ds...",
                rate_limiter.remaining_calls, POLL_INTERVAL_SEC,
            )
            time.sleep(POLL_INTERVAL_SEC)

    # ── Single trading cycle ─────────────────────────────────────

    def _trading_cycle(self):
        balance = self.client.get_balance()
        wallet  = balance.get("SpotWallet", balance.get("Wallet", {}))
        tickers = {pair: self.client.get_ticker_spread(pair) for pair in PAIRS}

        # Portfolio snapshot
        self.logger.info("--- Portfolio Snapshot ---")
        usd = wallet.get("USD", {})
        self.logger.info(
            "  USD   | Free: $%-12s | Lock: $%s",
            f"{usd.get('Free', 0):,.2f}", f"{usd.get('Lock', 0):,.2f}",
        )
        for coin, amounts in wallet.items():
            if coin == "USD":
                continue
            pair = f"{coin}/USD"
            pos  = self.positions.get(pair)
            pnl  = "N/A"
            if pos["side"] == "long" and pos["entry_price"] > 0:
                t = tickers.get(pair)
                if t and t["last"] > 0:
                    pct     = (t["last"] - pos["entry_price"]) / pos["entry_price"] * 100
                    usd_pnl = (t["last"] - pos["entry_price"]) * pos["quantity"]
                    pnl     = f"{pct:+.2f}% (${usd_pnl:+,.2f})"
            self.logger.info(
                "  %-5s | Free: %-12.6f | Lock: %-12.6f | Side: %-5s | "
                "Entry: $%-12.4f | SL: $%-10.4f | TP: $%-10.4f | PnL: %s",
                coin,
                amounts.get("Free", 0), amounts.get("Lock", 0),
                pos["side"], pos["entry_price"],
                pos["sl_price"], pos["tp_price"], pnl,
            )

        portfolio_value = usd.get("Free", 0) + usd.get("Lock", 0)
        for coin, amounts in wallet.items():
            if coin == "USD":
                continue
            qty = amounts.get("Free", 0) + amounts.get("Lock", 0)
            if qty > 0:
                t = tickers.get(f"{coin}/USD") or self.client.get_ticker_spread(f"{coin}/USD")
                if t and t["last"] > 0:
                    portfolio_value += qty * t["last"]
        self.logger.info("Portfolio value: $%s", f"{portfolio_value:,.2f}")

        # ── Daily loss circuit breaker ────────────────────────────
        today = datetime.utcnow().date()
        if self.daily_date != today:
            self.daily_start_value = portfolio_value
            self.daily_date        = today
            self.logger.info(
                "New trading day — daily baseline set at $%.2f (max loss: $%.2f / %.0f%%)",
                self.daily_start_value,
                self.daily_start_value * MAX_DAILY_LOSS_PCT,
                MAX_DAILY_LOSS_PCT * 100,
            )
        daily_loss_pct = (self.daily_start_value - portfolio_value) / self.daily_start_value \
                         if self.daily_start_value > 0 else 0.0
        daily_breaker_tripped = daily_loss_pct >= MAX_DAILY_LOSS_PCT
        if daily_breaker_tripped:
            self.logger.warning(
                "DAILY LOSS CIRCUIT BREAKER: down %.2f%% today (limit %.0f%%) — "
                "all new BUY entries blocked",
                daily_loss_pct * 100, MAX_DAILY_LOSS_PCT * 100,
            )

        # Count open positions for concurrent cap
        open_count = sum(
            1 for p in PAIRS if self.positions.get(p).get("side") == "long"
        )

        # Per-pair signal loop
        for pair in PAIRS:
            if pair not in self.smc_data:
                continue

            self.logger.info("\n  -- %s --", pair)
            ticker = tickers[pair]

            if ticker["spread_pct"] >= 999:
                self.logger.warning("  Ticker unavailable for %s, skipping", pair)
                continue

            self.logger.info(
                "  Price: $%s | Bid: $%s | Ask: $%s | Spread: %s%%",
                f"{ticker['last']:,.4f}", f"{ticker['bid']:,.4f}",
                f"{ticker['ask']:,.4f}", f"{ticker['spread_pct']:.4f}",
            )

            if self.cooldown.is_on_cooldown(pair):
                self.logger.info("  %s on cooldown — skipping", pair)
                continue

            position = self.positions.get(pair)
            df       = self.smc_data[pair]
            htf_df   = self.htf_data.get(pair)

            # ── Trailing SL (BEFORE signal gen so exit check uses updated SL) ──
            if position["side"] == "long" and position.get("entry_time"):
                new_sl = get_trailing_sl(
                    df,
                    entry_time=position["entry_time"],
                    current_sl=position["sl_price"],
                )
                if new_sl > position["sl_price"]:
                    self.logger.info(
                        "  Trailing SL: %.6f -> %.6f (+%.4f%%)",
                        position["sl_price"], new_sl,
                        (new_sl - position["sl_price"]) / position["sl_price"] * 100,
                    )
                    self.positions.update_sl(pair, new_sl)
                    position = self.positions.get(pair)

            signal = self.signal_gen.generate_signal(
                df, ticker["last"], ticker["spread_pct"], position, htf_df=htf_df
            )

            self.logger.info("  Signal: %s", signal["action"])
            for r in signal["reasons"]:
                self.logger.info("    %s", r)

            if signal["action"] == "BUY":
                if daily_breaker_tripped:
                    self.logger.info("  BUY blocked: daily loss circuit breaker active")
                elif open_count >= MAX_CONCURRENT_POSITIONS:
                    self.logger.info(
                        "  BUY blocked: %d/%d positions already open",
                        open_count, MAX_CONCURRENT_POSITIONS,
                    )
                else:
                    self._execute_buy(pair, ticker, portfolio_value, signal)
                    open_count += 1
            elif signal["action"] == "SELL":
                self._execute_sell(pair, ticker, signal)
                open_count -= 1

            self._log_signal(pair, signal, ticker, position)

    # ── Buy ──────────────────────────────────────────────────────

    def _execute_buy(self, pair: str, ticker: dict,
                     portfolio_value: float, signal: dict):
        entry_price = ticker["ask"]
        sl_price    = signal["sl_price"]
        tp_price    = signal["tp_price"]

        sizing       = calculate_position_size(portfolio_value, entry_price, sl_price)
        position_usd = sizing["position_usd"]

        if position_usd < 1.0:
            self.logger.info("  Position too small ($%.2f) — skipping", position_usd)
            return

        quantity = position_usd / entry_price

        pair_info       = self.exchange_info.get("TradePairs", {}).get(pair, {})
        qty_precision   = pair_info.get("AmountPrecision", 6)
        price_precision = pair_info.get("PricePrecision", 2)
        step            = pair_info.get("AmountStep", None)

        if step and step > 0:
            quantity = int(quantity / step) * step
            quantity = round(quantity, qty_precision)
        else:
            factor   = 10 ** qty_precision
            quantity = int(quantity * factor) / factor

        if quantity * entry_price < 1.0:
            self.logger.info("  Order too small after rounding — skipping")
            return

        actual_risk_pct = sizing["actual_risk_pct"]
        target_risk_pct = RISK_PER_TRADE_PCT * 100
        if actual_risk_pct > target_risk_pct * 1.5:
            self.logger.warning(
                "  Risk cap kicked in: actual risk %.2f%% >> target %.2f%%",
                actual_risk_pct, target_risk_pct,
            )

        self.logger.info(
            "  BUYING %s: qty=%.6f | ~$%.2f (%.1f%% of portfolio) | "
            "SL=%.4f | TP=%.4f | Target risk=%.2f%% | Actual risk=%.2f%% | SL-dist=%.4f%%",
            pair, quantity, position_usd,
            sizing["position_pct"] * 100,
            sl_price, tp_price,
            target_risk_pct, actual_risk_pct,
            sizing["sl_distance_pct"],
        )

        result = self.executor.execute_buy(pair, quantity, ticker, price_precision=price_precision)

        if result.get("Success"):
            filled     = result.get("OrderDetail", {}).get("FilledAverPrice", entry_price) or entry_price
            entry_time = datetime.utcnow().isoformat()
            self.positions.open_position(pair, float(filled), quantity, sl_price, tp_price, entry_time)
            self.logger.info("  BUY filled at $%.4f | entry_time=%s", filled, entry_time)
        else:
            self.logger.error("  BUY failed: %s", result.get("ErrMsg", "unknown"))

    # ── Sell ─────────────────────────────────────────────────────

    def _execute_sell(self, pair: str, ticker: dict, signal: dict):
        position        = self.positions.get(pair)
        quantity        = position.get("quantity", 0)
        pair_info       = self.exchange_info.get("TradePairs", {}).get(pair, {})
        price_precision = pair_info.get("PricePrecision", 2)

        if quantity <= 0:
            self.logger.warning("  No position to sell for %s", pair)
            return

        self.logger.info(
            "  SELLING %s: qty=%.6f | %s",
            pair, quantity,
            signal["reasons"][0] if signal["reasons"] else "signal",
        )

        result = self.executor.execute_sell(pair, quantity, ticker, price_precision=price_precision)

        if result.get("Success"):
            filled  = result.get("OrderDetail", {}).get("FilledAverPrice", ticker["bid"]) or ticker["bid"]
            entry   = position.get("entry_price", 0)
            pnl_pct = ((filled - entry) / entry * 100) if entry else 0
            self.positions.close_position(pair)
            self.cooldown.start_cooldown(pair)
            self.logger.info("  SELL filled at $%.4f | PnL: %+.2f%%", filled, pnl_pct)
        else:
            self.logger.error("  SELL failed: %s", result.get("ErrMsg", "unknown"))

    # ── Data refresh ─────────────────────────────────────────────

    def _maybe_refresh_data(self):
        now = datetime.utcnow()
        for pair in PAIRS:
            last = self.last_refresh.get(pair)
            if last is None:
                continue

            minutes_since = (now - last).total_seconds() / 60
            hours_since   = minutes_since / 60

            if minutes_since >= DATA_REFRESH_INTERVAL_MINUTES:
                self.logger.info(
                    "Appending new 5m candles for %s (%.1f min since last)", pair, minutes_since
                )
                try:
                    df = append_new_candles(pair)
                    if not df.empty:
                        self.smc_data[pair]     = compute_smc_indicators(df)
                        self.last_refresh[pair] = now
                        self.logger.info(
                            "%s: %d x 5m candles (incremental update)", pair, len(self.smc_data[pair])
                        )
                except Exception as e:
                    self.logger.error("5m append failed for %s: %s", pair, e)

            if hours_since >= DATA_REFRESH_INTERVAL_HOURS:
                self.logger.info("Refreshing 1H HTF data for %s", pair)
                try:
                    htf_df = load_or_fetch_htf(pair, force_refresh=True)
                    if not htf_df.empty:
                        self.htf_data[pair] = compute_smc_indicators(htf_df)
                except Exception as e:
                    self.logger.error("HTF refresh failed for %s: %s", pair, e)

    # ── Signal logger ────────────────────────────────────────────

    def _log_signal(self, pair: str, signal: dict, ticker: dict, position: dict):
        os.makedirs(LOG_DIR, exist_ok=True)
        entry = {
            "timestamp":  datetime.utcnow().isoformat(),
            "pair":       pair,
            "action":     signal["action"],
            "reasons":    signal["reasons"],
            "sl_price":   signal.get("sl_price"),
            "tp_price":   signal.get("tp_price"),
            "conf_score": signal.get("conf_score"),
            "ticker": {
                "last":       ticker["last"],
                "bid":        ticker["bid"],
                "ask":        ticker["ask"],
                "spread_pct": ticker["spread_pct"],
            },
            "position": position,
        }
        with open(LOG_SIGNALS_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SMC Crypto Trading Bot — FINAL V2")
    parser.parse_args()
    setup_logging()
    TradingBot().run()


if __name__ == "__main__":
    main()
