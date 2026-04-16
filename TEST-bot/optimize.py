"""
SMC Backtest Engine
===================
Backtests the EXACT same SMCSignalGenerator logic used by the live bot.

- Same pairs:      TRX/USD, TAO/USD, SOL/USD
- Same data:       5-minute candles + 1H HTF filter
- Same strategy:   strategy.py → SMCSignalGenerator (no separate model)
- Same SL/TP:      sweep-low SL, structural/ATR TP
- Same trailing:   get_trailing_sl() called every bar
- Same spread:     simulated at BACKTEST_SPREAD_PCT

Split: first 80% optimisation, last 20% out-of-sample test.
Reports: return, trades, win rate, max drawdown, profit factor, expectancy.

Usage:
    python optimize.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── make sure project root is on path ────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from data import load_or_fetch_historical, load_or_fetch_htf, compute_smc_indicators
from strategy import SMCSignalGenerator, get_trailing_sl

# ── Backtest settings ─────────────────────────────────────────────
STARTING_BALANCE    = 50_000.0
COMMISSION          = 0.0005          # 0.05% per side (Roostoo rate)
BACKTEST_SPREAD_PCT = 0.001           # 0.1% simulated spread
WARMUP_CANDLES      = 300             # candles needed before first signal
PAIRS               = ["TRX/USD", "TAO/USD", "SOL/USD", "FET/USD", "AVAX/USD", "BNB/USD"]


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_pair(pair: str):
    """
    Load 5m + 1H data for a pair, compute SMC indicators.
    Returns (df_5m, df_1h) or (empty, empty) on failure.
    """
    print(f"  Loading {pair}…", end=" ", flush=True)
    df5 = load_or_fetch_historical(pair, force_refresh=False)
    if df5.empty:
        print("NO 5m DATA")
        return pd.DataFrame(), pd.DataFrame()
    df5 = compute_smc_indicators(df5)

    df1h = load_or_fetch_htf(pair, force_refresh=False)
    if not df1h.empty:
        df1h = compute_smc_indicators(df1h)

    print(f"{len(df5)} × 5m  |  {len(df1h)} × 1H")
    return df5, df1h


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def _htf_slice(df1h: pd.DataFrame, current_time: pd.Timestamp) -> pd.DataFrame:
    """Return 1H rows whose open_time is at or before current_time."""
    if df1h.empty or "open_time" not in df1h.columns:
        return df1h
    return df1h[df1h["open_time"] <= current_time].reset_index(drop=True)


def run_backtest(pair_frames: dict) -> dict:
    """
    Simulate SMC trading on all pairs simultaneously.
    pair_frames: {pair: {"df5": DataFrame, "df1h": DataFrame}}
    Returns performance stats dict + list of trades.
    """
    generator = SMCSignalGenerator()
    cash      = STARTING_BALANCE

    positions = {
        pair: {
            "side": "none", "entry_price": 0.0,
            "quantity": 0.0, "sl_price": 0.0,
            "tp_price": 0.0, "entry_time": "",
        }
        for pair in pair_frames
    }

    trades       = []
    equity_curve = [STARTING_BALANCE]

    # Use shortest 5m series as the timeline
    min_len = min(len(v["df5"]) for v in pair_frames.values() if not v["df5"].empty)

    for i in range(WARMUP_CANDLES, min_len):
        # Mark-to-market equity snapshot
        bar_value = cash
        for pair, pos in positions.items():
            if pos["side"] == "long" and pos["quantity"] > 0:
                last = pair_frames[pair]["df5"].iloc[i]["close"]
                bar_value += pos["quantity"] * last
        equity_curve.append(bar_value)

        for pair, frames in pair_frames.items():
            df5  = frames["df5"]
            df1h = frames["df1h"]
            if df5.empty or i >= len(df5):
                continue

            df_slice = df5.iloc[: i + 1].reset_index(drop=True)
            row      = df_slice.iloc[-1]
            current_price = float(row["close"])

            # Simulate spread: ask slightly above close
            ask = current_price * (1 + BACKTEST_SPREAD_PCT / 2)
            bid = current_price * (1 - BACKTEST_SPREAD_PCT / 2)

            pos = positions[pair]

            # ── Trailing SL (before signal, so exit uses updated SL) ──
            if pos["side"] == "long" and pos.get("entry_time"):
                new_sl = get_trailing_sl(df_slice, pos["entry_time"], pos["sl_price"])
                if new_sl > pos["sl_price"]:
                    pos["sl_price"] = new_sl

            # ── HTF slice aligned to current bar time ─────────────────
            if "open_time" in row.index:
                htf_slice = _htf_slice(df1h, row["open_time"])
            else:
                htf_slice = df1h

            signal = generator.generate_signal(
                df_slice, current_price, BACKTEST_SPREAD_PCT, pos,
                htf_df=htf_slice if not htf_slice.empty else None,
            )

            # ── Execute BUY ───────────────────────────────────────────
            if signal["action"] == "BUY" and pos["side"] == "none":
                from strategy import calculate_position_size
                sl_price = signal["sl_price"]
                tp_price = signal["tp_price"]

                sizing       = calculate_position_size(cash, ask, sl_price)
                position_usd = sizing["position_usd"]

                if position_usd < 1.0 or cash < position_usd:
                    continue

                fill_price = ask * (1 + COMMISSION)
                quantity   = position_usd / fill_price
                cash      -= position_usd

                entry_time = str(row.get("open_time", i))
                positions[pair] = {
                    "side":        "long",
                    "entry_price": round(fill_price, 8),
                    "quantity":    quantity,
                    "sl_price":    sl_price,
                    "tp_price":    tp_price,
                    "entry_time":  entry_time,
                }
                trades.append({
                    "pair":       pair,
                    "side":       "BUY",
                    "price":      fill_price,
                    "qty":        quantity,
                    "bar":        i,
                    "time":       entry_time,
                    "conf_score": signal.get("conf_score"),
                })

            # ── Execute SELL ──────────────────────────────────────────
            elif signal["action"] == "SELL" and pos["side"] == "long":
                fill_price = bid * (1 - COMMISSION)
                qty        = pos["quantity"]
                entry      = pos["entry_price"]
                pnl        = (fill_price - entry) * qty
                cash      += qty * fill_price

                sell_time = str(row.get("open_time", i))
                trades.append({
                    "pair":    pair,
                    "side":    "SELL",
                    "price":   fill_price,
                    "qty":     qty,
                    "bar":     i,
                    "time":    sell_time,
                    "pnl":     round(pnl, 4),
                    "reason":  signal["reasons"][0] if signal["reasons"] else "",
                })
                positions[pair] = {
                    "side": "none", "entry_price": 0.0,
                    "quantity": 0.0, "sl_price": 0.0,
                    "tp_price": 0.0, "entry_time": "",
                }

    # Mark open positions to market at last bar
    portfolio_value = cash
    for pair, pos in positions.items():
        if pos["side"] == "long" and pos["quantity"] > 0:
            df5 = pair_frames[pair]["df5"]
            if not df5.empty:
                last_price       = float(df5.iloc[-1]["close"])
                portfolio_value += pos["quantity"] * last_price * (1 - COMMISSION)

    equity_arr = np.array(equity_curve)

    # Performance stats
    sells        = [t for t in trades if t["side"] == "SELL"]
    wins         = [t for t in sells if t.get("pnl", 0) > 0]
    losses       = [t for t in sells if t.get("pnl", 0) <= 0]
    total_pnl    = sum(t.get("pnl", 0) for t in sells)
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win   = gross_profit / len(wins)   if wins   else 0
    avg_loss  = gross_loss  / len(losses)  if losses else 0
    win_rate  = len(wins) / len(sells) if sells else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Max drawdown
    peak        = np.maximum.accumulate(equity_arr)
    drawdowns   = (equity_arr - peak) / peak * 100
    max_dd      = float(drawdowns.min())

    return {
        "final_balance":  round(portfolio_value, 2),
        "total_return":   round((portfolio_value - STARTING_BALANCE) / STARTING_BALANCE * 100, 2),
        "n_trades":       len(sells),
        "win_rate":       round(win_rate * 100, 1),
        "total_pnl":      round(total_pnl, 2),
        "profit_factor":  round(profit_factor, 3),
        "expectancy_usd": round(expectancy, 2),
        "avg_win_usd":    round(avg_win, 2),
        "avg_loss_usd":   round(avg_loss, 2),
        "max_drawdown":   round(max_dd, 2),
        "trades":         trades,
    }


# ═══════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═══════════════════════════════════════════════════════════════

def print_stats(label: str, stats: dict):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Starting balance : ${STARTING_BALANCE:>12,.2f}")
    print(f"  Final balance    : ${stats['final_balance']:>12,.2f}")
    print(f"  Total return     : {stats['total_return']:>+10.2f}%")
    print(f"  Total trades     : {stats['n_trades']:>12}")
    print(f"  Win rate         : {stats['win_rate']:>11.1f}%")
    print(f"  Profit factor    : {stats['profit_factor']:>12.3f}")
    print(f"  Expectancy       : ${stats['expectancy_usd']:>11.2f}  per trade")
    print(f"  Avg win          : ${stats['avg_win_usd']:>11.2f}")
    print(f"  Avg loss         : ${stats['avg_loss_usd']:>11.2f}")
    print(f"  Max drawdown     : {stats['max_drawdown']:>11.2f}%")
    print(f"  Total PnL        : ${stats['total_pnl']:>11.2f}")


def print_trade_log(trades: list, max_rows: int = 20):
    sells = [t for t in trades if t["side"] == "SELL"]
    if not sells:
        return
    print(f"\n  Last {min(max_rows, len(sells))} closed trades:")
    print(f"  {'#':<4} {'Pair':<10} {'Time':<22} {'PnL':>10}  Reason")
    for idx, t in enumerate(sells[-max_rows:], 1):
        pnl    = t.get("pnl", 0)
        symbol = "+" if pnl >= 0 else ""
        reason = t.get("reason", "")[:55]
        print(f"  {idx:<4} {t['pair']:<10} {str(t['time'])[:22]:<22} "
              f"{symbol}{pnl:>9.2f}  {reason}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SMC BACKTEST ENGINE")
    print(f"  Pairs : {', '.join(PAIRS)}")
    print(f"  Logic : strategy.py → SMCSignalGenerator (exact live logic)")
    print(f"  Split : 80% in-sample / 20% out-of-sample")
    print("=" * 65)

    # Load data
    print("\nLoading data…")
    all_frames = {}
    for pair in PAIRS:
        df5, df1h = load_pair(pair)
        if not df5.empty:
            all_frames[pair] = {"df5": df5, "df1h": df1h}

    if not all_frames:
        print("No data loaded. Run the bot once to populate the data/ cache.")
        return

    # Find split index
    min_len = min(len(v["df5"]) for v in all_frames.values())
    split   = int(min_len * 0.80)
    print(f"\nTotal candles (shortest pair): {min_len}")
    print(f"In-sample  : 0 → {split}  ({split} bars)")
    print(f"Out-sample : {split} → {min_len}  ({min_len - split} bars)")

    # ── In-sample backtest ──────────────────────────────────────
    print("\nRunning IN-SAMPLE backtest…")
    in_frames = {p: {"df5": v["df5"].iloc[:split].reset_index(drop=True),
                     "df1h": v["df1h"]}
                 for p, v in all_frames.items()}
    in_stats = run_backtest(in_frames)
    print_stats("IN-SAMPLE RESULTS (first 80%)", in_stats)
    print_trade_log(in_stats["trades"])

    # ── Out-of-sample backtest ──────────────────────────────────
    print("\nRunning OUT-OF-SAMPLE backtest…")
    oos_frames = {p: {"df5": v["df5"].iloc[split:].reset_index(drop=True),
                      "df1h": v["df1h"]}
                  for p, v in all_frames.items()}
    oos_stats = run_backtest(oos_frames)
    print_stats("OUT-OF-SAMPLE RESULTS (last 20%)", oos_stats)
    print_trade_log(oos_stats["trades"])

    # ── Per-pair breakdown ──────────────────────────────────────
    print(f"\n{'='*65}")
    print("  PER-PAIR TRADE BREAKDOWN (out-of-sample)")
    print(f"{'='*65}")
    for pair in PAIRS:
        pair_sells = [t for t in oos_stats["trades"]
                      if t["side"] == "SELL" and t["pair"] == pair]
        if not pair_sells:
            print(f"  {pair:<10}: no closed trades")
            continue
        pair_pnl  = sum(t.get("pnl", 0) for t in pair_sells)
        pair_wins = sum(1 for t in pair_sells if t.get("pnl", 0) > 0)
        print(f"  {pair:<10}: {len(pair_sells):>3} trades | "
              f"WR={pair_wins/len(pair_sells)*100:.0f}% | "
              f"PnL=${pair_pnl:>+,.2f}")

    print()


if __name__ == "__main__":
    main()
