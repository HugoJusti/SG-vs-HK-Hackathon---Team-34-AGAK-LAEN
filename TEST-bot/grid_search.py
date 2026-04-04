"""
SMC Parameter Grid Search
==========================
Exhaustively backtests the SMCSignalGenerator across a grid of key parameters
to find the combination that maximises risk-adjusted returns.

Parameters searched:
  MIN_RR_RATIO               minimum reward:risk ratio before entry
  BOS_LOOKBACK               swing-high lookback for BOS detection
  BOS_MAX_AGE_CANDLES        reject BOS older than N candles (staleness)
  MIN_CONFLUENCES            score threshold at Gate 4
  SWING_SIGNIFICANCE_ATR_MULT  fractal significance filter (higher = fewer, stronger swings)

Strategy:
  - First 80% of data  = training  (grid search here)
  - Last  20% of data  = test      (top-5 combos re-validated here)
  - Ranked by composite score = expectancy_pct * min(profit_factor, 3.0)
  - Per-parameter sensitivity table printed at end

Usage:
    python grid_search.py

Runtime: roughly 5-20 min depending on machine and how much data is cached.
"""

import os
import sys
import time
import warnings
import itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import config
import strategy as strat
from data import load_or_fetch_historical, load_or_fetch_htf, compute_smc_indicators
from config import PAIRS as DEFAULT_PAIRS
from strategy import SMCSignalGenerator, get_trailing_sl, calculate_position_size

# ── Grid definition ───────────────────────────────────────────────
# Edit these to narrow or widen the search.
PARAM_GRID = {
    "MIN_RR_RATIO":                [1.0, 1.5, 2.0],
    "BOS_LOOKBACK":                [5, 10, 30],
    "BOS_MAX_AGE_CANDLES":         [5, 10, 20, 30],
    "MIN_CONFLUENCES":             [3],
    "SWING_SIGNIFICANCE_ATR_MULT": [0.2, 0.5],
}

# ── Simulation constants ──────────────────────────────────────────
STARTING_BALANCE    = 50_000.0
COMMISSION          = 0.0005       # 0.05% per side
BACKTEST_SPREAD_PCT = 0.001        # 0.1% simulated spread
WARMUP_CANDLES      = 300          # bars needed before first signal is attempted
TRAIN_RATIO         = 0.80
PAIRS               = DEFAULT_PAIRS
TOP_N               = 5            # combos to re-run on test data


# ═══════════════════════════════════════════════════════════════
# PARAMETER PATCHING
# ═══════════════════════════════════════════════════════════════

# Names that live in strategy's module namespace (imported from config at load time)
_STRATEGY_ATTRS = {
    "MIN_RR_RATIO", "BOS_LOOKBACK", "BOS_MAX_AGE_CANDLES",
    "MIN_CONFLUENCES", "SWING_SIGNIFICANCE_ATR_MULT",
    # BOS_CLOSE_BUFFER_PCT and others also live here if needed later
}


def apply_params(params: dict):
    """
    Patch both the config module and the strategy module's local bindings.
    strategy.py uses `from config import X`, so the names are bound in its
    own namespace — we must patch strategy.X directly, not just config.X.
    """
    for key, val in params.items():
        setattr(config, key, val)
        if key in _STRATEGY_ATTRS:
            setattr(strat, key, val)


def restore_defaults():
    """Restore original config values after the grid search."""
    defaults = {
        "MIN_RR_RATIO":                1.5,
        "BOS_LOOKBACK":                30,
        "BOS_MAX_AGE_CANDLES":         10,
        "MIN_CONFLUENCES":             3,
        "SWING_SIGNIFICANCE_ATR_MULT": 0.3,
    }
    apply_params(defaults)


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_all_pairs() -> dict:
    """Return {pair: {"df5": DataFrame, "df1h": DataFrame}}."""
    frames = {}
    for pair in PAIRS:
        print(f"  Loading {pair}...", end=" ", flush=True)
        df5 = load_or_fetch_historical(pair, force_refresh=False)
        if df5.empty:
            print("NO 5m DATA — skipping")
            continue
        df5  = compute_smc_indicators(df5)
        df1h = load_or_fetch_htf(pair, force_refresh=False)
        if not df1h.empty:
            df1h = compute_smc_indicators(df1h)
        frames[pair] = {"df5": df5, "df1h": df1h}
        print(f"{len(df5)} x 5m  |  {len(df1h)} x 1H")
    return frames


def _htf_slice(df1h: pd.DataFrame, current_time) -> pd.DataFrame:
    if df1h.empty or "open_time" not in df1h.columns:
        return df1h
    return df1h[df1h["open_time"] <= current_time].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════
# SINGLE BACKTEST RUN
# ═══════════════════════════════════════════════════════════════

def run_single(pair_frames: dict) -> dict:
    """
    Simulate SMC trading across all pairs with current module-level params.
    SL/TP hit checked via candle high/low (not just close) for realism.
    """
    gen   = SMCSignalGenerator()
    cash  = STARTING_BALANCE
    pos   = {p: _empty_pos() for p in pair_frames}
    trades = []
    equity = [STARTING_BALANCE]

    min_len = min(len(v["df5"]) for v in pair_frames.values() if not v["df5"].empty)

    for i in range(WARMUP_CANDLES, min_len):
        bar_value = cash
        for pair, p in pos.items():
            if p["side"] == "long" and p["qty"] > 0:
                bar_value += p["qty"] * float(pair_frames[pair]["df5"].iloc[i]["close"])
        equity.append(bar_value)

        for pair, frames in pair_frames.items():
            df5  = frames["df5"]
            df1h = frames.get("df1h", pd.DataFrame())

            if df5.empty or i >= len(df5):
                continue

            df_slice = df5.iloc[:i + 1].reset_index(drop=True)
            row      = df_slice.iloc[-1]
            close    = float(row["close"])
            high     = float(row["high"])
            low      = float(row["low"])
            ask      = close * (1 + BACKTEST_SPREAD_PCT / 2)
            bid      = close * (1 - BACKTEST_SPREAD_PCT / 2)

            p = pos[pair]

            # ── Check SL/TP via candle range first (realistic) ──────
            if p["side"] == "long":
                # Update trailing SL before exit check
                if p.get("entry_time"):
                    new_sl = get_trailing_sl(df_slice, p["entry_time"], p["sl_price"])
                    if new_sl > p["sl_price"]:
                        p["sl_price"] = new_sl

                sl_hit = low  <= p["sl_price"]
                tp_hit = high >= p["tp_price"]

                if sl_hit or tp_hit:
                    fill   = p["sl_price"] if sl_hit else p["tp_price"]
                    pnl    = (fill - p["entry_price"]) * p["qty"]
                    cash  += p["qty"] * fill * (1 - COMMISSION)
                    trades.append({
                        "pair":    pair, "bar": i,
                        "pnl":     round(pnl, 4),
                        "pnl_pct": (fill - p["entry_price"]) / p["entry_price"] * 100,
                        "exit":    "SL" if sl_hit else "TP",
                    })
                    pos[pair] = _empty_pos()
                    continue

            # ── Signal generation (entry logic only when flat) ───────
            if p["side"] != "none":
                continue

            htf = _htf_slice(df1h, row["open_time"]) if "open_time" in row.index else df1h
            sig = gen.generate_signal(
                df_slice, close, BACKTEST_SPREAD_PCT,
                _sig_pos(),  # always present as flat
                htf_df=htf if not htf.empty else None,
            )

            if sig["action"] == "BUY":
                sl_price = sig["sl_price"]
                tp_price = sig["tp_price"]
                sizing   = calculate_position_size(cash, ask, sl_price)
                pos_usd  = sizing["position_usd"]

                if pos_usd < 1.0 or cash < pos_usd:
                    continue

                fill     = ask * (1 + COMMISSION)
                qty      = pos_usd / fill
                cash    -= pos_usd

                pos[pair] = {
                    "side":        "long",
                    "entry_price": round(fill, 8),
                    "qty":         qty,
                    "sl_price":    sl_price,
                    "tp_price":    tp_price,
                    "entry_time":  str(row.get("open_time", i)),
                }

    # Mark open positions at final bar
    final_val = cash
    for pair, p in pos.items():
        if p["side"] == "long" and p["qty"] > 0:
            df5 = pair_frames[pair]["df5"]
            if not df5.empty:
                final_val += p["qty"] * float(df5.iloc[-1]["close"]) * (1 - COMMISSION)

    return _calc_stats(final_val, trades, equity)


def _empty_pos():
    return {"side": "none", "entry_price": 0.0, "qty": 0.0,
            "sl_price": 0.0, "tp_price": 0.0, "entry_time": ""}


def _sig_pos():
    """Flat position dict for signal generator (no side)."""
    return {"side": "none", "entry_price": 0.0, "quantity": 0.0,
            "sl_price": 0.0, "tp_price": 0.0, "entry_time": ""}


def _calc_stats(final_val: float, trades: list, equity: list) -> dict:
    equity_arr   = np.array(equity)
    wins         = [t for t in trades if t["pnl"] > 0]
    losses       = [t for t in trades if t["pnl"] <= 0]
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss   = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )
    win_rate    = len(wins) / len(trades) if trades else 0
    avg_win_pct = np.mean([t["pnl_pct"] for t in wins])   if wins   else 0
    avg_los_pct = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
    # Expectancy in % per trade
    expectancy  = win_rate * avg_win_pct - (1 - win_rate) * abs(avg_los_pct)
    # Max drawdown
    peak   = np.maximum.accumulate(equity_arr)
    max_dd = float(((equity_arr - peak) / np.where(peak > 0, peak, 1) * 100).min())
    # Composite score: expectancy × clipped profit factor (rewards consistency, not outliers)
    score  = expectancy * min(profit_factor, 3.0) if len(trades) >= 3 else -999

    return {
        "final":          round(final_val, 2),
        "return_pct":     round((final_val - STARTING_BALANCE) / STARTING_BALANCE * 100, 2),
        "n_trades":       len(trades),
        "win_rate":       round(win_rate * 100, 1),
        "profit_factor":  round(profit_factor, 3),
        "expectancy_pct": round(expectancy, 4),
        "avg_win_pct":    round(avg_win_pct, 3),
        "avg_loss_pct":   round(avg_los_pct, 3),
        "max_drawdown":   round(max_dd, 2),
        "score":          round(score, 4),
        "trades":         trades,
    }


# ═══════════════════════════════════════════════════════════════
# GRID SEARCH
# ═══════════════════════════════════════════════════════════════

def grid_search(train_frames: dict) -> list:
    """
    Run every combination in PARAM_GRID on training data.
    Returns list of result dicts sorted by score descending.
    """
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total  = len(combos)

    print(f"\n  {total} parameter combinations to test across {len(train_frames)} pairs")
    print(f"  Parameters: {', '.join(keys)}")
    print(f"  This may take several minutes...\n")

    results = []
    t0      = time.time()

    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        apply_params(params)

        try:
            stats = run_single(train_frames)
        except Exception as e:
            stats = {"score": -999, "n_trades": 0, "return_pct": 0,
                     "win_rate": 0, "profit_factor": 0,
                     "expectancy_pct": 0, "max_drawdown": 0,
                     "avg_win_pct": 0, "avg_loss_pct": 0,
                     "final": STARTING_BALANCE, "trades": []}

        results.append({**params, **stats})

        # Progress update every 20 combos
        if idx % 20 == 0 or idx == total:
            elapsed  = time.time() - t0
            eta      = (elapsed / idx) * (total - idx)
            best     = max(results, key=lambda r: r["score"])
            print(f"  [{idx:>4}/{total}]  elapsed={elapsed:5.0f}s  ETA={eta:5.0f}s  "
                  f"| best so far: score={best['score']:.4f}  "
                  f"RR={best['MIN_RR_RATIO']}  BOS_LB={best['BOS_LOOKBACK']}  "
                  f"CONF={best['MIN_CONFLUENCES']}")

    restore_defaults()
    return sorted(results, key=lambda r: r["score"], reverse=True)


# ═══════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═══════════════════════════════════════════════════════════════

def _param_abbr(keys):
    abbr = {
        "MIN_RR_RATIO":                "RR",
        "BOS_LOOKBACK":                "BOS_LB",
        "BOS_MAX_AGE_CANDLES":         "BOS_AGE",
        "MIN_CONFLUENCES":             "CONF",
        "SWING_SIGNIFICANCE_ATR_MULT": "SWING_SIG",
    }
    return [abbr.get(k, k[:8]) for k in keys]


def print_leaderboard(results: list, label: str, n: int = 20):
    keys   = list(PARAM_GRID.keys())
    abbrs  = _param_abbr(keys)

    print(f"\n{'='*110}")
    print(f"  {label}  (top {min(n, len(results))})")
    print(f"{'='*110}")

    # Header
    param_hdr = "  ".join(f"{a:>9}" for a in abbrs)
    print(f"  {'#':>3}  {param_hdr}  "
          f"{'SCORE':>8}  {'RET%':>7}  {'N':>4}  {'WR%':>5}  "
          f"{'PF':>6}  {'EV%':>7}  {'DD%':>7}")
    print(f"  {'-'*105}")

    for rank, r in enumerate(results[:n], 1):
        param_vals = "  ".join(f"{r[k]:>9}" for k in keys)
        pf_str     = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 99 else "inf"
        print(f"  {rank:>3}  {param_vals}  "
              f"{r['score']:>8.4f}  {r['return_pct']:>+7.2f}%  {r['n_trades']:>4}  "
              f"{r['win_rate']:>5.1f}%  {pf_str:>6}  "
              f"{r['expectancy_pct']:>+7.4f}  {r['max_drawdown']:>+7.2f}%")


def print_sensitivity(results: list):
    """Show how each parameter affects the average score when held fixed."""
    keys  = list(PARAM_GRID.keys())
    abbrs = _param_abbr(keys)

    print(f"\n{'='*80}")
    print("  PARAMETER SENSITIVITY  (avg score per value, over all other params)")
    print(f"{'='*80}")

    for key, abbr in zip(keys, abbrs):
        vals       = PARAM_GRID[key]
        print(f"\n  {abbr} ({key})")
        print(f"  {'Value':>12}  {'Avg Score':>10}  {'Avg WR%':>8}  {'Avg PF':>8}  {'Avg EV%':>9}  Count")
        for v in vals:
            subset = [r for r in results if r[key] == v]
            if not subset:
                continue
            avg_score = np.mean([r["score"] for r in subset if r["score"] > -900])
            avg_wr    = np.mean([r["win_rate"] for r in subset])
            avg_pf    = np.mean([min(r["profit_factor"], 5) for r in subset])
            avg_ev    = np.mean([r["expectancy_pct"] for r in subset])
            print(f"  {v:>12}  {avg_score:>10.4f}  {avg_wr:>8.1f}%  "
                  f"{avg_pf:>8.3f}  {avg_ev:>+9.4f}  {len(subset)}")


def print_oos_results(oos_results: list, top_combos: list):
    """Show out-of-sample validation for top combos."""
    keys  = list(PARAM_GRID.keys())
    abbrs = _param_abbr(keys)

    print(f"\n{'='*90}")
    print(f"  OUT-OF-SAMPLE VALIDATION  (top {len(top_combos)} combos on held-out 20%)")
    print(f"{'='*90}")
    param_hdr = "  ".join(f"{a:>9}" for a in abbrs)
    print(f"  {'RANK':>4}  {param_hdr}  "
          f"{'TRAIN SC':>9}  {'OOS SC':>8}  {'OOS RET%':>9}  {'OOS N':>6}  {'OOS WR%':>7}")
    print(f"  {'-'*85}")

    for rank, (train_r, oos_r) in enumerate(zip(top_combos, oos_results), 1):
        param_vals = "  ".join(f"{train_r[k]:>9}" for k in keys)
        print(f"  {rank:>4}  {param_vals}  "
              f"{train_r['score']:>+9.4f}  {oos_r['score']:>+8.4f}  "
              f"{oos_r['return_pct']:>+9.2f}%  {oos_r['n_trades']:>6}  "
              f"{oos_r['win_rate']:>7.1f}%")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  SMC PARAMETER GRID SEARCH")
    print(f"  Pairs   : {', '.join(PAIRS)}")
    print(f"  Balance : ${STARTING_BALANCE:,.0f}  |  Commission: {COMMISSION*100:.3f}%/side")
    print(f"  Split   : {int(TRAIN_RATIO*100)}% train / {int((1-TRAIN_RATIO)*100)}% test")
    print("=" * 80)

    # ── Load data ─────────────────────────────────────────────────
    print("\nLoading data...")
    all_frames = load_all_pairs()
    if not all_frames:
        print("No data loaded. Run the bot at least once to populate data/ cache.")
        return

    min_len = min(len(v["df5"]) for v in all_frames.values())
    split   = int(min_len * TRAIN_RATIO)

    print(f"\n  Shortest series : {min_len} bars")
    print(f"  Training range  : bars 0–{split}  ({split} bars = {split*5/60:.0f}h of 5m data)")
    print(f"  Test range      : bars {split}–{min_len}  ({min_len-split} bars)")

    train_frames = {
        p: {"df5": v["df5"].iloc[:split].reset_index(drop=True), "df1h": v["df1h"]}
        for p, v in all_frames.items()
    }
    test_frames = {
        p: {"df5": v["df5"].iloc[split:].reset_index(drop=True), "df1h": v["df1h"]}
        for p, v in all_frames.items()
    }

    # ── Baseline (current config) ─────────────────────────────────
    print("\nBaseline run (current config values)...")
    try:
        baseline = run_single(train_frames)
        print(f"  Baseline: score={baseline['score']:.4f}  "
              f"return={baseline['return_pct']:+.2f}%  "
              f"n={baseline['n_trades']}  WR={baseline['win_rate']:.1f}%  "
              f"PF={baseline['profit_factor']:.3f}  "
              f"EV={baseline['expectancy_pct']:+.4f}%  "
              f"DD={baseline['max_drawdown']:.2f}%")
    except Exception as e:
        print(f"  Baseline failed: {e}")

    # ── Grid search ───────────────────────────────────────────────
    t_start  = time.time()
    all_results = grid_search(train_frames)
    elapsed  = time.time() - t_start
    print(f"\n  Grid search complete in {elapsed:.0f}s  ({elapsed/60:.1f} min)")

    # ── Leaderboard ───────────────────────────────────────────────
    print_leaderboard(all_results, "TRAINING LEADERBOARD", n=20)

    # ── Sensitivity analysis ──────────────────────────────────────
    print_sensitivity(all_results)

    # ── Out-of-sample test for top N ─────────────────────────────
    top_combos = [r for r in all_results[:TOP_N] if r["score"] > -900]
    print(f"\nRunning top {len(top_combos)} combos on test data...")

    oos_results = []
    for r in top_combos:
        params = {k: r[k] for k in PARAM_GRID}
        apply_params(params)
        try:
            oos = run_single(test_frames)
        except Exception as e:
            oos = {"score": -999, "return_pct": 0, "n_trades": 0, "win_rate": 0}
        oos_results.append(oos)

    restore_defaults()
    print_oos_results(oos_results, top_combos)

    # ── Recommendation ────────────────────────────────────────────
    # Pick the combo with best OOS score (not train score — avoids overfitting)
    if oos_results:
        best_oos_idx = max(range(len(oos_results)), key=lambda i: oos_results[i]["score"])
        best         = top_combos[best_oos_idx]
        print(f"\n{'='*80}")
        print("  RECOMMENDED PARAMETERS  (best OOS score)")
        print(f"{'='*80}")
        for k in PARAM_GRID:
            print(f"  {k:<35} = {best[k]}")
        print(f"\n  Training score : {best['score']:+.4f}")
        print(f"  OOS score      : {oos_results[best_oos_idx]['score']:+.4f}")
        print(f"  OOS return     : {oos_results[best_oos_idx]['return_pct']:+.2f}%")
        print(f"  OOS win rate   : {oos_results[best_oos_idx]['win_rate']:.1f}%")
        print(f"\n  Paste these into config.py to use the optimised settings.")
    print()


if __name__ == "__main__":
    main()
