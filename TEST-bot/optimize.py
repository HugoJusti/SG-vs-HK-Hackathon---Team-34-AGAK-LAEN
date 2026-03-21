"""
Backtest Optimizer
==================
Finds the best entry/exit parameters using random search over historical data.

- Splits data: first 80% for optimization, last 20% for out-of-sample test
- Uses random search (300 iterations) to maximize final portfolio value
- Simulates realistic trading: 0.05% commission, position sizing, stop loss, take profit
- Reports top 5 parameter sets and final balance starting from $50,000

Usage:
    python optimize.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import random
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────
STARTING_BALANCE   = 50_000.0
COMMISSION         = 0.0005       # 0.05% per trade (Roostoo rate)
N_RANDOM_TRIALS    = 50           # number of random parameter sets to try
PAIRS              = ["BTC/USD", "ETH/USD", "ZEC/USD"]
DATA_DIR           = "data"
MODEL_DIR          = "models"
EMA_SPAN           = 20           # matches ENTRY_MA_SHORT in config
MOMENTUM_PERIODS   = 10           # matches data.py

# ── Parameter search space ────────────────────────────────────
FIXED_STOP_LOSS    = 0.03    # fixed 3%
FIXED_TAKE_PROFIT  = 0.03    # fixed 3%

PARAM_SPACE = {
    "hmm_confidence":    (0.50, 0.90),   # ENTRY_HMM_CONFIDENCE
    "max_volatility":    (0.30, 3.00),   # ENTRY_MAX_VOLATILITY
    "momentum_min":      (-0.01, 0.02),  # ENTRY_MOMENTUM_MIN
    "vol_zscore_max":    (2.00, 6.00),   # ENTRY_VOLUME_ZSCORE_MAX
    "spread_max_pct":    (0.01, 0.10),   # ENTRY_SPREAD_MAX_PCT
    "position_pct":      (0.10, 0.40),   # MC_MAX_POSITION_PCT
}


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def load_data(pair: str) -> pd.DataFrame:
    coin = pair.split("/")[0]
    path = os.path.join(DATA_DIR, f"{coin}USDT_1h.csv")
    if not os.path.exists(path):
        print(f"  [WARN] No data file for {pair}, skipping.")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"]    = np.log(df["close"] / df["close"].shift(1))
    df["rolling_vol"]   = df["log_return"].rolling(20).std() * np.sqrt(105120)
    df["momentum"]      = df["close"].pct_change(periods=MOMENTUM_PERIODS)
    vol_mean            = df["volume"].rolling(20).mean()
    vol_std             = df["volume"].rolling(20).std()
    df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std
    df["ema20"]         = df["close"].ewm(span=EMA_SPAN, adjust=False).mean()
    df = df.dropna().reset_index(drop=True)
    return df


def load_hmm(pair: str):
    path = os.path.join(MODEL_DIR, f"hmm_{pair.replace('/', '_')}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_regime_probs(hmm, obs: np.ndarray) -> dict:
    if hmm is None or not hmm.is_trained:
        return {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34}
    recent = obs[-288:] if len(obs) > 288 else obs
    posteriors = hmm.model.predict_proba(recent)
    current = posteriors[-1]
    return {
        "bullish": float(current[hmm.bull_idx]),
        "bearish": float(current[hmm.bear_idx]),
        "neutral": float(current[hmm.neut_idx]),
    }


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_signal(row, hmm_probs, position, params):
    close       = row["close"]
    has_pos     = position["side"] == "long"
    entry_price = position.get("entry_price", close)

    # ── EXIT ──
    if has_pos:
        drawdown = (entry_price - close) / entry_price
        if drawdown >= FIXED_STOP_LOSS:
            return "SELL", "stop_loss"
        gain = (close - entry_price) / entry_price
        if gain >= FIXED_TAKE_PROFIT:
            return "SELL", "take_profit"

    # ── ENTRY ──
    if not has_pos:
        mom_ok  = row["momentum"]      > params["momentum_min"]
        ma_ok   = close                > row["ema20"]
        hmm_ok  = hmm_probs["bullish"] > params["hmm_confidence"]
        vol_ok  = row["rolling_vol"]   < params["max_volatility"]
        vz_ok   = abs(row["volume_zscore"]) < params["vol_zscore_max"]

        # Momentum + EMA20 mandatory; at least 2 of remaining 3 must pass
        if mom_ok and ma_ok:
            if sum([hmm_ok, vol_ok, vz_ok]) >= 2:
                return "BUY", "entry"

    return "HOLD", ""


# ═══════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════

def run_backtest(pair_data: dict, params: dict) -> dict:
    """
    Simulate trading on all pairs simultaneously.
    Returns final portfolio stats.
    """
    cash      = STARTING_BALANCE
    positions = {pair: {"side": "none", "entry_price": 0, "quantity": 0}
                 for pair in pair_data}
    trades    = []

    # Find common length (use shortest pair)
    min_len = min(len(v["df"]) for v in pair_data.values() if not v["df"].empty)

    for i in range(300, min_len):   # start at 300 to have enough HMM context
        for pair, v in pair_data.items():
            df      = v["df"]
            if df.empty or i >= len(df):
                continue

            row     = df.iloc[i]
            hmm     = v.get("hmm")
            obs_arr = v.get("obs")

            if hmm is None or obs_arr is None:
                continue

            obs_slice  = obs_arr[:i]
            hmm_probs  = get_regime_probs(hmm, obs_slice)
            pos        = positions[pair]
            signal, reason = generate_signal(row, hmm_probs, pos, params)

            if signal == "BUY" and pos["side"] == "none":
                position_usd = cash * params["position_pct"]
                if position_usd < 1.0 or cash < position_usd:
                    continue
                price    = row["close"] * (1 + COMMISSION)
                qty      = position_usd / price
                cash    -= position_usd
                positions[pair] = {"side": "long", "entry_price": price, "quantity": qty}
                trades.append({"pair": pair, "side": "BUY",  "price": price,
                               "qty": qty, "time": row["open_time"], "reason": reason})

            elif signal == "SELL" and pos["side"] == "long":
                price   = row["close"] * (1 - COMMISSION)
                qty     = pos["quantity"]
                cash   += qty * price
                pnl     = (price - pos["entry_price"]) * qty
                trades.append({"pair": pair, "side": "SELL", "price": price,
                               "qty": qty, "time": row["open_time"], "reason": reason,
                               "pnl": pnl})
                positions[pair] = {"side": "none", "entry_price": 0, "quantity": 0}

    # Mark-to-market open positions at last price
    portfolio_value = cash
    for pair, pos in positions.items():
        if pos["side"] == "long" and not pair_data[pair].get("df", pd.DataFrame()).empty:
            df    = pair_data[pair].get("df_full")
            if df is not None and not df.empty:
                last_price       = df["close"].iloc[-1]
                portfolio_value += pos["quantity"] * last_price * (1 - COMMISSION)

    wins   = [t for t in trades if t.get("side") == "SELL" and t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("side") == "SELL" and t.get("pnl", 0) <= 0]
    sells  = [t for t in trades if t.get("side") == "SELL"]
    total_pnl = sum(t.get("pnl", 0) for t in sells)

    return {
        "final_balance":  round(portfolio_value, 2),
        "total_return":   round((portfolio_value - STARTING_BALANCE) / STARTING_BALANCE * 100, 2),
        "n_trades":       len(sells),
        "win_rate":       round(len(wins) / len(sells) * 100, 1) if sells else 0,
        "total_pnl":      round(total_pnl, 2),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN OPTIMIZER
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  BACKTEST OPTIMIZER — Starting balance: $50,000")
    print("=" * 65)

    # Load data and HMMs
    print("\nLoading data and HMM models...")
    pair_data = {}
    for pair in PAIRS:
        df_raw = load_data(pair)
        if df_raw.empty:
            continue
        df = compute_features(df_raw)
        hmm = load_hmm(pair)

        feature_cols = ["log_return", "rolling_vol", "momentum", "volume_zscore"]
        obs = df[feature_cols].values

        # Split: 80% train/optimize, 20% out-of-sample test
        split = int(len(df) * 0.80)
        pair_data[pair] = {
            "df":      df.iloc[:split].reset_index(drop=True),
            "df_full": df,
            "obs":     obs[:split],
            "hmm":     hmm,
        }
        print(f"  {pair}: {len(df)} candles | train={split} | test={len(df)-split}")

    if not pair_data:
        print("No data loaded. Exiting.")
        return

    # Also add full obs for test phase
    for pair in list(pair_data.keys()):
        df_raw = load_data(pair)
        df_full = compute_features(df_raw)
        feature_cols = ["log_return", "rolling_vol", "momentum", "volume_zscore"]
        pair_data[pair]["obs_full"] = df_full[feature_cols].values

    # Random search
    print(f"\nRunning {N_RANDOM_TRIALS} random parameter trials on TRAIN data...\n")
    results = []

    for trial in range(N_RANDOM_TRIALS):
        params = {k: random.uniform(v[0], v[1]) for k, v in PARAM_SPACE.items()}
        stats  = run_backtest(pair_data, params)
        results.append({"params": params, **stats})

        if (trial + 1) % 50 == 0:
            best_so_far = max(results, key=lambda x: x["final_balance"])
            print(f"  Trial {trial+1:>3}/{N_RANDOM_TRIALS} | "
                  f"Best so far: ${best_so_far['final_balance']:>10,.2f} "
                  f"({best_so_far['total_return']:+.2f}%)")

    # Sort by final balance
    results.sort(key=lambda x: x["final_balance"], reverse=True)
    top5 = results[:5]

    # ── Print Top 5 ──
    print("\n" + "=" * 65)
    print("  TOP 5 PARAMETER SETS (on TRAIN data)")
    print("=" * 65)
    for rank, r in enumerate(top5, 1):
        p = r["params"]
        print(f"\n  #{rank} | Final: ${r['final_balance']:,.2f} | "
              f"Return: {r['total_return']:+.2f}% | "
              f"Trades: {r['n_trades']} | WinRate: {r['win_rate']:.1f}%")
        print(f"       HMM conf:     {p['hmm_confidence']:.3f}")
        print(f"       Max vol:      {p['max_volatility']:.3f}")
        print(f"       Momentum min: {p['momentum_min']:.4f}")
        print(f"       Vol z-max:    {p['vol_zscore_max']:.3f}")
        print(f"       Stop loss:    {FIXED_STOP_LOSS*100:.2f}% (fixed)")
        print(f"       Take profit:  {FIXED_TAKE_PROFIT*100:.2f}% (fixed)")
        print(f"       Position pct: {p['position_pct']*100:.1f}%")

    # ── Out-of-sample test with best params ──
    print("\n" + "=" * 65)
    print("  OUT-OF-SAMPLE TEST (last 20% of data) with best params")
    print("=" * 65)

    best_params = top5[0]["params"]

    # Rebuild pair_data for test period
    test_pair_data = {}
    for pair in PAIRS:
        df_raw = load_data(pair)
        if df_raw.empty:
            continue
        df = compute_features(df_raw)
        hmm = load_hmm(pair)
        feature_cols = ["log_return", "rolling_vol", "momentum", "volume_zscore"]
        obs = df[feature_cols].values
        split = int(len(df) * 0.80)
        test_pair_data[pair] = {
            "df":      df.iloc[split:].reset_index(drop=True),
            "df_full": df.iloc[split:].reset_index(drop=True),
            "obs":     obs[split:],
            "hmm":     hmm,
        }

    test_stats = run_backtest(test_pair_data, best_params)

    print(f"\n  Starting balance: ${STARTING_BALANCE:,.2f}")
    print(f"  Final balance:    ${test_stats['final_balance']:,.2f}")
    print(f"  Total return:     {test_stats['total_return']:+.2f}%")
    print(f"  Total trades:     {test_stats['n_trades']}")
    print(f"  Win rate:         {test_stats['win_rate']:.1f}%")
    print(f"  Total PnL:        ${test_stats['total_pnl']:,.2f}")

    print("\n" + "=" * 65)
    print("  RECOMMENDED CONFIG VALUES (copy to config.py)")
    print("=" * 65)
    p = best_params
    print(f"\n  ENTRY_HMM_CONFIDENCE    = {p['hmm_confidence']:.2f}")
    print(f"  ENTRY_MAX_VOLATILITY    = {p['max_volatility']:.2f}")
    print(f"  ENTRY_MOMENTUM_MIN      = {p['momentum_min']:.4f}")
    print(f"  ENTRY_VOLUME_ZSCORE_MAX = {p['vol_zscore_max']:.2f}")
    print(f"  EXIT_STOP_LOSS_PCT      = {FIXED_STOP_LOSS}  (fixed)")
    print(f"  EXIT_TAKE_PROFIT_PCT    = {FIXED_TAKE_PROFIT}  (fixed)")
    print(f"  MC_MAX_POSITION_PCT     = {p['position_pct']:.2f}")
    print()


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
