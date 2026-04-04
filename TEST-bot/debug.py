"""
SMC Debug Script
================
Prints the current signal state for each pair without placing any orders.
Useful for diagnosing which gate is failing and why.

Usage:
    python debug.py
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from data import load_or_fetch_historical, load_or_fetch_htf, compute_smc_indicators
from strategy import (
    SMCSignalGenerator,
    detect_swing_lows, build_support_zones,
    get_recent_support_zone,
    detect_liquidity_sweep, detect_bos,
    build_market_structure,
)
from config import PAIRS


def debug_pair(pair: str):
    print(f"\n{'='*60}")
    print(f"  {pair}")
    print(f"{'='*60}")

    # Load data
    df5 = load_or_fetch_historical(pair, force_refresh=False)
    if df5.empty:
        print("  ERROR: no 5m data")
        return
    df5 = compute_smc_indicators(df5)

    df1h = load_or_fetch_htf(pair, force_refresh=False)
    if not df1h.empty:
        df1h = compute_smc_indicators(df1h)

    last = df5.iloc[-1]
    current_price = float(last["close"])

    print(f"\n  5m candles loaded : {len(df5)}")
    print(f"  1H candles loaded : {len(df1h)}")
    print(f"  Current price     : {current_price:.6f}")
    print(f"  RSI               : {last.get('rsi', 'N/A'):.2f}" if 'rsi' in last.index else "  RSI: N/A")
    print(f"  MACD hist         : {last.get('macd_hist', 'N/A'):.6f}" if 'macd_hist' in last.index else "  MACD hist: N/A")
    print(f"  EMA50 (5m)        : {last.get('ema50', 'N/A'):.6f}" if 'ema50' in last.index else "  EMA50: N/A")
    print(f"  ATR               : {last.get('atr', 'N/A'):.6f}" if 'atr' in last.index else "  ATR: N/A")

    # HTF check
    if not df1h.empty:
        htf_last = df1h.iloc[-1]
        htf_pass = htf_last["close"] > htf_last.get("ema50", 0)
        print(f"\n  [Gate 1] HTF bias: 1H close={htf_last['close']:.4f}  "
              f"EMA50={htf_last.get('ema50', 0):.4f}  → {'PASS' if htf_pass else 'FAIL'}")
    else:
        print("\n  [Gate 1] HTF bias: no 1H data — gate SKIPPED")

    # Support zone (most recent structural low candle range)
    atr_val = float(df5["atr"].iloc[-1]) if "atr" in df5.columns else 0
    zone = get_recent_support_zone(df5)
    if zone:
        above    = current_price > zone["high"]
        dist_atr = (current_price - zone["high"]) / atr_val if atr_val > 0 else 0
        status   = "PASS" if above and dist_atr <= 1.5 else ("too far (>1.5x ATR)" if above else "below/inside zone")
        print(f"\n  [Gate 2] Recent support zone [{zone['label']}] sig={zone['significance']:.2f}")
        print(f"           zone:  {zone['low']:.6f} - {zone['high']:.6f}")
        print(f"           price: {current_price:.6f}  ({'above' if above else 'below/inside'}, {dist_atr:.2f}x ATR from ceiling)")
        print(f"           -> {status}")
    else:
        print(f"\n  [Gate 2] FAIL -- no structural swing low found")

    # Liquidity sweep
    sweep = detect_liquidity_sweep(df5, zones)
    if sweep["swept"]:
        print(f"\n  [Gate 3] LQ Sweep: PASS")
        print(f"           sweep_low={sweep['sweep_low']:.6f}  "
              f"wick_ratio={sweep['wick_ratio']:.3f}  "
              f"vol_spike={sweep['vol_spike']}")
        print(f"           zone: {sweep['zone']['low']:.6f}–{sweep['zone']['high']:.6f}")
    else:
        print(f"\n  [Gate 3] LQ Sweep: FAIL — no sweep in recent candles")

    # BOS
    if sweep["swept"]:
        bos = detect_bos(df5, sweep["candle_idx"], current_price)
        if bos["bos"]:
            live_tag = " (live)" if bos.get("live") else ""
            print(f"\n  [Gate 4] BOS: PASS{live_tag}")
            print(f"           bos_level={bos['bos_level']:.6f}  "
                  f"bos_close={bos['bos_close']:.6f}")
        else:
            print(f"\n  [Gate 4] BOS: FAIL — waiting for close above {bos.get('bos_level', '?'):.6f}")
    else:
        print(f"\n  [Gate 4] BOS: skipped (no sweep)")

    # Market structure
    structure = build_market_structure(df5)
    highs = [s for s in structure if s["type"] == "high"]
    lows  = [s for s in structure if s["type"] == "low"]
    print(f"\n  Market structure ({len(structure)} swings): "
          f"{len(highs)} highs, {len(lows)} lows")
    for s in structure[-6:]:
        print(f"    [{s['label']:<2}] {s['type']:<4}  price={s['price']:.6f}  "
              f"sig={s['significance']:.3f}")

    # Full signal
    no_position = {"side": "none", "entry_price": 0.0, "quantity": 0.0,
                   "sl_price": 0.0, "tp_price": 0.0, "entry_time": ""}
    gen    = SMCSignalGenerator()
    signal = gen.generate_signal(
        df5, current_price, 0.001, no_position,
        htf_df=df1h if not df1h.empty else None,
    )

    print(f"\n  FULL SIGNAL: {signal['action']}")
    for r in signal["reasons"]:
        print(f"    {r}")
    if signal["action"] == "BUY":
        print(f"\n  SL: {signal['sl_price']:.6f}  TP: {signal['tp_price']:.6f}  "
              f"Confluence: {signal.get('conf_score', '?')}/4")


def main():
    print("SMC DEBUG — live signal check (no orders placed)")
    for pair in PAIRS:
        debug_pair(pair)
    print()


if __name__ == "__main__":
    main()
