"""
Strategy: Smart Money Concept (SMC) — v3

Entry requires ALL of:
  1. 1H HTF close above 1H EMA50          — macro bullish bias
  2. Support zone identified               — demand area from fractal swing lows
  3. Liquidity sweep                       — wick below zone, close back above
  4. Break of Structure (BOS)              — close above most recent swing high before sweep
  5. Weighted confluence score ≥ 4 (out of 7)

Exit:
  - Stop loss:   price ≤ sl_price  (starts at sweep low, trails up to each new HL)
  - Take profit: price ≥ tp_price  (nearest structural high above BOS level, ATR fallback)

New in v3:
  - build_market_structure()  ATR-filtered swing classification (HH/LH/HL/LL)
  - get_structural_tp()       nearest resistance high above BOS as TP target
  - get_trailing_sl()         trail SL up to each new confirmed HL after entry
"""

import numpy as np
import pandas as pd
import logging
from config import (
    SWING_LEFT_BARS, SWING_RIGHT_BARS, SWING_LOOKBACK, SWING_SIGNIFICANCE_ATR_MULT,
    ZONE_MARGIN_PCT,
    SWEEP_TOLERANCE_PCT, SWEEP_LOOKBACK, SWEEP_WICK_RATIO_MIN,
    BOS_LOOKBACK,
    RSI_OVERBOUGHT,
    FVG_LOOKBACK, MIN_CONFLUENCES, CONFLUENCE_WEIGHTS,
    ENTRY_SPREAD_MAX_PCT,
    RISK_PER_TRADE_PCT, MAX_POSITION_PCT, MIN_POSITION_PCT,
    SL_BUFFER_PCT, TP_MIN_PCT, TP_MAX_PCT,
    VOLUME_SPIKE_MULT,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# MARKET STRUCTURE
# ═══════════════════════════════════════════════════════════════

def build_market_structure(df: pd.DataFrame, lookback: int = None) -> list:
    """
    Build an ATR-filtered, classified list of structural swing highs and lows.

    Each swing is checked for significance:
      significance = (pivot - mean_of_neighbours) / ATR
      Only included if significance ≥ SWING_SIGNIFICANCE_ATR_MULT (0.3)

    Each swing is labelled relative to the prior swing of the same type:
      Highs: HH (higher high) | LH (lower high) | SH (first/starting high)
      Lows:  HL (higher low)  | LL (lower low)  | SL (first/starting low)

    Returns list of swings in chronological order:
    [{"type":"high"|"low", "price":float, "index":int,
      "significance":float, "label":"HH"|"LH"|"HL"|"LL"|"SH"|"SL"}]
    """
    lb    = lookback or SWING_LOOKBACK
    start = max(0, len(df) - lb - SWING_RIGHT_BARS)
    highs = df["high"].values
    lows  = df["low"].values
    atrs  = df["atr"].values if "atr" in df.columns else np.full(len(df), 1e-6)

    swings = []

    for i in range(start + SWING_LEFT_BARS, len(df) - SWING_RIGHT_BARS):
        atr_val = float(atrs[i]) if atrs[i] > 1e-10 else 1e-6

        # ── Swing high ───────────────────────────────────────────
        pivot_h = highs[i]
        if all(highs[i - j] < pivot_h for j in range(1, SWING_LEFT_BARS + 1)) and \
           all(highs[i + j] < pivot_h for j in range(1, SWING_RIGHT_BARS + 1)):
            neighbours = np.concatenate([
                highs[max(0, i - SWING_LEFT_BARS): i],
                highs[i + 1: i + SWING_RIGHT_BARS + 1],
            ])
            sig = (pivot_h - float(np.mean(neighbours))) / atr_val
            if sig >= SWING_SIGNIFICANCE_ATR_MULT:
                swings.append({
                    "type":         "high",
                    "price":        float(pivot_h),
                    "index":        i,
                    "significance": round(sig, 3),
                    "label":        None,
                })

        # ── Swing low ────────────────────────────────────────────
        pivot_l = lows[i]
        if all(lows[i - j] > pivot_l for j in range(1, SWING_LEFT_BARS + 1)) and \
           all(lows[i + j] > pivot_l for j in range(1, SWING_RIGHT_BARS + 1)):
            neighbours = np.concatenate([
                lows[max(0, i - SWING_LEFT_BARS): i],
                lows[i + 1: i + SWING_RIGHT_BARS + 1],
            ])
            sig = (float(np.mean(neighbours)) - pivot_l) / atr_val
            if sig >= SWING_SIGNIFICANCE_ATR_MULT:
                swings.append({
                    "type":         "low",
                    "price":        float(pivot_l),
                    "index":        i,
                    "significance": round(sig, 3),
                    "label":        None,
                })

    # Sort chronologically, then label HH/LH/HL/LL
    swings.sort(key=lambda s: s["index"])
    last_high_price = None
    last_low_price  = None

    for s in swings:
        if s["type"] == "high":
            if last_high_price is None:
                s["label"] = "SH"
            elif s["price"] > last_high_price:
                s["label"] = "HH"
            else:
                s["label"] = "LH"
            last_high_price = s["price"]
        else:
            if last_low_price is None:
                s["label"] = "SL"
            elif s["price"] > last_low_price:
                s["label"] = "HL"
            else:
                s["label"] = "LL"
            last_low_price = s["price"]

    return swings


def get_structural_tp(structure: list, bos_level: float) -> float | None:
    """
    Return the nearest structural high above the BOS level.
    This is the next resistance the price is likely to react at after BOS.
    Returns None if no structural high exists above the BOS level.
    """
    candidates = [
        s for s in structure
        if s["type"] == "high" and s["price"] > bos_level
    ]
    if not candidates:
        return None
    return float(min(candidates, key=lambda s: s["price"])["price"])


def get_trailing_sl(df: pd.DataFrame, entry_time: str,
                    current_sl: float) -> float:
    """
    Scan for confirmed swing lows formed AFTER entry_time.
    Returns the highest qualifying SL candidate (swing_low × (1 - SL_BUFFER_PCT)).
    Never returns lower than current_sl — SL only moves up.

    Uses the same ATR significance filter as build_market_structure to avoid
    trailing to noise lows.
    """
    if not entry_time or "open_time" not in df.columns:
        return current_sl

    entry_dt = pd.Timestamp(entry_time)
    post_df  = df[df["open_time"] >= entry_dt].reset_index(drop=True)

    min_len = SWING_LEFT_BARS + SWING_RIGHT_BARS + 1
    if len(post_df) < min_len:
        return current_sl

    lows = post_df["low"].values
    atrs = post_df["atr"].values if "atr" in post_df.columns else np.full(len(post_df), 1e-6)
    best_sl = current_sl

    for i in range(SWING_LEFT_BARS, len(post_df) - SWING_RIGHT_BARS):
        pivot = lows[i]

        # Fractal check
        if not (all(lows[i - j] > pivot for j in range(1, SWING_LEFT_BARS + 1)) and
                all(lows[i + j] > pivot for j in range(1, SWING_RIGHT_BARS + 1))):
            continue

        # ATR significance filter — ignore noise lows
        atr_val = float(atrs[i]) if atrs[i] > 1e-10 else 1e-6
        neighbours = np.concatenate([
            lows[max(0, i - SWING_LEFT_BARS): i],
            lows[i + 1: i + SWING_RIGHT_BARS + 1],
        ])
        sig = (float(np.mean(neighbours)) - pivot) / atr_val
        if sig < SWING_SIGNIFICANCE_ATR_MULT:
            continue

        candidate = pivot * (1 - SL_BUFFER_PCT)
        if candidate > best_sl:
            best_sl = candidate
            logger.debug(
                "Trailing SL candidate: %.6f (pivot=%.6f, sig=%.3f)", candidate, pivot, sig
            )

    return round(best_sl, 6)


# ═══════════════════════════════════════════════════════════════
# LEGACY SWING HELPERS (used internally by detect_bos)
# ═══════════════════════════════════════════════════════════════

def detect_swing_lows(df: pd.DataFrame) -> list:
    """
    Fractal swing low detection for support zone building.
    Returns list of {"index": int, "price": float}.
    """
    start = max(0, len(df) - SWING_LOOKBACK - SWING_RIGHT_BARS)
    lows  = df["low"].values
    result = []

    for i in range(start + SWING_LEFT_BARS, len(df) - SWING_RIGHT_BARS):
        pivot = lows[i]
        if all(lows[i - j] > pivot for j in range(1, SWING_LEFT_BARS + 1)) and \
           all(lows[i + j] > pivot for j in range(1, SWING_RIGHT_BARS + 1)):
            result.append({"index": i, "price": float(pivot)})

    return result


def _most_recent_swing_high(df: pd.DataFrame) -> float | None:
    """
    Walk backwards and return the most recently confirmed fractal swing high.
    Used by detect_bos as the BOS level.
    """
    highs = df["high"].values
    n     = len(highs)

    for i in range(n - SWING_RIGHT_BARS - 1, SWING_LEFT_BARS - 1, -1):
        pivot = highs[i]
        if all(highs[i - j] < pivot for j in range(1, SWING_LEFT_BARS + 1)) and \
           all(highs[i + j] < pivot for j in range(1, SWING_RIGHT_BARS + 1)):
            return float(pivot)

    return None


# ═══════════════════════════════════════════════════════════════
# SUPPORT ZONES
# ═══════════════════════════════════════════════════════════════

def build_support_zones(swing_lows: list) -> list:
    """
    Cluster nearby swing lows into support zones, sorted by strength.
    {"low": float, "high": float, "mid": float, "strength": int}
    """
    if not swing_lows:
        return []

    prices = sorted(sl["price"] for sl in swing_lows)
    zones  = []
    i      = 0

    while i < len(prices):
        cluster = [prices[i]]
        j = i + 1
        while j < len(prices) and \
              (prices[j] - prices[i]) / prices[i] <= ZONE_MARGIN_PCT * 2:
            cluster.append(prices[j])
            j += 1

        zones.append({
            "low":      float(min(cluster)),
            "high":     float(max(cluster) * (1 + ZONE_MARGIN_PCT)),
            "mid":      float(np.mean(cluster)),
            "strength": len(cluster),
        })
        i = j

    return sorted(zones, key=lambda z: z["strength"], reverse=True)


# ═══════════════════════════════════════════════════════════════
# LIQUIDITY SWEEP
# ═══════════════════════════════════════════════════════════════

def detect_liquidity_sweep(df: pd.DataFrame, zones: list) -> dict:
    """
    Check the last SWEEP_LOOKBACK completed candles for a sweep of any support zone.
    Returns the most recent valid sweep or {"swept": False}.
    """
    if not zones:
        return {"swept": False}

    end   = len(df) - 1
    start = max(0, end - SWEEP_LOOKBACK)
    sub   = df.iloc[start:end]

    if sub.empty:
        return {"swept": False}

    opens  = sub["open"].values
    highs  = sub["high"].values
    lows   = sub["low"].values
    closes = sub["close"].values
    vols   = sub["volume"].values
    vol_ma = sub["vol_ma"].values if "vol_ma" in sub.columns else None

    for i in range(len(sub) - 1, -1, -1):
        candle_range = highs[i] - lows[i]
        if candle_range < 1e-10:
            continue

        lower_wick = min(opens[i], closes[i]) - lows[i]
        wick_ratio = lower_wick / candle_range

        for zone in zones:
            wick_below  = lows[i]   < zone["low"] * (1 - SWEEP_TOLERANCE_PCT)
            close_above = closes[i] > zone["low"]
            wick_ok     = wick_ratio >= SWEEP_WICK_RATIO_MIN

            if wick_below and close_above and wick_ok:
                vol_spike = True
                if vol_ma is not None and vol_ma[i] > 0:
                    vol_spike = vols[i] >= vol_ma[i] * VOLUME_SPIKE_MULT

                return {
                    "swept":       True,
                    "sweep_low":   float(lows[i]),
                    "sweep_close": float(closes[i]),
                    "zone":        zone,
                    "candle_idx":  start + i,
                    "vol_spike":   vol_spike,
                    "wick_ratio":  round(wick_ratio, 3),
                }

    return {"swept": False}


# ═══════════════════════════════════════════════════════════════
# BREAK OF STRUCTURE
# ═══════════════════════════════════════════════════════════════

def detect_bos(df: pd.DataFrame, sweep_candle_idx: int = None) -> dict:
    """
    BOS: a candle closes above the most recent confirmed swing high.

    sweep_candle_idx (optional):
      When a sweep was detected, anchor the swing-high search to the
      BOS_LOOKBACK candles before that sweep candle.
      When None (no sweep), search the last BOS_LOOKBACK candles of df.

    Live price check requires BOS_LIVE_BUFFER above the level to avoid wick entries.
    """
    if sweep_candle_idx is not None:
        pre_start = max(0, sweep_candle_idx - BOS_LOOKBACK)
        pre_df    = df.iloc[pre_start: sweep_candle_idx + 1]
        post_df   = df.iloc[sweep_candle_idx + 1:]
    else:
        # Standalone mode: find swing high in last BOS_LOOKBACK bars,
        # then check if any of the last 5 closed candles broke it.
        pre_df  = df.iloc[-BOS_LOOKBACK:] if len(df) >= BOS_LOOKBACK else df
        post_df = df.iloc[-5:]

    if pre_df.empty:
        return {"bos": False}

    bos_level = _most_recent_swing_high(pre_df)
    if bos_level is None:
        bos_level = float(pre_df["close"].max())

    for i, row in enumerate(post_df.itertuples()):
        if row.close > bos_level:
            return {
                "bos":            True,
                "bos_level":      round(bos_level, 6),
                "bos_candle_idx": (sweep_candle_idx + 1 + i) if sweep_candle_idx is not None else len(df) - 5 + i,
                "bos_close":      round(float(row.close), 6),
            }

    # No live price entry — only confirmed candle closes count
    return {"bos": False, "bos_level": round(bos_level, 6)}


# ═══════════════════════════════════════════════════════════════
# CONFLUENCE SCORING
# ═══════════════════════════════════════════════════════════════

def _detect_fvg(df: pd.DataFrame) -> list:
    """
    Bullish FVG: candle[i-2].high < candle[i].low.
    Only counted if the gap midpoint is within 2×ATR of the current close —
    a distant FVG from hours ago at a different price level is not confluent.
    """
    recent        = df.iloc[-FVG_LOOKBACK:] if len(df) >= FVG_LOOKBACK else df
    highs         = recent["high"].values
    lows          = recent["low"].values
    current_close = float(df["close"].iloc[-1])
    atr_val       = float(df["atr"].iloc[-1]) if "atr" in df.columns and df["atr"].iloc[-1] > 0 else 0
    proximity     = 2.0 * atr_val if atr_val > 0 else current_close * 0.02

    fvgs = []
    for i in range(2, len(recent)):
        if highs[i - 2] < lows[i]:
            bottom = float(highs[i - 2])
            top    = float(lows[i])
            mid    = (bottom + top) / 2
            if abs(mid - current_close) <= proximity:
                fvgs.append({"bottom": bottom, "top": top})
    return fvgs


def compute_confluences(df: pd.DataFrame, sweep_info: dict) -> tuple:
    """
    Weighted confluence scoring (max = 9, threshold = MIN_CONFLUENCES = 4).
      sweep (detected):  +2  — smart money stop-hunt fingerprint
      vol_spike on sweep:+1  — institutional volume confirmation
      fvg:               +2  — price inefficiency near entry
      ema50:             +1  — 5m trend filter
      macd:              +1  — momentum filter
      rsi:               +1  — not overbought
    Sweep is now scored, not a hard gate. No sweep = -3 pts but entry still possible.
    Returns (score: int, reasons: list[str]).
    """
    reasons = []
    score   = 0
    last    = df.iloc[-1]
    w       = CONFLUENCE_WEIGHTS

    # ── Sweep (optional bonus) ────────────────────────────────────
    if sweep_info.get("swept"):
        score += w["vol_spike"]   # reuse vol_spike weight (2) for sweep presence
        reasons.append(
            f"[PASS +{w['vol_spike']}] LQ sweep detected "
            f"(low={sweep_info['sweep_low']:.4f}, wick={sweep_info['wick_ratio']:.2f})"
        )
        if sweep_info.get("vol_spike"):
            score += 1
            reasons.append(f"[PASS +1] Volume spike on sweep candle")
        else:
            reasons.append(f"[    + 0] No volume spike on sweep candle")
    else:
        reasons.append(f"[    + 0] No liquidity sweep — entry on BOS alone (weaker)")

    # ── RSI ───────────────────────────────────────────────────────
    if "rsi" in df.columns:
        rsi_val = float(last["rsi"])
        if rsi_val < RSI_OVERBOUGHT:
            score += w["rsi"]
            reasons.append(f"[PASS +{w['rsi']}] RSI={rsi_val:.1f} < {RSI_OVERBOUGHT}")
        else:
            reasons.append(f"[FAIL  0] RSI={rsi_val:.1f} >= {RSI_OVERBOUGHT} (overbought)")

    # ── MACD ──────────────────────────────────────────────────────
    if "macd" in df.columns and "macd_signal" in df.columns:
        if last["macd"] > last["macd_signal"]:
            score += w["macd"]
            reasons.append(f"[PASS +{w['macd']}] MACD bullish")
        else:
            reasons.append(f"[FAIL  0] MACD bearish")

    # ── EMA50 ─────────────────────────────────────────────────────
    if "ema50" in df.columns:
        if last["close"] > last["ema50"]:
            score += w["ema50"]
            reasons.append(f"[PASS +{w['ema50']}] Price {last['close']:.4f} > EMA50 {last['ema50']:.4f}")
        else:
            reasons.append(f"[FAIL  0] Price {last['close']:.4f} <= EMA50 {last['ema50']:.4f}")

    # ── FVG ───────────────────────────────────────────────────────
    fvgs = _detect_fvg(df)
    if fvgs:
        score += w["fvg"]
        reasons.append(f"[PASS +{w['fvg']}] {len(fvgs)} FVG(s) near price in last {FVG_LOOKBACK} candles")
    else:
        reasons.append(f"[FAIL  0] No nearby FVG in last {FVG_LOOKBACK} candles")

    return score, reasons


# ═══════════════════════════════════════════════════════════════
# POSITION SIZING
# ═══════════════════════════════════════════════════════════════

def calculate_position_size(portfolio_value: float, entry_price: float,
                            sl_price: float) -> dict:
    """
    Risk-based sizing: risk RISK_PER_TRADE_PCT of portfolio.
      position_usd = risk_amount / sl_distance_pct
    Clamped to [MIN_POSITION_PCT, MAX_POSITION_PCT].
    """
    risk_amount = portfolio_value * RISK_PER_TRADE_PCT
    sl_distance = entry_price - sl_price

    if sl_distance > 0:
        sl_dist_pct  = sl_distance / entry_price
        position_usd = risk_amount / sl_dist_pct
    else:
        position_usd = portfolio_value * MIN_POSITION_PCT
        sl_dist_pct  = 0.0

    min_usd      = portfolio_value * MIN_POSITION_PCT
    max_usd      = portfolio_value * MAX_POSITION_PCT
    position_usd = max(min_usd, min(max_usd, position_usd))

    return {
        "position_usd":    round(position_usd, 2),
        "position_pct":    round(position_usd / portfolio_value, 4),
        "risk_amount":     round(risk_amount, 2),
        "sl_distance_pct": round(sl_dist_pct * 100, 4),
    }


# ═══════════════════════════════════════════════════════════════
# SMC SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════

class SMCSignalGenerator:
    """
    Entry:  HTF gate → support zone → LQ sweep → BOS → confluence ≥ 4
    TP:     nearest structural high above BOS level (ATR-based fallback)
    SL:     below sweep low; trailed up to each new confirmed HL post-entry
    """

    def generate_signal(self, df: pd.DataFrame, current_price: float,
                        spread_pct: float, current_position: dict,
                        htf_df: pd.DataFrame = None) -> dict:

        has_position = current_position.get("side") == "long"

        # ── EXIT LOGIC ─────────────────────────────────────────
        if has_position:
            entry = current_position.get("entry_price", current_price)
            sl    = current_position.get("sl_price", 0.0)
            tp    = current_position.get("tp_price", 0.0)

            if sl > 0 and current_price <= sl:
                loss_pct = (entry - current_price) / entry * 100
                return {
                    "action":   "SELL",
                    "reasons":  [f"STOP LOSS: {current_price:.4f} <= SL {sl:.4f} (-{loss_pct:.2f}%)"],
                    "sl_price": sl,
                    "tp_price": tp,
                }

            if tp > 0 and current_price >= tp:
                gain_pct = (current_price - entry) / entry * 100
                return {
                    "action":   "SELL",
                    "reasons":  [f"TAKE PROFIT: {current_price:.4f} >= TP {tp:.4f} (+{gain_pct:.2f}%)"],
                    "sl_price": sl,
                    "tp_price": tp,
                }

            unrealised_pct  = (current_price - entry) / entry * 100
            unrealised_usd  = (current_price - entry) * current_position.get("quantity", 0)
            sl_dist_pct     = (current_price - sl) / current_price * 100 if sl > 0 else 0
            tp_dist_pct     = (tp - current_price) / current_price * 100 if tp > 0 else 0
            return {
                "action":   "HOLD",
                "reasons":  [
                    f"Holding long | entry={entry:.4f} | now={current_price:.4f}",
                    f"  PnL:  {unrealised_pct:+.2f}%  (${unrealised_usd:+.2f})",
                    f"  SL:   {sl:.4f}  ({sl_dist_pct:.2f}% below price)",
                    f"  TP:   {tp:.4f}  ({tp_dist_pct:.2f}% above price)",
                ],
                "sl_price": sl,
                "tp_price": tp,
            }

        # ── ENTRY LOGIC ────────────────────────────────────────
        reasons = []

        if spread_pct > ENTRY_SPREAD_MAX_PCT:
            return {
                "action":  "HOLD",
                "reasons": [
                    f"[FAIL G0] Spread too wide: {spread_pct:.4f}% > max {ENTRY_SPREAD_MAX_PCT}%",
                    f"  ->skipping pair until spread narrows",
                ],
            }

        # Gate 1 — HTF bias
        if htf_df is not None and not htf_df.empty:
            htf_last = htf_df.iloc[-1]
            if "ema50" in htf_df.columns and htf_last["close"] <= htf_last["ema50"]:
                gap_pct = (htf_last["ema50"] - htf_last["close"]) / htf_last["ema50"] * 100
                return {
                    "action":  "HOLD",
                    "reasons": [
                        f"[FAIL G1] HTF bearish: 1H close {htf_last['close']:.4f} "
                        f"<= 1H EMA50 {htf_last['ema50']:.4f}  ({gap_pct:.2f}% below)",
                        f"  ->all 5m setups blocked until 1H closes above EMA50",
                    ],
                }
            reasons.append(
                f"[PASS G1] HTF bullish: 1H close {htf_last['close']:.4f} "
                f"> 1H EMA50 {htf_last['ema50']:.4f}"
            )

        # Gate 2 — Support zone within proximity of current price
        swing_lows = detect_swing_lows(df)
        zones      = build_support_zones(swing_lows)
        # Only zones within 5% of current price are relevant demand areas
        nearby_zones = [z for z in zones if abs(z["mid"] - current_price) / current_price <= 0.05]
        if not nearby_zones:
            all_zone_info = (f"{len(zones)} zone(s) found but all >5% away from price"
                             if zones else "no zones found at all")
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G2] No support zone within 5% of current price — {all_zone_info}",
                    f"  ->price not near a demand area, no setup context",
                ],
            }

        nearest = min(nearby_zones, key=lambda z: abs(z["mid"] - current_price))
        reasons.append(
            f"[PASS G2] {len(nearby_zones)} nearby zone(s) | "
            f"nearest: {nearest['low']:.4f}-{nearest['high']:.4f} "
            f"(strength={nearest['strength']}, "
            f"{abs(nearest['mid'] - current_price) / current_price * 100:.2f}% from price)"
        )

        # Gate 3 — Break of Structure (hard gate, sweep optional)
        # Try sweep-anchored BOS first; fall back to standalone BOS
        sweep = detect_liquidity_sweep(df, nearby_zones)
        if sweep["swept"]:
            bos = detect_bos(df, sweep["candle_idx"])
        else:
            sweep = {"swept": False}
            bos   = detect_bos(df, sweep_candle_idx=None)

        if not bos["bos"]:
            bos_level  = bos.get("bos_level", 0)
            gap_to_bos = (bos_level - current_price) / current_price * 100 if bos_level > current_price else 0
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G3] Awaiting BOS above {bos_level:.4f}  "
                    f"(price {current_price:.4f} is {gap_to_bos:.2f}% below)",
                    f"  ->need a candle close above that level (or live +0.1% buffer)",
                ],
            }

        sweep_tag = " + sweep" if sweep["swept"] else " (no sweep)"
        live_tag  = " live" if bos.get("live") else ""
        reasons.append(
            f"[PASS G3] BOS{sweep_tag}{live_tag}: "
            f"close {bos['bos_close']:.4f} > level {bos['bos_level']:.4f}"
        )

        # Gate 4 — Weighted confluences (sweep adds bonus points, not required)
        conf_score, conf_reasons = compute_confluences(df, sweep)
        reasons.extend(conf_reasons)
        if conf_score < MIN_CONFLUENCES:
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G4] Confluence score {conf_score}/{MIN_CONFLUENCES} — need ≥{MIN_CONFLUENCES} pts",
                    f"  ->missing enough confirmation to enter",
                ],
            }
        reasons.append(f"[PASS G4] Confluence score {conf_score} — all gates cleared")

        # ── SL / TP ────────────────────────────────────────────
        # SL: below sweep low if sweep present, else below nearest zone bottom
        if sweep["swept"]:
            sl_price = sweep["sweep_low"] * (1 - SL_BUFFER_PCT)
        else:
            sl_price = nearest["low"] * (1 - SL_BUFFER_PCT)

        # Build market structure for structural TP
        structure     = build_market_structure(df)
        structural_tp = get_structural_tp(structure, bos["bos_level"])

        if structural_tp is not None:
            # Clamp to minimum TP floor
            tp_price = max(structural_tp, current_price * (1 + TP_MIN_PCT))
            reasons.append(
                f"TP=structural high {structural_tp:.4f} | "
                f"SL={sl_price:.4f}"
            )
        else:
            # ATR fallback when no structural high is found above BOS
            atr      = float(df["atr"].iloc[-1]) if "atr" in df.columns and df["atr"].iloc[-1] > 0 else 0
            tp_raw   = (current_price + 2.0 * atr) if atr > 0 else current_price * (1 + TP_MAX_PCT)
            tp_price = float(np.clip(tp_raw,
                                     current_price * (1 + TP_MIN_PCT),
                                     current_price * (1 + TP_MAX_PCT)))
            reasons.append(
                f"TP=ATR fallback {tp_price:.4f} | SL={sl_price:.4f}"
            )

        return {
            "action":         "BUY",
            "reasons":        reasons,
            "sl_price":       round(sl_price, 6),
            "tp_price":       round(tp_price, 6),
            "sweep":          sweep,
            "bos":            bos,
            "conf_score":     conf_score,
            "structure_size": len(structure),
        }
