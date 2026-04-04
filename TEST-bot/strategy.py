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

def get_recent_support_zone(df: pd.DataFrame) -> dict | None:
    """
    Find the most recent ATR-significant structural swing low and return
    its CANDLE RANGE (low → high of that specific candle) as the support zone.

    This is the SMC definition: the demand zone is the exact candle that formed
    the last meaningful swing low — the last down-move before price turned up.
    The candle's HIGH is the upper boundary; price must be above it to be
    "out of the zone" and in a valid buying position.

    Returns:
        {
            "low":         candle low (bottom of demand zone)
            "high":        candle high (top of demand zone — key level)
            "mid":         midpoint
            "label":       HH/HL/LL/SL etc
            "significance": ATR significance score
            "index":       bar index in df
        }
        or None if no structural swing low found.
    """
    structure = build_market_structure(df)
    lows = [s for s in structure if s["type"] == "low"]
    if not lows:
        return None

    recent_low = lows[-1]   # most recently formed structural low
    idx        = recent_low["index"]

    if idx >= len(df):
        return None

    candle = df.iloc[idx]
    return {
        "low":          float(candle["low"]),
        "high":         float(candle["high"]),
        "mid":          (float(candle["low"]) + float(candle["high"])) / 2,
        "label":        recent_low["label"],
        "significance": recent_low["significance"],
        "index":        idx,
    }


def build_support_zones(swing_lows: list) -> list:
    """
    Cluster nearby swing lows into support zones, sorted by strength.
    {"low": float, "high": float, "mid": float, "strength": int}
    Kept for use by detect_liquidity_sweep (sweep still uses zone clusters).
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
# POI TOUCH DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_poi_touch(df: pd.DataFrame, poi_low: float, poi_high: float,
                     lookback: int = 30) -> dict:
    """
    Find the most recent candle within `lookback` bars where price
    visited (touched or entered) a Point of Interest zone.

    A touch = candle's low <= poi_high (price dipped into or through the zone).
    The close must NOT be far below poi_low (price shouldn't have crashed through).

    Returns:
        {"touched": True, "touch_idx": int, "touch_low": float}
        or {"touched": False}
    """
    end   = len(df) - 1
    start = max(0, end - lookback)

    for i in range(end, start - 1, -1):
        row = df.iloc[i]
        # Price entered the zone (low dipped at or below zone top)
        # but close must be at or above zone bottom — a close below means breakdown
        if row["low"] <= poi_high and row["close"] >= poi_low:
            return {
                "touched":     True,
                "touch_idx":   i,
                "touch_low":   float(row["low"]),
                "touch_close": float(row["close"]),
            }

    return {"touched": False}


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


def compute_confluences(df: pd.DataFrame, sweep_info: dict,
                        equil_level: float = None,
                        current_price: float = None,
                        atr_val: float = 0) -> tuple:
    """
    Weighted confluence scoring (max = 11, threshold = MIN_CONFLUENCES = 4).
      sweep detected:    +2  — smart money stop-hunt fingerprint
      vol spike on sweep:+1  — institutional volume confirmation
      equilibrium:       +2  — price at 50% of impulse leg (ideal retrace depth)
      fvg near price:    +2  — price inefficiency / imbalance at entry
      ema50:             +1  — 5m trend filter
      macd:              +1  — momentum filter
      rsi:               +1  — not overbought
    Sweep and equilibrium are bonuses, not hard gates.
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

    # ── Equilibrium ───────────────────────────────────────────────
    # 50% level of the impulse leg (zone low → BOS level).
    # Price near equilibrium means it retraced exactly halfway — the ideal
    # SMC entry: not too deep (structure broken), not too shallow (still risky).
    if equil_level is not None and current_price is not None:
        tolerance = atr_val if atr_val > 0 else current_price * 0.01
        dist      = abs(current_price - equil_level)
        if dist <= tolerance:
            score += 2
            reasons.append(
                f"[PASS +2] Equilibrium: price {current_price:.4f} "
                f"at/near 50% level {equil_level:.4f} (within {dist/equil_level*100:.2f}%)"
            )
        else:
            reasons.append(
                f"[    + 0] Equilibrium {equil_level:.4f} — "
                f"price {dist/equil_level*100:.2f}% away (>{tolerance/equil_level*100:.2f}% tolerance)"
            )

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

        atr_val = float(df["atr"].iloc[-1]) if "atr" in df.columns and df["atr"].iloc[-1] > 0 else 0

        # Gate 2 — Price must have TOUCHED a POI (support zone or FVG).
        # The sequence enforced here:
        #   price retraces INTO zone/FVG  →  candle touches it  →  THEN BOS from there
        # Both support zone and FVG are valid POIs. We find the best one,
        # confirm price actually visited it recently, then anchor BOS to that touch.

        zone    = get_recent_support_zone(df)
        fvgs    = _detect_fvg(df)
        fvg_poi = min(fvgs, key=lambda f: abs((f["bottom"] + f["top"]) / 2 - current_price)) if fvgs else None

        # Determine active POI (prefer support zone, fall back to FVG)
        poi_low = poi_high = None
        poi_label = ""
        if zone is not None:
            poi_low   = zone["low"]
            poi_high  = zone["high"]
            poi_label = f"support zone {zone['low']:.4f}-{zone['high']:.4f} [{zone['label']}]"
            if fvg_poi is not None:
                # If FVG overlaps or is closer, note it too
                fvg_mid = (fvg_poi["bottom"] + fvg_poi["top"]) / 2
                zone_mid = zone["mid"]
                if abs(fvg_mid - current_price) < abs(zone_mid - current_price):
                    poi_low   = min(zone["low"], fvg_poi["bottom"])
                    poi_high  = max(zone["high"], fvg_poi["top"])
                    poi_label += f" + FVG {fvg_poi['bottom']:.4f}-{fvg_poi['top']:.4f}"
        elif fvg_poi is not None:
            poi_low   = fvg_poi["bottom"]
            poi_high  = fvg_poi["top"]
            poi_label = f"FVG {fvg_poi['bottom']:.4f}-{fvg_poi['top']:.4f}"

        if poi_low is None:
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G2] No POI found — no structural support zone or nearby FVG",
                    f"  ->need a demand area for price to retrace into before BOS",
                ],
            }

        # Confirm price actually touched the POI recently (within 30 candles)
        touch = detect_poi_touch(df, poi_low, poi_high, lookback=30)
        if not touch["touched"]:
            dist_pct = (current_price - poi_high) / poi_high * 100 if current_price > poi_high else \
                       (poi_low - current_price) / poi_low * 100
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G2] POI exists ({poi_label}) but price hasn't touched it in last 30 candles",
                    f"  ->price is {dist_pct:.2f}% away — waiting for retrace into the zone",
                ],
            }

        reasons.append(
            f"[PASS G2] POI touched: {poi_label} | "
            f"touch low={touch['touch_low']:.4f} at bar {touch['touch_idx']}"
        )

        # Gate 3 — BOS anchored to the POI touch candle.
        # Only candles AFTER the touch count — BOS must come from the zone, not before it.
        # Also check for liquidity sweep at the touch (bonus, not required).
        swing_lows = detect_swing_lows(df)
        all_zones  = build_support_zones(swing_lows)
        sweep      = detect_liquidity_sweep(df, all_zones)

        # Anchor BOS to touch candle (sweep preferred if it aligns)
        if sweep["swept"] and sweep["candle_idx"] >= touch["touch_idx"]:
            bos = detect_bos(df, sweep["candle_idx"])
        else:
            sweep = {"swept": False}
            bos   = detect_bos(df, sweep_candle_idx=touch["touch_idx"])

        if not bos["bos"]:
            bos_level  = bos.get("bos_level", 0)
            gap_to_bos = (bos_level - current_price) / current_price * 100 if bos_level > current_price else 0
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G3] No BOS after POI touch — waiting for close above {bos_level:.4f} "
                    f"({gap_to_bos:.2f}% above current price)",
                    f"  ->price touched the zone but hasn't broken structure upward yet",
                ],
            }

        sweep_tag = " + sweep" if sweep["swept"] else ""
        reasons.append(
            f"[PASS G3] BOS{sweep_tag} after POI touch: "
            f"close {bos['bos_close']:.4f} > {bos['bos_level']:.4f}"
        )

        # Gate 4 — Confluences. Equilibrium = midpoint of POI low → BOS level.
        equil_level = (poi_low + bos["bos_level"]) / 2
        conf_score, conf_reasons = compute_confluences(df, sweep, equil_level, current_price, atr_val)
        reasons.extend(conf_reasons)
        if conf_score < MIN_CONFLUENCES:
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G4] Confluence score {conf_score}/{MIN_CONFLUENCES} — need ≥{MIN_CONFLUENCES} pts",
                ],
            }
        reasons.append(f"[PASS G4] Confluence score {conf_score} — all gates cleared")

        # ── SL / TP ────────────────────────────────────────────
        # SL anchored to deepest point: sweep low > touch low > poi low
        if sweep["swept"]:
            sl_anchor = sweep["sweep_low"]
        else:
            sl_anchor = min(touch["touch_low"], poi_low)
        sl_price = sl_anchor * (1 - SL_BUFFER_PCT)

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
