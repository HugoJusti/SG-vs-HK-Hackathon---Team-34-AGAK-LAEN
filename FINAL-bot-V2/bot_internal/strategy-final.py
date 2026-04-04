from __future__ import annotations
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
    BOS_LOOKBACK, BOS_CLOSE_BUFFER_PCT, BOS_MAX_AGE_CANDLES,
    POI_TOUCH_LOOKBACK,
    RSI_OVERBOUGHT,
    MIN_CONFLUENCES, CONFLUENCE_WEIGHTS,
    ENTRY_SPREAD_MAX_PCT,
    RISK_PER_TRADE_PCT, MAX_POSITION_PCT, MIN_POSITION_PCT,
    SL_BUFFER_PCT, TP_MIN_PCT, TP_MAX_PCT, MIN_RR_RATIO,
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


def get_structural_tp(structure: list, bos_level: float,
                      entry_price: float = None, sl_price: float = None,
                      min_rr: float = None) -> float | None:
    """
    Return the lowest structural high above BOS that also satisfies the
    minimum R:R ratio (if entry_price, sl_price, and min_rr are provided).

    Walks candidates in ascending order — skips any high that would give
    R:R < min_rr, so the nearest *valid* target is returned, not just the nearest.
    Returns None if no candidate clears both the level and the R:R requirement.
    """
    candidates = sorted(
        [s for s in structure if s["type"] == "high" and s["price"] > bos_level],
        key=lambda s: s["price"],
    )
    if not candidates:
        return None

    # No R:R filter requested — return nearest
    if entry_price is None or sl_price is None or min_rr is None:
        return float(candidates[0]["price"])

    sl_dist = entry_price - sl_price
    if sl_dist <= 0:
        return float(candidates[0]["price"])

    for c in candidates:
        rr = (c["price"] - entry_price) / sl_dist
        if rr >= min_rr:
            return float(c["price"])

    return None  # every structural high is too close relative to the SL


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


def detect_choch(df: pd.DataFrame, disrespect_idx: int) -> dict:
    """
    Change of Character (CHoCH): after a zone is disrespected (close below its floor),
    track the resulting selling move and detect when price closes back above the HIGH
    of that sell candle (+ BOS_CLOSE_BUFFER_PCT).

    When CHoCH fires, the LOW of the selling move becomes a new candidate demand zone.

    Args:
        df:               full 5m dataframe
        disrespect_idx:   bar index where the zone-breaking close occurred

    Returns:
        {
            "choch":        True/False
            "sell_low":     lowest low of the selling move (new zone bottom)
            "sell_high":    high of the selling move candle (CHoCH level)
            "choch_close":  close that confirmed CHoCH
            "new_zone":     {"low": sell_low, "high": sell_high, "mid": ..., "label": "CHoCH"}
        }
        or {"choch": False, "sell_low": float, "sell_high": float}
    """
    if disrespect_idx >= len(df) - 1:
        return {"choch": False}

    sell_segment = df.iloc[disrespect_idx:]
    if sell_segment.empty:
        return {"choch": False}

    # Find the lowest low and its candle high during the selling move
    sell_low_idx  = int(sell_segment["low"].idxmin())
    sell_low      = float(sell_segment["low"].min())
    sell_high     = float(df.iloc[sell_low_idx]["high"])   # high of the lowest candle
    choch_level   = sell_high * (1 + BOS_CLOSE_BUFFER_PCT)

    # Scan candles after the sell low for a close above that level
    post_sell = df.iloc[sell_low_idx + 1:]
    for i, row in post_sell.iterrows():
        if float(row["close"]) >= choch_level:
            new_zone = {
                "low":   round(sell_low, 6),
                "high":  round(sell_high, 6),
                "mid":   round((sell_low + sell_high) / 2, 6),
                "label": "CHoCH",
            }
            return {
                "choch":       True,
                "sell_low":    round(sell_low, 6),
                "sell_high":   round(sell_high, 6),
                "choch_close": round(float(row["close"]), 6),
                "choch_idx":   i,
                "new_zone":    new_zone,
            }

    return {
        "choch":      False,
        "sell_low":   round(sell_low, 6),
        "sell_high":  round(sell_high, 6),
    }


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
        # then check if any of the last 20 closed candles broke it.
        pre_df  = df.iloc[-BOS_LOOKBACK:] if len(df) >= BOS_LOOKBACK else df
        post_df = df.iloc[-20:]

    if pre_df.empty:
        return {"bos": False}

    bos_level = _most_recent_swing_high(pre_df)
    if bos_level is None:
        bos_level = float(pre_df["close"].max())

    required_close = bos_level * (1 + BOS_CLOSE_BUFFER_PCT)

    for i, row in enumerate(post_df.itertuples()):
        if row.close >= required_close:
            return {
                "bos":            True,
                "bos_level":      round(bos_level, 6),
                "bos_candle_idx": (sweep_candle_idx + 1 + i) if sweep_candle_idx is not None else len(df) - 5 + i,
                "bos_close":      round(float(row.close), 6),
                "bos_buffer_pct": BOS_CLOSE_BUFFER_PCT * 100,
            }

    # No confirmed close sufficiently above level
    return {"bos": False, "bos_level": round(bos_level, 6),
            "required_close": round(required_close, 6)}


# ═══════════════════════════════════════════════════════════════
# CONFLUENCE SCORING
# ═══════════════════════════════════════════════════════════════

def compute_confluences(df: pd.DataFrame, sweep_info: dict,
                        poi_low: float = None, bos_level: float = None,
                        current_price: float = None) -> tuple:
    """
    Weighted confluence scoring (max = 8, threshold = MIN_CONFLUENCES = 3).
      sweep detected:    +2  — smart money stop-hunt fingerprint
      vol spike on sweep:+1  — institutional volume confirmation
      equilibrium:       +2  — price in 40-60% band of POI-low to BOS impulse
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

    # ── Equilibrium ───────────────────────────────────────────────
    # Price is in the true 40–60% band of the impulse leg (poi_low → bos_level).
    # 40% level = poi_low + 0.40 * impulse
    # 60% level = poi_low + 0.60 * impulse
    # Midpoint (50%) = equilibrium level.
    if poi_low is not None and bos_level is not None and current_price is not None:
        impulse    = bos_level - poi_low
        if impulse > 1e-10:
            eq_low   = poi_low + 0.40 * impulse
            eq_high  = poi_low + 0.60 * impulse
            equil_mid = poi_low + 0.50 * impulse
            if eq_low <= current_price <= eq_high:
                score += 2
                pct_from = (current_price - equil_mid) / equil_mid * 100
                reasons.append(
                    f"[PASS +2] Equilibrium: price {current_price:.4f} "
                    f"in 40-60% band [{eq_low:.4f}-{eq_high:.4f}] "
                    f"({pct_from:+.2f}% from 50% level)"
                )
            else:
                side = "above" if current_price > eq_high else "below"
                reasons.append(
                    f"[    + 0] Equilibrium band [{eq_low:.4f}-{eq_high:.4f}] — "
                    f"price {current_price:.4f} is {side} the discount zone"
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

    actual_risk_usd = position_usd * sl_dist_pct
    actual_risk_pct = actual_risk_usd / portfolio_value if portfolio_value > 0 else 0.0

    return {
        "position_usd":    round(position_usd, 2),
        "position_pct":    round(position_usd / portfolio_value, 4),
        "risk_amount":     round(risk_amount, 2),
        "sl_distance_pct": round(sl_dist_pct * 100, 4),
        "actual_risk_pct": round(actual_risk_pct * 100, 4),
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

        # Gate 2 — Price must have TOUCHED a POI (structural support zone).
        # The sequence enforced here:
        #   price retraces INTO zone  ->  candle touches it  ->  THEN BOS from there

        zone = get_recent_support_zone(df)
        if zone is None:
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    "[FAIL G2] No structural support zone found",
                    "  ->need an ATR-significant swing low to define demand area",
                ],
            }
        poi_low   = zone["low"]
        poi_high  = zone["high"]
        poi_label = f"support zone {zone['low']:.4f}-{zone['high']:.4f} [{zone['label']}]"

        # Confirm price actually touched the POI recently
        touch = detect_poi_touch(df, poi_low, poi_high, lookback=POI_TOUCH_LOOKBACK)
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

        # Zone disrespect check — scan every candle from touch to now.
        # If any CLOSE is below poi_low the zone was broken, not just wicked through.
        post_touch = df.iloc[touch["touch_idx"]:]
        disrespect_candle = None
        for _, c in post_touch.iterrows():
            if float(c["close"]) < poi_low:
                disrespect_candle = c
                break

        if disrespect_candle is not None:
            # Zone broken — check for CHoCH (Change of Character).
            # If the selling move that followed already flipped back up,
            # the sell low becomes a new demand zone and we re-evaluate from there.
            try:
                disrespect_idx = df.index.get_loc(disrespect_candle.name)
            except Exception:
                disrespect_idx = touch["touch_idx"]
            choch = detect_choch(df, disrespect_idx)

            if choch["choch"]:
                # Promote the CHoCH zone as the new POI and continue evaluation
                poi_low   = choch["new_zone"]["low"]
                poi_high  = choch["new_zone"]["high"]
                poi_label = (
                    f"CHoCH zone {poi_low:.4f}-{poi_high:.4f} "
                    f"(sell low after disrespect of original zone)"
                )
                reasons.append(
                    f"[INFO] Original zone disrespected at {disrespect_candle['close']:.4f}, "
                    f"but CHoCH confirmed at {choch['choch_close']:.4f} — "
                    f"new demand zone set at {poi_low:.4f}-{poi_high:.4f}"
                )
                # Re-anchor touch to CHoCH candle so BOS is searched from there
                touch = {
                    "touched":     True,
                    "touch_idx":   choch["choch_idx"],
                    "touch_low":   choch["sell_low"],
                    "touch_close": choch["choch_close"],
                }
            else:
                return {
                    "action":  "HOLD",
                    "reasons": reasons + [
                        f"[FAIL G2] Zone disrespected — closed at "
                        f"{disrespect_candle['close']:.4f} below zone floor {poi_low:.4f}",
                        f"  ->watching for CHoCH above {choch.get('sell_high', 0):.4f} "
                        f"to form new demand zone at sell low {choch.get('sell_low', 0):.4f}",
                    ],
                }

        reasons.append(
            f"[PASS G2] POI touched & held: {poi_label} | "
            f"touch low={touch['touch_low']:.4f} at bar {touch['touch_idx']}"
        )

        # Gate 3 — BOS anchored to the POI touch candle.
        # Only candles AFTER the touch count — BOS must come from the zone, not before it.
        # Sweep check uses the SAME G2 zone (not a separate clustered zone set).
        g2_zone_list = [{"low": poi_low, "high": poi_high,
                         "mid": zone["mid"], "strength": 1}]
        sweep = detect_liquidity_sweep(df, g2_zone_list)

        # Anchor BOS to touch candle (sweep preferred if it aligns with touch)
        if sweep["swept"] and sweep["candle_idx"] >= touch["touch_idx"]:
            bos = detect_bos(df, sweep["candle_idx"])
        else:
            sweep = {"swept": False}
            bos   = detect_bos(df, sweep_candle_idx=touch["touch_idx"])

        if not bos["bos"]:
            bos_level_val  = bos.get("bos_level", 0)
            required_close = bos.get("required_close", bos_level_val)
            gap_to_bos     = (required_close - current_price) / current_price * 100 if required_close > current_price else 0
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G3] No BOS after POI touch — need close >= {required_close:.4f} "
                    f"(level {bos_level_val:.4f} + {BOS_CLOSE_BUFFER_PCT*100:.1f}% buffer, "
                    f"{gap_to_bos:.2f}% above current price)",
                    f"  ->price touched the zone but hasn't closed convincingly above structure",
                ],
            }

        # Staleness check — BOS that fired long ago means we'd be entering well into the move
        bos_age = len(df) - 1 - bos["bos_candle_idx"]
        if bos_age > BOS_MAX_AGE_CANDLES:
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL G3] BOS is stale — fired {bos_age} candles ago "
                    f"(max {BOS_MAX_AGE_CANDLES}). Entry price too far from BOS level.",
                    f"  ->waiting for a fresh BOS closer to current price",
                ],
            }

        sweep_tag = " + sweep" if sweep["swept"] else ""
        reasons.append(
            f"[PASS G3] BOS{sweep_tag} after POI touch: "
            f"close {bos['bos_close']:.4f} > {bos['bos_level']:.4f} "
            f"({bos_age} candle(s) ago)"
        )

        # Gate 4 — Confluences using proper 40-60% equilibrium band of the impulse.
        conf_score, conf_reasons = compute_confluences(
            df, sweep,
            poi_low=poi_low, bos_level=bos["bos_level"],
            current_price=current_price,
        )
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
        sl_dist  = current_price - sl_price

        # Build market structure for structural TP — skip any high too close for MIN_RR
        structure     = build_market_structure(df)
        structural_tp = get_structural_tp(
            structure, bos["bos_level"],
            entry_price=current_price, sl_price=sl_price, min_rr=MIN_RR_RATIO,
        )

        if structural_tp is not None:
            tp_price  = max(structural_tp, current_price * (1 + TP_MIN_PCT))
            rr_actual = (tp_price - current_price) / sl_dist if sl_dist > 0 else 0
            reasons.append(
                f"TP=structural high {structural_tp:.4f} | "
                f"SL={sl_price:.4f} | R:R={rr_actual:.2f}R"
            )
        else:
            # ATR fallback: target at least MIN_RR_RATIO × sl_dist, then clamp
            atr         = float(df["atr"].iloc[-1]) if "atr" in df.columns and df["atr"].iloc[-1] > 0 else 0
            tp_from_rr  = current_price + MIN_RR_RATIO * sl_dist
            tp_from_atr = (current_price + 2.0 * atr) if atr > 0 else tp_from_rr
            tp_raw      = max(tp_from_rr, tp_from_atr)
            tp_price    = float(np.clip(tp_raw,
                                        current_price * (1 + TP_MIN_PCT),
                                        current_price * (1 + TP_MAX_PCT)))
            rr_actual   = (tp_price - current_price) / sl_dist if sl_dist > 0 else 0

            # If TP_MAX_PCT cap prevents reaching MIN_RR, the trade has bad R:R — skip
            if rr_actual < MIN_RR_RATIO:
                min_tp_needed = current_price + MIN_RR_RATIO * sl_dist
                return {
                    "action":  "HOLD",
                    "reasons": reasons + [
                        f"[FAIL RR] R:R={rr_actual:.2f}R < min {MIN_RR_RATIO}R — "
                        f"SL too wide for available TP targets",
                        f"  ->SL={sl_price:.4f} ({sl_dist/current_price*100:.2f}% risk) | "
                        f"need TP >= {min_tp_needed:.4f} but max cap is "
                        f"{current_price * (1 + TP_MAX_PCT):.4f}",
                    ],
                }

            reasons.append(
                f"TP=ATR fallback {tp_price:.4f} | "
                f"SL={sl_price:.4f} | R:R={rr_actual:.2f}R"
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
