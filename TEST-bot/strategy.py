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
    BOS_LOOKBACK, BOS_LIVE_BUFFER,
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

def detect_bos(df: pd.DataFrame, sweep_candle_idx: int,
               current_price: float = None) -> dict:
    """
    BOS: a candle closes above the most recent confirmed swing high
    in the BOS_LOOKBACK candles before the sweep.

    Live price check requires BOS_LIVE_BUFFER above the level to avoid wick entries.
    """
    pre_start = max(0, sweep_candle_idx - BOS_LOOKBACK)
    pre_df    = df.iloc[pre_start: sweep_candle_idx + 1]

    if pre_df.empty:
        return {"bos": False}

    bos_level = _most_recent_swing_high(pre_df)
    if bos_level is None:
        bos_level = float(pre_df["close"].max())

    post_df = df.iloc[sweep_candle_idx + 1:]
    for i, row in enumerate(post_df.itertuples()):
        if row.close > bos_level:
            return {
                "bos":            True,
                "bos_level":      round(bos_level, 6),
                "bos_candle_idx": sweep_candle_idx + 1 + i,
                "bos_close":      round(float(row.close), 6),
            }

    if current_price is not None and \
       current_price > bos_level * (1 + BOS_LIVE_BUFFER):
        return {
            "bos":            True,
            "bos_level":      round(bos_level, 6),
            "bos_candle_idx": len(df),
            "bos_close":      round(current_price, 6),
            "live":           True,
        }

    return {"bos": False, "bos_level": round(bos_level, 6)}


# ═══════════════════════════════════════════════════════════════
# CONFLUENCE SCORING
# ═══════════════════════════════════════════════════════════════

def _detect_fvg(df: pd.DataFrame) -> list:
    """Bullish FVG: candle[i-2].high < candle[i].low."""
    recent = df.iloc[-FVG_LOOKBACK:] if len(df) >= FVG_LOOKBACK else df
    highs  = recent["high"].values
    lows   = recent["low"].values
    fvgs   = []
    for i in range(2, len(recent)):
        if highs[i - 2] < lows[i]:
            fvgs.append({"bottom": float(highs[i - 2]), "top": float(lows[i])})
    return fvgs


def compute_confluences(df: pd.DataFrame, sweep_info: dict) -> tuple:
    """
    Weighted confluence scoring (max = 7, threshold = MIN_CONFLUENCES = 4).
      vol_spike=2, fvg=2, ema50=1, macd=1, rsi=1
    Returns (score: int, reasons: list[str]).
    """
    reasons = []
    score   = 0
    last    = df.iloc[-1]
    w       = CONFLUENCE_WEIGHTS

    if "rsi" in df.columns:
        rsi_val = float(last["rsi"])
        if rsi_val < RSI_OVERBOUGHT:
            score += w["rsi"]
            reasons.append(f"[PASS +{w['rsi']}] RSI={rsi_val:.1f} < {RSI_OVERBOUGHT}")
        else:
            reasons.append(f"[FAIL  0] RSI={rsi_val:.1f} >= {RSI_OVERBOUGHT} (overbought)")

    if "macd" in df.columns and "macd_signal" in df.columns:
        if last["macd"] > last["macd_signal"]:
            score += w["macd"]
            reasons.append(f"[PASS +{w['macd']}] MACD bullish")
        else:
            reasons.append(f"[FAIL  0] MACD bearish")

    if "ema50" in df.columns:
        if last["close"] > last["ema50"]:
            score += w["ema50"]
            reasons.append(f"[PASS +{w['ema50']}] Price {last['close']:.4f} > EMA50 {last['ema50']:.4f}")
        else:
            reasons.append(f"[FAIL  0] Price {last['close']:.4f} <= EMA50 {last['ema50']:.4f}")

    fvgs = _detect_fvg(df)
    if fvgs:
        score += w["fvg"]
        reasons.append(f"[PASS +{w['fvg']}] {len(fvgs)} FVG(s) in last {FVG_LOOKBACK} candles")
    else:
        reasons.append(f"[FAIL  0] No FVG in last {FVG_LOOKBACK} candles")

    if sweep_info.get("vol_spike"):
        score += w["vol_spike"]
        reasons.append(f"[PASS +{w['vol_spike']}] Volume spike on sweep candle")
    else:
        reasons.append(f"[FAIL  0] No volume spike on sweep candle")

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

            return {
                "action":   "HOLD",
                "reasons":  [f"Holding | SL={sl:.4f} | TP={tp:.4f} | Now={current_price:.4f}"],
                "sl_price": sl,
                "tp_price": tp,
            }

        # ── ENTRY LOGIC ────────────────────────────────────────
        reasons = []

        if spread_pct > ENTRY_SPREAD_MAX_PCT:
            return {
                "action":  "HOLD",
                "reasons": [f"[FAIL] Spread {spread_pct:.4f}% > {ENTRY_SPREAD_MAX_PCT}%"],
            }

        # Gate 1 — HTF bias
        if htf_df is not None and not htf_df.empty:
            htf_last = htf_df.iloc[-1]
            if "ema50" in htf_df.columns and htf_last["close"] <= htf_last["ema50"]:
                return {
                    "action":  "HOLD",
                    "reasons": [
                        f"[FAIL] HTF bearish: 1H close {htf_last['close']:.4f} "
                        f"<= 1H EMA50 {htf_last['ema50']:.4f}"
                    ],
                }
            reasons.append(
                f"[PASS] HTF bullish: 1H close {htf_last['close']:.4f} "
                f"> 1H EMA50 {htf_last['ema50']:.4f}"
            )

        # Gate 2 — Support zones
        swing_lows = detect_swing_lows(df)
        zones      = build_support_zones(swing_lows)
        if not zones:
            return {"action": "HOLD", "reasons": reasons + ["[FAIL] No support zones found"]}

        reasons.append(
            f"[PASS] {len(zones)} zone(s) | "
            f"strongest: {zones[0]['low']:.4f}–{zones[0]['high']:.4f} "
            f"(strength={zones[0]['strength']})"
        )

        # Gate 3 — Liquidity sweep
        sweep = detect_liquidity_sweep(df, zones)
        if not sweep["swept"]:
            return {
                "action":  "HOLD",
                "reasons": reasons + ["[FAIL] No liquidity sweep in recent candles"],
            }

        reasons.append(
            f"[PASS] LQ Sweep {sweep['sweep_low']:.4f} | "
            f"zone={sweep['zone']['low']:.4f}–{sweep['zone']['high']:.4f} | "
            f"wick={sweep['wick_ratio']:.2f}"
        )

        # Gate 4 — Break of Structure
        bos = detect_bos(df, sweep["candle_idx"], current_price)
        if not bos["bos"]:
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL] Awaiting BOS above {bos.get('bos_level', '?'):.4f}"
                ],
            }

        live_tag = " (live +buffer)" if bos.get("live") else ""
        reasons.append(
            f"[PASS] BOS{live_tag}: {bos['bos_close']:.4f} > {bos['bos_level']:.4f}"
        )

        # Gate 5 — Weighted confluences
        conf_score, conf_reasons = compute_confluences(df, sweep)
        reasons.extend(conf_reasons)
        if conf_score < MIN_CONFLUENCES:
            return {
                "action":  "HOLD",
                "reasons": reasons + [
                    f"[FAIL] Confluence {conf_score}/{MIN_CONFLUENCES}"
                ],
            }
        reasons.append(f"[PASS] Confluence score {conf_score}/{MIN_CONFLUENCES}")

        # ── SL / TP ────────────────────────────────────────────
        sl_price = sweep["sweep_low"] * (1 - SL_BUFFER_PCT)

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
