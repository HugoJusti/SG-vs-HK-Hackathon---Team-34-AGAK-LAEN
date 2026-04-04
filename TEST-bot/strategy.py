"""
Strategy module: HMM regime detection, Monte Carlo position sizing, signal generation.
"""
import os
import pickle
import numpy as np
import logging
from hmmlearn.hmm import GaussianHMM
from config import (
    HMM_N_STATES, HMM_MODEL_DIR,
    ENTRY_HMM_CONFIDENCE, ENTRY_MAX_VOLATILITY, ENTRY_MOMENTUM_MIN,
    ENTRY_VOLUME_ZSCORE_MAX, ENTRY_SPREAD_MAX_PCT,
    EXIT_STOP_LOSS_PCT, EXIT_TAKE_PROFIT_PCT,
    MC_NUM_SIMULATIONS, MC_HORIZON_HOURS, MC_MAX_POSITION_PCT,
    MC_MIN_POSITION_PCT, MC_TAIL_RISK_LIMIT
)

logger = logging.getLogger(__name__)


def _merge_signal_params(params: dict | None = None) -> dict:
    merged = {
        "entry_hmm_confidence": ENTRY_HMM_CONFIDENCE,
        "entry_max_volatility": ENTRY_MAX_VOLATILITY,
        "entry_momentum_min": ENTRY_MOMENTUM_MIN,
        "entry_volume_zscore_max": ENTRY_VOLUME_ZSCORE_MAX,
        "entry_spread_max_pct": ENTRY_SPREAD_MAX_PCT,
        "exit_stop_loss_pct": EXIT_STOP_LOSS_PCT,
        "exit_take_profit_pct": EXIT_TAKE_PROFIT_PCT,
    }
    if params:
        for key in list(merged.keys()):
            if key in params and params[key] is not None:
                merged[key] = params[key]
    return merged


def _merge_mc_params(params: dict | None = None) -> dict:
    merged = {
        "mc_num_simulations": MC_NUM_SIMULATIONS,
        "mc_horizon_hours": MC_HORIZON_HOURS,
        "mc_max_position_pct": MC_MAX_POSITION_PCT,
        "mc_min_position_pct": MC_MIN_POSITION_PCT,
        "mc_tail_risk_limit": MC_TAIL_RISK_LIMIT,
    }
    if params:
        for key in list(merged.keys()):
            if key in params and params[key] is not None:
                merged[key] = params[key]
    return merged


# ═══════════════════════════════════════════════════════════════
# HMM REGIME MODEL
# ═══════════════════════════════════════════════════════════════

class RegimeHMM:
    """
    3-state Gaussian HMM for market regime detection.

    States are labelled post-training based on mean returns:
      - Highest mean return → "bullish" (state index stored in self.bull_idx)
      - Lowest mean return  → "bearish" (state index stored in self.bear_idx)
      - Middle              → "neutral" (state index stored in self.neut_idx)
    """

    def __init__(self):
        self.model = None
        self.bull_idx = None
        self.bear_idx = None
        self.neut_idx = None
        self.is_trained = False

    def train(self, observations: np.ndarray, pair: str = ""):
        """Train HMM on observation matrix (N x 4 features)."""
        logger.info(f"Training HMM for {pair} on {len(observations)} observations...")

        self.model = GaussianHMM(
            n_components=HMM_N_STATES,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=0.01
        )

        self.model.fit(observations)

        # Label states by mean log return (feature index 0)
        means = self.model.means_[:, 0]
        sorted_indices = np.argsort(means)
        self.bear_idx = sorted_indices[0]
        self.neut_idx = sorted_indices[1]
        self.bull_idx = sorted_indices[2]

        self.is_trained = True
        logger.info(
            f"HMM trained for {pair}. "
            f"Bull state={self.bull_idx} (mean_ret={means[self.bull_idx]:.6f}), "
            f"Bear state={self.bear_idx} (mean_ret={means[self.bear_idx]:.6f}), "
            f"Neutral state={self.neut_idx} (mean_ret={means[self.neut_idx]:.6f})"
        )

    def get_regime_probabilities(self, observations: np.ndarray) -> dict:
        """
        Given recent observations, return current regime probabilities.
        Uses last 288 candles (24h of 5m data) for context.
        Returns: {"bullish": float, "bearish": float, "neutral": float}
        """
        if not self.is_trained:
            return {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34}

        recent = observations[-288:] if len(observations) > 288 else observations
        posteriors = self.model.predict_proba(recent)
        current = posteriors[-1]

        return {
            "bullish": float(current[self.bull_idx]),
            "bearish": float(current[self.bear_idx]),
            "neutral": float(current[self.neut_idx]),
        }

    def get_regime_params(self) -> dict:
        """Extract per-regime parameters for Monte Carlo simulation."""
        if not self.is_trained:
            return None

        return {
            "means": self.model.means_,
            "covars": self.model.covars_,
            "transmat": self.model.transmat_,
            "bull_idx": self.bull_idx,
            "bear_idx": self.bear_idx,
            "neut_idx": self.neut_idx,
        }

    def save(self, pair: str):
        os.makedirs(HMM_MODEL_DIR, exist_ok=True)
        path = os.path.join(HMM_MODEL_DIR, f"hmm_{pair.replace('/', '_')}.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"HMM model saved to {path}")

    def load(self, pair: str) -> bool:
        path = os.path.join(HMM_MODEL_DIR, f"hmm_{pair.replace('/', '_')}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            self.model = loaded.model
            self.bull_idx = loaded.bull_idx
            self.bear_idx = loaded.bear_idx
            self.neut_idx = loaded.neut_idx
            self.is_trained = loaded.is_trained
            logger.info(f"HMM model loaded from {path}")
            return True
        return False


# ═══════════════════════════════════════════════════════════════
# MONTE CARLO POSITION SIZING
# ═══════════════════════════════════════════════════════════════

def monte_carlo_position_size(regime_params: dict, current_regime_idx: int,
                              portfolio_value: float, params: dict | None = None) -> dict:
    """
    Run Monte Carlo simulation to determine position size.
    MC no longer gates trades — only sizes them.
    Returns minimum position (5%) if tail risk is too high, else maximum (30%).
    """
    mc_params = _merge_mc_params(params)

    if regime_params is None:
        return {
            "position_pct": mc_params["mc_min_position_pct"],
            "position_usd": portfolio_value * mc_params["mc_min_position_pct"],
            "paths_positive_pct": 0.5,
            "median_return": 0.0,
            "tail_risk_5pct": 0.0,
        }

    means = regime_params["means"][:, 0]
    stds = np.sqrt(regime_params["covars"][:, 0, 0])
    transmat = regime_params["transmat"]

    n_sims = int(mc_params["mc_num_simulations"])
    horizon = int(mc_params["mc_horizon_hours"])
    final_returns = np.zeros(n_sims)

    for sim in range(n_sims):
        cumulative_return = 0.0
        state = current_regime_idx

        for _ in range(horizon):
            ret = np.random.normal(means[state], stds[state])
            cumulative_return += ret
            state = np.random.choice(HMM_N_STATES, p=transmat[state])

        final_returns[sim] = cumulative_return

    paths_positive = np.mean(final_returns > 0)
    median_return = np.median(final_returns)
    tail_risk = np.percentile(final_returns, 5)

    # MC only sizes — does not gate trades
    if tail_risk < -mc_params["mc_tail_risk_limit"]:
        position_pct = mc_params["mc_min_position_pct"]
    else:
        position_pct = mc_params["mc_max_position_pct"]

    result = {
        "position_pct": round(position_pct, 4),
        "position_usd": round(portfolio_value * position_pct, 2),
        "paths_positive_pct": round(paths_positive, 4),
        "median_return": round(median_return, 6),
        "tail_risk_5pct": round(tail_risk, 6),
    }

    logger.info(
        f"MC result: {paths_positive*100:.1f}% paths positive, "
        f"median ret={median_return*100:.3f}%, "
        f"tail risk={tail_risk*100:.3f}%, "
        f"position={position_pct*100:.1f}%"
    )

    return result


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════

class SignalGenerator:
    """
    Combines HMM regime probabilities, feature filters, EMA confirmation,
    and spread filter to produce BUY/SELL/HOLD signals.
    """

    def generate_signal(self, regime_probs: dict, features: dict,
                        spread_pct: float, current_position: dict,
                        params: dict | None = None) -> dict:
        """
        Entry logic:
          - Mandatory: momentum > 0 AND price > EMA20
          - Plus at least 2 of 4: HMM bullish, vol filter, z-score, spread

        Exit logic:
          - Stop loss: drawdown >= 3%
          - Take profit: gain >= 3%
        """
        signal_params = _merge_signal_params(params)
        action = "HOLD"
        reasons = []

        close = features["close"]
        has_position = current_position.get("side") == "long"

        # ── EXIT LOGIC ──
        if has_position:
            entry_price = current_position.get("entry_price", close)
            drawdown = (entry_price - close) / entry_price

            if drawdown >= signal_params["exit_stop_loss_pct"]:
                return {
                    "action": "SELL",
                    "reasons": [
                        f"STOP LOSS: drawdown {drawdown*100:.2f}% >= "
                        f"{signal_params['exit_stop_loss_pct']*100}%"
                    ],
                    "confidence": 1.0
                }

            gain = (close - entry_price) / entry_price
            if gain >= signal_params["exit_take_profit_pct"]:
                return {
                    "action": "SELL",
                    "reasons": [
                        f"TAKE PROFIT: gain {gain*100:.2f}% >= "
                        f"{signal_params['exit_take_profit_pct']*100}%"
                    ],
                    "confidence": 1.0
                }

        # ── ENTRY LOGIC ──
        if not has_position:
            entry_checks = []

            # 1. HMM bullish
            hmm_ok = regime_probs["bullish"] > signal_params["entry_hmm_confidence"]
            entry_checks.append(("HMM bullish", hmm_ok,
                                 f"P(bull)={regime_probs['bullish']:.3f} vs {signal_params['entry_hmm_confidence']}"))

            # 2. Volatility filter
            vol_ok = features["rolling_vol"] < signal_params["entry_max_volatility"]
            entry_checks.append(("Vol filter", vol_ok,
                                 f"vol={features['rolling_vol']:.4f} vs {signal_params['entry_max_volatility']}"))

            # 3. Momentum filter (mandatory)
            mom_ok = features["momentum"] > signal_params["entry_momentum_min"]
            entry_checks.append(("Momentum", mom_ok,
                                 f"mom={features['momentum']:.6f} vs {signal_params['entry_momentum_min']}"))

            # 4. Volume z-score filter
            vz_ok = abs(features["volume_zscore"]) < signal_params["entry_volume_zscore_max"]
            entry_checks.append(("Vol z-score", vz_ok,
                                 f"|z|={abs(features['volume_zscore']):.3f} vs {signal_params['entry_volume_zscore_max']}"))

            # 5. EMA20 confirmation (mandatory)
            ma_ok = close > features["ma20"]
            entry_checks.append(("EMA20 confirm", ma_ok,
                                 f"close={close:.2f} vs EMA20={features['ma20']:.2f}"))

            # 6. Spread filter
            spread_ok = spread_pct < signal_params["entry_spread_max_pct"]
            entry_checks.append(("Spread filter", spread_ok,
                                 f"spread={spread_pct:.4f}% vs {signal_params['entry_spread_max_pct']}%"))

            # Momentum and EMA20 are mandatory; at least 2 of remaining 4 must pass
            if not mom_ok or not ma_ok:
                all_pass = 0
            else:
                remaining = [hmm_ok, vol_ok, vz_ok, spread_ok]
                all_pass = 1 if sum(remaining) >= 2 else 0

            action = "BUY" if all_pass else "HOLD"
            reasons = [f"{'[PASS]' if c[1] else '[FAIL]'} {c[0]}: {c[2]}" for c in entry_checks]

            return {
                "action": action,
                "reasons": reasons,
                "confidence": regime_probs.get("bullish", 0) if all_pass else 0.0
            }

        # Holding — no exit triggered
        return {
            "action": "HOLD",
            "reasons": ["Position held — no exit conditions met"],
            "confidence": regime_probs.get("bullish", 0.5)
        }
