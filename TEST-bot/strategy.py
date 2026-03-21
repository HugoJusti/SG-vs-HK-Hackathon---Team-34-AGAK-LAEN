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
    ENTRY_VOLUME_ZSCORE_MAX, ENTRY_MA_SHORT, ENTRY_SPREAD_MAX_PCT,
    EXIT_STOP_LOSS_PCT, EXIT_TAKE_PROFIT_PCT,
    MC_NUM_SIMULATIONS, MC_HORIZON_HOURS, MC_MAX_POSITION_PCT,
    MC_MIN_POSITION_PCT, MC_TAIL_RISK_LIMIT
)

logger = logging.getLogger(__name__)


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
        """
        Train HMM on observation matrix (N x 4 features).
        """
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
        means = self.model.means_[:, 0]  # log_return means per state
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
        Returns: {"bullish": float, "bearish": float, "neutral": float}
        """
        if not self.is_trained:
            return {"bullish": 0.33, "bearish": 0.33, "neutral": 0.34}

        # Use last 100 observations for context
        recent = observations[-288:] if len(observations) > 288 else observations
        posteriors = self.model.predict_proba(recent)
        current = posteriors[-1]  # last time step

        return {
            "bullish": float(current[self.bull_idx]),
            "bearish": float(current[self.bear_idx]),
            "neutral": float(current[self.neut_idx]),
        }

    def get_regime_params(self) -> dict:
        """
        Extract per-regime parameters for Monte Carlo simulation.
        Returns mean and covariance for each regime, plus transition matrix.
        """
        if not self.is_trained:
            return None

        return {
            "means": self.model.means_,         # (3, 4) per-state feature means
            "covars": self.model.covars_,        # (3, 4, 4) per-state covariances
            "transmat": self.model.transmat_,    # (3, 3) transition probabilities
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
                              portfolio_value: float) -> dict:
    """
    Run Monte Carlo simulation to determine position size.
    
    Simulates MC_NUM_SIMULATIONS forward paths of MC_HORIZON_HOURS length,
    starting from current_regime_idx, using regime-specific return distributions
    and transition probabilities.
    
    Returns:
        {
            "position_pct": float,     # recommended % of portfolio
            "position_usd": float,     # dollar amount
            "paths_positive_pct": float,
            "median_return": float,
            "tail_risk_5pct": float,   # 5th percentile return
        }
    """
    if regime_params is None:
        return {
            "position_pct": MC_MIN_POSITION_PCT,
            "position_usd": portfolio_value * MC_MIN_POSITION_PCT,
            "paths_positive_pct": 0.5,
            "median_return": 0.0,
            "tail_risk_5pct": 0.0,
        }

    means = regime_params["means"][:, 0]       # log return means per regime
    stds = np.sqrt(regime_params["covars"][:, 0, 0])  # log return stds per regime
    transmat = regime_params["transmat"]

    n_sims = MC_NUM_SIMULATIONS
    horizon = MC_HORIZON_HOURS
    final_returns = np.zeros(n_sims)

    for sim in range(n_sims):
        cumulative_return = 0.0
        state = current_regime_idx

        for _ in range(horizon):
            # Draw return from current regime's distribution
            ret = np.random.normal(means[state], stds[state])
            cumulative_return += ret

            # Transition to next state
            state = np.random.choice(HMM_N_STATES, p=transmat[state])

        final_returns[sim] = cumulative_return

    # Analyse simulation results
    paths_positive = np.mean(final_returns > 0)
    median_return = np.median(final_returns)
    tail_risk = np.percentile(final_returns, 5)

    # Determine position size — MC no longer gates trades, only sizes them
    if tail_risk < -MC_TAIL_RISK_LIMIT:
        # Tail risk too high, use minimum
        position_pct = MC_MIN_POSITION_PCT
    else:
        # Use maximum position — entry confluences already confirmed the trade
        position_pct = MC_MAX_POSITION_PCT

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
    Combines HMM regime probabilities, feature filters, MA confirmation,
    and spread filter to produce BUY/SELL/HOLD signals.
    """

    def generate_signal(self, regime_probs: dict, features: dict,
                        spread_pct: float, current_position: dict) -> dict:
        """
        Args:
            regime_probs: {"bullish": float, "bearish": float, "neutral": float}
            features: dict with log_return, rolling_vol, momentum, volume_zscore, ma20, ma50, close
            spread_pct: current bid-ask spread as percentage
            current_position: {"side": "long"/"none", "entry_price": float, "pair": str}
        
        Returns:
            {"action": "BUY"/"SELL"/"HOLD", "reasons": [str], "confidence": float}
        """
        action = "HOLD"
        reasons = []

        close = features["close"]
        has_position = current_position.get("side") == "long"

        # ── EXIT LOGIC (check first — protecting capital is priority) ──
        if has_position:
            entry_price = current_position.get("entry_price", close)
            drawdown = (entry_price - close) / entry_price

            # Hard stop loss
            if drawdown >= EXIT_STOP_LOSS_PCT:
                return {
                    "action": "SELL",
                    "reasons": [f"STOP LOSS: drawdown {drawdown*100:.2f}% >= {EXIT_STOP_LOSS_PCT*100}%"],
                    "confidence": 1.0
                }

            # Take profit
            gain = (close - entry_price) / entry_price
            if gain >= EXIT_TAKE_PROFIT_PCT:
                return {
                    "action": "SELL",
                    "reasons": [f"TAKE PROFIT: gain {gain*100:.2f}% >= {EXIT_TAKE_PROFIT_PCT*100}%"],
                    "confidence": 1.0
                }

            # MA20 and HMM bearish exits disabled — relying on stop loss and take profit only

        # ── ENTRY LOGIC (all conditions must pass) ──
        if not has_position:
            entry_checks = []

            # 1. HMM bullish
            hmm_ok = regime_probs["bullish"] > ENTRY_HMM_CONFIDENCE
            entry_checks.append(("HMM bullish", hmm_ok,
                                 f"P(bull)={regime_probs['bullish']:.3f} vs {ENTRY_HMM_CONFIDENCE}"))

            # 2. Volatility filter
            vol_ok = features["rolling_vol"] < ENTRY_MAX_VOLATILITY
            entry_checks.append(("Vol filter", vol_ok,
                                 f"vol={features['rolling_vol']:.4f} vs {ENTRY_MAX_VOLATILITY}"))

            # 3. Momentum filter
            mom_ok = features["momentum"] > ENTRY_MOMENTUM_MIN
            entry_checks.append(("Momentum", mom_ok,
                                 f"mom={features['momentum']:.6f} vs {ENTRY_MOMENTUM_MIN}"))

            # 4. Volume z-score filter
            vz_ok = abs(features["volume_zscore"]) < ENTRY_VOLUME_ZSCORE_MAX
            entry_checks.append(("Vol z-score", vz_ok,
                                 f"|z|={abs(features['volume_zscore']):.3f} vs {ENTRY_VOLUME_ZSCORE_MAX}"))

            # 5. MA20 confirmation
            ma_ok = close > features["ma20"]
            entry_checks.append(("MA20 confirm", ma_ok,
                                 f"close={close:.2f} vs MA20={features['ma20']:.2f}"))

            # 6. Spread filter (Option B — not in HMM, just a gate)
            spread_ok = spread_pct < ENTRY_SPREAD_MAX_PCT
            entry_checks.append(("Spread filter", spread_ok,
                                 f"spread={spread_pct:.4f}% vs {ENTRY_SPREAD_MAX_PCT}%"))
            # Momentum and MA20 are mandatory
            if not mom_ok or not ma_ok:
                all_pass = 0
            else:
                # At least 2 of the remaining 4 must pass
                remaining = [hmm_ok, vol_ok, vz_ok, spread_ok]
                all_pass = 1 if sum(remaining) >= 2 else 0

            if all_pass:
                action = "BUY"
            else:
                action = "HOLD"
            reasons = [f"{'[PASS]' if c[1] else '[FAIL]'} {c[0]}: {c[2]}" for c in entry_checks]

            return {
                "action": action,
                "reasons": reasons,
                "confidence": regime_probs.get("bullish", 0) if all_pass else 0.0
            }

        # Holding position, no exit triggered
        return {
            "action": "HOLD",
            "reasons": ["Position held — no exit conditions met"],
            "confidence": regime_probs.get("bullish", 0.5)
        }