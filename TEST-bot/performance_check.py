"""
Offline performance evaluator for the TEST-bot strategy.

This script reuses the current strategy thresholds from config.py, runs an
offline simulation on recent Binance 5-minute candles, and reports:

- Pure return metrics
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Composite score = 0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar
- Heuristic entry / exit ratings for later ML work

Usage:
    python performance_check.py
    python performance_check.py --weeks 2
    python performance_check.py --weeks 4 --starting-balance 50000
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass, asdict, field
from typing import Dict, List

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    PAIRS,
    ENTRY_HMM_CONFIDENCE,
    ENTRY_MAX_VOLATILITY,
    ENTRY_MOMENTUM_MIN,
    ENTRY_VOLUME_ZSCORE_MAX,
    ENTRY_SPREAD_MAX_PCT,
    EXIT_STOP_LOSS_PCT,
    EXIT_TAKE_PROFIT_PCT,
    HMM_TRAINING_HOURS,
    LOG_DIR,
    MC_NUM_SIMULATIONS,
    MC_HORIZON_HOURS,
    MC_MAX_POSITION_PCT,
    MC_MIN_POSITION_PCT,
    MC_TAIL_RISK_LIMIT,
)
from data import compute_features, load_or_fetch_historical, pair_to_binance_symbol
from strategy import RegimeHMM, SignalGenerator, monte_carlo_position_size

warnings.filterwarnings("ignore", message="Model is not converging.*")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BARS_PER_HOUR = 12
BARS_PER_DAY = 24 * BARS_PER_HOUR
BARS_PER_WEEK = 7 * BARS_PER_DAY
PERIODS_PER_YEAR = 365 * 24 * BARS_PER_HOUR
DEFAULT_STARTING_BALANCE = 50_000.0
DEFAULT_COMMISSION = 0.0005
DEFAULT_ORDER_LOG_JSONL = os.path.join(SCRIPT_DIR, LOG_DIR, "performance_orders.jsonl")
DEFAULT_ORDER_LOG_CSV = os.path.join(SCRIPT_DIR, LOG_DIR, "performance_orders.csv")


@dataclass
class Position:
    side: str = "none"
    entry_price: float = 0.0
    quantity: float = 0.0
    entry_index: int = -1
    entry_time: str = ""
    entry_reasons: List[str] = field(default_factory=list)
    entry_confidence: float = 0.0
    entry_regime: str = "neutral"
    mc_position_pct: float = 0.0
    mc_position_usd: float = 0.0


def progress(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def print_order_event(event: Dict[str, object], enabled: bool = True) -> None:
    if not enabled:
        return

    if event["event"] == "BUY":
        progress(
            "[order] "
            f"{event['event']} #{event['order_no']} | {event['pair']} | {event['timestamp']} | "
            f"px=${event['price']:,.4f} | qty={event['quantity']:.8f} | "
            f"usd=${event['notional_usd']:,.2f} | regime={event['regime']} | "
            f"conf={event['signal_confidence']:.3f}"
        )
        return

    progress(
        "[order] "
        f"{event['event']} #{event['order_no']} | {event['pair']} | {event['timestamp']} | "
        f"px=${event['price']:,.4f} | pnl=${event['pnl']:,.2f} | "
        f"ret={event['return_pct']:.4f}% | hold={event['holding_hours']:.2f}h | "
        f"reason={event['reason']}"
    )


def runtime_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TEST-bot return performance.")
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="Recent history window to evaluate.",
    )
    parser.add_argument(
        "--starting-balance",
        type=float,
        default=DEFAULT_STARTING_BALANCE,
        help="Starting USD capital for the simulation.",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=DEFAULT_COMMISSION,
        help="Per-side execution cost. Default is 0.0005 (0.05%%).",
    )
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional pair list such as BTC BTC/USD ETH/USD.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh cached Binance candles before running if network is available.",
    )
    parser.add_argument(
        "--output-json",
        default=runtime_path(os.path.join(LOG_DIR, "performance_report.json")),
        help="Where to save the JSON report.",
    )
    parser.add_argument(
        "--output-csv",
        default=runtime_path(os.path.join(LOG_DIR, "performance_trades.csv")),
        help="Where to save the closed trade log CSV.",
    )
    parser.add_argument(
        "--output-orders-jsonl",
        default=DEFAULT_ORDER_LOG_JSONL,
        help="Where to save the simulated order event JSONL log.",
    )
    parser.add_argument(
        "--output-orders-csv",
        default=DEFAULT_ORDER_LOG_CSV,
        help="Where to save the simulated order event CSV log.",
    )
    parser.add_argument(
        "--no-order-prints",
        action="store_true",
        help="Disable per-order console output during the simulation.",
    )
    return parser.parse_args()


def default_parameter_snapshot() -> Dict[str, float]:
    return {
        "entry_hmm_confidence": ENTRY_HMM_CONFIDENCE,
        "entry_max_volatility": ENTRY_MAX_VOLATILITY,
        "entry_momentum_min": ENTRY_MOMENTUM_MIN,
        "entry_volume_zscore_max": ENTRY_VOLUME_ZSCORE_MAX,
        "entry_spread_max_pct": ENTRY_SPREAD_MAX_PCT,
        "exit_stop_loss_pct": EXIT_STOP_LOSS_PCT,
        "exit_take_profit_pct": EXIT_TAKE_PROFIT_PCT,
        "mc_num_simulations": MC_NUM_SIMULATIONS,
        "mc_horizon_hours": MC_HORIZON_HOURS,
        "mc_max_position_pct": MC_MAX_POSITION_PCT,
        "mc_min_position_pct": MC_MIN_POSITION_PCT,
        "mc_tail_risk_limit": MC_TAIL_RISK_LIMIT,
    }


def merge_parameter_snapshot(overrides: Dict[str, float] | None = None) -> Dict[str, float]:
    params = default_parameter_snapshot()
    if overrides:
        for key, value in overrides.items():
            if key in params and value is not None:
                params[key] = value

    params["mc_num_simulations"] = max(50, int(round(params["mc_num_simulations"])))
    params["mc_horizon_hours"] = max(1, int(round(params["mc_horizon_hours"])))
    params["mc_max_position_pct"] = float(np.clip(params["mc_max_position_pct"], 0.01, 1.0))
    params["mc_min_position_pct"] = float(np.clip(params["mc_min_position_pct"], 0.0, params["mc_max_position_pct"]))
    params["entry_hmm_confidence"] = float(np.clip(params["entry_hmm_confidence"], 0.0, 1.0))
    params["entry_volume_zscore_max"] = float(max(0.1, params["entry_volume_zscore_max"]))
    params["entry_spread_max_pct"] = float(max(0.0, params["entry_spread_max_pct"]))
    params["exit_stop_loss_pct"] = float(max(0.0005, params["exit_stop_loss_pct"]))
    params["exit_take_profit_pct"] = float(max(0.0005, params["exit_take_profit_pct"]))
    params["mc_tail_risk_limit"] = float(max(0.0001, params["mc_tail_risk_limit"]))
    return params


def normalize_pair_token(token: str) -> str:
    cleaned = token.strip().upper()
    if not cleaned:
        raise ValueError("Empty pair token.")
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        return f"{base}/{quote}"
    return f"{cleaned}/USD"


def resolve_pairs(pairs: List[str] | None = None) -> List[str]:
    requested = PAIRS if not pairs else [normalize_pair_token(pair) for pair in pairs]
    unknown = [pair for pair in requested if pair not in PAIRS]
    if unknown:
        raise ValueError(f"Unsupported pairs requested: {', '.join(unknown)}")
    return requested


def load_pair_dataframe(
    pair: str,
    refresh_data: bool,
    notes: List[str],
    progress_enabled: bool = True,
) -> pd.DataFrame:
    cache_path = runtime_path(os.path.join(DATA_DIR, f"{pair_to_binance_symbol(pair)}_1h.csv"))

    if refresh_data:
        try:
            progress(f"[data] Refreshing {pair} from Binance cache source...", enabled=progress_enabled)
            df_raw = load_or_fetch_historical(pair, force_refresh=True)
            if not df_raw.empty:
                return compute_features(df_raw)
        except Exception as exc:
            notes.append(f"{pair}: refresh failed, fell back to cache ({exc})")

    if os.path.exists(cache_path):
        progress(f"[data] Loading cached candles for {pair}...", enabled=progress_enabled)
        df_raw = pd.read_csv(cache_path, parse_dates=["open_time"])
        df_raw = df_raw.sort_values("open_time").reset_index(drop=True)
        return compute_features(df_raw)

    progress(f"[data] Fetching historical candles for {pair}...", enabled=progress_enabled)
    df_raw = load_or_fetch_historical(pair, force_refresh=False)
    if df_raw.empty:
        raise RuntimeError(f"No historical data available for {pair}")
    return compute_features(df_raw)


def align_pair_data(
    pairs: List[str], weeks: int, refresh_data: bool, progress_enabled: bool = True
) -> tuple[Dict[str, Dict[str, object]], List[str], List[str]]:
    notes: List[str] = []
    pair_data: Dict[str, Dict[str, object]] = {}
    eval_bars = weeks * BARS_PER_WEEK

    progress(
        f"[setup] Preparing recent {weeks}-week evaluation window for {len(pairs)} pairs...",
        enabled=progress_enabled,
    )
    for pair in pairs:
        df = load_pair_dataframe(
            pair,
            refresh_data=refresh_data,
            notes=notes,
            progress_enabled=progress_enabled,
        )
        if len(df) <= eval_bars + 300:
            notes.append(f"{pair}: skipped because there is not enough history for {weeks} weeks")
            continue

        time_to_index = {ts.isoformat(): idx for idx, ts in enumerate(df["open_time"])}
        obs = df[["log_return", "rolling_vol", "momentum", "volume_zscore"]].values
        pair_data[pair] = {
            "df": df,
            "obs": obs,
            "time_to_index": time_to_index,
        }
        progress(f"[setup] {pair}: {len(df)} feature rows ready", enabled=progress_enabled)

    if not pair_data:
        raise RuntimeError("No pair data could be prepared.")

    start_times = []
    for data in pair_data.values():
        df = data["df"]
        start_times.append(df["open_time"].iloc[-eval_bars])

    global_start = max(start_times)

    common_timestamps = None
    for data in pair_data.values():
        df = data["df"]
        current_times = {
            ts.isoformat() for ts in df.loc[df["open_time"] >= global_start, "open_time"]
        }
        common_timestamps = current_times if common_timestamps is None else common_timestamps & current_times

    timeline = sorted(common_timestamps)
    if not timeline:
        raise RuntimeError("No common timestamp range found across the selected pairs.")

    for pair, data in pair_data.items():
        start_idx = data["time_to_index"][timeline[0]]
        data["eval_start_idx"] = start_idx
        data["eval_end_idx"] = data["time_to_index"][timeline[-1]]

    progress(f"[setup] Common timeline built with {len(timeline)} bars", enabled=progress_enabled)
    return pair_data, timeline, notes


def clone_pair_data_for_timeline(
    pair_data: Dict[str, Dict[str, object]],
    timeline: List[str],
) -> Dict[str, Dict[str, object]]:
    if not timeline:
        raise ValueError("Timeline cannot be empty.")

    windowed: Dict[str, Dict[str, object]] = {}
    for pair, data in pair_data.items():
        cloned = dict(data)
        cloned["eval_start_idx"] = data["time_to_index"][timeline[0]]
        cloned["eval_end_idx"] = data["time_to_index"][timeline[-1]]
        windowed[pair] = cloned
    return windowed


def train_initial_hmms(
    pair_data: Dict[str, Dict[str, object]],
    progress_enabled: bool = True,
) -> Dict[str, RegimeHMM]:
    hmms: Dict[str, RegimeHMM] = {}
    train_bars = HMM_TRAINING_HOURS * BARS_PER_HOUR

    total_pairs = len(pair_data)
    for idx, (pair, data) in enumerate(pair_data.items(), start=1):
        start_idx = data["eval_start_idx"]
        train_start = max(0, start_idx - train_bars)
        train_obs = data["obs"][train_start:start_idx]
        if len(train_obs) < 300:
            raise RuntimeError(f"Not enough warmup observations to train HMM for {pair}")

        progress(
            f"[hmm] Training {pair} ({idx}/{total_pairs}) on {len(train_obs)} observations...",
            enabled=progress_enabled,
        )
        hmm = RegimeHMM()
        hmm.train(train_obs, pair)
        hmms[pair] = hmm
        progress(f"[hmm] Finished {pair}", enabled=progress_enabled)

    return hmms


def regime_probabilities(hmm: RegimeHMM, obs: np.ndarray) -> Dict[str, float]:
    return hmm.get_regime_probabilities(obs)


def dominant_regime(hmm: RegimeHMM, probs: Dict[str, float]) -> tuple[str, int]:
    label = max(probs, key=probs.get)
    mapping = {
        "bullish": hmm.bull_idx,
        "bearish": hmm.bear_idx,
        "neutral": hmm.neut_idx,
    }
    return label, mapping[label]


def mark_to_market(
    cash: float,
    positions: Dict[str, Position],
    current_prices: Dict[str, float],
    commission: float,
) -> float:
    equity = cash
    for pair, position in positions.items():
        if position.side == "long":
            equity += position.quantity * current_prices[pair] * (1 - commission)
    return equity


def compute_ratios(equity_curve: pd.DataFrame) -> Dict[str, float]:
    equity = equity_curve["equity"]
    returns = equity.pct_change().dropna()

    if returns.empty:
        return {
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "composite_score": 0.0,
        }

    mean_return = returns.mean()
    volatility = float(returns.std(ddof=0))
    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=0)) if not downside.empty else 0.0

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (PERIODS_PER_YEAR / len(returns)) - 1

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = abs(drawdown.min())

    if not np.isfinite(volatility):
        volatility = 0.0
    if not np.isfinite(downside_vol):
        downside_vol = 0.0
    if not np.isfinite(total_return):
        total_return = 0.0
    if not np.isfinite(annualized_return):
        annualized_return = 0.0
    if not np.isfinite(max_drawdown):
        max_drawdown = 0.0

    sharpe = 0.0 if volatility == 0 else (mean_return / volatility) * np.sqrt(PERIODS_PER_YEAR)
    sortino = 0.0 if downside_vol == 0 else (mean_return / downside_vol) * np.sqrt(PERIODS_PER_YEAR)
    calmar = 0.0 if max_drawdown == 0 else annualized_return / max_drawdown
    composite = 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar

    return {
        "total_return_pct": round(total_return * 100, 4),
        "annualized_return_pct": round(annualized_return * 100, 4),
        "max_drawdown_pct": round(max_drawdown * 100, 4),
        "sharpe_ratio": round(float(sharpe), 6),
        "sortino_ratio": round(float(sortino), 6),
        "calmar_ratio": round(float(calmar), 6),
        "composite_score": round(float(composite), 6),
    }


def summarize_closed_trades(trades: List[Dict[str, object]]) -> Dict[str, float]:
    if not trades:
        return {
            "closed_trades": 0,
            "win_rate_pct": 0.0,
            "avg_trade_return_pct": 0.0,
            "profit_factor": 0.0,
            "avg_holding_hours": 0.0,
            "realized_pnl": 0.0,
        }

    returns = np.array([trade["return_pct"] for trade in trades], dtype=float)
    pnls = np.array([trade["pnl"] for trade in trades], dtype=float)
    profits = pnls[pnls > 0].sum()
    losses = pnls[pnls < 0].sum()
    holding_hours = np.array([trade["holding_bars"] / BARS_PER_HOUR for trade in trades], dtype=float)

    profit_factor = 0.0 if losses == 0 else float(profits / abs(losses))

    return {
        "closed_trades": int(len(trades)),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 4),
        "avg_trade_return_pct": round(float(returns.mean() * 100), 4),
        "profit_factor": round(profit_factor, 6),
        "avg_holding_hours": round(float(holding_hours.mean()), 4),
        "realized_pnl": round(float(pnls.sum()), 4),
    }


def score_entries_and_exits(
    trades: List[Dict[str, object]],
    pair_data: Dict[str, Dict[str, object]],
) -> Dict[str, float]:
    if not trades:
        return {
            "entry_score": 0.0,
            "exit_score": 0.0,
            "entry_forward_1h_win_rate_pct": 0.0,
            "entry_forward_4h_win_rate_pct": 0.0,
            "entry_avg_forward_1h_pct": 0.0,
            "entry_avg_forward_4h_pct": 0.0,
            "exit_post_1h_success_rate_pct": 0.0,
            "exit_post_4h_success_rate_pct": 0.0,
            "exit_avg_post_1h_return_pct": 0.0,
            "exit_avg_post_4h_return_pct": 0.0,
        }

    entry_forward_1h = []
    entry_forward_4h = []
    exit_post_1h = []
    exit_post_4h = []
    realized_positive = []

    for trade in trades:
        pair = trade["pair"]
        df = pair_data[pair]["df"]
        entry_idx = trade["entry_index"]
        exit_idx = trade["exit_index"]
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]

        entry_idx_1h = min(entry_idx + 12, len(df) - 1)
        entry_idx_4h = min(entry_idx + 48, len(df) - 1)
        exit_idx_1h = min(exit_idx + 12, len(df) - 1)
        exit_idx_4h = min(exit_idx + 48, len(df) - 1)

        entry_forward_1h.append(df.iloc[entry_idx_1h]["close"] / entry_price - 1)
        entry_forward_4h.append(df.iloc[entry_idx_4h]["close"] / entry_price - 1)
        exit_post_1h.append(df.iloc[exit_idx_1h]["close"] / exit_price - 1)
        exit_post_4h.append(df.iloc[exit_idx_4h]["close"] / exit_price - 1)
        realized_positive.append(trade["return_pct"] > 0)

    entry_forward_1h = np.array(entry_forward_1h, dtype=float)
    entry_forward_4h = np.array(entry_forward_4h, dtype=float)
    exit_post_1h = np.array(exit_post_1h, dtype=float)
    exit_post_4h = np.array(exit_post_4h, dtype=float)
    realized_positive = np.array(realized_positive, dtype=bool)

    entry_score = 100 * (
        0.4 * realized_positive.mean()
        + 0.3 * (entry_forward_1h > 0).mean()
        + 0.3 * (entry_forward_4h > 0).mean()
    )
    exit_score = 100 * (
        0.5 * (exit_post_1h <= 0).mean()
        + 0.5 * (exit_post_4h <= 0).mean()
    )

    return {
        "entry_score": round(float(entry_score), 4),
        "exit_score": round(float(exit_score), 4),
        "entry_forward_1h_win_rate_pct": round(float((entry_forward_1h > 0).mean() * 100), 4),
        "entry_forward_4h_win_rate_pct": round(float((entry_forward_4h > 0).mean() * 100), 4),
        "entry_avg_forward_1h_pct": round(float(entry_forward_1h.mean() * 100), 4),
        "entry_avg_forward_4h_pct": round(float(entry_forward_4h.mean() * 100), 4),
        "exit_post_1h_success_rate_pct": round(float((exit_post_1h <= 0).mean() * 100), 4),
        "exit_post_4h_success_rate_pct": round(float((exit_post_4h <= 0).mean() * 100), 4),
        "exit_avg_post_1h_return_pct": round(float(exit_post_1h.mean() * 100), 4),
        "exit_avg_post_4h_return_pct": round(float(exit_post_4h.mean() * 100), 4),
    }


def per_pair_summary(trades: List[Dict[str, object]], pairs: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for pair in pairs:
        pair_trades = [trade for trade in trades if trade["pair"] == pair]
        summary[pair] = summarize_closed_trades(pair_trades)
    return summary


def run_simulation(
    pair_data: Dict[str, Dict[str, object]],
    timeline: List[str],
    starting_balance: float,
    commission: float,
    params: Dict[str, float] | None = None,
    hmms: Dict[str, RegimeHMM] | None = None,
    print_orders: bool = True,
    progress_enabled: bool = True,
) -> Dict[str, object]:
    parameter_snapshot = merge_parameter_snapshot(params)
    start_time = time.time()
    if hmms is None:
        hmms = train_initial_hmms(pair_data, progress_enabled=progress_enabled)
    signal_gen = SignalGenerator()

    cash = starting_balance
    positions = {pair: Position() for pair in pair_data}
    trades: List[Dict[str, object]] = []
    order_events: List[Dict[str, object]] = []
    equity_rows: List[Dict[str, object]] = []
    total_steps = len(timeline)
    progress_interval = max(1, total_steps // 20)

    progress(
        f"[sim] Starting virtual-money simulation across {total_steps} bars...",
        enabled=progress_enabled,
    )
    for step_index, ts in enumerate(timeline, start=1):
        current_prices = {}
        row_cache = {}
        index_cache = {}

        for pair, data in pair_data.items():
            idx = data["time_to_index"][ts]
            row = data["df"].iloc[idx]
            current_prices[pair] = float(row["close"])
            row_cache[pair] = row
            index_cache[pair] = idx

        for pair, data in pair_data.items():
            hmm = hmms[pair]
            idx = index_cache[pair]
            row = row_cache[pair]
            position = positions[pair]
            obs_now = data["obs"][: idx + 1]

            probs = regime_probabilities(hmm, obs_now)
            regime_label, regime_idx = dominant_regime(hmm, probs)

            features = {
                "log_return": float(row["log_return"]),
                "rolling_vol": float(row["rolling_vol"]),
                "momentum": float(row["momentum"]),
                "volume_zscore": float(row["volume_zscore"]),
                "ma20": float(row["ma20"]),
                "ma50": float(row["ma50"]),
                "close": float(row["close"]),
            }
            signal = signal_gen.generate_signal(
                probs,
                features,
                spread_pct=0.0,
                current_position=asdict(position),
                params=parameter_snapshot,
            )

            if signal["action"] == "BUY" and position.side == "none":
                portfolio_value = mark_to_market(cash, positions, current_prices, commission)
                mc_result = monte_carlo_position_size(
                    hmm.get_regime_params(),
                    regime_idx,
                    portfolio_value,
                    params=parameter_snapshot,
                )
                allocation = min(float(mc_result["position_usd"]), cash)
                if allocation < 1.0:
                    continue

                fill_price = float(row["close"]) * (1 + commission)
                quantity = allocation / fill_price
                if quantity <= 0:
                    continue

                cash -= allocation
                positions[pair] = Position(
                    side="long",
                    entry_price=fill_price,
                    quantity=quantity,
                    entry_index=idx,
                    entry_time=ts,
                    entry_reasons=list(signal["reasons"]),
                    entry_confidence=float(signal["confidence"]),
                    entry_regime=regime_label,
                    mc_position_pct=float(mc_result["position_pct"]),
                    mc_position_usd=allocation,
                )
                equity_after = mark_to_market(cash, positions, current_prices, commission)
                order_event = {
                    "order_no": len(order_events) + 1,
                    "event": "BUY",
                    "timestamp": ts,
                    "pair": pair,
                    "price": round(fill_price, 8),
                    "market_price": round(float(row["close"]), 8),
                    "quantity": round(quantity, 8),
                    "notional_usd": round(allocation, 8),
                    "cash_after": round(float(cash), 8),
                    "equity_after": round(float(equity_after), 8),
                    "regime": regime_label,
                    "signal_confidence": round(float(signal["confidence"]), 8),
                    "reason": " | ".join(signal["reasons"]),
                    "mc_position_pct": round(float(mc_result["position_pct"]), 8),
                    "mc_position_usd": round(float(mc_result["position_usd"]), 8),
                }
                order_events.append(order_event)
                print_order_event(order_event, enabled=print_orders)

            elif signal["action"] == "SELL" and position.side == "long":
                fill_price = float(row["close"]) * (1 - commission)
                proceeds = position.quantity * fill_price
                cost_basis = position.quantity * position.entry_price
                pnl = proceeds - cost_basis
                return_pct = fill_price / position.entry_price - 1
                cash += proceeds
                holding_hours = (idx - position.entry_index) / BARS_PER_HOUR
                order_event = {
                    "order_no": len(order_events) + 1,
                    "event": "SELL",
                    "timestamp": ts,
                    "pair": pair,
                    "price": round(fill_price, 8),
                    "market_price": round(float(row["close"]), 8),
                    "quantity": round(position.quantity, 8),
                    "notional_usd": round(proceeds, 8),
                    "cash_after": round(float(cash), 8),
                    "equity_after": round(float(cash), 8),
                    "regime": regime_label,
                    "signal_confidence": round(float(signal["confidence"]), 8),
                    "reason": " | ".join(signal["reasons"]),
                    "pnl": round(float(pnl), 8),
                    "return_pct": round(float(return_pct * 100), 8),
                    "holding_hours": round(float(holding_hours), 8),
                }
                order_events.append(order_event)
                print_order_event(order_event, enabled=print_orders)

                trades.append(
                    {
                        "pair": pair,
                        "entry_time": position.entry_time,
                        "exit_time": ts,
                        "entry_index": position.entry_index,
                        "exit_index": idx,
                        "entry_price": round(position.entry_price, 8),
                        "exit_price": round(fill_price, 8),
                        "quantity": round(position.quantity, 8),
                        "return_pct": round(float(return_pct), 8),
                        "pnl": round(float(pnl), 8),
                        "holding_bars": int(idx - position.entry_index),
                        "entry_regime": position.entry_regime,
                        "exit_regime": regime_label,
                        "entry_confidence": round(position.entry_confidence, 8),
                        "exit_confidence": round(float(signal["confidence"]), 8),
                        "entry_reasons": " | ".join(position.entry_reasons),
                        "exit_reasons": " | ".join(signal["reasons"]),
                        "mc_position_pct": round(position.mc_position_pct, 8),
                        "mc_position_usd": round(position.mc_position_usd, 8),
                    }
                )
                positions[pair] = Position()

        equity = mark_to_market(cash, positions, current_prices, commission)
        equity_rows.append(
            {
                "timestamp": ts,
                "equity": round(float(equity), 8),
                "cash": round(float(cash), 8),
                "open_positions": int(sum(pos.side == "long" for pos in positions.values())),
            }
        )

        if step_index == 1 or step_index % progress_interval == 0 or step_index == total_steps:
            pct = (step_index / total_steps) * 100
            elapsed = time.time() - start_time
            progress(
                f"[sim] {pct:5.1f}% | bar {step_index}/{total_steps} | "
                f"equity=${equity:,.2f} | trades={len(trades)} | elapsed={elapsed:.1f}s",
                enabled=progress_enabled,
            )

    if timeline:
        final_ts = timeline[-1]
        final_prices = {}
        for pair, data in pair_data.items():
            final_idx = data["time_to_index"][final_ts]
            final_prices[pair] = float(data["df"].iloc[final_idx]["close"])

        for pair, position in positions.items():
            if position.side != "long":
                continue

            fill_price = final_prices[pair] * (1 - commission)
            proceeds = position.quantity * fill_price
            cost_basis = position.quantity * position.entry_price
            pnl = proceeds - cost_basis
            return_pct = fill_price / position.entry_price - 1
            cash += proceeds
            holding_hours = (pair_data[pair]["time_to_index"][final_ts] - position.entry_index) / BARS_PER_HOUR
            order_event = {
                "order_no": len(order_events) + 1,
                "event": "SELL_FORCED",
                "timestamp": final_ts,
                "pair": pair,
                "price": round(fill_price, 8),
                "market_price": round(final_prices[pair], 8),
                "quantity": round(position.quantity, 8),
                "notional_usd": round(proceeds, 8),
                "cash_after": round(float(cash), 8),
                "equity_after": round(float(cash), 8),
                "regime": "forced_end",
                "signal_confidence": 0.0,
                "reason": "forced_close_end_of_test",
                "pnl": round(float(pnl), 8),
                "return_pct": round(float(return_pct * 100), 8),
                "holding_hours": round(float(holding_hours), 8),
            }
            order_events.append(order_event)
            print_order_event(order_event, enabled=print_orders)

            trades.append(
                {
                    "pair": pair,
                    "entry_time": position.entry_time,
                    "exit_time": final_ts,
                    "entry_index": position.entry_index,
                    "exit_index": pair_data[pair]["time_to_index"][final_ts],
                    "entry_price": round(position.entry_price, 8),
                    "exit_price": round(fill_price, 8),
                    "quantity": round(position.quantity, 8),
                    "return_pct": round(float(return_pct), 8),
                    "pnl": round(float(pnl), 8),
                    "holding_bars": int(pair_data[pair]["time_to_index"][final_ts] - position.entry_index),
                    "entry_regime": position.entry_regime,
                    "exit_regime": "forced_end",
                    "entry_confidence": round(position.entry_confidence, 8),
                    "exit_confidence": 0.0,
                    "entry_reasons": " | ".join(position.entry_reasons),
                    "exit_reasons": "forced_close_end_of_test",
                    "mc_position_pct": round(position.mc_position_pct, 8),
                    "mc_position_usd": round(position.mc_position_usd, 8),
                }
            )
            positions[pair] = Position()

        if equity_rows:
            equity_rows[-1]["equity"] = round(float(cash), 8)
            equity_rows[-1]["cash"] = round(float(cash), 8)
            equity_rows[-1]["open_positions"] = 0

    equity_curve = pd.DataFrame(equity_rows)
    trade_summary = summarize_closed_trades(trades)
    ratio_summary = compute_ratios(equity_curve)
    timing_summary = score_entries_and_exits(trades, pair_data)
    progress("[report] Building summary metrics and saving outputs...", enabled=progress_enabled)

    return {
        "equity_curve": equity_curve,
        "trades": trades,
        "order_events": order_events,
        "parameters": parameter_snapshot,
        "trade_summary": trade_summary,
        "ratio_summary": ratio_summary,
        "timing_summary": timing_summary,
        "per_pair": per_pair_summary(trades, list(pair_data.keys())),
        "final_balance": round(float(cash), 4),
    }


def make_report(
    pair_data: Dict[str, Dict[str, object]],
    timeline: List[str],
    notes: List[str],
    results: Dict[str, object],
    *,
    weeks: int,
    starting_balance: float,
    commission: float,
) -> Dict[str, object]:
    report = {
        "window": {
            "weeks": weeks,
            "start_time": timeline[0],
            "end_time": timeline[-1],
            "bars": len(timeline),
            "pairs": list(pair_data.keys()),
        },
        "simulation": {
            "starting_balance": starting_balance,
            "final_balance": results["final_balance"],
            "commission": commission,
        },
        "parameters": results["parameters"],
        "return_metrics": results["ratio_summary"],
        "trade_metrics": results["trade_summary"],
        "order_metrics": {
            "total_orders": len(results["order_events"]),
            "buy_orders": sum(event["event"] == "BUY" for event in results["order_events"]),
            "sell_orders": sum(event["event"] == "SELL" for event in results["order_events"]),
            "forced_sell_orders": sum(event["event"] == "SELL_FORCED" for event in results["order_events"]),
        },
        "timing_metrics": results["timing_summary"],
        "per_pair": results["per_pair"],
        "assumptions": [
            "Uses Binance 5-minute OHLCV candles from the local TEST-bot cache unless refresh is requested.",
            "Historical spread is unavailable in OHLCV data, so offline spread_pct is fixed at 0.0 for signal checks.",
            "Orders are filled at candle close with configurable per-side commission.",
            "Entry and exit scores are heuristic ratings meant for parameter search, not ground-truth labels.",
        ],
        "notes": notes,
    }
    return report


def evaluate_strategy(
    *,
    pairs: List[str] | None = None,
    weeks: int = 4,
    starting_balance: float = DEFAULT_STARTING_BALANCE,
    commission: float = DEFAULT_COMMISSION,
    refresh_data: bool = False,
    params: Dict[str, float] | None = None,
    print_orders: bool = True,
    progress_enabled: bool = True,
    pair_data: Dict[str, Dict[str, object]] | None = None,
    timeline: List[str] | None = None,
    hmms: Dict[str, RegimeHMM] | None = None,
) -> Dict[str, object]:
    selected_pairs = resolve_pairs(pairs)

    notes: List[str] = []
    if pair_data is None or timeline is None:
        pair_data, timeline, notes = align_pair_data(
            selected_pairs,
            weeks,
            refresh_data,
            progress_enabled=progress_enabled,
        )
    else:
        pair_data = clone_pair_data_for_timeline(pair_data, timeline)

    results = run_simulation(
        pair_data=pair_data,
        timeline=timeline,
        starting_balance=starting_balance,
        commission=commission,
        params=params,
        hmms=hmms,
        print_orders=print_orders,
        progress_enabled=progress_enabled,
    )
    report = make_report(
        pair_data,
        timeline,
        notes,
        results,
        weeks=weeks,
        starting_balance=starting_balance,
        commission=commission,
    )
    return {
        "pair_data": pair_data,
        "timeline": timeline,
        "notes": notes,
        "results": results,
        "report": report,
    }


def print_report(report: Dict[str, object]) -> None:
    window = report["window"]
    sim = report["simulation"]
    returns = report["return_metrics"]
    trades = report["trade_metrics"]
    orders = report["order_metrics"]
    timing = report["timing_metrics"]

    print("=" * 72)
    print("TEST-bot Performance Check")
    print("=" * 72)
    print(
        f"Window: {window['start_time']} -> {window['end_time']} "
        f"({window['weeks']} weeks, {window['bars']} bars)"
    )
    print(f"Pairs:  {', '.join(window['pairs'])}")
    print(f"Start:  ${sim['starting_balance']:,.2f}")
    print(f"Final:  ${sim['final_balance']:,.2f}")
    print()

    print("Return Metrics")
    print(f"  Total return:       {returns['total_return_pct']:+.4f}%")
    print(f"  Annualized return:  {returns['annualized_return_pct']:+.4f}%")
    print(f"  Max drawdown:       {returns['max_drawdown_pct']:.4f}%")
    print(f"  Sharpe ratio:       {returns['sharpe_ratio']:.6f}")
    print(f"  Sortino ratio:      {returns['sortino_ratio']:.6f}")
    print(f"  Calmar ratio:       {returns['calmar_ratio']:.6f}")
    print(f"  Composite score:    {returns['composite_score']:.6f}")
    print()

    print("Trade Metrics")
    print(f"  Closed trades:      {trades['closed_trades']}")
    print(f"  Orders logged:      {orders['total_orders']}")
    print(f"  Win rate:           {trades['win_rate_pct']:.4f}%")
    print(f"  Avg trade return:   {trades['avg_trade_return_pct']:+.4f}%")
    print(f"  Profit factor:      {trades['profit_factor']:.6f}")
    print(f"  Avg holding hours:  {trades['avg_holding_hours']:.4f}")
    print(f"  Realized PnL:       ${trades['realized_pnl']:,.2f}")
    print()

    print("Entry / Exit Ratings")
    print(f"  Entry score:        {timing['entry_score']:.4f} / 100")
    print(f"  Exit score:         {timing['exit_score']:.4f} / 100")
    print(f"  Entry 1h win rate:  {timing['entry_forward_1h_win_rate_pct']:.4f}%")
    print(f"  Entry 4h win rate:  {timing['entry_forward_4h_win_rate_pct']:.4f}%")
    print(f"  Exit 1h success:    {timing['exit_post_1h_success_rate_pct']:.4f}%")
    print(f"  Exit 4h success:    {timing['exit_post_4h_success_rate_pct']:.4f}%")
    print()

    print("Parameters Used")
    for key, value in report["parameters"].items():
        print(f"  {key}: {value}")

    if report["notes"]:
        print()
        print("Notes")
        for note in report["notes"]:
            print(f"  - {note}")


def save_outputs(
    report: Dict[str, object],
    trades: List[Dict[str, object]],
    order_events: List[Dict[str, object]],
    output_json: str,
    output_csv: str,
    output_orders_jsonl: str,
    output_orders_csv: str,
) -> None:
    output_json = runtime_path(output_json)
    output_csv = runtime_path(output_csv)
    output_orders_jsonl = runtime_path(output_orders_jsonl)
    output_orders_csv = runtime_path(output_orders_csv)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(output_orders_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(output_orders_csv), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if trades:
        pd.DataFrame(trades).to_csv(output_csv, index=False)
    else:
        pd.DataFrame(columns=["pair", "entry_time", "exit_time"]).to_csv(output_csv, index=False)

    with open(output_orders_jsonl, "w", encoding="utf-8") as handle:
        for event in order_events:
            handle.write(json.dumps(event) + "\n")

    if order_events:
        pd.DataFrame(order_events).to_csv(output_orders_csv, index=False)
    else:
        pd.DataFrame(columns=["event", "timestamp", "pair"]).to_csv(output_orders_csv, index=False)


def main() -> None:
    args = parse_args()
    progress("[start] performance_check.py")
    evaluation = evaluate_strategy(
        pairs=args.pairs,
        weeks=args.weeks,
        starting_balance=args.starting_balance,
        commission=args.commission,
        refresh_data=args.refresh_data,
        print_orders=not args.no_order_prints,
        progress_enabled=True,
    )
    report = evaluation["report"]
    print_report(report)
    save_outputs(
        report,
        evaluation["results"]["trades"],
        evaluation["results"]["order_events"],
        args.output_json,
        args.output_csv,
        args.output_orders_jsonl,
        args.output_orders_csv,
    )
    progress(
        "[done] Saved report to "
        f"{args.output_json}, trades to {args.output_csv}, "
        f"and order logs to {args.output_orders_jsonl} / {args.output_orders_csv}"
    )


if __name__ == "__main__":
    main()
