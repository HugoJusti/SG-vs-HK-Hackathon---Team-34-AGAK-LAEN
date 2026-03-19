"""
Performance metrics and plotting utilities for the trading bot.

This module reads the bot's JSONL signal and trade logs, reconstructs closed
trades, computes summary metrics, and saves a chart image that covers:
1. Cumulative return
2. Sharpe and Sortino
3. Drawdown
4. Turnover and trading costs
5. Signal edge via IC and hit rate
"""
import json
import math
import os
from typing import Dict, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_jsonl(path: str) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame()

    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return pd.DataFrame()

    frame = pd.json_normalize(rows)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        frame = frame.sort_values("timestamp")
    return frame.reset_index(drop=True)


def _safe_float(value, default: float = 0.0) -> float:
    """Convert to float while tolerating None/NaN/string noise."""
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw trade log rows."""
    if trades.empty:
        return trades

    frame = trades.copy()
    if "side" not in frame.columns:
        frame["side"] = ""
    if "success" not in frame.columns:
        frame["success"] = False
    if "quantity" not in frame.columns:
        frame["quantity"] = 0.0
    if "commission" not in frame.columns:
        frame["commission"] = 0.0
    if "filled_price" not in frame.columns:
        frame["filled_price"] = 0.0
    if "price" not in frame.columns:
        frame["price"] = 0.0

    frame["side"] = frame["side"].astype(str).str.upper()
    frame["success"] = frame["success"].fillna(False).astype(bool)
    frame["quantity"] = frame["quantity"].apply(_safe_float)
    frame["commission"] = frame["commission"].apply(_safe_float)

    filled = frame["filled_price"].apply(_safe_float)
    requested = frame["price"].apply(_safe_float)
    frame["effective_price"] = np.where(filled > 0, filled, requested)

    return frame[
        frame["success"]
        & frame["side"].isin(["BUY", "SELL"])
        & (frame["quantity"] > 0)
        & (frame["effective_price"] > 0)
    ].copy()


def _build_round_trips(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct round trips from BUY then SELL events.

    The current bot only carries one long position per pair, so a simple
    per-pair state machine is sufficient.
    """
    if trades.empty:
        return pd.DataFrame()

    open_positions: Dict[str, Dict[str, float]] = {}
    round_trips = []

    for row in trades.sort_values("timestamp").itertuples(index=False):
        pair = getattr(row, "pair", "")
        side = getattr(row, "side", "")
        price = _safe_float(getattr(row, "effective_price", 0))
        quantity = _safe_float(getattr(row, "quantity", 0))
        commission = _safe_float(getattr(row, "commission", 0))
        timestamp = getattr(row, "timestamp", pd.NaT)

        if side == "BUY":
            open_positions[pair] = {
                "entry_time": timestamp,
                "entry_price": price,
                "quantity": quantity,
                "buy_commission": commission,
            }
            continue

        if side != "SELL" or pair not in open_positions:
            continue

        entry = open_positions.pop(pair)
        closed_quantity = min(quantity, entry["quantity"])
        if closed_quantity <= 0 or entry["entry_price"] <= 0:
            continue

        buy_notional = entry["entry_price"] * closed_quantity
        sell_notional = price * closed_quantity
        total_commission = entry["buy_commission"] + commission
        gross_pnl = sell_notional - buy_notional
        net_pnl = gross_pnl - total_commission
        net_return = net_pnl / buy_notional if buy_notional else 0.0
        holding_hours = np.nan
        if pd.notna(entry["entry_time"]) and pd.notna(timestamp):
            holding_hours = (timestamp - entry["entry_time"]).total_seconds() / 3600.0

        round_trips.append(
            {
                "pair": pair,
                "entry_time": entry["entry_time"],
                "exit_time": timestamp,
                "entry_price": entry["entry_price"],
                "exit_price": price,
                "quantity": closed_quantity,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "turnover": buy_notional + sell_notional,
                "commission": total_commission,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "net_return": net_return,
                "holding_hours": holding_hours,
            }
        )

    if not round_trips:
        return pd.DataFrame()

    frame = pd.DataFrame(round_trips).sort_values("exit_time").reset_index(drop=True)
    frame["equity_curve"] = (1.0 + frame["net_return"]).cumprod()
    frame["cumulative_return"] = frame["equity_curve"] - 1.0
    rolling_peak = frame["equity_curve"].cummax()
    frame["drawdown"] = frame["equity_curve"] / rolling_peak - 1.0

    window = max(3, min(10, len(frame)))
    rolling_mean = frame["net_return"].rolling(window).mean()
    rolling_std = frame["net_return"].rolling(window).std(ddof=0).replace(0, np.nan)
    downside = frame["net_return"].where(frame["net_return"] < 0, 0.0)
    downside_std = downside.rolling(window).std(ddof=0).replace(0, np.nan)

    frame["rolling_sharpe"] = (rolling_mean / rolling_std) * math.sqrt(window)
    frame["rolling_sortino"] = (rolling_mean / downside_std) * math.sqrt(window)

    frame["cum_turnover"] = frame["turnover"].cumsum()
    frame["cum_commission"] = frame["commission"].cumsum()
    return frame


def _prepare_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Normalize signal log rows and compute forward returns."""
    if signals.empty:
        return signals

    frame = signals.copy()
    price_col = "ticker.last"
    bull_col = "regime.bullish"
    bear_col = "regime.bearish"

    if price_col not in frame.columns:
        return pd.DataFrame()

    if "action" not in frame.columns:
        frame["action"] = "HOLD"

    frame["price"] = frame[price_col].apply(_safe_float)
    frame["bullish"] = frame[bull_col].apply(_safe_float) if bull_col in frame.columns else 0.0
    frame["bearish"] = frame[bear_col].apply(_safe_float) if bear_col in frame.columns else 0.0
    frame["edge_score"] = frame["bullish"] - frame["bearish"]
    frame["action_score"] = frame["action"].map({"BUY": 1, "SELL": -1}).fillna(0)

    frame = frame.sort_values(["pair", "timestamp"]).reset_index(drop=True)
    frame["next_price"] = frame.groupby("pair")["price"].shift(-1)
    frame["forward_return"] = frame["next_price"] / frame["price"] - 1.0

    valid = frame["price"] > 0
    return frame[valid].copy()


def _compute_summary(round_trips: pd.DataFrame, signals: pd.DataFrame) -> Dict[str, float]:
    """Aggregate summary metrics for the chart text panel."""
    summary = {
        "closed_trades": 0,
        "total_return": np.nan,
        "avg_trade_return": np.nan,
        "sharpe": np.nan,
        "sortino": np.nan,
        "max_drawdown": np.nan,
        "turnover": 0.0,
        "commission": 0.0,
        "ic": np.nan,
        "hit_rate": np.nan,
    }

    if not round_trips.empty:
        returns = round_trips["net_return"]
        downside = returns[returns < 0]

        summary["closed_trades"] = int(len(round_trips))
        summary["total_return"] = round_trips["cumulative_return"].iloc[-1]
        summary["avg_trade_return"] = returns.mean()
        std = returns.std(ddof=0)
        downside_std = downside.std(ddof=0)
        summary["sharpe"] = returns.mean() / std if std and not np.isnan(std) else np.nan
        summary["sortino"] = (
            returns.mean() / downside_std if downside_std and not np.isnan(downside_std) else np.nan
        )
        summary["max_drawdown"] = round_trips["drawdown"].min()
        summary["turnover"] = round_trips["turnover"].sum()
        summary["commission"] = round_trips["commission"].sum()

    if not signals.empty and "forward_return" in signals.columns:
        valid = signals.dropna(subset=["edge_score", "forward_return"])
        if len(valid) >= 2:
            summary["ic"] = valid["edge_score"].corr(valid["forward_return"], method="spearman")

        action_valid = signals[
            signals["action_score"].ne(0) & signals["forward_return"].notna()
        ].copy()
        if not action_valid.empty:
            summary["hit_rate"] = (
                np.sign(action_valid["action_score"]) == np.sign(action_valid["forward_return"])
            ).mean()
        elif len(valid) >= 1:
            summary["hit_rate"] = (
                np.sign(valid["edge_score"]) == np.sign(valid["forward_return"])
            ).mean()

    return summary


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_ratio(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}"


def plot_metrics_report(
    signals_path: str,
    trades_path: str,
    output_path: str,
) -> Tuple[Dict[str, float], str]:
    """Create and save a performance chart image."""
    raw_signals = _load_jsonl(signals_path)
    raw_trades = _load_jsonl(trades_path)

    signals = _prepare_signals(raw_signals)
    trades = _prepare_trades(raw_trades)
    round_trips = _build_round_trips(trades)
    summary = _compute_summary(round_trips, signals)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle("Trading Bot Performance Report", fontsize=16, fontweight="bold")
    ax_return, ax_risk, ax_drawdown, ax_costs, ax_edge, ax_summary = axes.flatten()

    if not round_trips.empty:
        ax_return.plot(
            round_trips["exit_time"],
            round_trips["cumulative_return"] * 100.0,
            color="#0f766e",
            linewidth=2.2,
        )
        ax_return.set_title("Return: cumulative closed-trade return")
        ax_return.set_ylabel("Return (%)")
        ax_return.grid(alpha=0.25)

        ax_risk.plot(
            round_trips["exit_time"],
            round_trips["rolling_sharpe"],
            label="Sharpe",
            color="#1d4ed8",
            linewidth=2.0,
        )
        ax_risk.plot(
            round_trips["exit_time"],
            round_trips["rolling_sortino"],
            label="Sortino",
            color="#7c3aed",
            linewidth=2.0,
        )
        ax_risk.axhline(0, color="#6b7280", linewidth=1)
        ax_risk.set_title("Efficiency: rolling Sharpe and Sortino")
        ax_risk.set_ylabel("Ratio")
        ax_risk.legend()
        ax_risk.grid(alpha=0.25)

        ax_drawdown.fill_between(
            round_trips["exit_time"],
            round_trips["drawdown"] * 100.0,
            0,
            color="#dc2626",
            alpha=0.35,
        )
        ax_drawdown.set_title("Survivability: drawdown")
        ax_drawdown.set_ylabel("Drawdown (%)")
        ax_drawdown.grid(alpha=0.25)

        ax_costs.plot(
            round_trips["exit_time"],
            round_trips["cum_turnover"],
            label="Cumulative turnover",
            color="#ea580c",
            linewidth=2.0,
        )
        ax_costs.plot(
            round_trips["exit_time"],
            round_trips["cum_commission"],
            label="Cumulative commissions",
            color="#b91c1c",
            linewidth=2.0,
        )
        ax_costs.set_title("Tradability: turnover and costs")
        ax_costs.set_ylabel("USD")
        ax_costs.legend()
        ax_costs.grid(alpha=0.25)
    else:
        for axis, title in [
            (ax_return, "Return: cumulative closed-trade return"),
            (ax_risk, "Efficiency: rolling Sharpe and Sortino"),
            (ax_drawdown, "Survivability: drawdown"),
            (ax_costs, "Tradability: turnover and costs"),
        ]:
            axis.set_title(title)
            axis.text(0.5, 0.5, "No closed trades logged yet", ha="center", va="center")
            axis.set_xticks([])
            axis.set_yticks([])

    if not signals.empty and signals["forward_return"].notna().any():
        edge_data = signals.dropna(subset=["edge_score", "forward_return"])
        if not edge_data.empty:
            colors = edge_data["pair"].astype("category").cat.codes
            scatter = ax_edge.scatter(
                edge_data["edge_score"],
                edge_data["forward_return"] * 100.0,
                c=colors,
                cmap="viridis",
                alpha=0.65,
            )
            ax_edge.axhline(0, color="#6b7280", linewidth=1)
            ax_edge.axvline(0, color="#6b7280", linewidth=1)
            ax_edge.set_title("Signal edge: IC and forward return hit rate")
            ax_edge.set_xlabel("Signal score (bullish - bearish)")
            ax_edge.set_ylabel("Next logged return (%)")
            ax_edge.grid(alpha=0.25)
            legend_pairs = edge_data["pair"].astype("category").cat.categories
            handles = scatter.legend_elements()[0]
            ax_edge.legend(handles, legend_pairs, title="Pair", loc="best")
        else:
            ax_edge.set_title("Signal edge: IC and forward return hit rate")
            ax_edge.text(0.5, 0.5, "Not enough signal history", ha="center", va="center")
            ax_edge.set_xticks([])
            ax_edge.set_yticks([])
    else:
        ax_edge.set_title("Signal edge: IC and forward return hit rate")
        ax_edge.text(0.5, 0.5, "No signal log history found", ha="center", va="center")
        ax_edge.set_xticks([])
        ax_edge.set_yticks([])

    summary_lines = [
        f"Closed trades: {summary['closed_trades']}",
        f"Total return: {_format_pct(summary['total_return'])}",
        f"Average trade return: {_format_pct(summary['avg_trade_return'])}",
        f"Sharpe: {_format_ratio(summary['sharpe'])}",
        f"Sortino: {_format_ratio(summary['sortino'])}",
        f"Max drawdown: {_format_pct(summary['max_drawdown'])}",
        f"Turnover: ${summary['turnover']:,.2f}",
        f"Commissions: ${summary['commission']:,.2f}",
        f"IC: {_format_ratio(summary['ic'])}",
        f"Hit rate: {_format_pct(summary['hit_rate'])}",
    ]
    ax_summary.set_title("Summary")
    ax_summary.axis("off")
    ax_summary.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=11,
        family="monospace",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return summary, output_path
