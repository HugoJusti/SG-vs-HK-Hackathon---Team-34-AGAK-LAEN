import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# ----------------------------
# Feature engineering
# ----------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Log return
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling volatility (20-candle std of log returns)
    df["rolling_vol"] = df["log_ret"].rolling(20).std()

    # Short-term momentum (6-candle percentage change)
    df["momentum"] = df["close"].pct_change(6)

    # Short moving average
    df["ma20"] = df["close"].rolling(20).mean()

    # Volume z-score
    vol_mean = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    df["vol_z"] = (df["volume"] - vol_mean) / vol_std

    df = df.dropna().reset_index(drop=True)
    return df


# ----------------------------
# HMM training and labeling
# ----------------------------

def train_hmm(df: pd.DataFrame, n_states: int = 3):
    features = df[["log_ret", "rolling_vol", "momentum", "vol_z"]].values

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=300,
        random_state=42
    )
    model.fit(features)

    df = df.copy()
    df["state"] = model.predict(features)

    return model, df


def label_states(df: pd.DataFrame) -> dict:
    """
    Assign human meaning to HMM states.
    """
    summary = df.groupby("state").agg(
        mean_ret=("log_ret", "mean"),
        mean_vol=("rolling_vol", "mean")
    )

    bullish_state = summary["mean_ret"].idxmax()
    bearish_state = summary["mean_ret"].idxmin()

    # The remaining state becomes choppy
    remaining_states = [s for s in summary.index if s not in [bullish_state, bearish_state]]
    if remaining_states:
        choppy_state = remaining_states[0]
    else:
        choppy_state = summary["mean_vol"].idxmax()

    return {
        bullish_state: "bullish",
        bearish_state: "bearish",
        choppy_state: "choppy"
    }


def predict_latest_regime(model, df: pd.DataFrame, state_map: dict) -> str:
    x = df[["log_ret", "rolling_vol", "momentum", "vol_z"]].values
    latest_state = model.predict(x)[-1]
    return state_map[latest_state]


# ----------------------------
# Monte Carlo downside filter
# ----------------------------

def monte_carlo_downside_risk(
    current_price: float,
    recent_log_returns: pd.Series,
    horizon: int = 12,
    n_sims: int = 1000,
    percentile: float = 5
):
    """
    Simulate future price paths using recent mean and std of log returns.
    Returns the downside percentile of final returns.
    """
    mu = recent_log_returns.mean()
    sigma = recent_log_returns.std()

    simulated_final_returns = []

    for _ in range(n_sims):
        shocks = np.random.normal(mu, sigma, horizon)
        path_log_return = shocks.sum()
        final_price = current_price * np.exp(path_log_return)
        final_return = (final_price / current_price) - 1
        simulated_final_returns.append(final_return)

    simulated_final_returns = np.array(simulated_final_returns)

    # Example: 5th percentile return = bad-case outcome
    downside_percentile = np.percentile(simulated_final_returns, percentile)

    return downside_percentile, simulated_final_returns.mean()


# ----------------------------
# Entry / exit rules
# ----------------------------

def check_entry(row, regime: str, downside_p5: float) -> bool:
    """
    Entry conditions for a long-only strategy.
    """
    return (
        regime == "bullish" and
        row["close"] > row["ma20"] and
        row["momentum"] > 0 and
        row["vol_z"] > -0.5 and
        downside_p5 > -0.03   # reject if 5th percentile worse than -3%
    )


def check_exit(row, regime: str, entry_price: float, stop_loss: float, take_profit: float) -> bool:
    """
    Exit conditions.
    """
    current_return = (row["close"] / entry_price) - 1

    if regime in ["bearish", "choppy"]:
        return True

    if row["close"] < row["ma20"]:
        return True

    if current_return <= -stop_loss:
        return True

    if current_return >= take_profit:
        return True

    return False


# ----------------------------
# Position sizing
# ----------------------------

def position_size_by_risk(
    capital: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_pct: float
) -> float:
    """
    Risk-based sizing:
    risk amount = capital * risk_per_trade
    units = risk amount / (entry_price * stop_loss_pct)
    """
    risk_amount = capital * risk_per_trade
    unit_risk = entry_price * stop_loss_pct

    if unit_risk <= 0:
        return 0.0

    units = risk_amount / unit_risk
    return units


# ----------------------------
# Simple backtest loop
# ----------------------------

def run_backtest(df: pd.DataFrame):
    df = add_features(df)

    # Train HMM on all available data for demo purposes.
    # In real use, you should retrain periodically on rolling windows.
    model, df = train_hmm(df, n_states=3)
    state_map = label_states(df)

    capital = 10000.0
    cash = capital
    units = 0.0
    in_position = False
    entry_price = 0.0

    stop_loss_pct = 0.02     # 2%
    take_profit_pct = 0.04   # 4%
    risk_per_trade = 0.01    # risk 1% of capital each trade

    trade_log = []

    for i in range(30, len(df)):
        row = df.iloc[i]
        window = df.iloc[:i+1].copy()

        # Predict regime using data up to current row
        regime = predict_latest_regime(model, window, state_map)

        # Use recent returns for Monte Carlo
        recent_rets = window["log_ret"].tail(50)
        downside_p5, mc_mean = monte_carlo_downside_risk(
            current_price=row["close"],
            recent_log_returns=recent_rets,
            horizon=12,
            n_sims=500,
            percentile=5
        )

        if not in_position:
            if check_entry(row, regime, downside_p5):
                size = position_size_by_risk(
                    capital=cash,
                    risk_per_trade=risk_per_trade,
                    entry_price=row["close"],
                    stop_loss_pct=stop_loss_pct
                )

                cost = size * row["close"]

                # Safety cap: don't use more than 20% of cash
                max_cost = 0.20 * cash
                if cost > max_cost:
                    size = max_cost / row["close"]
                    cost = size * row["close"]

                if size > 0 and cost <= cash:
                    units = size
                    cash -= cost
                    entry_price = row["close"]
                    in_position = True

                    trade_log.append({
                        "type": "BUY",
                        "index": i,
                        "price": row["close"],
                        "regime": regime,
                        "mc_downside_p5": downside_p5,
                        "mc_mean": mc_mean,
                        "units": units,
                        "cash_after": cash
                    })

        else:
            if check_exit(row, regime, entry_price, stop_loss_pct, take_profit_pct):
                proceeds = units * row["close"]
                pnl = proceeds - (units * entry_price)

                cash += proceeds
                trade_log.append({
                    "type": "SELL",
                    "index": i,
                    "price": row["close"],
                    "regime": regime,
                    "pnl": pnl,
                    "cash_after": cash
                })

                units = 0.0
                entry_price = 0.0
                in_position = False

    # Mark-to-market if still holding
    final_value = cash
    if in_position:
        final_value += units * df.iloc[-1]["close"]

    return {
        "final_value": final_value,
        "return_pct": (final_value / capital - 1) * 100,
        "trade_log": trade_log,
        "state_map": state_map
    }