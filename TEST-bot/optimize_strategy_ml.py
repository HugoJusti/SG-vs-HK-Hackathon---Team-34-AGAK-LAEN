"""
Offline ML optimizer for the Binance strategy thresholds.

The script trains an adjustable-epoch MLPClassifier on engineered OHLCV features,
uses chronological train/test splits, plots train vs validation accuracy, and
searches the strategy parameter space against a weighted objective:

0.4 * Sortino + 0.3 * Sharpe + 0.3 * Calmar

To make the model learn a more meaningful market structure, the supervised target
is framed as a 3-class problem:
  - bullish: future move is materially positive
  - neutral: future move is inside a noise band
  - bearish: future move is materially negative

The MLP's bullish and bearish probabilities are then used as the confidence
signal during the offline threshold search.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
ANNUALIZATION_FACTOR = 24 * 365
EPSILON = 1e-12
RANDOM_SEED = 42
DEFAULT_EPOCHS = 120
DEFAULT_PATIENCE = 20
DEFAULT_LOOKBACK_WINDOW = 24
DEFAULT_MIN_MOVE_THRESHOLD = 0.006
DEFAULT_NEUTRAL_VOL_MULTIPLIER = 0.80
DEFAULT_STARTING_CAPITAL = 10_000.0
TARGET_CLASSES = np.array([-1, 0, 1], dtype=int)
TARGET_LABEL_NAMES = {
    -1: "bearish",
    0: "neutral",
    1: "bullish",
}

SEQUENCE_BASE_FEATURES = [
    "log_return",
    "return_3",
    "return_6",
    "return_12",
    "rolling_vol_6",
    "rolling_vol_20",
    "rolling_vol_48",
    "momentum_10",
    "rsi_14",
    "atr_pct_14",
    "macd_hist",
    "bollinger_z_20",
    "volume_zscore_20",
    "close_to_ma20",
    "close_to_ma50",
    "ma_gap_20_50",
    "intrabar_range_pct",
    "candle_body_pct",
]

DEFAULT_PARAMS = {
    "ENTRY_HMM_CONFIDENCE": 0.65,
    "ENTRY_MAX_VOLATILITY": 0.055,
    "ENTRY_MOMENTUM_MIN": 0.0,
    "ENTRY_VOLUME_ZSCORE_MAX": 2.0,
    "ENTRY_MA_SHORT": 20,
    "ENTRY_SPREAD_MAX_PCT": 0.05,
    "EXIT_HMM_CONFIDENCE": 0.60,
    "EXIT_MA_LONG": 50,
    "EXIT_STOP_LOSS_PCT": 0.03,
}

SEARCH_BOUNDS = {
    "ENTRY_HMM_CONFIDENCE": (0.20, 0.90),
    "ENTRY_MAX_VOLATILITY": (0.02, 0.20),
    "ENTRY_MOMENTUM_MIN": (-0.05, 0.05),
    "ENTRY_VOLUME_ZSCORE_MAX": (0.50, 5.00),
    "ENTRY_MA_SHORT": (5, 60),
    "ENTRY_SPREAD_MAX_PCT": (0.01, 0.20),
    "EXIT_HMM_CONFIDENCE": (0.15, 0.85),
    "EXIT_MA_LONG": (20, 140),
    "EXIT_STOP_LOSS_PCT": (0.005, 0.10),
}


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    train_full: pd.DataFrame
    test: pd.DataFrame


def parse_hidden_layers(raw_value: str) -> tuple[int, ...]:
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one hidden layer size is required.")
    return tuple(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an MLP on Binance OHLCV data and optimize strategy thresholds."
    )
    parser.add_argument(
        "--csv-path",
        default=str(BASE_DIR / "binance_data.csv"),
        help="Path to the historical Binance OHLCV CSV.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Chronological train/test split ratio.",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.20,
        help="Validation share taken from the end of the training split.",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=12,
        help="Future horizon in hours for the classifier target.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Maximum number of training epochs for the MLP. 120 is the recommended starting point.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Patience window used to flag when validation stopped improving. The plot still runs all requested epochs.",
    )
    parser.add_argument(
        "--hidden-layers",
        default="128,64,32",
        help="Comma-separated hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="L2 regularization strength for the MLP.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate for the MLP optimizer.",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=DEFAULT_LOOKBACK_WINDOW,
        help="How many hourly feature snapshots to flatten into each ML sample.",
    )
    parser.add_argument(
        "--min-move-threshold",
        type=float,
        default=DEFAULT_MIN_MOVE_THRESHOLD,
        help="Minimum future return magnitude used to label bullish or bearish moves.",
    )
    parser.add_argument(
        "--neutral-vol-multiplier",
        type=float,
        default=DEFAULT_NEUTRAL_VOL_MULTIPLIER,
        help="Volatility multiplier used to create the neutral-band threshold around zero return.",
    )
    parser.add_argument(
        "--search-iterations",
        type=int,
        default=300,
        help="Number of random-search candidates for threshold optimization.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=6,
        help="Minimum preferred number of closed trades before a penalty is applied.",
    )
    parser.add_argument(
        "--params-output",
        default=str(BASE_DIR / "optimized_strategy_params.json"),
        help="Where to save the optimized parameter JSON output.",
    )
    parser.add_argument(
        "--plot-output",
        default=str(BASE_DIR / "ml_accuracy.png"),
        help="Where to save the matplotlib training accuracy plot.",
    )
    parser.add_argument(
        "--model-output",
        default=str(BASE_DIR / "strategy_mlp_model.pkl"),
        help="Where to save the trained MLP model bundle.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--starting-capital",
        type=float,
        default=DEFAULT_STARTING_CAPITAL,
        help="Starting capital used for the backtest profit simulation summary.",
    )
    return parser.parse_args()


def clip_metric(value: float, limit: float = 10.0) -> float:
    if np.isnan(value):
        return 0.0
    return float(np.clip(value, -limit, limit))


def print_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_accuracy: float,
    train_balanced_accuracy: float,
    validation_accuracy: float,
    validation_balanced_accuracy: float,
    train_loss: float,
    validation_loss: float,
    optimizer_loss: float,
    best_validation_accuracy: float,
    best_validation_balanced_accuracy: float,
    wait: int,
    improved: bool,
    patience_reached: bool,
) -> None:
    status_flags: list[str] = []
    if improved:
        status_flags.append("BEST")
    if patience_reached:
        status_flags.append("PATIENCE")
    status = ",".join(status_flags) if status_flags else "-"

    print(
        f"Epoch {epoch:03d}/{total_epochs:03d} | "
        f"train_acc={train_accuracy:.4f} | "
        f"train_bal_acc={train_balanced_accuracy:.4f} | "
        f"val_acc={validation_accuracy:.4f} | "
        f"val_bal_acc={validation_balanced_accuracy:.4f} | "
        f"train_loss={train_loss:.5f} | "
        f"val_loss={validation_loss:.5f} | "
        f"fit_loss={optimizer_loss:.5f} | "
        f"best_val_acc={best_validation_accuracy:.4f} | "
        f"best_val_bal_acc={best_validation_balanced_accuracy:.4f} | "
        f"wait={wait:02d} | "
        f"status={status}"
    )


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + EPSILON)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    high_low = frame["high"] - frame["low"]
    high_close = (frame["high"] - previous_close).abs()
    low_close = (frame["low"] - previous_close).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=window).mean()


def build_lagged_feature_columns(
    frame: pd.DataFrame,
    base_features: list[str],
    lookback_window: int,
) -> tuple[pd.DataFrame, list[str]]:
    if lookback_window < 1:
        raise ValueError("lookback_window must be at least 1.")

    enriched = frame.copy()
    feature_columns: list[str] = []

    for lag in range(lookback_window):
        lagged = enriched[base_features].shift(lag)
        renamed = {column: f"{column}_lag_{lag}" for column in base_features}
        lagged = lagged.rename(columns=renamed)
        enriched = pd.concat([enriched, lagged], axis=1)
        feature_columns.extend(renamed.values())

    return enriched, feature_columns


def summarise_class_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["target"].value_counts().to_dict()
    return {
        TARGET_LABEL_NAMES[label]: int(counts.get(label, 0))
        for label in TARGET_CLASSES
    }


def load_feature_frame(
    csv_path: Path,
    target_horizon: int,
    lookback_window: int,
    min_move_threshold: float,
    neutral_vol_multiplier: float,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    frame = pd.read_csv(csv_path, parse_dates=["open_time"])
    frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)

    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["return_3"] = frame["close"].pct_change(periods=3)
    frame["return_6"] = frame["close"].pct_change(periods=6)
    frame["return_12"] = frame["close"].pct_change(periods=12)
    frame["rolling_vol_6"] = frame["log_return"].rolling(window=6).std() * np.sqrt(ANNUALIZATION_FACTOR)
    frame["rolling_vol_20"] = frame["log_return"].rolling(window=20).std() * np.sqrt(ANNUALIZATION_FACTOR)
    frame["rolling_vol_48"] = frame["log_return"].rolling(window=48).std() * np.sqrt(ANNUALIZATION_FACTOR)
    frame["momentum_10"] = frame["close"].pct_change(periods=10)

    volume_mean = frame["volume"].rolling(window=20).mean()
    volume_std = frame["volume"].rolling(window=20).std()
    frame["volume_zscore_20"] = (frame["volume"] - volume_mean) / (volume_std + EPSILON)

    ma20 = frame["close"].rolling(window=20).mean()
    ma50 = frame["close"].rolling(window=50).mean()
    frame["ma_gap_20_50"] = ma20 / (ma50 + EPSILON) - 1.0
    frame["close_to_ma20"] = frame["close"] / ma20 - 1.0
    frame["close_to_ma50"] = frame["close"] / ma50 - 1.0
    frame["intrabar_range_pct"] = (frame["high"] - frame["low"]) / frame["close"]
    frame["volume_change_1"] = frame["volume"].pct_change()
    frame["candle_body_pct"] = (frame["close"] - frame["open"]).abs() / frame["close"]

    rolling_mean_20 = frame["close"].rolling(window=20).mean()
    rolling_std_20 = frame["close"].rolling(window=20).std()
    frame["bollinger_z_20"] = (frame["close"] - rolling_mean_20) / (rolling_std_20 + EPSILON)

    ema12 = frame["close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = macd - macd_signal

    frame["rsi_14"] = (compute_rsi(frame["close"], window=14) - 50.0) / 50.0
    frame["atr_pct_14"] = compute_atr(frame, window=14) / frame["close"]

    # The CSV has no bid/ask spread column, so use 5% of the recent candle range
    # as a conservative spread proxy expressed in percentage points.
    spread_proxy = frame["intrabar_range_pct"].rolling(window=3, min_periods=1).median()
    frame["estimated_spread_pct"] = spread_proxy * 100.0 * 0.05

    frame["future_return"] = frame["close"].shift(-target_horizon) / frame["close"] - 1.0
    hourly_volatility = frame["rolling_vol_20"] / np.sqrt(ANNUALIZATION_FACTOR)
    future_move_threshold = np.maximum(
        min_move_threshold,
        hourly_volatility * np.sqrt(target_horizon) * neutral_vol_multiplier,
    )
    frame["future_move_threshold"] = future_move_threshold
    frame["target"] = np.select(
        [
            frame["future_return"] > frame["future_move_threshold"],
            frame["future_return"] < -frame["future_move_threshold"],
        ],
        [1, -1],
        default=0,
    )

    frame, model_features = build_lagged_feature_columns(
        frame=frame,
        base_features=SEQUENCE_BASE_FEATURES,
        lookback_window=lookback_window,
    )

    required_columns = model_features + [
        "future_return",
        "future_move_threshold",
        "target",
        "rolling_vol_20",
        "momentum_10",
        "volume_zscore_20",
        "estimated_spread_pct",
    ]
    frame = frame.dropna(subset=required_columns).reset_index(drop=True)
    if frame.empty:
        raise ValueError("Feature engineering produced no usable rows.")

    class_distribution = summarise_class_distribution(frame)
    if any(count == 0 for count in class_distribution.values()):
        raise ValueError(
            f"Target classes are not all present after feature engineering: {class_distribution}"
        )

    return frame, model_features, class_distribution


def build_splits(frame: pd.DataFrame, train_ratio: float, validation_ratio: float) -> SplitData:
    if not 0.5 < train_ratio < 0.95:
        raise ValueError("train_ratio must be between 0.5 and 0.95.")
    if not 0.05 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0.05 and 0.5.")

    split_index = int(len(frame) * train_ratio)
    train_full = frame.iloc[:split_index].copy()
    test = frame.iloc[split_index:].copy()

    validation_index = int(len(train_full) * (1.0 - validation_ratio))
    train = train_full.iloc[:validation_index].copy()
    validation = train_full.iloc[validation_index:].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("Split configuration produced an empty train, validation, or test segment.")

    return SplitData(train=train, validation=validation, train_full=train_full, test=test)


def rebalance_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    class_indices: dict[int, np.ndarray] = {
        int(label): np.flatnonzero(y_train == label)
        for label in TARGET_CLASSES
    }

    missing = [
        TARGET_LABEL_NAMES[label]
        for label, indices in class_indices.items()
        if len(indices) == 0
    ]
    if missing:
        raise ValueError(f"Training split is missing target classes: {', '.join(missing)}")

    max_count = max(len(indices) for indices in class_indices.values())
    balanced_indices = []

    for label in TARGET_CLASSES:
        indices = class_indices[int(label)]
        sampled = rng.choice(indices, size=max_count, replace=True)
        balanced_indices.append(sampled)

    balanced_indices = np.concatenate(balanced_indices)
    rng.shuffle(balanced_indices)

    class_distribution = {
        TARGET_LABEL_NAMES[int(label)]: int(len(class_indices[int(label)]))
        for label in TARGET_CLASSES
    }
    return X_train[balanced_indices], y_train[balanced_indices], class_distribution


def compute_actionable_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not np.any(mask):
        return float("nan")
    return float(accuracy_score(y_true[mask], y_pred[mask]))


def fit_for_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classes: np.ndarray,
    hidden_layers: tuple[int, ...],
    alpha: float,
    learning_rate: float,
    epochs: int,
    seed: int,
) -> MLPClassifier:
    X_balanced_seed, _, _ = rebalance_training_data(X_train=X_train, y_train=y_train, seed=seed)
    batch_size = min(256, max(32, len(X_balanced_seed) // 10))
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        random_state=seed,
        shuffle=True,
        max_iter=1,
    )

    for epoch in range(epochs):
        X_epoch, y_epoch, _ = rebalance_training_data(
            X_train=X_train,
            y_train=y_train,
            seed=seed + epoch,
        )
        if epoch == 0:
            model.partial_fit(X_epoch, y_epoch, classes=classes)
        else:
            model.partial_fit(X_epoch, y_epoch)

    return model


def train_mlp_classifier(
    split_data: SplitData,
    model_features: list[str],
    hidden_layers: tuple[int, ...],
    alpha: float,
    learning_rate: float,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[StandardScaler, MLPClassifier, list[dict[str, float]], dict[str, float]]:
    X_train = split_data.train[model_features].to_numpy(dtype=float)
    y_train = split_data.train["target"].to_numpy(dtype=int)
    X_validation = split_data.validation[model_features].to_numpy(dtype=float)
    y_validation = split_data.validation["target"].to_numpy(dtype=int)

    scaler = StandardScaler()
    X_train_scaled_unbalanced = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_train_scaled_seed, _, training_class_distribution = rebalance_training_data(
        X_train=X_train_scaled_unbalanced,
        y_train=y_train,
        seed=seed,
    )

    batch_size = min(256, max(32, len(X_train_scaled_seed) // 10))
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate,
        random_state=seed,
        shuffle=True,
        max_iter=1,
    )

    classes = TARGET_CLASSES.copy()
    history: list[dict[str, float]] = []
    best_epoch = 1
    best_validation_accuracy = -np.inf
    best_validation_balanced_accuracy = -np.inf
    best_checkpoint_found = False
    wait = 0
    early_stop_epoch = None

    for epoch in range(1, epochs + 1):
        X_train_epoch, y_train_epoch, _ = rebalance_training_data(
            X_train=X_train_scaled_unbalanced,
            y_train=y_train,
            seed=seed + epoch,
        )
        if epoch == 1:
            model.partial_fit(X_train_epoch, y_train_epoch, classes=classes)
        else:
            model.partial_fit(X_train_epoch, y_train_epoch)

        train_probabilities = model.predict_proba(X_train_scaled_unbalanced)
        validation_probabilities = model.predict_proba(X_validation_scaled)
        train_predictions = model.classes_[np.argmax(train_probabilities, axis=1)]
        validation_predictions = model.classes_[np.argmax(validation_probabilities, axis=1)]

        train_accuracy = accuracy_score(y_train, train_predictions)
        train_balanced_accuracy = balanced_accuracy_score(y_train, train_predictions)
        validation_accuracy = accuracy_score(y_validation, validation_predictions)
        validation_balanced_accuracy = balanced_accuracy_score(y_validation, validation_predictions)
        train_loss = log_loss(y_train, train_probabilities, labels=classes.tolist())
        validation_loss = log_loss(y_validation, validation_probabilities, labels=classes.tolist())
        optimizer_loss = float(model.loss_)
        improved = (
            validation_balanced_accuracy > best_validation_balanced_accuracy + 1e-6
            or (
                abs(validation_balanced_accuracy - best_validation_balanced_accuracy) <= 1e-6
                and validation_accuracy > best_validation_accuracy + 1e-6
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_accuracy": float(train_accuracy),
                "train_balanced_accuracy": float(train_balanced_accuracy),
                "validation_accuracy": float(validation_accuracy),
                "validation_balanced_accuracy": float(validation_balanced_accuracy),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "optimizer_loss": optimizer_loss,
            }
        )

        if improved:
            best_validation_accuracy = float(validation_accuracy)
            best_validation_balanced_accuracy = float(validation_balanced_accuracy)
            best_epoch = epoch
            best_checkpoint_found = True
            wait = 0
        else:
            wait += 1

        if wait >= patience and early_stop_epoch is None:
            early_stop_epoch = epoch

        print_epoch_metrics(
            epoch=epoch,
            total_epochs=epochs,
            train_accuracy=float(train_accuracy),
            train_balanced_accuracy=float(train_balanced_accuracy),
            validation_accuracy=float(validation_accuracy),
            validation_balanced_accuracy=float(validation_balanced_accuracy),
            train_loss=float(train_loss),
            validation_loss=float(validation_loss),
            optimizer_loss=optimizer_loss,
            best_validation_accuracy=float(best_validation_accuracy),
            best_validation_balanced_accuracy=float(best_validation_balanced_accuracy),
            wait=wait,
            improved=improved,
            patience_reached=early_stop_epoch == epoch,
        )

    if not best_checkpoint_found:
        raise RuntimeError("MLP training failed to produce a usable model.")

    scaler_full = StandardScaler()
    X_train_full_unbalanced = scaler_full.fit_transform(split_data.train_full[model_features].to_numpy(dtype=float))
    y_train_full = split_data.train_full["target"].to_numpy(dtype=int)
    _, _, train_full_class_distribution = rebalance_training_data(
        X_train=X_train_full_unbalanced,
        y_train=y_train_full,
        seed=seed + 1,
    )
    final_model = fit_for_epochs(
        X_train=X_train_full_unbalanced,
        y_train=y_train_full,
        classes=classes,
        hidden_layers=hidden_layers,
        alpha=alpha,
        learning_rate=learning_rate,
        epochs=best_epoch,
        seed=seed,
    )

    full_train_probabilities = final_model.predict_proba(X_train_full_unbalanced)
    full_train_predictions = final_model.classes_[np.argmax(full_train_probabilities, axis=1)]
    full_train_accuracy = accuracy_score(y_train_full, full_train_predictions)
    full_train_balanced_accuracy = balanced_accuracy_score(y_train_full, full_train_predictions)
    full_train_actionable_accuracy = compute_actionable_accuracy(y_train_full, full_train_predictions)

    X_test = scaler_full.transform(split_data.test[model_features].to_numpy(dtype=float))
    y_test = split_data.test["target"].to_numpy(dtype=int)
    test_probabilities = final_model.predict_proba(X_test)
    test_predictions = final_model.classes_[np.argmax(test_probabilities, axis=1)]
    test_accuracy = accuracy_score(y_test, test_predictions)
    test_balanced_accuracy = balanced_accuracy_score(y_test, test_predictions)
    test_actionable_accuracy = compute_actionable_accuracy(y_test, test_predictions)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_validation_accuracy": float(best_validation_accuracy),
        "full_train_accuracy": float(full_train_accuracy),
        "full_train_balanced_accuracy": float(full_train_balanced_accuracy),
        "full_train_actionable_accuracy": full_train_actionable_accuracy,
        "test_accuracy": float(test_accuracy),
        "test_balanced_accuracy": float(test_balanced_accuracy),
        "test_actionable_accuracy": test_actionable_accuracy,
        "selection_train_accuracy": float(history[best_epoch - 1]["train_accuracy"]),
        "selection_train_balanced_accuracy": float(history[best_epoch - 1]["train_balanced_accuracy"]),
        "best_validation_balanced_accuracy": float(history[best_epoch - 1]["validation_balanced_accuracy"]),
        "early_stop_epoch": None if early_stop_epoch is None else int(early_stop_epoch),
        "training_class_distribution": training_class_distribution,
        "train_full_class_distribution": train_full_class_distribution,
    }

    return scaler_full, final_model, history, metrics


def plot_accuracy_history(
    history: list[dict[str, float]],
    best_epoch: int,
    test_accuracy: float,
    test_balanced_accuracy: float,
    early_stop_epoch: int | None,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    validation_accuracy = [row["validation_accuracy"] for row in history]
    train_balanced_accuracy = [row["train_balanced_accuracy"] for row in history]
    validation_balanced_accuracy = [row["validation_balanced_accuracy"] for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracy, label="Train accuracy", linewidth=2.0, color="#1d4ed8")
    plt.plot(epochs, validation_accuracy, label="Validation accuracy", linewidth=2.0, color="#15803d")
    plt.plot(
        epochs,
        train_balanced_accuracy,
        label="Train balanced accuracy",
        linewidth=1.5,
        linestyle="--",
        color="#60a5fa",
    )
    plt.plot(
        epochs,
        validation_balanced_accuracy,
        label="Validation balanced accuracy",
        linewidth=1.5,
        linestyle="--",
        color="#4ade80",
    )
    plt.axvline(best_epoch, color="#b45309", linestyle="--", linewidth=1.5, label=f"Best epoch = {best_epoch}")
    if early_stop_epoch is not None:
        plt.axvline(
            early_stop_epoch,
            color="#dc2626",
            linestyle="-.",
            linewidth=1.2,
            label=f"Patience threshold = {early_stop_epoch}",
        )
    plt.axhline(test_accuracy, color="#7c3aed", linestyle=":", linewidth=1.5, label=f"Test accuracy = {test_accuracy:.3f}")
    plt.axhline(
        test_balanced_accuracy,
        color="#f97316",
        linestyle=":",
        linewidth=1.5,
        label=f"Test balanced accuracy = {test_balanced_accuracy:.3f}",
    )
    plt.title("MLP Accuracy Across Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def attach_probabilities(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    model: MLPClassifier,
    model_features: list[str],
) -> pd.DataFrame:
    features = scaler.transform(frame[model_features].to_numpy(dtype=float))
    probabilities = model.predict_proba(features)
    class_to_index = {
        int(label): index
        for index, label in enumerate(model.classes_)
    }

    enriched = frame.copy()
    enriched["bearish_prob"] = probabilities[:, class_to_index[-1]]
    enriched["neutral_prob"] = probabilities[:, class_to_index[0]]
    enriched["bullish_prob"] = probabilities[:, class_to_index[1]]
    enriched["predicted_regime"] = model.classes_[np.argmax(probabilities, axis=1)]
    return enriched


def get_cached_ma(frame: pd.DataFrame, window: int, cache: dict[int, np.ndarray]) -> np.ndarray:
    if window not in cache:
        cache[window] = frame["close"].rolling(window=window).mean().to_numpy(dtype=float)
    return cache[window]


def compute_risk_metrics(returns: np.ndarray) -> dict[str, float]:
    if len(returns) == 0:
        return {
            "objective_raw": -999.0,
            "sortino_ratio": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
        }

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=0))
    downside = np.minimum(returns, 0.0)
    downside_std = float(np.sqrt(np.mean(np.square(downside))))

    sharpe_ratio = 0.0 if std_return <= EPSILON else mean_return / std_return * np.sqrt(ANNUALIZATION_FACTOR)
    sortino_ratio = 0.0 if downside_std <= EPSILON else mean_return / downside_std * np.sqrt(ANNUALIZATION_FACTOR)

    equity_curve = np.cumprod(np.clip(1.0 + returns, 0.001, None))
    rolling_peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / np.maximum(rolling_peak, EPSILON) - 1.0
    max_drawdown = float(np.min(drawdowns))
    total_return = float(equity_curve[-1] - 1.0)

    if equity_curve[-1] <= EPSILON:
        annualized_return = -1.0
    else:
        annualized_return = float(equity_curve[-1] ** (ANNUALIZATION_FACTOR / len(returns)) - 1.0)

    if max_drawdown < -EPSILON:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = annualized_return if annualized_return > 0 else 0.0

    sharpe_ratio = clip_metric(sharpe_ratio)
    sortino_ratio = clip_metric(sortino_ratio)
    calmar_ratio = clip_metric(calmar_ratio)

    objective_raw = 0.4 * sortino_ratio + 0.3 * sharpe_ratio + 0.3 * calmar_ratio
    return {
        "objective_raw": float(objective_raw),
        "sortino_ratio": float(sortino_ratio),
        "sharpe_ratio": float(sharpe_ratio),
        "calmar_ratio": float(calmar_ratio),
        "annualized_return": float(annualized_return),
        "max_drawdown": float(max_drawdown),
        "total_return": float(total_return),
    }


def build_simulation_summary(
    metrics: dict[str, float],
    starting_capital: float,
) -> dict[str, float]:
    starting_capital = float(starting_capital)
    total_return = float(metrics["total_return"])
    ending_capital = starting_capital * (1.0 + total_return)
    net_profit = ending_capital - starting_capital

    return {
        "starting_capital": starting_capital,
        "ending_capital": float(ending_capital),
        "net_profit": float(net_profit),
        "profit_pct": float(total_return * 100.0),
        "annualized_return_pct": float(metrics["annualized_return"] * 100.0),
        "max_drawdown_pct": float(metrics["max_drawdown"] * 100.0),
        "objective": float(metrics["objective"]),
        "objective_raw": float(metrics["objective_raw"]),
        "sortino_ratio": float(metrics["sortino_ratio"]),
        "sharpe_ratio": float(metrics["sharpe_ratio"]),
        "calmar_ratio": float(metrics["calmar_ratio"]),
        "closed_trades": int(metrics["closed_trades"]),
        "win_rate_pct": float(metrics["win_rate"] * 100.0),
        "avg_trade_return_pct": float(metrics["avg_trade_return"] * 100.0),
        "avg_holding_hours": float(metrics["avg_holding_hours"]),
        "exposure_ratio_pct": float(metrics["exposure_ratio"] * 100.0),
        "trade_penalty": float(metrics["trade_penalty"]),
    }


def build_simulation_report(
    optimization_results: dict[str, object],
    starting_capital: float,
) -> dict[str, dict[str, float]]:
    return {
        "optimized_train": build_simulation_summary(
            optimization_results["best_train_metrics"],
            starting_capital,
        ),
        "optimized_test": build_simulation_summary(
            optimization_results["best_test_metrics"],
            starting_capital,
        ),
        "default_train": build_simulation_summary(
            optimization_results["default_train_metrics"],
            starting_capital,
        ),
        "default_test": build_simulation_summary(
            optimization_results["default_test_metrics"],
            starting_capital,
        ),
    }


def print_simulation_summary(title: str, summary: dict[str, float]) -> None:
    print(title)
    print(f"  Starting capital: ${summary['starting_capital']:,.2f}")
    print(f"  Ending capital:   ${summary['ending_capital']:,.2f}")
    print(f"  Net profit:       ${summary['net_profit']:,.2f}")
    print(f"  Profit:           {summary['profit_pct']:.2f}%")
    print(f"  Annualized return:{summary['annualized_return_pct']:.2f}%")
    print(f"  Max drawdown:     {summary['max_drawdown_pct']:.2f}%")
    print(f"  Sortino/Sharpe/Calmar: {summary['sortino_ratio']:.4f} / {summary['sharpe_ratio']:.4f} / {summary['calmar_ratio']:.4f}")
    print(f"  Closed trades:    {summary['closed_trades']}")
    print(f"  Win rate:         {summary['win_rate_pct']:.2f}%")
    print(f"  Avg trade return: {summary['avg_trade_return_pct']:.2f}%")
    print(f"  Avg holding time: {summary['avg_holding_hours']:.2f}h")
    print(f"  Exposure ratio:   {summary['exposure_ratio_pct']:.2f}%")


def backtest_strategy(
    frame: pd.DataFrame,
    params: dict[str, float],
    min_trades: int,
    ma_cache: dict[int, np.ndarray],
) -> dict[str, float]:
    close = frame["close"].to_numpy(dtype=float)
    bullish_prob = frame["bullish_prob"].to_numpy(dtype=float)
    bearish_prob = frame["bearish_prob"].to_numpy(dtype=float)
    rolling_vol = frame["rolling_vol_20"].to_numpy(dtype=float)
    momentum = frame["momentum_10"].to_numpy(dtype=float)
    volume_zscore = frame["volume_zscore_20"].to_numpy(dtype=float)
    spread_pct = frame["estimated_spread_pct"].to_numpy(dtype=float)

    ma_short = get_cached_ma(frame, int(params["ENTRY_MA_SHORT"]), ma_cache)
    ma_long = get_cached_ma(frame, int(params["EXIT_MA_LONG"]), ma_cache)

    hourly_returns = np.zeros(len(frame), dtype=float)
    trade_returns: list[float] = []
    holding_hours: list[int] = []

    in_position = False
    exited_this_bar = False
    entry_price = 0.0
    entry_cost = 0.0
    entry_index = -1
    exposure_hours = 0

    for idx in range(len(frame) - 1):
        exited_this_bar = False
        if np.isnan(ma_short[idx]) or np.isnan(ma_long[idx]):
            continue

        if in_position:
            drawdown = 0.0 if entry_price <= 0 else (entry_price - close[idx]) / entry_price
            should_exit = (
                drawdown >= params["EXIT_STOP_LOSS_PCT"]
                or close[idx] < ma_long[idx]
                or bearish_prob[idx] > params["EXIT_HMM_CONFIDENCE"]
            )
            if should_exit:
                exit_cost = spread_pct[idx] / 100.0 / 2.0
                hourly_returns[idx] -= exit_cost
                trade_return = (close[idx] / entry_price - 1.0) - entry_cost - exit_cost
                trade_returns.append(float(trade_return))
                holding_hours.append(max(1, idx - entry_index))
                in_position = False
                exited_this_bar = True
                entry_price = 0.0
                entry_cost = 0.0
                entry_index = -1

        if not in_position and not exited_this_bar:
            should_enter = (
                bullish_prob[idx] > params["ENTRY_HMM_CONFIDENCE"]
                and rolling_vol[idx] < params["ENTRY_MAX_VOLATILITY"]
                and momentum[idx] > params["ENTRY_MOMENTUM_MIN"]
                and abs(volume_zscore[idx]) < params["ENTRY_VOLUME_ZSCORE_MAX"]
                and close[idx] >= ma_short[idx]
                and spread_pct[idx] < params["ENTRY_SPREAD_MAX_PCT"]
            )
            if should_enter:
                in_position = True
                entry_price = close[idx]
                entry_cost = spread_pct[idx] / 100.0 / 2.0
                hourly_returns[idx] -= entry_cost
                entry_index = idx

        if in_position:
            hourly_returns[idx + 1] += close[idx + 1] / close[idx] - 1.0
            exposure_hours += 1

    if in_position and entry_price > 0:
        exit_cost = spread_pct[-1] / 100.0 / 2.0
        hourly_returns[-1] -= exit_cost
        trade_return = (close[-1] / entry_price - 1.0) - entry_cost - exit_cost
        trade_returns.append(float(trade_return))
        holding_hours.append(max(1, len(frame) - 1 - entry_index))

    metrics = compute_risk_metrics(hourly_returns)
    closed_trades = len(trade_returns)
    trade_penalty = max(0, min_trades - closed_trades) * 0.15

    metrics["objective"] = float(metrics["objective_raw"] - trade_penalty)
    metrics["trade_penalty"] = float(trade_penalty)
    metrics["closed_trades"] = int(closed_trades)
    metrics["win_rate"] = float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0
    metrics["avg_trade_return"] = float(np.mean(trade_returns)) if trade_returns else 0.0
    metrics["avg_holding_hours"] = float(np.mean(holding_hours)) if holding_hours else 0.0
    metrics["exposure_ratio"] = float(exposure_hours / max(len(frame) - 1, 1))
    return metrics


def sample_float(
    rng: np.random.Generator,
    low: float,
    high: float,
    center: float,
    bias_scale: float = 0.20,
) -> float:
    if rng.random() < 0.40:
        return float(rng.uniform(low, high))
    std = max((high - low) * bias_scale, EPSILON)
    return float(np.clip(rng.normal(center, std), low, high))


def sample_int(
    rng: np.random.Generator,
    low: int,
    high: int,
    center: int,
    bias_scale: float = 0.20,
) -> int:
    if rng.random() < 0.40:
        return int(rng.integers(low, high + 1))
    std = max((high - low) * bias_scale, 1.0)
    value = int(round(rng.normal(center, std)))
    return int(np.clip(value, low, high))


def sample_candidates(
    iterations: int,
    seed: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    candidates = [DEFAULT_PARAMS.copy()]
    seen = {
        tuple(DEFAULT_PARAMS[key] for key in DEFAULT_PARAMS.keys())
    }

    while len(candidates) < iterations:
        candidate = {
            "ENTRY_HMM_CONFIDENCE": sample_float(
                rng,
                *SEARCH_BOUNDS["ENTRY_HMM_CONFIDENCE"],
                DEFAULT_PARAMS["ENTRY_HMM_CONFIDENCE"],
            ),
            "ENTRY_MAX_VOLATILITY": sample_float(
                rng,
                *SEARCH_BOUNDS["ENTRY_MAX_VOLATILITY"],
                DEFAULT_PARAMS["ENTRY_MAX_VOLATILITY"],
            ),
            "ENTRY_MOMENTUM_MIN": sample_float(
                rng,
                *SEARCH_BOUNDS["ENTRY_MOMENTUM_MIN"],
                DEFAULT_PARAMS["ENTRY_MOMENTUM_MIN"],
                bias_scale=0.15,
            ),
            "ENTRY_VOLUME_ZSCORE_MAX": sample_float(
                rng,
                *SEARCH_BOUNDS["ENTRY_VOLUME_ZSCORE_MAX"],
                DEFAULT_PARAMS["ENTRY_VOLUME_ZSCORE_MAX"],
            ),
            "ENTRY_MA_SHORT": sample_int(
                rng,
                *SEARCH_BOUNDS["ENTRY_MA_SHORT"],
                DEFAULT_PARAMS["ENTRY_MA_SHORT"],
            ),
            "ENTRY_SPREAD_MAX_PCT": sample_float(
                rng,
                *SEARCH_BOUNDS["ENTRY_SPREAD_MAX_PCT"],
                DEFAULT_PARAMS["ENTRY_SPREAD_MAX_PCT"],
                bias_scale=0.12,
            ),
            "EXIT_HMM_CONFIDENCE": sample_float(
                rng,
                *SEARCH_BOUNDS["EXIT_HMM_CONFIDENCE"],
                DEFAULT_PARAMS["EXIT_HMM_CONFIDENCE"],
            ),
            "EXIT_STOP_LOSS_PCT": sample_float(
                rng,
                *SEARCH_BOUNDS["EXIT_STOP_LOSS_PCT"],
                DEFAULT_PARAMS["EXIT_STOP_LOSS_PCT"],
                bias_scale=0.15,
            ),
        }

        min_exit_ma = max(candidate["ENTRY_MA_SHORT"] + 5, SEARCH_BOUNDS["EXIT_MA_LONG"][0])
        candidate["EXIT_MA_LONG"] = sample_int(
            rng,
            min_exit_ma,
            SEARCH_BOUNDS["EXIT_MA_LONG"][1],
            max(DEFAULT_PARAMS["EXIT_MA_LONG"], min_exit_ma),
        )

        candidate["ENTRY_HMM_CONFIDENCE"] = round(candidate["ENTRY_HMM_CONFIDENCE"], 4)
        candidate["ENTRY_MAX_VOLATILITY"] = round(candidate["ENTRY_MAX_VOLATILITY"], 4)
        candidate["ENTRY_MOMENTUM_MIN"] = round(candidate["ENTRY_MOMENTUM_MIN"], 4)
        candidate["ENTRY_VOLUME_ZSCORE_MAX"] = round(candidate["ENTRY_VOLUME_ZSCORE_MAX"], 4)
        candidate["ENTRY_SPREAD_MAX_PCT"] = round(candidate["ENTRY_SPREAD_MAX_PCT"], 4)
        candidate["EXIT_HMM_CONFIDENCE"] = round(candidate["EXIT_HMM_CONFIDENCE"], 4)
        candidate["EXIT_STOP_LOSS_PCT"] = round(candidate["EXIT_STOP_LOSS_PCT"], 4)

        candidate_key = tuple(candidate[key] for key in DEFAULT_PARAMS.keys())
        if candidate_key in seen:
            continue

        seen.add(candidate_key)
        candidates.append(candidate)

    return candidates


def optimize_parameters(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    iterations: int,
    min_trades: int,
    seed: int,
) -> dict[str, object]:
    candidates = sample_candidates(iterations=iterations, seed=seed)
    train_ma_cache: dict[int, np.ndarray] = {}
    test_ma_cache: dict[int, np.ndarray] = {}

    best_params = None
    best_train_metrics = None

    for candidate in candidates:
        train_metrics = backtest_strategy(
            frame=train_frame,
            params=candidate,
            min_trades=min_trades,
            ma_cache=train_ma_cache,
        )
        if best_train_metrics is None or train_metrics["objective"] > best_train_metrics["objective"]:
            best_params = candidate
            best_train_metrics = train_metrics

    if best_params is None or best_train_metrics is None:
        raise RuntimeError("Parameter search failed to find a usable candidate.")

    best_test_metrics = backtest_strategy(
        frame=test_frame,
        params=best_params,
        min_trades=min_trades,
        ma_cache=test_ma_cache,
    )
    default_train_metrics = backtest_strategy(
        frame=train_frame,
        params=DEFAULT_PARAMS,
        min_trades=min_trades,
        ma_cache=train_ma_cache,
    )
    default_test_metrics = backtest_strategy(
        frame=test_frame,
        params=DEFAULT_PARAMS,
        min_trades=min_trades,
        ma_cache=test_ma_cache,
    )

    return {
        "best_parameters": best_params,
        "best_train_metrics": best_train_metrics,
        "best_test_metrics": best_test_metrics,
        "default_train_metrics": default_train_metrics,
        "default_test_metrics": default_test_metrics,
        "search_candidates_evaluated": len(candidates),
    }


def save_model_bundle(
    output_path: Path,
    scaler: StandardScaler,
    model: MLPClassifier,
    model_features: list[str],
    best_epoch: int,
    target_horizon: int,
    hidden_layers: Iterable[int],
    lookback_window: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scaler": scaler,
        "model": model,
        "feature_columns": model_features,
        "class_labels": [int(label) for label in model.classes_],
        "best_epoch": int(best_epoch),
        "target_horizon_hours": int(target_horizon),
        "hidden_layers": list(hidden_layers),
        "lookback_window": int(lookback_window),
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_parameter_report(
    output_path: Path,
    csv_path: Path,
    frame: pd.DataFrame,
    split_data: SplitData,
    model_features: list[str],
    class_distribution: dict[str, int],
    ml_metrics: dict[str, float],
    optimization_results: dict[str, object],
    hidden_layers: tuple[int, ...],
    epochs_requested: int,
    patience: int,
    target_horizon: int,
    lookback_window: int,
    min_move_threshold: float,
    neutral_vol_multiplier: float,
    starting_capital: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_report = build_simulation_report(
        optimization_results=optimization_results,
        starting_capital=starting_capital,
    )
    report = {
        "dataset": {
            "csv_path": str(csv_path),
            "rows_after_feature_engineering": int(len(frame)),
            "train_rows": int(len(split_data.train_full)),
            "test_rows": int(len(split_data.test)),
            "target_horizon_hours": int(target_horizon),
            "lookback_window_hours": int(lookback_window),
            "overall_target_distribution": class_distribution,
            "train_target_distribution": summarise_class_distribution(split_data.train_full),
            "validation_target_distribution": summarise_class_distribution(split_data.validation),
            "test_target_distribution": summarise_class_distribution(split_data.test),
        },
        "model": {
            "type": "MLPClassifier",
            "feature_count": int(len(model_features)),
            "base_sequence_features": SEQUENCE_BASE_FEATURES,
            "hidden_layers": list(hidden_layers),
            "epochs_requested": int(epochs_requested),
            "best_epoch": int(ml_metrics["best_epoch"]),
            "early_stop_epoch": ml_metrics["early_stop_epoch"],
            "patience": int(patience),
            "selection_train_accuracy": float(ml_metrics["selection_train_accuracy"]),
            "selection_train_balanced_accuracy": float(ml_metrics["selection_train_balanced_accuracy"]),
            "best_validation_accuracy": float(ml_metrics["best_validation_accuracy"]),
            "best_validation_balanced_accuracy": float(ml_metrics["best_validation_balanced_accuracy"]),
            "full_train_accuracy": float(ml_metrics["full_train_accuracy"]),
            "full_train_balanced_accuracy": float(ml_metrics["full_train_balanced_accuracy"]),
            "full_train_actionable_accuracy": ml_metrics["full_train_actionable_accuracy"],
            "test_accuracy": float(ml_metrics["test_accuracy"]),
            "test_balanced_accuracy": float(ml_metrics["test_balanced_accuracy"]),
            "test_actionable_accuracy": ml_metrics["test_actionable_accuracy"],
            "training_class_distribution": ml_metrics["training_class_distribution"],
            "train_full_class_distribution": ml_metrics["train_full_class_distribution"],
        },
        "objective_weights": {
            "sortino_ratio": 0.4,
            "sharpe_ratio": 0.3,
            "calmar_ratio": 0.3,
        },
        "target_definition": {
            "classes": TARGET_LABEL_NAMES,
            "minimum_move_threshold": float(min_move_threshold),
            "neutral_volatility_multiplier": float(neutral_vol_multiplier),
            "rule": (
                "bullish if future_return > max(min_move_threshold, hourly_vol * sqrt(horizon) * neutral_volatility_multiplier); "
                "bearish if future_return < -that threshold; otherwise neutral"
            ),
        },
        "assumptions": {
            "confidence_source": (
                "ENTRY_HMM_CONFIDENCE and EXIT_HMM_CONFIDENCE are optimized against "
                "MLP bullish and bearish class probabilities from a 3-class bull/neutral/bear model."
            ),
            "spread_proxy": (
                "The CSV has no bid/ask spread, so estimated_spread_pct uses 5% of the "
                "3-candle median intrabar range in percentage points."
            ),
            "positioning": "Long-only, one position at a time, using close-to-close returns.",
            "cost_model": "Half of the synthetic spread is charged on entry and exit.",
        },
        "default_parameters": DEFAULT_PARAMS,
        "optimized_parameters": optimization_results["best_parameters"],
        "optimized_train_metrics": optimization_results["best_train_metrics"],
        "optimized_test_metrics": optimization_results["best_test_metrics"],
        "default_train_metrics": optimization_results["default_train_metrics"],
        "default_test_metrics": optimization_results["default_test_metrics"],
        "simulation": simulation_report,
        "search_candidates_evaluated": optimization_results["search_candidates_evaluated"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    args = parse_args()
    hidden_layers = parse_hidden_layers(args.hidden_layers)

    csv_path = Path(args.csv_path).resolve()
    params_output = Path(args.params_output).resolve()
    plot_output = Path(args.plot_output).resolve()
    model_output = Path(args.model_output).resolve()

    frame, model_features, class_distribution = load_feature_frame(
        csv_path=csv_path,
        target_horizon=args.target_horizon,
        lookback_window=args.lookback_window,
        min_move_threshold=args.min_move_threshold,
        neutral_vol_multiplier=args.neutral_vol_multiplier,
    )
    split_data = build_splits(
        frame=frame,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    scaler, model, history, ml_metrics = train_mlp_classifier(
        split_data=split_data,
        model_features=model_features,
        hidden_layers=hidden_layers,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )

    plot_accuracy_history(
        history=history,
        best_epoch=int(ml_metrics["best_epoch"]),
        test_accuracy=float(ml_metrics["test_accuracy"]),
        test_balanced_accuracy=float(ml_metrics["test_balanced_accuracy"]),
        early_stop_epoch=ml_metrics["early_stop_epoch"],
        output_path=plot_output,
    )

    train_scored = attach_probabilities(split_data.train_full, scaler, model, model_features)
    test_scored = attach_probabilities(split_data.test, scaler, model, model_features)
    optimization_results = optimize_parameters(
        train_frame=train_scored,
        test_frame=test_scored,
        iterations=args.search_iterations,
        min_trades=args.min_trades,
        seed=args.seed,
    )

    save_model_bundle(
        output_path=model_output,
        scaler=scaler,
        model=model,
        model_features=model_features,
        best_epoch=int(ml_metrics["best_epoch"]),
        target_horizon=args.target_horizon,
        hidden_layers=hidden_layers,
        lookback_window=args.lookback_window,
    )
    save_parameter_report(
        output_path=params_output,
        csv_path=csv_path,
        frame=frame,
        split_data=split_data,
        model_features=model_features,
        class_distribution=class_distribution,
        ml_metrics=ml_metrics,
        optimization_results=optimization_results,
        hidden_layers=hidden_layers,
        epochs_requested=args.epochs,
        patience=args.patience,
        target_horizon=args.target_horizon,
        lookback_window=args.lookback_window,
        min_move_threshold=args.min_move_threshold,
        neutral_vol_multiplier=args.neutral_vol_multiplier,
        starting_capital=args.starting_capital,
    )

    optimized_test_simulation = build_simulation_summary(
        optimization_results["best_test_metrics"],
        args.starting_capital,
    )
    default_test_simulation = build_simulation_summary(
        optimization_results["default_test_metrics"],
        args.starting_capital,
    )

    print(f"Saved optimized parameters to: {params_output}")
    print(f"Saved MLP accuracy plot to: {plot_output}")
    print(f"Saved trained model bundle to: {model_output}")
    print("Best parameters:")
    for key, value in optimization_results["best_parameters"].items():
        print(f"  {key} = {value}")
    print("Model accuracy:")
    print(f"  Train accuracy: {ml_metrics['full_train_accuracy']:.4f}")
    print(f"  Test accuracy:  {ml_metrics['test_accuracy']:.4f}")
    print(f"  Test balanced accuracy:  {ml_metrics['test_balanced_accuracy']:.4f}")
    print(f"  Test actionable accuracy: {ml_metrics['test_actionable_accuracy']:.4f}")
    print_simulation_summary("Simulated optimized bot performance on test split:", optimized_test_simulation)
    print_simulation_summary("Simulated default bot performance on test split:", default_test_simulation)


if __name__ == "__main__":
    main()
