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
RANDOM_SEED = 41
DEFAULT_EPOCHS = 67
DEFAULT_PATIENCE = 8
DEFAULT_LOOKBACK_WINDOW = 24
DEFAULT_MIN_MOVE_THRESHOLD = 0.006
DEFAULT_NEUTRAL_VOL_MULTIPLIER = 0.80
DEFAULT_STARTING_CAPITAL = 10_000.0
DEFAULT_RECENT_DAYS = 90
DEFAULT_HIDDEN_LAYERS = "64,32"
DEFAULT_ALPHA = 5e-4
DEFAULT_LEARNING_RATE = 5e-5
DEFAULT_POSITION_PCT = 0.379
EARLY_STOP_MIN_DELTA = 1e-3
DEFAULT_CSV_PATHS = [
    str(BASE_DIR / "data" / "BTCUSDT_1h.csv"),
    str(BASE_DIR / "data" / "ETHUSDT_1h.csv"),
]
DEFAULT_OUTPUT_DIR = BASE_DIR / "ml_mlp_outputs"
DEFAULT_SUMMARY_OUTPUT = BASE_DIR / "optimized_strategy_params_mlp_multi.json"
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
    "ENTRY_HMM_CONFIDENCE": 0.586,
    "ENTRY_MAX_VOLATILITY": 0.657,
    "ENTRY_MOMENTUM_MIN": 0.0181,
    "ENTRY_VOLUME_ZSCORE_MAX": 4.284,
    "ENTRY_MA_SHORT": 20,
    "ENTRY_SPREAD_MAX_PCT": 0.05,
    "EXIT_HMM_CONFIDENCE": 0.60,
    "EXIT_MA_LONG": 20,
    "EXIT_STOP_LOSS_PCT": 0.02,
    "EXIT_TAKE_PROFIT_PCT": 0.02,
}

OPTIMIZED_PARAMETER_KEYS = [
    "ENTRY_HMM_CONFIDENCE",
    "ENTRY_MAX_VOLATILITY",
    "ENTRY_MOMENTUM_MIN",
    "ENTRY_VOLUME_ZSCORE_MAX",
    "ENTRY_MA_SHORT",
    "ENTRY_SPREAD_MAX_PCT",
    "EXIT_HMM_CONFIDENCE",
    "EXIT_MA_LONG",
]
FIXED_PARAMETER_KEYS = [
    "EXIT_STOP_LOSS_PCT",
    "EXIT_TAKE_PROFIT_PCT",
]

SEARCH_BOUNDS = {
    "ENTRY_HMM_CONFIDENCE": (0.20, 0.90),
    "ENTRY_MAX_VOLATILITY": (0.05, 4.00),
    "ENTRY_MOMENTUM_MIN": (-0.05, 0.05),
    "ENTRY_VOLUME_ZSCORE_MAX": (0.50, 6.00),
    "ENTRY_MA_SHORT": (5, 60),
    "ENTRY_SPREAD_MAX_PCT": (0.01, 0.20),
    "EXIT_HMM_CONFIDENCE": (0.20, 0.90),
    "EXIT_MA_LONG": (5, 80),
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
        description="Train an MLP on crypto OHLCV data and optimize strategy thresholds per asset."
    )
    parser.add_argument(
        "--csv-paths",
        nargs="+",
        default=DEFAULT_CSV_PATHS,
        help="One or more historical OHLCV CSVs. Each asset is trained independently.",
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
        "--recent-days",
        type=int,
        default=DEFAULT_RECENT_DAYS,
        help="Use only the most recent N days from each CSV before training. Use 0 to keep the full file.",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=12,
        help="Future horizon in bars for the classifier target.",
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
        help="Patience window used to mark the first sustained validation plateau; training still runs all requested epochs.",
    )
    parser.add_argument(
        "--hidden-layers",
        default=DEFAULT_HIDDEN_LAYERS,
        help="Comma-separated hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="L2 regularization strength for the MLP.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for the MLP optimizer.",
    )
    parser.add_argument(
        "--lookback-window",
        type=int,
        default=DEFAULT_LOOKBACK_WINDOW,
        help="How many historical bars to flatten into each ML sample.",
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
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where per-asset plots, parameter reports, and model bundles are saved.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_OUTPUT),
        help="Where to save the combined multi-asset parameter summary JSON output.",
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


def normalize_asset_name(csv_path: Path) -> str:
    stem = csv_path.stem.upper()
    asset = stem.split("_")[0]
    if asset.endswith("USDT"):
        asset = f"{asset[:-4]}USD"
    return asset


def infer_bar_minutes(frame: pd.DataFrame) -> float:
    diffs = frame["open_time"].diff().dropna()
    if diffs.empty:
        return 60.0

    minutes = diffs.dt.total_seconds() / 60.0
    minutes = minutes[minutes > 0]
    if minutes.empty:
        return 60.0

    return float(minutes.median())


def get_frame_annualization_factor(frame: pd.DataFrame) -> float:
    return float(frame.attrs.get("annualization_factor", ANNUALIZATION_FACTOR))


def get_frame_bar_hours(frame: pd.DataFrame) -> float:
    return float(frame.attrs.get("bar_hours", 1.0))


def propagate_frame_attrs(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    target.attrs = dict(source.attrs)
    return target


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
    best_validation_loss: float,
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
        f"best_val_loss={best_validation_loss:.5f} | "
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
    recent_days: int,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    frame = pd.read_csv(csv_path, parse_dates=["open_time"])
    frame = frame.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    asset_name = normalize_asset_name(csv_path)

    if recent_days > 0:
        latest_time = frame["open_time"].max()
        cutoff = latest_time - pd.Timedelta(days=int(recent_days))
        frame = frame.loc[frame["open_time"] >= cutoff].reset_index(drop=True)

    bar_minutes = infer_bar_minutes(frame)
    bar_hours = bar_minutes / 60.0
    annualization_factor = (60.0 / max(bar_minutes, EPSILON)) * 24.0 * 365.0

    frame["log_return"] = np.log(frame["close"] / frame["close"].shift(1))
    frame["return_3"] = frame["close"].pct_change(periods=3)
    frame["return_6"] = frame["close"].pct_change(periods=6)
    frame["return_12"] = frame["close"].pct_change(periods=12)
    frame["rolling_vol_6"] = frame["log_return"].rolling(window=6).std() * np.sqrt(annualization_factor)
    frame["rolling_vol_20"] = frame["log_return"].rolling(window=20).std() * np.sqrt(annualization_factor)
    frame["rolling_vol_48"] = frame["log_return"].rolling(window=48).std() * np.sqrt(annualization_factor)
    frame["momentum_10"] = frame["close"].pct_change(periods=10)

    volume_mean = frame["volume"].rolling(window=20).mean()
    volume_std = frame["volume"].rolling(window=20).std()
    frame["volume_zscore_20"] = (frame["volume"] - volume_mean) / (volume_std + EPSILON)

    ma20 = frame["close"].ewm(span=20, adjust=False).mean()
    ma50 = frame["close"].ewm(span=50, adjust=False).mean()
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
    bar_volatility = frame["rolling_vol_20"] / np.sqrt(annualization_factor)
    future_move_threshold = np.maximum(
        min_move_threshold,
        bar_volatility * np.sqrt(target_horizon) * neutral_vol_multiplier,
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

    frame.attrs["asset_name"] = asset_name
    frame.attrs["bar_minutes"] = float(bar_minutes)
    frame.attrs["bar_hours"] = float(bar_hours)
    frame.attrs["annualization_factor"] = float(annualization_factor)
    frame.attrs["target_horizon_bars"] = int(target_horizon)
    frame.attrs["lookback_window_bars"] = int(lookback_window)
    frame.attrs["recent_days"] = int(recent_days)
    frame.attrs["start_time"] = frame["open_time"].min().isoformat()
    frame.attrs["end_time"] = frame["open_time"].max().isoformat()

    return frame, model_features, class_distribution


def build_splits(frame: pd.DataFrame, train_ratio: float, validation_ratio: float) -> SplitData:
    if not 0.5 < train_ratio < 0.95:
        raise ValueError("train_ratio must be between 0.5 and 0.95.")
    if not 0.05 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0.05 and 0.5.")

    split_index = int(len(frame) * train_ratio)
    train_full = propagate_frame_attrs(frame, frame.iloc[:split_index].copy())
    test = propagate_frame_attrs(frame, frame.iloc[split_index:].copy())

    validation_index = int(len(train_full) * (1.0 - validation_ratio))
    train = propagate_frame_attrs(train_full, train_full.iloc[:validation_index].copy())
    validation = propagate_frame_attrs(train_full, train_full.iloc[validation_index:].copy())

    if train.empty or validation.empty or test.empty:
        raise ValueError("Split configuration produced an empty train, validation, or test segment.")

    return SplitData(train=train, validation=validation, train_full=train_full, test=test)


def build_class_balanced_sample_weight(y_train: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    counts = {
        int(label): int(np.sum(y_train == label))
        for label in TARGET_CLASSES
    }
    missing = [
        TARGET_LABEL_NAMES[label]
        for label, count in counts.items()
        if count == 0
    ]
    if missing:
        raise ValueError(f"Training split is missing target classes: {', '.join(missing)}")

    total = len(y_train)
    class_weights = {
        label: total / (len(TARGET_CLASSES) * count)
        for label, count in counts.items()
    }
    sample_weight = np.array([class_weights[int(label)] for label in y_train], dtype=float)
    class_distribution = {
        TARGET_LABEL_NAMES[int(label)]: counts[int(label)]
        for label in TARGET_CLASSES
    }
    return sample_weight, class_distribution


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
    sample_weight, _ = build_class_balanced_sample_weight(y_train)
    batch_size = min(128, max(32, len(X_train) // 20))
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
        if epoch == 0:
            model.partial_fit(X_train, y_train, classes=classes, sample_weight=sample_weight)
        else:
            model.partial_fit(X_train, y_train, sample_weight=sample_weight)

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
    train_sample_weight, training_class_distribution = build_class_balanced_sample_weight(y_train)

    batch_size = min(128, max(32, len(X_train_scaled_unbalanced) // 20))
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
    best_validation_loss = np.inf
    best_validation_accuracy = -np.inf
    best_validation_balanced_accuracy = -np.inf
    best_checkpoint_found = False
    wait = 0
    early_stop_epoch = None

    for epoch in range(1, epochs + 1):
        if epoch == 1:
            model.partial_fit(
                X_train_scaled_unbalanced,
                y_train,
                classes=classes,
                sample_weight=train_sample_weight,
            )
        else:
            model.partial_fit(
                X_train_scaled_unbalanced,
                y_train,
                sample_weight=train_sample_weight,
            )

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
            validation_loss < best_validation_loss - EARLY_STOP_MIN_DELTA
            or (
                abs(validation_loss - best_validation_loss) <= EARLY_STOP_MIN_DELTA
                and validation_balanced_accuracy > best_validation_balanced_accuracy + 1e-6
            )
            or (
                abs(validation_loss - best_validation_loss) <= EARLY_STOP_MIN_DELTA
                and abs(validation_balanced_accuracy - best_validation_balanced_accuracy) <= 1e-6
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
            best_validation_loss = float(validation_loss)
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
            best_validation_loss=float(best_validation_loss),
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
    _, train_full_class_distribution = build_class_balanced_sample_weight(y_train_full)
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
        "epochs_trained": int(len(history)),
        "best_validation_loss": float(best_validation_loss),
        "best_validation_accuracy": float(best_validation_accuracy),
        "full_train_accuracy": float(full_train_accuracy),
        "full_train_balanced_accuracy": float(full_train_balanced_accuracy),
        "full_train_actionable_accuracy": full_train_actionable_accuracy,
        "test_accuracy": float(test_accuracy),
        "test_balanced_accuracy": float(test_balanced_accuracy),
        "test_actionable_accuracy": test_actionable_accuracy,
        "selection_train_accuracy": float(history[best_epoch - 1]["train_accuracy"]),
        "selection_train_balanced_accuracy": float(history[best_epoch - 1]["train_balanced_accuracy"]),
        "selection_train_loss": float(history[best_epoch - 1]["train_loss"]),
        "selection_validation_loss": float(history[best_epoch - 1]["validation_loss"]),
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
    asset_name: str,
    clip_epoch: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if clip_epoch is not None:
        history = [row for row in history if int(row["epoch"]) <= int(clip_epoch)]

    epochs = [row["epoch"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    validation_accuracy = [row["validation_accuracy"] for row in history]
    train_balanced_accuracy = [row["train_balanced_accuracy"] for row in history]
    validation_balanced_accuracy = [row["validation_balanced_accuracy"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]

    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    accuracy_ax, loss_ax = axes

    accuracy_ax.plot(epochs, train_accuracy, label="Train accuracy", linewidth=2.0, color="#1d4ed8")
    accuracy_ax.plot(epochs, validation_accuracy, label="Validation accuracy", linewidth=2.0, color="#15803d")
    accuracy_ax.plot(
        epochs,
        train_balanced_accuracy,
        label="Train balanced accuracy",
        linewidth=1.5,
        linestyle="--",
        color="#60a5fa",
    )
    accuracy_ax.plot(
        epochs,
        validation_balanced_accuracy,
        label="Validation balanced accuracy",
        linewidth=1.5,
        linestyle="--",
        color="#4ade80",
    )
    accuracy_ax.axvline(best_epoch, color="#b45309", linestyle="--", linewidth=1.5, label=f"Best epoch = {best_epoch}")
    if early_stop_epoch is not None:
        accuracy_ax.axvline(
            early_stop_epoch,
            color="#dc2626",
            linestyle="-.",
            linewidth=1.2,
            label=f"Patience threshold = {early_stop_epoch}",
        )
        loss_ax.axvline(
            early_stop_epoch,
            color="#dc2626",
            linestyle="-.",
            linewidth=1.2,
        )
    accuracy_ax.axhline(
        test_accuracy,
        color="#7c3aed",
        linestyle=":",
        linewidth=1.5,
        label=f"Test accuracy = {test_accuracy:.3f}",
    )
    accuracy_ax.axhline(
        test_balanced_accuracy,
        color="#f97316",
        linestyle=":",
        linewidth=1.5,
        label=f"Test balanced accuracy = {test_balanced_accuracy:.3f}",
    )
    accuracy_ax.set_title(f"{asset_name} MLP Accuracy Across Epochs")
    accuracy_ax.set_ylabel("Accuracy")
    accuracy_ax.set_ylim(0.0, 1.0)
    accuracy_ax.grid(alpha=0.25)
    accuracy_ax.legend()

    loss_ax.plot(epochs, train_loss, label="Train loss", linewidth=2.0, color="#2563eb")
    loss_ax.plot(epochs, validation_loss, label="Validation loss", linewidth=2.0, color="#16a34a")
    loss_ax.axvline(best_epoch, color="#b45309", linestyle="--", linewidth=1.5, label=f"Best epoch = {best_epoch}")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()

    figure.tight_layout()
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
        cache[window] = frame["close"].ewm(span=window, adjust=False).mean().to_numpy(dtype=float)
    return cache[window]


def compute_risk_metrics(
    returns: np.ndarray,
    annualization_factor: float,
) -> dict[str, float]:
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

    sharpe_ratio = 0.0 if std_return <= EPSILON else mean_return / std_return * np.sqrt(annualization_factor)
    sortino_ratio = 0.0 if downside_std <= EPSILON else mean_return / downside_std * np.sqrt(annualization_factor)

    equity_curve = np.cumprod(np.clip(1.0 + returns, 0.001, None))
    rolling_peak = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / np.maximum(rolling_peak, EPSILON) - 1.0
    max_drawdown = float(np.min(drawdowns))
    total_return = float(equity_curve[-1] - 1.0)

    if equity_curve[-1] <= EPSILON:
        annualized_return = -1.0
    else:
        annualized_return = float(equity_curve[-1] ** (annualization_factor / len(returns)) - 1.0)

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
        "position_pct": float(DEFAULT_POSITION_PCT * 100.0),
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
    print(f"  Position size:    {summary['position_pct']:.2f}%")
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
    annualization_factor = get_frame_annualization_factor(frame)
    bar_hours = get_frame_bar_hours(frame)
    position_pct = float(DEFAULT_POSITION_PCT)

    ma_short = get_cached_ma(frame, int(params["ENTRY_MA_SHORT"]), ma_cache)
    ma_long = get_cached_ma(frame, int(params["EXIT_MA_LONG"]), ma_cache)
    fixed_stop_loss = float(DEFAULT_PARAMS["EXIT_STOP_LOSS_PCT"])
    fixed_take_profit = float(DEFAULT_PARAMS["EXIT_TAKE_PROFIT_PCT"])

    period_returns = np.zeros(len(frame), dtype=float)
    trade_returns: list[float] = []
    holding_hours: list[float] = []

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
            gain = 0.0 if entry_price <= 0 else (close[idx] - entry_price) / entry_price
            should_exit = (
                drawdown >= fixed_stop_loss
                or gain >= fixed_take_profit
                or close[idx] < ma_long[idx]
                or bearish_prob[idx] > params["EXIT_HMM_CONFIDENCE"]
            )
            if should_exit:
                exit_cost = position_pct * spread_pct[idx] / 100.0 / 2.0
                period_returns[idx] -= exit_cost
                trade_return = position_pct * (close[idx] / entry_price - 1.0) - entry_cost - exit_cost
                trade_returns.append(float(trade_return))
                holding_hours.append(max(1, idx - entry_index) * bar_hours)
                in_position = False
                exited_this_bar = True
                entry_price = 0.0
                entry_cost = 0.0
                entry_index = -1

        if not in_position and not exited_this_bar:
            hmm_ok = bullish_prob[idx] > params["ENTRY_HMM_CONFIDENCE"]
            vol_ok = rolling_vol[idx] < params["ENTRY_MAX_VOLATILITY"]
            mom_ok = momentum[idx] > params["ENTRY_MOMENTUM_MIN"]
            volume_ok = abs(volume_zscore[idx]) < params["ENTRY_VOLUME_ZSCORE_MAX"]
            ma_ok = close[idx] > ma_short[idx]
            spread_ok = spread_pct[idx] < params["ENTRY_SPREAD_MAX_PCT"]
            should_enter = mom_ok and ma_ok and sum([hmm_ok, vol_ok, volume_ok, spread_ok]) >= 2
            if should_enter:
                in_position = True
                entry_price = close[idx]
                entry_cost = position_pct * spread_pct[idx] / 100.0 / 2.0
                period_returns[idx] -= entry_cost
                entry_index = idx

        if in_position:
            period_returns[idx + 1] += position_pct * (close[idx + 1] / close[idx] - 1.0)
            exposure_hours += 1

    if in_position and entry_price > 0:
        exit_cost = position_pct * spread_pct[-1] / 100.0 / 2.0
        period_returns[-1] -= exit_cost
        trade_return = position_pct * (close[-1] / entry_price - 1.0) - entry_cost - exit_cost
        trade_returns.append(float(trade_return))
        holding_hours.append(max(1, len(frame) - 1 - entry_index) * bar_hours)

    metrics = compute_risk_metrics(period_returns, annualization_factor=annualization_factor)
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
            "EXIT_MA_LONG": sample_int(
                rng,
                *SEARCH_BOUNDS["EXIT_MA_LONG"],
                DEFAULT_PARAMS["EXIT_MA_LONG"],
            ),
            "EXIT_STOP_LOSS_PCT": DEFAULT_PARAMS["EXIT_STOP_LOSS_PCT"],
            "EXIT_TAKE_PROFIT_PCT": DEFAULT_PARAMS["EXIT_TAKE_PROFIT_PCT"],
        }

        candidate["ENTRY_HMM_CONFIDENCE"] = round(candidate["ENTRY_HMM_CONFIDENCE"], 4)
        candidate["ENTRY_MAX_VOLATILITY"] = round(candidate["ENTRY_MAX_VOLATILITY"], 4)
        candidate["ENTRY_MOMENTUM_MIN"] = round(candidate["ENTRY_MOMENTUM_MIN"], 4)
        candidate["ENTRY_VOLUME_ZSCORE_MAX"] = round(candidate["ENTRY_VOLUME_ZSCORE_MAX"], 4)
        candidate["ENTRY_SPREAD_MAX_PCT"] = round(candidate["ENTRY_SPREAD_MAX_PCT"], 4)
        candidate["EXIT_HMM_CONFIDENCE"] = round(candidate["EXIT_HMM_CONFIDENCE"], 4)

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
    bar_minutes: float,
    asset_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "asset_name": asset_name,
        "scaler": scaler,
        "model": model,
        "feature_columns": model_features,
        "class_labels": [int(label) for label in model.classes_],
        "best_epoch": int(best_epoch),
        "target_horizon_bars": int(target_horizon),
        "hidden_layers": list(hidden_layers),
        "lookback_window_bars": int(lookback_window),
        "bar_interval_minutes": float(bar_minutes),
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
    asset_name: str,
    bar_minutes: float,
    recent_days: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_report = build_simulation_report(
        optimization_results=optimization_results,
        starting_capital=starting_capital,
    )
    report = {
        "dataset": {
            "asset_name": asset_name,
            "csv_path": str(csv_path),
            "rows_after_feature_engineering": int(len(frame)),
            "recent_days_used": int(recent_days),
            "start_time": frame.attrs.get("start_time"),
            "end_time": frame.attrs.get("end_time"),
            "train_rows": int(len(split_data.train_full)),
            "test_rows": int(len(split_data.test)),
            "bar_interval_minutes": float(bar_minutes),
            "target_horizon_bars": int(target_horizon),
            "lookback_window_bars": int(lookback_window),
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
            "epochs_trained": int(ml_metrics["epochs_trained"]),
            "early_stop_epoch": ml_metrics["early_stop_epoch"],
            "patience": int(patience),
            "selection_metric": "lowest validation log loss, then balanced accuracy, then accuracy",
            "selection_train_accuracy": float(ml_metrics["selection_train_accuracy"]),
            "selection_train_balanced_accuracy": float(ml_metrics["selection_train_balanced_accuracy"]),
            "selection_train_loss": float(ml_metrics["selection_train_loss"]),
            "selection_validation_loss": float(ml_metrics["selection_validation_loss"]),
            "best_validation_loss": float(ml_metrics["best_validation_loss"]),
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
                "bullish if future_return > max(min_move_threshold, bar_volatility * sqrt(horizon_bars) * neutral_volatility_multiplier); "
                "bearish if future_return < -that threshold; otherwise neutral"
            ),
        },
        "assumptions": {
            "training_balance": (
                "The MLP is trained on the original chronological samples using class-balanced "
                "sample weights instead of per-epoch random oversampling."
            ),
            "confidence_source": (
                "ENTRY_HMM_CONFIDENCE and EXIT_HMM_CONFIDENCE are optimized against "
                "MLP bullish and bearish class probabilities from a 3-class "
                "bull/neutral/bear model. EXIT_MA_LONG controls the live long-MA "
                "exit, while stop-loss and take-profit stay fixed at the config values."
            ),
            "spread_proxy": (
                "The CSV has no bid/ask spread, so estimated_spread_pct uses 5% of the "
                "3-candle median intrabar range in percentage points."
            ),
            "fixed_strategy_parameters": {
                "EXIT_STOP_LOSS_PCT": DEFAULT_PARAMS["EXIT_STOP_LOSS_PCT"],
                "EXIT_TAKE_PROFIT_PCT": DEFAULT_PARAMS["EXIT_TAKE_PROFIT_PCT"],
            },
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


def build_asset_artifact_paths(output_dir: Path, asset_name: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_slug = asset_name.lower()
    return {
        "params_output": output_dir / f"optimized_strategy_params_{asset_slug}.json",
        "plot_output": output_dir / f"ml_accuracy_{asset_slug}.png",
        "model_output": output_dir / f"strategy_mlp_model_{asset_slug}.pkl",
    }


def save_combined_summary(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_asset_pipeline(
    csv_path: Path,
    args: argparse.Namespace,
    hidden_layers: tuple[int, ...],
    output_dir: Path,
) -> dict[str, object]:
    asset_name = normalize_asset_name(csv_path)
    artifact_paths = build_asset_artifact_paths(output_dir=output_dir, asset_name=asset_name)

    print(f"\n[{asset_name}] Loading dataset from {csv_path}")
    frame, model_features, class_distribution = load_feature_frame(
        csv_path=csv_path,
        target_horizon=args.target_horizon,
        lookback_window=args.lookback_window,
        min_move_threshold=args.min_move_threshold,
        neutral_vol_multiplier=args.neutral_vol_multiplier,
        recent_days=args.recent_days,
    )
    split_data = build_splits(
        frame=frame,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    bar_minutes = float(frame.attrs.get("bar_minutes", 60.0))
    print(
        f"[{asset_name}] rows={len(frame)}, bar_interval={bar_minutes:.1f}m, "
        f"train={len(split_data.train_full)}, test={len(split_data.test)}"
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
        output_path=artifact_paths["plot_output"],
        asset_name=asset_name,
        clip_epoch=None,
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
        output_path=artifact_paths["model_output"],
        scaler=scaler,
        model=model,
        model_features=model_features,
        best_epoch=int(ml_metrics["best_epoch"]),
        target_horizon=args.target_horizon,
        hidden_layers=hidden_layers,
        lookback_window=args.lookback_window,
        bar_minutes=bar_minutes,
        asset_name=asset_name,
    )
    save_parameter_report(
        output_path=artifact_paths["params_output"],
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
        asset_name=asset_name,
        bar_minutes=bar_minutes,
        recent_days=args.recent_days,
    )

    optimized_test_simulation = build_simulation_summary(
        optimization_results["best_test_metrics"],
        args.starting_capital,
    )
    default_test_simulation = build_simulation_summary(
        optimization_results["default_test_metrics"],
        args.starting_capital,
    )

    print(f"[{asset_name}] Saved optimized parameters to: {artifact_paths['params_output']}")
    print(f"[{asset_name}] Saved MLP accuracy plot to: {artifact_paths['plot_output']}")
    print(f"[{asset_name}] Saved trained model bundle to: {artifact_paths['model_output']}")
    print(f"[{asset_name}] Best parameters:")
    for key, value in optimization_results["best_parameters"].items():
        print(f"  {key} = {value}")
    print(f"[{asset_name}] Model accuracy:")
    print(f"  Train accuracy: {ml_metrics['full_train_accuracy']:.4f}")
    print(f"  Test accuracy:  {ml_metrics['test_accuracy']:.4f}")
    print(f"  Test balanced accuracy:  {ml_metrics['test_balanced_accuracy']:.4f}")
    print(f"  Test actionable accuracy: {ml_metrics['test_actionable_accuracy']:.4f}")
    print_simulation_summary(
        f"[{asset_name}] Simulated optimized bot performance on test split:",
        optimized_test_simulation,
    )
    print_simulation_summary(
        f"[{asset_name}] Simulated default bot performance on test split:",
        default_test_simulation,
    )

    return {
        "asset_name": asset_name,
        "csv_path": str(csv_path),
        "bar_interval_minutes": bar_minutes,
        "recent_days_used": int(args.recent_days),
        "optimized_parameters": optimization_results["best_parameters"],
        "default_parameters": DEFAULT_PARAMS,
        "model_accuracy": {
            "train_accuracy": float(ml_metrics["full_train_accuracy"]),
            "test_accuracy": float(ml_metrics["test_accuracy"]),
            "test_balanced_accuracy": float(ml_metrics["test_balanced_accuracy"]),
            "test_actionable_accuracy": float(ml_metrics["test_actionable_accuracy"]),
            "best_epoch": int(ml_metrics["best_epoch"]),
            "best_validation_loss": float(ml_metrics["best_validation_loss"]),
        },
        "simulation": {
            "optimized_test": optimized_test_simulation,
            "default_test": default_test_simulation,
        },
        "artifacts": {
            key: str(value)
            for key, value in artifact_paths.items()
        },
    }


def main() -> None:
    args = parse_args()
    hidden_layers = parse_hidden_layers(args.hidden_layers)
    output_dir = Path(args.output_dir).resolve()
    summary_output = Path(args.summary_output).resolve()

    csv_paths = [Path(raw_path).resolve() for raw_path in args.csv_paths]
    asset_summaries: dict[str, object] = {}

    for csv_path in csv_paths:
        asset_summary = run_asset_pipeline(
            csv_path=csv_path,
            args=args,
            hidden_layers=hidden_layers,
            output_dir=output_dir,
        )
        asset_summaries[asset_summary["asset_name"]] = asset_summary

    combined_summary = {
        "training_window": {
            "recent_days_used": int(args.recent_days),
            "policy": "Each asset is trained on the most recent N days ending at the latest timestamp available in that CSV.",
        },
        "strategy_alignment": {
            "optimized_parameters": OPTIMIZED_PARAMETER_KEYS,
            "fixed_parameters": FIXED_PARAMETER_KEYS,
            "exit_logic": (
                "Stop-loss and take-profit stay fixed at config values, while the "
                "MLP optimizer tunes the bearish-probability exit and long-MA exit "
                "to match strategy.py."
            ),
        },
        "assets": asset_summaries,
    }
    save_combined_summary(summary_output, combined_summary)
    print(f"\nSaved combined multi-asset summary to: {summary_output}")


if __name__ == "__main__":
    main()
