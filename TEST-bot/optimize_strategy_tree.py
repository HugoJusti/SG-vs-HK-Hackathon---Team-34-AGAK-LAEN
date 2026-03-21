"""
Offline tree-model optimizer for the Binance strategy thresholds.

This script keeps the same feature engineering, target construction, and
parameter-search objective as optimize_strategy_ml.py, but swaps the classifier
for a gradient-boosted tree model that is usually a better fit for tabular
technical features.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from optimize_strategy_ml import (
    BASE_DIR,
    RANDOM_SEED,
    DEFAULT_PATIENCE,
    DEFAULT_LOOKBACK_WINDOW,
    DEFAULT_MIN_MOVE_THRESHOLD,
    DEFAULT_NEUTRAL_VOL_MULTIPLIER,
    DEFAULT_PARAMS,
    TARGET_CLASSES,
    TARGET_LABEL_NAMES,
    SEQUENCE_BASE_FEATURES,
    load_feature_frame,
    build_splits,
    summarise_class_distribution,
    compute_actionable_accuracy,
    sample_candidates,
    backtest_strategy,
)
from tree_training_outputs import plot_tree_history, save_tree_history_csv


DEFAULT_TREE_ITERATIONS = 300
DEFAULT_TREE_LEARNING_RATE = 0.05
DEFAULT_TREE_MAX_DEPTH = 3
DEFAULT_TREE_SUBSAMPLE = 0.8
DEFAULT_TREE_MIN_SAMPLES_LEAF = 20


def print_stage(message: str) -> None:
    print(f"[tree] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gradient-boosted tree model on Binance OHLCV data and optimize strategy thresholds."
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
        "--n-estimators",
        type=int,
        default=DEFAULT_TREE_ITERATIONS,
        help="Maximum number of boosting iterations for the tree model.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Patience window used to flag when validation stopped improving. The plot still runs all requested iterations.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_TREE_LEARNING_RATE,
        help="Learning rate for gradient boosting.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_TREE_MAX_DEPTH,
        help="Maximum depth of each decision tree.",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=DEFAULT_TREE_SUBSAMPLE,
        help="Row subsampling ratio for stochastic boosting.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=DEFAULT_TREE_MIN_SAMPLES_LEAF,
        help="Minimum samples per leaf to regularize the trees.",
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
        default=str(BASE_DIR / "optimized_strategy_params_tree.json"),
        help="Where to save the optimized parameter JSON output for the tree model.",
    )
    parser.add_argument(
        "--plot-output",
        default=str(BASE_DIR / "tree_accuracy.png"),
        help="Where to save the tree-model accuracy plot.",
    )
    parser.add_argument(
        "--history-output",
        default=str(BASE_DIR / "tree_training_history.csv"),
        help="Where to save the per-iteration training history CSV.",
    )
    parser.add_argument(
        "--model-output",
        default=str(BASE_DIR / "strategy_tree_model.pkl"),
        help="Where to save the trained tree-model bundle.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def build_class_weighted_sample_weight(y: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    counts = {
        int(label): int(np.sum(y == label))
        for label in TARGET_CLASSES
    }
    missing = [
        TARGET_LABEL_NAMES[label]
        for label, count in counts.items()
        if count == 0
    ]
    if missing:
        raise ValueError(f"Training split is missing target classes: {', '.join(missing)}")

    total = len(y)
    class_weights = {
        label: total / (len(TARGET_CLASSES) * count)
        for label, count in counts.items()
    }
    sample_weight = np.array([class_weights[int(label)] for label in y], dtype=float)
    class_distribution = {
        TARGET_LABEL_NAMES[label]: counts[label]
        for label in TARGET_CLASSES
    }
    return sample_weight, class_distribution


def compute_iteration_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    predictions = classes[np.argmax(probabilities, axis=1)]
    accuracy = accuracy_score(y_true, predictions)
    balanced_accuracy = balanced_accuracy_score(y_true, predictions)
    loss = log_loss(y_true, probabilities, labels=classes.tolist())
    return float(accuracy), float(balanced_accuracy), float(loss), predictions


def print_iteration_metrics(
    iteration: int,
    total_iterations: int,
    train_accuracy: float,
    train_balanced_accuracy: float,
    validation_accuracy: float,
    validation_balanced_accuracy: float,
    train_loss: float,
    validation_loss: float,
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
        f"Epoch {iteration:03d}/{total_iterations:03d} | "
        f"train_acc={train_accuracy:.4f} | "
        f"train_bal_acc={train_balanced_accuracy:.4f} | "
        f"val_acc={validation_accuracy:.4f} | "
        f"val_bal_acc={validation_balanced_accuracy:.4f} | "
        f"train_loss={train_loss:.5f} | "
        f"val_loss={validation_loss:.5f} | "
        f"best_val_acc={best_validation_accuracy:.4f} | "
        f"best_val_bal_acc={best_validation_balanced_accuracy:.4f} | "
        f"wait={wait:02d} | "
        f"status={status}"
    )


def make_tree_model(
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    min_samples_leaf: int,
    seed: int,
) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        loss="log_loss",
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        subsample=subsample,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )


def optimize_parameters_with_progress(
    train_frame,
    test_frame,
    iterations: int,
    min_trades: int,
    seed: int,
) -> dict[str, object]:
    candidates = sample_candidates(iterations=iterations, seed=seed)
    train_ma_cache: dict[int, np.ndarray] = {}
    test_ma_cache: dict[int, np.ndarray] = {}

    best_params = None
    best_train_metrics = None
    total = len(candidates)
    print_stage(f"Starting parameter search across {total} candidates...")

    for index, candidate in enumerate(candidates, start=1):
        train_metrics = backtest_strategy(
            frame=train_frame,
            params=candidate,
            min_trades=min_trades,
            ma_cache=train_ma_cache,
        )
        if best_train_metrics is None or train_metrics["objective"] > best_train_metrics["objective"]:
            best_params = candidate
            best_train_metrics = train_metrics
            print_stage(
                "New best candidate "
                f"{index}/{total}: objective={train_metrics['objective']:.4f}, "
                f"sortino={train_metrics['sortino_ratio']:.4f}, "
                f"sharpe={train_metrics['sharpe_ratio']:.4f}, "
                f"calmar={train_metrics['calmar_ratio']:.4f}, "
                f"trades={train_metrics['closed_trades']}"
            )
        elif index == 1 or index % 25 == 0 or index == total:
            print_stage(
                f"Evaluated candidate {index}/{total} | current_best_objective={best_train_metrics['objective']:.4f}"
            )

    if best_params is None or best_train_metrics is None:
        raise RuntimeError("Parameter search failed to find a usable candidate.")

    print_stage("Scoring best candidate on the test split...")
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
        "search_candidates_evaluated": total,
    }


def train_tree_classifier(
    split_data,
    model_features: list[str],
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    min_samples_leaf: int,
    patience: int,
    seed: int,
) -> tuple[GradientBoostingClassifier, list[dict[str, float]], dict[str, float]]:
    X_train = split_data.train[model_features].to_numpy(dtype=float)
    y_train = split_data.train["target"].to_numpy(dtype=int)
    X_validation = split_data.validation[model_features].to_numpy(dtype=float)
    y_validation = split_data.validation["target"].to_numpy(dtype=int)

    sample_weight, training_class_distribution = build_class_weighted_sample_weight(y_train)
    model = make_tree_model(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )
    print_stage(
        "Fitting gradient-boosted tree model "
        f"(iterations={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}, "
        f"subsample={subsample}, min_samples_leaf={min_samples_leaf})..."
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    print_stage("Initial tree fit complete. Replaying staged metrics for every boosting iteration...")

    classes = model.classes_
    history: list[dict[str, float]] = []
    best_iteration = 1
    best_validation_accuracy = -np.inf
    best_validation_balanced_accuracy = -np.inf
    wait = 0
    early_stop_iteration = None
    found_checkpoint = False

    train_stages = model.staged_predict_proba(X_train)
    validation_stages = model.staged_predict_proba(X_validation)

    for iteration, (train_probabilities, validation_probabilities) in enumerate(
        zip(train_stages, validation_stages),
        start=1,
    ):
        train_accuracy, train_balanced_accuracy, train_loss, _ = compute_iteration_metrics(
            y_true=y_train,
            probabilities=train_probabilities,
            classes=classes,
        )
        validation_accuracy, validation_balanced_accuracy, validation_loss, _ = compute_iteration_metrics(
            y_true=y_validation,
            probabilities=validation_probabilities,
            classes=classes,
        )

        improved = (
            validation_balanced_accuracy > best_validation_balanced_accuracy + 1e-6
            or (
                abs(validation_balanced_accuracy - best_validation_balanced_accuracy) <= 1e-6
                and validation_accuracy > best_validation_accuracy + 1e-6
            )
        )

        history.append(
            {
                "iteration": iteration,
                "train_accuracy": train_accuracy,
                "train_balanced_accuracy": train_balanced_accuracy,
                "validation_accuracy": validation_accuracy,
                "validation_balanced_accuracy": validation_balanced_accuracy,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        if improved:
            best_iteration = iteration
            best_validation_accuracy = validation_accuracy
            best_validation_balanced_accuracy = validation_balanced_accuracy
            wait = 0
            found_checkpoint = True
        else:
            wait += 1

        if wait >= patience and early_stop_iteration is None:
            early_stop_iteration = iteration

        print_iteration_metrics(
            iteration=iteration,
            total_iterations=n_estimators,
            train_accuracy=train_accuracy,
            train_balanced_accuracy=train_balanced_accuracy,
            validation_accuracy=validation_accuracy,
            validation_balanced_accuracy=validation_balanced_accuracy,
            train_loss=train_loss,
            validation_loss=validation_loss,
            best_validation_accuracy=best_validation_accuracy,
            best_validation_balanced_accuracy=best_validation_balanced_accuracy,
            wait=wait,
            improved=improved,
            patience_reached=early_stop_iteration == iteration,
        )

    if not found_checkpoint:
        raise RuntimeError("Tree-model training failed to produce a usable checkpoint.")

    X_train_full = split_data.train_full[model_features].to_numpy(dtype=float)
    y_train_full = split_data.train_full["target"].to_numpy(dtype=int)
    X_test = split_data.test[model_features].to_numpy(dtype=float)
    y_test = split_data.test["target"].to_numpy(dtype=int)

    sample_weight_full, train_full_class_distribution = build_class_weighted_sample_weight(y_train_full)
    final_model = make_tree_model(
        n_estimators=best_iteration,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_leaf=min_samples_leaf,
        seed=seed,
    )
    print_stage(f"Refitting final tree model on the full training split using best_iteration={best_iteration}...")
    final_model.fit(X_train_full, y_train_full, sample_weight=sample_weight_full)

    full_train_probabilities = final_model.predict_proba(X_train_full)
    full_train_accuracy, full_train_balanced_accuracy, full_train_loss, full_train_predictions = compute_iteration_metrics(
        y_true=y_train_full,
        probabilities=full_train_probabilities,
        classes=final_model.classes_,
    )
    full_train_actionable_accuracy = compute_actionable_accuracy(y_train_full, full_train_predictions)

    test_probabilities = final_model.predict_proba(X_test)
    test_accuracy, test_balanced_accuracy, test_loss, test_predictions = compute_iteration_metrics(
        y_true=y_test,
        probabilities=test_probabilities,
        classes=final_model.classes_,
    )
    test_actionable_accuracy = compute_actionable_accuracy(y_test, test_predictions)

    metrics = {
        "best_iteration": int(best_iteration),
        "best_validation_accuracy": float(best_validation_accuracy),
        "best_validation_balanced_accuracy": float(best_validation_balanced_accuracy),
        "full_train_accuracy": float(full_train_accuracy),
        "full_train_balanced_accuracy": float(full_train_balanced_accuracy),
        "full_train_loss": float(full_train_loss),
        "full_train_actionable_accuracy": full_train_actionable_accuracy,
        "test_accuracy": float(test_accuracy),
        "test_balanced_accuracy": float(test_balanced_accuracy),
        "test_loss": float(test_loss),
        "test_actionable_accuracy": test_actionable_accuracy,
        "selection_train_accuracy": float(history[best_iteration - 1]["train_accuracy"]),
        "selection_train_balanced_accuracy": float(history[best_iteration - 1]["train_balanced_accuracy"]),
        "selection_train_loss": float(history[best_iteration - 1]["train_loss"]),
        "selection_validation_loss": float(history[best_iteration - 1]["validation_loss"]),
        "early_stop_iteration": None if early_stop_iteration is None else int(early_stop_iteration),
        "training_class_distribution": training_class_distribution,
        "train_full_class_distribution": train_full_class_distribution,
    }
    return final_model, history, metrics


def attach_probabilities_tree(
    frame,
    model: GradientBoostingClassifier,
    model_features: list[str],
):
    features = frame[model_features].to_numpy(dtype=float)
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


def get_top_feature_importances(
    model: GradientBoostingClassifier,
    model_features: list[str],
    top_n: int = 20,
) -> list[dict[str, float]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    order = np.argsort(importances)[::-1][:top_n]
    return [
        {
            "feature": model_features[int(index)],
            "importance": float(importances[int(index)]),
        }
        for index in order
    ]


def save_tree_model_bundle(
    output_path: Path,
    model: GradientBoostingClassifier,
    model_features: list[str],
    best_iteration: int,
    target_horizon: int,
    lookback_window: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    min_samples_leaf: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": model_features,
        "class_labels": [int(label) for label in model.classes_],
        "best_iteration": int(best_iteration),
        "target_horizon_hours": int(target_horizon),
        "lookback_window": int(lookback_window),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "min_samples_leaf": int(min_samples_leaf),
    }
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle)


def save_tree_parameter_report(
    output_path: Path,
    csv_path: Path,
    frame,
    split_data,
    model_features: list[str],
    class_distribution: dict[str, int],
    tree_metrics: dict[str, float],
    optimization_results: dict[str, object],
    n_estimators: int,
    patience: int,
    target_horizon: int,
    lookback_window: int,
    min_move_threshold: float,
    neutral_vol_multiplier: float,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    min_samples_leaf: int,
    top_feature_importances: list[dict[str, float]],
    history_output: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
            "type": "GradientBoostingClassifier",
            "feature_count": int(len(model_features)),
            "base_sequence_features": SEQUENCE_BASE_FEATURES,
            "n_estimators_requested": int(n_estimators),
            "best_iteration": int(tree_metrics["best_iteration"]),
            "early_stop_iteration": tree_metrics["early_stop_iteration"],
            "patience": int(patience),
            "learning_rate": float(learning_rate),
            "max_depth": int(max_depth),
            "subsample": float(subsample),
            "min_samples_leaf": int(min_samples_leaf),
            "selection_train_accuracy": float(tree_metrics["selection_train_accuracy"]),
            "selection_train_balanced_accuracy": float(tree_metrics["selection_train_balanced_accuracy"]),
            "selection_train_loss": float(tree_metrics["selection_train_loss"]),
            "selection_validation_loss": float(tree_metrics["selection_validation_loss"]),
            "best_validation_accuracy": float(tree_metrics["best_validation_accuracy"]),
            "best_validation_balanced_accuracy": float(tree_metrics["best_validation_balanced_accuracy"]),
            "full_train_accuracy": float(tree_metrics["full_train_accuracy"]),
            "full_train_balanced_accuracy": float(tree_metrics["full_train_balanced_accuracy"]),
            "full_train_loss": float(tree_metrics["full_train_loss"]),
            "full_train_actionable_accuracy": tree_metrics["full_train_actionable_accuracy"],
            "test_accuracy": float(tree_metrics["test_accuracy"]),
            "test_balanced_accuracy": float(tree_metrics["test_balanced_accuracy"]),
            "test_loss": float(tree_metrics["test_loss"]),
            "test_actionable_accuracy": tree_metrics["test_actionable_accuracy"],
            "training_class_distribution": tree_metrics["training_class_distribution"],
            "train_full_class_distribution": tree_metrics["train_full_class_distribution"],
            "top_feature_importances": top_feature_importances,
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
                "tree-model bullish and bearish class probabilities from a 3-class bull/neutral/bear model."
            ),
            "spread_proxy": (
                "The CSV has no bid/ask spread, so estimated_spread_pct uses 5% of the "
                "3-candle median intrabar range in percentage points."
            ),
            "positioning": "Long-only, one position at a time, using close-to-close returns.",
            "cost_model": "Half of the synthetic spread is charged on entry and exit.",
        },
        "artifacts": {
            "training_history_csv": str(history_output),
        },
        "default_parameters": DEFAULT_PARAMS,
        "optimized_parameters": optimization_results["best_parameters"],
        "optimized_train_metrics": optimization_results["best_train_metrics"],
        "optimized_test_metrics": optimization_results["best_test_metrics"],
        "default_train_metrics": optimization_results["default_train_metrics"],
        "default_test_metrics": optimization_results["default_test_metrics"],
        "search_candidates_evaluated": optimization_results["search_candidates_evaluated"],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv_path).resolve()
    params_output = Path(args.params_output).resolve()
    plot_output = Path(args.plot_output).resolve()
    history_output = Path(args.history_output).resolve()
    model_output = Path(args.model_output).resolve()

    print_stage(f"Loading dataset from {csv_path}...")
    frame, model_features, class_distribution = load_feature_frame(
        csv_path=csv_path,
        target_horizon=args.target_horizon,
        lookback_window=args.lookback_window,
        min_move_threshold=args.min_move_threshold,
        neutral_vol_multiplier=args.neutral_vol_multiplier,
    )
    print_stage(
        f"Feature engineering complete: rows={len(frame)}, feature_count={len(model_features)}, "
        f"class_distribution={class_distribution}"
    )
    split_data = build_splits(
        frame=frame,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    print_stage(
        f"Chronological split ready: train={len(split_data.train)}, "
        f"validation={len(split_data.validation)}, test={len(split_data.test)}"
    )

    model, history, tree_metrics = train_tree_classifier(
        split_data=split_data,
        model_features=model_features,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        min_samples_leaf=args.min_samples_leaf,
        patience=args.patience,
        seed=args.seed,
    )

    print_stage(f"Saving per-epoch history CSV to {history_output}...")
    save_tree_history_csv(
        history=history,
        output_path=history_output,
    )
    print_stage(f"Saving training graph to {plot_output}...")
    plot_tree_history(
        history=history,
        best_iteration=int(tree_metrics["best_iteration"]),
        test_accuracy=float(tree_metrics["test_accuracy"]),
        test_balanced_accuracy=float(tree_metrics["test_balanced_accuracy"]),
        early_stop_iteration=tree_metrics["early_stop_iteration"],
        output_path=plot_output,
    )

    train_scored = attach_probabilities_tree(split_data.train_full, model, model_features)
    test_scored = attach_probabilities_tree(split_data.test, model, model_features)
    optimization_results = optimize_parameters_with_progress(
        train_frame=train_scored,
        test_frame=test_scored,
        iterations=args.search_iterations,
        min_trades=args.min_trades,
        seed=args.seed,
    )

    top_feature_importances = get_top_feature_importances(model, model_features)

    print_stage(f"Saving trained tree model to {model_output}...")
    save_tree_model_bundle(
        output_path=model_output,
        model=model,
        model_features=model_features,
        best_iteration=int(tree_metrics["best_iteration"]),
        target_horizon=args.target_horizon,
        lookback_window=args.lookback_window,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        min_samples_leaf=args.min_samples_leaf,
    )
    print_stage(f"Saving parameter report to {params_output}...")
    save_tree_parameter_report(
        output_path=params_output,
        csv_path=csv_path,
        frame=frame,
        split_data=split_data,
        model_features=model_features,
        class_distribution=class_distribution,
        tree_metrics=tree_metrics,
        optimization_results=optimization_results,
        n_estimators=args.n_estimators,
        patience=args.patience,
        target_horizon=args.target_horizon,
        lookback_window=args.lookback_window,
        min_move_threshold=args.min_move_threshold,
        neutral_vol_multiplier=args.neutral_vol_multiplier,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        min_samples_leaf=args.min_samples_leaf,
        top_feature_importances=top_feature_importances,
        history_output=history_output,
    )

    print(f"Saved optimized parameters to: {params_output}")
    print(f"Saved tree accuracy plot to: {plot_output}")
    print(f"Saved tree history CSV to: {history_output}")
    print(f"Saved trained tree model bundle to: {model_output}")
    print("Best parameters:")
    for key, value in optimization_results["best_parameters"].items():
        print(f"  {key} = {value}")
    print("Tree model accuracy:")
    print(f"  Train accuracy: {tree_metrics['full_train_accuracy']:.4f}")
    print(f"  Test accuracy:  {tree_metrics['test_accuracy']:.4f}")
    print(f"  Test balanced accuracy: {tree_metrics['test_balanced_accuracy']:.4f}")
    print(f"  Test actionable accuracy: {tree_metrics['test_actionable_accuracy']:.4f}")


if __name__ == "__main__":
    main()
