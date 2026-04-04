"""
ML-style parameter optimizer for TEST-bot.

This script uses the offline simulator in performance_check.py as the scoring
engine, searches for better strategy parameters, and writes the best result to
separate output files.

The optimizer is pair-aware, so you can search on:
    - BTC only
    - ETH only
    - any subset of supported pairs
    - all configured pairs

Examples:
    python ml_parameter_optimizer.py --pairs BTC --weeks 4
    python ml_parameter_optimizer.py --pairs BTC ETH --iterations 8
    python ml_parameter_optimizer.py --pairs BTC/USD --initial-samples 4 --iterations 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import LOG_DIR
from performance_check import (
    DEFAULT_COMMISSION,
    DEFAULT_STARTING_BALANCE,
    align_pair_data,
    clone_pair_data_for_timeline,
    merge_parameter_snapshot,
    progress,
    resolve_pairs,
    run_simulation,
    train_initial_hmms,
)


PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "entry_hmm_confidence": (0.35, 0.98),
    "entry_max_volatility": (0.10, 1.50),
    "entry_momentum_min": (-0.03, 0.03),
    "entry_volume_zscore_max": (0.75, 6.00),
    "exit_stop_loss_pct": (0.002, 0.030),
    "exit_take_profit_pct": (0.004, 0.060),
    "mc_max_position_pct": (0.05, 0.40),
    "mc_min_position_pct": (0.01, 0.15),
    "mc_tail_risk_limit": (0.005, 0.050),
}
PARAM_KEYS = list(PARAM_BOUNDS.keys())
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize TEST-bot parameters with a surrogate model.")
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Optional pair list such as BTC BTC/USD ETH/USD.",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="Recent evaluation window size.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.50,
        help="Fraction of the recent timeline used for training search statistics.",
    )
    parser.add_argument(
        "--validation-frac",
        type=float,
        default=0.25,
        help="Fraction of the recent timeline used for validation scoring.",
    )
    parser.add_argument(
        "--starting-balance",
        type=float,
        default=DEFAULT_STARTING_BALANCE,
        help="Starting USD capital for each simulation.",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=DEFAULT_COMMISSION,
        help="Per-side execution cost used by the simulator.",
    )
    parser.add_argument(
        "--initial-samples",
        type=int,
        default=6,
        help="Number of seed parameter sets to evaluate before surrogate steps.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of surrogate optimization iterations.",
    )
    parser.add_argument(
        "--surrogate-pool",
        type=int,
        default=48,
        help="Number of cheap surrogate candidates sampled per iteration.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.18,
        help="Base surrogate step size. Lower is slower and usually safer.",
    )
    parser.add_argument(
        "--exploration-scale",
        type=float,
        default=0.25,
        help="Initial exploration noise around the best candidate.",
    )
    parser.add_argument(
        "--search-mc-simulations",
        type=int,
        default=300,
        help="Monte Carlo paths used during the search loop to keep runtime manageable.",
    )
    parser.add_argument(
        "--final-mc-simulations",
        type=int,
        default=1000,
        help="Monte Carlo paths used for the final best-parameter report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible search steps.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh Binance data before running if network is available.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path for the optimizer summary JSON.",
    )
    parser.add_argument(
        "--output-config",
        default=None,
        help="Optional path for the generated config-style parameter file.",
    )
    parser.add_argument(
        "--output-history-csv",
        default=None,
        help="Optional path for the candidate history CSV.",
    )
    return parser.parse_args()


def pair_slug(pairs: List[str]) -> str:
    return "_".join(pair.replace("/", "_") for pair in pairs)


def default_output_paths(pairs: List[str]) -> Dict[str, str]:
    slug = pair_slug(pairs)
    return {
        "json": os.path.join(SCRIPT_DIR, LOG_DIR, f"optimized_params_{slug}.json"),
        "config": os.path.join(SCRIPT_DIR, LOG_DIR, f"optimized_params_{slug}.py"),
        "history_csv": os.path.join(SCRIPT_DIR, LOG_DIR, f"optimizer_history_{slug}.csv"),
    }


def runtime_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(SCRIPT_DIR, path)


def split_timeline(
    timeline: List[str],
    train_frac: float,
    validation_frac: float,
) -> Tuple[List[str], List[str], List[str]]:
    if not (0.2 <= train_frac <= 0.8):
        raise ValueError("train_frac must be between 0.2 and 0.8")
    if not (0.1 <= validation_frac <= 0.4):
        raise ValueError("validation_frac must be between 0.1 and 0.4")
    if train_frac + validation_frac >= 0.95:
        raise ValueError("train_frac + validation_frac must leave room for a test split")

    total = len(timeline)
    train_end = int(total * train_frac)
    valid_end = int(total * (train_frac + validation_frac))

    if train_end < 200 or valid_end - train_end < 100 or total - valid_end < 100:
        raise ValueError("Timeline split is too small. Increase weeks or adjust fractions.")

    return timeline[:train_end], timeline[train_end:valid_end], timeline[valid_end:]


def vector_from_params(params: Dict[str, float]) -> np.ndarray:
    merged = merge_parameter_snapshot(params)
    vector = []
    for key in PARAM_KEYS:
        lo, hi = PARAM_BOUNDS[key]
        value = merged[key]
        vector.append((value - lo) / (hi - lo))
    return np.array(vector, dtype=float)


def params_from_vector(vector: np.ndarray, template: Dict[str, float] | None = None) -> Dict[str, float]:
    vector = np.clip(vector, 0.0, 1.0)
    params = merge_parameter_snapshot(template)
    for idx, key in enumerate(PARAM_KEYS):
        lo, hi = PARAM_BOUNDS[key]
        params[key] = lo + float(vector[idx]) * (hi - lo)
    return merge_parameter_snapshot(params)


def build_feature_matrix(x: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.ones((len(x), 1)),
            x,
            x ** 2,
        ],
        axis=1,
    )


def fit_surrogate_model(x: np.ndarray, y: np.ndarray, ridge_alpha: float = 1e-3) -> np.ndarray:
    phi = build_feature_matrix(x)
    regularizer = ridge_alpha * np.eye(phi.shape[1])
    regularizer[0, 0] = 0.0
    lhs = phi.T @ phi + regularizer
    rhs = phi.T @ y
    return np.linalg.pinv(lhs) @ rhs


def predict_surrogate(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return build_feature_matrix(x) @ weights


def evaluate_simulation(
    pair_data: Dict[str, Dict[str, object]],
    timeline: List[str],
    hmms: Dict[str, object],
    params: Dict[str, float],
    starting_balance: float,
    commission: float,
    seed: int,
    mc_simulations: int,
) -> Dict[str, object]:
    eval_params = dict(params)
    eval_params["mc_num_simulations"] = mc_simulations

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return run_simulation(
            pair_data=pair_data,
            timeline=timeline,
            starting_balance=starting_balance,
            commission=commission,
            params=eval_params,
            hmms=hmms,
            print_orders=False,
            progress_enabled=False,
        )
    finally:
        np.random.set_state(state)


def objective_score(train_results: Dict[str, object], valid_results: Dict[str, object]) -> float:
    train_comp = float(train_results["ratio_summary"]["composite_score"])
    valid_comp = float(valid_results["ratio_summary"]["composite_score"])
    valid_return = float(valid_results["ratio_summary"]["total_return_pct"])
    valid_profit_factor = float(valid_results["trade_summary"]["profit_factor"])
    valid_trades = int(valid_results["trade_summary"]["closed_trades"])

    if not np.isfinite(train_comp):
        train_comp = -1e6
    if not np.isfinite(valid_comp):
        valid_comp = -1e6
    if not np.isfinite(valid_return):
        valid_return = -1e6
    if not np.isfinite(valid_profit_factor):
        valid_profit_factor = 0.0

    overfit_gap = max(0.0, train_comp - valid_comp)
    low_trade_penalty = max(0.0, 5 - valid_trades) * 0.25

    score = (
        valid_comp
        + 0.12 * valid_return
        + 0.10 * min(valid_profit_factor, 3.0)
        - 0.25 * overfit_gap
        - low_trade_penalty
    )
    return round(float(score), 6)


def summarize_results(results: Dict[str, object], prefix: str) -> Dict[str, float]:
    ratios = results["ratio_summary"]
    trades = results["trade_summary"]
    return {
        f"{prefix}_final_balance": results["final_balance"],
        f"{prefix}_total_return_pct": ratios["total_return_pct"],
        f"{prefix}_composite_score": ratios["composite_score"],
        f"{prefix}_sharpe_ratio": ratios["sharpe_ratio"],
        f"{prefix}_sortino_ratio": ratios["sortino_ratio"],
        f"{prefix}_calmar_ratio": ratios["calmar_ratio"],
        f"{prefix}_max_drawdown_pct": ratios["max_drawdown_pct"],
        f"{prefix}_realized_pnl": trades["realized_pnl"],
        f"{prefix}_profit_factor": trades["profit_factor"],
        f"{prefix}_closed_trades": trades["closed_trades"],
        f"{prefix}_win_rate_pct": trades["win_rate_pct"],
    }


def write_config_file(
    path: str,
    pairs: List[str],
    best_record: Dict[str, object],
    baseline_record: Dict[str, object],
) -> None:
    params = best_record["params"]
    lines = [
        '"""Auto-generated optimized parameters from ml_parameter_optimizer.py."""',
        "",
        f"PAIR_SCOPE = {pairs!r}",
        "",
        f"ENTRY_HMM_CONFIDENCE = {params['entry_hmm_confidence']:.6f}",
        f"ENTRY_MAX_VOLATILITY = {params['entry_max_volatility']:.6f}",
        f"ENTRY_MOMENTUM_MIN = {params['entry_momentum_min']:.6f}",
        f"ENTRY_VOLUME_ZSCORE_MAX = {params['entry_volume_zscore_max']:.6f}",
        f"ENTRY_SPREAD_MAX_PCT = {params['entry_spread_max_pct']:.6f}",
        f"EXIT_STOP_LOSS_PCT = {params['exit_stop_loss_pct']:.6f}",
        f"EXIT_TAKE_PROFIT_PCT = {params['exit_take_profit_pct']:.6f}",
        f"MC_MAX_POSITION_PCT = {params['mc_max_position_pct']:.6f}",
        f"MC_MIN_POSITION_PCT = {params['mc_min_position_pct']:.6f}",
        f"MC_TAIL_RISK_LIMIT = {params['mc_tail_risk_limit']:.6f}",
        "",
        f"EXPECTED_TEST_TOTAL_RETURN_PCT = {best_record['test_total_return_pct']:.6f}",
        f"EXPECTED_TEST_REALIZED_PNL = {best_record['test_realized_pnl']:.6f}",
        f"EXPECTED_TEST_COMPOSITE_SCORE = {best_record['test_composite_score']:.6f}",
        f"BASELINE_TEST_TOTAL_RETURN_PCT = {baseline_record['test_total_return_pct']:.6f}",
        f"BASELINE_TEST_COMPOSITE_SCORE = {baseline_record['test_composite_score']:.6f}",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    selected_pairs = resolve_pairs(args.pairs)
    outputs = default_output_paths(selected_pairs)
    output_json = runtime_path(args.output_json) if args.output_json else outputs["json"]
    output_config = runtime_path(args.output_config) if args.output_config else outputs["config"]
    output_history_csv = runtime_path(args.output_history_csv) if args.output_history_csv else outputs["history_csv"]

    rng = np.random.default_rng(args.seed)
    progress("[start] ml_parameter_optimizer.py")
    progress(f"[setup] Pair scope: {', '.join(selected_pairs)}")

    pair_data_full, timeline_full, notes = align_pair_data(
        selected_pairs,
        weeks=args.weeks,
        refresh_data=args.refresh_data,
        progress_enabled=True,
    )
    train_timeline, valid_timeline, test_timeline = split_timeline(
        timeline_full,
        args.train_frac,
        args.validation_frac,
    )

    train_pair_data = clone_pair_data_for_timeline(pair_data_full, train_timeline)
    valid_pair_data = clone_pair_data_for_timeline(pair_data_full, valid_timeline)
    test_pair_data = clone_pair_data_for_timeline(pair_data_full, test_timeline)

    progress(
        "[split] "
        f"train={len(train_timeline)} bars | validation={len(valid_timeline)} bars | "
        f"test={len(test_timeline)} bars"
    )

    progress("[hmm] Pretraining reusable HMMs for each split...")
    train_hmms = train_initial_hmms(train_pair_data, progress_enabled=True)
    valid_hmms = train_initial_hmms(valid_pair_data, progress_enabled=True)
    test_hmms = train_initial_hmms(test_pair_data, progress_enabled=True)

    history: List[Dict[str, object]] = []
    seen_vectors: List[np.ndarray] = []
    baseline_params = merge_parameter_snapshot()

    def evaluate_candidate(params: Dict[str, float], source: str, step: int) -> Dict[str, object]:
        merged = merge_parameter_snapshot(params)
        vector = vector_from_params(merged)

        train_results = evaluate_simulation(
            train_pair_data,
            train_timeline,
            train_hmms,
            merged,
            args.starting_balance,
            args.commission,
            args.seed + 11,
            args.search_mc_simulations,
        )
        valid_results = evaluate_simulation(
            valid_pair_data,
            valid_timeline,
            valid_hmms,
            merged,
            args.starting_balance,
            args.commission,
            args.seed + 17,
            args.search_mc_simulations,
        )
        objective = objective_score(train_results, valid_results)

        record: Dict[str, object] = {
            "candidate_id": len(history) + 1,
            "step": step,
            "source": source,
            "objective": objective,
            "params": merged,
            "vector": vector,
        }
        record.update(summarize_results(train_results, "train"))
        record.update(summarize_results(valid_results, "valid"))
        history.append(record)
        seen_vectors.append(vector)

        progress(
            "[eval] "
            f"#{record['candidate_id']:02d} {source:<12} | objective={objective:>8.4f} | "
            f"valid_return={record['valid_total_return_pct']:+7.3f}% | "
            f"valid_comp={record['valid_composite_score']:>8.4f} | "
            f"valid_pnl=${record['valid_realized_pnl']:>10,.2f}"
        )
        return record

    progress("[baseline] Evaluating current config...")
    baseline_record = evaluate_candidate(baseline_params, source="baseline", step=0)

    progress("[search] Evaluating initial samples...")
    for sample_idx in range(max(0, args.initial_samples - 1)):
        if sample_idx % 2 == 0:
            center = vector_from_params(baseline_params)
            candidate_vector = np.clip(center + rng.normal(0.0, 0.20, size=len(PARAM_KEYS)), 0.0, 1.0)
        else:
            candidate_vector = rng.uniform(0.0, 1.0, size=len(PARAM_KEYS))
        evaluate_candidate(
            params_from_vector(candidate_vector, baseline_params),
            source="seed_random",
            step=sample_idx + 1,
        )

    for iteration in range(args.iterations):
        best_record = max(history, key=lambda item: item["objective"])
        x = np.vstack([record["vector"] for record in history])
        y = np.array([record["objective"] for record in history], dtype=float)
        weights = fit_surrogate_model(x, y)

        current_lr = args.learning_rate / math.sqrt(iteration + 1)
        current_noise = args.exploration_scale / math.sqrt(iteration + 1)

        pool_vectors = []
        best_vector = best_record["vector"]
        top_vectors = [record["vector"] for record in sorted(history, key=lambda item: item["objective"], reverse=True)[:3]]

        for _ in range(args.surrogate_pool):
            if rng.random() < 0.5:
                seed_vector = top_vectors[rng.integers(0, len(top_vectors))]
                pool_vectors.append(np.clip(seed_vector + rng.normal(0.0, current_noise, size=len(PARAM_KEYS)), 0.0, 1.0))
            else:
                pool_vectors.append(rng.uniform(0.0, 1.0, size=len(PARAM_KEYS)))

        pool_matrix = np.vstack(pool_vectors)
        pool_scores = predict_surrogate(weights, pool_matrix)
        target_vector = pool_matrix[int(np.argmax(pool_scores))]

        exploit_vector = np.clip(
            best_vector + current_lr * (target_vector - best_vector) + rng.normal(0.0, current_noise * 0.20, size=len(PARAM_KEYS)),
            0.0,
            1.0,
        )
        explore_vector = np.clip(
            best_vector + rng.normal(0.0, current_noise, size=len(PARAM_KEYS)),
            0.0,
            1.0,
        )

        candidates = [
            ("surrogate_step", exploit_vector),
            ("exploration", explore_vector),
        ]

        progress(
            "[search] "
            f"iteration={iteration + 1}/{args.iterations} | "
            f"lr={current_lr:.4f} | noise={current_noise:.4f} | "
            f"best_objective={best_record['objective']:.4f}"
        )

        for label, vector in candidates:
            if any(np.linalg.norm(vector - seen) < 1e-3 for seen in seen_vectors):
                vector = rng.uniform(0.0, 1.0, size=len(PARAM_KEYS))
            evaluate_candidate(
                params_from_vector(vector, baseline_params),
                source=label,
                step=args.initial_samples + iteration,
            )

    best_record = max(history, key=lambda item: item["objective"])
    progress("[final] Evaluating baseline and best candidate on the holdout test split...")

    baseline_test_results = evaluate_simulation(
        test_pair_data,
        test_timeline,
        test_hmms,
        baseline_record["params"],
        args.starting_balance,
        args.commission,
        args.seed + 23,
        args.final_mc_simulations,
    )
    best_test_results = evaluate_simulation(
        test_pair_data,
        test_timeline,
        test_hmms,
        best_record["params"],
        args.starting_balance,
        args.commission,
        args.seed + 29,
        args.final_mc_simulations,
    )

    baseline_record.update(summarize_results(baseline_test_results, "test"))
    best_record.update(summarize_results(best_test_results, "test"))

    top_records = sorted(history, key=lambda item: item["objective"], reverse=True)[:5]
    json_summary = {
        "meta": {
            "pairs": selected_pairs,
            "weeks": args.weeks,
            "train_frac": args.train_frac,
            "validation_frac": args.validation_frac,
            "search_mc_simulations": args.search_mc_simulations,
            "final_mc_simulations": args.final_mc_simulations,
            "seed": args.seed,
            "notes": notes,
        },
        "baseline": {
            key: value for key, value in baseline_record.items() if key != "vector"
        },
        "best": {
            key: value for key, value in best_record.items() if key != "vector"
        },
        "top_candidates": [
            {key: value for key, value in record.items() if key != "vector"}
            for record in top_records
        ],
    }

    history_rows = []
    for record in history:
        row = {key: value for key, value in record.items() if key not in {"params", "vector"}}
        row.update(record["params"])
        history_rows.append(row)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    os.makedirs(os.path.dirname(output_config), exist_ok=True)
    os.makedirs(os.path.dirname(output_history_csv), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(json_summary, handle, indent=2)
    write_config_file(output_config, selected_pairs, best_record, baseline_record)
    pd.DataFrame(history_rows).to_csv(output_history_csv, index=False)

    progress("=" * 72)
    progress("ML Parameter Optimizer")
    progress("=" * 72)
    progress(f"Pairs: {', '.join(selected_pairs)}")
    progress(
        "Baseline test: "
        f"return={baseline_record['test_total_return_pct']:+.4f}% | "
        f"composite={baseline_record['test_composite_score']:.6f} | "
        f"pnl=${baseline_record['test_realized_pnl']:,.2f}"
    )
    progress(
        "Best test:     "
        f"return={best_record['test_total_return_pct']:+.4f}% | "
        f"composite={best_record['test_composite_score']:.6f} | "
        f"pnl=${best_record['test_realized_pnl']:,.2f}"
    )
    progress("Best parameters:")
    for key in PARAM_KEYS:
        progress(f"  {key}: {best_record['params'][key]}")
    progress(f"[done] Wrote summary to {output_json}")
    progress(f"[done] Wrote config snippet to {output_config}")
    progress(f"[done] Wrote history to {output_history_csv}")


if __name__ == "__main__":
    main()
