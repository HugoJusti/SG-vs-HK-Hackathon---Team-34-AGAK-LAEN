from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_tree_history_csv(
    history: list[dict[str, float]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_path, index=False)
    return output_path


def plot_tree_history(
    history: list[dict[str, float]],
    best_iteration: int,
    test_accuracy: float,
    test_balanced_accuracy: float,
    early_stop_iteration: int | None,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [row["iteration"] for row in history]
    train_accuracy = [row["train_accuracy"] for row in history]
    validation_accuracy = [row["validation_accuracy"] for row in history]
    train_balanced_accuracy = [row["train_balanced_accuracy"] for row in history]
    validation_balanced_accuracy = [row["validation_balanced_accuracy"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["validation_loss"] for row in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax_accuracy, ax_loss = axes

    ax_accuracy.plot(iterations, train_accuracy, label="Train accuracy", linewidth=2.0, color="#1d4ed8")
    ax_accuracy.plot(iterations, validation_accuracy, label="Validation accuracy", linewidth=2.0, color="#15803d")
    ax_accuracy.plot(
        iterations,
        train_balanced_accuracy,
        label="Train balanced accuracy",
        linewidth=1.5,
        linestyle="--",
        color="#60a5fa",
    )
    ax_accuracy.plot(
        iterations,
        validation_balanced_accuracy,
        label="Validation balanced accuracy",
        linewidth=1.5,
        linestyle="--",
        color="#4ade80",
    )
    ax_accuracy.axvline(best_iteration, color="#b45309", linestyle="--", linewidth=1.5, label=f"Best iteration = {best_iteration}")
    if early_stop_iteration is not None:
        ax_accuracy.axvline(
            early_stop_iteration,
            color="#dc2626",
            linestyle="-.",
            linewidth=1.2,
            label=f"Patience threshold = {early_stop_iteration}",
        )
    ax_accuracy.axhline(test_accuracy, color="#7c3aed", linestyle=":", linewidth=1.5, label=f"Test accuracy = {test_accuracy:.3f}")
    ax_accuracy.axhline(
        test_balanced_accuracy,
        color="#f97316",
        linestyle=":",
        linewidth=1.5,
        label=f"Test balanced accuracy = {test_balanced_accuracy:.3f}",
    )
    ax_accuracy.set_title("Tree Model Accuracy Across Boosting Iterations")
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.set_ylim(0.0, 1.0)
    ax_accuracy.grid(alpha=0.25)
    ax_accuracy.legend()

    ax_loss.plot(iterations, train_loss, label="Train loss", linewidth=2.0, color="#2563eb")
    ax_loss.plot(iterations, validation_loss, label="Validation loss", linewidth=2.0, color="#dc2626")
    ax_loss.axvline(best_iteration, color="#b45309", linestyle="--", linewidth=1.5)
    if early_stop_iteration is not None:
        ax_loss.axvline(
            early_stop_iteration,
            color="#dc2626",
            linestyle="-.",
            linewidth=1.2,
        )
    ax_loss.set_title("Tree Model Loss Across Boosting Iterations")
    ax_loss.set_xlabel("Boosting iteration")
    ax_loss.set_ylabel("Log loss")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path
