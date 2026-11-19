"""Loss-curve figures for a cross-validation run, drawn from the histories saved in `results.json`."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID_PER_EPOCH = 10  # points per epoch on the common axis of the train curve in the mean figure


def fold_histories(folds: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The `(fold, history)` pairs of the folds that have a loss history (iterative models only)."""
    return [(fold, fold["history"]) for fold in folds if fold.get("history")]


def _title(fold: dict[str, Any]) -> str:
    val = f", val {'+'.join(fold['val_groups'])}" if fold.get("val_groups") else ""
    return f"Fold {fold['fold']}: test {'+'.join(fold['test_groups'])}{val}"


def plot_fold_curves(folds: list[dict[str, Any]], path: Path) -> Path | None:
    """One panel per fold: train loss at every step, val loss at every evaluation.

    A dashed line marks the step whose weights were kept (the lowest val loss).

    Returns:
        The path of the figure, or None if no fold has a loss history.
    """
    pairs = fold_histories(folds)
    if not pairs:
        return None
    cols = min(2, len(pairs))
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.6 * rows), squeeze=False, sharey=True)
    for ax, (fold, history) in zip(axes.flat, pairs):
        ax.plot(history["train_steps"], history["train_loss"], color="tab:blue", alpha=0.45, lw=0.8, label="train (every step)")
        if history["val_loss"]:
            ax.plot(history["val_steps"], history["val_loss"], color="tab:orange", marker="o", ms=3, lw=1.4,
                    label="validation")
        if history["best_step"] is not None:
            ax.axvline(history["best_step"], color="gray", ls="--", lw=1, label="kept weights")
        if history["stopped_early"]:
            ax.axvline(history["train_steps"][-1], color="tab:red", ls=":", lw=1, label="early stop")
        ax.set(title=_title(fold), xlabel="Training step", ylabel="Weighted cross-entropy")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _on_grid(x: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Values interpolated onto `grid`; NaN where the curve does not reach (a fold that stopped earlier)."""
    return np.interp(grid, x, values, left=np.nan, right=np.nan)


def mean_curves(folds: list[dict[str, Any]]) -> dict[str, np.ndarray] | None:
    """Mean and standard deviation over folds of the train and val loss, on an epoch axis.

    Folds have different numbers of steps per epoch and stop at different times, so every curve is put on
    an epoch axis first. A point is only used where at least two folds have a value.

    - Train loss: averaged over one epoch of steps (a single step is too noisy), then put on a grid with
      `GRID_PER_EPOCH` points per epoch.
    - Val loss: only the epochs where it was really computed, so no value is interpolated.

    Returns:
        Arrays `epoch`, `train_mean`, `train_std` and `val_epoch`, `val_mean`, `val_std`, or None if no
        fold has a loss history.
    """
    pairs = fold_histories(folds)
    if not pairs:
        return None
    last_epoch = max(h["train_steps"][-1] / h["steps_per_epoch"] for _, h in pairs)
    grid = np.arange(1, int(np.ceil(last_epoch * GRID_PER_EPOCH)) + 1) / GRID_PER_EPOCH
    val_epochs = np.unique(np.round([step / h["steps_per_epoch"] for _, h in pairs for step in h["val_steps"]], 2))

    def reduce(curves: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
        stack = np.vstack(curves) if curves else np.full((1, width), np.nan)
        n = np.sum(~np.isnan(stack), axis=0)
        with np.errstate(all="ignore"):
            mean, std = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
        mean[n < 2], std[n < 2] = np.nan, np.nan
        return mean, std

    train, val = [], []
    for _, h in pairs:
        spe = h["steps_per_epoch"]
        smooth = np.asarray(pd.Series(h["train_loss"]).rolling(spe, min_periods=1).mean())
        train.append(_on_grid(np.asarray(h["train_steps"]) / spe, smooth, grid))
        if h["val_loss"]:
            at = np.round(np.asarray(h["val_steps"]) / spe, 2)
            val.append(np.array([h["val_loss"][list(at).index(e)] if e in at else np.nan for e in val_epochs]))
    train_mean, train_std = reduce(train, len(grid))
    val_mean, val_std = reduce(val, len(val_epochs))
    return {"epoch": grid, "train_mean": train_mean, "train_std": train_std,
            "val_epoch": val_epochs, "val_mean": val_mean, "val_std": val_std}


def plot_mean_curves(folds: list[dict[str, Any]], path: Path) -> Path | None:
    """Train and val loss against epochs, as the mean over folds with a shaded band of ±1 standard deviation.

    Returns:
        The path of the figure, or None if no fold has a loss history.
    """
    curves = mean_curves(folds)
    if curves is None:
        return None
    n_total = len(fold_histories(folds))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for name, color, label, x, marker in (
        ("train", "tab:blue", "train (mean of 1 epoch of steps)", curves["epoch"], None),
        ("val", "tab:orange", "validation", curves["val_epoch"], "o"),
    ):
        mean, std = curves[f"{name}_mean"], curves[f"{name}_std"]
        if np.all(np.isnan(mean)):
            continue
        ax.plot(x, mean, color=color, label=label, lw=1.6, marker=marker, ms=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, lw=0)
    ax.set(xlabel="Epoch", ylabel="Weighted cross-entropy",
           title=f"Loss over {n_total} folds (mean, shaded band = ±1 std across folds)")
    ax.text(0.99, 0.97, "a point needs at least 2 folds still training", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="gray")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
