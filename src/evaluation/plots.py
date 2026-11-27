"""Draw fold and mean loss curves from saved training histories.

Use a non-interactive plotting backend so figures can be saved without a display.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.fitting import History
from evaluation.records import FoldResult

GRID_PER_EPOCH = 10  # Training-curve samples per epoch on the shared plotting grid.


def fold_histories(folds: list[FoldResult]) -> list[tuple[FoldResult, History]]:
    """Return (fold, history) pairs only for folds with iterative training histories."""
    return [(fold, fold.history) for fold in folds if fold.history]


def _title(fold: FoldResult) -> str:
    """Format the fold number and held-out people for a plot title."""
    val = f", val {'+'.join(fold.val_groups)}" if fold.val_groups else ""
    return f"Fold {fold.fold}: test {'+'.join(fold.test_groups)}{val}"


def plot_fold_curves(folds: list[FoldResult], path: Path) -> Path | None:
    """Save one panel per fold, showing training and validation loss by step.

    Mark the selected weights with a dashed line and early stopping with a
    dotted line when those steps are recorded.

    Returns:
        Figure path, or None when no fold has a training history.
    """
    pairs = fold_histories(folds)
    if not pairs:
        return None
    cols = min(2, len(pairs))
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 3.6 * rows), squeeze=False, sharey=True)
    for ax, (fold, history) in zip(axes.flat, pairs):
        ax.plot(history.train_steps, history.train_loss, color="tab:blue", alpha=0.45, lw=0.8, label="train (every step)")
        if history.val_loss:
            ax.plot(history.val_steps, history.val_loss, color="tab:orange", marker="o", ms=3, lw=1.4,
                    label="validation")
        if history.best_step is not None:
            ax.axvline(history.best_step, color="gray", ls="--", lw=1, label="kept weights")
        if history.stopped_early:
            ax.axvline(history.train_steps[-1], color="tab:red", ls=":", lw=1, label="early stop")
        ax.set(title=_title(fold), xlabel="Training step", ylabel="Weighted cross-entropy")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(pairs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _on_grid(x: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate a curve onto grid, using NaN outside its recorded range."""
    return np.interp(grid, x, values, left=np.nan, right=np.nan)


def mean_curves(folds: list[FoldResult]) -> dict[str, np.ndarray] | None:
    """Compute loss means and standard deviations across folds on an epoch axis.

    Smooth training loss over one epoch of steps, then interpolate it onto a
    common grid. Keep validation loss only at recorded evaluation epochs.
    Each point needs values from at least two folds; others remain NaN.

    Returns:
        Arrays named epoch, train_mean, train_std, val_epoch, val_mean and val_std,
        or None when no fold has a training history.
    """
    pairs = fold_histories(folds)
    if not pairs:
        return None
    last_epoch = max(h.train_steps[-1] / h.steps_per_epoch for _, h in pairs)
    grid = np.arange(1, int(np.ceil(last_epoch * GRID_PER_EPOCH)) + 1) / GRID_PER_EPOCH
    val_epochs = np.unique(np.round([step / h.steps_per_epoch for _, h in pairs for step in h.val_steps], 2))

    def reduce(curves: list[np.ndarray], width: int) -> tuple[np.ndarray, np.ndarray]:
        """Compute column means and deviations only where at least two folds contribute."""
        stack = np.vstack(curves) if curves else np.full((1, width), np.nan)
        n = np.sum(~np.isnan(stack), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Empty columns remain NaN.
            mean, std = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
        mean[n < 2], std[n < 2] = np.nan, np.nan
        return mean, std

    train, val = [], []
    for _, h in pairs:
        spe = h.steps_per_epoch
        smooth = np.asarray(pd.Series(h.train_loss).rolling(spe, min_periods=1).mean())
        train.append(_on_grid(np.asarray(h.train_steps) / spe, smooth, grid))
        if h.val_loss:
            at = np.round(np.asarray(h.val_steps) / spe, 2)
            val.append(np.array([h.val_loss[list(at).index(e)] if e in at else np.nan for e in val_epochs]))
    train_mean, train_std = reduce(train, len(grid))
    val_mean, val_std = reduce(val, len(val_epochs))
    return {"epoch": grid, "train_mean": train_mean, "train_std": train_std,
            "val_epoch": val_epochs, "val_mean": val_mean, "val_std": val_std}


def plot_mean_curves(folds: list[FoldResult], path: Path) -> Path | None:
    """Save mean training and validation losses with a one-standard-deviation band.

    Returns:
        Figure path, or None when no fold has a training history.
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
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
