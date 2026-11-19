"""Hyperparameter tuning inside one training fold: the inner loop of nested cross-validation.

For an outer fold, every setting of the grid is trained with leave-one-person-out over the fold's training
people (the outer test person is never in it). The setting with the best pooled validation macro-F1 wins,
and the epoch count is the median best epoch of its inner folds. The caller then refits on all the training
people with `Tuner.build`, which has no validation person left and so trains for exactly that many epochs.
"""

import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from data.resample import RandomOversampler
from models.model import BaseModel, History
from utils.metrics import macro_f1
from utils.splitter import inner_splits

logger = logging.getLogger(__name__)

Setting = dict[str, Any]  # overrides of the model's hyperparameters, e.g. {"hidden": 64, "lr": 3e-4}


def validate_grid(grid: Sequence[Setting], allowed: Sequence[str]) -> None:
    """Checks that every setting only names hyperparameters the model has.

    Raises:
        ValueError: If a setting has an unknown key, or the same setting appears twice.
    """
    for setting in grid:
        unknown = sorted(set(setting) - set(allowed))
        if unknown:
            raise ValueError(f"unknown hyperparameter {unknown} in {setting}, choose from {sorted(allowed)}")
    if len({tuple(sorted(s.items())) for s in grid}) != len(grid):
        raise ValueError("the tuning grid has the same setting twice")


def best_epoch(history: History | None) -> int | None:
    """The epoch (1-based) whose weights were kept, or the last epoch trained if none were chosen."""
    if history is None:
        return None
    step = history.best_step or (history.train_steps[-1] if history.train_steps else None)
    return None if step is None else math.ceil(step / history.steps_per_epoch)


def median_epoch(epochs: Sequence[int]) -> int:
    """The median of the best epochs, rounded up, and at least 1."""
    return max(1, math.ceil(float(np.median(epochs))))


@dataclass
class Tuning:
    """The outcome of tuning one outer fold."""

    setting: Setting  # the winning setting
    epochs: int  # median best epoch of the winner's inner folds
    score: float | None  # the winner's pooled validation macro-F1
    candidates: list[dict[str, Any]] = field(default_factory=list)  # every setting with its inner results

    def to_dict(self) -> dict[str, Any]:
        """Plain values, ready for JSON."""
        return {"setting": self.setting, "epochs": self.epochs, "score": self.score, "candidates": self.candidates}


@dataclass
class Tuner:
    make_model: Callable[[Setting], BaseModel]  # a fresh, untrained model with the setting applied
    grid: list[Setting]
    resampler: RandomOversampler | None = None  # applied to the inner fit windows only, like in the trainer

    def __post_init__(self) -> None:
        if not self.grid:
            raise ValueError("the tuning grid is empty")

    def tune(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray, train: np.ndarray,
             classes: np.ndarray) -> Tuning:
        """Picks the best setting using only the people in `train`.

        Args:
            x, y, groups: All windows, their labels and their people.
            train: Indices of the outer training fold. The outer test person must not be in it.
            classes: The fixed class set the validation macro-F1 is scored against.

        Returns:
            The winning setting, its epoch count and every candidate's inner results. Ties go to the
            earlier setting in the grid. A setting whose macro-F1 is undefined ranks last.
        """
        inner = inner_splits(train, groups)
        candidates = []
        for number, setting in enumerate(self.grid, start=1):
            true, pred, folds, epochs = [], [], [], []
            for fit_idx, val_idx in inner:
                if self.resampler is not None:
                    fit_idx = fit_idx[self.resampler.indices(y[fit_idx])]
                model = self.make_model(setting)
                history = model.fit(x[fit_idx], y[fit_idx], val=(x[val_idx], y[val_idx]))
                predicted = model.predict(x[val_idx])
                epoch = best_epoch(history)
                person = str(groups[val_idx][0])
                folds.append({"val_group": person, "best_epoch": epoch,
                              "macro_f1": macro_f1(y[val_idx], predicted, classes)})
                true.append(y[val_idx])
                pred.append(predicted)
                if epoch is not None:
                    epochs.append(epoch)
            if not epochs:
                raise ValueError("the model reports no best epoch, so no epoch count can be chosen")
            score = macro_f1(np.concatenate(true), np.concatenate(pred), classes)
            candidates.append({"setting": setting, "score": score, "epochs": median_epoch(epochs), "folds": folds,
                               "predicted_classes": dict(Counter(np.concatenate(pred).tolist()))})
            logger.info("tune %d/%d %s: val macro_f1=%s, best epochs %s", number, len(self.grid), setting,
                        "N/A" if score is None else f"{score:.3f}", epochs)

        winner = max(range(len(candidates)), key=lambda i: -math.inf if candidates[i]["score"] is None
                     else candidates[i]["score"])  # max keeps the first of equal scores
        best = candidates[winner]
        logger.info("tune: best setting %s, %d epochs", best["setting"], best["epochs"])
        return Tuning(best["setting"], best["epochs"], best["score"], candidates)

    def build(self, tuning: Tuning) -> BaseModel:
        """A fresh model with the winning setting, set to train for the chosen number of epochs."""
        return self.make_model({**tuning.setting, "epochs": tuning.epochs})
