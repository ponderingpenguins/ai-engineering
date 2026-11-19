"""Hyperparameter tuning inside one training fold: the inner loop of nested cross-validation.

For an outer fold, every setting of the grid is trained with leave-one-person-out over the fold's training
people (the outer test person is never in it). The setting with the lowest mean class-balanced validation
loss (at the weights early stopping kept) wins, and the epoch count is the median best epoch of its inner
folds. The class weights of that loss are pooled over all the fold's training people, the same for every validation
person and setting, so the losses are comparable. The caller then refits on all the training people with `Tuner.build`, which has no validation person
left and so trains for exactly that many epochs.

Across the outer folds, `best_overall` picks the setting to use for the final model, and `save_tuned` /
`load_tuned` hand it to a later full run.
"""

import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from data.resample import RandomOversampler
from models.model import BaseModel, History
from utils.config import ModelConfig, RNNConfig
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


def kept_val_loss(history: History | None) -> float | None:
    """The validation loss at the weights that were kept (the lowest one if no step was marked)."""
    if history is None or not history.val_loss:
        return None
    if history.best_step in history.val_steps:
        return history.val_loss[history.val_steps.index(history.best_step)]
    return min(history.val_loss)


def median_epoch(epochs: Sequence[int]) -> int:
    """The median of the best epochs, rounded up, and at least 1."""
    return max(1, math.ceil(float(np.median(epochs))))


@dataclass
class Tuning:
    """The outcome of tuning one outer fold."""

    setting: Setting  # the winning setting
    epochs: int  # median best epoch of the winner's inner folds
    score: float  # the winner's mean class-balanced validation loss over its inner folds (lower is better)
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
            classes: The fixed class set the (informational) validation macro-F1 is scored against.

        Returns:
            The winning setting, its epoch count and every candidate's inner results. The winner has the
            lowest mean validation loss; ties go to the earlier setting in the grid.

        Raises:
            ValueError: If the model reports no validation loss or no best epoch.
        """
        inner = inner_splits(train, groups)
        candidates = []
        for number, setting in enumerate(self.grid, start=1):
            true, pred, folds, epochs, losses = [], [], [], [], []
            for fit_idx, val_idx in inner:
                if self.resampler is not None:
                    fit_idx = fit_idx[self.resampler.indices(y[fit_idx])]
                model = self.make_model(setting)
                history = model.fit(x[fit_idx], y[fit_idx], val=(x[val_idx], y[val_idx]), val_weight_labels=y[train])
                predicted = model.predict(x[val_idx])
                epoch, loss = best_epoch(history), kept_val_loss(history)
                if loss is None:
                    raise ValueError("the model reports no validation loss, so settings cannot be compared")
                folds.append({"val_group": str(groups[val_idx][0]), "best_epoch": epoch, "val_loss": loss,
                              "macro_f1": macro_f1(y[val_idx], predicted, classes)})
                losses.append(loss)
                true.append(y[val_idx])
                pred.append(predicted)
                if epoch is not None:
                    epochs.append(epoch)
            if not epochs:
                raise ValueError("the model reports no best epoch, so no epoch count can be chosen")
            score, f1 = float(np.mean(losses)), macro_f1(np.concatenate(true), np.concatenate(pred), classes)
            candidates.append({"setting": setting, "score": score, "epochs": median_epoch(epochs), "folds": folds,
                               "macro_f1": f1, "predicted_classes": dict(Counter(np.concatenate(pred).tolist()))})
            logger.info("tune %d/%d %s: val loss %.4f (macro_f1 %s), best epochs %s", number, len(self.grid), setting,
                        score, "N/A" if f1 is None else f"{f1:.3f}", epochs)

        winner = min(range(len(candidates)), key=lambda i: candidates[i]["score"])  # min keeps the first tie
        best = candidates[winner]
        logger.info("tune: best setting %s, %d epochs", best["setting"], best["epochs"])
        return Tuning(best["setting"], best["epochs"], best["score"], candidates)

    def build(self, tuning: Tuning) -> BaseModel:
        """A fresh model with the winning setting, set to train for the chosen number of epochs."""
        return self.make_model({**tuning.setting, "epochs": tuning.epochs})


def best_overall(tunings: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The setting to use for the final model, from the `Tuning.to_dict()` of every outer fold.

    The setting that won the most outer folds. Ties go to the lower mean validation loss of its wins. Its
    epoch count is the median of the epochs it won with.
    """
    wins: dict[str, list[dict[str, Any]]] = {}
    for tuning in tunings:
        wins.setdefault(json.dumps(tuning["setting"], sort_keys=True), []).append(tuning)
    key = min(wins, key=lambda k: (-len(wins[k]), float(np.mean([t["score"] for t in wins[k]]))))
    won = wins[key]
    return {"setting": won[0]["setting"], "epochs": median_epoch([t["epochs"] for t in won]),
            "wins": len(won), "of_folds": len(tunings), "mean_val_loss": float(np.mean([t["score"] for t in won]))}


def save_tuned(path: Path, best: dict[str, Any], model: str, task: dict[str, Any]) -> None:
    """Writes the chosen setting to a YAML file, with the model and task it was tuned for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"model": model, "task": task, **best}), path)


def apply_tuned(model_cfg: ModelConfig, tuned: dict[str, Any], task: dict[str, Any]) -> ModelConfig:
    """The model config with a tuned setting and epoch count applied, for the full run.

    Raises:
        ValueError: If the setting was tuned for a different model or task than the current run, or names
            a hyperparameter the model does not have.
    """
    if tuned.get("model") != model_cfg.name or tuned.get("task") != task:
        raise ValueError(f"the tuned config is for model {tuned.get('model')!r} and task {tuned.get('task')}, "
                         f"but this run is {model_cfg.name!r} with {task}; tune again")
    validate_grid([tuned["setting"]], [f.name for f in fields(RNNConfig)])
    return replace(model_cfg, rnn=replace(model_cfg.rnn, **{**tuned["setting"], "epochs": tuned["epochs"]}))


def load_tuned(path: Path) -> dict[str, Any]:
    """Reads a file written by `save_tuned`.

    Raises:
        FileNotFoundError: If the file does not exist, with a hint to run the tuning first.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no tuned config at {path}; run the tuning first (set tune.grid)")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]
