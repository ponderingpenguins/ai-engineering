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
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import OmegaConf

from data.resample import RandomOversampler
from models.model import BaseModel
from config import RunConfig, check_run_config
from training.fitting import FitConfig, History, fit_fold
from evaluation.records import BestSetting, Candidate, InnerFold, Setting, Tuning
from evaluation.metrics import macro_f1
from training.splitting import inner_splits

logger = logging.getLogger(__name__)


def hyperparameter_names(cfg: RunConfig) -> list[str]:
    """Config paths that can vary between candidate models without changing splits."""
    if cfg.model.name == "baseline":
        raise ValueError("tuning is only defined for trainable models, not baseline")
    names = [f"model.{section}.{f.name}" for section in ("arch", "augment")
             for f in fields(getattr(cfg.model, section))]
    names += [f"training.{f.name}" for f in fields(cfg.training)
              if f.name not in ("validation", "oversample_ratio")]
    return names


def with_setting(cfg: RunConfig, setting: Setting) -> RunConfig:
    """Apply typed dotted overrides to one candidate; the original config is untouched."""
    validate_grid([setting], hyperparameter_names(cfg))
    override = OmegaConf.create({})
    for key, value in setting.items():
        OmegaConf.update(override, key, value)
    merged = OmegaConf.merge(OmegaConf.structured(cfg), override)
    candidate = cast(RunConfig, OmegaConf.to_object(merged))
    check_run_config(candidate)
    return candidate


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
class Tuner:
    make_candidate: Callable[[Setting], FitConfig | BaseModel]  # fresh model and fitting settings
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
                candidate = self._candidate(setting)
                model = candidate.model
                history = fit_fold(model, x, y, fit_idx, val_idx, training=candidate.training,
                                   augment=candidate.augment, seed=candidate.seed, resampler=self.resampler,
                                   val_weight_labels=y[train]).history  # the same weights for every person
                predicted = model.predict(x[val_idx])
                epoch, loss = best_epoch(history), kept_val_loss(history)
                if loss is None:
                    raise ValueError("the model reports no validation loss, so settings cannot be compared")
                folds.append(InnerFold(str(groups[val_idx][0]), epoch, loss, macro_f1(y[val_idx], predicted, classes)))
                losses.append(loss)
                true.append(y[val_idx])
                pred.append(predicted)
                if epoch is not None:
                    epochs.append(epoch)
            if not epochs:
                raise ValueError("the model reports no best epoch, so no epoch count can be chosen")
            score, f1 = float(np.mean(losses)), macro_f1(np.concatenate(true), np.concatenate(pred), classes)
            candidates.append(Candidate(setting, score, median_epoch(epochs), folds, f1,
                                        dict(Counter(np.concatenate(pred).tolist()))))
            logger.info("tune %d/%d %s: val loss %.4f (macro_f1 %s), best epochs %s", number, len(self.grid), setting,
                        score, "N/A" if f1 is None else f"{f1:.3f}", epochs)

        best = min(candidates, key=lambda c: c.score)  # min keeps the first tie
        logger.info("tune: best setting %s, %d epochs", best.setting, best.epochs)
        return Tuning(best.setting, best.epochs, best.score, candidates)

    def _candidate(self, setting: Setting) -> FitConfig:
        candidate = self.make_candidate(setting)
        return candidate if isinstance(candidate, FitConfig) else FitConfig(candidate)

    def build(self, tuning: Tuning) -> FitConfig:
        """A fresh model and fitting settings, using the winning setting and chosen epoch count."""
        return self._candidate({**tuning.setting, "training.epochs": tuning.epochs})


def best_overall(tunings: Sequence[Tuning]) -> BestSetting:
    """The setting to use for the final model, from the tuning of every outer fold.

    The setting that won the most outer folds. Ties go to the lower mean validation loss of its wins. Its
    epoch count is the median of the epochs it won with.
    """
    wins: dict[str, list[Tuning]] = {}
    for tuning in tunings:
        wins.setdefault(json.dumps(tuning.setting, sort_keys=True), []).append(tuning)
    key = min(wins, key=lambda k: (-len(wins[k]), float(np.mean([t.score for t in wins[k]]))))
    won = wins[key]
    return BestSetting(won[0].setting, median_epoch([t.epochs for t in won]), len(won), len(tunings),
                       float(np.mean([t.score for t in won])))


def save_tuned(path: Path, best: BestSetting, model: str, windowing: dict[str, Any]) -> None:
    """Writes the chosen setting with its model and window specification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"model": model, "windowing": dict(windowing), **asdict(best)}), path)


def apply_tuned(cfg: RunConfig, tuned: dict[str, Any]) -> RunConfig:
    """The model config with a tuned setting and epoch count applied, for the full run.

    Raises:
        ValueError: If the setting was tuned for a different model or window specification than the current run, or names
            a hyperparameter the model does not have.
    """
    saved = tuned.get("windowing", tuned.get("window_spec"))
    if tuned.get("model") != cfg.model.name or saved != cfg.windowing.signature():
        raise ValueError("the tuned model or windowing settings differ from this run; tune again")
    return with_setting(cfg, {**tuned["setting"], "training.epochs": tuned["epochs"]})



def load_tuned(path: Path) -> dict[str, Any]:
    """Reads a file written by `save_tuned`.

    Raises:
        FileNotFoundError: If the file does not exist, with a hint to run the tuning first.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no tuned config at {path}; run tuning first (python main.py tune 'tune.grid=[...]')")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]
