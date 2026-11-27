"""Choose hyperparameters using inner leave-one-person-out validation.

Compare candidates by validation loss with class weights pooled over training people.
Choose refit epochs from the median best epoch and save the final choice in tuned.yaml.
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
from config.training import RunConfig
from config.loader import check_run_config
from training.fitting import FitConfig, History, fit_fold
from evaluation.records import BestSetting, Candidate, InnerFold, Setting, Tuning
from evaluation.metrics import macro_f1
from training.splitting import inner_splits

logger = logging.getLogger(__name__)


def hyperparameter_names(cfg: RunConfig) -> list[str]:
    """List tunable config paths that leave data selection and splits unchanged.

    Raises:
        ValueError: If the selected model is the majority-class baseline.
    """
    if cfg.model.name == "baseline":
        raise ValueError("tuning is only defined for trainable models, not baseline")
    names = [f"model.{section}.{f.name}" for section in ("arch", "augment")
             for f in fields(getattr(cfg.model, section))]
    names += [f"training.{f.name}" for f in fields(cfg.training)
              if f.name not in ("validation", "oversample_ratio")]
    return names


def with_setting(cfg: RunConfig, setting: Setting) -> RunConfig:
    """Return a validated config copy with one candidate's dotted overrides applied."""
    validate_grid([setting], hyperparameter_names(cfg))
    override = OmegaConf.create({})
    for key, value in setting.items():
        OmegaConf.update(override, key, value)
    merged = OmegaConf.merge(OmegaConf.structured(cfg), override)
    candidate = cast(RunConfig, OmegaConf.to_object(merged))
    check_run_config(candidate)
    return candidate


def validate_grid(grid: Sequence[Setting], allowed: Sequence[str]) -> None:
    """Check candidate keys against the allowed hyperparameters and reject duplicates.

    Raises:
        ValueError: If a key is unsupported or a setting appears more than once.
    """
    for setting in grid:
        unknown = sorted(set(setting) - set(allowed))
        if unknown:
            raise ValueError(f"unknown hyperparameter {unknown} in {setting}, choose from {sorted(allowed)}")
    if len({tuple(sorted(s.items())) for s in grid}) != len(grid):
        raise ValueError("the tuning grid has the same setting twice")


def best_epoch(history: History | None) -> int | None:
    """Return the one-based epoch containing the selected training step.

    Fall back to the last trained step. Return None if no training step exists.
    """
    if history is None:
        return None
    step = history.best_step or (history.train_steps[-1] if history.train_steps else None)
    return None if step is None else math.ceil(step / history.steps_per_epoch)


def kept_val_loss(history: History | None) -> float | None:
    """Return loss at the selected step, or the minimum recorded loss; None if unavailable."""
    if history is None or not history.val_loss:
        return None
    if history.best_step in history.val_steps:
        return history.val_loss[history.val_steps.index(history.best_step)]
    return min(history.val_loss)


def median_epoch(epochs: Sequence[int]) -> int:
    """Return the median epoch count, rounded up and bounded below by one."""
    return max(1, math.ceil(float(np.median(epochs))))


@dataclass
class Tuner:
    """Compare settings on inner person-wise folds and build the chosen refit model."""

    make_candidate: Callable[[Setting], FitConfig | BaseModel]  # Fresh model and fitting settings.
    grid: list[Setting]
    resampler: RandomOversampler | None = None  # Repeat inner training windows only.

    def __post_init__(self) -> None:
        """Reject an empty candidate grid before starting any training."""
        if not self.grid:
            raise ValueError("the tuning grid is empty")

    def tune(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray, train: np.ndarray,
             classes: np.ndarray) -> Tuning:
        """Choose settings using only people in the outer training fold.

        Args:
            x: Model inputs for the full dataset.
            y: String window labels for the full dataset.
            groups: Person names aligned with x and y.
            train: Outer training indices, excluding every outer test person.
            classes: Fixed label set used for informational validation macro F1.

        Returns:
            The setting with the lowest mean inner validation loss, its median best
            epoch count and all candidate results. Ties favour the earlier grid entry.

        Raises:
            ValueError: If inner splits need more people, a candidate has no validation
                loss, or no usable epoch count is recorded.
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
                                   val_weight_labels=y[train]).history  # Shared source for validation weights.
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

        best = min(candidates, key=lambda c: c.score)  # Equal scores keep the earlier grid entry.
        logger.info("tune: best setting %s, %d epochs", best.setting, best.epochs)
        return Tuning(best.setting, best.epochs, best.score, candidates)

    def _candidate(self, setting: Setting) -> FitConfig:
        """Construct a fresh candidate, wrapping bare models in default fitting settings."""
        candidate = self.make_candidate(setting)
        return candidate if isinstance(candidate, FitConfig) else FitConfig(candidate)

    def build(self, tuning: Tuning) -> FitConfig:
        """Build a fresh candidate with the winning overrides and selected epoch count."""
        return self._candidate({**tuning.setting, "training.epochs": tuning.epochs})


def best_overall(tunings: Sequence[Tuning]) -> BestSetting:
    """Choose the final setting by its number of outer-fold wins.

    Break ties using the mean validation loss of the wins. Use the median epoch
    count of those wins, rounded up. tunings must contain at least one result.
    """
    wins: dict[str, list[Tuning]] = {}
    for tuning in tunings:
        wins.setdefault(json.dumps(tuning.setting, sort_keys=True), []).append(tuning)
    key = min(wins, key=lambda k: (-len(wins[k]), float(np.mean([t.score for t in wins[k]]))))
    won = wins[key]
    return BestSetting(won[0].setting, median_epoch([t.epochs for t in won]), len(won), len(tunings),
                       float(np.mean([t.score for t in won])))


def save_tuned(path: Path, best: BestSetting, model: str, windowing: dict[str, Any]) -> None:
    """Save the winning setting, epoch count, model name and window signature as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"model": model, "windowing": dict(windowing), **asdict(best)}), path)


def apply_tuned(cfg: RunConfig, tuned: dict[str, Any]) -> RunConfig:
    """Return a config copy with the saved winner and epoch count applied.

    Raises:
        ValueError: If model or window settings differ, or the saved overrides
            include unsupported hyperparameters or invalid values.
    """
    saved = tuned.get("windowing", tuned.get("window_spec"))
    if tuned.get("model") != cfg.model.name or saved != cfg.windowing.signature():
        raise ValueError("the tuned model or windowing settings differ from this run; tune again")
    return with_setting(cfg, {**tuned["setting"], "training.epochs": tuned["epochs"]})



def load_tuned(path: Path) -> dict[str, Any]:
    """Load a YAML file written by save_tuned.

    Raises:
        FileNotFoundError: If the file is missing; the message explains how to tune.
    """
    if not path.is_file():
        raise FileNotFoundError(f"no tuned config at {path}; run tuning first (python main.py tune 'tune.grid=[...]')")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]
