"""Evaluate models on held-out people and save fold and pooled results.

Each fold optionally tunes or holds out a training person for early stopping.
Save checkpoints, results.json and a report when an output directory is supplied.
"""

import json
import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from data.resample import RandomOversampler
from models.model import BaseModel
from config.training import TrainingConfig
from training.fitting import BatchAugment, FitConfig, fit_fold
from evaluation.metrics import accuracy, fmt, macro_f1, per_class, scores
from evaluation.records import FoldResult, Pooled, RunResults
from evaluation.report import write_report
from training.splitting import GroupKFoldSplitter, validation_split
from training.tuning import Tuner, best_overall

logger = logging.getLogger(__name__)

Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float | None]  # (y_true, y_pred, classes)


@dataclass
class CrossValidator:
    """Fit fresh models on person-wise folds and collect held-out predictions."""

    make_model: Callable[[], BaseModel]  # Create a fresh model for each untuned fold.
    splitter: GroupKFoldSplitter = field(default_factory=GroupKFoldSplitter)
    metrics: dict[str, Metric] = field(default_factory=lambda: {"accuracy": accuracy, "macro_f1": macro_f1})
    out_dir: Path | None = None  # Optional destination for results, checkpoints and reports.
    config: dict[str, Any] | None = None  # Run settings saved with the results.
    windowing: dict[str, Any] | None = None  # Window signature stored in each checkpoint.
    seed: int = 0  # Base seed for choosing validation people and fitting untuned models.
    resampler: RandomOversampler | None = None  # Repeat training windows only.
    validate: bool = True  # For untuned folds, reserve one training person for early stopping.
    tuner: Tuner | None = None  # Tune within each training fold, then refit on all its people.

    training: TrainingConfig | None = None
    augment: BatchAugment | None = None

    def cross_validate(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> RunResults:
        """Evaluate each outer test fold once and combine its held-out predictions.

        Use the full dataset's class list for every metric. Hyperparameter tuning
        and early stopping use training people only.

        Args:
            x: Model inputs with one window per row.
            y: String window labels of shape (N,).
            groups: Person names of shape (N,), aligned with x and y.

        Returns:
            Per-fold, pooled and per-person results, with tuning details when enabled.
            Save artifacts and a report if out_dir is set.
        """
        folds = self.splitter.split(y, groups)
        k = len(folds)
        classes = np.unique(y)
        all_true, all_pred, all_groups = [], [], []
        fold_results = []

        for i, (train, test) in enumerate(folds, start=1):
            held_out = np.unique(groups[test])
            logger.info("fold %d/%d: test on %s", i, k, ", ".join(sorted(held_out)))

            tuning = None
            if self.tuner is not None:  # Choose settings and epochs using training people only.
                tuning = self.tuner.tune(x, y, groups, train, classes)
                fit_idx, val_idx = train, None
            else:  # Optionally reserve one training person for early stopping.
                fit_idx, val_idx = validation_split(train, groups, self.seed + i) if self.validate else (train, None)
            val_people = [] if val_idx is None else sorted(np.unique(groups[val_idx]))
            if val_idx is None and self.validate and tuning is None:
                logger.warning("fold %d/%d: fewer than two training people, no validation data", i, k)
            elif val_idx is not None:
                logger.info("fold %d/%d: validate on %s", i, k, ", ".join(val_people))

            if tuning is None:
                candidate = FitConfig(self.make_model(), self.training, self.augment, self.seed)
            else:
                assert self.tuner is not None
                candidate = self.tuner.build(tuning)
            model = candidate.model
            fit = fit_fold(model, x, y, fit_idx, val_idx, training=candidate.training, augment=candidate.augment,
                           seed=candidate.seed, resampler=self.resampler, log_as=f"fold {i}/{k}")
            train_idx = fit.train_idx
            checkpoint = self._save_checkpoint(model, i, held_out)
            started = perf_counter()
            pred = model.predict(x[test])
            predict_seconds = perf_counter() - started
            fold_scores = {name: fn(y[test], pred, classes) for name, fn in self.metrics.items()}
            logger.info("fold %d/%d: n=%d %s", i, k, len(test),
                        " ".join(f"{name}={fmt(score)}" for name, score in fold_scores.items()))
            fold_results.append(FoldResult(
                fold=i,
                test_groups=sorted(held_out),
                val_groups=val_people,
                n_train=len(fit_idx),
                n_train_after_resampling=len(train_idx),
                train_classes=dict(Counter(y[train_idx].tolist())),
                n_val=0 if val_idx is None else len(val_idx),
                n_test=len(test),
                test_classes=dict(Counter(y[test].tolist())),
                metrics=fold_scores,
                per_class=per_class(y[test], pred, classes),
                history=fit.history,
                tuning=tuning,
                checkpoint=checkpoint,
                train_seconds=fit.seconds,
                predict_seconds=predict_seconds,
            ))
            all_true.append(y[test])
            all_pred.append(pred)
            all_groups.append(groups[test])

        # Pool held-out predictions so the main score includes every person's rare-class windows.
        true, pred, held_groups = np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_groups)
        pooled = Pooled(
            n=len(true),
            metrics={name: fn(true, pred, classes) for name, fn in self.metrics.items()},
            per_class=per_class(true, pred, classes),
            predicted_classes=dict(Counter(pred.tolist())),
        )
        logger.info("pooled: n=%d %s", len(true), " ".join(f"{n}={fmt(s)}" for n, s in pooled.metrics.items()))
        for name, row in pooled.per_class.items():
            logger.info("  %-12s %s", name, row)

        # Keep per-person metrics with class support to show how much evidence each has.
        per_person = {
            str(person): {"n": int(np.sum(held_groups == person)),
                          **scores(true[held_groups == person], pred[held_groups == person], classes)}
            for person in np.unique(held_groups)
        }
        for person, row in per_person.items():
            logger.info("  person %-10s n=%d %s", person, row["n"], {key: value for key, value in row.items() if key.endswith(("_recall", "_support"))})

        results = RunResults(self.config, fold_results, pooled, per_person)
        if self.tuner is not None:  # Choose the saved setting for final training.
            results.tuned = best_overall([fold.tuning for fold in fold_results if fold.tuning])
            logger.info("tuned: %s", results.tuned)
        self._save(results)
        return results

    def _save_checkpoint(self, model: BaseModel, fold: int, held_out: np.ndarray) -> str | None:
        """Save the fold's fitted model under out_dir/checkpoints.

        Returns:
            A path relative to out_dir, or None when output or window settings are
            missing. Restore the checkpoint with BaseModel.load.
        """
        if self.out_dir is None or self.windowing is None:
            return None
        relative = Path("checkpoints") / f"fold{fold}_{'+'.join(sorted(held_out))}.pt"
        model.save(self.out_dir / relative, self.windowing)
        return str(relative)

    def _save(self, results: RunResults) -> None:
        """Write results.json, loss plots and report.md when out_dir is set."""
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "results.json").write_text(json.dumps(results.to_dict(), indent=2))
        logger.info("saved results to %s", self.out_dir)
        report = write_report(results, self.out_dir)
        logger.info("report: %s", report)
