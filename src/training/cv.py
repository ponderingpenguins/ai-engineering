"""Person-wise k-fold cross-validation, with the results saved next to the config."""

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
from training.fit import fit_fold
from evaluation.metrics import accuracy, fmt, macro_f1, per_class, scores
from evaluation.records import FinalResult, FoldResult, Pooled, RunResults
from evaluation.report import write_report
from training.splits import GroupKFoldSplitter, validation_split
from training.tuner import Tuner, best_overall

logger = logging.getLogger(__name__)

Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float | None]  # (y_true, y_pred, classes)


@dataclass
class Trainer:
    make_model: Callable[[], BaseModel]  # called once per fold, so every fold starts from scratch
    splitter: GroupKFoldSplitter = field(default_factory=GroupKFoldSplitter)
    metrics: dict[str, Metric] = field(default_factory=lambda: {"accuracy": accuracy, "macro_f1": macro_f1})
    out_dir: Path | None = None  # results, checkpoints, loss curves and the report are saved here when set
    config: dict[str, Any] | None = None  # saved together with the results
    windowing: dict[str, Any] | None = None  # stored in each checkpoint
    seed: int = 0  # picks the validation person of each fold
    resampler: RandomOversampler | None = None  # applied to each fold's training windows only
    validate: bool = True  # False: no validation person, the model trains on all training people for its epochs
    tuner: Tuner | None = None  # set: each fold tunes on its training people, then refits on all of them

    def cross_validate(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> RunResults:
        """Each fold is held out once as test data. Returns per-fold, pooled and per-person results.

        Every metric is scored against the classes of the whole dataset, so a fold without a rare class
        is scored like the others. The pooled score over all held-out predictions is the headline number.
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
            if self.tuner is not None:  # inner leave-one-person-out picks the setting and epochs, then refit on all
                tuning = self.tuner.tune(x, y, groups, train, classes)
                fit_idx, val_idx = train, None
            else:  # one training person is held out for early stopping; the test person is never used for that
                fit_idx, val_idx = validation_split(train, groups, self.seed + i) if self.validate else (train, None)
            val_people = [] if val_idx is None else sorted(np.unique(groups[val_idx]))
            if val_idx is None and self.validate and tuning is None:
                logger.warning("fold %d/%d: fewer than two training people, no validation data", i, k)
            elif val_idx is not None:
                logger.info("fold %d/%d: validate on %s", i, k, ", ".join(val_people))

            if tuning is None:
                model = self.make_model()
            else:
                assert self.tuner is not None
                model = self.tuner.build(tuning)
            fit = fit_fold(model, x, y, fit_idx, val_idx, resampler=self.resampler, log_as=f"fold {i}/{k}")
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

        # score all the held-out predictions together: a single fold may hold almost none of the rare class
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

        # how each held-out person does, with the support behind the number
        per_person = {
            str(person): {"n": int(np.sum(held_groups == person)),
                          **scores(true[held_groups == person], pred[held_groups == person], classes)}
            for person in np.unique(held_groups)
        }
        for person, row in per_person.items():
            logger.info("  person %-10s n=%d %s", person, row["n"], {key: value for key, value in row.items() if key.endswith(("_recall", "_support"))})

        results = RunResults(self.config, fold_results, pooled, per_person)
        if self.tuner is not None:  # the setting to use for the final model, see training.tuner.best_overall
            results.tuned = best_overall([fold.tuning for fold in fold_results if fold.tuning])
            logger.info("tuned: %s", results.tuned)
        self._save(results)
        return results

    def train_final(self, x: np.ndarray, y: np.ndarray) -> FinalResult:
        """Trains one model on all windows, with no validation and no held-out score.

        The model comes from `make_model`, which should already carry the tuned setting and epoch count.
        The result is saved as `final.pt` and `final.json` in `out_dir`, when it and `windowing` are set.
        """
        model = self.make_model()
        fit = fit_fold(model, x, y, np.arange(len(y)), resampler=self.resampler)
        final = FinalResult(config=self.config, n_windows=len(y), n_train_after_resampling=len(fit.train_idx),
                            train_classes=dict(Counter(y[fit.train_idx].tolist())), train_seconds=fit.seconds,
                            history=fit.history, checkpoint=None)
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            if self.windowing is not None:
                model.save(self.out_dir / "final.pt", self.windowing)
                final.checkpoint = "final.pt"
            (self.out_dir / "final.json").write_text(json.dumps(final.to_dict(), indent=2))
        logger.info("final model trained on %d windows in %.1f s; checkpoint %s", len(fit.train_idx), fit.seconds,
                    final.checkpoint)
        return final

    def _save_checkpoint(self, model: BaseModel, fold: int, held_out: np.ndarray) -> str | None:
        """Saves the fold's trained model (its best weights) under `out_dir/checkpoints`.

        Returns the path relative to `out_dir`, or None if nothing is saved. Load it with
        `BaseModel.load`.
        """
        if self.out_dir is None or self.windowing is None:
            return None
        relative = Path("checkpoints") / f"fold{fold}_{'+'.join(sorted(held_out))}.pt"
        model.save(self.out_dir / relative, self.windowing)
        return str(relative)

    def _save(self, results: RunResults) -> None:
        """Writes `results.json`, the loss curves and the report to `out_dir`, if it is set."""
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "results.json").write_text(json.dumps(results.to_dict(), indent=2))
        logger.info("saved results to %s", self.out_dir)
        report = write_report(results, self.out_dir)
        logger.info("report: %s", report)
