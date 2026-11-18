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

from models.model import BaseModel
from utils.metrics import accuracy, macro_f1, per_class, scores
from utils.splitter import GroupKFoldSplitter

logger = logging.getLogger(__name__)

Metric = Callable[[np.ndarray, np.ndarray, np.ndarray], float | None]  # (y_true, y_pred, classes)


def fmt(value: float | None) -> str:
    """Three decimals, or N/A for an undefined metric."""
    return "N/A" if value is None else f"{value:.3f}"


@dataclass
class Trainer:
    make_model: Callable[[], BaseModel]  # called once per fold, so every fold starts from scratch
    splitter: GroupKFoldSplitter = field(default_factory=GroupKFoldSplitter)
    metrics: dict[str, Metric] = field(default_factory=lambda: {"accuracy": accuracy, "macro_f1": macro_f1})
    out_dir: Path | None = None  # results are saved here when set
    config: dict[str, Any] | None = None  # saved together with the results

    def cross_validate(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       collect_predictions: bool = False) -> dict[str, Any]:
        """Each fold is held out once as test data. Returns per-fold, pooled and per-person results.

        Every metric is scored against the classes of the whole dataset, so a fold without a rare class
        is scored like the others. The pooled score over all held-out predictions is the headline number.
        """
        folds = self.splitter.split(y, groups)
        k = len(folds)
        classes = np.unique(y)
        all_true, all_pred, all_groups = [], [], []
        fold_results = []
        predictions = []

        for i, (train, test) in enumerate(folds, start=1):
            held_out = np.unique(groups[test])
            logger.info("fold %d/%d: test on %s", i, k, ", ".join(sorted(held_out)))

            model = self.make_model()
            started = perf_counter()
            model.fit(x[train], y[train])  # no val data: the test fold must not pick the best epoch
            train_seconds = perf_counter() - started
            started = perf_counter()
            pred = model.predict(x[test])
            predict_seconds = perf_counter() - started
            if collect_predictions:
                predictions.extend({"sample_id": int(j), "fold": i,
                                    "true_label": str(y[j]), "predicted_label": str(p)}
                                   for j, p in zip(test, pred))

            fold_scores = {name: fn(y[test], pred, classes) for name, fn in self.metrics.items()}
            logger.info("fold %d/%d: n=%d %s", i, k, len(test),
                        " ".join(f"{name}={fmt(score)}" for name, score in fold_scores.items()))
            fold_results.append({
                "fold": i,
                "test_groups": sorted(held_out),
                "n_train": len(train),
                "n_test": len(test),
                "test_classes": dict(Counter(y[test].tolist())),
                "metrics": fold_scores,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            })
            all_true.append(y[test])
            all_pred.append(pred)
            all_groups.append(groups[test])

        # score all the held-out predictions together: a single fold may hold almost none of the rare class
        true, pred, held_groups = np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_groups)
        pooled: dict[str, Any] = {
            "n": len(true),
            "metrics": {name: fn(true, pred, classes) for name, fn in self.metrics.items()},
            "per_class": per_class(true, pred, classes),
            "predicted_classes": dict(Counter(pred.tolist())),
        }
        logger.info("pooled: n=%d %s", len(true), " ".join(f"{n}={fmt(s)}" for n, s in pooled["metrics"].items()))
        for name, row in pooled["per_class"].items():
            logger.info("  %-12s %s", name, row)

        # how each held-out person does, with the support behind the number
        per_person = {
            str(person): {"n": int(np.sum(held_groups == person)),
                          **scores(true[held_groups == person], pred[held_groups == person], classes)}
            for person in np.unique(held_groups)
        }
        for person, row in per_person.items():
            logger.info("  person %-10s n=%d %s", person, row["n"], {key: value for key, value in row.items() if key.endswith(("_recall", "_support"))})

        results = {"config": self.config, "folds": fold_results, "pooled": pooled, "per_person": per_person}
        if collect_predictions:
            results["predictions"] = predictions
        self._save(results)
        return results

    def _save(self, results: dict[str, Any]) -> None:
        """Writes the results to `out_dir/results.json`, if `out_dir` is set.

        Args:
            results: The JSON-serializable results.
        """
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "results.json").write_text(json.dumps(results, indent=2))
        logger.info("saved results to %s", self.out_dir)
