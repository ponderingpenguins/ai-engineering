"""Person-wise k-fold cross-validation, with the results saved next to the config."""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from models.model import BaseModel
from utils.metrics import accuracy, macro_f1, per_class

logger = logging.getLogger(__name__)


def group_folds(groups: np.ndarray, k: int) -> list[np.ndarray]:
    """Plain k-fold over the groups. All windows of a group end up in the same fold.

    With k equal to the number of groups this is leave-one-group-out.
    """
    names = np.unique(groups)
    if not 2 <= k <= len(names):
        raise ValueError(f"folds must be between 2 and the number of groups ({len(names)}), got {k}")
    return np.array_split(names, k)


@dataclass
class Trainer:
    make_model: Callable[[], BaseModel]  # called once per fold, so every fold starts from scratch
    metrics: dict[str, Callable] = field(default_factory=lambda: {"accuracy": accuracy, "macro_f1": macro_f1})
    out_dir: Path | None = None  # results are saved here when set
    config: dict[str, Any] | None = None  # saved together with the results

    def cross_validate(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int) -> dict[str, Any]:
        """Each fold is held out once as test data. Returns per-fold and pooled results."""
        folds = group_folds(groups, k)
        all_true, all_pred = [], []
        fold_results = []

        for i, held_out in enumerate(folds, start=1):
            test = np.isin(groups, held_out)
            logger.info("fold %d/%d: test on %s", i, k, ", ".join(sorted(held_out)))

            model = self.make_model()
            model.fit(x[~test], y[~test])  # no val data: the test fold must not pick the best epoch
            pred = model.predict(x[test])

            scores = {name: fn(y[test], pred) for name, fn in self.metrics.items()}
            logger.info("fold %d/%d: n=%d %s", i, k, test.sum(),
                        " ".join(f"{name}={score:.3f}" for name, score in scores.items()))
            fold_results.append({
                "fold": i,
                "test_groups": sorted(held_out),
                "n_train": int((~test).sum()),
                "n_test": int(test.sum()),
                "test_classes": dict(Counter(y[test].tolist())),
                "metrics": scores,
            })
            all_true.append(y[test])
            all_pred.append(pred)

        # score all the held-out predictions together: a single fold may hold almost none of the rare class
        true, pred = np.concatenate(all_true), np.concatenate(all_pred)
        pooled: dict[str, Any] = {
            "n": len(true),
            "metrics": {name: fn(true, pred) for name, fn in self.metrics.items()},
            "per_class": per_class(true, pred),
            "predicted_classes": dict(Counter(pred.tolist())),
        }
        logger.info("pooled: n=%d %s", len(true), " ".join(f"{name}={score:.3f}" for name, score in pooled["metrics"].items()))
        for name, row in pooled["per_class"].items():
            logger.info("  %-12s %s", name, row)

        results = {"config": self.config, "folds": fold_results, "pooled": pooled}
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
