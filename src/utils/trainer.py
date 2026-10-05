import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from models.model import BaseModel
from utils.metrics import accuracy, macro_f1

logger = logging.getLogger(__name__)
Data = tuple[np.ndarray, np.ndarray]

@dataclass
class Trainer:
    model: BaseModel
    metrics: dict[str, Callable] = field(default_factory=lambda: {"accuracy": accuracy, "macro_f1": macro_f1})

    def run(self, train: Data, evals: dict[str, Data]) -> dict[str, dict[str, float]]:
        """Fit on train, score on every split in `evals`."""
        self.model.fit(*train, val=evals.get("val"))

        # evaluate
        results = {}
        for name, (x, y) in evals.items():
            if len(y) == 0:
                logger.info("%-5s: no labelled windows, skipped", name)
                continue
            pred = self.model.predict(x)
            results[name] = {k: fn(y, pred) for k, fn in self.metrics.items()}
            logger.info("%-5s: n=%s %s", name, f"{len(y):,}",
                        " ".join(f"{k}={v:.3f}" for k, v in results[name].items()))
        return results
