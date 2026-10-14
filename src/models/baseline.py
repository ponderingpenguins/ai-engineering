"""Majority-class baseline: always predict the most common training label."""

from collections import Counter
from typing import Any, Self

import numpy as np

from .model import BaseModel


class MajorityClass(BaseModel):
    needs_heatmaps = False
    model_name = "baseline"

    def _checkpoint_state(self) -> dict[str, Any]:
        if not hasattr(self, "label_"):
            raise ValueError("Set the majority label before saving the baseline.")
        return {"label": str(self.label_)}

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict[str, Any]) -> Self:
        model = cls()
        model.label_ = checkpoint["label"]
        return model

    def fit(self, x: np.ndarray, y: np.ndarray, val:tuple[np.ndarray, np.ndarray] | None=None) -> None:
        """Remembers the most common label.

        Args:
            x: Ignored.
            y: Training labels, shape (N,).
            val: Ignored.
        """
        self.label_ = Counter(y).most_common(1)[0][0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the most common training label for every sample.

        Args:
            x: Samples, only their number is used.

        Returns:
            Array of shape (len(x),) holding the majority label.
        """
        n = len(x)
        return np.asarray([self.label_] * n)
