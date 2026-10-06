"""Majority-class baseline: always predict the most common training label."""

from collections import Counter
from .model import BaseModel
import numpy as np


class MajorityClass(BaseModel):
    needs_heatmaps = False

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
