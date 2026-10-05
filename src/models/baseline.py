"""Majority-class baseline: always predict the most common training label."""

from collections import Counter
from .model import BaseModel
import numpy as np


class MajorityClass(BaseModel):
    def fit(self, x: np.ndarray, y: np.ndarray, val:tuple[np.ndarray, np.ndarray] | None=None) -> None:
        self.label_ = Counter(y).most_common(1)[0][0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        return np.asarray([self.label_] * n)
