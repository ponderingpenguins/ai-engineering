"""Majority-class baseline: always predict the most common training label."""

from collections import Counter
from .model import BaseModel
import numpy as np


class MajorityClass(BaseModel):
    def fit(self, x: np.ndarray, y: np.ndarray) -> "MajorityClass":
        self.label_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, x: np.ndarray) -> list[str]:
        n = len(x)
        return [self.label_] * n
