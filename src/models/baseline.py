"""Majority-class baseline: always predict the most common training label."""

from collections import Counter


class MajorityClass:
    def fit(self, y: list[str]) -> "MajorityClass":
        self.label_ = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, n: int) -> list[str]:
        return [self.label_] * n
