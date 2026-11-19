"""Training-time resampling: changes how often each training window is seen, never the held-out data.

Applied to the training part of each fold only. It repeats existing windows, so it adds no new
information; it only stops the rare class from being drowned out.
"""

import numpy as np


class RandomOversampler:
    """Repeats randomly chosen windows of the rarer classes until they reach a share of the largest class.

    Every original window stays in, so nothing is thrown away; classes are only topped up.
    """

    def __init__(self, ratio: float = 1.0, seed: int = 0) -> None:
        """Args:
        ratio: Target size of each class relative to the largest class. 1.0 balances the classes,
            0.5 brings each rarer class up to half the size of the largest. Classes already at or
            above the target are left alone.
        seed: Seed of the random choice of repeated windows.
        """
        if not 0 < ratio <= 1:
            raise ValueError(f"oversample ratio must be in (0, 1], got {ratio}")
        self.ratio = ratio
        self.seed = seed

    def indices(self, y: np.ndarray) -> np.ndarray:
        """Indices into `y` for the resampled training set: all originals, then the repeats.

        Args:
            y: The training labels.

        Returns:
            An index array. Use it as `x[idx]`, `y[idx]`. It is longer than `y` if any class was topped up.
        """
        rng = np.random.default_rng(self.seed)
        classes, counts = np.unique(y, return_counts=True)
        target = int(round(counts.max() * self.ratio))
        extra = [
            rng.choice(np.flatnonzero(y == c), size=target - n, replace=True)
            for c, n in zip(classes, counts)
            if n < target
        ]
        return np.concatenate([np.arange(len(y)), *extra]).astype(int)
