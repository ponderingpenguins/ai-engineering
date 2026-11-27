"""Repeat training windows to increase the share of rare classes.

Oversampling repeats existing examples; validation and test windows stay unchanged.
"""

import numpy as np


class RandomOversampler:
    """Repeat rare-class windows until they reach a target size.

    The target is a fraction of the largest class. Keep every original window
    and add duplicates only for classes below that target.
    """

    def __init__(self, ratio: float = 1.0, seed: int = 0) -> None:
        """Set the target class-size ratio and sampling seed.

        Args:
            ratio: Target size relative to the largest class, in (0, 1]. A value
                of 1 balances classes; 0.5 raises smaller classes to half that size.
            seed: Random seed used to choose repeated windows.

        Raises:
            ValueError: If ratio is outside (0, 1].
        """
        if not 0 < ratio <= 1:
            raise ValueError(f"oversample ratio must be in (0, 1], got {ratio}")
        self.ratio = ratio
        self.seed = seed

    def indices(self, y: np.ndarray) -> np.ndarray:
        """Return training indices containing all originals followed by repeats.

        Args:
            y: Non-empty array of training labels.

        Returns:
            Integer indices for selecting x[idx] and y[idx]. The result is longer
            than y only when a class needs additional windows.
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
