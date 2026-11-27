"""Split windows by person for testing, validation and inner tuning folds.

All windows from a person stay together within each split.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class GroupKFoldSplitter:
    """Keep each person in exactly one test fold, balancing labels where possible.

    With k=None, use one test fold per person. A fixed seed makes grouped
    stratified folds repeatable for the same inputs.
    """

    def __init__(self, k: int | None = None, seed: int = 0) -> None:
        """Set the outer fold count and split seed.

        Args:
            k: Number of folds, or None for leave-one-person-out evaluation.
            seed: Random seed for the grouped stratified split.
        """
        self.k = k
        self.seed = seed

    def n_splits(self, groups: np.ndarray) -> int:
        """Return the fold count for these person groups.

        Raises:
            ValueError: If the count is outside 2 to the number of people.
        """
        n_groups = len(np.unique(groups))
        k = n_groups if self.k is None else self.k
        if not 2 <= k <= n_groups:
            raise ValueError(f"folds must be between 2 and the number of groups ({n_groups}), got {k}")
        return k

    def split(self, y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return training and test indices for every outer fold.

        Args:
            y: Window labels of shape (N,).
            groups: Person names of shape (N,), aligned with y.

        Returns:
            One (train_idx, test_idx) pair per fold, with no person in both parts.

        Raises:
            ValueError: If the fold count is invalid or a test fold is empty.
        """
        k = self.n_splits(groups)
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.seed)
        folds = list(splitter.split(np.zeros(len(y)), y, groups))

        tested = [set(groups[test]) for _, test in folds]
        if any(not group_set for group_set in tested):
            raise ValueError("a fold has no test data; try a different seed or fewer folds")
        return folds


def validation_split(train: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Hold out one whole training person for early-stopping validation.

    Args:
        train: Window indices of the outer training fold.
        groups: Person names for the full dataset.
        seed: Random seed used to choose the validation person.

    Returns:
        (fit_idx, val_idx), or (train, None) if fewer than two training people
        are available. Indices refer to the full dataset.
    """
    people = np.unique(groups[train])
    if len(people) < 2:
        return train, None
    return _hold_out(train, groups, np.random.default_rng(seed).choice(people))


def inner_splits(train: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hold out each outer-training person once for inner tuning validation.

    Args:
        train: Outer training indices, excluding every outer test person.
        groups: Person names for the full dataset.

    Returns:
        One (fit_idx, val_idx) pair per training person, sorted by person name.
        Indices refer to the full dataset.

    Raises:
        ValueError: If fewer than two training people are available.
    """
    people = np.unique(groups[train])
    if len(people) < 2:
        raise ValueError(f"inner splits need at least two people, got {len(people)}")
    return [_hold_out(train, groups, person) for person in people]


def _hold_out(train: np.ndarray, groups: np.ndarray, person: str) -> tuple[np.ndarray, np.ndarray]:
    """Put all requested person's windows in validation and the remaining windows in fit."""
    is_val = groups[train] == person
    return train[~is_val], train[is_val]
