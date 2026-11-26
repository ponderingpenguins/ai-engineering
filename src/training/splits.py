"""Person-wise k-fold cross-validation splits."""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class GroupKFoldSplitter:
    """Stratified k-fold over groups: every group (person) lands in exactly one test fold.

    Folds keep the share of each class roughly equal where the groups allow it. With `k=None` there is
    one fold per group, which is leave-one-group-out.
    """

    def __init__(self, k: int | None = None, seed: int = 0) -> None:
        """Args:
        k: Number of folds, or None for one fold per group.
        seed: Seed for the shuffle, so the same data always gives the same folds.
        """
        self.k = k
        self.seed = seed

    def n_splits(self, groups: np.ndarray) -> int:
        """The number of folds for these groups.

        Raises:
            ValueError: If `k` is not between 2 and the number of groups.
        """
        n_groups = len(np.unique(groups))
        k = n_groups if self.k is None else self.k
        if not 2 <= k <= n_groups:
            raise ValueError(f"folds must be between 2 and the number of groups ({n_groups}), got {k}")
        return k

    def split(self, y: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns one `(train_idx, test_idx)` pair per fold.

        Raises:
            ValueError: If `k` is out of range, or a fold has no test data.
        """
        k = self.n_splits(groups)
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.seed)
        folds = list(splitter.split(np.zeros(len(y)), y, groups))

        tested = [set(groups[test]) for _, test in folds]
        if any(not group_set for group_set in tested):
            raise ValueError("a fold has no test data; try a different seed or fewer folds")
        return folds


def validation_split(train: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Splits a training fold into fit and validation indices by holding out one whole person.

    A person is never in both parts, so the validation loss measures generalisation to a new person
    (as the test fold does) and early stopping is not tuned on someone the model has trained on.

    Args:
        train: Indices of the training fold.
        groups: The person of every window, indexed like the full dataset.
        seed: Chooses which person is held out.

    Returns:
        `(fit_idx, val_idx)`, or `(train, None)` if the fold has fewer than two people, in which case
        there is nothing to validate on.
    """
    people = np.unique(groups[train])
    if len(people) < 2:
        return train, None
    return _hold_out(train, groups, np.random.default_rng(seed).choice(people))


def inner_splits(train: np.ndarray, groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Leave-one-person-out inside a training fold, for tuning without touching the test person.

    Every person in the training fold is the validation person once, and the others are fit on.

    Args:
        train: Indices of the training fold (the outer test person is not in it).
        groups: The person of every window, indexed like the full dataset.

    Returns:
        One `(fit_idx, val_idx)` pair per person in `train`, sorted by the person's name.

    Raises:
        ValueError: If `train` has fewer than two people, so there is nothing to fit on or validate with.
    """
    people = np.unique(groups[train])
    if len(people) < 2:
        raise ValueError(f"inner splits need at least two people, got {len(people)}")
    return [_hold_out(train, groups, person) for person in people]


def _hold_out(train: np.ndarray, groups: np.ndarray, person: str) -> tuple[np.ndarray, np.ndarray]:
    """`(fit_idx, val_idx)` with all of `person`'s windows in `val_idx`."""
    is_val = groups[train] == person
    return train[~is_val], train[is_val]
