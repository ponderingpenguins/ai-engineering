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
            ValueError: If `k` is out of range, or the folds are not a partition of the groups
                (an empty fold, or a group tested more than once).
        """
        k = self.n_splits(groups)
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=self.seed)
        folds = list(splitter.split(np.zeros(len(y)), y, groups))

        tested = [set(groups[test]) for _, test in folds]
        if any(not group_set for group_set in tested):
            raise ValueError("a fold has no test data; try a different seed or fewer folds")
        if sum(len(group_set) for group_set in tested) != len(set(groups)):
            raise ValueError("a group is in more than one test fold")
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
    held_out = np.random.default_rng(seed).choice(people)
    is_val = groups[train] == held_out
    return train[~is_val], train[is_val]
