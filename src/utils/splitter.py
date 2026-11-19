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
