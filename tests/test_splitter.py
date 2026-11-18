import numpy as np
import pytest

from utils.splitter import GroupKFoldSplitter

PEOPLE = {"Bob": 40, "Dave": 30, "Frank": 60, "Ian": 70}
LYING = {"Bob": 1, "Dave": 1, "Frank": 3, "Ian": 2}


@pytest.fixture
def data():
    groups, y = [], []
    for person, n in PEOPLE.items():
        groups += [person] * n
        y += ["lying"] * LYING[person] + ["not_lying"] * (n - LYING[person])
    return np.array(y), np.array(groups)


def test_default_is_one_person_per_fold(data):
    y, groups = data
    folds = GroupKFoldSplitter().split(y, groups)
    assert len(folds) == len(PEOPLE)
    assert sorted(tuple(np.unique(groups[test])) for _, test in folds) == sorted((p,) for p in PEOPLE)


@pytest.mark.parametrize("seed", range(8))
def test_every_person_is_tested_exactly_once_and_never_in_train(data, seed):
    y, groups = data
    folds = GroupKFoldSplitter(seed=seed).split(y, groups)
    tested = []
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])  # no person in both train and test
        assert len(np.intersect1d(train, test)) == 0
        tested += list(np.unique(groups[test]))
    assert sorted(tested) == sorted(PEOPLE)


def test_k_below_number_of_persons(data):
    y, groups = data
    folds = GroupKFoldSplitter(k=2).split(y, groups)
    assert len(folds) == 2
    assert all(len(test) > 0 for _, test in folds)
    assert sorted(i for _, test in folds for i in test) == list(range(len(y)))  # a partition of all windows


def test_same_seed_same_folds(data):
    y, groups = data
    a = GroupKFoldSplitter(k=2, seed=3).split(y, groups)
    b = GroupKFoldSplitter(k=2, seed=3).split(y, groups)
    assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))


@pytest.mark.parametrize("k", [1, 5])
def test_k_out_of_range_raises(data, k):
    y, groups = data
    with pytest.raises(ValueError, match="between 2 and the number of groups"):
        GroupKFoldSplitter(k=k).split(y, groups)
