import numpy as np
import pytest

from models.model import History
from utils.splitter import GroupKFoldSplitter
from utils.trainer import Trainer
from utils.tuning import Tuner, best_epoch, median_epoch, validate_grid

CLASSES = np.array(["lying", "not_lying"])


class Fake:
    """Predicts from a label hint in column 1 of x when `good`, else always not_lying. Records what it saw."""

    seen: list = []  # (kind, row ids) of every fit and validation call

    def __init__(self, setting):
        self.setting = setting

    def fit(self, x, y, val=None):
        Fake.seen.append(("fit", set(x[:, 0].astype(int))))
        if val is not None:
            Fake.seen.append(("val", set(val[0][:, 0].astype(int))))
        return History(steps_per_epoch=10, train_steps=[10, 20, 30], best_step=self.setting.get("best_step", 20))

    def predict(self, x):
        if not self.setting.get("good"):
            return np.array(["not_lying"] * len(x))
        return np.where(x[:, 1] == 1, "lying", "not_lying")


def make_data():
    groups = np.array([p for p in ["Bob", "Dave", "Frank", "Ian"] for _ in range(12)])
    y = np.array((["lying"] + ["not_lying"] * 5) * 8)
    x = np.stack([np.arange(len(y)), (y == "lying").astype(float)], axis=1).astype(np.float32)
    return x, y, groups


def test_tuner_picks_the_setting_with_the_better_validation_macro_f1():
    x, y, groups = make_data()
    grid = [{"good": False}, {"good": True}, {"good": False}]
    train = np.flatnonzero(groups != "Ian")
    tuning = Tuner(Fake, grid).tune(x, y, groups, train, CLASSES)
    assert tuning.setting == {"good": True}
    assert tuning.score == 1.0
    assert [c["setting"] for c in tuning.candidates] == grid
    assert [f["val_group"] for f in tuning.candidates[0]["folds"]] == ["Bob", "Dave", "Frank"]


def test_ties_go_to_the_earlier_setting():
    x, y, groups = make_data()
    tuning = Tuner(Fake, [{"good": True, "tag": "a"}, {"good": True, "tag": "b"}]).tune(
        x, y, groups, np.arange(len(y)), CLASSES)
    assert tuning.setting["tag"] == "a"


def test_the_outer_test_person_is_never_seen_while_tuning():
    x, y, groups = make_data()
    train = np.flatnonzero(groups != "Ian")
    Fake.seen = []
    Tuner(Fake, [{"good": True}, {"good": False}]).tune(x, y, groups, train, CLASSES)
    test_rows = set(np.flatnonzero(groups == "Ian"))
    assert Fake.seen and all(not (rows & test_rows) for _, rows in Fake.seen)
    for (kind_a, fit), (kind_b, val) in zip(Fake.seen[::2], Fake.seen[1::2]):
        assert (kind_a, kind_b) == ("fit", "val") and not fit & val  # a person is not in both parts


def test_epochs_are_the_median_best_epoch_and_build_uses_them():
    x, y, groups = make_data()
    tuner = Tuner(Fake, [{"good": True, "best_step": 25}])  # step 25 of 10 per epoch is epoch 3
    tuning = tuner.tune(x, y, groups, np.flatnonzero(groups != "Ian"), CLASSES)
    assert tuning.epochs == 3
    assert tuner.build(tuning).setting == {"good": True, "best_step": 25, "epochs": 3}


def test_median_epoch_rounds_up_and_is_at_least_one():
    assert median_epoch([2, 3, 7]) == 3
    assert median_epoch([2, 3]) == 3
    assert median_epoch([0]) == 1


def test_best_epoch_falls_back_to_the_last_step_trained():
    assert best_epoch(History(steps_per_epoch=10, train_steps=[10, 20, 25], best_step=None)) == 3
    assert best_epoch(History(steps_per_epoch=10, best_step=11)) == 2
    assert best_epoch(None) is None


def test_empty_grid_and_unknown_or_duplicate_settings_raise():
    with pytest.raises(ValueError, match="empty"):
        Tuner(Fake, [])
    with pytest.raises(ValueError, match="unknown hyperparameter"):
        validate_grid([{"hiden": 3}], ["hidden", "lr"])
    with pytest.raises(ValueError, match="twice"):
        validate_grid([{"lr": 1}, {"lr": 1}], ["hidden", "lr"])
    validate_grid([{"lr": 1}, {"lr": 2, "hidden": 3}], ["hidden", "lr"])


def test_trainer_tunes_each_outer_fold_on_its_training_people_only():
    x, y, groups = make_data()
    Fake.seen = []
    tuner = Tuner(Fake, [{"good": False}, {"good": True}])
    results = Trainer(lambda: Fake({}), splitter=GroupKFoldSplitter(), tuner=tuner).cross_validate(x, y, groups)
    assert len(results["folds"]) == 4
    for fold in results["folds"]:
        assert fold["tuning"]["setting"] == {"good": True}
        assert fold["val_groups"] == [] and fold["n_val"] == 0  # the refit has no validation person
        assert fold["n_train"] == 36  # all three training people
        assert fold["metrics"]["macro_f1"] == 1.0
    assert results["pooled"]["metrics"]["macro_f1"] == 1.0
