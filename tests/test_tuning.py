from dataclasses import asdict, replace
import numpy as np
import pytest

from models.baseline import MajorityClass
from models.model import History
from utils.splitter import GroupKFoldSplitter
from utils.trainer import Trainer
from utils.results import BestSetting, Tuning
from utils.task import Task
from utils.tuning import (Tuner, apply_tuned, best_epoch, best_overall, kept_val_loss, load_tuned, median_epoch, save_tuned,
                          validate_grid)

CLASSES = np.array(["lying", "not_lying"])


class Fake:
    """Validation loss and best step come from the setting; predictions are right only when `good`.

    Records the row ids (column 0 of x) it was fitted and validated on.
    """

    seen: list = []

    def __init__(self, setting):
        self.setting = setting

    def fit(self, x, y, val=None, **kwargs):
        Fake.seen.append(("fit", set(x[:, 0].astype(int))))
        if val is not None:
            Fake.seen.append(("val", set(val[0][:, 0].astype(int))))
        loss = self.setting.get("loss", 1.0)
        return History(steps_per_epoch=10, train_steps=[10, 20, 30], train_loss=[1.0, 0.9, 0.8], val_steps=[10, 20, 30],
                       val_loss=[loss + 0.1, loss, loss + 0.2], best_step=self.setting.get("best_step", 20))

    def predict(self, x):
        if not self.setting.get("good"):
            return np.array(["not_lying"] * len(x))
        return np.where(x[:, 1] == 1, "lying", "not_lying")


def make_data():
    groups = np.array([p for p in ["Bob", "Dave", "Frank", "Ian"] for _ in range(12)])
    y = np.array((["lying"] + ["not_lying"] * 5) * 8)
    x = np.stack([np.arange(len(y)), (y == "lying").astype(float)], axis=1).astype(np.float32)
    return x, y, groups


def test_tuner_picks_the_lowest_validation_loss_not_the_best_macro_f1():
    x, y, groups = make_data()
    grid = [{"loss": 0.9, "good": True}, {"loss": 0.4, "good": False}, {"loss": 0.7, "good": True}]
    train = np.flatnonzero(groups != "Ian")
    tuning = Tuner(Fake, grid).tune(x, y, groups, train, CLASSES)
    assert tuning.setting == grid[1]  # it predicts badly, but its loss is the lowest
    assert tuning.score == pytest.approx(0.4)
    assert [c.setting for c in tuning.candidates] == grid
    assert tuning.candidates[0].macro_f1 == 1.0 and tuning.candidates[1].macro_f1 < 1.0
    assert [f.val_group for f in tuning.candidates[0].folds] == ["Bob", "Dave", "Frank"]


def test_ties_go_to_the_earlier_setting():
    x, y, groups = make_data()
    tuning = Tuner(Fake, [{"tag": "a"}, {"tag": "b"}]).tune(x, y, groups, np.arange(len(y)), CLASSES)
    assert tuning.setting["tag"] == "a"


def test_the_outer_test_person_is_never_seen_while_tuning():
    x, y, groups = make_data()
    train = np.flatnonzero(groups != "Ian")
    Fake.seen = []
    Tuner(Fake, [{"loss": 0.5}, {"loss": 0.6}]).tune(x, y, groups, train, CLASSES)
    test_rows = set(np.flatnonzero(groups == "Ian"))
    assert Fake.seen and all(not (rows & test_rows) for _, rows in Fake.seen)
    for (kind_a, fit), (kind_b, val) in zip(Fake.seen[::2], Fake.seen[1::2]):
        assert (kind_a, kind_b) == ("fit", "val") and not fit & val  # a person is not in both parts


def test_epochs_are_the_median_best_epoch_and_build_uses_them():
    x, y, groups = make_data()
    tuner = Tuner(Fake, [{"best_step": 25}])  # step 25 at 10 steps per epoch is epoch 3
    tuning = tuner.tune(x, y, groups, np.flatnonzero(groups != "Ian"), CLASSES)
    assert tuning.epochs == 3
    assert tuner.build(tuning).setting == {"best_step": 25, "epochs": 3}


def test_kept_val_loss_is_the_loss_at_the_best_step():
    history = History(steps_per_epoch=10, val_steps=[10, 20, 30], val_loss=[0.9, 0.5, 0.7], best_step=20)
    assert kept_val_loss(history) == 0.5
    assert kept_val_loss(History(steps_per_epoch=10, val_steps=[10, 20], val_loss=[0.9, 0.6])) == 0.6  # no best step
    assert kept_val_loss(History(steps_per_epoch=10)) is None and kept_val_loss(None) is None


def test_model_without_validation_loss_cannot_be_tuned():
    class NoLoss(Fake):
        def fit(self, x, y, val=None, **kwargs):
            return History(steps_per_epoch=10, train_steps=[10], best_step=10)

    x, y, groups = make_data()
    with pytest.raises(ValueError, match="no validation loss"):
        Tuner(NoLoss, [{}]).tune(x, y, groups, np.flatnonzero(groups != "Ian"), CLASSES)


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


def tuning(setting, epochs, score):
    return Tuning(setting, epochs, score)


def test_best_overall_is_the_setting_that_won_most_folds_with_the_median_epoch():
    a, b = {"hidden": 32}, {"hidden": 64}
    best = best_overall([tuning(a, 3, 0.5), tuning(b, 9, 0.4), tuning(a, 7, 0.6), tuning(a, 5, 0.5)])
    assert best.setting == a and best.epochs == 5  # median of 3, 7, 5
    assert (best.wins, best.of_folds) == (3, 4)
    assert best.mean_val_loss == pytest.approx((0.5 + 0.6 + 0.5) / 3)


def test_best_overall_breaks_a_tie_by_the_lower_mean_loss():
    a, b = {"hidden": 32}, {"hidden": 64}
    assert best_overall([tuning(a, 3, 0.5), tuning(b, 3, 0.4)]).setting == b


def test_tuned_setting_round_trips_through_the_yaml_file(tmp_path):
    best = BestSetting({"hidden": 64, "lr": 0.0003}, 7, 3, 4, 0.5)
    task = Task(target="label_pose", collapse_lying=True, window_len=16, label_rule="any_lying")
    path = tmp_path / "sub" / "tuned.yaml"
    save_tuned(path, best, "rnn", task)
    assert load_tuned(path) == {"model": "rnn", "task": task.to_dict(), **asdict(best)}


def test_loading_a_missing_tuned_file_says_to_tune_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the tuning first"):
        load_tuned(tmp_path / "none.yaml")


def test_trainer_tunes_each_outer_fold_on_its_training_people_only():
    x, y, groups = make_data()
    Fake.seen = []
    tuner = Tuner(Fake, [{"loss": 0.8, "good": False}, {"loss": 0.3, "good": True}])
    results = Trainer(lambda: Fake({}), splitter=GroupKFoldSplitter(), tuner=tuner).cross_validate(x, y, groups)
    assert len(results.folds) == 4
    for fold in results.folds:
        assert fold.tuning.setting == {"loss": 0.3, "good": True}
        assert fold.val_groups == [] and fold.n_val == 0  # the refit has no validation person
        assert fold.n_train == 36  # all three training people
        assert fold.metrics["macro_f1"] == 1.0
    assert results.tuned.setting == {"loss": 0.3, "good": True}
    assert (results.tuned.wins, results.tuned.of_folds) == (4, 4)


def test_train_final_fits_on_all_windows_and_saves_a_checkpoint(tmp_path):
    x, y, groups = make_data()
    task = Task(target="label_pose", collapse_lying=True, window_len=16, label_rule="any_lying")
    info = Trainer(MajorityClass, out_dir=tmp_path, task=task).train_final(x, y)
    assert info.n_windows == len(y) and info.checkpoint == "final.pt"
    assert (tmp_path / "final.pt").is_file() and (tmp_path / "final.json").is_file()
    assert info.train_classes == {"lying": 8, "not_lying": 40}


def test_report_has_a_tuning_section_with_every_setting(tmp_path):
    x, y, groups = make_data()
    tuner = Tuner(Fake, [{"loss": 0.8}, {"loss": 0.3, "good": True}])
    Trainer(lambda: Fake({}), splitter=GroupKFoldSplitter(), tuner=tuner, out_dir=tmp_path).cross_validate(x, y, groups)
    report = (tmp_path / "report.md").read_text()
    assert "## Tuning" in report and "**Chosen for the final model:**" in report
    assert report.count("### Fold") == 4
    assert "{'loss': 0.8}" in report and "**{'loss': 0.3, 'good': True}**" in report  # the winner is bold


TASK = Task(target="label_pose", collapse_lying=True, window_len=16, label_rule="any_lying")
TUNED = {"model": "rnn", "task": TASK.to_dict(), "setting": {"hidden": 64, "epochs": 99}, "epochs": 7}


def test_apply_tuned_sets_the_setting_and_the_tuned_epochs():
    from utils.config import ModelConfig
    cfg = apply_tuned(ModelConfig(), TUNED, TASK)
    assert cfg.rnn.hidden == 64 and cfg.rnn.epochs == 7  # the tuned epoch count beats a grid "epochs"
    assert cfg.rnn.lr == ModelConfig().rnn.lr  # everything else stays at its default


def test_apply_tuned_refuses_a_different_task_or_model():
    from utils.config import ModelConfig
    with pytest.raises(ValueError, match="tune again"):
        apply_tuned(ModelConfig(), TUNED, replace(TASK, window_len=8))
    with pytest.raises(ValueError, match="tune again"):
        apply_tuned(ModelConfig(name="baseline"), TUNED, TASK)
    with pytest.raises(ValueError, match="unknown hyperparameter"):
        apply_tuned(ModelConfig(), {**TUNED, "setting": {"hiden": 1}}, TASK)


def test_a_pointnet_setting_is_applied_to_the_pointnet_config():
    from utils.config import ModelConfig
    from utils.tuning import hyperparameter_names, with_setting

    cfg = with_setting(ModelConfig(name="pointnet"), {"width": 64, "point_drop": 0.1})
    assert cfg.pointnet.width == 64 and cfg.pointnet.point_drop == 0.1
    assert cfg.rnn == ModelConfig().rnn  # the other model's settings are untouched
    assert "width" in hyperparameter_names(cfg) and "hidden" not in hyperparameter_names(cfg)

    tuned = {"model": "pointnet", "task": TASK.to_dict(), "setting": {"width": 64}, "epochs": 5}
    applied = apply_tuned(ModelConfig(name="pointnet"), tuned, TASK)
    assert applied.pointnet.width == 64 and applied.pointnet.epochs == 5
    with pytest.raises(ValueError, match="unknown hyperparameter"):  # an RNN setting does not fit PointNet
        apply_tuned(ModelConfig(name="pointnet"), {**tuned, "setting": {"hidden": 8}}, TASK)


def test_the_baseline_cannot_be_tuned():
    from utils.config import ModelConfig
    from utils.tuning import hyperparameter_names

    with pytest.raises(ValueError, match="only defined for"):
        hyperparameter_names(ModelConfig(name="baseline"))


def test_nested_sub_configs_are_not_tunable():
    from utils.config import ModelConfig
    from utils.tuning import hyperparameter_names

    for name in ("rnn", "pointnet"):
        names = hyperparameter_names(ModelConfig(name=name))
        assert "augment" not in names and "lr" in names
