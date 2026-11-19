import json

import numpy as np
import pytest

from models.baseline import MajorityClass
from models.model import BaseModel, History
from models.rnn import RNN
from utils.curves import mean_curves, plot_fold_curves, plot_mean_curves
from utils.run_report import write_report
from utils.splitter import GroupKFoldSplitter, validation_split
from utils.trainer import Trainer

TASK = {"target": "label_pose", "collapse_lying": True, "window_len": 16, "label_rule": "any_lying"}


# ---- validation split -------------------------------------------------------------------------

def test_validation_split_holds_out_exactly_one_whole_person():
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    train = np.arange(len(groups))
    fit, val = validation_split(train, groups, seed=0)
    assert len(set(groups[val])) == 1
    assert not set(groups[fit]) & set(groups[val])
    assert sorted(np.concatenate([fit, val])) == list(train)


def test_validation_split_never_touches_the_test_fold():
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    train = np.flatnonzero(groups != "c")  # "c" is the test person
    fit, val = validation_split(train, groups, seed=3)
    assert "c" not in set(groups[fit]) | set(groups[val])


def test_validation_split_is_seeded_and_needs_two_people():
    groups = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    train = np.arange(12)
    assert np.array_equal(validation_split(train, groups, 1)[1], validation_split(train, groups, 1)[1])
    fit, val = validation_split(np.arange(4), groups, 0)  # one person only
    assert val is None and len(fit) == 4


# ---- early stopping ---------------------------------------------------------------------------

@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 2, 1, 256, 16)).astype(np.float32)
    y = np.array(["lying", "not_lying"] * 8)
    xv = rng.normal(size=(8, 2, 1, 256, 16)).astype(np.float32)
    yv = np.array(["lying", "not_lying"] * 4)
    return x, y, xv, yv


def test_without_validation_data_it_trains_all_epochs(tiny_data):
    x, y, _, _ = tiny_data
    history = RNN(epochs=3, batch_size=8, hidden=8, feat_dim=8).fit(x, y)
    assert history.steps_per_epoch == 2
    assert history.train_steps == list(range(1, 7))
    assert history.val_loss == [] and history.best_step is None and not history.stopped_early


def test_early_stopping_stops_and_keeps_the_best_weights(tiny_data):
    x, y, xv, yv = tiny_data
    model = RNN(epochs=40, batch_size=8, hidden=8, feat_dim=8, patience=3, lr=1e-2)
    history = model.fit(x, y, val=(xv, yv))
    assert history.stopped_early and history.train_steps[-1] < 40 * history.steps_per_epoch
    assert history.best_step in history.val_steps
    best = history.val_steps.index(history.best_step)
    assert history.val_loss[best] == min(history.val_loss)
    # patience is counted in evaluations after the best one
    assert len(history.val_steps) - 1 - best == 3


def test_validation_loss_is_computed_per_evaluation_not_per_step(tiny_data):
    x, y, xv, yv = tiny_data
    history = RNN(epochs=2, batch_size=4, hidden=8, feat_dim=8, patience=99, evals_per_epoch=2).fit(x, y, val=(xv, yv))
    assert history.steps_per_epoch == 4
    assert history.val_steps == [2, 4, 6, 8]  # every 2 steps
    assert len(history.train_loss) == 8  # every step


def test_kept_weights_are_restored_so_predict_uses_the_best_epoch(tiny_data):
    x, y, xv, yv = tiny_data
    model = RNN(epochs=5, batch_size=8, hidden=8, feat_dim=8, patience=1, seed=1)
    model.fit(x, y, val=(xv, yv))
    assert set(model.predict(xv)) <= {"lying", "not_lying"}


# ---- trainer: checkpoints, curves, report ------------------------------------------------------

def make_cv_data():
    groups = np.array(["Bob"] * 6 + ["Dave"] * 6 + ["Frank"] * 6 + ["Ian"] * 6)
    y = np.array(["lying", "not_lying", "not_lying"] * 8)
    return np.zeros((len(y), 4), dtype=np.float32), y, groups


def test_trainer_writes_checkpoints_results_and_report(tmp_path):
    x, y, groups = make_cv_data()
    out = tmp_path / "run"
    trainer = Trainer(MajorityClass, splitter=GroupKFoldSplitter(), out_dir=out, task=TASK,
                      config={"model": {"name": "baseline"}, "eval": {"folds": None, "seed": 0}})
    results = trainer.cross_validate(x, y, groups)

    assert len(results["folds"]) == 4
    for fold in results["folds"]:
        assert fold["val_groups"] and not set(fold["val_groups"]) & set(fold["test_groups"])
        assert (out / fold["checkpoint"]).exists()
        model, task = BaseModel.load(out / fold["checkpoint"])
        assert task["window_len"] == 16 and task["label_rule"] == "any_lying"
    assert json.loads((out / "results.json").read_text())["pooled"]["n"] == len(y)
    report = (out / "report.md").read_text()
    assert "## Per fold" in report and "## Pooled results" in report and "Recall lying" in report
    assert "does not train iteratively" in report  # the baseline has no loss curves


def test_trainer_without_out_dir_saves_nothing(tmp_path):
    x, y, groups = make_cv_data()
    results = Trainer(MajorityClass).cross_validate(x, y, groups)
    assert all(fold["checkpoint"] is None for fold in results["folds"])
    assert not list(tmp_path.iterdir())


# ---- curves and report ------------------------------------------------------------------------

def fake_history(steps, steps_per_epoch, val_every, stopped=False):
    rng = np.random.default_rng(steps)
    train = list(1.0 / np.arange(1, steps + 1) + rng.normal(0, 0.02, steps))
    val_steps = list(range(val_every, steps + 1, val_every))
    return History(steps_per_epoch=steps_per_epoch, train_steps=list(range(1, steps + 1)), train_loss=train,
                   val_steps=val_steps, val_loss=[1.0 / (i + 1) + 0.1 for i in range(len(val_steps))],
                   best_step=val_steps[-1], stopped_early=stopped).to_dict()


def fake_folds():
    return [
        {"fold": 1, "test_groups": ["Bob"], "val_groups": ["Dave"], "history": fake_history(40, 10, 10)},
        {"fold": 2, "test_groups": ["Dave"], "val_groups": ["Ian"], "history": fake_history(30, 6, 6, stopped=True)},
        {"fold": 3, "test_groups": ["Frank"], "val_groups": ["Ian"], "history": fake_history(50, 10, 10)},
    ]


def test_mean_curves_use_only_real_val_points_and_need_two_folds():
    folds = fake_folds()
    folds[2]["history"] = fake_history(60, 10, 10)  # fold 3 trains for 6 epochs, fold 1 for 4, fold 2 for 5
    curves = mean_curves(folds)
    assert curves is not None
    assert list(curves["val_epoch"]) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # evaluated once per epoch, no interpolation
    assert not np.isnan(curves["val_mean"][:5]).any()  # at least two folds reached epochs 1 to 5
    assert np.isnan(curves["val_mean"][5]) and np.isnan(curves["val_std"][5])  # only fold 3 reached epoch 6
    assert np.isnan(curves["train_mean"][curves["epoch"] > 5.0 + 1e-9]).all()


def test_mean_curves_none_without_histories():
    assert mean_curves([{"fold": 1, "history": None}]) is None


def test_figures_are_written(tmp_path):
    folds = fake_folds()
    assert plot_fold_curves(folds, tmp_path / "a.png").stat().st_size > 0
    assert plot_mean_curves(folds, tmp_path / "b.png").stat().st_size > 0
    assert plot_fold_curves([{"fold": 1, "history": None}], tmp_path / "c.png") is None


def test_report_embeds_both_figures_when_there_are_histories(tmp_path):
    folds = [dict(f, n_test=5, metrics={"accuracy": 0.9, "macro_f1": 0.5}, checkpoint=f"checkpoints/f{f['fold']}.pt",
                  per_class={"lying": {"recall": 0.5, "support": 2}, "not_lying": {"recall": 1.0, "support": 3}})
             for f in fake_folds()]
    results = {"config": {"model": {"name": "rnn", "rnn": {"epochs": 30, "patience": 5, "batch_size": 32,
                                                          "lr": 0.001, "evals_per_epoch": 1}},
                          "eval": {"folds": None, "seed": 0, "window_len": 16, "stride": 8, "label_rule": "any_lying"},
                          "window_timing": {"window_s": 3.2, "stride_s": 1.6}},
               "folds": folds,
               "pooled": {"n": 15, "metrics": {"accuracy": 0.9, "macro_f1": 0.5},
                          "per_class": {"lying": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 6},
                                        "not_lying": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 9}}},
               "per_person": {"Bob": {"n": 5, "lying_recall": 0.5, "lying_support": 2,
                                      "not_lying_recall": 1.0, "not_lying_support": 3}}}
    report = write_report(results, tmp_path).read_text()
    assert "![Loss per fold](loss_per_fold.png)" in report and "![Loss across folds](loss_mean.png)" in report
    assert "3.2 s" in report and "patience 5" in report and "early" in report
    assert (tmp_path / "loss_per_fold.png").exists() and (tmp_path / "loss_mean.png").exists()


# ---- fixed epochs without validation -------------------------------------------------------------

def test_without_validation_every_fold_trains_on_all_its_training_people():
    x, y, groups = make_cv_data()
    seen = []

    class Recorder(MajorityClass):
        def fit(self, x, y, val=None):
            seen.append(val)
            return super().fit(x, y, val)

    results = Trainer(Recorder, splitter=GroupKFoldSplitter(), validate=False).cross_validate(x, y, groups)
    assert all(val is None for val in seen)
    assert all(fold["val_groups"] == [] and fold["n_val"] == 0 for fold in results["folds"])
    assert all(fold["n_train"] == 18 for fold in results["folds"])  # 3 of 4 people, 6 windows each


def test_report_and_figures_work_without_validation_loss(tmp_path):
    folds = []
    for f in fake_folds():
        h = dict(f["history"], val_steps=[], val_loss=[], best_step=None, stopped_early=False)
        folds.append(dict(f, val_groups=[], history=h, n_test=5, metrics={"accuracy": 0.9, "macro_f1": 0.5},
                          per_class={"lying": {"recall": 0.5, "support": 2}}, checkpoint=None))
    results = {"config": {"model": {"name": "rnn", "rnn": {"epochs": 3, "batch_size": 32, "lr": 0.001, "seed": 0}},
                          "train": {"validation": False}, "eval": {"folds": None, "seed": 0}},
               "folds": folds,
               "pooled": {"n": 15, "metrics": {"accuracy": 0.9, "macro_f1": 0.5},
                          "per_class": {"lying": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 6}}},
               "per_person": {}}
    report = write_report(results, tmp_path).read_text()
    assert "exactly 3 epochs" in report and "No validation data and no early stopping" in report
    assert "Every fold trains on all its training people" in report
    assert (tmp_path / "loss_per_fold.png").exists() and (tmp_path / "loss_mean.png").exists()
