import numpy as np

from data.resample import RandomOversampler
from models.baseline import MajorityClass
from models.model import History
from training.fit import fit_fold


class Recorder(MajorityClass):
    """Remembers what `fit` got."""

    calls: list = []

    def fit(self, x, y, val=None, val_weight_labels=None):
        Recorder.calls.append({"y": np.asarray(y).copy(), "val": val, "weights": val_weight_labels})
        return History(steps_per_epoch=1)


def data():
    y = np.array((["lying"] + ["not_lying"] * 5) * 4)
    x = np.arange(len(y), dtype=np.float32).reshape(-1, 1)
    return x, y


def test_fit_fold_trains_on_the_fit_windows_and_validates_on_the_validation_windows():
    x, y = data()
    Recorder.calls = []
    fit = fit_fold(Recorder(), x, y, np.arange(0, 12), np.arange(12, 18), val_weight_labels=y)
    call = Recorder.calls[0]
    assert len(call["y"]) == 12 and len(call["val"][1]) == 6 and call["weights"] is y
    assert (fit.train_idx == np.arange(12)).all() and fit.seconds >= 0 and fit.history.steps_per_epoch == 1


def test_fit_fold_without_validation_passes_none():
    x, y = data()
    Recorder.calls = []
    fit_fold(Recorder(), x, y, np.arange(12))
    assert Recorder.calls[0]["val"] is None and Recorder.calls[0]["weights"] is None


def test_the_resampler_repeats_training_windows_only():
    x, y = data()
    Recorder.calls = []
    fit = fit_fold(Recorder(), x, y, np.arange(12), np.arange(12, 18), resampler=RandomOversampler(1.0))
    call = Recorder.calls[0]
    assert (call["y"] == "lying").sum() == (call["y"] == "not_lying").sum()  # balanced
    assert len(call["val"][1]) == 6 and (call["val"][1] == y[12:18]).all()  # validation untouched
    assert len(fit.train_idx) == len(call["y"]) > 12


def test_a_model_that_does_not_train_iteratively_returns_no_history():
    x, y = data()
    model = MajorityClass()
    fit = fit_fold(model, x, y, np.arange(12), np.arange(12, 18))
    assert fit.history is None and model.label_ == "not_lying"
