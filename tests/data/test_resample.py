import numpy as np
import pytest
import torch

from data.resample import RandomOversampler
from models.baseline import MajorityClass
from models.rnn import RNN
from config.training import TrainingConfig
from training.fitting import _balanced_weights, _to_int, fit_torch
from training.splitting import GroupKFoldSplitter
from training.cross_validation import CrossValidator


def counts(y):
    return {c: int(n) for c, n in zip(*np.unique(y, return_counts=True))}


def test_oversampling_balances_classes_and_keeps_every_original():
    y = np.array(["not_lying"] * 20 + ["lying"] * 3)
    idx = RandomOversampler(1.0, seed=0).indices(y)
    assert counts(y[idx]) == {"lying": 20, "not_lying": 20}
    assert set(range(len(y))) <= set(idx)  # nothing is dropped
    assert list(idx[: len(y)]) == list(range(len(y)))  # originals first, repeats after
    assert set(y[idx[len(y):]]) == {"lying"}  # only the rare class is repeated


def test_ratio_tops_the_rare_class_up_to_a_share_of_the_largest():
    y = np.array(["not_lying"] * 20 + ["lying"] * 3)
    assert counts(y[RandomOversampler(0.5).indices(y)]) == {"lying": 10, "not_lying": 20}


def test_balanced_input_is_unchanged_and_result_is_seeded():
    y = np.array(["a"] * 5 + ["b"] * 5)
    assert list(RandomOversampler().indices(y)) == list(range(10))
    y2 = np.array(["a"] * 9 + ["b"] * 2)
    assert np.array_equal(RandomOversampler(seed=4).indices(y2), RandomOversampler(seed=4).indices(y2))


@pytest.mark.parametrize("ratio", [0, -1, 1.5])
def test_bad_ratio_raises(ratio):
    with pytest.raises(ValueError, match="ratio"):
        RandomOversampler(ratio)


class Recorder(MajorityClass):
    """Remembers what it was trained on."""

    seen: list = []

    def fit(self, x, y, val=None, **kwargs):
        Recorder.seen.append((np.asarray(y).copy(), None if val is None else np.asarray(val[1]).copy()))
        return super().fit(x, y, val, **kwargs)


def test_trainer_oversamples_training_windows_only():
    groups = np.array(["Bob"] * 12 + ["Dave"] * 12 + ["Frank"] * 12)
    y = np.array((["lying"] + ["not_lying"] * 5) * 6)
    x = np.zeros((len(y), 4), dtype=np.float32)
    Recorder.seen = []
    results = CrossValidator(Recorder, splitter=GroupKFoldSplitter(), resampler=RandomOversampler(1.0)).cross_validate(x, y, groups)
    for (train_y, val_y), fold in zip(Recorder.seen, results.folds):
        assert counts(train_y)["lying"] == counts(train_y)["not_lying"]  # balanced
        assert counts(val_y) == {"lying": 2, "not_lying": 10}  # validation untouched
        assert fold.n_test == 12  # test untouched
        assert fold.n_train_after_resampling > fold.n_train


# ---- the validation loss no longer depends on the training weights ----------------------------

def test_balanced_weights_come_from_the_validation_labels():
    model = RNN()
    weights = _balanced_weights(torch.tensor([0, 0, 0, 1]), 3, model.device)
    assert weights.tolist() == pytest.approx([4 / (2 * 3), 4 / (2 * 1), 0.0])  # class 2 absent: weight 0


def test_validation_loss_is_the_mean_of_per_class_losses():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 2, 1, 256, 16)).astype(np.float32)
    y = np.array(["lying", "not_lying"] * 6)
    xv = rng.normal(size=(9, 2, 1, 256, 16)).astype(np.float32)
    yv = np.array(["lying"] * 2 + ["not_lying"] * 7)  # imbalanced validation set
    model = RNN(hidden=8, feat_dim=8)
    history = fit_torch(model, x, y, training=TrainingConfig(epochs=1, batch_size=6), val=(xv, yv))

    model.net.eval()
    with torch.no_grad():
        per_window = torch.nn.functional.cross_entropy(
            model.net(torch.as_tensor(xv).to(model.device)),
            torch.as_tensor(np.searchsorted(model.classes_, yv)).to(model.device), reduction="none").cpu().numpy()
    expected = np.mean([per_window[yv == c].mean() for c in model.classes_])
    # one step per epoch evaluation; the kept weights are the only ones, so the recorded loss matches
    assert history.val_loss[-1] == pytest.approx(expected, rel=1e-4)


def test_validation_weights_can_be_pooled_over_people():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 2, 1, 256, 16)).astype(np.float32)
    y = np.array(["lying", "not_lying"] * 6)
    xv = rng.normal(size=(5, 2, 1, 256, 16)).astype(np.float32)
    yv = np.array(["not_lying"] * 5)  # a validation person with one class only
    pooled = np.array(["lying"] * 2 + ["not_lying"] * 8)
    own = fit_torch(RNN(hidden=8, feat_dim=8), x, y, training=TrainingConfig(epochs=1, batch_size=6), val=(xv, yv)).val_loss[-1]
    shared = fit_torch(RNN(hidden=8, feat_dim=8), x, y, training=TrainingConfig(epochs=1, batch_size=6), val=(xv, yv), val_weight_labels=pooled).val_loss[-1]
    # one class: any positive weight gives the same weighted mean, so the loss is the plain cross-entropy either way
    assert shared == pytest.approx(own, rel=1e-4)
    model = RNN()
    model.classes_ = np.array(["lying", "not_lying"])
    weights = _balanced_weights(_to_int(model, pooled), 2, model.device)
    assert weights.tolist() == pytest.approx([10 / (2 * 2), 10 / (2 * 8)])
