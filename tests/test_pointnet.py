from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
from datasets import Dataset

from data.windows import HEATMAP_SHAPE, _window_points, make_windows
from models.factory import MODEL_CLASSES, build_model
from models.model import BaseModel
from models.pointnet import MiniPointNet, PointNet
from utils.config import ModelConfig
from utils.splitter import GroupKFoldSplitter
from utils.task import Task
from utils.trainer import Trainer

PERIOD = 0.2


def cloud(*points):
    """A radar frame: each point is (x, y, z, doppler, snr); the radar's noise and track values are appended."""
    return [list(p) + [40.0, 255.0] for p in points]


# ---- windows -----------------------------------------------------------------------------------

def test_window_points_layout_padding_and_time():
    clouds = [np.array(cloud((1, 2, 3, 0.1, 20)), dtype=np.float32), np.zeros((0, 7), dtype=np.float32),
              np.array(cloud((4, 5, 6, 0.2, 30), (7, 8, 9, 0.3, 10)), dtype=np.float32)]
    out = _window_points(clouds, np.array([0, 1, 2]), max_points=5)
    assert out.shape == (5, 7)
    assert out[:, 6].tolist() == [1, 1, 1, 0, 0]  # three real points, then padding
    assert out[0, :5].tolist() == pytest.approx([1, 2, 3, 0.1, 20])
    assert out[:3, 5].tolist() == pytest.approx([0.0, 1.0, 1.0])  # time: first frame 0, last frame 1
    assert not out[3:].any()  # padding is all zeros


def test_window_points_empty_window_is_all_padding():
    out = _window_points([np.zeros((0, 7), dtype=np.float32)] * 3, np.array([0, 1, 2]), max_points=4)
    assert out.shape == (4, 7) and not out.any()


def test_window_points_keeps_the_strongest_by_snr_in_original_order():
    pts = np.array(cloud((0, 0, 0, 0, 5), (1, 0, 0, 0, 50), (2, 0, 0, 0, 40), (3, 0, 0, 0, 1)), dtype=np.float32)
    out = _window_points([pts], np.array([0]), max_points=2)
    assert out[:, 0].tolist() == [1.0, 2.0]  # x of the two strongest, still in frame order
    assert out[:, 6].tolist() == [1, 1]


def make_pc_ds(clouds, labels=None):
    n = len(clouds)
    labels = labels or ["standing"] * n
    start = datetime(2026, 1, 1)
    return Dataset.from_dict({
        "recording": ["r1"] * n, "person": ["Bob"] * n, "frame_idx": list(range(n)),
        "timestamp": [start + timedelta(seconds=PERIOD * i) for i in range(n)],
        "label_pose": labels, "heatmap_ok": [True] * n,
        "num_range_bins": [HEATMAP_SHAPE[0]] * n, "num_doppler_bins": [HEATMAP_SHAPE[1]] * n,
        "pointcloud": clouds,
    })


def test_make_windows_pointcloud_shape_and_labels_match_the_other_inputs():
    clouds = [cloud((i, 0, 0, 0, 20)) if i % 2 == 0 else [] for i in range(32)]  # every other frame is empty
    labels = ["standing"] * 32
    labels[5] = "lying_left"
    ds = make_pc_ds(clouds, labels)
    x, y, groups = make_windows(ds, window_len=16, stride=8, inputs="pointcloud", max_points=20, label_rule="any_lying")
    _, y_none, groups_none = make_windows(ds, window_len=16, stride=8, inputs="none", label_rule="any_lying")
    assert x.shape == (3, 20, 7) and x.dtype == np.float32
    assert list(y) == list(y_none) and list(groups) == list(groups_none)  # same windows whatever the input
    assert x[0, :, 6].sum() == 8  # 8 of the first window's 16 frames have a point


def test_make_windows_rejects_unknown_input_type():
    with pytest.raises(ValueError, match="inputs"):
        make_windows(make_pc_ds([cloud((0, 0, 0, 0, 1))] * 16), inputs="lidar")


# ---- the network -------------------------------------------------------------------------------

def batch(n=4, points=10, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, points, 7), dtype=np.float32)
    for i in range(n):
        k = rng.integers(0, points + 1)
        x[i, :k, :5] = rng.normal(size=(k, 5))
        x[i, :k, 5] = rng.random(k)
        x[i, :k, 6] = 1
    return torch.as_tensor(x)


def test_forward_shape_and_empty_window_is_finite():
    net = MiniPointNet(2).eval()
    x = batch()
    x[0] = 0  # a window without points
    out = net(x)
    assert out.shape == (4, 2) and torch.isfinite(out).all()


def test_output_does_not_depend_on_point_order_or_padding():
    net = MiniPointNet(2).eval()
    x = batch(n=3, points=8)
    k = int(x[0, :, 6].sum())
    shuffled = x.clone()
    shuffled[0, :k] = x[0, torch.randperm(k)]
    padded = torch.cat([x, torch.zeros(3, 12, 7)], dim=1)  # twice as much padding
    assert torch.allclose(net(x), net(shuffled), atol=1e-5)
    assert torch.allclose(net(x), net(padded), atol=1e-5)


def test_padding_values_never_leak_into_the_features():
    net = MiniPointNet(2).eval()
    x = batch(n=2, points=8)
    garbage = x.clone()
    garbage[x[..., 6] == 0, :6] = 1e6  # junk in the padded rows, flag still 0
    assert torch.allclose(net(x), net(garbage), atol=1e-5)


# ---- the model ---------------------------------------------------------------------------------

def separable_data(n=60, seed=0):
    """Lying windows have low points (small z), the others high ones, so a working model must learn it."""
    rng = np.random.default_rng(seed)
    y = np.array(["lying", "not_lying"] * (n // 2))
    x = np.zeros((n, 12, 7), dtype=np.float32)
    for i, label in enumerate(y):
        k = rng.integers(4, 12)
        x[i, :k, 0] = rng.normal(0, 0.3, k)
        x[i, :k, 1] = rng.normal(2, 0.3, k)
        x[i, :k, 2] = rng.normal(0.2 if label == "lying" else 1.2, 0.1, k)
        x[i, :k, 3] = rng.normal(0, 0.2, k)
        x[i, :k, 4] = rng.normal(20, 3, k)
        x[i, :k, 5] = rng.random(k)
        x[i, :k, 6] = 1
    return x, y


def test_it_learns_a_separable_toy_problem_and_records_a_history():
    x, y = separable_data()
    xv, yv = separable_data(30, seed=1)
    model = PointNet(epochs=30, batch_size=16, width=32, lr=1e-2, patience=10)
    history = model.fit(x, y, val=(xv, yv))
    assert (model.predict(xv) == yv).mean() > 0.9
    assert history.train_steps and history.val_loss and history.best_step in history.val_steps


def test_it_is_registered_and_built_from_the_config():
    assert MODEL_CLASSES["pointnet"] is PointNet and PointNet.input_type == "pointcloud"
    model = build_model(ModelConfig(name="pointnet"))
    assert isinstance(model, PointNet) and model.width == 128


def test_checkpoint_round_trip_gives_the_same_predictions(tmp_path):
    x, y = separable_data()
    model = PointNet(epochs=3, batch_size=16, width=16)
    model.fit(x, y)
    task = Task(target="label_pose", collapse_lying=True, window_len=16, label_rule="any_lying", max_points=12)
    model.save(tmp_path / "pn.pt", task)
    loaded, loaded_task = BaseModel.load(tmp_path / "pn.pt")
    assert isinstance(loaded, PointNet) and loaded_task == task
    assert (loaded.predict(x) == model.predict(x)).all()
    assert torch.allclose(loaded.net.feat_mean.cpu(), model.net.feat_mean.cpu())  # the standardisation travels with the weights


def test_input_standardisation_ignores_padding():
    x, y = separable_data(20)
    model = PointNet(epochs=1, batch_size=16, width=16)
    model.fit(x, y)
    real = x[x[..., 6] > 0.5][:, :5]
    assert model.net.feat_mean.cpu().numpy() == pytest.approx(real.mean(axis=0), abs=1e-4)


def test_trainer_runs_cross_validation_with_the_pointnet(tmp_path):
    x, y = separable_data(48)
    groups = np.array(["Bob", "Dave", "Frank", "Ian"] * 12)
    results = Trainer(lambda: PointNet(epochs=2, batch_size=16, width=16), splitter=GroupKFoldSplitter(),
                      out_dir=tmp_path, task=Task(target="label_pose", collapse_lying=True, window_len=16,
                                                label_rule="any_lying", max_points=12),
                      config={"model": {"name": "pointnet", "pointnet": {"epochs": 2, "batch_size": 16, "lr": 0.001,
                                                                          "patience": 5, "evals_per_epoch": 1}},
                              "eval": {"folds": None, "seed": 0, "max_points": 12}, "input_type": "pointcloud"}).cross_validate(x, y, groups)
    assert len(results["folds"]) == 4 and all(f["history"] for f in results["folds"])
    assert all((tmp_path / f["checkpoint"]).exists() for f in results["folds"])
    assert "point cloud" in (tmp_path / "report.md").read_text()


def test_regularisation_only_acts_in_training_and_checkpoint_keeps_it():
    net = MiniPointNet(2, width=16, dropout=0.5, point_dropout=0.5, point_drop=0.5, jitter=0.5)
    x = torch.randn(4, 10, 7)
    x[..., 6] = 1.0
    net.eval()
    assert torch.equal(net(x), net(x))  # deterministic in eval mode
    net.train()
    assert not torch.equal(net(x), net(x))  # dropout, point dropping and jitter are random in training
    model = PointNet(point_dropout=0.1, point_drop=0.2, jitter=0.03)
    assert model._build_net(2).point_drop == 0.2


def test_rnn_regularisation_only_acts_in_training():
    from models.rnn import HeatmapGRU
    net = HeatmapGRU(2, feat_dim=8, hidden=8, dropout=0.5, feature_dropout=0.5, noise=0.5)
    x = torch.randn(2, 3, 1, 256, 16)
    net.eval()
    assert torch.equal(net(x), net(x))
    net.train()
    assert not torch.equal(net(x), net(x))


def test_point_count_feature_ignores_augmentation_dropping():
    from models.pointnet import MiniPointNet

    torch.manual_seed(0)
    x = torch.rand(8, 40, 7)
    x[..., 6] = 1.0
    net = MiniPointNet(2, point_drop=0.9)
    net.train()
    seen = []
    net.head.register_forward_hook(lambda module, inputs, out: seen.append(inputs[0][:, -1]))
    net(x)
    expected = torch.log1p(torch.tensor(40.0)) / 5.0  # all 40 points are real, however many were dropped
    assert torch.allclose(seen[0], expected.expand(8))
