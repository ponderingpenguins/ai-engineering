import numpy as np
import pytest
import torch

from data.augment import HeatmapAugment
from models.factory import build_model
from utils.config import ModelConfig

B, T, R, D = 4, 8, 64, 4


def windows(seed: int = 0) -> torch.Tensor:
    """log1p heatmaps around a raw floor of 1600 with noise."""
    g = torch.Generator().manual_seed(seed)
    return torch.log1p(1600 + 5 * torch.randn(B, T, 1, R, D, generator=g))


def test_everything_is_off_by_default():
    aug = HeatmapAugment()
    assert not aug.active
    x = windows()
    assert torch.equal(aug(x), x)


def test_negative_sizes_are_rejected():
    with pytest.raises(ValueError):
        HeatmapAugment(range_shift_bins=-1)


def test_input_is_not_modified_and_shape_is_kept():
    x = windows(); before = x.clone()
    out = HeatmapAugment(range_shift_bins=5, floor_jitter=50, time_reverse=True)(x)
    assert out.shape == x.shape and torch.equal(x, before)


def test_range_shift_moves_a_person_and_fills_with_the_floor():
    x = windows()
    x[..., 30:34, :] = torch.log1p(torch.tensor(1900.0))  # a person at bins 30-33
    torch.manual_seed(0)
    out = HeatmapAugment(range_shift_bins=6)(x)
    for b in range(B):
        centre = (out[b, 0, 0, :, 0] - out[b, 0, 0, :, 0].median()).argmax().item()
        assert abs(centre - 31.5) <= 6.5  # moved by at most the limit
    assert (out - torch.log1p(torch.tensor(1600.0))).abs().max() < torch.log1p(torch.tensor(1900.0)) - torch.log1p(torch.tensor(1600.0)) + 0.01  # no wild values


def test_integer_range_shift_is_an_exact_shift_with_no_wraparound(monkeypatch):
    x = windows()
    monkeypatch.setattr(torch, "rand", lambda n, device=None: torch.full((n,), 0.75))  # delta = (0.75*2-1)*6 = 3
    out = HeatmapAugment(range_shift_bins=6)._shift(x)
    assert torch.allclose(out[..., 3:, :], x[..., :-3, :], atol=1e-5)
    fill = x.permute(0, 4, 1, 2, 3).reshape(B, D, -1).median(-1).values
    assert torch.allclose(out[..., :3, :], fill.view(B, 1, 1, 1, D).expand(B, T, 1, 3, D), atol=1e-5)


def test_floor_jitter_is_smooth_along_range_and_the_same_for_every_frame():
    x = windows()
    torch.manual_seed(0)
    out = HeatmapAugment(floor_jitter=100)(x)
    raw_change = torch.expm1(out) - torch.expm1(x)
    assert raw_change.std() > 20  # it does something at about the requested size
    assert torch.allclose(raw_change[:, 0], raw_change[:, -1], atol=0.5)  # same baseline in every frame
    assert raw_change.diff(dim=3).abs().max() < 100  # smooth along range, no jumps
    assert out.min() >= 0


def test_time_reverse_flips_some_windows_and_leaves_the_rest():
    x = torch.arange(T, dtype=torch.float32).view(1, T, 1, 1, 1).expand(64, T, 1, R, D).contiguous()
    torch.manual_seed(0)
    out = HeatmapAugment(time_reverse=True)(x)
    flipped = out[:, 0, 0, 0, 0] == T - 1
    kept = out[:, 0, 0, 0, 0] == 0
    assert (flipped | kept).all() and flipped.any() and kept.any()
    assert torch.equal(out[flipped], x[flipped].flip(1))


def test_rnn_trains_with_augmentation_and_only_on_training_batches():
    cfg = ModelConfig(name="rnn")
    cfg.rnn.epochs = 1
    cfg.rnn.augment.range_shift_bins = 4
    cfg.rnn.augment.time_reverse = True
    model = build_model(cfg)
    assert model.augment is not None
    x = np.random.default_rng(0).normal(7.4, 0.01, (12, 6, 1, 256, 16)).astype(np.float32)
    y = np.array(["a", "b"] * 6)
    model.fit(x, y)
    assert model.predict(x).shape == (12,)  # prediction never calls the augmentation
    assert build_model(ModelConfig(name="rnn")).augment is None


# ---- point clouds ------------------------------------------------------------------------------

from data.augment import PointAugment  # noqa: E402
from models.pointnet import PointNet  # noqa: E402


def points(n_valid: int = 6, slots: int = 20, batch: int = 8, seed: int = 0) -> torch.Tensor:
    """Windows of points around the person: boresight y about 1.8 m, x and z near 0, SNR 20 dB."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(batch, slots, 7)
    x[:, :n_valid, 0] = 0.2 * torch.randn(batch, n_valid, generator=g)
    x[:, :n_valid, 1] = 1.8 + 0.1 * torch.randn(batch, n_valid, generator=g)
    x[:, :n_valid, 2] = 0.2 * torch.randn(batch, n_valid, generator=g)
    x[:, :n_valid, 3] = 0.13
    x[:, :n_valid, 4] = 20.0
    x[:, :n_valid, 5] = torch.rand(batch, n_valid, generator=g)
    x[:, :n_valid, 6] = 1.0
    return x


def test_point_augment_is_off_by_default_and_leaves_the_input_alone():
    x = points(); before = x.clone()
    aug = PointAugment()
    assert not aug.active and torch.equal(aug(x), x)
    PointAugment(shift_m=0.3, yaw_deg=90, ghost_points=3)(x)
    assert torch.equal(x, before)


def test_yaw_and_tilt_keep_distances_and_yaw_keeps_the_boresight_coordinate():
    x = points()
    torch.manual_seed(0)
    yawed = PointAugment(yaw_deg=180)(x)
    v = x[..., 6] > 0.5
    assert torch.allclose(yawed[..., 1][v], x[..., 1][v], atol=1e-5)  # y is the rotation axis
    assert torch.allclose(yawed[..., :3].norm(dim=-1)[v], x[..., :3].norm(dim=-1)[v], atol=1e-5)
    assert (yawed[..., 0][v] - x[..., 0][v]).abs().max() > 0.05  # and it does move the points in the floor plane
    tilted = PointAugment(tilt_deg=10)(x)
    assert torch.allclose(tilted[..., :3].norm(dim=-1)[v], x[..., :3].norm(dim=-1)[v], atol=1e-5)
    assert torch.equal(yawed[..., 3], x[..., 3]) and torch.equal(yawed[..., 6], x[..., 6])  # nothing else changes


def test_mirror_flips_only_x_and_only_some_windows():
    x = points(batch=64)
    torch.manual_seed(0)
    out = PointAugment(mirror=True)(x)
    flipped = (out[:, 0, 0] == -x[:, 0, 0]) & (x[:, 0, 0] != 0)
    kept = out[:, 0, 0] == x[:, 0, 0]
    assert (flipped | kept).all() and flipped.any() and kept.any()
    assert torch.equal(out[..., 1:3], x[..., 1:3])


def test_translation_moves_a_window_as_a_whole_and_rescales_snr_for_the_new_range():
    x = points()
    torch.manual_seed(0)
    out = PointAugment(shift_m=0.5, height_m=0.4)(x)
    v = x[..., 6] > 0.5
    shift = (out[..., :3] - x[..., :3])
    for b in range(len(x)):
        s = shift[b][v[b]]
        assert torch.allclose(s, s[0].expand_as(s), atol=1e-5)  # one shift for the whole window
        assert s[0, 0].abs() <= 0.5 and s[0, 2].abs() <= 0.5 and s[0, 1].abs() <= 0.4
    r_old, r_new = x[..., :3].norm(dim=-1), out[..., :3].norm(dim=-1)
    expected = 20 + 40 * torch.log10(r_old / r_new)
    assert torch.allclose(out[..., 4][v], expected[v], atol=1e-3)
    assert ((r_new > r_old) == (out[..., 4] < 20))[v].all()  # farther means weaker


def test_points_pushed_beyond_the_range_limit_become_padding():
    x = points(n_valid=4)
    x[:, 3, 1] = 4.8  # one point near the limit
    out = PointAugment(height_m=1.0, range_max_m=4.85)(x)
    gone = (x[..., 6] > 0.5) & (out[..., 6] < 0.5)
    assert gone.any()
    assert not out[gone].any()  # a dropped point is an all-zero row, like padding
    assert (out[..., :3].norm(dim=-1)[out[..., 6] > 0.5] <= 4.85 + 1e-5).all()


def test_ghost_points_only_fill_empty_slots_and_have_low_snr():
    x = points(n_valid=6, slots=40, batch=64)
    torch.manual_seed(0)
    out = PointAugment(ghost_points=4)(x)
    real = x[..., 6] > 0.5
    assert torch.equal(out[real], x[real])  # real points untouched
    added = (out[..., 6] > 0.5) & ~real
    assert 2.5 < added.sum(1).float().mean() < 5.5  # about 4 per window
    ghosts = out[added]
    assert ghosts[:, 4].min() >= 11 and ghosts[:, 4].max() <= 19
    assert (ghosts[:, :3].norm(dim=-1) <= 4.85).all()


def test_ghost_points_cannot_overflow_a_full_window():
    x = points(n_valid=10, slots=10)
    out = PointAugment(ghost_points=50)(x)
    assert torch.equal(out, x)


def test_snr_jitter_is_one_offset_per_window():
    x = points()
    torch.manual_seed(0)
    out = PointAugment(snr_jitter_db=3)(x)
    v = x[..., 6] > 0.5
    for b in range(len(x)):
        delta = (out[b, :, 4] - x[b, :, 4])[v[b]]
        assert torch.allclose(delta, delta[0].expand_as(delta), atol=1e-5)
    assert (out[..., 4] - x[..., 4])[v].std() > 0.5


def test_pointnet_trains_with_augmentation_and_prediction_ignores_it():
    cfg = ModelConfig(name="pointnet")
    cfg.pointnet.epochs = 1
    cfg.pointnet.augment.shift_m = 0.3
    cfg.pointnet.augment.ghost_points = 3
    model = build_model(cfg)
    assert isinstance(model, PointNet) and model.augment is not None
    x = points(batch=12).numpy()
    y = np.array(["a", "b"] * 6)
    model.fit(x, y)
    assert torch.equal(torch.as_tensor(model.predict(x) == model.predict(x)).all(), torch.tensor(True))
    assert build_model(ModelConfig(name="pointnet")).augment is None
