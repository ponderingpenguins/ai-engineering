import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from data.extract import NO_PERSON
from data.preprocess import (
    CONFIG_4,
    DropRecordings,
    DropUnknownPerson,
    KeepConfig4,
    RangeNormalise,
    TemporalBackground,
    build_pipeline,
)
from config import DataConfig, PreprocessConfig


@pytest.fixture
def ds():
    return Dataset.from_dict({
        "recording": ["r1", "r2", "r3", "r4"],
        "cfg_file": [CONFIG_4, CONFIG_4, "Config_3.cfg", CONFIG_4],
        "person": ["Bob", NO_PERSON, "Dave", "Ian"],
        "heatmap": [[0.0], [0.0], [0.0], [0.0]],
    })


def test_keep_config_4(ds):
    assert KeepConfig4()(ds)["recording"] == ["r1", "r2", "r4"]


def test_drop_unknown_person(ds):
    assert DropUnknownPerson()(ds)["recording"] == ["r1", "r3", "r4"]


def test_pipeline_applies_steps_in_order_and_keeps_other_columns(ds):
    out = build_pipeline(["config_4", "known_person"])(ds)
    assert out["recording"] == ["r1", "r4"]
    assert out.column_names == ds.column_names


def test_empty_pipeline_only_drops_unknown_persons(ds):
    assert build_pipeline([])(ds)["recording"] == ["r1", "r3", "r4"]


def test_unknown_step_name_raises():
    with pytest.raises(ValueError, match="unknown preprocessor"):
        build_pipeline(["nope"])


def test_named_recordings_are_dropped_from_the_pipeline(ds):
    assert DropRecordings(["r1", "r4"])(ds)["recording"] == ["r2", "r3"]
    out = build_pipeline(["config_4"], exclude=["r1"])(ds)
    assert out["recording"] == ["r4"]  # r2 has no person, r3 is Config 3, r1 is excluded
    assert build_pipeline([], exclude=[])(ds)["recording"] == ["r1", "r3", "r4"]  # nothing excluded by default


def test_excluding_a_recording_that_does_not_exist_warns(ds, caplog):
    with caplog.at_level("WARNING"):
        out = DropRecordings(["no_such_recording"])(ds)
    assert out["recording"] == ds["recording"]
    assert "no_such_recording" in caplog.text


def test_the_default_config_excludes_the_two_non_top_down_recordings():
    assert DataConfig().selection.exclude_recordings == ["2311__20231124_152926", "2311__20231124_153817"]


def test_unknown_persons_are_dropped_even_if_not_requested(ds):
    out = build_pipeline(["config_4"])(ds)
    assert "NA" not in out["person"]
    assert "NA" not in build_pipeline([])(ds)["person"]


R, D = 64, 4  # a small heatmap, enough to exercise the transforms


def heatmap_ds(recordings: dict[str, np.ndarray], ok: dict[str, list[bool]] | None = None) -> Dataset:
    """A dataset with one row per frame; `recordings` maps a recording name to (n, R, D) heatmaps."""
    rows: dict[str, list] = {k: [] for k in ("recording", "timestamp", "heatmap", "heatmap_ok", "num_range_bins", "num_doppler_bins")}
    for name, frames in recordings.items():
        flags = (ok or {}).get(name, [True] * len(frames))
        for t, (frame, good) in enumerate(zip(frames, flags)):
            rows["recording"].append(name)
            rows["timestamp"].append(pd.Timestamp("2023-11-20") + pd.Timedelta(seconds=0.2 * t))
            rows["heatmap"].append(frame.astype(np.uint16).tolist() if good else [])
            rows["heatmap_ok"].append(good)
            rows["num_range_bins"].append(R)
            rows["num_doppler_bins"].append(D)
    return Dataset.from_dict(rows)


def noisy(n: int, level: np.ndarray, seed: int = 0) -> np.ndarray:
    """n frames whose range bins sit at `level` (R,), plus noise."""
    return level[None, :, None] + np.random.default_rng(seed).normal(0, 5, (n, R, D))


def test_range_normalise_flattens_the_floor_and_keeps_a_person():
    floor = np.linspace(1500, 1800, R)  # the floor rises with range
    frames = noisy(200, floor)
    frames[:, 30:36, :] += 100  # a person in the room for the whole recording
    out = np.stack(RangeNormalise(smooth_bins=21)(heatmap_ds({"r": frames}))["heatmap"])
    assert out.shape == frames.shape and out.min() >= 0
    quiet = np.r_[0:25, 45:R]
    assert out[:, quiet, :].mean() < 15  # the range trend is gone
    assert out[:, 30:36, :].mean() > 70  # and the person is still there


def test_range_normalise_is_per_recording():
    a, b = noisy(100, np.full(R, 1500.0)), noisy(100, np.full(R, 1800.0), seed=1)
    out = RangeNormalise(smooth_bins=21)(heatmap_ds({"a": a, "b": b}))
    arr = np.stack(out["heatmap"])
    assert arr[:100].mean() < 15 and arr[100:].mean() < 15  # each recording is judged against its own floor


def test_temporal_background_removes_static_and_keeps_a_new_arrival():
    frames = noisy(400, np.full(R, 1600.0))
    frames[:, 10:14, :] += 200  # furniture, there from the start
    frames[300:, 40:44, :] += 150  # a person who walks in at frame 300
    out = np.stack(TemporalBackground(tau_frames=50)(heatmap_ds({"r": frames}))["heatmap"])
    assert out[100:300, 10:14, :].mean() < 20  # the furniture has faded into the background
    assert out[300:305, 40:44, :].mean() > 100  # the arrival shows at nearly full strength


def test_temporal_background_is_causal():
    frames = noisy(100, np.full(R, 1600.0))
    full = np.stack(TemporalBackground()(heatmap_ds({"r": frames}))["heatmap"])
    cut = np.stack(TemporalBackground()(heatmap_ds({"r": frames[:60]}))["heatmap"])
    np.testing.assert_allclose(full[:60], cut, atol=1e-3)


def test_heatmap_transforms_keep_rows_and_leave_bad_frames_alone():
    frames = noisy(10, np.full(R, 1600.0))
    ok = {"r": [True] * 4 + [False] + [True] * 5}
    ds = heatmap_ds({"r": frames}, ok)
    for step in (RangeNormalise(), TemporalBackground()):
        out = step(ds)
        assert len(out) == len(ds) and out.column_names == ds.column_names
        assert out["heatmap"][4] == [] and out["heatmap_ok"][4] is False


def test_heatmap_transforms_follow_capture_order_not_row_order():
    frames = noisy(60, np.full(R, 1600.0))
    frames[30:, :, :] += 300  # the scene changes half way
    ds = heatmap_ds({"r": frames})
    shuffled = ds.select(list(range(59, -1, -1)))  # same frames, reversed row order
    out = np.stack(TemporalBackground(tau_frames=20)(shuffled)["heatmap"])[::-1]
    np.testing.assert_allclose(out, np.stack(TemporalBackground(tau_frames=20)(ds)["heatmap"]), atol=1e-3)


def test_new_steps_are_registered_and_off_by_default():
    assert isinstance(build_pipeline(["range_normalise", "temporal_background"]).steps[1], RangeNormalise)  # steps[0] is known_person
    assert DataConfig().selection.sensor_config == CONFIG_4
    assert not PreprocessConfig().range_normalise.enabled and not PreprocessConfig().temporal_background.enabled


def test_range_normalise_default_window_ignores_a_person_as_wide_as_a_lying_one():
    frames = np.random.default_rng(0).normal(1600, 5, (200, 256, 4))  # a Config_4 sized range axis
    frames[:, 70:87, :] += 100  # a person 17 bins (0.4 m) deep, there for the whole recording
    transform = RangeNormalise()
    assert transform.smooth_bins >= 35  # more than twice the person
    out = transform.transform(frames.astype(np.float32))
    assert out[:, 70:87, :].mean() > 90  # not absorbed into the floor
    assert abs(out[:, 120:200, :]).mean() < 15
