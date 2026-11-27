import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from data.annotations import NO_PERSON
from data.preprocess import (
    CONFIG_4,
    DropRecordings,
    DropUnknownPerson,
    KeepSensorConfig,
    RangeNormalise,
    TemporalBackground,
    denoise_heatmaps,
    preprocess_dataset,
    select_recordings,
)
from config.data import DataConfig, PreprocessConfig, SelectionConfig


@pytest.fixture
def ds():
    return Dataset.from_dict({
        "recording": ["r1", "r2", "r3", "r4"],
        "cfg_file": [CONFIG_4, CONFIG_4, "Config_3.cfg", CONFIG_4],
        "person": ["Bob", NO_PERSON, "Dave", "Ian"],
        "heatmap": [[0.0], [0.0], [0.0], [0.0]],
    })


def test_keep_sensor_config_defaults_to_config_4(ds):
    assert KeepSensorConfig()(ds)["recording"] == ["r1", "r2", "r4"]
    assert KeepSensorConfig("Config_3.cfg")(ds)["recording"] == ["r3"]


def test_drop_unknown_person(ds):
    assert DropUnknownPerson()(ds)["recording"] == ["r1", "r3", "r4"]


def test_selection_applies_gates_in_order_and_preserves_other_columns(ds, caplog):
    selection = SelectionConfig(exclude_recordings=["r1"])
    with caplog.at_level("INFO", logger="data.preprocess"):
        out = select_recordings(ds, selection)
    assert out.to_dict() == ds.select([3]).to_dict()
    assert out.features == ds.features
    assert caplog.messages == [
        "preprocess known_person: 4 -> 3 frames",
        "preprocess exclude_recordings: 3 -> 2 frames",
        "preprocess sensor_config: 2 -> 1 frames",
    ]


def test_disabling_optional_selection_gates_only_drops_unknown_persons(ds, caplog):
    selection = SelectionConfig(sensor_config=None, exclude_recordings=[])
    with caplog.at_level("INFO", logger="data.preprocess"):
        out = select_recordings(ds, selection)
    assert out["recording"] == ["r1", "r3", "r4"]
    assert caplog.messages == ["preprocess known_person: 4 -> 3 frames"]


def test_selection_can_keep_a_different_sensor_config(ds):
    selection = SelectionConfig(sensor_config="Config_3.cfg", exclude_recordings=[])
    assert select_recordings(ds, selection)["recording"] == ["r3"]


def test_named_recordings_are_dropped(ds):
    assert DropRecordings(["r1", "r4"])(ds)["recording"] == ["r2", "r3"]
    out = select_recordings(ds, SelectionConfig(exclude_recordings=["r1"]))
    assert out["recording"] == ["r4"]  # r2 has no person, r3 is Config 3, r1 is excluded


def test_excluding_a_recording_that_does_not_exist_warns(ds, caplog):
    with caplog.at_level("WARNING"):
        out = select_recordings(ds, SelectionConfig(sensor_config=None, exclude_recordings=["no_such_recording"]))
    assert out["recording"] == ["r1", "r3", "r4"]
    assert "no_such_recording" in caplog.text


def test_the_default_config_excludes_the_two_non_top_down_recordings():
    assert DataConfig().selection.exclude_recordings == ["2311__20231124_152926", "2311__20231124_153817"]


def test_selection_can_drop_every_row_without_changing_the_schema(ds):
    out = select_recordings(ds, SelectionConfig(exclude_recordings=["r1", "r3", "r4"]))
    assert len(out) == 0 and out.features == ds.features


def test_disabled_denoising_returns_the_original_dataset(ds):
    # This dataset lacks transform metadata; disabled transforms must not read it.
    assert denoise_heatmaps(ds, PreprocessConfig()) is ds


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


def test_denoising_is_off_by_default():
    assert DataConfig().selection.sensor_config == CONFIG_4
    assert not PreprocessConfig().range_normalise.enabled and not PreprocessConfig().temporal_background.enabled


@pytest.mark.parametrize("range_enabled,temporal_enabled", [(True, False), (False, True), (True, True)])
def test_denoising_uses_enabled_transforms_in_order_with_configured_parameters(range_enabled, temporal_enabled, caplog):
    frames = noisy(40, np.linspace(1500, 1800, R))
    frames[20:, 30:36, :] += 100
    ds = heatmap_ds({"r": frames})
    config = PreprocessConfig()
    config.range_normalise.enabled = range_enabled
    config.range_normalise.percentile = 25
    config.range_normalise.smooth_bins = 9
    config.temporal_background.enabled = temporal_enabled
    config.temporal_background.tau_frames = 7
    expected, names = ds, []
    if range_enabled:
        expected = RangeNormalise(percentile=25, smooth_bins=9)(expected)
        names.append("range_normalise")
    if temporal_enabled:
        expected = TemporalBackground(tau_frames=7)(expected)
        names.append("temporal_background")

    with caplog.at_level("INFO", logger="data.preprocess"):
        out = denoise_heatmaps(ds, config)
    np.testing.assert_allclose(out["heatmap"], expected["heatmap"], atol=1e-3)
    metadata = [column for column in ds.column_names if column != "heatmap"]
    assert out.select_columns(metadata).to_dict() == ds.select_columns(metadata).to_dict()
    assert caplog.messages == [f"preprocess {name}: 40 -> 40 frames" for name in names]


def test_preprocessing_selects_recordings_before_denoising(caplog):
    frames = noisy(10, np.full(R, 1600.0))
    ds = heatmap_ds({name: frames for name in ("kept", "unknown", "excluded", "other_config")})
    ds = ds.add_column("person", ["Bob"] * 10 + [NO_PERSON] * 10 + ["Dave"] * 20)
    ds = ds.add_column("cfg_file", [CONFIG_4] * 30 + ["Config_3.cfg"] * 10)
    selection = SelectionConfig(exclude_recordings=["excluded"])
    denoising = PreprocessConfig()
    denoising.range_normalise.enabled = True
    denoising.temporal_background.enabled = True

    with caplog.at_level("INFO", logger="data.preprocess"):
        out = preprocess_dataset(ds, selection, denoising)
    assert out["recording"] == ["kept"] * 10
    assert np.asarray(out["heatmap"]).shape == (10, R, D)
    assert np.asarray(out["heatmap"]).mean() < 15
    assert caplog.messages == [
        "preprocess known_person: 40 -> 30 frames",
        "preprocess exclude_recordings: 30 -> 20 frames",
        "preprocess sensor_config: 20 -> 10 frames",
        "preprocess range_normalise: 10 -> 10 frames",
        "preprocess temporal_background: 10 -> 10 frames",
    ]


def test_range_normalise_default_window_ignores_a_person_as_wide_as_a_lying_one():
    frames = np.random.default_rng(0).normal(1600, 5, (200, 256, 4))  # a Config_4 sized range axis
    frames[:, 70:87, :] += 100  # a person 17 bins (0.4 m) deep, there for the whole recording
    transform = RangeNormalise()
    assert transform.smooth_bins >= 35  # more than twice the person
    out = transform.transform(frames.astype(np.float32))
    assert out[:, 70:87, :].mean() > 90  # not absorbed into the floor
    assert abs(out[:, 120:200, :]).mean() < 15
