from datetime import datetime, timedelta

import numpy as np
import pytest
from datasets import Dataset, concatenate_datasets

from config import WindowConfig
from data.windowing import (
    HEATMAP_SHAPE, build_window_inputs, build_windows, collapse_lying, filter_valid_frames,
    make_windows, slice_window_indices, window_timing,
)

PERIOD = 0.2  # seconds between frames, as in the real recordings


def make_ds(labels, person="Bob", recording="r1", drop=()):
    """One recording; `drop` removes frames to create gaps in capture order."""
    start = datetime(2026, 1, 1)
    rows = [i for i in range(len(labels)) if i not in set(drop)]
    return Dataset.from_dict({
        "recording": [recording] * len(rows),
        "person": [person] * len(rows),
        "frame_idx": rows,
        "timestamp": [start + timedelta(seconds=PERIOD * i) for i in rows],
        "label_pose": [labels[i] for i in rows],
        "heatmap_ok": [True] * len(rows),
        "num_range_bins": [HEATMAP_SHAPE[0]] * len(rows),
        "num_doppler_bins": [HEATMAP_SHAPE[1]] * len(rows),
    })


def windows(ds, **kwargs):
    x, y, groups = make_windows(ds, window_len=16, stride=8, inputs="none", **kwargs)
    return y, groups


@pytest.fixture
def blip():
    """32 frames; a single lying frame (0.2 s) at index 5, inside the first window only."""
    labels = ["standing"] * 32
    labels[5] = "lying_left"
    return labels


def test_last_rule_misses_a_lying_blip(blip):
    y, _ = windows(make_ds(blip), label_rule="last")
    assert list(y) == ["standing", "standing", "standing"]


def test_any_lying_rule_keeps_the_blip(blip):
    y, _ = windows(make_ds(blip), label_rule="any_lying")
    assert list(y) == ["lying_left", "standing", "standing"]


def test_any_lying_keeps_left_vs_right_and_collapses(blip):
    labels = list(blip)
    labels[6] = "lying_right"  # the last lying frame in the window decides
    y, _ = windows(make_ds(labels), label_rule="any_lying")
    assert y[0] == "lying_right"
    y, _ = windows(make_ds(labels), label_rule="any_lying", relabel=collapse_lying)
    assert list(y) == ["lying", "not_lying", "not_lying"]


def test_default_rule_is_last(blip):
    y, _ = windows(make_ds(blip))
    assert "lying_left" not in y


def test_unknown_label_rule_raises(blip):
    with pytest.raises(ValueError, match="label_rule"):
        windows(make_ds(blip), label_rule="majority")


def test_window_with_a_dropped_frame_is_skipped(blip):
    y, _ = windows(make_ds(blip, drop=[10]), label_rule="any_lying")
    assert len(y) < 3  # windows spanning the gap are not used


def test_group_is_the_person(blip):
    _, groups = windows(make_ds(blip, person="Frank"))
    assert set(groups) == {"Frank"}


def test_window_timing_in_seconds(blip):
    timing = window_timing(make_ds(blip), window_len=16, stride=8)
    assert timing["frame_period_s"] == pytest.approx(PERIOD)
    assert timing["window_s"] == pytest.approx(3.2)
    assert timing["stride_s"] == pytest.approx(1.6)


def test_window_timing_ignores_long_gaps():
    ds = make_ds(["standing"] * 40, drop=range(10, 30))  # a 4 s hole in the recording
    assert window_timing(ds, 16, 8)["frame_period_s"] == pytest.approx(PERIOD)


def test_limit_per_group_keeps_rare_windows_and_caps_every_person():
    from data.windowing import limit_per_group

    groups = np.array(["a"] * 20 + ["b"] * 20 + ["c"] * 3)
    y = np.array((["lying"] * 2 + ["not_lying"] * 18) + ["not_lying"] * 20 + ["not_lying"] * 3)
    keep = limit_per_group(y, groups, 5, seed=1)
    assert (np.diff(keep) > 0).all()  # sorted, no repeats
    for g, expected in (("a", 5), ("b", 5), ("c", 3)):
        assert (groups[keep] == g).sum() == expected
    assert (y[keep][groups[keep] == "a"] == "lying").sum() == 2  # both lying windows survive
    assert np.array_equal(keep, limit_per_group(y, groups, 5, seed=1))  # seeded
    assert len(limit_per_group(y, groups, 100)) == len(y)  # a big limit keeps everything


def test_make_windows_can_limit_windows_per_person():
    labels = ["standing"] * 80
    y_all, g_all = windows(make_ds(labels))
    assert len(y_all) > 3
    x, y, groups = make_windows(make_ds(labels), window_len=16, stride=8, inputs="none", max_per_person=3)
    assert len(x) == len(y) == len(groups) == 3


def test_frame_coverage_counts_each_rejection_once_and_keeps_capture_positions():
    ds = make_ds(["standing", None, "standing", "standing", None, "lying_left", "standing", None])
    ds = ds.remove_columns(["heatmap_ok", "num_range_bins"])
    ds = ds.add_column("heatmap_ok", [True, False, False, True, True, True, True, False])
    ds = ds.add_column("num_range_bins", [256, 256, 256, 128, 128, 256, 256, 256])
    ds = concatenate_datasets([ds, make_ds([None, None], recording="waiting")])
    original = ds.to_dict()

    frames, meta, coverage = filter_valid_frames(ds, "label_pose")
    assert coverage.recordings_used == ["r1"]
    assert coverage.recordings_unlabelled == ["waiting"]
    assert (coverage.frames_total, coverage.frames_unlabelled, coverage.frames_bad_heatmap,
            coverage.frames_wrong_shape, coverage.frames_kept) == (10, 5, 1, 1, 3)
    assert coverage.frames_total == (coverage.frames_unlabelled + coverage.frames_bad_heatmap
                                     + coverage.frames_wrong_shape + coverage.frames_kept)
    assert frames["frame_idx"] == meta.frame_idx.tolist() == [0, 5, 6]
    assert meta.capture_idx.tolist() == [0, 5, 6]
    assert meta.index.tolist() == list(range(len(frames)))
    assert ds.to_dict() == original  # source frames are preserved


def test_windows_crossing_filtered_frames_are_counted_and_skipped(caplog):
    ds = make_ds(["standing"] * 4 + [None] + ["standing"] * 4)
    _, meta, _ = filter_valid_frames(ds, "label_pose")
    frames, y, groups, skipped = slice_window_indices(meta, "label_pose", 3, 1, "last")
    np.testing.assert_array_equal(frames, [[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]])
    assert y.tolist() == ["standing"] * 4 and groups.tolist() == ["Bob"] * 4
    assert skipped == 2
    with caplog.at_level("INFO", logger="data.windowing"):
        _, _, _, coverage = build_windows(
            ds, WindowConfig(target="label_pose", window_len=3, stride=1), "none",
        )
    assert coverage.windows_gap_skipped == 2 and coverage.windows_kept == 4
    assert "9 total -> 8 labelled (-1 unlabelled)" in caplog.text
    assert "windows 4 kept, 2 gap-spanning skipped" in caplog.text


@pytest.mark.parametrize("rule,collapse,expected", [
    ("last", False, ["standing", "standing"]),
    ("any_lying", False, ["lying_left", "lying_right"]),
    ("any_lying", True, ["lying", "lying"]),
])
def test_slicing_uses_capture_order_and_preserves_dataset_positions(rule, collapse, expected):
    ds = make_ds(["standing", "lying_left", "standing", "lying_right", "standing", "standing"])
    ds = ds.remove_columns("frame_idx").add_column("frame_idx", [0, 4, 5, 1, 2, 3])
    ds = ds.select([3, 0, 5, 1, 4, 2])
    _, meta, _ = filter_valid_frames(ds, "label_pose")
    frames, y, groups, skipped = slice_window_indices(meta, "label_pose", 3, 2, rule, collapse)
    np.testing.assert_array_equal(frames, [[1, 3, 5], [5, 0, 4]])
    assert y.tolist() == expected and groups.tolist() == ["Bob", "Bob"]
    assert skipped == 0


def test_slicing_rejects_an_unknown_person_anywhere_in_a_recording():
    ds = make_ds(["standing"] * 4)
    ds = ds.remove_columns("person").add_column("person", ["Bob", "NA", "Bob", "Bob"])
    _, meta, _ = filter_valid_frames(ds, "label_pose")
    with pytest.raises(ValueError, match="no known person"):
        slice_window_indices(meta, "label_pose", 2, 1, "last")


@pytest.mark.parametrize("window_len,stride", [(0, 1), (2, 0)])
def test_slicing_requires_positive_window_length_and_stride(window_len, stride):
    _, meta, _ = filter_valid_frames(make_ds(["standing"] * 4), "label_pose")
    with pytest.raises(ValueError, match="must be positive"):
        slice_window_indices(meta, "label_pose", window_len, stride, "last")


@pytest.mark.parametrize("input_type", ["heatmap", "pointcloud", "none"])
def test_input_builders_gather_the_same_frames_in_window_order(input_type):
    heatmaps = np.arange(24).reshape(3, 4, 2)
    clouds = [[[i, 0, 0, 0, 20, 40, 255]] for i in range(3)]
    ds = Dataset.from_dict({"heatmap": heatmaps.tolist(), "pointcloud": clouds})
    windows = np.array([[2, 0], [1, 2]])
    x = build_window_inputs(ds, windows, input_type, shape=(4, 2), max_points=3)
    assert x.dtype == np.float32
    if input_type == "heatmap":
        np.testing.assert_allclose(x, np.log1p(heatmaps.astype(np.float32))[windows, None])
    elif input_type == "pointcloud":
        assert x.shape == (2, 3, 7)
        np.testing.assert_array_equal(x[:, :2, 0], windows)
        np.testing.assert_array_equal(x[:, :2, 5], [[0, 1], [0, 1]])
        assert (x[:, :2, 6] == 1).all() and not x[:, 2].any()
    else:
        assert x.shape == (2, 2) and not x.any()


@pytest.mark.parametrize("input_type,expected", [
    ("heatmap", (0, 3, 1, 4, 2)), ("pointcloud", (0, 5, 7)), ("none", (0, 3)),
])
def test_empty_input_builders_keep_shapes_without_reading_payload_columns(input_type, expected):
    x = build_window_inputs(Dataset.from_dict({}), np.empty((0, 3), dtype=int), input_type,
                            shape=(4, 2), max_points=5)
    assert x.shape == expected and x.dtype == np.float32


def test_build_windows_uses_config_and_reports_the_final_capped_window_count(blip):
    config = WindowConfig(target="label_pose", window_len=8, stride=4, label_rule="any_lying", collapse_lying=True)
    x, y, groups, coverage = build_windows(make_ds(blip), config, "none", max_per_person=3, seed=7)
    assert x.shape == (3, 8) and groups.tolist() == ["Bob"] * 3
    assert sorted(y.tolist()) == ["lying", "lying", "not_lying"]
    assert coverage.frames_total == coverage.frames_kept == 32
    assert coverage.windows_kept == 3 and coverage.windows_gap_skipped == 0
    _, uncollapsed, _, _ = build_windows(make_ds(blip), WindowConfig(
        target="label_pose", window_len=8, stride=4, label_rule="any_lying", collapse_lying=False,
    ), "none")
    assert "lying_left" in uncollapsed


@pytest.mark.parametrize("input_type,expected", [
    ("heatmap", (0, 3, 1, 256, 16)), ("pointcloud", (0, 5, 7)), ("none", (0, 3)),
])
def test_unlabelled_recordings_return_empty_windows_and_stay_in_the_dataset(input_type, expected, caplog):
    ds = make_ds([None] * 6)
    x, y, groups, coverage = build_windows(
        ds, WindowConfig(target="label_pose", window_len=3, max_points=5), input_type,
    )
    assert x.shape == expected and len(y) == len(groups) == 0
    assert coverage.frames_total == coverage.frames_unlabelled == 6
    assert coverage.frames_kept == coverage.windows_kept == coverage.windows_gap_skipped == 0
    assert coverage.recordings_used == [] and coverage.recordings_unlabelled == ["r1"]
    assert len(ds) == 6 and "source frames remain on disk" in caplog.text
