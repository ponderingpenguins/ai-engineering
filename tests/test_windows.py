from datetime import datetime, timedelta

import numpy as np
import pytest
from datasets import Dataset

from data.windows import HEATMAP_SHAPE, collapse_lying, make_windows, window_timing

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
    from data.windows import limit_per_group

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
