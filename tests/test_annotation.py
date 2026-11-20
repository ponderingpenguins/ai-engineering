from datetime import datetime, timedelta

import pandas as pd
import pytest
from datasets import Dataset

from data import extract
from data.annotation import (LYING, NOT_LYING, capture_positions, check_interval, lying_labels, make_marks, merge,
                             read_annotations, write_annotations)

START = datetime(2026, 1, 1)


def frames(recording, idx_in_time_order):
    """One recording; `idx_in_time_order` lists the frame_idx of each frame in the order they were captured."""
    return pd.DataFrame({"recording": recording, "frame_idx": idx_in_time_order,
                         "timestamp": [START + timedelta(seconds=0.2 * i) for i in range(len(idx_in_time_order))]})


def out_of_order():
    """Like the two 20231115 recordings: frames 6-11 were captured between frames 1 and 2 but stored after them."""
    return frames("r", [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]).sample(frac=1, random_state=0).reset_index(drop=True)


# ---- capture order ---------------------------------------------------------------------------------

def test_capture_position_is_the_place_in_time_order_not_the_frame_idx():
    meta = out_of_order()
    positions = capture_positions(meta)
    by_position = meta.assign(position=positions).sort_values("position").frame_idx.tolist()
    assert by_position == [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]
    assert sorted(positions) == list(range(12))


def test_capture_positions_are_counted_per_recording():
    meta = pd.concat([frames("a", [0, 1, 2]), frames("b", [0, 1])], ignore_index=True)
    assert capture_positions(meta).tolist() == [0, 1, 2, 0, 1]


def test_equal_timestamps_are_ordered_by_frame_idx():
    meta = pd.DataFrame({"recording": "r", "frame_idx": [2, 0, 1], "timestamp": [START] * 3})
    assert capture_positions(meta).tolist() == [2, 0, 1]


# ---- intervals -------------------------------------------------------------------------------------

def test_merge_joins_overlapping_and_touching_intervals_and_sorts():
    assert merge([(10, 12), (3, 4), (5, 6), (11, 15), (20, 20)]) == ((3, 6), (10, 15), (20, 20))
    assert merge([]) == ()


@pytest.mark.parametrize("first, last", [(-1, 3), (5, 4)])
def test_invalid_intervals_are_refused(first, last):
    with pytest.raises(ValueError, match="first <= last"):
        check_interval(first, last)


def marks(intervals, reviewed=True):
    return make_marks(intervals, reviewed)


def test_file_round_trips_with_marker_rows_drafts_and_sorted_rows(tmp_path):
    path = tmp_path / "lying.csv"
    write_annotations({"b": marks([(7, 8), (1, 2), (3, 3)]), "a": marks([]), "c": marks([(0, 0)], reviewed=False),
                       "unmarked": marks([], reviewed=False)}, path)
    assert path.read_text().splitlines() == [
        "recording,first_capture_idx,last_capture_idx,reviewed",
        "a,,,true", "b,1,3,true", "b,7,8,true", "c,0,0,false"]  # a recording with nothing to save has no row
    assert read_annotations(path) == {"a": marks([]), "b": marks([(1, 3), (7, 8)]),
                                      "c": marks([(0, 0)], reviewed=False)}
    assert [p.name for p in tmp_path.iterdir()] == ["lying.csv"]  # no temporary file is left behind


def test_a_missing_file_means_nothing_was_marked(tmp_path):
    assert read_annotations(tmp_path / "none.csv") == {}


def test_an_invalid_interval_is_refused_before_anything_is_written():
    with pytest.raises(ValueError):
        make_marks([(5, 4)], reviewed=True)


def test_a_crash_while_replacing_keeps_the_old_file_and_leaves_no_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "lying.csv"
    write_annotations({"a": marks([(1, 2)])}, path)

    def crash(*_):
        raise OSError("disk gone")

    monkeypatch.setattr("data.annotation.os.replace", crash)
    with pytest.raises(OSError):
        write_annotations({"a": marks([(7, 8)])}, path)
    monkeypatch.undo()
    assert read_annotations(path) == {"a": marks([(1, 2)])}
    assert [p.name for p in tmp_path.iterdir()] == ["lying.csv"]


@pytest.mark.parametrize("text, message", [
    ("recording,first_capture_idx,last_capture_idx,reviewed\na,3,,true\n", "only one position"),
    ("recording,first_capture_idx,last_capture_idx,reviewed\na,5,4,true\n", "first <= last"),
    ("recording,first_capture_idx,last_capture_idx,reviewed\na,1,2,maybe\n", "expected true or false"),
    ("recording,first_capture_idx,last_capture_idx,reviewed\na,1,2,true\na,5,6,false\n", "disagree on reviewed"),
    ("recording,first_capture_idx,last_capture_idx\na,1,2\n", "must have the columns"),
])
def test_bad_files_are_refused(tmp_path, text, message):
    path = tmp_path / "lying.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        read_annotations(path)


# ---- the per-frame labels --------------------------------------------------------------------------

def test_labels_follow_capture_order_not_frame_idx():
    meta = out_of_order()
    labels = lying_labels(meta, {"r": marks([(2, 3)])})  # the 3rd and 4th frames captured: frame_idx 6 and 7
    lying = sorted(meta.frame_idx[[label == LYING for label in labels]])
    assert lying == [6, 7]
    assert labels.count(NOT_LYING) == 10


def test_unreviewed_recordings_get_none_and_reviewed_ones_without_lying_get_not_lying():
    meta = pd.concat([frames("reviewed", [0, 1]), frames("empty", [0, 1]), frames("unreviewed", [0, 1])],
                     ignore_index=True)
    labels = lying_labels(meta, {"reviewed": marks([(1, 1)]), "empty": marks([]), "unreviewed": marks([(0, 0)], reviewed=False)})
    assert labels == [NOT_LYING, LYING, NOT_LYING, NOT_LYING, None, None]


def test_an_interval_past_the_end_is_skipped_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        labels = lying_labels(frames("r", [0, 1, 2]), {"r": marks([(0, 0), (2, 5)])})
    assert labels == [LYING, NOT_LYING, NOT_LYING] and "skipped the interval (2, 5)" in caplog.text


def test_annotations_for_unknown_recordings_warn(caplog):
    with caplog.at_level("WARNING"):
        lying_labels(frames("r", [0]), {"r": marks([]), "gone": marks([])})
    assert "gone" in caplog.text


# ---- the join at bake time -------------------------------------------------------------------------

def test_extract_adds_the_lying_annotation_column(tmp_path, monkeypatch):
    path = tmp_path / "lying.csv"
    write_annotations({"a": marks([(1, 1)])}, path)
    monkeypatch.setattr(extract, "LYING_INTERVALS", path)
    ds = Dataset.from_pandas(pd.concat([frames("a", [0, 1, 2]), frames("b", [0, 1])], ignore_index=True))
    out = extract._add_lying_annotation(ds)
    assert out["lying_annotation"] == [NOT_LYING, LYING, NOT_LYING, None, None]


def test_extract_without_a_file_gives_none_everywhere(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(extract, "LYING_INTERVALS", tmp_path / "none.csv")
    with caplog.at_level("WARNING"):
        out = extract._add_lying_annotation(Dataset.from_pandas(frames("a", [0, 1])))
    assert out["lying_annotation"] == [None, None] and "None everywhere" in caplog.text
