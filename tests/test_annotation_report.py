from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from data.annotation import make_marks
from utils.annotation_report import compare, report, runs, tracks, window_flags

START = datetime(2026, 1, 1)


def flags(n, *positions):
    out = np.zeros(n, dtype=bool)
    out[list(positions)] = True
    return out


def test_runs_are_the_stretches_of_consecutive_true_values():
    assert runs(flags(10, 0, 1, 4, 9)) == [(0, 1), (4, 4), (9, 9)]
    assert runs(flags(5)) == []


def test_a_window_is_lying_if_any_of_its_frames_is():
    # windows of 8 frames every 4: starts 0, 4, 8, ... 24. Frame 5 is in the windows starting at 0 and 4.
    assert window_flags(flags(32, 5), 8, 4).tolist() == [True, True, False, False, False, False, False]


def test_compare_counts_frames_intervals_and_windows():
    pose = flags(32, 5, 6, 25)                 # runs (5, 6) and (25, 25)
    annotation = flags(32, 4, 5, 6, 7, 20)     # runs (4, 7) and (20, 20)
    result = compare("r", pose, annotation, window_len=8, stride=4)
    assert result.pose_runs == [(5, 6), (25, 25)] and result.annotation_runs == [(4, 7), (20, 20)]
    assert result.pose_runs_found == [True, False]        # (5, 6) overlaps (4, 7); (25, 25) overlaps nothing
    assert result.annotation_runs_found == [True, False]
    assert result.frame_counts == {"both": 2, "only_pose": 1, "only_annotation": 3, "neither": 26}
    # 7 windows start at 0, 4, ... 24. Pose-label lying is in the windows starting at 0, 4, 20 and 24, the
    # annotation in those starting at 0, 4, 16 and 20.
    assert result.window_counts == {"both": 3, "only_pose": 1, "only_annotation": 1, "neither": 2}


def test_touching_but_not_overlapping_runs_are_not_found():
    result = compare("r", flags(10, 3, 4), flags(10, 5, 6), 4, 2)
    assert result.pose_runs_found == [False] and result.annotation_runs_found == [False]


def meta_of(recording, labels_in_capture_order, shuffle=True):
    """A frame table; the rows are shuffled, and frame_idx is not the capture order."""
    n = len(labels_in_capture_order)
    table = pd.DataFrame({"recording": recording, "frame_idx": np.arange(n)[::-1],  # frame_idx runs backwards
                          "timestamp": [START + timedelta(seconds=0.2 * i) for i in range(n)],
                          "label_pose": labels_in_capture_order})
    return table.sample(frac=1, random_state=1).reset_index(drop=True) if shuffle else table


def test_tracks_are_in_capture_order_and_only_for_reviewed_recordings_with_pose_labels():
    pose = ["standing", "lying_left", "lying_right", "standing", "standing", "moving"]
    meta = pd.concat([meta_of("a", pose), meta_of("unreviewed", pose), meta_of("no_pose", [None] * 6)],
                     ignore_index=True)
    annotations = {"a": make_marks([(2, 3)], True), "unreviewed": make_marks([(2, 3)], False),
                   "no_pose": make_marks([], True)}
    found = tracks(meta, annotations)
    assert list(found) == ["a"]
    assert found["a"][0].tolist() == [False, True, True, False, False, False]   # capture order, not frame_idx order
    assert found["a"][1].tolist() == [False, False, True, True, False, False]


def test_the_report_has_the_headline_the_other_levels_and_links_into_the_tool():
    pose = ["standing"] * 40
    for i in (5, 6, 30):
        pose[i] = "lying_left"
    meta = pd.concat([meta_of("2311__a", pose), meta_of("2311__waiting", pose), meta_of("2311__nopose", [None] * 40)],
                     ignore_index=True)
    annotations = {"2311__a": make_marks([(4, 7), (20, 20)], True), "2311__nopose": make_marks([], True)}
    text = report(meta, annotations, 8, 4, "http://127.0.0.1:9000/")
    assert "Compared: 1 reviewed recordings" in text
    assert "Reviewed but without pose labels, so not compared: nopose." in text
    assert "Not reviewed yet, so not compared: waiting." in text
    assert "| a | 2 | 1 | 2 | 1 |" in text                      # pose runs, found; annotation runs, found
    assert "| a | 2 | 1 | 3 | 34 |" in text                     # frames: both, only pose, only annotation, neither
    assert "[a @ 30](http://127.0.0.1:9000/#rec=2311__a&f=30), 1 frames" in text   # the pose run the annotator missed
    assert "[a @ 20](http://127.0.0.1:9000/#rec=2311__a&f=20), 1 frames" in text   # the annotation run without pose label


def test_the_report_says_when_there_is_nothing_to_compare():
    meta = meta_of("2311__a", ["standing"] * 10)
    text = report(meta, {}, 8, 4, "http://127.0.0.1:9000/")
    assert "Nothing to compare yet." in text and "Not reviewed yet" in text
