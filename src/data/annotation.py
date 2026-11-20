"""The hand-made lying annotation: lying intervals per recording, and the per-frame labels built from them.

An annotator watches the camera video and marks the stretches where the person is lying. A stretch is an
interval of capture positions (first and last frame included), see `capture_positions`. The marks are stored in
`data/processed/lying_intervals.csv`:

    recording,first_capture_idx,last_capture_idx,reviewed
    2311__20231120_144541,340,341,true
    2311__20231115_140642,,,true
    2311__20231124_152926,10,10,false

A *reviewed* recording is finished: every frame outside its intervals is not lying. A reviewed recording with
no lying has one row with empty positions. The marks of a recording that is not reviewed yet are saved too, so
no work is lost, but they do not count: that recording has no annotation, which is different from having no lying.
"""

import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
LYING_INTERVALS = REPO / "data" / "processed" / "lying_intervals.csv"
COLUMNS = ["recording", "first_capture_idx", "last_capture_idx", "reviewed"]
LYING, NOT_LYING = "lying", "not_lying"  # the labels of the `lying_annotation` column

Interval = tuple[int, int]


@dataclass(frozen=True)
class Marks:
    """The lying intervals of one recording, and whether the annotator has finished it."""

    intervals: tuple[Interval, ...] = ()
    reviewed: bool = False


Annotations = dict[str, Marks]  # recording -> its marks; recordings that were never marked are not in it


def capture_positions(meta: pd.DataFrame) -> pd.Series:
    """The place of every frame in the time order of its recording, counting from 0.

    Frames are ordered by timestamp, and by `frame_idx` when timestamps are equal. This is not always the order
    of `frame_idx`: in two recordings a block of frames was stored out of time order.

    Args:
        meta: A frame table with `recording`, `timestamp` and `frame_idx`, with a unique index.

    Returns:
        The positions, aligned to the index of `meta`.
    """
    ordered = meta.sort_values(["recording", "timestamp", "frame_idx"], kind="stable")
    return ordered.groupby("recording", sort=False).cumcount().reindex(meta.index)


def merge(intervals: Sequence[Interval]) -> tuple[Interval, ...]:
    """Sorted intervals in which overlapping or touching intervals are joined."""
    merged: list[list[int]] = []
    for first, last in sorted(intervals):
        if merged and first <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    return tuple((a, b) for a, b in merged)


def check_interval(first: int, last: int) -> Interval:
    """The interval as ints.

    Raises:
        ValueError: If a position is negative or the interval ends before it starts.
    """
    first, last = int(first), int(last)
    if first < 0 or last < first:
        raise ValueError(f"an interval needs 0 <= first <= last, got ({first}, {last})")
    return first, last


def make_marks(intervals: Sequence[Sequence[int]], reviewed: bool) -> Marks:
    """Marks from raw intervals, checked and merged.

    Raises:
        ValueError: If an interval is not valid.
    """
    return Marks(merge([check_interval(first, last) for first, last in intervals]), bool(reviewed))


def read_annotations(path: Path = LYING_INTERVALS) -> Annotations:
    """The marks of every recording in the file. A missing file means nothing was marked.

    Raises:
        ValueError: If the columns are wrong, a row has only one of its two positions, an interval is not valid,
            or the rows of one recording disagree on whether it is reviewed.
    """
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype={"recording": str, "reviewed": str})
    if list(table.columns) != COLUMNS:
        raise ValueError(f"{path.name} must have the columns {COLUMNS}, got {list(table.columns)}")
    intervals: dict[str, list[Interval]] = {}
    reviewed: dict[str, bool] = {}
    for recording, first, last, flag in zip(table.recording, table.first_capture_idx, table.last_capture_idx,
                                            table.reviewed):
        if flag not in ("true", "false"):
            raise ValueError(f"{path.name}: {recording} has reviewed={flag!r}, expected true or false")
        if reviewed.setdefault(recording, flag == "true") != (flag == "true"):
            raise ValueError(f"{path.name}: the rows of {recording} disagree on reviewed")
        intervals.setdefault(recording, [])
        if pd.isna(first) and pd.isna(last):
            continue  # the marker row of a recording without lying
        if pd.isna(first) or pd.isna(last):
            raise ValueError(f"{path.name}: {recording} has a row with only one position")
        intervals[recording].append(check_interval(first, last))
    return {recording: make_marks(intervals[recording], reviewed[recording]) for recording in intervals}


def write_annotations(annotations: Mapping[str, Marks], path: Path = LYING_INTERVALS) -> None:
    """Writes the file. The old file is replaced in one step, so a crash never leaves half a file.

    A recording that is not reviewed and has no intervals has nothing to save and gets no row.
    """
    rows: list[tuple] = []
    for recording in sorted(annotations):
        marks = annotations[recording]
        flag = "true" if marks.reviewed else "false"
        rows += [(recording, a, b, flag) for a, b in marks.intervals]
        if marks.reviewed and not marks.intervals:
            rows.append((recording, "", "", flag))
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", newline="") as file:
            pd.DataFrame(rows, columns=pd.Index(COLUMNS)).to_csv(file, index=False)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def lying_labels(meta: pd.DataFrame, annotations: Mapping[str, Marks]) -> list[str | None]:
    """The `lying_annotation` label of every frame: `lying`, `not_lying`, or None if not reviewed.

    Only reviewed recordings get labels. An interval that reaches past the end of its recording is skipped with
    a warning, because the data it was made for has changed.

    Args:
        meta: A frame table with `recording`, `timestamp` and `frame_idx`, with a unique index.
        annotations: The marks of the recordings, see `read_annotations`.

    Returns:
        One label per row of `meta`, in the order of `meta`.
    """
    positions = capture_positions(meta).to_numpy()
    recordings = meta.recording.to_numpy()
    labels = np.full(len(meta), None, dtype=object)
    for recording in np.unique(recordings):
        marks = annotations.get(recording)
        if marks is None or not marks.reviewed:
            continue
        rows = np.flatnonzero(recordings == recording)
        lying = np.zeros(rows.size, dtype=bool)
        for first, last in marks.intervals:
            if last >= rows.size:
                logger.warning("%s: skipped the interval (%d, %d), the recording has only %d frames",
                               recording, first, last, rows.size)
                continue
            lying[first : last + 1] = True
        labels[rows] = np.where(lying[positions[rows]], LYING, NOT_LYING)
    unknown = sorted(set(annotations) - set(recordings))
    if unknown:
        logger.warning("lying annotations for recordings that are not in the dataset: %s", unknown)
    return labels.tolist()
