"""Read, save and attach manual person, presence and lying annotations.

Capture positions follow timestamp order within each recording.
Reviewed lying intervals become frame labels; unreviewed recordings stay unlabelled.
"""

import logging
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, Value

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
PERSON_MAP = REPO / "data" / "annotations" / "person_map.csv"
PRESENCE_EVENTS = REPO / "data" / "annotations" / "presence_events.csv"
LYING_INTERVALS = REPO / "data" / "annotations" / "lying_intervals.csv"
NO_PERSON = "NA"
COLUMNS = ["recording", "first_capture_idx", "last_capture_idx", "reviewed"]
LYING, NOT_LYING = "lying", "not_lying"  # Values of the lying_annotation column.

Interval = tuple[int, int]


@dataclass(frozen=True)
class Marks:
    """Inclusive lying intervals and the review status of one recording."""

    intervals: tuple[Interval, ...] = ()
    reviewed: bool = False


Annotations = dict[str, Marks]  # Recording name -> saved marks; untouched recordings are absent.


def capture_positions(meta: pd.DataFrame) -> pd.Series:
    """Return each frame's zero-based position in its recording's capture order.

    Sort by timestamp, using frame_idx to break ties. Stored frame indices can
    differ from capture order, so annotation intervals use these positions.

    Args:
        meta: Frame table with recording, timestamp and frame_idx columns and
            a unique index.

    Returns:
        Capture positions aligned with the original index of meta.
    """
    ordered = meta.sort_values(["recording", "timestamp", "frame_idx"], kind="stable")
    return ordered.groupby("recording", sort=False).cumcount().reindex(meta.index)


def merge(intervals: Sequence[Interval]) -> tuple[Interval, ...]:
    """Return sorted intervals, joining overlapping or adjacent inclusive ranges."""
    merged: list[list[int]] = []
    for first, last in sorted(intervals):
        if merged and first <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    return tuple((a, b) for a, b in merged)


def check_interval(first: int, last: int) -> Interval:
    """Convert interval endpoints to integers and check their order.

    Raises:
        ValueError: If either endpoint is negative or last is less than first.
    """
    first, last = int(first), int(last)
    if first < 0 or last < first:
        raise ValueError(f"an interval needs 0 <= first <= last, got ({first}, {last})")
    return first, last


def make_marks(intervals: Sequence[Sequence[int]], reviewed: bool) -> Marks:
    """Build recording marks from checked and merged inclusive intervals.

    Raises:
        ValueError: If an interval is invalid.
    """
    return Marks(merge([check_interval(first, last) for first, last in intervals]), bool(reviewed))


def read_annotations(path: Path = LYING_INTERVALS) -> Annotations:
    """Read intervals and review status from the annotation CSV.

    A missing file returns an empty mapping. Empty interval endpoints record
    review status without marking any lying frames.

    Raises:
        ValueError: If columns, interval endpoints or review flags are invalid,
            or rows for one recording disagree on review status.
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
            continue  # Preserve review status for a recording with no lying intervals.
        if pd.isna(first) or pd.isna(last):
            raise ValueError(f"{path.name}: {recording} has a row with only one position")
        intervals[recording].append(check_interval(first, last))
    return {recording: make_marks(intervals[recording], reviewed[recording]) for recording in intervals}


def write_annotations(annotations: Mapping[str, Marks], path: Path = LYING_INTERVALS) -> None:
    """Save marks to a temporary CSV, then atomically replace the destination.

    Reviewed recordings without lying get one row with empty endpoints.
    Unreviewed recordings without intervals are omitted.

    Args:
        annotations: Recording names mapped to inclusive intervals and review status.
        path: Destination CSV; missing parent directories are created.
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
    """Expand reviewed intervals into lying, not_lying or missing frame labels.

    Unreviewed recordings get None. Intervals beyond the recording's final
    capture position are skipped with a warning.

    Args:
        meta: Frame table with recording, timestamp and frame_idx columns and
            a unique index.
        annotations: Recording names mapped to intervals and review status.

    Returns:
        One label per row, in the original order of meta.
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


def _replace_column(ds: Dataset, name: str, values: list[Any], dtype: str) -> Dataset:
    """Replace an annotation column with explicitly typed values, including all-null columns."""
    if name in ds.column_names:
        ds = ds.remove_columns(name)
    # Set the feature on insertion to avoid casting/copying unrelated radar columns.
    return ds.add_column(name, values, feature=Value(dtype))


def attach_person(ds: Dataset, annotations_dir: Path | None = None) -> Dataset:
    """Attach person_map.csv identities; recordings without an identity get "NA"."""
    # Preserve literal names such as "NA" rather than treating them as missing values.
    df = pd.read_csv(PERSON_MAP if annotations_dir is None else annotations_dir / "person_map.csv", keep_default_na=False)
    mapping = {r: p.strip() or NO_PERSON for r, p in zip(df.recording, df.person)}
    recs = ds.select_columns(["recording"]).to_pandas().recording
    unlisted = sorted(set(recs) - set(mapping))
    if unlisted:
        logger.warning("not in %s, person set to %s: %s", PERSON_MAP.name, NO_PERSON, unlisted)
    return _replace_column(ds, "person", recs.map(lambda r: mapping.get(r, NO_PERSON)).tolist(), "string")


def attach_presence(ds: Dataset, annotations_dir: Path | None = None) -> Dataset:
    """Attach person_present flags from presence_events.csv.

    Frames before the first enter event are absent. Presence lasts from enter
    until leave or the recording ends. Recordings with no events get None.
    """
    path = PRESENCE_EVENTS if annotations_dir is None else annotations_dir / "presence_events.csv"
    events = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["recording", "frame_idx", "event"])
    if events.empty:
        logger.warning("no events in %s, person_present is None everywhere", PRESENCE_EVENTS.name)
    meta = ds.select_columns(["recording", "frame_idx"]).to_pandas()
    present = np.full(len(meta), None, dtype=object)
    for rec, evs in events.sort_values("frame_idx").groupby("recording"):
        rows = np.flatnonzero(meta.recording == rec)
        if rows.size == 0:
            logger.warning("%s in %s but not in the dataset", rec, PRESENCE_EVENTS.name)
            continue
        state, last = False, 0
        flags = np.zeros(rows.size, dtype=bool)
        for frame, event in zip(evs.frame_idx, evs.event):
            flags[last:frame] = state
            state, last = event == "enter", int(frame)
        flags[last:] = state
        present[rows[np.argsort(meta.frame_idx.values[rows])]] = flags  # Events use frame_idx order.
    unannotated = sorted(set(meta.recording) - set(events.recording))
    if unannotated:
        logger.warning("not in %s, person_present set to None: %s", PRESENCE_EVENTS.name, unannotated)
    return _replace_column(ds, "person_present", present.tolist(), "bool")


def attach_lying_annotation(ds: Dataset, annotations_dir: Path | None = None) -> Dataset:
    """Attach reviewed lying labels from lying_intervals.csv.

    Labels are lying or not_lying; unreviewed recordings get None so that
    incomplete annotation is distinct from a reviewed recording with no lying.
    """
    annotations = read_annotations(LYING_INTERVALS if annotations_dir is None else annotations_dir / "lying_intervals.csv")
    meta = ds.select_columns(["recording", "frame_idx", "timestamp"]).to_pandas()
    labels = lying_labels(meta, annotations)
    reviewed = {rec for rec, marks in annotations.items() if marks.reviewed} & set(meta.recording)
    if not reviewed:
        logger.warning("no reviewed recordings in %s, lying_annotation is None everywhere", LYING_INTERVALS.name)
    logger.info("lying_annotation: %d recordings reviewed, %d not reviewed", len(reviewed),
                meta.recording.nunique() - len(reviewed))
    return _replace_column(ds, "lying_annotation", labels, "string")


def attach_annotations(ds: Dataset, annotations_dir: Path | None = None) -> Dataset:
    """Attach or refresh person, presence and lying labels from their source CSVs."""
    ds = attach_person(ds, annotations_dir)
    ds = attach_presence(ds, annotations_dir)
    return attach_lying_annotation(ds, annotations_dir)
