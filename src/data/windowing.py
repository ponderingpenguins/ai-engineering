"""Filter frames, slice labelled windows and build model inputs with coverage counts."""

import logging
from dataclasses import dataclass, field
from typing import Callable, cast

import numpy as np
import pandas as pd
from datasets import Dataset

from config import WindowConfig
from data.annotations import NO_PERSON, capture_positions

logger = logging.getLogger(__name__)

# Every labelled recording is Config_4: 256 range bins x 16 Doppler bins (see data/preprocess.py, which drops
# the other sensor configs).
HEATMAP_SHAPE = (256, 16)  # (range_bins, doppler_bins)

# How a window gets its label from the labels of its frames: "last" or "any_lying".
# See "How windows are labelled" in docs/modeling.md and `_window_label`.
LABEL_RULES = ("last", "any_lying")

# What a window's input array holds:
#   "heatmap":    (N, window_len, 1, 256, 16) log-scaled range-Doppler heatmaps.
#   "pointcloud": (N, max_points, 7) the radar points of all frames of the window, see _window_points.
#   "none":       (N, window_len) zeros, for models that ignore the input (the labels are all they need).
INPUT_TYPES = ("heatmap", "pointcloud", "none")
POINT_FEATURES = ("x", "y", "z", "doppler", "snr")  # the first five of the radar's 7 point values


def collapse_lying(y: np.ndarray) -> np.ndarray:
    """Collapse labels to either 'lying' or 'not_lying'"""
    return np.where(np.char.startswith(y, "lying"), "lying", "not_lying")


def _window_label(frame_labels: np.ndarray, rule: str) -> str:
    """The label of one window, given the labels of its frames in time order."""
    if rule == "any_lying":
        lying = [label for label in frame_labels if str(label).startswith("lying")]
        if lying:
            return lying[-1]
    return frame_labels[-1]


def window_timing(ds: Dataset, window_len: int, stride: int) -> dict[str, float]:
    """How long windows are in seconds, from the capture timestamps of `ds`.

    The frame period is the median time between consecutive frames of a recording. Gaps longer than a
    second (dropped frames, pauses) are ignored.
    """
    meta = ds.select_columns(["recording", "timestamp"]).to_pandas()
    gaps = meta.sort_values(["recording", "timestamp"]).groupby("recording").timestamp.diff().dt.total_seconds()
    period = float(gaps[gaps <= 1.0].median())
    return {"frame_period_s": period, "window_s": window_len * period, "stride_s": stride * period}


def limit_per_group(y: np.ndarray, groups: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Indices of at most `n` windows per group, for quick test runs.

    Every group keeps its windows of the rarer classes first (they are what makes a run meaningful) and
    fills up with random windows of the most common class. If the rarer classes alone exceed `n`, a random
    `n` of all of the group's windows are kept. The result is sorted, so the window order is unchanged.
    """
    rng = np.random.default_rng(seed)
    keep = []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        if len(idx) > n:
            labels, counts = np.unique(y[idx], return_counts=True)
            rare = idx[y[idx] != labels[counts.argmax()]]
            common = idx[y[idx] == labels[counts.argmax()]]
            if len(rare) >= n:
                idx = rng.choice(idx, n, replace=False)
            else:
                idx = np.concatenate([rare, rng.choice(common, n - len(rare), replace=False)])
        keep.append(idx)
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)


def _window_points(clouds: list[np.ndarray], frames: np.ndarray, max_points: int) -> np.ndarray:
    """All radar points of one window as a fixed-size array of shape (max_points, 7).

    Columns are the five point features (see POINT_FEATURES), the time of the point's frame inside the window
    (0 for the first frame, 1 for the last) and a validity flag (1 for a real point, 0 for padding). The
    radar returns 0 to 16 points per frame, so a window of 16 frames has about 40 on average (a few have
    none). If a window has more than `max_points`, the strongest by SNR are kept.
    """
    span = max(len(frames) - 1, 1)
    rows = [
        np.column_stack([clouds[pos][:, :5], np.full(len(clouds[pos]), t / span, dtype=np.float32)])
        for t, pos in enumerate(frames)
        if len(clouds[pos])
    ]
    out = np.zeros((max_points, 7), dtype=np.float32)
    if rows:
        points = np.concatenate(rows)
        if len(points) > max_points:
            keep = np.sort(np.argsort(-points[:, 4], kind="stable")[:max_points])  # strongest, original order
            points = points[keep]
        out[: len(points), :6] = points
        out[: len(points), 6] = 1.0
    return out


def _prep(h, shape: tuple[int, int]) -> np.ndarray:
    """Make sure the shapes match"""
    a = np.log1p(np.asarray(h, dtype=np.float32))
    if a.shape != shape:
        raise ValueError(f"heatmap is {a.shape}, expected {shape}")
    return a[None]


@dataclass
class WindowCoverage:
    """Coverage from selected frames to final windows.

    Frame drops are exclusive: missing labels first, then bad heatmaps, then
    wrong sensor shapes. Recording lists describe target-label availability
    before frame filtering; windows_kept includes any per-person cap.
    """

    recordings_used: list[str] = field(default_factory=list)
    recordings_unlabelled: list[str] = field(default_factory=list)
    frames_total: int = 0
    frames_unlabelled: int = 0
    frames_bad_heatmap: int = 0
    frames_wrong_shape: int = 0
    frames_kept: int = 0
    windows_gap_skipped: int = 0
    windows_kept: int = 0

    def log_summary(self, target: str) -> None:
        """Log the frame/window funnel and name recordings awaiting target labels."""
        logger.info(
            "target %s coverage: %d recordings used, %d unlabelled; "
            "frames %d total -> %d labelled (-%d unlabelled) -> %d heatmap OK (-%d bad) "
            "-> %d kept (-%d wrong shape); windows %d kept, %d gap-spanning skipped",
            target, len(self.recordings_used), len(self.recordings_unlabelled),
            self.frames_total, self.frames_total - self.frames_unlabelled, self.frames_unlabelled,
            self.frames_total - self.frames_unlabelled - self.frames_bad_heatmap, self.frames_bad_heatmap,
            self.frames_kept, self.frames_wrong_shape, self.windows_kept, self.windows_gap_skipped,
        )
        if self.recordings_unlabelled:
            logger.warning(
                "no labels in %s for %d recordings, they are not used: %s; source frames remain on disk",
                target, len(self.recordings_unlabelled), self.recordings_unlabelled,
            )


def filter_valid_frames(
    ds: Dataset, target: str, shape: tuple[int, int] = HEATMAP_SHAPE,
) -> tuple[Dataset, pd.DataFrame, WindowCoverage]:
    """Keep labelled frames with usable heatmaps of the expected sensor shape.

    Capture positions are assigned before filtering, so windows can detect holes
    left by rejected frames without changing annotation frame indices.
    """
    meta_all = cast(pd.DataFrame, ds.select_columns([
        "recording", "person", "frame_idx", "timestamp", target,
        "heatmap_ok", "num_range_bins", "num_doppler_bins",
    ]).to_pandas())
    meta_all["capture_idx"] = capture_positions(meta_all)
    has_label = meta_all[target].notna().to_numpy()
    has_heatmap = meta_all.heatmap_ok.to_numpy(dtype=bool)
    right_shape = ((meta_all.num_range_bins == shape[0]) & (meta_all.num_doppler_bins == shape[1])).to_numpy()
    keep = has_label & has_heatmap & right_shape
    labelled = meta_all.groupby("recording")[target].count() > 0
    coverage = WindowCoverage(
        recordings_used=sorted(labelled[labelled].index.tolist()),
        recordings_unlabelled=sorted(labelled[~labelled].index.tolist()),
        frames_total=len(meta_all),
        frames_unlabelled=int((~has_label).sum()),
        frames_bad_heatmap=int((has_label & ~has_heatmap).sum()),
        frames_wrong_shape=int((has_label & has_heatmap & ~right_shape).sum()),
        frames_kept=int(keep.sum()),
    )
    return ds.select(np.flatnonzero(keep)), meta_all.loc[keep].reset_index(drop=True), coverage


def slice_window_indices(
    meta: pd.DataFrame, target: str, window_len: int, stride: int, label_rule: str, collapse: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Slice each recording in capture order, skipping windows over rejected frames."""
    if window_len < 1 or stride < 1:
        raise ValueError("window_len and stride must be positive")
    if label_rule not in LABEL_RULES:
        raise ValueError(f"label_rule must be one of {LABEL_RULES}, got {label_rule!r}")
    frames_per_window: list[np.ndarray] = []
    window_labels: list[str] = []
    window_groups: list[str] = []
    n_gap_skipped = 0
    for recording, rec in meta.groupby("recording", sort=False):
        rec = rec.sort_values("capture_idx")
        if (rec.person == NO_PERSON).any():
            raise ValueError(f"recording {recording} has no known person; run the preprocessing pipeline first")
        positions = rec.index.to_numpy()
        capture_idx = rec.capture_idx.to_numpy()
        labels = rec[target].to_numpy()
        group = str(rec.person.iloc[0])
        for start in range(0, len(rec) - window_len + 1, stride):
            last = start + window_len - 1
            if capture_idx[last] - capture_idx[start] != window_len - 1:
                n_gap_skipped += 1
                continue
            frames_per_window.append(positions[start : last + 1])
            window_labels.append(_window_label(labels[start : last + 1], label_rule))
            window_groups.append(group)
    window_frames = np.asarray(frames_per_window, dtype=int).reshape(-1, window_len)
    y = np.asarray(window_labels, dtype=str)
    if collapse:
        y = collapse_lying(y)
    return window_frames, y, np.asarray(window_groups, dtype=str), n_gap_skipped


def _build_heatmap_inputs(frames_ds: Dataset, window_frames: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if len(window_frames) == 0:
        return np.zeros((0, window_frames.shape[1], 1, *shape), dtype=np.float32)
    heatmap_batches = frames_ds.select_columns(["heatmap"]).iter(256)
    all_frames = np.concatenate(
        [np.stack([_prep(h, shape) for h in cast(dict, batch)["heatmap"]]) for batch in heatmap_batches]
    )
    return all_frames[window_frames]


def _build_pointcloud_inputs(frames_ds: Dataset, window_frames: np.ndarray, max_points: int) -> np.ndarray:
    if len(window_frames) == 0:
        return np.zeros((0, max_points, 7), dtype=np.float32)
    clouds = [np.asarray(c, dtype=np.float32).reshape(-1, 7) for c in frames_ds["pointcloud"]]
    return np.stack([_window_points(clouds, frames, max_points) for frames in window_frames])


def _build_empty_inputs(window_frames: np.ndarray) -> np.ndarray:
    return np.zeros(window_frames.shape, dtype=np.float32)


def build_window_inputs(
    frames_ds: Dataset, window_frames: np.ndarray, input_type: str,
    shape: tuple[int, int] = HEATMAP_SHAPE, max_points: int = 128,
) -> np.ndarray:
    """Build heatmap, point-cloud or placeholder inputs for the same window indices."""
    if input_type == "heatmap":
        return _build_heatmap_inputs(frames_ds, window_frames, shape)
    if input_type == "pointcloud":
        return _build_pointcloud_inputs(frames_ds, window_frames, max_points)
    if input_type == "none":
        return _build_empty_inputs(window_frames)
    raise ValueError(f"inputs must be one of {INPUT_TYPES}, got {input_type!r}")


def _build_windows(
    ds: Dataset, windowing: WindowConfig, input_type: str, *,
    max_per_person: int | None = None, seed: int = 0, shape: tuple[int, int] = HEATMAP_SHAPE,
    relabel: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WindowCoverage]:
    """Shared orchestration for configured runs and explicit make_windows calls."""
    if input_type not in INPUT_TYPES:
        raise ValueError(f"inputs must be one of {INPUT_TYPES}, got {input_type!r}")
    frames_ds, meta, coverage = filter_valid_frames(ds, windowing.target, shape)
    window_frames, y, groups, coverage.windows_gap_skipped = slice_window_indices(
        meta, windowing.target, windowing.window_len, windowing.stride, windowing.label_rule,
        collapse=windowing.collapse_lying,
    )
    if relabel is not None:
        y = relabel(y)
    if max_per_person is not None:
        keep = limit_per_group(y, groups, max_per_person, seed)
        window_frames, y, groups = window_frames[keep], y[keep], groups[keep]
    x = build_window_inputs(frames_ds, window_frames, input_type, shape, windowing.max_points)
    coverage.windows_kept = len(y)
    coverage.log_summary(windowing.target)
    return x, y, groups, coverage


def make_windows(
    ds: Dataset,
    target: str = "label_pose",
    window_len: int = 16,
    stride: int = 8,
    shape: tuple[int, int] = HEATMAP_SHAPE,
    inputs: str = "heatmap",
    relabel: Callable[[np.ndarray], np.ndarray] | None = None,
    label_rule: str = "last",
    max_per_person: int | None = None,
    seed: int = 0,
    max_points: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build windows with explicit settings; return inputs, labels and person groups.

    Heatmaps have shape (N, window_len, 1, *shape), point clouds (N, max_points, 7),
    and placeholder inputs (N, window_len). Run recording selection first.
    """
    windowing = WindowConfig(target=target, collapse_lying=False, window_len=window_len,
                             stride=stride, label_rule=label_rule, max_points=max_points)
    x, y, groups, _ = _build_windows(ds, windowing, inputs, max_per_person=max_per_person,
                                     seed=seed, shape=shape, relabel=relabel)
    return x, y, groups


def build_windows(
    ds: Dataset, windowing: WindowConfig, input_type: str, *, max_per_person: int | None = None, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WindowCoverage]:
    """Build windows from config and return their frame/window coverage summary."""
    return _build_windows(ds, windowing, input_type, max_per_person=max_per_person, seed=seed)
