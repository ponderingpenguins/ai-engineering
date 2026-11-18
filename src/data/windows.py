from typing import Callable, cast

import numpy as np
import pandas as pd
from datasets import Dataset

from data.extract import NO_PERSON

# Every labelled recording is Config_4: 256 range bins x 16 Doppler bins (see data/preprocess.py, which drops
# the other sensor configs).
HEATMAP_SHAPE = (256, 16)  # (range_bins, doppler_bins)

# How a window gets its label from the labels of its frames: "last" or "any_lying".
# See "How windows are labelled" in the README and `_window_label`.
LABEL_RULES = ("last", "any_lying")

# What a window's input array holds:
#   "heatmap":    (N, window_len, 1, 256, 16) log-scaled range-Doppler heatmaps.
#   "pointcloud": (N, max_points, 7) the radar points of all frames of the window, see _window_points.
#   "none":       (N, window_len) zeros, for models that ignore the input (the labels are all they need).
INPUT_TYPES = ("heatmap", "pointcloud", "none")
POINT_FEATURES = ("x", "y", "z", "doppler", "snr")  # the first five of the radar's 7 point values


def collapse_lying(y: np.ndarray) -> np.ndarray:
    """Relable all labels to either 'lying' or 'no_lying'"""
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


def make_windows(
    ds: Dataset,
    target: str = "label_pose",
    window_len: int = 16,  # frames per window
    stride: int = 8,  # frames between window starts
    shape: tuple[int, int] = HEATMAP_SHAPE,
    inputs: str = "heatmap",
    relabel: Callable[[np.ndarray], np.ndarray] | None = None,
    label_rule: str = "last",
    max_per_person: int | None = None,
    seed: int = 0,
    max_points: int = 128,
) -> tuple:
    """Cut each recording into windows of `window_len` frames.

    Returns x, y (N,) and groups (N,). x depends on `inputs` (see INPUT_TYPES): heatmaps
    (N, window_len, 1, *shape), point clouds (N, max_points, 7), or placeholder zeros. A window's label comes from its
    frames' labels by `label_rule` (see LABEL_RULES; "last" is the last frame's label). The group is the
    person (the windows of one group must not be split between train and test). Run the preprocessing
    pipeline first to drop recordings without a known person. `max_per_person` keeps at most that many windows
    per person (see `limit_per_group`), for quick test runs; None keeps all.
    """
    if inputs not in INPUT_TYPES:
        raise ValueError(f"inputs must be one of {INPUT_TYPES}, got {inputs!r}")
    if label_rule not in LABEL_RULES:
        raise ValueError(f"label_rule must be one of {LABEL_RULES}, got {label_rule!r}")
    # 1. keep only frames we can use: valid heatmap, a label, and the expected shape
    has_heatmap = np.asarray(ds["heatmap_ok"], dtype=bool)
    has_label = np.array([label is not None for label in ds[target]])
    right_shape = (np.asarray(ds["num_range_bins"]) == shape[0]) & (np.asarray(ds["num_doppler_bins"]) == shape[1])
    keep = has_heatmap & has_label & right_shape
    # Use capture order without changing the frame_idx used by hand annotations.
    meta_all = cast(pd.DataFrame, ds.select_columns(
        ["recording", "person", "frame_idx", "timestamp", target]
    ).to_pandas())
    chronological = meta_all.sort_values(["recording", "timestamp", "frame_idx"], kind="stable")
    meta_all["capture_idx"] = chronological.groupby("recording", sort=False).cumcount()
    frames_ds = ds.select(np.flatnonzero(keep))
    meta = meta_all.loc[keep].reset_index(drop=True)

    # 2. slide a window over each recording, and remember which frames it covers
    frames_per_window = []  # for each window: positions of its frames in frames_ds
    window_labels = []
    window_groups = []
    for _, rec in meta.groupby("recording", sort=False):
        rec = rec.sort_values("capture_idx")
        positions = rec.index.to_numpy()
        frame_nums = rec.capture_idx.to_numpy()
        labels = rec[target].to_numpy()
        person = rec.person.iloc[0]
        if person == NO_PERSON:
            raise ValueError(f"recording {rec.recording.iloc[0]} has no known person; run the preprocessing pipeline first")
        group = str(person)

        for start in range(0, len(rec) - window_len + 1, stride):
            last = start + window_len - 1
            has_gap = frame_nums[last] - frame_nums[start] != window_len - 1
            if has_gap:  # frames were dropped in between, so skip this window
                continue
            frames_per_window.append(positions[start : last + 1])
            window_labels.append(_window_label(labels[start : last + 1], label_rule))
            window_groups.append(group)

    window_frames = np.asarray(frames_per_window, dtype=int).reshape(-1, window_len)
    y = np.asarray(window_labels, dtype=str)
    groups = np.asarray(window_groups, dtype=str)
    if relabel is not None:
        y = relabel(y)
    if max_per_person is not None:
        keep_windows = limit_per_group(y, groups, max_per_person, seed)
        window_frames, y, groups = window_frames[keep_windows], y[keep_windows], groups[keep_windows]

    # 3. load the inputs and gather them into windows
    if inputs == "none":
        return np.zeros((len(window_frames), window_len), dtype=np.float32), y, groups
    if inputs == "pointcloud":
        clouds = [np.asarray(c, dtype=np.float32).reshape(-1, 7) for c in frames_ds["pointcloud"]]
        points = [_window_points(clouds, frames, max_points) for frames in window_frames]
        return (np.stack(points) if points else np.zeros((0, max_points, 7), dtype=np.float32)), y, groups
    if len(window_frames) == 0:
        return np.zeros((0, window_len, 1, *shape), dtype=np.float32), y, groups
    heatmap_batches = frames_ds.select_columns(["heatmap"]).iter(256)
    all_frames = np.concatenate(
        [np.stack([_prep(h, shape) for h in cast(dict, batch)["heatmap"]]) for batch in heatmap_batches]
    )
    return all_frames[window_frames], y, groups
