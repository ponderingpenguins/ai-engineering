from typing import Callable, cast

import numpy as np
import pandas as pd
from datasets import Dataset

# Every labelled recording is Config_4: 256 range bins x 16 Doppler bins (see data/preprocess.py, which drops
# the other sensor configs).
HEATMAP_SHAPE = (256, 16)  # (range_bins, doppler_bins)

# How a window gets its label from the labels of its frames: "last" or "any_lying".
# See "How windows are labelled" in the README and `_window_label`.
LABEL_RULES = ("last", "any_lying")


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
    heatmaps: bool = True,
    relabel: Callable[[np.ndarray], np.ndarray] | None = None,
    return_metadata: bool = False,
    label_rule: str = "last",
) -> tuple:
    """Cut each recording into windows of `window_len` frames.

    Returns x (N, window_len, 1, *shape), y (N,) and groups (N,). A window's label comes from its
    frames' labels by `label_rule` (see LABEL_RULES; "last" is the last frame's label). The group is the person (the windows of one group must not be split between train and
    test). Run the preprocessing pipeline first to drop recordings without a known person.

    With return_metadata=True, also returns one metadata dict per window, including
    ordered original frame IDs and capture timestamps for investigating predictions.
    """
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
    window_metadata = []
    for recording, rec in meta.groupby("recording", sort=False):
        rec = rec.sort_values("capture_idx")
        positions = rec.index.to_numpy()
        frame_nums = rec.capture_idx.to_numpy()
        labels = rec[target].to_numpy()
        person = rec.person.iloc[0]
        group = str(person)

        for start in range(0, len(rec) - window_len + 1, stride):
            last = start + window_len - 1
            has_gap = frame_nums[last] - frame_nums[start] != window_len - 1
            if has_gap:  # frames were dropped in between, so skip this window
                continue
            frames_per_window.append(positions[start : last + 1])
            window_labels.append(_window_label(labels[start : last + 1], label_rule))
            window_groups.append(group)
            if return_metadata:
                part = rec.iloc[start : last + 1]
                window_metadata.append({
                    "recording": str(recording), "person": str(person), "group": group,
                    "frame_ids": part.frame_idx.astype(int).tolist(),
                    "start_timestamp": part.timestamp.iloc[0].isoformat(),
                    "end_timestamp": part.timestamp.iloc[-1].isoformat(),
                    "capture_start": int(frame_nums[start]), "capture_end": int(frame_nums[last]),
                })

    window_frames = np.asarray(frames_per_window, dtype=int).reshape(-1, window_len)
    y = np.asarray(window_labels, dtype=str)
    groups = np.asarray(window_groups, dtype=str)
    if relabel is not None:
        y = relabel(y)

    def result(x):
        base = (x, y, groups)
        return (*base, window_metadata) if return_metadata else base

    # 3. load the heatmaps and gather them into windows
    if len(window_frames) == 0:
        return result(np.zeros((0, window_len, 1, *shape), dtype=np.float32))
    if not heatmaps:
        return result(np.zeros((len(window_frames), window_len), dtype=np.float32))
    heatmap_batches = frames_ds.select_columns(["heatmap"]).iter(256)
    all_frames = np.concatenate(
        [np.stack([_prep(h, shape) for h in cast(dict, batch)["heatmap"]]) for batch in heatmap_batches]
    )
    return result(all_frames[window_frames])
