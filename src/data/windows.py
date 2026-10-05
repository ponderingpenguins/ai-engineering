from typing import Callable, cast

import numpy as np
import pandas as pd
from datasets import Dataset

# Every labelled recording is Config_4: 256 range bins x 16 Doppler bins. Other sensor
# configs (64/128 range bins) carry no pose labels, so they are dropped.
HEATMAP_SHAPE = (256, 16)  # (range_bins, doppler_bins)


def collapse_lying(y: np.ndarray) -> np.ndarray:
    """Relable all labels to either 'lying' or 'no_lying'"""
    return np.where(np.char.startswith(y, "lying"), "lying", "not_lying")


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
) -> tuple[np.ndarray, np.ndarray]:
    """Cut each recording into windows of `window_len` frames.

    Returns x (N, window_len, 1, *shape) and y (N,). A window gets the label of its last frame.
    """
    # 1. keep only frames we can use: valid heatmap, a label, and the expected shape
    has_heatmap = np.asarray(ds["heatmap_ok"], dtype=bool)
    has_label = np.array([label is not None for label in ds[target]])
    right_shape = (np.asarray(ds["num_range_bins"]) == shape[0]) & (np.asarray(ds["num_doppler_bins"]) == shape[1])
    frames_ds = ds.select(np.flatnonzero(has_heatmap & has_label & right_shape))
    meta = cast(pd.DataFrame, frames_ds.select_columns(["recording", "frame_idx", target]).to_pandas())

    # 2. slide a window over each recording, and remember which frames it covers
    frames_per_window = []  # for each window: positions of its frames in frames_ds
    window_labels = []
    for _, rec in meta.groupby("recording", sort=False):
        rec = rec.sort_values("frame_idx")
        positions = rec.index.to_numpy()
        frame_nums = rec.frame_idx.to_numpy()
        labels = rec[target].to_numpy()

        for start in range(0, len(rec) - window_len + 1, stride):
            last = start + window_len - 1
            has_gap = frame_nums[last] - frame_nums[start] != window_len - 1
            if has_gap:  # frames were dropped in between, so skip this window
                continue
            frames_per_window.append(positions[start : last + 1])
            window_labels.append(labels[last])

    window_frames = np.asarray(frames_per_window, dtype=int).reshape(-1, window_len)
    y = np.asarray(window_labels, dtype=str)
    if relabel is not None:
        y = relabel(y)

    # 3. load the heatmaps and gather them into windows
    if len(window_frames) == 0:
        return np.zeros((0, window_len, 1, *shape), dtype=np.float32), y
    if not heatmaps:
        return np.zeros((len(window_frames), window_len), dtype=np.float32), y
    heatmap_batches = frames_ds.select_columns(["heatmap"]).iter(256)
    all_frames = np.concatenate(
        [np.stack([_prep(h, shape) for h in cast(dict, batch)["heatmap"]]) for batch in heatmap_batches]
    )
    return all_frames[window_frames], y
