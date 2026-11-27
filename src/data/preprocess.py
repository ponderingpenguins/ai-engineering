"""Select recordings and optionally clean heatmaps before building windows.

Apply the same rules to training and held-out recordings.
Each denoising step estimates its statistics within one recording.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Value, concatenate_datasets
from datasets import Sequence as ListOf
from scipy.signal import lfilter

from config.data import PreprocessConfig, SelectionConfig
from data.annotations import NO_PERSON

logger = logging.getLogger(__name__)

CONFIG_4 = "Config_4.cfg"  # Default sensor config: 256 range bins by 16 Doppler bins.


class Preprocessor(ABC):
    """A named dataset-to-dataset preprocessing step."""

    name: str

    @abstractmethod
    def __call__(self, ds: Dataset) -> Dataset:
        """Return the dataset after applying this step."""


class RowFilter(Preprocessor):
    """Keep rows selected by a boolean condition on metadata columns."""

    columns: tuple[str, ...]

    @abstractmethod
    def keep(self, meta) -> np.ndarray:
        """Return one boolean per row of the metadata table containing self.columns."""

    def __call__(self, ds: Dataset) -> Dataset:
        """Read the required metadata and select rows where keep returns True."""
        meta = ds.select_columns(list(self.columns)).to_pandas()
        mask = np.asarray(self.keep(meta), dtype=bool)
        return ds if mask.all() else ds.select(np.flatnonzero(mask))


class KeepSensorConfig(RowFilter):
    """Keep recordings made with the requested sensor configuration."""

    name = "sensor_config"
    columns = ("cfg_file",)

    def __init__(self, sensor_config: str = CONFIG_4) -> None:
        """Store the sensor configuration filename to retain."""
        self.sensor_config = sensor_config

    def keep(self, meta):
        """Match each frame's sensor configuration filename."""
        return meta.cfg_file == self.sensor_config


class DropUnknownPerson(RowFilter):
    """Remove unknown participants so they cannot cross person-wise split boundaries."""

    name = "known_person"
    columns = ("person",)

    def keep(self, meta):
        """Keep rows whose person value differs from the unknown-person marker."""
        return meta.person != NO_PERSON


class DropRecordings(RowFilter):
    """Exclude named recordings, such as recordings with an unsuitable camera view."""

    name = "exclude_recordings"
    columns = ("recording",)

    def __init__(self, recordings: Sequence[str]) -> None:
        """Store the recording names to exclude."""
        self.recordings = set(recordings)

    def keep(self, meta):
        """Reject excluded recordings and warn about names absent from the dataset."""
        missing = self.recordings - set(meta.recording)
        if missing:
            logger.warning("selection.exclude_recordings names recordings that are not in the dataset: %s", sorted(missing))
        return ~meta.recording.isin(self.recordings)


def _as_2d(heatmap) -> np.ndarray:
    """Convert a heatmap to float32, representing an empty heatmap with shape (0, 0)."""
    a = np.asarray(heatmap, dtype=np.float32)
    return a if a.ndim == 2 else a.reshape(0, 0)


class HeatmapTransform(Preprocessor):
    """Rewrite usable heatmaps one recording at a time.

    Subclasses implement transform and estimate statistics only within that
    recording. Heatmaps are clipped to zero and stored as float32 for log1p
    window inputs. Invalid frames and recordings with mismatched shapes are skipped.
    """

    columns = ("recording", "timestamp", "heatmap_ok", "num_range_bins", "num_doppler_bins")

    @abstractmethod
    def transform(self, frames: np.ndarray) -> np.ndarray:
        """Transform the usable heatmaps of one recording.

        Args:
            frames: Float32 array of shape (N, range_bins, doppler_bins), sorted by
                timestamp with ties kept in their stored order.

        Returns:
            Transformed heatmaps with the same shape.
        """

    def __call__(self, ds: Dataset) -> Dataset:
        """Transform each recording and replace only the dataset's heatmap column."""
        meta = cast(pd.DataFrame, ds.select_columns(list(self.columns)).to_pandas())
        new: dict[int, np.ndarray] = {}
        for _, rec in meta.groupby("recording", sort=False):
            rec = rec[rec.heatmap_ok.astype(bool)].sort_values("timestamp", kind="stable")
            shape = (int(rec.num_range_bins.iloc[0]), int(rec.num_doppler_bins.iloc[0])) if len(rec) else None
            if shape is None:
                continue
            rows = rec.index.to_numpy()
            frames = np.stack([np.asarray(h, dtype=np.float32) for h in ds.select(rows)["heatmap"]])
            if frames.shape[1:] != shape:
                continue
            out = np.maximum(self.transform(frames), 0.0).astype(np.float32)
            new.update(zip(rows.tolist(), out))
        if not new:
            return ds
        # Only the heatmap column is rewritten; the image and payload columns stay memory-mapped.
        feature = Features({"heatmap": ListOf(ListOf(Value("float32")))})
        heatmaps = ds.select_columns(["heatmap"]).map(
            lambda example, i: {"heatmap": new[i] if i in new else _as_2d(example["heatmap"])},
            with_indices=True,
            features=feature,
            keep_in_memory=True,  # Avoid writing cache files beside the baked dataset.
            new_fingerprint=uuid.uuid4().hex,  # Avoid hashing every heatmap in the closure.
        )
        return concatenate_datasets([ds.remove_columns("heatmap"), heatmaps], axis=1).select_columns(ds.column_names)


class RangeNormalise(HeatmapTransform):
    """Subtract a smoothed noise-floor estimate for each range bin.

    Estimate a low percentile over time and Doppler, then smooth along range
    with a running median. This uses the whole recording, including future
    frames, so it is suitable for offline processing rather than live prediction.
    """

    name = "range_normalise"

    def __init__(self, percentile: float = 10.0, smooth_bins: int = 41) -> None:
        """Store the noise-floor percentile and range smoothing width.

        Args:
            percentile: Percentile over the recording's time and Doppler axes.
            smooth_bins: Positive odd width of the running median, in range bins.
        """
        self.percentile = percentile
        self.smooth_bins = smooth_bins  # Odd running-median width along range.

    def transform(self, frames: np.ndarray) -> np.ndarray:
        """Subtract the recording's smoothed range profile from every frame."""
        floor = np.percentile(frames, self.percentile, axis=(0, 2))
        half = self.smooth_bins // 2
        padded = np.pad(floor, half, mode="edge")
        smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, self.smooth_bins), axis=1)
        return frames - smooth[None, :, None]


class TemporalBackground(HeatmapTransform):
    """Subtract each heatmap cell's previous exponential moving average.

    After initialising from the first frame, use only earlier frames for each
    background estimate. Persistent signals fade, including a person lying still.
    The transform is disabled by default.
    """

    name = "temporal_background"

    def __init__(self, tau_frames: float = 300.0) -> None:
        """Store the moving-average time constant in frames."""
        self.tau_frames = tau_frames  # At Config_4's 0.2 s frame period, 300 frames is 60 s.

    def transform(self, frames: np.ndarray) -> np.ndarray:
        """Subtract the previous background estimate from each frame."""
        alpha = 1.0 / self.tau_frames
        first = frames[0]
        ema, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], frames, axis=0, zi=((1.0 - alpha) * first)[None])
        background = np.concatenate([first[None], ema[:-1]])  # Shift EMA to the previous frame.
        return frames - background


def _apply_steps(ds: Dataset, steps: Sequence[Preprocessor]) -> Dataset:
    """Apply steps in order, logging frame counts at each step."""
    for step in steps:
        before = len(ds)
        ds = step(ds)
        logger.info("preprocess %s: %d -> %d frames", step.name, before, len(ds))
    return ds


def select_recordings(ds: Dataset, selection: SelectionConfig) -> Dataset:
    """Apply participant, recording and sensor gates without changing signal values."""
    steps: list[RowFilter] = []
    if selection.require_known_person:
        steps.append(DropUnknownPerson())
    if selection.exclude_recordings:
        steps.append(DropRecordings(selection.exclude_recordings))
    if selection.sensor_config is not None:
        steps.append(KeepSensorConfig(selection.sensor_config))
    return _apply_steps(ds, steps)


def denoise_heatmaps(ds: Dataset, denoising: PreprocessConfig) -> Dataset:
    """Apply enabled range normalisation, then temporal background subtraction.

    If neither transform is enabled, return the original dataset untouched.
    """
    steps: list[HeatmapTransform] = []
    rn, tb = denoising.range_normalise, denoising.temporal_background
    if rn.enabled:
        steps.append(RangeNormalise(rn.percentile, rn.smooth_bins))
    if tb.enabled:
        steps.append(TemporalBackground(tb.tau_frames))
    return _apply_steps(ds, steps)


def preprocess_dataset(ds: Dataset, selection: SelectionConfig, denoising: PreprocessConfig) -> Dataset:
    """Select recordings, then denoise their heatmaps before windowing."""
    return denoise_heatmaps(select_recordings(ds, selection), denoising)
