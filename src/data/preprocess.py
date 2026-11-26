"""Frame-level preprocessing, run once on the loaded dataset before it is cut into windows.

A preprocessor filters or cleans the data identically for training and testing. Steps that create
extra training examples (augmentation) are a separate stage and do not belong here.
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

from config import PreprocessConfig, SelectionConfig
from data.annotations import NO_PERSON

logger = logging.getLogger(__name__)

CONFIG_4 = "Config_4.cfg"  # 256 range bins x 16 Doppler bins; the only sensor config with pose labels


class Preprocessor(ABC):
    """One preprocessing step: takes a dataset, returns a dataset."""

    name: str

    @abstractmethod
    def __call__(self, ds: Dataset) -> Dataset:
        """Returns the preprocessed dataset."""


class RowFilter(Preprocessor):
    """A preprocessor that keeps the rows where a condition on some metadata columns holds."""

    columns: tuple[str, ...]

    @abstractmethod
    def keep(self, meta) -> np.ndarray:
        """Boolean mask over the rows of `meta`, a DataFrame with `columns`."""

    def __call__(self, ds: Dataset) -> Dataset:
        meta = ds.select_columns(list(self.columns)).to_pandas()
        mask = np.asarray(self.keep(meta), dtype=bool)
        return ds if mask.all() else ds.select(np.flatnonzero(mask))


class KeepConfig4(RowFilter):
    """Keeps only recordings made with sensor Config 4; the other configs are experimental."""

    name = "config_4"
    columns = ("cfg_file",)

    def __init__(self, sensor_config: str = CONFIG_4) -> None:
        self.sensor_config = sensor_config

    def keep(self, meta):
        return meta.cfg_file == self.sensor_config


class DropUnknownPerson(RowFilter):
    """Drops recordings without a known participant, which could hide a test participant in training."""

    name = "known_person"
    columns = ("person",)

    def keep(self, meta):
        return meta.person != NO_PERSON


class DropRecordings(RowFilter):
    """Drops the named recordings, for example ones that do not have the usual top-down camera view."""

    name = "exclude_recordings"
    columns = ("recording",)

    def __init__(self, recordings: Sequence[str]) -> None:
        self.recordings = set(recordings)

    def keep(self, meta):
        missing = self.recordings - set(meta.recording)
        if missing:
            logger.warning("selection.exclude_recordings names recordings that are not in the dataset: %s", sorted(missing))
        return ~meta.recording.isin(self.recordings)


def _as_2d(heatmap) -> np.ndarray:
    """A heatmap as a float32 2-D array; the empty heatmap (`[]`) of a bad frame becomes shape (0, 0)."""
    a = np.asarray(heatmap, dtype=np.float32)
    return a if a.ndim == 2 else a.reshape(0, 0)


class HeatmapTransform(Preprocessor):
    """A preprocessor that rewrites the heatmaps, one recording at a time.

    Subclasses only compute `transform`. Every statistic must come from the frames of the one recording
    being transformed, never from other recordings: the pipeline runs separately on each split, so a
    statistic pooled across recordings would differ between training and testing.

    The transformed heatmaps are float32 and clipped at 0 (the windowing step takes log1p of them).
    Rows without a usable heatmap (`heatmap_ok` False, or an unexpected shape) are left as they are.
    """

    columns = ("recording", "timestamp", "heatmap_ok", "num_range_bins", "num_doppler_bins")

    @abstractmethod
    def transform(self, frames: np.ndarray) -> np.ndarray:
        """Transforms the heatmaps of one recording.

        Args:
            frames: Float32 array (n, range_bins, doppler_bins), in capture order.

        Returns:
            An array of the same shape.
        """

    def __call__(self, ds: Dataset) -> Dataset:
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
            keep_in_memory=True,  # no cache files next to the baked dataset
            new_fingerprint=uuid.uuid4().hex,  # hashing the closure would hash every heatmap
        )
        return concatenate_datasets([ds.remove_columns("heatmap"), heatmaps], axis=1).select_columns(ds.column_names)


class RangeNormalise(HeatmapTransform):
    """Subtracts each range bin's noise floor, so a pixel means "above the floor at this range".

    The floor of an empty room changes with range by about as much as a person adds (roughly 100-180
    against 40-120 heatmap units). It is a low percentile over the recording's time and Doppler axes, then
    smoothed along range with a running median wider than a person, so a person who is in the room for
    most of the recording does not pull the floor up under themselves.

    Not causal: the floor comes from the whole recording, later frames included. Fine for recorded data, but it
    could not run online. A recording whose heatmaps have an unexpected shape is left unchanged.
    """

    name = "range_normalise"

    def __init__(self, percentile: float = 10.0, smooth_bins: int = 41) -> None:
        self.percentile = percentile
        self.smooth_bins = smooth_bins  # odd; wider than twice a lying person (about 17 bins of 0.0237 m)

    def transform(self, frames: np.ndarray) -> np.ndarray:
        floor = np.percentile(frames, self.percentile, axis=(0, 2))
        half = self.smooth_bins // 2
        padded = np.pad(floor, half, mode="edge")
        smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, self.smooth_bins), axis=1)
        return frames - smooth[None, :, None]


class TemporalBackground(HeatmapTransform):
    """Subtracts a slow exponential moving average of each cell, so only recent change is left.

    Causal: the background of a frame uses earlier frames only, so it also works online. Anything that stays
    put for much longer than `tau_frames` fades into the background, and that includes a person lying still,
    so this trades static clutter against a still person. It is not on by default.
    """

    name = "temporal_background"

    def __init__(self, tau_frames: float = 300.0) -> None:
        self.tau_frames = tau_frames  # frames are 0.2 s apart, so 300 is a minute

    def transform(self, frames: np.ndarray) -> np.ndarray:
        alpha = 1.0 / self.tau_frames
        first = frames[0]
        ema, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], frames, axis=0, zi=((1.0 - alpha) * first)[None])
        background = np.concatenate([first[None], ema[:-1]])  # the background before each frame
        return frames - background


PREPROCESSORS: dict[str, type[Preprocessor]] = {
    cls.name: cls for cls in (KeepConfig4, DropUnknownPerson, RangeNormalise, TemporalBackground)
}


class Pipeline:
    """Runs preprocessors in order and logs what each one removes."""

    def __init__(self, steps: Sequence[Preprocessor]) -> None:
        self.steps = list(steps)

    def __call__(self, ds: Dataset) -> Dataset:
        for step in self.steps:
            before = len(ds)
            ds = step(ds)
            logger.info("preprocess %s: %d -> %d frames", step.name, before, len(ds))
        return ds


def configured_pipeline(selection: SelectionConfig, denoise: PreprocessConfig | None = None) -> Pipeline:
    """Wire the central config to existing steps; None omits signal transforms for annotation tools."""
    steps: list[Preprocessor] = []
    if selection.require_known_person:
        steps.append(DropUnknownPerson())
    if selection.exclude_recordings:
        steps.append(DropRecordings(selection.exclude_recordings))
    if selection.sensor_config is not None:
        steps.append(KeepConfig4(selection.sensor_config))
    if denoise is not None:
        rn, tb = denoise.range_normalise, denoise.temporal_background
        if rn.enabled:
            steps.append(RangeNormalise(rn.percentile, rn.smooth_bins))
        if tb.enabled:
            steps.append(TemporalBackground(tb.tau_frames))
    return Pipeline(steps)


def build_pipeline(names: Sequence[str], exclude: Sequence[str] = ()) -> Pipeline:
    """A pipeline that first drops unknown persons and the `exclude` recordings, then runs the named preprocessors.

    Dropping recordings without a known person is not optional: they would all share one pseudo-person in the
    person-level splits. `known_person` in `names` is accepted and ignored. The `exclude` recordings are dropped
    from training and testing alike.

    Raises:
        ValueError: If a name is not in `PREPROCESSORS`.
    """
    unknown = [name for name in names if name not in PREPROCESSORS]
    if unknown:
        raise ValueError(f"unknown preprocessor {unknown}, choose from {list(PREPROCESSORS)}")
    steps: list[Preprocessor] = [DropUnknownPerson()]
    if exclude:
        steps.append(DropRecordings(exclude))
    steps += [PREPROCESSORS[name]() for name in names if name != DropUnknownPerson.name]
    return Pipeline(steps)
