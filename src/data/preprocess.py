"""Frame-level preprocessing, run once on the loaded dataset before it is cut into windows.

A preprocessor filters or cleans the data identically for training and testing. Steps that create
extra training examples (augmentation) are a separate stage and do not belong here.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from datasets import Dataset

from data.extract import NO_PERSON

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

    def keep(self, meta):
        return meta.cfg_file == CONFIG_4


class DropUnknownPerson(RowFilter):
    """Drops recordings without a known participant, which could hide a test participant in training."""

    name = "known_person"
    columns = ("person",)

    def keep(self, meta):
        return meta.person != NO_PERSON


PREPROCESSORS: dict[str, type[Preprocessor]] = {cls.name: cls for cls in (KeepConfig4, DropUnknownPerson)}


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


def build_pipeline(names: Sequence[str]) -> Pipeline:
    """A pipeline of the named preprocessors, in the given order.

    Raises:
        ValueError: If a name is not in `PREPROCESSORS`.
    """
    unknown = [name for name in names if name not in PREPROCESSORS]
    if unknown:
        raise ValueError(f"unknown preprocessor {unknown}, choose from {list(PREPROCESSORS)}")
    return Pipeline([PREPROCESSORS[name]() for name in names])
