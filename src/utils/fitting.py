"""Fits one model on the windows of one fold. The outer loop and the tuning loop both use it."""

import logging
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from data.resample import RandomOversampler
from models.model import BaseModel, History

logger = logging.getLogger(__name__)


@dataclass
class FoldFit:
    """What `fit_fold` did."""

    history: History | None  # None for models that do not train iteratively
    train_idx: np.ndarray  # the windows the model was trained on, after resampling
    seconds: float


def fit_fold(
    model: BaseModel,
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    *,
    resampler: RandomOversampler | None = None,
    val_weight_labels: np.ndarray | None = None,
    log_as: str | None = None,
) -> FoldFit:
    """Trains `model` on the windows `fit_idx`, with `val_idx` as validation data if given.

    The resampler repeats training windows only. The validation windows keep their real class balance, and
    the test windows are never passed in.

    Args:
        model: A fresh, untrained model.
        x, y: All windows and their labels.
        fit_idx: Indices of the training windows, before resampling.
        val_idx: Indices of the validation windows, or None to train without validation.
        resampler: Oversamples the training windows, or None to use them as they are.
        val_weight_labels: Labels whose class frequencies weight the validation loss, see `BaseModel.fit`.
        log_as: A name for the log line about oversampling (for example "fold 2/4"). None logs nothing.

    Returns:
        The loss history, the training windows used and the training time.
    """
    train_idx = fit_idx
    if resampler is not None:
        train_idx = fit_idx[resampler.indices(y[fit_idx])]
        if log_as:
            logger.info("%s: oversampled training windows %s -> %s", log_as,
                        dict(Counter(y[fit_idx].tolist())), dict(Counter(y[train_idx].tolist())))
    started = perf_counter()
    history = model.fit(x[train_idx], y[train_idx], val=None if val_idx is None else (x[val_idx], y[val_idx]),
                        val_weight_labels=val_weight_labels)
    return FoldFit(history, train_idx, perf_counter() - started)
