"""Run cv, tune and train-final from validated settings to saved outputs.

Load baked data, select and denoise frames, build windows and start the chosen workflow.
The command-line parser in main.py calls these functions.
"""

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from datasets import Dataset, load_from_disk

from data.preprocess import preprocess_dataset
from data.resample import RandomOversampler
from data.windowing import build_windows, window_timing
from models.factory import MODEL_CLASSES, build_model
from config.training import RunConfig
from evaluation.records import FinalResult, RunResults
from training.splitting import GroupKFoldSplitter
from training.cross_validation import CrossValidator
from training.fitting import FitConfig, build_augment, fit_fold
from training.tuning import Tuner, apply_tuned, hyperparameter_names, load_tuned, save_tuned, validate_grid, with_setting

logger = logging.getLogger(__name__)


def tuned_model_config(cfg: RunConfig, root: Path) -> tuple[RunConfig, dict[str, Any]]:
    """Load and apply the saved tuning winner for train-final.

    Returns:
        Updated run settings and the saved tuning metadata.

    Raises:
        ValueError: If the model cannot be tuned, the tuned file is missing,
            or saved settings conflict with the requested model or windows.
    """
    hyperparameter_names(cfg)  # Reject the baseline, which has no tunable parameters.
    try:
        tuned = load_tuned(root / cfg.tune.best_path)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e
    return apply_tuned(cfg, tuned), tuned


@dataclass
class _PreparedData:
    """Window arrays, saved configuration metadata and output naming for one run."""

    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    windowing: dict[str, Any]
    resampler: RandomOversampler | None
    config: dict[str, Any]
    stamp: str


def _prepare_data(cfg: RunConfig, root: Path) -> _PreparedData:
    """Load and preprocess baked frames, then build and describe labelled windows."""
    path = root / cfg.paths.baked_dir  # Absolute paths pass through unchanged.
    if not path.is_dir():
        raise ValueError(f"No baked dataset at {path}.\nBuild it with: python main.py bake")
    ds = preprocess_dataset(cast(Dataset, load_from_disk(str(path))), cfg.selection, cfg.denoising)

    windowing = cfg.windowing.signature()
    resampler = None if cfg.training.oversample_ratio is None else RandomOversampler(cfg.training.oversample_ratio, cfg.seed)
    input_type = MODEL_CLASSES[cfg.model.name].input_type
    x, y, groups, coverage = build_windows(
        ds, cfg.windowing, input_type, max_per_person=cfg.cv.max_windows_per_person, seed=cfg.seed,
    )
    timing = window_timing(ds, cfg.windowing.window_len, cfg.windowing.stride)
    logger.info(
        "windows: %d frames = %.1f s, stride %d frames = %.1f s (%.2f s per frame), label rule %r",
        cfg.windowing.window_len, timing["window_s"], cfg.windowing.stride, timing["stride_s"],
        timing["frame_period_s"], cfg.windowing.label_rule,
    )
    logger.info("%d windows from %d groups, classes: %s", len(y), len(set(groups)), dict(Counter(y.tolist())))
    config = {**asdict(cfg), "window_timing": timing, "input_type": input_type,
              "recordings_used": coverage.recordings_used, "recordings_unlabelled": coverage.recordings_unlabelled,
              "window_coverage": asdict(coverage)}
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model.name}"
    return _PreparedData(x, y, groups, windowing, resampler, config, stamp)


def _cross_validator(cfg: RunConfig, root: Path, data: _PreparedData,
                     tuner: Tuner | None = None) -> CrossValidator:
    """Check split feasibility and construct the configured cross-validation runner."""
    splitter = GroupKFoldSplitter(cfg.cv.folds, cfg.seed)
    splitter.n_splits(data.groups)
    return CrossValidator(
        lambda: build_model(cfg.model), splitter=splitter, out_dir=root / cfg.out_dir / data.stamp,
        windowing=data.windowing, seed=cfg.seed, resampler=data.resampler, validate=cfg.training.validation,
        tuner=tuner, config=data.config, training=cfg.training, augment=build_augment(cfg.model),
    )


def run_cv(cfg: RunConfig, root: Path) -> RunResults:
    """Cross-validate the configured model on held-out people and save the results."""
    data = _prepare_data(cfg, root)
    return _cross_validator(cfg, root, data).cross_validate(data.x, data.y, data.groups)


def run_tune(cfg: RunConfig, root: Path) -> RunResults:
    """Tune inside each outer training fold, evaluate refits and save the overall winner."""
    if not cfg.tune.grid:
        raise ValueError("tune requires a non-empty tune.grid")
    validate_grid(cfg.tune.grid, hyperparameter_names(cfg))
    # Validate candidate values as well as names before loading the baked data.
    for setting in cfg.tune.grid:
        with_setting(cfg, setting)
    data = _prepare_data(cfg, root)

    def make_candidate(setting: dict[str, Any]) -> FitConfig:
        """Construct a fresh model and training settings from one validated candidate."""
        candidate = with_setting(cfg, setting)
        return FitConfig(build_model(candidate.model), candidate.training,
                         build_augment(candidate.model), candidate.seed)

    tuner = Tuner(make_candidate, cfg.tune.grid, data.resampler)
    results = _cross_validator(cfg, root, data, tuner).cross_validate(data.x, data.y, data.groups)
    if results.tuned is not None:
        for target in (root / cfg.tune.best_path, root / cfg.out_dir / data.stamp / "tuned.yaml"):
            save_tuned(target, results.tuned, cfg.model.name, data.windowing)
        logger.info("saved the best setting to %s", root / cfg.tune.best_path)
    return results


def run_train_final(cfg: RunConfig, root: Path) -> FinalResult:
    """Train the saved tuning winner on all selected windows and save final artifacts.

    Use its chosen epoch count without a validation or test split. Return final
    training metadata; no held-out performance score is produced.
    """
    cfg, tuned = tuned_model_config(cfg, root)
    logger.info("full run with the tuned setting %s for %d epochs", tuned["setting"], tuned["epochs"])
    data = _prepare_data(cfg, root)
    model = build_model(cfg.model)
    fit = fit_fold(model, data.x, data.y, np.arange(len(data.y)), training=cfg.training,
                   augment=build_augment(cfg.model), seed=cfg.seed, resampler=data.resampler)
    final = FinalResult(config={**data.config, "tuned": tuned}, n_windows=len(data.y),
                        n_train_after_resampling=len(fit.train_idx),
                        train_classes=dict(Counter(data.y[fit.train_idx].tolist())),
                        train_seconds=fit.seconds, history=fit.history, checkpoint="final.pt")
    out_dir = root / cfg.out_dir / f"{data.stamp}_final"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "final.pt", data.windowing)
    (out_dir / "final.json").write_text(json.dumps(final.to_dict(), indent=2))
    logger.info("final model trained on %d windows in %.1f s; checkpoint %s", len(fit.train_idx), fit.seconds,
                out_dir / "final.pt")
    return final
