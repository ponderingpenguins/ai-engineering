"""One experiment: from a `RunConfig` to results, with the checks that come before it.

`main.py` only reads the command line and turns errors into exit codes. Everything between a valid `RunConfig`
and the results on disk lives here, so tests can run it on a small dataset.
"""

import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_from_disk

from data.preprocess import configured_pipeline
from data.resample import RandomOversampler
from data.windows import collapse_lying, make_windows, window_timing
from models.factory import MODEL_CLASSES, MODELS, build_model
from models.model import BaseModel
from config import RunConfig
from utils.results import FinalResult, RunResults
from utils.splitter import GroupKFoldSplitter
from utils.trainer import Trainer
from utils.tuning import Tuner, apply_tuned, hyperparameter_names, load_tuned, save_tuned, validate_grid, with_setting

logger = logging.getLogger(__name__)


def tuned_model_config(cfg: RunConfig, root: Path) -> tuple[RunConfig, dict[str, Any]]:
    """The model config with the setting saved by a tuning run applied, for `train-final`.

    Raises:
        ValueError: If tuning is also requested, the model cannot be tuned, the tuned file is missing, or it
            was tuned for a different model or window specification than the current settings.
    """
    hyperparameter_names(cfg)  # raises for a model that has nothing to tune, so there is no tuned setting
    try:
        tuned = load_tuned(root / cfg.tune.best_path)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e
    return apply_tuned(cfg, tuned), tuned


def recordings_by_label(ds: Dataset, target: str) -> tuple[list[str], list[str]]:
    """The recordings that have at least one label in the `target` column, and the ones that have none.

    The second list is the recordings a run on this target cannot use: for `lying_annotation` these are the
    recordings that are not reviewed yet.
    """
    meta = ds.select_columns(["recording", target]).to_pandas()
    has_labels = meta.groupby("recording")[target].apply(lambda labels: bool(labels.notna().any()))
    return sorted(has_labels[has_labels].index), sorted(has_labels[~has_labels].index)


def run(cfg: RunConfig, root: Path, mode: str = "cv") -> RunResults | FinalResult:
    """Cross-validates the configured model on the baked dataset, or trains the final model.

    Args:
        cfg: The full run configuration; `check_run_config` has accepted it.
        root: The repository folder. Relative paths in `cfg` are resolved against it.

    Returns:
        The results of the cross-validation, or the information about the final model for `train-final`.

    Raises:
        ValueError: If the baked dataset is missing, `cv.folds` is out of range, the tuning grid is invalid,
            or the full run has no matching tuned setting.
    """
    if mode not in ("cv", "tune", "train-final"):
        raise ValueError("mode must be cv, tune or train-final")
    path = root / cfg.paths.baked_dir  # absolute paths pass through unchanged
    if not path.is_dir():
        raise ValueError(f"No baked dataset at {path}.\nBuild it with: python main.py bake")
    ds = configured_pipeline(cfg.selection, cfg.denoising)(cast(Dataset, load_from_disk(str(path))))

    used, unlabelled = recordings_by_label(ds, cfg.windowing.target)
    logger.info("target %s: %d recordings used", cfg.windowing.target, len(used))
    if unlabelled:
        logger.warning("no labels in %s for %d recordings, they are not used: %s", cfg.windowing.target, len(unlabelled),
                       unlabelled)
    windowing = cfg.windowing.signature()
    resampler = None if cfg.training.oversample_ratio is None else RandomOversampler(cfg.training.oversample_ratio, cfg.seed)
    model_cfg, tuned = cfg, None
    if mode == "train-final":
        model_cfg, tuned = tuned_model_config(cfg, root)
        logger.info("full run with the tuned setting %s for %d epochs", tuned["setting"], tuned["epochs"])

    def make_model() -> BaseModel:
        """Builds a fresh, untrained model, so every fold starts from scratch.

        Returns:
            A new model as configured by `model_cfg` (`cfg.model`, with the tuned setting for a full run).
        """
        return build_model(model_cfg.model, model_cfg.training, seed=model_cfg.seed)

    input_type = MODEL_CLASSES[cfg.model.name].input_type
    x, y, groups = make_windows(
        ds,
        cfg.windowing.target,
        window_len=cfg.windowing.window_len,
        stride=cfg.windowing.stride,
        inputs=input_type,
        max_points=cfg.windowing.max_points,
        relabel=collapse_lying if cfg.windowing.collapse_lying else None,
        label_rule=cfg.windowing.label_rule,
        max_per_person=cfg.cv.max_windows_per_person,
        seed=cfg.seed,
    )
    timing = window_timing(ds, cfg.windowing.window_len, cfg.windowing.stride)
    logger.info(
        "windows: %d frames = %.1f s, stride %d frames = %.1f s (%.2f s per frame), label rule %r",
        cfg.windowing.window_len, timing["window_s"], cfg.windowing.stride, timing["stride_s"],
        timing["frame_period_s"], cfg.windowing.label_rule,
    )
    logger.info("%d windows from %d groups, classes: %s", len(y), len(set(groups)), dict(Counter(y.tolist())))
    config = {**asdict(cfg), "window_timing": timing, "input_type": input_type,
              "recordings_used": used, "recordings_unlabelled": unlabelled}
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model.name}"
    if mode == "train-final":
        out_dir = root / cfg.out_dir / f"{stamp}_final"
        return Trainer(make_model, out_dir=out_dir, windowing=windowing, resampler=resampler,
                       config={**config, "tuned": tuned}).train_final(x, y)
    splitter = GroupKFoldSplitter(cfg.cv.folds, cfg.seed)
    splitter.n_splits(groups)  # raises ValueError if cv.folds is out of range

    out_dir = root / cfg.out_dir / stamp
    tuner = None
    if mode == "tune":
        validate_grid(cfg.tune.grid, hyperparameter_names(cfg))
        def tuned_model(setting):
            candidate = with_setting(cfg, setting)
            return build_model(candidate.model, candidate.training, seed=candidate.seed)
        tuner = Tuner(tuned_model, cfg.tune.grid, resampler)
    trainer = Trainer(make_model, splitter=splitter, out_dir=out_dir, windowing=windowing, seed=cfg.seed,
                      resampler=resampler, validate=cfg.training.validation, tuner=tuner, config=config)
    results = trainer.cross_validate(x, y, groups)
    if tuner is not None and results.tuned is not None:  # the full run (train-final) picks this up
        for target in (root / cfg.tune.best_path, out_dir / "tuned.yaml"):
            save_tuned(target, results.tuned, cfg.model.name, windowing)
        logger.info("saved the best setting to %s", root / cfg.tune.best_path)
    return results
