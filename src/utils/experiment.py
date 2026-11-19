"""One experiment: from a `Config` to results, with the checks that come before it.

`main.py` only reads the command line and turns errors into exit codes. Everything between a valid `Config`
and the results on disk lives here, so tests can run it on a small dataset.
"""

import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, load_from_disk

from data.preprocess import build_pipeline
from data.resample import RandomOversampler
from data.windows import collapse_lying, make_windows, window_timing
from models.factory import MODEL_CLASSES, MODELS, build_model
from models.model import BaseModel
from utils.config import Config, ModelConfig
from utils.results import FinalResult, RunResults
from utils.splitter import GroupKFoldSplitter
from utils.task import Task
from utils.trainer import Trainer
from utils.tuning import Tuner, apply_tuned, hyperparameter_names, load_tuned, save_tuned, validate_grid, with_setting

logger = logging.getLogger(__name__)


def check_config(cfg: Config) -> None:
    """Checks the rules that involve more than one setting.

    Raises:
        ValueError: If the model is unknown or an augmentation does not fit the model or the task.
    """
    if cfg.model.name not in MODELS:
        raise ValueError(f"model.name must be one of {MODELS}, got {cfg.model.name!r}")
    for name in MODELS:
        augment = getattr(getattr(cfg.model, name, None), "augment", None)
        if augment is None or not any(asdict(augment).values()):
            continue
        if cfg.model.name != name:
            raise ValueError(f"model.{name}.augment only applies to model.name={name}")
        if "config_4" not in cfg.data.preprocess:
            raise ValueError("the augmentation sizes are for Config_4, keep 'config_4' in data.preprocess")
    if cfg.model.rnn.augment.time_reverse and cfg.eval.label_rule != "any_lying":
        raise ValueError("model.rnn.augment.time_reverse needs eval.label_rule=any_lying, "
                         "because with 'last' the label depends on which frame is last")
    point_augment = cfg.model.pointnet.augment
    if (point_augment.yaw_deg or point_augment.mirror) and not cfg.eval.collapse_lying:
        raise ValueError("model.pointnet.augment.yaw_deg and mirror change which way a person lies, "
                         "so they need eval.collapse_lying=true")


def tuned_model_config(cfg: Config, task: Task, root: Path) -> tuple[ModelConfig, dict[str, Any]]:
    """The model config with the setting saved by a tuning run applied, for `train.full_run`.

    Raises:
        ValueError: If tuning is also requested, the model cannot be tuned, the tuned file is missing, or it
            was tuned for a different model or task than the current settings.
    """
    if cfg.tune.grid:
        raise ValueError("train.full_run and tune.grid cannot be combined: tune first, then run the full run")
    hyperparameter_names(cfg.model)  # raises for a model that has nothing to tune, so there is no tuned setting
    try:
        tuned = load_tuned(root / cfg.tune.best_path)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e
    return apply_tuned(cfg.model, tuned, task), tuned


def run(cfg: Config, root: Path) -> RunResults | FinalResult:
    """Cross-validates the configured model on the baked dataset, or trains the final model.

    Args:
        cfg: The full run configuration; `check_config` has accepted it.
        root: The repository folder. Relative paths in `cfg` are resolved against it.

    Returns:
        The results of the cross-validation, or the information about the final model for `train.full_run`.

    Raises:
        ValueError: If the baked dataset is missing, `eval.folds` is out of range, the tuning grid is invalid,
            or the full run has no matching tuned setting.
    """
    path = root / cfg.data.baked  # absolute paths pass through unchanged
    if not path.is_dir():
        raise ValueError(f"No baked dataset at {path}.\nBuild it with: python -m data.extract")
    ds = build_pipeline(cfg.data.preprocess, cfg.data.exclude)(cast(Dataset, load_from_disk(str(path))))

    task = Task.from_eval(cfg.eval)
    resampler = None if cfg.train.oversample is None else RandomOversampler(cfg.train.oversample, cfg.eval.seed)
    model_cfg, tuned = cfg.model, None
    if cfg.train.full_run:
        model_cfg, tuned = tuned_model_config(cfg, task, root)
        logger.info("full run with the tuned setting %s for %d epochs", tuned["setting"], tuned["epochs"])

    def make_model() -> BaseModel:
        """Builds a fresh, untrained model, so every fold starts from scratch.

        Returns:
            A new model as configured by `model_cfg` (`cfg.model`, with the tuned setting for a full run).
        """
        return build_model(model_cfg)

    input_type = MODEL_CLASSES[cfg.model.name].input_type
    x, y, groups = make_windows(
        ds,
        cfg.eval.target,
        window_len=cfg.eval.window_len,
        stride=cfg.eval.stride,
        inputs=input_type,
        max_points=cfg.eval.max_points,
        relabel=collapse_lying if cfg.eval.collapse_lying else None,
        label_rule=cfg.eval.label_rule,
        max_per_person=cfg.eval.max_windows_per_person,
        seed=cfg.eval.seed,
    )
    timing = window_timing(ds, cfg.eval.window_len, cfg.eval.stride)
    logger.info(
        "windows: %d frames = %.1f s, stride %d frames = %.1f s (%.2f s per frame), label rule %r",
        cfg.eval.window_len, timing["window_s"], cfg.eval.stride, timing["stride_s"],
        timing["frame_period_s"], cfg.eval.label_rule,
    )
    logger.info("%d windows from %d groups, classes: %s", len(y), len(set(groups)), dict(Counter(y.tolist())))
    config = {**asdict(cfg), "window_timing": timing, "input_type": input_type}
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model.name}"
    if cfg.train.full_run:
        out_dir = root / cfg.eval.out_dir / f"{stamp}_final"
        return Trainer(make_model, out_dir=out_dir, task=task, resampler=resampler,
                       config={**config, "tuned": tuned}).train_final(x, y)
    splitter = GroupKFoldSplitter(cfg.eval.folds, cfg.eval.seed)
    splitter.n_splits(groups)  # raises ValueError if eval.folds is out of range

    out_dir = root / cfg.eval.out_dir / stamp
    tuner = None
    if cfg.tune.grid:
        validate_grid(cfg.tune.grid, hyperparameter_names(cfg.model))
        tuner = Tuner(lambda setting: build_model(with_setting(cfg.model, setting)), cfg.tune.grid, resampler)
    trainer = Trainer(make_model, splitter=splitter, out_dir=out_dir, task=task, seed=cfg.eval.seed,
                      resampler=resampler, validate=cfg.train.validation, tuner=tuner, config=config)
    results = trainer.cross_validate(x, y, groups)
    if tuner is not None and results.tuned is not None:  # the full run (train.full_run=true) picks this up
        for target in (root / cfg.tune.best_path, out_dir / "tuned.yaml"):
            save_tuned(target, results.tuned, cfg.model.name, task)
        logger.info("saved the best setting to %s", root / cfg.tune.best_path)
    return results
