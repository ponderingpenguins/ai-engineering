"""
Loads the baked dataset, cuts it into windows and cross-validates a model, person by person.

Everything is configured in config/config.yaml and can be overridden from the command line.
By default there is one fold per person; set eval.folds=<k> for k-fold instead:

    python main.py model.name=baseline
    python main.py model.rnn.epochs=5 eval.folds=3 eval.collapse_lying=false

Tune first, then train the final model with the best setting (see "Evaluation workflow" in the README):

    python main.py 'tune.grid=[{hidden: 64}, {hidden: 128, lr: 0.0003}]'
    python main.py train.full_run=true

TODO:
- data preprocessing
"""

import sys
import logging
from collections import Counter
from dataclasses import asdict, fields, replace
from datetime import datetime
from pathlib import Path
from typing import cast

from datasets import Dataset, load_from_disk
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from data.preprocess import build_pipeline
from data.resample import RandomOversampler
from data.windows import collapse_lying, make_windows, window_timing
from models.factory import MODELS, build_model
from models.model import BaseModel
from utils.config import Config, ModelConfig, RNNConfig
from utils.splitter import GroupKFoldSplitter
from utils.trainer import Trainer
from utils.tuning import Tuner, apply_tuned, load_tuned, save_tuned, validate_grid

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent


def tuned_model_config(cfg: Config, task: dict) -> tuple[ModelConfig, dict]:
    """The model config with the setting saved by a tuning run applied, for `train.full_run`.

    Raises:
        ValueError: If tuning is also requested, the model is not an RNN, the tuned file is missing, or it
            was tuned for a different model or task than the current settings.
    """
    if cfg.tune.grid:
        raise ValueError("train.full_run and tune.grid cannot be combined: tune first, then run the full run")
    if cfg.model.name != "rnn":
        raise ValueError(f"train.full_run needs a tuned setting, which only exists for model.name=rnn, got {cfg.model.name!r}")
    path = REPO / cfg.tune.best_path
    try:
        tuned = load_tuned(path)
    except FileNotFoundError as e:
        raise ValueError(str(e)) from e
    return apply_tuned(cfg.model, tuned, task), tuned


def run(cfg: Config) -> None:
    """Cross-validates the configured model on the baked dataset.

    Args:
        cfg: The full run configuration.

    Raises:
        SystemExit: If the baked dataset is missing or `eval.folds` is out of range.
    """
    path = REPO / cfg.data.baked  # absolute paths pass through unchanged
    if not path.is_dir():
        logger.error("No baked dataset at %s.\nBuild it with: python -m data.extract", path)
        sys.exit(1)
    ds = build_pipeline(cfg.data.preprocess)(cast(Dataset, load_from_disk(str(path))))

    task = {key: getattr(cfg.eval, key) for key in ("target", "collapse_lying", "window_len", "label_rule")}
    resampler = None if cfg.train.oversample is None else RandomOversampler(cfg.train.oversample, cfg.eval.seed)
    model_cfg, tuned = cfg.model, None
    if cfg.train.full_run:
        try:
            model_cfg, tuned = tuned_model_config(cfg, task)
        except ValueError as e:
            logger.error("%s", e)
            sys.exit(1)
        logger.info("full run with the tuned setting %s for %d epochs", tuned["setting"], tuned["epochs"])

    def make_model() -> BaseModel:
        """Builds a fresh, untrained model, so every fold starts from scratch.

        Returns:
            A new model as configured by `model_cfg` (`cfg.model`, with the tuned setting for a full run).
        """
        return build_model(model_cfg)

    x, y, groups = make_windows(
        ds,
        cfg.eval.target,
        window_len=cfg.eval.window_len,
        stride=cfg.eval.stride,
        heatmaps=make_model().needs_heatmaps,
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
    if cfg.train.full_run:
        out_dir = REPO / cfg.eval.out_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model.name}_final"
        Trainer(make_model, out_dir=out_dir, task=task, resampler=resampler,
                config={**asdict(cfg), "window_timing": timing, "tuned": tuned}).train_final(x, y)
        return
    splitter = GroupKFoldSplitter(cfg.eval.folds, cfg.eval.seed)
    try:
        splitter.n_splits(groups)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    out_dir = REPO / cfg.eval.out_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model.name}"
    tuner = None
    if cfg.tune.grid:
        try:
            if cfg.model.name != "rnn":
                raise ValueError(f"tuning is only defined for model.name=rnn, got {cfg.model.name!r}")
            validate_grid(cfg.tune.grid, [f.name for f in fields(RNNConfig)])
        except ValueError as e:
            logger.error("%s", e)
            sys.exit(1)
        tuner = Tuner(lambda setting: build_model(replace(cfg.model, rnn=replace(cfg.model.rnn, **setting))),
                      cfg.tune.grid, resampler)
    trainer = Trainer(make_model, splitter=splitter, out_dir=out_dir, task=task, seed=cfg.eval.seed,
                      resampler=resampler, validate=cfg.train.validation, tuner=tuner,
                      config={**asdict(cfg), "window_timing": timing})
    results = trainer.cross_validate(x, y, groups)
    if tuner is not None:  # the full run (train.full_run=true) picks this up
        for target in (REPO / cfg.tune.best_path, out_dir / "tuned.yaml"):
            save_tuned(target, results["tuned"], cfg.model.name, task)
        logger.info("saved the best setting to %s", REPO / cfg.tune.best_path)


def main() -> None:
    """main function"""
    schema = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(REPO / "config" / "config.yaml")
    cli_cfg = OmegaConf.from_cli()
    try:
        # unknown keys fail the merge (ConfigKeyError), bad types fail to_object
        merged = OmegaConf.merge(schema, file_cfg, cli_cfg)
        cfg = cast(Config, OmegaConf.to_object(merged))
        if cfg.model.name not in MODELS:
            raise ValueError(f"model.name must be one of {MODELS}, got {cfg.model.name!r}")
    except (TypeError, ValueError, OmegaConfBaseException) as e:  # pylint: disable=broad-exception-raised
        logger.error(
            "Error: %s\n\nUsage: python main.py [data.baked=<path>] [model.name=%s] "
            "[model.rnn.epochs=<n>] [train.oversample=<ratio>] [train.validation=true|false] [eval.folds=<k>] [eval.collapse_lying=true|false] [tune.grid=<list of settings>] [train.full_run=true|false]",
            e,
            "|".join(MODELS),
        )
        sys.exit(1)

    run(cfg)


if __name__ == "__main__":
    main()
