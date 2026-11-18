"""
Loads the baked dataset, cuts it into windows and cross-validates a model, person by person.

Everything is configured in config/config.yaml and can be overridden from the command line.
By default there is one fold per person; set eval.folds=<k> for k-fold instead:

    python main.py model.name=baseline
    python main.py model.rnn.epochs=5 eval.folds=3 eval.collapse_lying=false

TODO:
- data preprocessing
"""

import sys
import logging
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import cast

from datasets import Dataset, load_from_disk
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from data.preprocess import build_pipeline
from data.windows import make_windows, collapse_lying
from models.factory import MODELS, build_model
from models.model import BaseModel
from utils.config import Config
from utils.splitter import GroupKFoldSplitter
from utils.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent


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

    def make_model() -> BaseModel:
        """Builds a fresh, untrained model, so every fold starts from scratch.

        Returns:
            A new model as configured by `cfg.model`.
        """
        return build_model(cfg.model)

    x, y, groups = make_windows(
        ds,
        cfg.eval.target,
        window_len=cfg.eval.window_len,
        stride=cfg.eval.stride,
        heatmaps=make_model().needs_heatmaps,
        relabel=collapse_lying if cfg.eval.collapse_lying else None,
    )
    logger.info("%d windows from %d groups, classes: %s", len(y), len(set(groups)), dict(Counter(y.tolist())))
    splitter = GroupKFoldSplitter(cfg.eval.folds, cfg.eval.seed)
    try:
        splitter.n_splits(groups)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    out_dir = REPO / cfg.eval.out_dir / f"{datetime.now():%Y%m%d-%H%M%S}_{cfg.model.name}"
    trainer = Trainer(make_model, splitter=splitter, out_dir=out_dir, config=asdict(cfg))
    trainer.cross_validate(x, y, groups)


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
            "[model.rnn.epochs=<n>] [eval.folds=<k>] [eval.collapse_lying=true|false]",
            e,
            "|".join(MODELS),
        )
        sys.exit(1)

    run(cfg)


if __name__ == "__main__":
    main()
