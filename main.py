"""
Loads both datasets

TODO:
- data preprocessing
- Implement models
- train models
- evaluate models
"""

import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

# import numpy as np
# import pandas as pd
from datasets import DatasetDict, load_from_disk
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from data.extract import split_paths
from models.baseline import MajorityClass
from utils.metrics import accuracy, macro_f1
from utils.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent

# per-recording sensor config (constant within a recording)
CFG_COLUMNS = [
    "cfg_file",
    "range_resolution_m",
    "range_max_m",
    "num_range_bins",
    "num_doppler_bins",
    "doppler_resolution_m_s",
    "max_velocity_m_s",
    "num_tx",
    "num_rx",
]


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        # OmegaConf.to_container hands nested configs back as plain dicts
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)


def load_splits(path: Path) -> dict[str, DatasetDict]:
    """Room-wise and person-wise train/test splits of the baked dataset, see `python -m data.extract`"""
    splits = {}
    for name, split_path in split_paths(path).items():
        if not split_path.is_dir():
            logger.error(
                "No %s split at %s.\n"
                "Rebuild with: python -m data.extract overwrite=true",
                name,
                split_path,
            )
            sys.exit(1)
        splits[name] = cast(DatasetDict, load_from_disk(str(split_path)))
    return splits


def run_baseline(split: DatasetDict, target: str = "label_pose") -> None:
    """Fit the majority-class baseline on train, report accuracy and macro-F1 on val/test."""
    labels = {name: [y for y in ds[target] if y is not None] for name, ds in split.items()}  # unlabelled frames dropped
    model = MajorityClass().fit(labels["train"])
    logger.info("baseline predicts %r for every frame (%s)", model.label_, target)
    for name in ("val", "test"):
        if not labels.get(name):
            logger.info("%-5s: no labelled frames, skipped", name)
            continue
        y, pred = labels[name], model.predict(len(labels[name]))
        logger.info("%-5s: n=%s accuracy=%.3f macro-F1=%.3f", name, f"{len(y):,}", accuracy(y, pred), macro_f1(y, pred))


def run(cfg: Config) -> None:
    path = REPO / cfg.data.baked  # absolute paths pass through unchanged
    splits = load_splits(path)
    room_split, person_split = splits["room"], splits["person"]

    for name, split in splits.items():
        logger.info("--- baseline, %s split ---", name)
        run_baseline(split)


def main() -> None:
    """main function"""
    schema = OmegaConf.structured(Config)
    file_cfg = OmegaConf.load(REPO / "config" / "config.yaml")
    cli_cfg = OmegaConf.from_cli()
    try:
        # unknown keys fail the merge (ConfigKeyError), bad fields fail Config(**raw)
        merged = OmegaConf.merge(schema, file_cfg, cli_cfg)
        raw = cast(dict[str, Any], OmegaConf.to_container(merged, resolve=True))
        cfg = Config(**raw)
    except (TypeError, OmegaConfBaseException) as e:  # pylint: disable=broad-exception-raised
        logger.error("Error: %s\n\nUsage: python main.py [data.baked=<path>]", e)
        sys.exit(1)

    run(cfg)


if __name__ == "__main__":
    main()
