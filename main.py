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

from datasets import DatasetDict, load_from_disk
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from data.extract import split_paths
from models.baseline import MajorityClass
from models.model import BaseModel
from models.rnn import RNN
from utils.config import DataConfig
from data.windows import make_windows, collapse_lying
from utils.trainer import Trainer

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

def run_model(model:BaseModel, split:DatasetDict, heatmaps:bool, target:str = "label_pose") -> dict:
    data = {n: make_windows(ds, target, heatmaps=heatmaps, relabel=collapse_lying) for n, ds in split.items()}
    evals = {n: d for n, d in data.items() if n != "train"}
    return Trainer(model).run(data["train"], evals)

def run(cfg: Config) -> None:
    path = REPO / cfg.data.baked  # absolute paths pass through unchanged
    splits = load_splits(path)
    room_split, person_split = splits["room"], splits["person"]
    model = RNN()

    for name, split in splits.items():
        logger.info("--- baseline, %s split ---", name)
        # run_model(MajorityClass(), split, heatmaps=False)
        run_model(model, split, heatmaps=True)


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
