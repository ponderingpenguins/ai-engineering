import os
import sys
import glob
import inspect
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from utils.config import DataConfig

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent

try:
    from loguru import logger as loguru_logger
    from vt_radar.data.dataset import TLVDataset
    from vt_radar.data.rangedoppler import RangeDoppler
    from vt_radar.config.ti_config import ConfigFile
    loguru_logger.remove()  # the loader is chatty; silence it
except ModuleNotFoundError:
    raise SystemExit(
        "Did you forget to activate your environment?\n\n"
        "source .venv/bin/activate"
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self) -> None:
        # OmegaConf.to_container hands nested configs back as plain dicts
        if isinstance(self.data, dict):
            self.data = DataConfig(**self.data)


def load_cfg(root: str, rec: str) -> tuple[ConfigFile, str]:
    '''Parse the .cfg that shipped with a recording.'''
    d = rec_dir(root, rec)
    cfgs = sorted(glob.glob(os.path.join(d, "*.cfg")))
    if not cfgs:
        raise FileNotFoundError(f"no .cfg in {d}")
    c = ConfigFile()
    c.parseCfg(cfgs[0])
    return c, os.path.basename(cfgs[0])


def rec_dir(root: str, rec: str) -> str:
    '''Directory for a recording name, whichever room it is in.'''
    hits = glob.glob(os.path.join(root, "data", "*", rec))
    if not hits:
        raise FileNotFoundError(f"{rec} is not in this folder (another zip part?)")
    return hits[0]


def cfg_summary(root: str, rec: str) -> dict[str, Any]:
    c, name = load_cfg(root, rec)
    s = c.sensor()
    return {
        "recording": rec,
        "cfg": name,
        "range_res_m": round(s.range_resolution, 4),
        "range_max_m": round(s.range_max, 2),
        "range_bins": s.num_range_bins,
        "doppler_bins": s.num_doppler_bins,
        "doppler_res_m_s": round(s.doppler_resolution, 4),
        "max_velocity_m_s": round(s.max_velocity, 3),
        "tx": s.num_tx_antennas,
        "rx": s.num_rx_antennas,
    }


def run(cfg: Config) -> None:
    root = str(REPO / cfg.data.root)  # absolute paths pass through unchanged
    if not os.path.isdir(os.path.join(root, "data")):
        logger.error(
            "No data/ folder under %s.\n"
            "Set data.root in config/config.yaml to the folder that contains data/, recorder/ and previews/.",
            root,
        )
        sys.exit(1)

    logger.info("bundle  : %s", root)
    logger.info("python  : %s", sys.executable)
    logger.info("reader  : %s", os.path.dirname(inspect.getfile(TLVDataset)))

    man = pd.read_csv(os.path.join(root, "MANIFEST.csv"))
    shapes = pd.read_csv(os.path.join(root, "HEATMAP_SHAPES.csv"))
    man = man.merge(shapes[["rec", "heatmap", "cfg"]], left_on="dataset", right_on="rec", how="left")

    available: set[str] = {os.path.basename(p) for p in glob.glob(os.path.join(root, "data", "*", "*"))}
    man["present"] = man.dataset.isin(available)

    logger.info("%d of %d recordings present in this folder", man.present.sum(), len(man))
    logger.info("%s frames, %.0f MB", f"{man.loc[man.present, 'frames'].sum():,}", man.loc[man.present, "size_mb"].sum())
    logger.info("\n%s", man.loc[man.present, ["dataset", "room", "frames", "size_mb", "heatmap", "recorded", "has_labels"]])

    rows: list[dict[str, Any]] = []
    for rec in man.dataset:
        try:
            rows.append(cfg_summary(root, rec))
        except Exception as e:
            logger.warning("%s", e)
            rows.append({"recording": rec, "cfg": f"<{type(e).__name__}>"})
    cfgs = pd.DataFrame(rows)
    logger.info("\n%s", cfgs.head())

    REC = man.dataset.iloc[0]  # change this to explore another recording
    ds = TLVDataset(rec_dir(root, REC))
    frames = list(ds)
    logger.info("%s: %d frames", REC, len(frames))

    f = frames[len(frames) // 2]
    logger.info("timestamp     : %s", f.timestamp)
    logger.info("heatmap       : %s", np.asarray(f.output.range_doppler_heatmap).shape)
    logger.info("point cloud   : %s (x, y, z, doppler, snr, noise, track)", np.asarray(f.output.pointcloud).shape)
    logger.info("camera image  : %s", np.asarray(f.image).shape)
    logger.info("tracks        : %s", f.output.num_detected_tracks)


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
        logger.error("Error: %s\n\nUsage: python main.py data.root=<path>", e)
        sys.exit(1)

    run(cfg)


if __name__ == "__main__":
    main()
