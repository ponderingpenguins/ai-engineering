import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk
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

# per-recording sensor config, constant within a recording
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


def load_baked(path: Path) -> Dataset:
    '''Memory-mapped baked dataset, see `python -m data.extract`.'''
    if not path.is_dir():
        logger.error(
            "No baked dataset at %s.\n"
            "Build it first with: python -m data.extract",
            path,
        )
        sys.exit(1)
    return cast(Dataset, load_from_disk(str(path)))


def recording_summary(ds: Dataset) -> pd.DataFrame:
    '''One row per recording: room, frame count, sensor config, heatmap validity.'''
    # skip the heavy columns (image, payload, heatmap); this reads only the light ones
    df = ds.select_columns(["recording", "room", *CFG_COLUMNS, "n_tracks", "heatmap_ok"]).to_pandas()
    return (
        df.groupby("recording", sort=False)
        .agg(
            room=("room", "first"),
            frames=("recording", "size"),
            bad_heatmaps=("heatmap_ok", lambda s: int((~s).sum())),
            **{c: (c, "first") for c in CFG_COLUMNS},
        )
        .reset_index()
    )


def run(cfg: Config) -> None:
    path = REPO / cfg.data.out  # absolute paths pass through unchanged
    ds = load_baked(path)

    logger.info("dataset : %s", path)
    logger.info("rows    : %s", f"{len(ds):,}")
    logger.info("columns : %s", ds.column_names)

    recs = recording_summary(ds)
    logger.info("%d recordings, %s frames", len(recs), f"{recs.frames.sum():,}")
    logger.info("\n%s", recs.drop(columns=["cfg_file"]).head())

    labels = [c for c in ds.column_names if c.startswith("label_") and not c.endswith("_confidence")]
    logger.info("label tracks: %s", labels)

    rec = recs.recording.iloc[0]  # change this to explore another recording
    sub = ds.filter(lambda r: r == rec, input_columns="recording")
    logger.info("%s: %d frames", rec, len(sub))

    f = sub[len(sub) // 2]
    heatmap = np.asarray(f["heatmap"])
    if heatmap.size:
        # stored raw with zero velocity at doppler bin 0
        heatmap = np.fft.fftshift(heatmap, axes=-1)
    logger.info("timestamp     : %s", f["timestamp"])
    logger.info("heatmap       : %s", heatmap.shape)
    logger.info("point cloud   : %s (x, y, z, doppler, snr, noise, track)", np.asarray(f["pointcloud"]).shape)
    logger.info("camera image  : %s", np.asarray(f["image"]).shape if f["image"] is not None else None)
    logger.info("tracks        : %s", f["n_tracks"])


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
        logger.error("Error: %s\n\nUsage: python main.py [data.out=<path>]", e)
        sys.exit(1)

    run(cfg)


if __name__ == "__main__":
    main()
