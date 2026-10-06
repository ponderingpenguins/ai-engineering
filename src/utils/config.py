from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    # relative paths are resolved against the repo root
    root: str = MISSING  # raw dataset bundle (MANIFEST.csv, data/, ...)
    baked: str = "data/processed/floor_posture"  # baked HF dataset (gitignored)
