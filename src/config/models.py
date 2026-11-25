"""Only the selected model's architecture and augmentation settings are loaded."""
from dataclasses import dataclass, field
from typing import Any

@dataclass
class HeatmapAugmentConfig:
    noise: float = 0.05
    range_shift_bins: float = 0.0
    floor_jitter: float = 0.0
    time_reverse: bool = False


@dataclass
class PointAugmentConfig:
    point_drop: float = 0.3
    jitter: float = 0.05
    shift_m: float = 0.0
    height_m: float = 0.0
    yaw_deg: float = 0.0
    tilt_deg: float = 0.0
    mirror: bool = False
    ghost_points: float = 0.0
    snr_jitter_db: float = 0.0


@dataclass
class RNNArchitecture:
    hidden: int = 128
    feat_dim: int = 64
    dropout: float = 0.5
    feature_dropout: float = 0.3


@dataclass
class PointNetArchitecture:
    width: int = 128
    dropout: float = 0.5
    point_dropout: float = 0.2


@dataclass
class ModelConfig:
    name: str = "rnn"
    arch: Any = None
    augment: Any = None

    def __post_init__(self) -> None:
        if self.name not in MODEL_SCHEMAS:
            raise ValueError(f"model must be one of {tuple(MODEL_SCHEMAS)}, got {self.name!r}")
        architecture, augmentation = MODEL_SCHEMAS[self.name]
        if self.arch is None:
            self.arch = architecture()
        if self.augment is None:
            self.augment = augmentation()


MODEL_SCHEMAS = {
    "rnn": (RNNArchitecture, HeatmapAugmentConfig),
    "pointnet": (PointNetArchitecture, PointAugmentConfig),
    "baseline": (dict, dict),
}
