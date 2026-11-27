"""Settings for dataset paths, recording selection, denoising and windows.

These schemas define the defaults and types used by config/data.yaml.
"""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PathsConfig:
    """Raw, baked and annotation paths, resolved from the repository root."""

    raw_root: str = "data/raw/floor_posture_dataset"
    baked_dir: str = "data/baked/floor_posture"
    annotations_dir: str = "data/annotations"


@dataclass
class SelectionConfig:
    """Recording filters applied before denoising and windowing."""

    sensor_config: str | None = "Config_4.cfg"
    exclude_recordings: list[str] = field(default_factory=lambda: ["2311__20231124_152926", "2311__20231124_153817"])

    @property
    def require_known_person(self) -> bool:
        """Always require a known person to keep person-wise splits meaningful."""
        return True


@dataclass
class WindowConfig:
    """Frame labels, window sizes and point limits used to build model inputs."""

    target: str = "lying_annotation"
    collapse_lying: bool = True
    window_len: int = 16
    stride: int = 8
    label_rule: str = "any_lying"
    max_points: int = 128

    def signature(self) -> dict[str, Any]:
        """Return the window settings stored with checkpoints and tuned settings.

        Stride is excluded because it changes sampling, not the contents of one window.
        """
        values = asdict(self)
        values.pop("stride")
        return values


@dataclass
class RangeNormaliseConfig:
    """Settings for subtracting a smoothed noise floor along the range axis."""

    enabled: bool = False
    percentile: float = 10.0
    smooth_bins: int = 41


@dataclass
class TemporalBackgroundConfig:
    """Settings for subtracting a slowly changing background from each heatmap cell."""

    enabled: bool = False
    tau_frames: float = 300.0


@dataclass
class PreprocessConfig:
    """Optional heatmap denoising steps, applied in the order defined by preprocessing."""

    range_normalise: RangeNormaliseConfig = field(default_factory=RangeNormaliseConfig)
    temporal_background: TemporalBackgroundConfig = field(default_factory=TemporalBackgroundConfig)


@dataclass
class DataConfig:
    """Settings shared by baking, annotation tools and model data preparation."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    denoising: PreprocessConfig = field(default_factory=PreprocessConfig)
    windowing: WindowConfig = field(default_factory=WindowConfig)
