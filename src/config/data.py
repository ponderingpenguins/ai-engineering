"""Settings for dataset paths, selection, denoising and labelled windows."""
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PathsConfig:
    raw_root: str = "data/raw/floor_posture_dataset"
    baked_dir: str = "data/baked/floor_posture"
    annotations_dir: str = "data/annotations"


@dataclass
class SelectionConfig:
    sensor_config: str | None = "Config_4.cfg"
    exclude_recordings: list[str] = field(default_factory=lambda: ["2311__20231124_152926", "2311__20231124_153817"])

    @property
    def require_known_person(self) -> bool:
        """Existing selection pipelines always discard unknown participants."""
        return True


@dataclass
class WindowConfig:
    target: str = "lying_annotation"
    collapse_lying: bool = True
    window_len: int = 16
    stride: int = 8
    label_rule: str = "any_lying"
    max_points: int = 128

    def signature(self) -> dict[str, Any]:
        """Prediction settings saved with artifacts; stride only changes sampling."""
        values = asdict(self)
        values.pop("stride")
        return values


@dataclass
class RangeNormaliseConfig:
    enabled: bool = False
    percentile: float = 10.0
    smooth_bins: int = 41


@dataclass
class TemporalBackgroundConfig:
    enabled: bool = False
    tau_frames: float = 300.0


@dataclass
class PreprocessConfig:
    range_normalise: RangeNormaliseConfig = field(default_factory=RangeNormaliseConfig)
    temporal_background: TemporalBackgroundConfig = field(default_factory=TemporalBackgroundConfig)


@dataclass
class DataConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    denoising: PreprocessConfig = field(default_factory=PreprocessConfig)
    windowing: WindowConfig = field(default_factory=WindowConfig)
