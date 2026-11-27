"""Settings for model fitting, cross-validation and hyperparameter tuning.

RunConfig combines data settings, the selected model and shared training controls.
"""

from dataclasses import dataclass, field
from typing import Any
from .data import DataConfig
from .models import ModelConfig


@dataclass
class TrainingConfig:
    """Optimiser, batching, early-stopping and training class-balance settings."""

    epochs: int = 30
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 0.01
    validation: bool = True
    patience: int = 5
    evals_per_epoch: int = 1
    class_weights: bool = True
    oversample_ratio: float | None = None


@dataclass
class CVConfig:
    """Outer fold count and an optional per-person window cap for quick checks."""

    folds: int | None = None
    max_windows_per_person: int | None = None


@dataclass
class TuneConfig:
    """Candidate overrides and the file used to pass the winner to final training."""

    grid: list[dict[str, Any]] = field(default_factory=list)
    best_path: str = "results/tuned.yaml"


@dataclass
class RunConfig(DataConfig):
    """Complete data, model, training and evaluation settings for one run."""

    model: ModelConfig = field(default_factory=ModelConfig)
    seed: int = 0
    out_dir: str = "results"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    tune: TuneConfig = field(default_factory=TuneConfig)
