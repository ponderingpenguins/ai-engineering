"""Construct the classifier selected by the configuration.

MODEL_CLASSES is also used to choose input types and restore checkpoints.
"""

from dataclasses import asdict

from config.models import ModelConfig

from .baseline import MajorityClass
from .model import BaseModel
from .pointnet import PointNet
from .rnn import RNN

MODEL_CLASSES: dict[str, type[BaseModel]] = {"baseline": MajorityClass, "rnn": RNN, "pointnet": PointNet}
MODELS = tuple(MODEL_CLASSES)


def build_model(cfg: ModelConfig) -> BaseModel:
    """Build an architecture without training or augmentation settings."""
    model_class = MODEL_CLASSES.get(cfg.name)
    if model_class is None:
        raise ValueError(f"unknown model {cfg.name!r}, choose one of {MODELS}")
    return model_class() if cfg.name == "baseline" else model_class(**asdict(cfg.arch))
