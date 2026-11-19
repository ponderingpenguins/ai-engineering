"""Build a model from the config, so the model can be picked from the command line."""

from dataclasses import asdict, is_dataclass

from utils.config import ModelConfig

from .baseline import MajorityClass
from .model import BaseModel
from .pointnet import PointNet
from .rnn import RNN

MODEL_CLASSES: dict[str, type[BaseModel]] = {"baseline": MajorityClass, "rnn": RNN, "pointnet": PointNet}
MODELS = tuple(MODEL_CLASSES)


def build_model(cfg: ModelConfig) -> BaseModel:
    """A fresh, untrained model."""
    model_class = MODEL_CLASSES.get(cfg.name)
    if model_class is None:
        raise ValueError(f"unknown model {cfg.name!r}, choose one of {MODELS}")
    params = getattr(cfg, cfg.name, None)  # the model's own hyperparameters, if it has any
    return model_class(**asdict(params)) if is_dataclass(params) else model_class()
