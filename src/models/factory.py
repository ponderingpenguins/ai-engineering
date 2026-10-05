"""Build a model from the config, so the model can be picked from the command line."""

from dataclasses import asdict

from utils.config import ModelConfig
from .baseline import MajorityClass
from .model import BaseModel
from .rnn import RNN

MODELS = ("baseline", "rnn")


def build_model(cfg: ModelConfig) -> BaseModel:
    """A fresh, untrained model."""
    if cfg.name == "baseline":
        return MajorityClass()
    if cfg.name == "rnn":
        return RNN(**asdict(cfg.rnn))
    raise ValueError(f"unknown model {cfg.name!r}, choose one of {MODELS}")
