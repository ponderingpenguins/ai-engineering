"""Build a model from the config, so the model can be picked from the command line."""

from dataclasses import asdict

from config import ModelConfig, TrainingConfig

from .baseline import MajorityClass
from .model import BaseModel
from .pointnet import PointNet
from .rnn import RNN
from .torch_classifier import TorchClassifier

MODEL_CLASSES: dict[str, type[BaseModel]] = {"baseline": MajorityClass, "rnn": RNN, "pointnet": PointNet}
MODELS = tuple(MODEL_CLASSES)


def build_model(cfg: ModelConfig, training: TrainingConfig | None = None, *, seed: int = 0) -> BaseModel:
    """Build with model-specific architecture/augmentation and one set of loop defaults."""
    model_class = MODEL_CLASSES.get(cfg.name)
    if model_class is None:
        raise ValueError(f"unknown model {cfg.name!r}, choose one of {MODELS}")
    if cfg.name == "baseline":
        return model_class()
    training = training if training is not None else TrainingConfig()
    loop = asdict(training)
    for name in ("validation", "oversample_ratio", "class_weights"):
        loop.pop(name)
    transforms = asdict(cfg.augment)
    network_augment = ("noise",) if cfg.name == "rnn" else ("point_drop", "jitter")
    settings = {**asdict(cfg.arch), **loop, "seed": seed}
    for name in network_augment:
        settings[name] = transforms.pop(name)
    settings["augment"] = transforms
    model = model_class(**settings)
    if isinstance(model, TorchClassifier):
        model.class_weights = training.class_weights
    return model
