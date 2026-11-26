from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import torch
from torch import nn

from config import WindowConfig


class BaseModel(ABC):
    # What the windows must hold for this model: "heatmap" (N, T, 1, R, D), "pointcloud" (N, P, 7), or "none"
    # for models that ignore x, so nothing needs to be loaded. See data.windowing.INPUT_TYPES.
    input_type: str = "heatmap"
    model_name: str  # Key used to register this model in models.factory.MODEL_CLASSES.

    def save(self, path: str | Path, windowing: dict[str, Any]) -> None:
        """Save model state with its input and label specification.

        Subclasses supply tensors and plain values through _checkpoint_state.
        Whole Python model objects are not pickled into the checkpoint.
        """
        checkpoint = {
            **self._checkpoint_state(),
            "format_version": 3,
            "model": self.model_name,
            "windowing": dict(windowing),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple[Self, dict[str, Any]]:
        """Restore a model and its window specification without training.

        BaseModel.load selects the registered subclass automatically. Calling
        a subclass directly, such as RNN.load, checks that the type matches.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") not in (1, 2, 3):
            raise ValueError("Unsupported model checkpoint format.")
        version = checkpoint["format_version"]
        values = checkpoint.get({1: "evaluation", 2: "window_spec", 3: "windowing"}[version])
        if not isinstance(values, dict) or values.keys() != WindowConfig().signature().keys():
            raise ValueError("Checkpoint is missing valid windowing settings.")
        windowing = dict(values)

        model_class: type[BaseModel] = cls
        if cls is BaseModel:
            # Import here to avoid a cycle while model subclasses are imported.
            from .factory import MODEL_CLASSES

            model_name = checkpoint.get("model")
            registered = MODEL_CLASSES.get(model_name) if isinstance(model_name, str) else None
            if registered is None:
                raise ValueError(f"Unknown checkpoint model: {checkpoint.get('model')!r}")
            model_class = registered
        elif checkpoint.get("model") != cls.model_name:
            raise ValueError(f"Expected a {cls.model_name!r} checkpoint.")
        return cast(Self, model_class._from_checkpoint(checkpoint)), windowing

    @abstractmethod
    def _checkpoint_state(self) -> dict[str, Any]:
        """Return learned state and constructor settings as tensors/plain values."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_checkpoint(cls, checkpoint: dict[str, Any]) -> Self:
        """Reconstruct a prediction-ready model from its saved state."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts a label for each window.

        Args:
            x: Windows to classify.

        Returns:
            Array of shape (len(x),) with the predicted labels.
        """
        pass


def pick_device() -> torch.device:
    """Picks the fastest available device.

    Returns:
        CUDA if available, else MPS, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchModel(BaseModel):
    """Neural architecture, batched inference and checkpoints; fitting lives in training."""

    net: nn.Module
    classes_: np.ndarray
    preprocessing: str
    parameter_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.device = pick_device()

    @abstractmethod
    def _build_net(self, n_classes: int) -> nn.Module:
        """Build a fresh network with one output per class."""

    def _set_input_stats(self, x: np.ndarray) -> torch.Tensor | None:
        """Set training-input statistics and return their scale, if applicable."""
        return None

    def _checkpoint_state(self) -> dict[str, Any]:
        """The hyperparameters, class names and weights. A subclass only lists its `parameter_names`."""
        if not hasattr(self, "net"):
            raise ValueError("Train the model before saving it.")
        parameters = {name: getattr(self, name) for name in self.parameter_names}
        return {
            "parameters": parameters,
            "classes": self.classes_.tolist(),
            "state_dict": self.net.state_dict(),
            "preprocessing": self.preprocessing,
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict[str, Any]) -> Self:
        """Rebuilds the model from a checkpoint made by `_checkpoint_state`."""
        if checkpoint.get("preprocessing") != cls.preprocessing:
            raise ValueError("Checkpoint preprocessing does not match the current pipeline.")
        # Legacy checkpoints also stored optimiser and augmentation settings.
        # Inference only needs the architecture; learned input statistics are buffers.
        parameters = {name: value for name, value in checkpoint["parameters"].items()
                      if name in cls.parameter_names}
        model = cls(**parameters)
        model.classes_ = np.asarray(checkpoint["classes"], dtype=str)
        if len(model.classes_) == 0:
            raise ValueError("Checkpoint has no class labels.")
        model.net = model._build_net(len(model.classes_)).to(model.device)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.eval()
        return model

    @torch.no_grad()
    def predict(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Predicts a label for each window.

        Args:
            x: Heatmaps (N, T, 1, R, D) or point clouds (N, P, 7).
            batch_size: Maximum windows per inference batch.

        Returns:
            String labels of shape (N,).
        """
        self.net.eval()
        predictions = []
        for start in range(0, len(x), batch_size):  # in batches, so a big test set fits in memory
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32).to(self.device)
            predictions.append(self.net(xb).argmax(dim=1).cpu().numpy())
        if not predictions:
            return np.asarray([], dtype=self.classes_.dtype)
        return self.classes_[np.concatenate(predictions)]
