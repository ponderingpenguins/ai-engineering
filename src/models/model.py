"""Shared prediction interfaces and checkpoint storage for posture models.

BaseModel saves learned state with the window settings needed for prediction.
TorchModel adds network construction, batched prediction and input statistics.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import torch
from torch import nn

from config.data import WindowConfig


class BaseModel(ABC):
    # Input layouts: heatmap (N, T, 1, R, D), pointcloud (N, P, 7), or none (N, T).
    # The baseline uses zero placeholders; see data.windowing.INPUT_TYPES.
    """Shared label-prediction interface and portable checkpoint format."""

    input_type: str = "heatmap"
    model_name: str  # Key used to register this model in models.factory.MODEL_CLASSES.

    def save(self, path: str | Path, windowing: dict[str, Any]) -> None:
        """Save learned state with the input and label settings used for prediction.

        Subclasses provide tensors and plain values through _checkpoint_state;
        the checkpoint does not store a pickled model instance.

        Args:
            path: Checkpoint destination; missing parent directories are created.
            windowing: WindowConfig.signature() values for the training inputs.
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
        """Restore a model and its saved window settings without training.

        BaseModel.load chooses the registered model class. Calling a subclass,
        such as RNN.load, also checks that the saved model type matches.

        Args:
            path: Checkpoint file written by save.

        Returns:
            The prediction-ready model and its window settings. Callers must compare
            those values with the settings used to build new inputs.

        Raises:
            ValueError: If the format, window-setting keys, model type or saved
                preprocessing identifier is unsupported.
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
        """Return learned state and constructor settings as tensors and plain values."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_checkpoint(cls, checkpoint: dict[str, Any]) -> Self:
        """Reconstruct a prediction-ready model from its saved state."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict one label per input window.

        Args:
            x: Model inputs with one window per row; shape depends on input_type.

        Returns:
            String labels of shape (N,).
        """
        pass


def pick_device() -> torch.device:
    """Choose CUDA if available, otherwise MPS, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchModel(BaseModel):
    """Neural network construction, batched prediction and checkpoint restoration."""

    net: nn.Module
    classes_: np.ndarray
    preprocessing: str
    parameter_names: tuple[str, ...] = ()

    def __init__(self) -> None:
        """Select the device used when the network is built or restored."""
        self.device = pick_device()

    @abstractmethod
    def _build_net(self, n_classes: int) -> nn.Module:
        """Build a fresh network with one output per class."""

    def _set_input_stats(self, x: np.ndarray) -> torch.Tensor | None:
        """Set training-only input statistics and return their scale when needed."""
        return None

    def _checkpoint_state(self) -> dict[str, Any]:
        """Return architecture parameters, class names, weights and preprocessing ID.

        Subclasses list constructor settings in parameter_names. Learned input
        statistics are already included in the network's state_dict buffers.
        """
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
        """Rebuild the network, load its saved buffers and weights, and enable evaluation mode."""
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
        """Predict one label per window in batches without tracking gradients.

        Args:
            x: Heatmaps (N, T, 1, R, D) or point clouds (N, P, 7).
            batch_size: Maximum windows per inference batch.

        Returns:
            String labels of shape (N,).
        """
        self.net.eval()
        predictions = []
        for start in range(0, len(x), batch_size):  # Bound device memory use during prediction.
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32).to(self.device)
            predictions.append(self.net(xb).argmax(dim=1).cpu().numpy())
        if not predictions:
            return np.asarray([], dtype=self.classes_.dtype)
        return self.classes_[np.concatenate(predictions)]
