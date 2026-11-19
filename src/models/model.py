from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch


class BaseModel(ABC):
    needs_heatmaps = True  # False for models that ignore x, so the heatmaps need not be loaded
    model_name: str  # Key used to register this model in models.factory.MODEL_CLASSES.

    def save(self, path: str | Path, evaluation: dict[str, Any]) -> None:
        """Save model state with the task settings used during training.

        Subclasses supply tensors and plain values through _checkpoint_state.
        Whole Python model objects are not pickled into the checkpoint.
        """
        required = {"target", "collapse_lying", "window_len"}
        if not required <= set(evaluation):
            raise ValueError(f"Checkpoint evaluation settings require {sorted(required)}")
        checkpoint = {
            **self._checkpoint_state(),
            "format_version": 1,
            "model": self.model_name,
            # label_rule is optional so checkpoints saved before it existed still load, as "last"
            "evaluation": {**{key: evaluation[key] for key in required},
                           "label_rule": evaluation.get("label_rule", "last")},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> tuple[Self, dict[str, Any]]:
        """Restore a checkpoint and its task settings without training.

        BaseModel.load selects the registered subclass automatically. Calling
        a subclass directly, such as RNN.load, checks that the type matches.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported model checkpoint format.")
        evaluation = checkpoint.get("evaluation")
        required = {"target", "collapse_lying", "window_len"}
        if not isinstance(evaluation, dict) or not required <= set(evaluation):
            raise ValueError("Checkpoint is missing evaluation settings.")
        evaluation.setdefault("label_rule", "last")  # checkpoints from before window label rules existed

        model_class = cls
        if cls is BaseModel:
            # Import here to avoid a cycle while model subclasses are imported.
            from .factory import MODEL_CLASSES

            model_class = MODEL_CLASSES.get(checkpoint.get("model"))
            if model_class is None:
                raise ValueError(f"Unknown checkpoint model: {checkpoint.get('model')!r}")
        elif checkpoint.get("model") != cls.model_name:
            raise ValueError(f"Expected a {cls.model_name!r} checkpoint.")
        return model_class._from_checkpoint(checkpoint), evaluation

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
    def fit(self, x: np.ndarray, y:np.ndarray, val:tuple[np.ndarray, np.ndarray] | None = None) -> None:
        """Trains the model from scratch.

        Args:
            x: Training windows.
            y: Training labels, shape (N,).
            val: Optional (x, y) used to choose the best epoch. Must not contain test data.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts a label for each window.

        Args:
            x: Windows to classify.

        Returns:
            Array of shape (len(x),) with the predicted labels.
        """
        pass
