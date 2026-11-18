from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch


@dataclass
class History:
    """Losses recorded while training one model.

    The train loss is recorded at every optimiser step, the validation loss only every few steps.
    Both are class-weighted cross-entropy, so they are on the same scale.
    """

    steps_per_epoch: int
    train_steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_step: int | None = None  # step of the lowest validation loss; its weights are the ones kept
    stopped_early: bool = False  # True if patience ran out before the last epoch

    def to_dict(self) -> dict[str, Any]:
        """Plain lists and numbers, ready for JSON."""
        return asdict(self)


class BaseModel(ABC):
    # What the windows must hold for this model: "heatmap" (N, T, 1, R, D), "pointcloud" (N, P, 7), or "none"
    # for models that ignore x, so nothing needs to be loaded. See data.windows.INPUT_TYPES.
    input_type: str = "heatmap"
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
            # optional keys: checkpoints saved before they existed still load, with the old behaviour
            "evaluation": {**{key: evaluation[key] for key in required},
                           "label_rule": evaluation.get("label_rule", "last"),
                           "max_points": evaluation.get("max_points", 128)},
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
        evaluation.setdefault("max_points", 128)  # and from before point-cloud windows existed

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
    def fit(self, x: np.ndarray, y: np.ndarray, val: tuple[np.ndarray, np.ndarray] | None = None,
            val_weight_labels: np.ndarray | None = None) -> History | None:
        """Trains the model from scratch.

        Args:
            x: Training windows.
            y: Training labels, shape (N,).
            val: Optional (x, y) used for early stopping and to choose the best weights.
                Must not contain test data.
            val_weight_labels: Optional labels whose class frequencies weight the validation loss. Pass the
                pooled labels of many people so validation losses of different people are comparable. Default:
                the labels of `val` itself.

        Returns:
            The recorded losses, or None for models that do not train iteratively.
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
