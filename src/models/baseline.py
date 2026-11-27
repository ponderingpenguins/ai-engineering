"""Predict the most common training label for every window.

This baseline uses labels only; its input arrays are placeholders.
"""

from collections import Counter
from typing import Any, Self

import numpy as np

from .model import BaseModel


class MajorityClass(BaseModel):
    """A label-only classifier that remembers the most frequent training label."""

    input_type = "none"
    model_name = "baseline"

    def _checkpoint_state(self) -> dict[str, Any]:
        """Return the learned majority label, raising ValueError if it is unset."""
        if not hasattr(self, "label_"):
            raise ValueError("Set the majority label before saving the baseline.")
        return {"label": str(self.label_)}

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict[str, Any]) -> Self:
        """Restore the learned majority label from a checkpoint."""
        model = cls()
        model.label_ = checkpoint["label"]
        return model

    def fit(self, x: np.ndarray, y: np.ndarray, val: tuple[np.ndarray, np.ndarray] | None = None,
            val_weight_labels: np.ndarray | None = None) -> None:
        """Remember the most frequent training label.

        Args:
            x: Ignored placeholder inputs.
            y: Non-empty training labels of shape (N,).
            val: Ignored validation data.
            val_weight_labels: Ignored validation-weight labels.
        """
        self.label_ = Counter(y).most_common(1)[0][0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Repeat the learned majority label for every input window.

        Args:
            x: Placeholder inputs; only the number of rows is used.

        Returns:
            String labels of shape (N,).
        """
        n = len(x)
        return np.asarray([self.label_] * n)
