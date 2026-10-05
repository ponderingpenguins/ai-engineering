from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    needs_heatmaps = True  # False for models that ignore x, so the heatmaps need not be loaded

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
