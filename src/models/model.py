from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y:np.ndarray, val:tuple[np.ndarray, np.ndarray] | None = None) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass
