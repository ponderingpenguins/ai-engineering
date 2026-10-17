import copy
import logging
from typing import Any, Self

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.metrics import macro_f1

from .model import BaseModel

logger = logging.getLogger(__name__)


def _pick_device() -> torch.device:
    """Picks the fastest available device.

    Returns:
        CUDA if available, else MPS, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _conv_block(c_in: int, c_out: int, pool: tuple[int, int]) -> list[nn.Module]:
    """Builds a conv, batch norm, ReLU, max pool block.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        pool: Max pooling window over (range, Doppler).

    Returns:
        The layers of the block, in order.
    """
    return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(pool)]


class HeatmapEncoder(nn.Module):
    """Turns each radar frame into one feature vector: (B, T, 1, R, D) -> (B, T, feat_dim).

    R = 256 range bins, D = 16 Doppler bins. Only range is pooled hard, Doppler is already short.
    """

    # registered as buffers below, declared here so type checkers know they are tensors
    input_mean: torch.Tensor
    input_std: torch.Tensor

    def __init__(self, feat_dim: int = 64) -> None:
        """Initializes the encoder.

        Args:
            feat_dim: Size of the feature vector per frame.
        """
        super().__init__()
        self.conv = nn.Sequential(
            *_conv_block(1, 16, (2, 1)),  # (16, 128, 16)
            *_conv_block(16, 32, (2, 1)),  # (32, 64, 16)
            *_conv_block(32, 64, (2, 2)),  # (64, 32, 8)
            nn.AdaptiveAvgPool2d((4, 2)),  # (64, 4, 2): keeps coarse position along range
            nn.Flatten(),  # 512
        )
        self.fc = nn.Linear(64 * 4 * 2, feat_dim)
        # input standardization: identity until set from the training data
        self.register_buffer("input_mean", torch.zeros(1))
        self.register_buffer("input_std", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes every frame of every window.

        Args:
            x: Heatmaps of shape (B, T, 1, R, D).

        Returns:
            Features of shape (B, T, feat_dim).
        """
        B, T = x.shape[:2]
        x = (x - self.input_mean) / self.input_std
        z = self.conv(x.flatten(0, 1))  # all B*T frames go through the same CNN
        return self.fc(z).view(B, T, -1)


class HeatmapGRU(nn.Module):
    def __init__(self, n_classes:int, feat_dim:int=64, hidden:int=128, *args: Any, **kwargs: Any) -> None:
        """Initializes the encoder, the GRU and the classification head.

        Args:
            n_classes: Number of output classes.
            feat_dim: Size of the feature vector per frame.
            hidden: Size of the GRU hidden state.
            *args: Unused.
            **kwargs: Unused.
        """
        super().__init__()
        # encoder
        self.encoder = HeatmapEncoder(feat_dim)
        # backbone
        self.gru = nn.GRU(feat_dim, hidden_size=hidden, batch_first=True)
        # classification head
        self.pose = nn.Linear(hidden, n_classes)

    def forward(self, x):
        """Classifies a window from its last GRU state.

        Args:
            x: Heatmaps of shape (B, T, 1, R, D).

        Returns:
            Class logits of shape (B, n_classes).
        """
        # (B, T, 1, R, D)
        z = self.encoder(x)
        return self.pose(self.gru(z)[0][:, -1])


class RNN(BaseModel):
    """CNN + GRU classifier on windows of heatmaps. x: (N, T, 1, R, D), y: (N,) string labels."""

    model_name = "rnn"

    def __init__(
        self,
        hidden: int = 128,
        feat_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 1,
        batch_size: int = 32,
        seed: int = 0,
    ) -> None:
        """Stores the hyperparameters. The network is built in `fit`.

        Args:
            hidden: Size of the GRU hidden state.
            feat_dim: Size of the feature vector per frame.
            lr: Adam learning rate.
            weight_decay: Adam weight decay.
            epochs: Number of passes over the training data.
            batch_size: Batch size for training and prediction.
            seed: Torch seed, set at the start of `fit`.
        """
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = _pick_device()

    def _checkpoint_state(self) -> dict[str, Any]:
        """Include normalization buffers along with weights and class names."""
        if not hasattr(self, "net"):
            raise ValueError("Train the model before saving it.")
        return {
            "parameters": {
                "hidden": self.hidden, "feat_dim": self.feat_dim,
                "lr": self.lr, "weight_decay": self.weight_decay,
                "epochs": self.epochs, "batch_size": self.batch_size, "seed": self.seed,
            },
            "classes": self.classes_.tolist(),
            "state_dict": self.net.state_dict(),
            "preprocessing": "log1p_raw_heatmap_v1",
        }

    @classmethod
    def _from_checkpoint(cls, checkpoint: dict[str, Any]) -> Self:
        """Restore the RNN architecture, normalization and learned weights."""
        if checkpoint.get("preprocessing") != "log1p_raw_heatmap_v1":
            raise ValueError("Checkpoint preprocessing does not match the current pipeline.")
        model = cls(**checkpoint["parameters"])
        model.classes_ = np.asarray(checkpoint["classes"], dtype=str)
        if len(model.classes_) == 0:
            raise ValueError("Checkpoint has no class labels.")
        model.net = HeatmapGRU(len(model.classes_), model.feat_dim, model.hidden).to(model.device)
        model.net.load_state_dict(checkpoint["state_dict"])
        model.net.eval()
        return model

    def fit(self, x: np.ndarray, y: np.ndarray, val: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        """Trains a new network on windows of heatmaps.

        Classes are weighted by inverse frequency. If `val` is given, the weights of the
        epoch with the best validation macro F1 are kept, otherwise those of the last epoch.

        Args:
            x: Heatmaps of shape (N, T, 1, R, D).
            y: String labels of shape (N,).
            val: Optional (x, y) validation data, used to choose the best epoch.
        """
        torch.manual_seed(self.seed)

        # labels -> integers 0..K-1, classes_ remembers which name is which
        self.classes_, y_int = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        self.net = HeatmapGRU(n_classes, self.feat_dim, self.hidden).to(self.device)
        # standardize the input with statistics from the training data only
        self.net.encoder.input_mean.fill_(float(x.mean()))
        self.net.encoder.input_std.fill_(max(float(x.std()), 1e-6))

        # rare classes get a bigger weight, otherwise "always predict the common class" wins
        counts = np.bincount(y_int, minlength=n_classes)
        weights = torch.as_tensor(len(y_int) / (n_classes * counts), dtype=torch.float32)
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(self.device))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_data = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y_int, dtype=torch.long))
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        best_score, best_state = -1.0, None
        for epoch in tqdm(range(1, self.epochs + 1)):
            loss = self._train_epoch(loader, optimizer, loss_fn)
            message = f"epoch {epoch}/{self.epochs} loss={loss:.4f}"

            if val is not None and len(val[1]) > 0:
                score = macro_f1(val[1], self.predict(val[0]))
                message += f" val_macro_f1={score:.3f}"
                if score > best_score:  # remember the best epoch on validation data
                    best_score, best_state = score, copy.deepcopy(self.net.state_dict())
            logger.info(message)

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module) -> float:
        """One pass over the training data, returns the mean loss."""
        self.net.train()
        total_loss, n_samples = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            loss = loss_fn(self.net(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            n_samples += len(xb)
        return total_loss / n_samples

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts a label for each window.

        Args:
            x: Heatmaps of shape (N, T, 1, R, D).

        Returns:
            String labels of shape (N,).
        """
        self.net.eval()
        predictions = []
        for start in range(0, len(x), self.batch_size):  # in batches, so a big test set fits in memory
            xb = torch.as_tensor(x[start : start + self.batch_size], dtype=torch.float32).to(self.device)
            predictions.append(self.net(xb).argmax(dim=1).cpu().numpy())
        if not predictions:
            return np.asarray([], dtype=self.classes_.dtype)
        return self.classes_[np.concatenate(predictions)]

