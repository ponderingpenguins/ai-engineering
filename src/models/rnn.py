import copy
import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import macro_f1
from .model import BaseModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _conv_block(c_in: int, c_out: int, pool: tuple[int, int]) -> list[nn.Module]:
    return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(pool)]


class HeatmapEncoder(nn.Module):
    """Turns each radar frame into one feature vector: (B, T, 1, R, D) -> (B, T, feat_dim).

    R = 256 range bins, D = 16 Doppler bins. Only range is pooled hard, Doppler is already short.
    """

    def __init__(self, feat_dim: int = 64) -> None:
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
        B, T = x.shape[:2]
        x = (x - self.input_mean) / self.input_std
        z = self.conv(x.flatten(0, 1))  # all B*T frames go through the same CNN
        return self.fc(z).view(B, T, -1)


class HeatmapGRU(nn.Module):
    def __init__(self, n_classes:int, feat_dim:int=64, hidden:int=128, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        # encoder
        self.encoder = HeatmapEncoder(feat_dim)
        # backbone
        self.gru = nn.GRU(feat_dim, hidden_size=hidden, batch_first=True)
        # classification head
        self.pose = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # (B, T, 1, R, D)
        z = self.encoder(x)
        return self.pose(self.gru(z)[0][:, -1])


class RNN(BaseModel):
    """CNN + GRU classifier on windows of heatmaps. x: (N, T, 1, R, D), y: (N,) string labels."""

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
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = _pick_device()

    def fit(self, x: np.ndarray, y: np.ndarray, val: tuple[np.ndarray, np.ndarray] | None = None) -> None:
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
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            optimizer.zero_grad()
            loss = loss_fn(self.net(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        predictions = []
        for start in range(0, len(x), self.batch_size):  # in batches, so a big test set fits in memory
            xb = torch.as_tensor(x[start : start + self.batch_size], dtype=torch.float32).to(self.device)
            predictions.append(self.net(xb).argmax(dim=1).cpu().numpy())
        if not predictions:
            return np.asarray([], dtype=self.classes_.dtype)
        return self.classes_[np.concatenate(predictions)]

