import logging
from typing import Any

import numpy as np
import torch
from torch import nn

from data.augment import HeatmapAugment

from .torch_classifier import TorchClassifier

logger = logging.getLogger(__name__)


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

    def __init__(self, feat_dim: int = 64, dropout: float = 0.0) -> None:
        """Initializes the encoder.

        Args:
            feat_dim: Size of the feature vector per frame.
            dropout: Dropout on the per-frame feature vector.
        """
        super().__init__()
        self.feature_dropout = nn.Dropout(dropout)
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
        return self.feature_dropout(self.fc(z)).view(B, T, -1)


class HeatmapGRU(nn.Module):
    def __init__(self, n_classes: int, feat_dim: int = 64, hidden: int = 128, dropout: float = 0.0,
                 feature_dropout: float = 0.0, *args: Any, **kwargs: Any) -> None:
        """Initializes the encoder, the GRU and the classification head.

        Args:
            n_classes: Number of output classes.
            feat_dim: Size of the feature vector per frame.
            hidden: Size of the GRU hidden state.
            dropout: Dropout on the GRU output before the classification layer.
            feature_dropout: Dropout on the per-frame CNN features.
            *args: Unused.
            **kwargs: Unused.
        """
        super().__init__()
        # encoder
        self.encoder = HeatmapEncoder(feat_dim, feature_dropout)
        # backbone
        self.gru = nn.GRU(feat_dim, hidden_size=hidden, batch_first=True)
        # classification head
        self.dropout = nn.Dropout(dropout)
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
        return self.pose(self.dropout(self.gru(z)[0][:, -1]))


class RNN(TorchClassifier):
    """CNN + GRU classifier on windows of heatmaps. x: (N, T, 1, R, D), y: (N,) string labels."""

    net: HeatmapGRU
    augment: HeatmapAugment | None

    model_name = "rnn"
    input_type = "heatmap"
    preprocessing = "log1p_raw_heatmap_v1"
    parameter_names = ("hidden", "feat_dim", "dropout", "feature_dropout", "lr", "weight_decay", "epochs",
                       "batch_size", "seed", "patience", "evals_per_epoch", "augment")

    def __init__(
        self,
        hidden: int = 128,
        feat_dim: int = 64,
        dropout: float = 0.5,
        feature_dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        epochs: int = 30,
        batch_size: int = 32,
        seed: int = 0,
        patience: int = 5,
        evals_per_epoch: int = 1,
        augment: dict[str, Any] | None = None,
    ) -> None:
        """Stores the hyperparameters; `RNNConfig` says what each one does. The network is built in `fit`.

        `augment` holds the settings of `HeatmapAugment`; None or all off means no augmentation.
        """
        super().__init__(lr, weight_decay, epochs, batch_size, seed, patience, evals_per_epoch)
        self._set_augment(augment, HeatmapAugment)
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.dropout = dropout
        self.feature_dropout = feature_dropout

    def _build_net(self, n_classes: int) -> HeatmapGRU:
        return HeatmapGRU(n_classes, self.feat_dim, self.hidden, self.dropout, self.feature_dropout)

    def _set_input_stats(self, x: np.ndarray) -> None:
        """Standardise the heatmaps with the mean and std of the training windows."""
        encoder = self.net.encoder
        encoder.input_mean.fill_(float(x.mean()))
        encoder.input_std.fill_(max(float(x.std()), 1e-6))
        if self.augment is not None:
            self.augment.set_input_std(encoder.input_std)
