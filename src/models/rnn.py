"""Classify heatmap sequences with a per-frame CNN followed by a GRU.

HeatmapEncoder extracts frame features; HeatmapGRU predicts from the final GRU state.
Inputs are log1p heatmaps with shape (N, T, 1, 256, 16).
"""

import numpy as np
import torch
from torch import nn

from .model import TorchModel

def _conv_block(c_in: int, c_out: int, pool: tuple[int, int]) -> list[nn.Module]:
    """Build a convolution, batch normalisation, ReLU and max-pooling block.

    Args:
        c_in: Number of input channels.
        c_out: Number of output channels.
        pool: Pooling window over the (range, Doppler) axes.

    Returns:
        The layers in execution order.
    """
    return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(pool)]


class HeatmapEncoder(nn.Module):
    """Encode (B, T, 1, R, D) heatmaps as (B, T, feat_dim) frame features.

    Config_4 uses R=256 and D=16. Pool mainly along range because the Doppler
    axis is already small, then retain a coarse 4-by-2 spatial grid.
    """

    # Tensor buffers declared here for type checkers and registered below.
    input_mean: torch.Tensor
    input_std: torch.Tensor

    def __init__(self, feat_dim: int = 64, dropout: float = 0.0) -> None:
        """Build the frame encoder and initially neutral input statistics.

        Args:
            feat_dim: Number of features per frame.
            dropout: Dropout probability on the final frame features.
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
        # Leave inputs unchanged until training statistics are set.
        self.register_buffer("input_mean", torch.zeros(1))
        self.register_buffer("input_std", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise and encode every frame in the batch.

        Args:
            x: Heatmaps of shape (B, T, 1, R, D).

        Returns:
            Frame features of shape (B, T, feat_dim).
        """
        B, T = x.shape[:2]
        x = (x - self.input_mean) / self.input_std
        z = self.conv(x.flatten(0, 1))  # Share one CNN across all B*T frames.
        return self.feature_dropout(self.fc(z)).view(B, T, -1)


class HeatmapGRU(nn.Module):
    """Encode each frame, then classify the window from the final GRU output."""

    def __init__(self, n_classes: int, feat_dim: int = 64, hidden: int = 128, dropout: float = 0.0,
                 feature_dropout: float = 0.0) -> None:
        """Build the frame encoder, GRU and classification head.

        Args:
            n_classes: Number of output classes.
            feat_dim: Number of CNN features per frame.
            hidden: Number of GRU hidden units.
            dropout: Dropout probability before the classification layer.
            feature_dropout: Dropout probability on the CNN's frame features.
        """
        super().__init__()
        self.encoder = HeatmapEncoder(feat_dim, feature_dropout)
        self.gru = nn.GRU(feat_dim, hidden_size=hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.pose = nn.Linear(hidden, n_classes)

    def forward(self, x):
        """Classify windows using their final GRU output.

        Args:
            x: Heatmaps of shape (B, T, 1, R, D).

        Returns:
            Unnormalised class scores of shape (B, n_classes).
        """
        z = self.encoder(x)
        return self.pose(self.dropout(self.gru(z)[0][:, -1]))


class RNN(TorchModel):
    """CNN and GRU model for heatmap windows, with one string label per window."""

    net: HeatmapGRU

    model_name = "rnn"
    input_type = "heatmap"
    preprocessing = "log1p_raw_heatmap_v1"
    parameter_names = ("hidden", "feat_dim", "dropout", "feature_dropout")

    def __init__(
        self,
        hidden: int = 128,
        feat_dim: int = 64,
        dropout: float = 0.5,
        feature_dropout: float = 0.3,
    ) -> None:
        """Store architecture settings; the network is built when fitted or loaded."""
        super().__init__()
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.dropout = dropout
        self.feature_dropout = feature_dropout

    def _build_net(self, n_classes: int) -> HeatmapGRU:
        """Construct the CNN and GRU with one output score per class."""
        return HeatmapGRU(n_classes, self.feat_dim, self.hidden, self.dropout, self.feature_dropout)

    def _set_input_stats(self, x: np.ndarray) -> torch.Tensor:
        """Store the training heatmap mean and standard deviation; return the scale."""
        encoder = self.net.encoder
        encoder.input_mean.fill_(float(x.mean()))
        encoder.input_std.fill_(max(float(x.std()), 1e-6))
        return encoder.input_std
