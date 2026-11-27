"""Classify a window of radar points with a small PointNet network.

Inputs have shape (N, P, 7): x, y, z, Doppler, SNR, relative time and validity.
The network pools only valid points and returns one class score per window and class.
"""

import numpy as np
import torch
from torch import nn

from .model import TorchModel

N_STANDARDISED = 5  # Standardise x, y, z, Doppler and SNR; preserve time and validity.


class MiniPointNet(nn.Module):
    """Encode points independently, then pool valid points to classify each window.

    Inputs are (B, P, 7): x, y, z, Doppler, SNR, relative time and validity.
    A shared multilayer perceptron produces features for max and mean pooling.
    The head also uses log1p of the point count; padding contributes to neither pool.
    """

    feat_mean: torch.Tensor  # Tensor buffers declared here for type checkers.
    feat_std: torch.Tensor

    def __init__(self, n_classes: int, width: int = 128, dropout: float = 0.2, point_dropout: float = 0.0) -> None:
        """Build the per-point network, normalisation buffers and classification head.

        Args:
            n_classes: Number of output classes.
            width: Number of features per point before pooling.
            dropout: Dropout probability in the classification head.
            point_dropout: Dropout probability on point features before pooling;
                this does not remove input points.
        """
        super().__init__()
        self.feature_dropout = nn.Dropout(point_dropout)  # Drop features before pooling.
        self.register_buffer("feat_mean", torch.zeros(N_STANDARDISED))
        self.register_buffer("feat_std", torch.ones(N_STANDARDISED))
        self.point_mlp = nn.Sequential(
            nn.Linear(6, width // 4), nn.ReLU(),
            nn.Linear(width // 4, width // 2), nn.ReLU(),
            nn.Linear(width // 2, width), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * width + 1, width // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width // 2, n_classes),
        )

    def forward(self, x: torch.Tensor, n_points: torch.Tensor | None = None) -> torch.Tensor:
        """Return unnormalised class scores of shape (B, n_classes).

        Args:
            x: Point windows (B, P, 7); the final feature marks real points.
            n_points: Optional counts (B, 1) taken before random point dropping.
                If omitted, count valid points in x.
        """
        valid = x[..., 6] > 0.5  # (B, P)
        if n_points is None:
            n_points = valid.sum(dim=1, keepdim=True)
        coords = (x[..., :N_STANDARDISED] - self.feat_mean) / self.feat_std
        h = self.feature_dropout(self.point_mlp(torch.cat([coords, x[..., 5:6]], dim=-1)))  # (B, P, width)

        mask = valid.unsqueeze(-1)
        n_valid = valid.sum(dim=1, keepdim=True)  # (B, 1)
        # Empty windows pool to zeros rather than negative infinity or NaN.
        pooled_max = h.masked_fill(~mask, float("-inf")).max(dim=1).values
        pooled_max = torch.where(n_valid > 0, pooled_max, torch.zeros_like(pooled_max))
        pooled_mean = (h * mask).sum(dim=1) / n_valid.clamp(min=1)
        count = torch.log1p(n_points.float()) / 5.0  # Roughly 0 to 1 for 0 to 150 points.
        return self.head(torch.cat([pooled_max, pooled_mean, count], dim=-1))


class PointNet(TorchModel):
    """Point-cloud window classifier with one string label per window."""

    net: MiniPointNet

    model_name = "pointnet"
    input_type = "pointcloud"
    preprocessing = "pointcloud_xyz_doppler_snr_time_v1"
    parameter_names = ("width", "dropout", "point_dropout")

    def __init__(
        self,
        width: int = 128,
        dropout: float = 0.2,
        point_dropout: float = 0.0,
    ) -> None:
        """Store architecture settings; the network is built when fitted or loaded."""
        super().__init__()
        self.width = width
        self.dropout = dropout
        self.point_dropout = point_dropout

    def _build_net(self, n_classes: int) -> MiniPointNet:
        """Construct a PointNet with one output score per class."""
        return MiniPointNet(n_classes, self.width, self.dropout, self.point_dropout)

    def _set_input_stats(self, x: np.ndarray) -> torch.Tensor:
        """Store training means and standard deviations for the five point features.

        Exclude padding and leave relative time and validity unchanged. Return the
        feature scales so augmentation noise can use standardised units.
        """
        points = x[x[..., 6] > 0.5][:, :N_STANDARDISED]
        if len(points):
            self.net.feat_mean.copy_(torch.as_tensor(points.mean(axis=0)))
            self.net.feat_std.copy_(torch.as_tensor(np.maximum(points.std(axis=0), 1e-6)))
        return self.net.feat_std
