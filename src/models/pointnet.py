"""A mini PointNet on the radar point cloud of a window, instead of on the range-Doppler heatmaps."""

from typing import Any, Self

import numpy as np
import torch
from torch import nn

from data.augment import PointAugment

from .torch_classifier import TorchClassifier

N_STANDARDISED = 5  # x, y, z, doppler, snr are standardised; time and the validity flag are used as they are


class MiniPointNet(nn.Module):
    """PointNet without the alignment networks: a shared MLP on every point, then a symmetric pooling.

    Input x has shape (B, P, 7): per point x, y, z, doppler, snr, time inside the window, and a validity
    flag (1 for a real point, 0 for padding; see `data.windows._window_points`). The order of the points
    does not matter, and padding never reaches the pooled features. Pooling is max and mean over the valid
    points, plus the log of how many there are (lying windows tend to contain more points).
    """

    feat_mean: torch.Tensor  # buffers, declared so type checkers know they are tensors
    feat_std: torch.Tensor

    def __init__(self, n_classes: int, width: int = 128, dropout: float = 0.2, point_dropout: float = 0.0,
                 point_drop: float = 0.0, jitter: float = 0.0) -> None:
        super().__init__()
        self.point_drop = point_drop
        self.jitter = jitter
        self.feature_dropout = nn.Dropout(point_dropout)  # on the per-point features, before pooling
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class scores of shape (B, n_classes)."""
        valid = x[..., 6] > 0.5  # (B, P)
        n_points = valid.sum(dim=1, keepdim=True)  # real points, counted before augmentation drops any
        coords = (x[..., :N_STANDARDISED] - self.feat_mean) / self.feat_std
        if self.training:  # augmentation, training only: drop random points and jitter the rest
            if self.point_drop > 0:
                valid = valid & (torch.rand_like(x[..., 6]) >= self.point_drop)
            if self.jitter > 0:
                coords = coords + self.jitter * torch.randn_like(coords)
        h = self.feature_dropout(self.point_mlp(torch.cat([coords, x[..., 5:6]], dim=-1)))  # (B, P, width)

        mask = valid.unsqueeze(-1)
        n_valid = valid.sum(dim=1, keepdim=True)  # (B, 1)
        # a window without any point pools to zeros instead of -inf / nan
        pooled_max = h.masked_fill(~mask, float("-inf")).max(dim=1).values
        pooled_max = torch.where(n_valid > 0, pooled_max, torch.zeros_like(pooled_max))
        pooled_mean = (h * mask).sum(dim=1) / n_valid.clamp(min=1)
        count = torch.log1p(n_points.float()) / 5.0  # about 0 to 1 for 0 to 150 points
        return self.head(torch.cat([pooled_max, pooled_mean, count], dim=-1))


class PointNet(TorchClassifier):
    """Mini PointNet classifier on windows of radar points. x: (N, P, 7), y: (N,) string labels."""

    model_name = "pointnet"
    input_type = "pointcloud"
    preprocessing = "pointcloud_xyz_doppler_snr_time_v1"
    parameter_names = ("width", "dropout", "point_dropout", "point_drop", "jitter", "lr", "weight_decay", "epochs",
                       "batch_size", "seed", "patience", "evals_per_epoch", "augment")

    def __init__(
        self,
        width: int = 128,
        dropout: float = 0.2,
        point_dropout: float = 0.0,
        point_drop: float = 0.0,
        jitter: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 32,
        seed: int = 0,
        patience: int = 5,
        evals_per_epoch: int = 1,
        augment: dict[str, Any] | None = None,
    ) -> None:
        """Stores the hyperparameters. The network is built in `fit`.

        Args:
            width: Number of features per point after the shared MLP (the pooled vector has twice this).
            dropout: Dropout in the classifier head.
            point_dropout: Dropout on the per-point features before pooling.
            point_drop: Probability of dropping each point while training (augmentation).
            jitter: Std of Gaussian noise added to the standardised point features while training (augmentation).
            lr, weight_decay, epochs, batch_size, seed, patience, evals_per_epoch: see `TorchClassifier`.
            augment: Settings of `data.augment.PointAugment` (moves, rotations, ghost points), applied to training
                batches only. None or all off means no geometric augmentation.
        """
        super().__init__(lr, weight_decay, epochs, batch_size, seed, patience, evals_per_epoch)
        self._set_augment(augment, PointAugment)
        self.width = width
        self.dropout = dropout
        self.point_dropout = point_dropout
        self.point_drop = point_drop
        self.jitter = jitter

    def _build_net(self, n_classes: int) -> nn.Module:
        return MiniPointNet(n_classes, self.width, self.dropout, self.point_dropout, self.point_drop, self.jitter)

    def _set_input_stats(self, x: np.ndarray) -> None:
        """Standardise x, y, z, doppler and snr with the mean and std of the training points (padding excluded)."""
        points = x[x[..., 6] > 0.5][:, :N_STANDARDISED]
        if len(points) == 0:
            return
        self.net.feat_mean.copy_(torch.as_tensor(points.mean(axis=0)))  # type: ignore[union-attr]
        self.net.feat_std.copy_(torch.as_tensor(np.maximum(points.std(axis=0), 1e-6)))  # type: ignore[union-attr]
