"""Fit individual models for cross-validation, tuning and final training.

Build batch augmentations, apply training-only resampling and run the PyTorch loop.
Record losses and restore the best validation weights when validation is available.
"""

from __future__ import annotations

import copy
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from config.models import ModelConfig
from config.training import TrainingConfig
from data.augment import HeatmapAugment, PointAugment
from data.resample import RandomOversampler
from models.model import BaseModel, TorchModel

logger = logging.getLogger(__name__)
BatchAugment = HeatmapAugment | PointAugment


@dataclass
class History:
    """Training losses, validation evaluations and the selected stopping point.

    Record training loss at every optimiser step and validation loss at the
    configured step interval. Validation uses class weights independently of
    the training class_weights setting.
    """

    steps_per_epoch: int
    train_steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_step: int | None = None  # Step with the lowest validation loss; keep these weights.
    stopped_early: bool = False  # True when patience triggers, including in the final epoch.

    def to_dict(self) -> dict[str, Any]:
        """Return plain lists, numbers and flags ready for JSON serialisation."""
        return asdict(self)


@dataclass
class FoldFit:
    """Training history, sampled training indices and elapsed fitting time."""

    history: History | None  # None for models that do not train iteratively
    train_idx: np.ndarray  # Actual training indices, including any oversampling repeats.
    seconds: float


def fit_fold(
    model: BaseModel,
    x: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    *,
    training: TrainingConfig | None = None,
    augment: BatchAugment | None = None,
    seed: int = 0,
    resampler: RandomOversampler | None = None,
    val_weight_labels: np.ndarray | None = None,
    log_as: str | None = None,
) -> FoldFit:
    """Fit one model on the supplied training indices, optionally with validation.

    Args:
        model: Model to fit in place.
        x: All model inputs, indexed by window.
        y: All string labels, aligned with x.
        fit_idx: Training-window indices; must exclude validation and test data.
        val_idx: Optional validation-window indices; must exclude test data.
        training: Loop settings, or default TrainingConfig values for neural models.
        augment: Optional transform applied to training batches only.
        seed: Seed used for neural-network fitting.
        resampler: Optional oversampler applied only to fit_idx.
        val_weight_labels: Label pool used to compute validation class weights.
        log_as: Optional prefix for the oversampling log message.

    Returns:
        History, actual training indices after resampling, and fitting time.
    """
    train_idx = fit_idx
    if resampler is not None:
        train_idx = fit_idx[resampler.indices(y[fit_idx])]
        if log_as:
            logger.info("%s: oversampled training windows %s -> %s", log_as,
                        dict(Counter(y[fit_idx].tolist())), dict(Counter(y[train_idx].tolist())))
    started = perf_counter()
    history = fit_model(model, x[train_idx], y[train_idx], training=training, augment=augment, seed=seed,
                        val=None if val_idx is None else (x[val_idx], y[val_idx]),
                        val_weight_labels=val_weight_labels)
    return FoldFit(history, train_idx, perf_counter() - started)


@dataclass
class FitConfig:
    """A fresh model, fitting settings, training augmentation and random seed."""

    model: BaseModel
    training: TrainingConfig | None = None
    augment: BatchAugment | None = None
    seed: int = 0


def build_augment(cfg: ModelConfig) -> BatchAugment | None:
    """Build training augmentation from the selected model's settings.

    Return None for the baseline or when all transforms are disabled.
    """
    if cfg.name == "baseline":
        return None
    augment_classes: dict[str, type[HeatmapAugment] | type[PointAugment]] = {
        "rnn": HeatmapAugment,
        "pointnet": PointAugment,
    }
    augment = augment_classes[cfg.name](**asdict(cfg.augment))
    return augment if augment.active else None


def fit_model(
    model: BaseModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    training: TrainingConfig | None = None,
    augment: BatchAugment | None = None,
    seed: int = 0,
    val: tuple[np.ndarray, np.ndarray] | None = None,
    val_weight_labels: np.ndarray | None = None,
) -> History | None:
    """Fit a neural model or call a simpler model's own fit method.

    Forward fitting settings to fit_torch for neural models. Other models
    receive inputs, labels and optional validation data directly.

    Returns:
        Training history for neural models, or the simple model's fit result.
    """
    if isinstance(model, TorchModel):
        return fit_torch(model, x, y, training=training, augment=augment, seed=seed,
                         val=val, val_weight_labels=val_weight_labels)
    return cast(Any, model).fit(x, y, val=val, val_weight_labels=val_weight_labels)


def _forward_train(model: TorchModel, xb: torch.Tensor, augment: BatchAugment | None) -> torch.Tensor:
    """Augment a training batch and return its class scores.

    For PointNet, pass the count before random point dropping so its count
    feature does not change merely because augmentation removed points.
    """
    if isinstance(augment, PointAugment):
        xb, n_points = augment.apply_with_count(xb)
        return model.net(xb, n_points=n_points)
    return model.net(xb if augment is None else augment(xb))


def fit_torch(
    model: TorchModel,
    x: np.ndarray,
    y: np.ndarray,
    *,
    training: TrainingConfig | None = None,
    augment: BatchAugment | None = None,
    seed: int = 0,
    val: tuple[np.ndarray, np.ndarray] | None = None,
    val_weight_labels: np.ndarray | None = None,
) -> History:
    """Build and train a fresh neural network, recording its loss history.

    With usable validation data, keep the weights with the lowest weighted
    validation loss and stop after patience evaluations without improvement.
    Otherwise, train for all epochs and keep the final weights.

    Args:
        model: Neural model to initialise and fit in place.
        x: Training heatmaps (N, T, 1, R, D) or point clouds (N, P, 7).
        y: Training string labels of shape (N,).
        training: Optimiser and loop settings; None uses TrainingConfig defaults.
            class_weights controls inverse-frequency training loss only.
        augment: Optional transform for training batches before standardisation.
        seed: PyTorch random seed for initialisation and training.
        val: Optional (inputs, labels) validation pair, excluding test data.
            Labels absent from training are dropped from loss evaluation.
        val_weight_labels: Labels used to compute inverse-frequency validation
            weights; None uses validation labels. Tuning passes all outer-training
            labels to use the same weight source for people and candidates.

    Returns:
        Training loss per step, validation loss per evaluation, best step and
        early-stopping status. Input statistics use training windows only.
    """
    training = training if training is not None else TrainingConfig()
    torch.manual_seed(seed)

    # Encode labels as integers 0..K-1 and retain their names for prediction.
    model.classes_, y_int = np.unique(y, return_inverse=True)
    n_classes = len(model.classes_)

    model.net = model._build_net(n_classes).to(model.device)
    std = model._set_input_stats(x)  # Estimate input statistics from training data only.
    if augment is not None and std is not None:
        augment.set_input_std(std)

    # Optional inverse-frequency weights give rare training classes more influence.
    counts = np.bincount(y_int, minlength=n_classes)
    weights = torch.as_tensor(len(y_int) / (n_classes * counts), dtype=torch.float32).to(model.device)
    loss_fn = nn.CrossEntropyLoss(weight=weights if training.class_weights else None)
    optimizer = torch.optim.Adam(model.net.parameters(), lr=training.lr, weight_decay=training.weight_decay)

    train_data = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y_int, dtype=torch.long))
    loader = DataLoader(train_data, batch_size=training.batch_size, shuffle=True)
    steps_per_epoch = len(loader)
    eval_every = max(1, steps_per_epoch // training.evals_per_epoch)
    history = History(steps_per_epoch=steps_per_epoch)
    val_data = _val_tensors(model, val)
    val_weights = None
    if val_data is not None:
        weight_y = val_data[1] if val_weight_labels is None else _to_int(model, val_weight_labels)
        val_weights = _balanced_weights(weight_y, n_classes, model.device)

    best_loss, best_state, stale_evals, step = float("inf"), None, 0, 0
    for epoch in tqdm(range(1, training.epochs + 1)):
        model.net.train()
        for xb, yb in loader:
            xb, yb = xb.to(model.device), yb.to(model.device)
            optimizer.zero_grad()
            loss = loss_fn(_forward_train(model, xb, augment), yb)
            loss.backward()
            optimizer.step()
            step += 1
            history.train_steps.append(step)
            history.train_loss.append(loss.item())

            if val_data is None or step % eval_every:
                continue
            assert val_weights is not None
            val_loss = _val_loss(model, val_data[0], val_data[1], val_weights, training.batch_size)
            history.val_steps.append(step)
            history.val_loss.append(val_loss)
            model.net.train()  # Resume training mode after validation.
            if val_loss < best_loss:  # Keep the weights with the lowest validation loss.
                best_loss, best_state, stale_evals = val_loss, copy.deepcopy(model.net.state_dict()), 0
                history.best_step = step
            else:
                stale_evals += 1
            if stale_evals >= training.patience:
                history.stopped_early = True
                break
        logger.info("epoch %d/%d step %d train_loss=%.4f val_loss=%s", epoch, training.epochs, step,
                    history.train_loss[-1], f"{history.val_loss[-1]:.4f}" if history.val_loss else "n/a")
        if history.stopped_early:
            logger.info("early stopping at step %d: no val loss improvement in %d evaluations (best step %s)",
                        step, training.patience, history.best_step)
            break

    if best_state is not None:
        model.net.load_state_dict(best_state)
    model.net.eval()
    return history


def _val_tensors(model: TorchModel, val: tuple[np.ndarray, np.ndarray] | None) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Convert validation windows to tensors, dropping labels unknown to the model.

    Return None when no validation windows remain.
    """
    if val is None or len(val[1]) == 0:
        return None
    xv, yv = val
    known = np.isin(yv, model.classes_)
    if not known.any():
        return None
    if not known.all():
        logger.warning("dropping %d validation windows of classes not in the training data", int((~known).sum()))
    y_int = np.searchsorted(model.classes_, yv[known])
    return torch.as_tensor(xv[known], dtype=torch.float32), torch.as_tensor(y_int, dtype=torch.long)


def _to_int(model: TorchModel, y: np.ndarray) -> torch.Tensor:
    """Encode known labels as integer class IDs, ignoring labels absent from training."""
    y = np.asarray(y)
    return torch.as_tensor(np.searchsorted(model.classes_, y[np.isin(y, model.classes_)]), dtype=torch.long)


def _balanced_weights(y_int: torch.Tensor, n_classes: int, device: torch.device) -> torch.Tensor:
    """Compute inverse-frequency class weights, using zero for absent classes."""
    counts = torch.bincount(y_int, minlength=n_classes).float()
    present = counts > 0
    weights = torch.zeros(n_classes)
    weights[present] = len(y_int) / (int(present.sum()) * counts[present])
    return weights.to(device)


@torch.no_grad()
def _val_loss(model: TorchModel, xv: torch.Tensor, yv: torch.Tensor, weights: torch.Tensor,
              batch_size: int) -> float:
    """Compute weighted cross-entropy over all validation batches.

    Divide summed weighted loss by the total weight of the observed labels,
    rather than averaging batch means. With weights from validation itself,
    this equals the mean of the observed classes' mean losses.
    """
    model.net.eval()
    total, weight_sum = 0.0, 0.0
    for start in range(0, len(yv), batch_size):
        xb = xv[start : start + batch_size].to(model.device)
        yb = yv[start : start + batch_size].to(model.device)
        total += nn.functional.cross_entropy(model.net(xb), yb, weight=weights, reduction="sum").item()
        weight_sum += weights[yb].sum().item()
    return total / weight_sum

