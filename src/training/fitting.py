"""Fits one model on the windows of one fold. The outer loop and the tuning loop both use it."""

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

from config import ModelConfig, TrainingConfig
from data.augment import HeatmapAugment, PointAugment
from data.resample import RandomOversampler
from models.model import BaseModel, TorchModel

logger = logging.getLogger(__name__)
BatchAugment = HeatmapAugment | PointAugment


@dataclass
class History:
    """Losses recorded while training one model.

    The train loss is recorded at every optimiser step, the validation loss only every few steps.
    Validation loss is class-balanced; training weighting follows TrainingConfig.
    """

    steps_per_epoch: int
    train_steps: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_step: int | None = None  # step of the lowest validation loss; its weights are the ones kept
    stopped_early: bool = False  # True if patience ran out before the last epoch

    def to_dict(self) -> dict[str, Any]:
        """Plain lists and numbers, ready for JSON."""
        return asdict(self)


@dataclass
class FoldFit:
    """What `fit_fold` did."""

    history: History | None  # None for models that do not train iteratively
    train_idx: np.ndarray  # the windows the model was trained on, after resampling
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
    """Trains `model` on the windows `fit_idx`, with `val_idx` as validation data if given.

    Only the training windows are resampled. Validation windows keep their real class balance, and test
    windows are never passed in. `val_weight_labels` sets the validation class weights (see `fit_torch`).
    `log_as` names the log line about oversampling (for example "fold 2/4"); None logs nothing.
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
    """A fresh model and the settings needed to fit one tuning candidate."""

    model: BaseModel
    training: TrainingConfig | None = None
    augment: BatchAugment | None = None
    seed: int = 0


def build_augment(cfg: ModelConfig) -> BatchAugment | None:
    """Build batch augmentation separately from the model architecture."""
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
    """Fit a neural model, or delegate to a model's own simple fitting method."""
    if isinstance(model, TorchModel):
        return fit_torch(model, x, y, training=training, augment=augment, seed=seed,
                         val=val, val_weight_labels=val_weight_labels)
    return cast(Any, model).fit(x, y, val=val, val_weight_labels=val_weight_labels)


def _forward_train(model: TorchModel, xb: torch.Tensor, augment: BatchAugment | None) -> torch.Tensor:
    """Keep the pre-drop point count when augmenting PointNet training batches."""
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
    """Train a fresh neural network on heatmap or point-cloud windows.

    Classes are weighted by inverse frequency of the training windows (a balanced training set gets weight 1
    each). The validation loss is class-balanced, with weights from `val_weight_labels` (default: the labels of
    `val`). If `val` is given, the validation loss is computed
    `evals_per_epoch` times per epoch. Training stops when it has not improved for `patience`
    evaluations in a row, and the weights with the lowest validation loss are kept. Without `val`
    the network trains for all epochs and the last weights are kept.

    Args:
        x: Heatmaps (N, T, 1, R, D) or point clouds (N, P, 7).
        y: String labels of shape (N,).
        val: Optional (x, y) validation data. Must not contain test data.
        val_weight_labels: Labels whose class frequencies give the validation class weights. Tuning passes the
            pooled labels of all people in the training fold, so every validation person and every setting is
            scored with the same weights.

    Returns:
        The train loss per step and the validation loss per evaluation.
    """
    training = training if training is not None else TrainingConfig()
    torch.manual_seed(seed)

    # labels -> integers 0..K-1, classes_ remembers which name is which
    model.classes_, y_int = np.unique(y, return_inverse=True)
    n_classes = len(model.classes_)

    model.net = model._build_net(n_classes).to(model.device)
    std = model._set_input_stats(x)  # training data only
    if augment is not None and std is not None:
        augment.set_input_std(std)

    # rare classes get a bigger weight, otherwise "always predict the common class" wins
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
            model.net.train()  # _val_loss switched to eval mode
            if val_loss < best_loss:  # remember the best weights on validation data
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
    """Validation data as tensors; windows of a class the training data never had are dropped."""
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
    """Integer class ids of the labels the network knows; other labels are ignored."""
    y = np.asarray(y)
    return torch.as_tensor(np.searchsorted(model.classes_, y[np.isin(y, model.classes_)]), dtype=torch.long)


def _balanced_weights(y_int: torch.Tensor, n_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights from `y_int`; classes that do not occur get weight 0."""
    counts = torch.bincount(y_int, minlength=n_classes).float()
    present = counts > 0
    weights = torch.zeros(n_classes)
    weights[present] = len(y_int) / (int(present.sum()) * counts[present])
    return weights.to(device)


@torch.no_grad()
def _val_loss(model: TorchModel, xv: torch.Tensor, yv: torch.Tensor, weights: torch.Tensor,
              batch_size: int) -> float:
    """Class-balanced cross-entropy over all validation windows.

    With weights that are the inverse class frequencies of the validation set itself, this is the mean of the
    per-class mean losses: a model that predicts 50/50 scores ln 2 whatever the class sizes. With weights
    pooled over many people, every person's loss is on one scale. Either way it does not depend on how the
    training data was weighted or resampled.
    """
    model.net.eval()
    total, weight_sum = 0.0, 0.0
    for start in range(0, len(yv), batch_size):
        xb = xv[start : start + batch_size].to(model.device)
        yb = yv[start : start + batch_size].to(model.device)
        total += nn.functional.cross_entropy(model.net(xb), yb, weight=weights, reduction="sum").item()
        weight_sum += weights[yb].sum().item()
    return total / weight_sum

