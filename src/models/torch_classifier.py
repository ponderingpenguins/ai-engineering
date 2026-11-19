"""Shared training loop for the PyTorch classifiers (class weighting, validation loss, early stopping)."""

import copy
import logging
from abc import abstractmethod
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .model import BaseModel, History

logger = logging.getLogger(__name__)


def pick_device() -> torch.device:
    """Picks the fastest available device.

    Returns:
        CUDA if available, else MPS, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchClassifier(BaseModel):
    """A classifier on string labels trained with Adam on class-weighted cross-entropy.

    A subclass supplies the network (`_build_net`), may standardise its input (`_set_input_stats`) and
    stores its own hyperparameters and checkpoint state. Everything else lives here, so the models are
    trained and early-stopped in exactly the same way and their results are comparable.
    """

    net: nn.Module
    classes_: np.ndarray
    # applied to each training batch (never to validation or test data); a subclass may set it
    augment: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __init__(
        self,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 32,
        seed: int = 0,
        patience: int = 5,
        evals_per_epoch: int = 1,
    ) -> None:
        """Stores the training hyperparameters. The network is built in `fit`.

        Args:
            lr: Adam learning rate.
            weight_decay: Adam weight decay.
            epochs: Maximum number of passes over the training data.
            batch_size: Batch size for training and prediction.
            seed: Torch seed, set at the start of `fit`.
            patience: Stop after this many validation evaluations in a row without a lower
                validation loss. Only applies when `fit` gets validation data.
            evals_per_epoch: How often the validation loss is computed, per epoch.
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.patience = patience
        self.evals_per_epoch = evals_per_epoch
        self.device = pick_device()

    @abstractmethod
    def _build_net(self, n_classes: int) -> nn.Module:
        """A new, untrained network with `n_classes` outputs."""

    def _set_input_stats(self, x: np.ndarray) -> None:
        """Hook: set the network's input standardisation from the training windows. Does nothing by default."""

    def fit(self, x: np.ndarray, y: np.ndarray, val: tuple[np.ndarray, np.ndarray] | None = None,
            val_weight_labels: np.ndarray | None = None) -> History:
        """Trains a new network on windows of heatmaps.

        Classes are weighted by inverse frequency of the training windows (a balanced training set gets weight 1
        each). The validation loss is class-balanced, with weights from `val_weight_labels` (default: the labels of
        `val`). If `val` is given, the validation loss is computed
        `evals_per_epoch` times per epoch. Training stops when it has not improved for `patience`
        evaluations in a row, and the weights with the lowest validation loss are kept. Without `val`
        the network trains for all epochs and the last weights are kept.

        Args:
            x: Heatmaps of shape (N, T, 1, R, D).
            y: String labels of shape (N,).
            val: Optional (x, y) validation data. Must not contain test data.
            val_weight_labels: Labels whose class frequencies give the validation class weights. Tuning passes the
                pooled labels of all people in the training fold, so every validation person and every setting is
                scored with the same weights.

        Returns:
            The train loss per step and the validation loss per evaluation.
        """
        torch.manual_seed(self.seed)

        # labels -> integers 0..K-1, classes_ remembers which name is which
        self.classes_, y_int = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        self.net = self._build_net(n_classes).to(self.device)
        self._set_input_stats(x)  # standardize the input with statistics from the training data only

        # rare classes get a bigger weight, otherwise "always predict the common class" wins
        counts = np.bincount(y_int, minlength=n_classes)
        weights = torch.as_tensor(len(y_int) / (n_classes * counts), dtype=torch.float32).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        train_data = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y_int, dtype=torch.long))
        loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        steps_per_epoch = len(loader)
        eval_every = max(1, steps_per_epoch // self.evals_per_epoch)
        history = History(steps_per_epoch=steps_per_epoch)
        val_data = self._val_tensors(val)
        val_weights = None
        if val_data is not None:
            weight_y = val_data[1] if val_weight_labels is None else self._to_int(val_weight_labels)
            val_weights = self._balanced_weights(weight_y, n_classes)

        best_loss, best_state, stale_evals, step = float("inf"), None, 0, 0
        for epoch in tqdm(range(1, self.epochs + 1)):
            self.net.train()
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                if self.augment is not None:
                    xb = self.augment(xb)
                optimizer.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                optimizer.step()
                step += 1
                history.train_steps.append(step)
                history.train_loss.append(loss.item())

                if val_data is None or step % eval_every:
                    continue
                val_loss = self._val_loss(*val_data, val_weights)
                history.val_steps.append(step)
                history.val_loss.append(val_loss)
                self.net.train()  # _val_loss switched to eval mode
                if val_loss < best_loss:  # remember the best weights on validation data
                    best_loss, best_state, stale_evals = val_loss, copy.deepcopy(self.net.state_dict()), 0
                    history.best_step = step
                else:
                    stale_evals += 1
                if stale_evals >= self.patience:
                    history.stopped_early = True
                    break
            logger.info("epoch %d/%d step %d train_loss=%.4f val_loss=%s", epoch, self.epochs, step,
                        history.train_loss[-1], f"{history.val_loss[-1]:.4f}" if history.val_loss else "n/a")
            if history.stopped_early:
                logger.info("early stopping at step %d: no val loss improvement in %d evaluations (best step %s)",
                            step, self.patience, history.best_step)
                break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return history

    def _val_tensors(self, val: tuple[np.ndarray, np.ndarray] | None) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Validation data as tensors; windows of a class the training data never had are dropped."""
        if val is None or len(val[1]) == 0:
            return None
        xv, yv = val
        known = np.isin(yv, self.classes_)
        if not known.any():
            return None
        if not known.all():
            logger.warning("dropping %d validation windows of classes not in the training data", int((~known).sum()))
        y_int = np.searchsorted(self.classes_, yv[known])
        return torch.as_tensor(xv[known], dtype=torch.float32), torch.as_tensor(y_int, dtype=torch.long)

    def _to_int(self, y: np.ndarray) -> torch.Tensor:
        """Integer class ids of the labels the network knows; other labels are ignored."""
        y = np.asarray(y)
        return torch.as_tensor(np.searchsorted(self.classes_, y[np.isin(y, self.classes_)]), dtype=torch.long)

    def _balanced_weights(self, y_int: torch.Tensor, n_classes: int) -> torch.Tensor:
        """Inverse-frequency class weights from `y_int`; classes that do not occur get weight 0."""
        counts = torch.bincount(y_int, minlength=n_classes).float()
        present = counts > 0
        weights = torch.zeros(n_classes)
        weights[present] = len(y_int) / (int(present.sum()) * counts[present])
        return weights.to(self.device)

    @torch.no_grad()
    def _val_loss(self, xv: torch.Tensor, yv: torch.Tensor, weights: torch.Tensor) -> float:
        """Class-balanced cross-entropy over all validation windows.

        With weights that are the inverse class frequencies of the validation set itself, this is the mean of the
        per-class mean losses: a model that predicts 50/50 scores ln 2 whatever the class sizes. With weights
        pooled over many people, every person's loss is on one scale. Either way it does not depend on how the
        training data was weighted or resampled.
        """
        self.net.eval()
        total, weight_sum = 0.0, 0.0
        for start in range(0, len(yv), self.batch_size):
            xb = xv[start : start + self.batch_size].to(self.device)
            yb = yv[start : start + self.batch_size].to(self.device)
            total += nn.functional.cross_entropy(self.net(xb), yb, weight=weights, reduction="sum").item()
            weight_sum += weights[yb].sum().item()
        return total / weight_sum

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

