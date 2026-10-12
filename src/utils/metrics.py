"""Classification metrics."""

from collections.abc import Sequence

import numpy as np

Labels = Sequence[str] | np.ndarray


def macro_f1(y_true: Labels, y_pred: Labels) -> float:
    """Computes the F1 score per true class and averages them.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        The unweighted mean F1 over the classes present in `y_true`.
    """
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in np.unique(true):
        tp = np.sum((true == c) & (pred == c))
        precision = tp / max(np.sum(pred == c), 1)
        recall = tp / np.sum(true == c)
        scores.append(0.0 if tp == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def accuracy(y_true: Labels, y_pred: Labels) -> float:
    """Computes the fraction of correct predictions.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy between 0 and 1.
    """
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def per_class(y_true: Labels, y_pred: Labels) -> dict[str, dict[str, float | int | None]]:
    """Precision, recall, F1 and support for each class."""
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    report: dict[str, dict[str, float | int | None]] = {}
    for c in np.unique(np.concatenate([true, pred])):
        tp = int(np.sum((true == c) & (pred == c)))
        n_true, n_pred = int(np.sum(true == c)), int(np.sum(pred == c))
        precision = tp / n_pred if n_pred else None
        recall = tp / n_true if n_true else None
        f1 = 2 * tp / (n_true + n_pred) if n_true + n_pred else None
        report[str(c)] = {"precision": precision, "recall": recall, "f1": f1, "support": n_true}
    return report
