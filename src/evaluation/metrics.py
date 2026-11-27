"""Score classification predictions against a consistent set of labels.

Compute accuracy, macro F1 and per-class precision, recall, F1 and support.
Undefined precision, recall or F1 values are None; support is always a count.
"""

from collections.abc import Sequence

import numpy as np

Labels = Sequence[str] | np.ndarray


def _classes(true: np.ndarray, pred: np.ndarray, classes: Labels | None) -> list[str]:
    """Return the supplied class list, or the union of true and predicted labels."""
    if classes is None:
        classes = np.unique(np.concatenate([true, pred]))
    return [str(c) for c in classes]


def per_class(y_true: Labels, y_pred: Labels, classes: Labels | None = None) -> dict[str, dict[str, float | int | None]]:
    """Compute precision, recall, F1 and support for each requested class.

    Args:
        y_true: True labels of shape (N,).
        y_pred: Predicted labels aligned with y_true.
        classes: Fixed class list, or None to use all labels in truth and predictions.

    Returns:
        Metrics by class. Support counts true examples. Precision is None with
        no predictions; recall is None with no true examples. F1 is None only
        when the class is absent from both truth and predictions.
    """
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    report: dict[str, dict[str, float | int | None]] = {}
    for c in _classes(true, pred, classes):
        tp = int(np.sum((true == c) & (pred == c)))
        n_true, n_pred = int(np.sum(true == c)), int(np.sum(pred == c))
        report[c] = {
            "precision": tp / n_pred if n_pred else None,
            "recall": tp / n_true if n_true else None,
            "f1": 2 * tp / (n_true + n_pred) if n_true + n_pred else None,
            "support": n_true,
        }
    return report


def macro_f1(y_true: Labels, y_pred: Labels, classes: Labels | None = None) -> float | None:
    """Average the class F1 scores that are defined.

    Args:
        y_true: True labels of shape (N,).
        y_pred: Predicted labels aligned with y_true.
        classes: Fixed class list, or None to use all labels in truth and predictions.

    Returns:
        The unweighted mean of defined F1 scores, or None if none are defined.
    """
    f1s = [row["f1"] for row in per_class(y_true, y_pred, classes).values() if row["f1"] is not None]
    return float(np.mean(f1s)) if f1s else None


def accuracy(y_true: Labels, y_pred: Labels, classes: Labels | None = None) -> float:
    """Return the fraction of matching labels for aligned, non-empty input arrays.

    classes is accepted for the shared metric interface but does not affect accuracy.
    """
    del classes  # accuracy does not depend on the class set
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def scores(y_true: Labels, y_pred: Labels, classes: Labels) -> dict[str, float | int | None]:
    """Return accuracy, macro F1 and flattened per-class precision, recall and support.

    Per-class F1 is available through per_class; it is not included in this mapping.
    """
    result: dict[str, float | int | None] = {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, classes),
    }
    for label, row in per_class(y_true, y_pred, classes).items():
        result[f"{label}_precision"] = row["precision"]
        result[f"{label}_recall"] = row["recall"]
        result[f"{label}_support"] = row["support"]
    return result


def fmt(value: float | None, digits: int = 3) -> str:
    """Format a metric with fixed decimal places, or N/A when its value is None."""
    return "N/A" if value is None else f"{value:.{digits}f}"
