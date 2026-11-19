"""Classification metrics.

Every metric that depends on the set of classes takes it explicitly, so a fold that happens to hold no
example of a rare class is scored against the same classes as every other fold.

For one class, F1 = 2 * TP / (n_true + n_pred). It is undefined (None) when the class is neither in the
truth nor in the predictions. A class that is predicted but absent from the truth scores 0, because every
prediction of it is a false positive. Macro F1 is the mean over the classes where F1 is defined.
"""

from collections.abc import Sequence

import numpy as np

Labels = Sequence[str] | np.ndarray


def _classes(true: np.ndarray, pred: np.ndarray, classes: Labels | None) -> list[str]:
    """The fixed class set, or every class seen in `true` or `pred` if none is given."""
    if classes is None:
        classes = np.unique(np.concatenate([true, pred]))
    return [str(c) for c in classes]


def per_class(y_true: Labels, y_pred: Labels, classes: Labels | None = None) -> dict[str, dict[str, float | int | None]]:
    """Precision, recall, F1 and support for each class; None where the ratio is undefined."""
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
    """Mean F1 over the classes where it is defined.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        classes: The fixed class set. Defaults to every class seen in `y_true` or `y_pred`.

    Returns:
        The macro F1, or None if no class has a defined F1.
    """
    f1s = [row["f1"] for row in per_class(y_true, y_pred, classes).values() if row["f1"] is not None]
    return float(np.mean(f1s)) if f1s else None


def accuracy(y_true: Labels, y_pred: Labels, classes: Labels | None = None) -> float:
    """Computes the fraction of correct predictions. `classes` is accepted for a uniform signature."""
    del classes  # accuracy does not depend on the class set
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def scores(y_true: Labels, y_pred: Labels, classes: Labels) -> dict[str, float | int | None]:
    """Accuracy, macro F1 and per-class precision, recall and support, flattened into one dict."""
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
    """A metric for logs and reports: fixed decimals, or N/A when the metric is undefined."""
    return "N/A" if value is None else f"{value:.{digits}f}"
