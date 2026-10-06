"""Classification metrics."""

import numpy as np


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for c in np.unique(true):
        tp = np.sum((true == c) & (pred == c))
        precision = tp / max(np.sum(pred == c), 1)
        recall = tp / np.sum(true == c)
        scores.append(0.0 if tp == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores))


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
