import numpy as np
import pytest

from evaluation.metrics import accuracy, macro_f1, per_class, scores

CLASSES = ["lying", "not_lying"]


def test_perfect_prediction():
    y = np.array(["lying", "not_lying", "not_lying"])
    assert macro_f1(y, y, CLASSES) == 1.0


def test_majority_only_model_scores_half():
    # 99% accurate, but it never finds lying: macro F1 must expose that
    true = np.array(["lying"] + ["not_lying"] * 99)
    pred = np.array(["not_lying"] * 100)
    assert accuracy(true, pred) == pytest.approx(0.99)
    assert macro_f1(true, pred, CLASSES) == pytest.approx(0.4975, abs=1e-3)
    assert per_class(true, pred, CLASSES)["lying"]["recall"] == 0.0


def test_class_absent_and_never_predicted_is_undefined_not_perfect():
    true = pred = np.array(["not_lying"] * 5)
    row = per_class(true, pred, CLASSES)["lying"]
    assert row["f1"] is None and row["recall"] is None and row["precision"] is None
    # the old behaviour averaged only the present class; undefined classes are skipped, not counted as 1.0
    assert macro_f1(true, pred, CLASSES) == 1.0
    assert scores(true, pred, CLASSES)["lying_support"] == 0


def test_class_absent_but_predicted_scores_zero():
    true = np.array(["not_lying"] * 4)
    pred = np.array(["lying", "not_lying", "not_lying", "not_lying"])
    assert per_class(true, pred, CLASSES)["lying"]["f1"] == 0.0
    assert macro_f1(true, pred, CLASSES) < 0.9  # the false positive is penalised


def test_fixed_class_set_is_used_even_if_a_class_never_occurs():
    y = np.array(["a", "a"])
    assert set(per_class(y, y, ["a", "b", "c"])) == {"a", "b", "c"}


def test_macro_f1_undefined_when_nothing_is_defined():
    assert macro_f1(np.array([]), np.array([]), CLASSES) is None


def test_classes_default_to_union_of_true_and_predicted():
    true = np.array(["a", "a"])
    pred = np.array(["a", "b"])
    assert set(per_class(true, pred)) == {"a", "b"}
