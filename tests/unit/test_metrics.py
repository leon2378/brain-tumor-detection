from __future__ import annotations

import numpy as np
import pytest

from btd.training.metrics import (
    classification_summary,
    confusion_matrix,
    decide,
    dice_iou,
    latency_summary,
    tumor_screening,
    tune_threshold,
)

LABELS = ["glioma", "meningioma", "pituitary", "no_tumor"]


def test_confusion_and_summary() -> None:
    y_true = ["glioma", "glioma", "meningioma", "no_tumor", "pituitary", "no_tumor"]
    y_pred = ["glioma", "meningioma", "meningioma", "no_tumor", "pituitary", "glioma"]
    cm = confusion_matrix(y_true, y_pred, LABELS)
    assert cm.sum() == 6
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[3, 0] == 1
    s = classification_summary(cm, LABELS)
    assert s["accuracy"] == pytest.approx(4 / 6, abs=1e-4)
    assert s["per_class"]["pituitary"]["f1"] == 1.0
    assert 0 < s["macro_f1"] < 1


def test_screening() -> None:
    s = tumor_screening(
        ["glioma", "no_tumor", "no_tumor", "pituitary"], ["no_tumor", "no_tumor", "glioma", "pituitary"]
    )
    assert s["sensitivity"] == 0.5 and s["specificity"] == 0.5
    assert s["false_negatives"] == 1 and s["false_positives"] == 1


def test_dice_iou() -> None:
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    assert dice_iou(a, b)[:2] == (1.0, 1.0)
    a[:5] = True
    b[:5, :5] = True
    dice, iou, inter, union = dice_iou(a, b)
    assert inter == 25 and union == 50
    assert iou == 0.5 and dice == pytest.approx(2 * 25 / 75)


def test_decide_and_tune() -> None:
    names = ["glioma", "meningioma", "pituitary"]
    assert decide(np.array([0.1, 0.2, 0.05]), 0.5, names) == "no_tumor"
    assert decide(np.array([0.1, 0.7, 0.05]), 0.5, names) == "meningioma"
    scores = np.array([[0.9, 0, 0], [0.0, 0.35, 0], [0.3, 0, 0], [0.1, 0, 0], [0, 0, 0.8]])
    y_true = ["glioma", "meningioma", "no_tumor", "no_tumor", "pituitary"]
    thr, table = tune_threshold(scores, y_true, names)
    assert 0.3 < thr <= 0.35  # keeps the 0.35 meningioma, rejects the 0.3 false positive
    assert max(r["macro_f1"] for r in table) == 1.0


def test_latency_summary() -> None:
    s = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert s["p50_ms"] == 2.5 and s["max_ms"] == 4.0
    assert latency_summary([]) == {}
