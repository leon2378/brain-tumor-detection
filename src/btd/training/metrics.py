"""Framework-free evaluation metrics (numpy only) for image-level classification and segmentation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from btd.constants import NO_TUMOR


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Rows = ground truth, columns = prediction."""
    idx = {lbl: i for i, lbl in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[idx[t], idx[p]] += 1
    return cm


def classification_summary(cm: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    total = cm.sum()
    per_class = {
        lbl: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, lbl in enumerate(labels)
    }
    return {
        "accuracy": round(float(tp.sum() / total), 4) if total else 0.0,
        "macro_f1": round(float(f1[present].mean()), 4) if present.any() else 0.0,
        "balanced_accuracy": round(float(recall[present].mean()), 4) if present.any() else 0.0,
        "per_class": per_class,
    }


def tumor_screening(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, float]:
    """Binary view: 'any tumour' vs healthy — the clinically relevant miss/false-alarm trade-off."""
    t = np.array([y != NO_TUMOR for y in y_true])
    p = np.array([y != NO_TUMOR for y in y_pred])
    tp, tn = int((t & p).sum()), int((~t & ~p).sum())
    fp, fn = int((~t & p).sum()), int((t & ~p).sum())
    return {
        "sensitivity": round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "false_negatives": fn,
        "false_positives": fp,
    }


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, int, int]:
    """(dice, iou, intersection, union) for boolean masks. Empty-vs-empty counts as perfect."""
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    total = int(pred.sum()) + int(gt.sum())
    if union == 0:
        return 1.0, 1.0, 0, 0
    return 2.0 * inter / total, inter / union, inter, union


def decide(class_scores: np.ndarray, threshold: float, class_names: Sequence[str]) -> str:
    """Image label from per-class max detection scores (same rule as ``ImageDecision``)."""
    if class_scores.size == 0 or float(class_scores.max()) < threshold:
        return NO_TUMOR
    return class_names[int(class_scores.argmax())]


def tune_threshold(
    class_scores: np.ndarray,
    y_true: Sequence[str],
    class_names: Sequence[str],
    grid: Sequence[float] | None = None,
) -> tuple[float, list[dict[str, float]]]:
    """Pick the operating threshold maximising macro-F1 (ties → higher balanced accuracy, then higher τ)."""
    labels = [*class_names, NO_TUMOR]
    grid = list(grid) if grid is not None else [round(float(x), 2) for x in np.arange(0.05, 0.951, 0.05)]
    table: list[dict[str, float]] = []
    for thr in grid:
        y_pred = [decide(s, thr, class_names) for s in class_scores]
        summ = classification_summary(confusion_matrix(y_true, y_pred, labels), labels)
        screen = tumor_screening(y_true, y_pred)
        table.append(
            {
                "threshold": thr,
                "macro_f1": summ["macro_f1"],
                "balanced_accuracy": summ["balanced_accuracy"],
                "accuracy": summ["accuracy"],
                "sensitivity": screen["sensitivity"],
                "specificity": screen["specificity"],
            }
        )
    best = max(table, key=lambda r: (r["macro_f1"], r["balanced_accuracy"], r["threshold"]))
    return float(best["threshold"]), table


def latency_summary(values_ms: Sequence[float]) -> dict[str, float]:
    if not values_ms:
        return {}
    arr = np.asarray(values_ms, dtype=float)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "max_ms": round(float(arr.max()), 2),
    }
