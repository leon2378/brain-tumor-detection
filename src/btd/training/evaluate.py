"""Evaluation.

Two complementary views:

* :func:`ultralytics_metrics` — standard COCO-style box/mask mAP from Ultralytics (works for ``.pt`` and exported
  ONNX files, so quantised models are scored with the same evaluator).
* :func:`evaluate_engine` — what the *deployed* service actually does: the torch-free ONNX engine is run on every
  slice and scored on image-level classification (4 classes incl. no tumour), tumour screening
  sensitivity/specificity, and pixel-level Dice/IoU of the predicted tumour region against expert masks.
"""

from __future__ import annotations

import csv
import logging
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from btd.constants import NO_TUMOR
from btd.training.metrics import (
    classification_summary,
    confusion_matrix,
    decide,
    dice_iou,
    latency_summary,
    tumor_screening,
    tune_threshold,
)
from btd.utils import imread, imread_gray, utc_now

LOGGER = logging.getLogger(__name__)
MIN_EVAL_CONF = 0.05  # lowest threshold considered when tuning; detections below it are never shown


@dataclass(frozen=True)
class Sample:
    key: str
    label: str
    image: Path
    mask: Path | None


def load_split(dataset_dir: str | Path, split: str, limit: int | None = None) -> list[Sample]:
    root = Path(dataset_dir)
    with (root / "splits.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    samples = []
    for r in rows[:limit] if limit else rows:
        mask = root / "masks" / split / f"{r['key']}.png"
        samples.append(Sample(r["key"], r["label"], root / r["image"], mask if mask.is_file() else None))
    if not samples:
        raise ValueError(f"No samples for split {split!r} in {root / 'splits.csv'}")
    return samples


def run_engine(engine: Any, samples: Sequence[Sample]) -> list[dict[str, Any]]:
    """Predict every sample once at ``MIN_EVAL_CONF`` (masks included) — later thresholds are applied offline."""
    out = []
    names = engine.class_names
    for i, s in enumerate(samples):
        img = imread(s.image)
        pred = engine.predict(img, conf=MIN_EVAL_CONF)
        scores = np.zeros(len(names), dtype=np.float32)
        for d in pred.detections:
            scores[d.class_id] = max(scores[d.class_id], d.confidence)
        out.append({"sample": s, "pred": pred, "scores": scores})
        if (i + 1) % 200 == 0:
            LOGGER.info("  %d/%d", i + 1, len(samples))
    return out


@dataclass
class _SegAccumulator:
    dice: list[float] = field(default_factory=list)
    iou: list[float] = field(default_factory=list)
    inter: int = 0
    union: int = 0


def segmentation_scores(results: Sequence[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Dice/IoU of the union of predicted masks (conf ≥ threshold) vs the expert mask, on tumour slices."""
    per_class: dict[str, _SegAccumulator] = {}
    for r in results:
        s: Sample = r["sample"]
        if s.mask is None or s.label == NO_TUMOR:
            continue
        gt = imread_gray(s.mask) > 127
        pred_mask = np.zeros(gt.shape, dtype=bool)
        for d in r["pred"].detections:
            if d.confidence >= threshold and d.mask is not None and d.mask.size:
                x0, y0 = d.mask_origin
                h, w = d.mask.shape
                pred_mask[y0 : y0 + h, x0 : x0 + w] |= d.mask
        dice, iou, inter, union = dice_iou(pred_mask, gt)
        acc = per_class.setdefault(s.label, _SegAccumulator())
        acc.dice.append(dice)
        acc.iou.append(iou)
        acc.inter += inter
        acc.union += union

    classes: dict[str, dict[str, Any]] = {}
    for label, acc in sorted(per_class.items()):
        classes[label] = {
            "images": len(acc.dice),
            "mean_dice": round(float(np.mean(acc.dice)), 4),
            "mean_iou": round(float(np.mean(acc.iou)), 4),
            "dataset_iou": round(acc.inter / max(acc.union, 1), 4),
        }
    all_dice = [d for acc in per_class.values() for d in acc.dice]
    all_iou = [v for acc in per_class.values() for v in acc.iou]
    n = len(all_dice)
    return {
        "images": n,
        "mean_dice": round(float(np.mean(all_dice)), 4) if n else None,
        "mean_iou": round(float(np.mean(all_iou)), 4) if n else None,
        "weighted_dataset_iou": round(sum(c["dataset_iou"] * c["images"] for c in classes.values()) / n, 4)
        if n
        else None,
        "per_class": classes,
    }


def evaluate_engine(
    model_path: str | Path,
    dataset_dir: str | Path,
    split: str = "test",
    threshold: float | None = None,
    tune: bool = False,
    providers: str = "auto",
    limit: int | None = None,
) -> dict[str, Any]:
    """Score the deployable ONNX engine on a split. With ``tune=True`` the operating threshold is chosen on this
    split (only ever do that on *val*)."""
    from btd.inference.engine import SegmentationEngine

    engine = SegmentationEngine(model_path, providers=providers, verify_checksum=False)
    samples = load_split(dataset_dir, split, limit)
    LOGGER.info(
        "Evaluating %s on %d %s slices (%s)", Path(model_path).name, len(samples), split, engine.providers[0]
    )
    engine.warmup()
    results = run_engine(engine, samples)
    names = list(engine.class_names)
    labels = [*names, NO_TUMOR]
    y_true = [r["sample"].label for r in results]
    scores = np.stack([r["scores"] for r in results])

    sweep: list[dict[str, float]] | None = None
    if tune:
        thr, sweep = tune_threshold(scores, y_true, names)
    else:
        thr = float(threshold if threshold is not None else engine.conf_threshold)
    y_pred = [decide(s, thr, names) for s in scores]
    cm = confusion_matrix(y_true, y_pred, labels)

    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "model": Path(model_path).name,
        "precision": engine.precision,
        "provider": engine.providers[0],
        "split": split,
        "images": len(samples),
        "threshold": thr,
        "threshold_tuned_on_this_split": tune,
        "classification": classification_summary(cm, labels),
        "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
        "screening": tumor_screening(y_true, y_pred),
        "segmentation": segmentation_scores(results, thr),
        "latency": {
            "total": latency_summary([r["pred"].timings_ms["total_ms"] for r in results]),
            "inference": latency_summary([r["pred"].timings_ms["inference_ms"] for r in results]),
        },
    }
    if sweep is not None:
        report["threshold_sweep"] = sweep
    return report


def ultralytics_metrics(
    model_path: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int = 640,
    nms: bool | None = None,
    device: str | int | None = None,
) -> dict[str, Any]:
    """Box/mask mAP via Ultralytics. ``nms=False`` scores YOLO26's NMS-free head for ``.pt`` weights."""
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="segment")
    kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": split,
        "imgsz": imgsz,
        "conf": 0.001,
        "iou": 0.7,
        "plots": False,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device
    if str(model_path).endswith(".onnx"):
        kwargs["batch"] = 1
    if nms is not None:
        kwargs["nms"] = nms
    # Ultralytics creates a save_dir even with plots=False; point it at a temp dir so runs/segment/ doesn't
    # collect an empty val, val-2, ... folder on every call.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        r = model.val(**kwargs, project=tmp, name="val", exist_ok=True)
    names = [r.names[i] for i in sorted(r.names)] if isinstance(r.names, dict) else list(r.names)
    seg_maps = getattr(r.seg, "maps", None)
    return {
        "box_map50": round(float(r.box.map50), 4),
        "box_map50_95": round(float(r.box.map), 4),
        "mask_map50": round(float(r.seg.map50), 4),
        "mask_map50_95": round(float(r.seg.map), 4),
        "box_precision": round(float(r.box.mp), 4),
        "box_recall": round(float(r.box.mr), 4),
        "per_class_mask_map50_95": {n: round(float(seg_maps[i]), 4) for i, n in enumerate(names)}
        if seg_maps is not None
        else {},
    }


def render_markdown(reports: Sequence[dict[str, Any]], title: str = "Evaluation") -> str:
    lines = [f"# {title}", ""]
    lines += [
        "| model | precision | split | acc | macro-F1 | sensitivity | specificity | Dice | IoU (dataset) | p50 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        c, s, seg = r["classification"], r["screening"], r["segmentation"]
        lines.append(
            f"| {r['model']} | {r['precision']} | {r['split']} | {c['accuracy']:.4f} | {c['macro_f1']:.4f} | "
            f"{s['sensitivity']:.4f} | {s['specificity']:.4f} | {seg['mean_dice'] or 0:.4f} | "
            f"{seg['weighted_dataset_iou'] or 0:.4f} | {r['latency']['total'].get('p50_ms', 0):.1f} |"
        )
    if reports:
        r = reports[0]
        lines += ["", f"Confusion matrix ({r['model']}, rows = truth):", ""]
        labels = r["confusion_matrix"]["labels"]
        lines += ["| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
        for lbl, row in zip(labels, r["confusion_matrix"]["matrix"], strict=True):
            lines.append(f"| **{lbl}** | " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def plot_confusion_matrix(report: dict[str, Any], path: str | Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a training extra
        return None
    labels = report["confusion_matrix"]["labels"]
    cm = np.array(report["confusion_matrix"]["matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if norm[i, j] > 0.5 else "black",
            )
    ax.set_title(f"{report['model']} · {report['split']}")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return Path(path)
