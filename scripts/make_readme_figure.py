"""Build the README figure: typical predictions of the released model on the BRISC 2025 test split.

    python scripts/make_readme_figure.py [--model models/model.onnx] [--data data/processed/brisc-yolo]

Panels are chosen by rule, not by hand, so the figure shows typical behaviour rather than best cases:

* each tumour class: the correct prediction with the median Dice among that class's correct predictions
* healthy: the first (by file name) healthy slice correctly left without detections
* the most frequent mistake between tumour types, at its median confidence
* the first (by file name) tumour slice the model called healthy

Filled colour is the prediction (drawn exactly like the API's /v1/predict/overlay), white outline the expert mask.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from btd.constants import NO_TUMOR, TUMOR_CLASSES
from btd.inference.engine import SegmentationEngine
from btd.inference.visualize import draw_overlay
from btd.training.evaluate import load_split
from btd.training.metrics import dice_iou
from btd.utils import imread, imread_gray, imwrite

REPO = Path(__file__).resolve().parents[1]
TILE = 320
CAPTION_H = 52
GAP = 4
PRETTY = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary", NO_TUMOR: "healthy"}


def score_test_split(engine: SegmentationEngine, data: Path) -> list[dict[str, Any]]:
    rows = []
    for s in load_split(data, "test"):
        pred = engine.predict(imread(s.image))
        gt = imread_gray(s.mask) > 127 if s.mask else None
        dice = dice_iou(pred.union_mask(), gt)[0] if gt is not None else None
        rows.append({"sample": s, "pred": pred.decision.label, "score": pred.decision.score, "dice": dice})
    return rows


def pick_panels(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    def median(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        return sorted(items, key=lambda r: (r[key], r["sample"].key))[len(items) // 2]

    panels = []
    for cls in TUMOR_CLASSES:
        correct = [r for r in rows if r["sample"].label == cls and r["pred"] == cls]
        panels.append((f"{PRETTY[cls].capitalize()}: correct", median(correct, "dice")))
    healthy = sorted(
        (r for r in rows if r["sample"].label == NO_TUMOR and r["pred"] == NO_TUMOR),
        key=lambda r: r["sample"].key,
    )
    panels.append(("Healthy: correct", healthy[0]))

    confusions = Counter(
        (r["sample"].label, r["pred"])
        for r in rows
        if r["sample"].label != r["pred"] and NO_TUMOR not in (r["sample"].label, r["pred"])
    )
    if confusions:
        (truth, wrong), _ = confusions.most_common(1)[0]
        cases = [r for r in rows if r["sample"].label == truth and r["pred"] == wrong]
        panels.append((f"Most common mistake: {truth} read as {wrong}", median(cases, "score")))
    missed = sorted(
        (r for r in rows if r["sample"].label != NO_TUMOR and r["pred"] == NO_TUMOR),
        key=lambda r: r["sample"].key,
    )
    if missed:
        panels.append((f"Missed tumour: {missed[0]['sample'].label} read as healthy", missed[0]))
    return panels


def render(engine: SegmentationEngine, title: str, row: dict[str, Any]) -> np.ndarray:
    s = row["sample"]
    img = imread(s.image)
    pred = engine.predict(img)
    out = draw_overlay(img, pred, show_disclaimer=False)
    if s.mask:
        gt = (imread_gray(s.mask) > 127).astype(np.uint8)
        contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), max(1, round(min(img.shape[:2]) / 250)))

    h, w = out.shape[:2]
    scale = TILE / max(h, w)
    small = cv2.resize(out, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    tile = np.zeros((TILE + CAPTION_H, TILE, 3), np.uint8)
    y0, x0 = (TILE - small.shape[0]) // 2, (TILE - small.shape[1]) // 2
    tile[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small
    tile[TILE:] = (40, 40, 40)

    facts = [f"score {pred.decision.score:.2f}"] if pred.decision.label != NO_TUMOR else ["no detections"]
    if row["dice"] is not None:
        facts.append(f"Dice {row['dice']:.2f}")
    for i, text in enumerate((title, ", ".join(facts))):
        scale_txt = 0.45
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale_txt, 1)
        scale_txt *= min(1.0, (TILE - 24) / tw)
        cv2.putText(
            tile,
            text,
            (6, TILE + 20 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale_txt,
            (255, 255, 255) if i == 0 else (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return tile


def join(images: list[np.ndarray], axis: int) -> np.ndarray:
    """Stack images side by side (axis=1) or on top of each other (axis=0) with a white gap between them."""
    parts: list[np.ndarray] = []
    for im in images:
        if parts:
            gap_shape = (im.shape[0], GAP, 3) if axis == 1 else (GAP, im.shape[1], 3)
            parts.append(np.full(gap_shape, 255, np.uint8))
        parts.append(im)
    return np.concatenate(parts, axis=axis)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(REPO / "models" / "model.onnx"))
    ap.add_argument("--data", default=str(REPO / "data" / "processed" / "brisc-yolo"))
    ap.add_argument("--out", default=str(REPO / "docs" / "images" / "predictions.png"))
    args = ap.parse_args()

    engine = SegmentationEngine(args.model, providers="auto")
    rows = score_test_split(engine, Path(args.data))
    wrong = Counter((r["sample"].label, r["pred"]) for r in rows if r["sample"].label != r["pred"])
    print(f"{len(rows)} test slices, {sum(wrong.values())} wrong: {dict(wrong.most_common())}")

    panels = pick_panels(rows)
    for title, r in panels:
        print(f"  {title:<50} {r['sample'].key}")
    tiles = [render(engine, title, r) for title, r in panels]
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = join([join(tiles[i : i + cols], axis=1) for i in range(0, len(tiles), cols)], axis=0)
    imwrite(args.out, grid, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"wrote {args.out} ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
