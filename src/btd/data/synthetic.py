"""Synthetic BRISC-shaped dataset for CI and local smoke tests.

Produces the exact BRISC 2025 folder layout, filename convention, masks and manifest so the full pipeline
(prepare → train → export → quantise → serve) can run in minutes on a CPU without downloading anything.
The images are cartoons of axial/coronal/sagittal slices with class-specific tumour appearance/location:

* glioma      — irregular, ring-enhancing lesion inside a hemisphere
* meningioma  — round, homogeneous, bright lesion attached to the skull
* pituitary   — small bright lesion in the sellar region (midline, inferior)
* no tumour   — healthy slice

They are NOT medical images; they only exercise the code paths.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import cv2
import numpy as np

from btd.constants import TUMOR_CODES
from btd.utils import imwrite, sha256_file

CODE_BY_LABEL = {v: k for k, v in TUMOR_CODES.items()}
PLANES = ("ax", "co", "sa")


def _texture(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), sigma)


def make_slice(
    rng: np.random.Generator, label: str, size: int = 256, plane: str = "ax"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (grayscale uint8 image, uint8 mask with 255 = tumour)."""
    img = np.zeros((size, size), np.float32)
    cx = size / 2 + rng.normal(0, size * 0.015)
    cy = size / 2 + rng.normal(0, size * 0.015)
    ax = size * rng.uniform(0.32, 0.38) * (1.12 if plane == "sa" else 1.0)
    ay = size * rng.uniform(0.38, 0.44) * (0.92 if plane == "sa" else 1.0)
    ang = float(rng.uniform(-8, 8))
    center = (round(cx), round(cy))

    brain = np.zeros_like(img)
    cv2.ellipse(brain, center, (round(ax * 0.9), round(ay * 0.9)), ang, 0, 360, 1.0, -1)
    img += brain * (95 + 12 * _texture(rng, size, size / 40))
    cv2.ellipse(img, center, (round(ax), round(ay)), ang, 0, 360, 185, max(2, size // 40))
    for side in (-1, 1):  # ventricles
        vc = (round(cx + side * ax * 0.14), round(cy - ay * 0.05))
        cv2.ellipse(
            img, vc, (max(2, round(ax * 0.07)), max(3, round(ay * 0.18))), side * 12.0, 0, 360, 35, -1
        )

    mask = np.zeros((size, size), np.uint8)
    if label == "glioma":
        hemisphere = int(rng.choice([-1, 1]))
        gx = cx + hemisphere * ax * rng.uniform(0.3, 0.5)
        gy = cy + ay * rng.uniform(-0.35, 0.3)
        r = size * rng.uniform(0.06, 0.10)
        for _ in range(int(rng.integers(3, 6))):
            ox, oy = rng.normal(0, r * 0.45, 2)
            cv2.circle(
                mask, (round(gx + ox), round(gy + oy)), max(2, round(r * rng.uniform(0.55, 0.9))), 255, -1
            )
        ring = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        core = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
        img[mask > 0] = 150 + 15 * _texture(rng, size, 2)[mask > 0]
        img[core] = 60
        img[ring] = 215
    elif label == "meningioma":
        theta = rng.uniform(0, 2 * math.pi)
        r = size * rng.uniform(0.05, 0.09)
        mx = cx + (ax * 0.9 - r * 0.8) * math.cos(theta)
        my = cy + (ay * 0.9 - r * 0.8) * math.sin(theta)
        cv2.circle(mask, (round(mx), round(my)), max(2, round(r)), 255, -1)
        img[mask > 0] = 205 + 8 * _texture(rng, size, 3)[mask > 0]
    elif label == "pituitary":
        r = size * rng.uniform(0.03, 0.05)
        px = cx + rng.normal(0, size * 0.01)
        py = cy + ay * rng.uniform(0.35, 0.45)
        cv2.ellipse(
            mask, (round(px), round(py)), (max(2, round(r * 1.2)), max(2, round(r))), 0, 0, 360, 255, -1
        )
        img[mask > 0] = 225
    elif label != "no_tumor":
        raise ValueError(f"Unknown label {label!r}")

    noisy = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 4, img.shape).astype(np.float32)
    return np.clip(noisy, 0, 255).astype(np.uint8), mask


def generate(
    out_dir: str | Path,
    n_train_per_class: int = 24,
    n_test_per_class: int = 8,
    size: int = 256,
    seed: int = 0,
) -> Path:
    """Write a synthetic ``brisc2025`` tree under ``out_dir`` and return its root."""
    rng = np.random.default_rng(seed)
    root = Path(out_dir) / "brisc2025"
    rows: list[dict[str, str]] = []
    for split, n in (("train", n_train_per_class), ("test", n_test_per_class)):
        index = 0
        for label in ("glioma", "meningioma", "pituitary", "no_tumor"):
            for _ in range(n):
                index += 1
                plane = str(rng.choice(PLANES))
                img, mask = make_slice(rng, label, size=size, plane=plane)
                stem = f"brisc2025_{split}_{index:05d}_{CODE_BY_LABEL[label]}_{plane}_t1"
                cls_path = root / "classification_task" / split / label / f"{stem}.jpg"
                imwrite(cls_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(_row(root, cls_path, "classification", split, label, plane))
                if label != "no_tumor":
                    seg_img = root / "segmentation_task" / split / "images" / f"{stem}.jpg"
                    seg_mask = root / "segmentation_task" / split / "masks" / f"{stem}.png"
                    imwrite(seg_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    imwrite(seg_mask, mask)
                    rows.append(_row(root, seg_img, "segmentation", split, label, plane))
                    rows.append(_row(root, seg_mask, "segmentation", split, label, plane))
    with (root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


def _row(root: Path, path: Path, task: str, split: str, label: str, plane: str) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "task": task,
        "split": split,
        "tumor_code": CODE_BY_LABEL[label],
        "plane_code": plane,
        "sha256": sha256_file(path),
    }
