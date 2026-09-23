"""Convert BRISC 2025 into an Ultralytics YOLO-seg dataset with a leakage-safe validation split.

Output layout::

    <out>/images/{train,val,test}/*.jpg     slices (no-tumour slices are background images with empty labels)
    <out>/labels/{train,val,test}/*.txt     YOLO-seg polygons: "<cls> x1 y1 x2 y2 ..." (normalised)
    <out>/masks/{train,val,test}/*.png      binary ground-truth masks (for Dice/IoU evaluation)
    <out>/data.yaml                         Ultralytics dataset file
    <out>/splits.csv                        one row per slice: split, label, plane, cluster, polygon fidelity
    <out>/prepare_report.json               counts + conversion quality

* The official BRISC test split is kept untouched and is only used for the final evaluation.
* Training slices that are exact or near duplicates of any test slice are **dropped**. Despite the release notes,
  BRISC 2025 ships such pairs across its train/test split, and keeping them would inflate test metrics.
* Validation is carved out of BRISC train, stratified by (label, plane) and **grouped by near-duplicate
  clusters**, so near-identical slices can never straddle train and val.
"""

from __future__ import annotations

import csv
import logging
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from btd.constants import NO_TUMOR, TUMOR_CLASSES
from btd.data.audit import Item, compute_entries
from btd.data.brisc import BriscRecord, index_brisc, verify_manifest
from btd.data.hashing import UnionFind, hamming_pairs
from btd.utils import imread_gray, imwrite, utc_now, write_json

LOGGER = logging.getLogger(__name__)
CLASS_TO_ID = {name: i for i, name in enumerate(TUMOR_CLASSES)}
MARKER = "prepare_report.json"


@dataclass
class Converted:
    polygons: list[np.ndarray]  # each (k, 2) normalised to [0, 1]
    fidelity_iou: float | None  # IoU between the original mask and the rasterised polygons


def binarise(mask: np.ndarray) -> np.ndarray:
    return mask > (127 if mask.max() > 1 else 0)


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 16.0, epsilon_px: float = 1.0) -> Converted:
    """External contours of a binary mask → simplified, normalised polygons (+ rasterisation fidelity)."""
    binary = binarise(mask).astype(np.uint8)
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys: list[np.ndarray] = []
    raster = np.zeros_like(binary)
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        approx = cv2.approxPolyDP(c, epsilon_px, True).reshape(-1, 2)
        if approx.shape[0] < 3:
            continue
        cv2.fillPoly(raster, [approx.astype(np.int32)], 1)
        polys.append(np.clip(approx.astype(np.float64) / [w, h], 0.0, 1.0))
    if not binary.any():
        return Converted(polys, None)
    inter = np.logical_and(raster, binary).sum()
    union = np.logical_or(raster, binary).sum()
    return Converted(polys, float(inter / union) if union else None)


def yolo_label_lines(class_id: int, polygons: list[np.ndarray]) -> list[str]:
    return [f"{class_id} " + " ".join(f"{v:.6f}" for v in poly.ravel()) for poly in polygons]


def near_duplicate_clusters(
    records: list[BriscRecord], max_hamming: int = 10, min_corr: float = 0.95, workers: int = 8
) -> list[int]:
    """Cluster id per record (records in the same cluster must land in the same split)."""
    entries = compute_entries([Item(r.image_path, r.split, r.label) for r in records], workers=workers)
    hashes = np.array([e.phash for e in entries], dtype=np.uint64)
    thumbs = np.stack([e.thumb for e in entries])
    uf = UnionFind(len(records))
    for i, j, _ in hamming_pairs(hashes, max_hamming):
        if entries[i].pixel_md5 == entries[j].pixel_md5 or float(thumbs[i] @ thumbs[j]) >= min_corr:
            uf.union(i, j)
    return [uf.find(i) for i in range(len(records))]


def stratified_group_split(
    records: list[BriscRecord], clusters: list[int], val_fraction: float, seed: int
) -> set[str]:
    """Return the keys that go to validation. Whole clusters move together; quotas are per (label, plane)."""
    rng = random.Random(seed)
    by_cluster: dict[int, list[int]] = {}
    for i, c in enumerate(clusters):
        by_cluster.setdefault(c, []).append(i)
    strata: dict[tuple[str, str], list[list[int]]] = {}
    for members in by_cluster.values():
        first = records[members[0]]
        strata.setdefault((first.label, first.plane), []).append(members)
    val: set[str] = set()
    for key in sorted(strata):
        groups = strata[key]
        rng.shuffle(groups)
        total = sum(len(g) for g in groups)
        target = round(val_fraction * total)
        taken = 0
        for g in groups:
            if taken >= target:
                break
            val.update(records[i].key for i in g)
            taken += len(g)
    return val


def prepare_dataset(
    src: str | Path,
    out: str | Path,
    val_fraction: float = 0.15,
    seed: int = 42,
    overwrite: bool = False,
    verify_checksums: bool = True,
    min_area_px: float = 16.0,
    epsilon_px: float = 1.0,
    workers: int = 8,
) -> dict[str, Any]:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty; pass --overwrite to rebuild it")
        if not (out / MARKER).is_file():
            raise FileExistsError(f"Refusing to delete {out}: it was not created by `btd data prepare`")
        shutil.rmtree(out)

    index = index_brisc(src)
    manifest = verify_manifest(index.root) if verify_checksums else {"status": "skipped"}
    if manifest.get("mismatched"):
        raise RuntimeError(
            f"{len(manifest['mismatched'])} files fail their manifest SHA-256 - re-download the data"
        )

    train = index.by_split("train")
    test = index.by_split("test")
    LOGGER.info("Clustering %d train + %d test slices for leakage-safe splits...", len(train), len(test))
    all_clusters = near_duplicate_clusters(train + test, workers=workers)
    train_clusters = all_clusters[: len(train)]
    test_clusters = set(all_clusters[len(train) :])
    keep = [c not in test_clusters for c in train_clusters]
    dropped = [r.key for r, k in zip(train, keep, strict=True) if not k]
    clusters = [c for c, k in zip(train_clusters, keep, strict=True) if k]
    train = [r for r, k in zip(train, keep, strict=True) if k]
    if dropped:
        LOGGER.warning("Dropped %d training slices that duplicate a test slice", len(dropped))
    val_keys = stratified_group_split(train, clusters, val_fraction, seed)
    cluster_of = {r.key: c for r, c in zip(train, clusters, strict=True)}

    rows: list[dict[str, Any]] = []
    fidelity: list[float] = []
    skipped: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for rec in train + test:
        split = "test" if rec.split == "test" else ("val" if rec.key in val_keys else "train")
        lines: list[str] = []
        n_inst = 0
        fid: float | None = None
        if rec.is_tumor:
            if rec.mask_path is None:
                skipped.append(rec.key)
                continue
            mask = imread_gray(rec.mask_path)
            img_hw = imread_gray(rec.image_path).shape[:2]
            if mask.shape[:2] != img_hw:
                LOGGER.warning("Mask/image size mismatch for %s; resizing mask", rec.key)
                mask = cv2.resize(mask, (img_hw[1], img_hw[0]), interpolation=cv2.INTER_NEAREST)
            conv = mask_to_polygons(mask, min_area_px=min_area_px, epsilon_px=epsilon_px)
            if not conv.polygons:
                skipped.append(rec.key)
                continue
            lines = yolo_label_lines(CLASS_TO_ID[rec.label], conv.polygons)
            n_inst = len(conv.polygons)
            fid = conv.fidelity_iou
            if fid is not None:
                fidelity.append(fid)
            imwrite(out / "masks" / split / f"{rec.key}.png", binarise(mask).astype(np.uint8) * 255)

        dst_img = out / "images" / split / f"{rec.key}{rec.image_path.suffix.lower()}"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec.image_path, dst_img)
        label_file = out / "labels" / split / f"{rec.key}.txt"
        label_file.parent.mkdir(parents=True, exist_ok=True)
        label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        counts.setdefault(split, {}).setdefault(rec.label, 0)
        counts[split][rec.label] += 1
        rows.append(
            {
                "key": rec.key,
                "split": split,
                "label": rec.label,
                "plane": rec.plane,
                "cluster": cluster_of.get(rec.key, ""),
                "instances": n_inst,
                "polygon_iou": "" if fid is None else round(fid, 4),
                "image": dst_img.relative_to(out).as_posix(),
            }
        )

    data_yaml = {
        "path": out.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(TUMOR_CLASSES)),
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    with (out / "splits.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "generated_at": utc_now(),
        "source": index.summary(),
        "manifest_check": {k: v if not isinstance(v, list) else len(v) for k, v in manifest.items()},
        "val_fraction": val_fraction,
        "seed": seed,
        "counts": {s: dict(sorted(c.items())) for s, c in sorted(counts.items())},
        "background_images": {s: c.get(NO_TUMOR, 0) for s, c in counts.items()},
        "skipped_tumor_slices_without_usable_mask": skipped,
        "train_slices_dropped_as_test_duplicates": dropped,
        "polygon_fidelity_iou": {
            "mean": round(float(np.mean(fidelity)), 4) if fidelity else None,
            "p05": round(float(np.percentile(fidelity, 5)), 4) if fidelity else None,
            "min": round(float(np.min(fidelity)), 4) if fidelity else None,
        },
        "near_duplicate_clusters_in_train": sum(1 for n in Counter(clusters).values() if n > 1),
        "data_yaml": (out / "data.yaml").as_posix(),
    }
    write_json(out / MARKER, report)
    LOGGER.info("Prepared dataset at %s: %s", out, report["counts"])
    return report
