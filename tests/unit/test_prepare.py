from __future__ import annotations

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from btd.data.prepare import mask_to_polygons, prepare_dataset
from btd.data.synthetic import generate


def test_mask_to_polygons_fidelity() -> None:
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (80, 90), 40, 255, -1)
    cv2.rectangle(mask, (150, 150), (190, 185), 255, -1)
    cv2.circle(mask, (10, 10), 1, 255, -1)  # speck below min area → dropped
    conv = mask_to_polygons(mask, min_area_px=16)
    assert len(conv.polygons) == 2
    assert conv.fidelity_iou is not None and conv.fidelity_iou > 0.95
    for poly in conv.polygons:
        assert poly.min() >= 0.0 and poly.max() <= 1.0


def test_prepare_dataset(synthetic_brisc: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed yolo"
    report = prepare_dataset(synthetic_brisc, out, val_fraction=0.25, seed=1, workers=2)
    data = yaml.safe_load((out / "data.yaml").read_text())
    assert data["names"] == {0: "glioma", 1: "meningioma", 2: "pituitary"}
    assert Path(data["path"]).is_absolute()
    counts = report["counts"]
    dropped = report["train_slices_dropped_as_test_duplicates"]
    assert sum(counts["test"].values()) == 16  # official test split untouched
    assert sum(counts["train"].values()) + sum(counts["val"].values()) + len(dropped) == 48
    assert counts["val"]["no_tumor"] >= 1  # background images are stratified too

    rows = list(csv.DictReader((out / "splits.csv").open(encoding="utf-8")))
    assert {r["key"] for r in rows}.isdisjoint(dropped)
    for r in rows:
        label_file = out / "labels" / r["split"] / f"{r['key']}.txt"
        lines = label_file.read_text().strip().splitlines()
        if r["label"] == "no_tumor":
            assert lines == []  # background image
        else:
            cls = int(lines[0].split()[0])
            assert cls == ["glioma", "meningioma", "pituitary"].index(r["label"])
            assert (out / "masks" / r["split"] / f"{r['key']}.png").is_file()
    # clusters never straddle train/val
    split_of_cluster: dict[str, set[str]] = {}
    for r in rows:
        if r["cluster"] != "":
            split_of_cluster.setdefault(r["cluster"], set()).add(r["split"])
    assert all(len(s) == 1 for s in split_of_cluster.values())

    with pytest.raises(FileExistsError):
        prepare_dataset(synthetic_brisc, out)
    prepare_dataset(synthetic_brisc, out, overwrite=True, workers=2)


def test_prepare_drops_train_duplicates_of_test(tmp_path: Path) -> None:
    root = generate(tmp_path / "raw", n_train_per_class=6, n_test_per_class=2, size=96, seed=3)
    cls = root / "classification_task"
    test_img = sorted((cls / "test" / "no_tumor").glob("*.jpg"))[0]
    train_img = sorted((cls / "train" / "no_tumor").glob("*.jpg"))[0]
    shutil.copyfile(test_img, train_img)  # plant an exact train/test duplicate
    out = tmp_path / "out"
    report = prepare_dataset(root, out, workers=2, verify_checksums=False)
    assert train_img.stem in report["train_slices_dropped_as_test_duplicates"]
    rows = list(csv.DictReader((out / "splits.csv").open(encoding="utf-8")))
    assert train_img.stem not in {r["key"] for r in rows}
    assert sum(r["split"] == "test" for r in rows) == 8  # official test split untouched


def test_prepare_refuses_to_delete_foreign_dir(synthetic_brisc: Path, tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("user data")
    with pytest.raises(FileExistsError, match="Refusing"):
        prepare_dataset(synthetic_brisc, tmp_path, overwrite=True)
    assert (tmp_path / "keep.txt").exists()
