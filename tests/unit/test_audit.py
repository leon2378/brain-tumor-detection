from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from btd.data.audit import run_audit, scan_folders
from btd.data.synthetic import make_slice
from btd.utils import imwrite


def _kaggle_like(root: Path) -> None:
    """Training/Testing/<class> layout with planted leakage, like the Kaggle 'Brain Tumor MRI Dataset'."""
    rng = np.random.default_rng(42)
    classes = {
        "glioma": "glioma",
        "meningioma": "meningioma",
        "pituitary": "pituitary",
        "notumor": "no_tumor",
    }
    kept = {}
    for split, n in (("Training", 6), ("Testing", 2)):
        for folder, label in classes.items():
            for i in range(n):
                img, _ = make_slice(rng, label, 128)
                imwrite(root / split / folder / f"{split[:2]}-{folder[:2]}_{i}.jpg", img)
                kept[(split, folder, i)] = img
    # 1) exact copy of a training glioma inside Testing
    imwrite(root / "Testing" / "glioma" / "Te-gl_dup.jpg", kept[("Training", "glioma", 0)])
    # 2) re-encoded near-duplicate of a training pituitary inside Testing
    _, buf = cv2.imencode(".jpg", kept[("Training", "pituitary", 1)], [cv2.IMWRITE_JPEG_QUALITY, 35])
    (root / "Testing" / "pituitary" / "Te-pi_near.jpg").write_bytes(buf.tobytes())
    # 3) the same image filed under two labels in Training
    imwrite(root / "Training" / "notumor" / "Tr-no_conflict.jpg", kept[("Training", "meningioma", 2)])


def test_scan_folders_maps_kaggle_names(tmp_path: Path) -> None:
    _kaggle_like(tmp_path)
    items = scan_folders(tmp_path)
    assert {i.split for i in items} == {"train", "test"}
    assert {i.label for i in items} == {"glioma", "meningioma", "pituitary", "no_tumor"}


def test_audit_finds_leakage_and_conflicts(tmp_path: Path) -> None:
    data = tmp_path / "Brain Tumor MRI Data"
    _kaggle_like(data)
    report = run_audit(data, tmp_path / "report", workers=2)
    assert report["source"]["layout"] == "folders"
    assert report["exact_duplicates"]["groups"] >= 2
    leak = report["leakage"]["test"]
    assert leak["with_exact_copy_in_train"] >= 1
    assert leak["with_exact_or_near_copy_in_train"] >= 2
    assert report["label_conflicts"]["clusters"] >= 1
    for name in ("audit.json", "audit.md", "duplicate_pairs.csv"):
        assert (tmp_path / "report" / name).is_file()
    assert json.loads((tmp_path / "report" / "audit.json").read_text())["images"] == report["images"]
    assert "leakage" in (tmp_path / "report" / "audit.md").read_text().lower()


def test_audit_on_brisc_layout(synthetic_brisc: Path, tmp_path: Path) -> None:
    report = run_audit(synthetic_brisc.parent, tmp_path / "r", workers=2)
    assert report["source"]["layout"] == "brisc"
    assert report["counts"]["train"]["glioma"] == 12
    assert report["exact_duplicates"]["groups"] == 0
