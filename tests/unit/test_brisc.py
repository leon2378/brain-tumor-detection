from __future__ import annotations

import csv
from pathlib import Path

import pytest

from btd.data.brisc import find_brisc_root, index_brisc, parse_name, verify_manifest


def test_parse_name() -> None:
    n = parse_name("brisc2025_test_00010_gl_ax_t1")
    assert n is not None
    assert (n.split, n.index, n.label, n.plane, n.sequence) == ("test", 10, "glioma", "axial", "t1")
    # real BRISC 2025 no-tumour files use "no", e.g. brisc2025_train_02477_no_ax_t1.jpg
    assert parse_name("brisc2025_train_02477_no_ax_t1").label == "no_tumor"  # type: ignore[union-attr]
    assert parse_name("Tr-gl_0010") is None
    assert parse_name("brisc2025_train_00001_xx_ax_t1") is None


def test_index_synthetic(synthetic_brisc: Path) -> None:
    index = index_brisc(synthetic_brisc.parent)  # finds the nested brisc2025 folder
    assert index.root == synthetic_brisc
    summary = index.summary()
    assert summary["images"] == 4 * 12 + 4 * 4
    assert summary["with_masks"] == 3 * 12 + 3 * 4  # every tumour slice has a mask
    assert all(v == 0 for v in summary["issues"].values())
    rec = next(r for r in index.records if r.label == "glioma")
    assert rec.sources == {"segmentation", "classification"}


def test_manifest_verification_detects_tampering(synthetic_brisc: Path, tmp_path: Path) -> None:
    assert verify_manifest(synthetic_brisc)["status"] == "ok"
    rows = list(csv.DictReader((synthetic_brisc / "manifest.csv").open(encoding="utf-8")))
    victim = synthetic_brisc / rows[0]["relative_path"]
    original = victim.read_bytes()
    try:
        victim.write_bytes(original + b"tampered")
        res = verify_manifest(synthetic_brisc)
        assert res["status"] == "problems" and res["mismatched"] == [rows[0]["relative_path"]]
    finally:
        victim.write_bytes(original)


def test_missing_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_brisc_root(tmp_path)
