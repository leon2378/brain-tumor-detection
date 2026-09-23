"""Index the BRISC 2025 release into one record per unique slice.

The release ships two task folders that share filenames::

    brisc2025/
      classification_task/{train,test}/<class>/*.jpg          # all 6,000 slices (incl. no-tumour)
      segmentation_task/{train,test}/{images,masks}/*.{jpg,png} # 4,793 tumour slices + expert masks
      manifest.csv / manifest.json                            # per-file metadata incl. SHA-256

Labels come from the filename code (``brisc2025_<split>_<idx>_<gl|me|pi|no>_<ax|co|sa>_<seq>``) which is
authoritative; folder names are only cross-checked. The loader is tolerant of folder naming so it keeps working if
class folders are renamed between dataset versions.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from btd.constants import IMAGE_EXTENSIONS, NO_TUMOR, PLANE_CODES, TUMOR_CODES, canonical_label
from btd.utils import sha256_file

LOGGER = logging.getLogger(__name__)

FILENAME_RE = re.compile(
    r"^brisc2025_(?P<split>train|test)_(?P<index>\d+)_(?P<tumor>[a-z]{2})_(?P<plane>[a-z]{2})_(?P<seq>[a-z0-9]+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BriscName:
    split: str
    index: int
    tumor_code: str
    plane_code: str
    sequence: str

    @property
    def label(self) -> str:
        return TUMOR_CODES[self.tumor_code]

    @property
    def plane(self) -> str:
        return PLANE_CODES.get(self.plane_code, self.plane_code)


def parse_name(stem: str) -> BriscName | None:
    m = FILENAME_RE.match(stem)
    if not m or m["tumor"].lower() not in TUMOR_CODES:
        return None
    return BriscName(
        split=m["split"].lower(),
        index=int(m["index"]),
        tumor_code=m["tumor"].lower(),
        plane_code=m["plane"].lower(),
        sequence=m["seq"].lower(),
    )


@dataclass
class BriscRecord:
    key: str
    split: str
    label: str
    plane: str
    image_path: Path
    mask_path: Path | None = None
    sources: set[str] = field(default_factory=set)

    @property
    def is_tumor(self) -> bool:
        return self.label != NO_TUMOR


@dataclass
class BriscIndex:
    root: Path
    records: list[BriscRecord]
    issues: dict[str, list[str]]

    def by_split(self, split: str) -> list[BriscRecord]:
        return [r for r in self.records if r.split == split]

    def summary(self) -> dict[str, Any]:
        counts: dict[str, dict[str, int]] = {}
        for r in self.records:
            counts.setdefault(r.split, {}).setdefault(r.label, 0)
            counts[r.split][r.label] += 1
        return {
            "root": self.root.as_posix(),
            "images": len(self.records),
            "with_masks": sum(r.mask_path is not None for r in self.records),
            "counts": counts,
            "issues": {k: len(v) for k, v in self.issues.items()},
        }


def find_brisc_root(path: str | Path, max_depth: int = 3) -> Path:
    """Locate the folder that contains ``classification_task`` / ``segmentation_task``."""
    base = Path(path)
    candidates = [base]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for c in candidates:
            if (c / "segmentation_task").is_dir() or (c / "classification_task").is_dir():
                return c
            if c.is_dir():
                nxt.extend(p for p in c.iterdir() if p.is_dir() and not p.name.startswith("."))
        candidates = nxt
    raise FileNotFoundError(
        f"No BRISC layout (classification_task/ or segmentation_task/) found under {base}. "
        "Run `btd data download` or pass --src pointing at the extracted brisc2025 folder."
    )


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def index_brisc(path: str | Path) -> BriscIndex:
    root = find_brisc_root(path)
    records: dict[str, BriscRecord] = {}
    issues: dict[str, list[str]] = {
        "unparsed": [],
        "label_conflicts": [],
        "split_conflicts": [],
        "tumor_without_mask": [],
        "orphan_masks": [],
    }

    seg = root / "segmentation_task"
    seg_splits = sorted(p for p in seg.iterdir() if p.is_dir()) if seg.is_dir() else []
    for split_dir in seg_splits:
        masks = {p.stem: p for p in _images(split_dir / "masks")}
        images = _images(split_dir / "images")
        image_stems = {p.stem for p in images}
        issues["orphan_masks"].extend(str(m) for s, m in masks.items() if s not in image_stems)
        for img in images:
            name = parse_name(img.stem)
            if name is None:
                issues["unparsed"].append(str(img))
                continue
            if name.split != split_dir.name.lower():
                issues["split_conflicts"].append(str(img))
            records[img.stem] = BriscRecord(
                key=img.stem,
                split=name.split,
                label=name.label,
                plane=name.plane,
                image_path=img,
                mask_path=masks.get(img.stem),
                sources={"segmentation"},
            )

    cls = root / "classification_task"
    for img in _images(cls):
        name = parse_name(img.stem)
        if name is None:
            issues["unparsed"].append(str(img))
            continue
        folder_label = canonical_label(img.parent.name)
        if folder_label is not None and folder_label != name.label:
            issues["label_conflicts"].append(f"{img} (folder={folder_label}, filename={name.label})")
        rec = records.get(img.stem)
        if rec is None:
            records[img.stem] = BriscRecord(
                key=img.stem,
                split=name.split,
                label=name.label,
                plane=name.plane,
                image_path=img,
                sources={"classification"},
            )
        else:
            rec.sources.add("classification")

    for rec in records.values():
        if rec.is_tumor and rec.mask_path is None:
            issues["tumor_without_mask"].append(rec.key)

    out = sorted(records.values(), key=lambda r: (r.split, r.key))
    if not out:
        raise FileNotFoundError(f"No BRISC images found under {root}")
    for k, v in issues.items():
        if v:
            LOGGER.warning("BRISC index: %d %s (e.g. %s)", len(v), k.replace("_", " "), v[0])
    return BriscIndex(root=root, records=out, issues=issues)


def verify_manifest(root: str | Path, limit: int | None = None) -> dict[str, Any]:
    """Check files against ``manifest.csv`` SHA-256 checksums when the manifest is present."""
    root = Path(root)
    manifest = next((p for p in (root / "manifest.csv", root.parent / "manifest.csv") if p.is_file()), None)
    if manifest is None:
        return {"status": "no-manifest", "checked": 0, "missing": [], "mismatched": []}
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"status": "empty-manifest", "checked": 0, "missing": [], "mismatched": []}
    cols = {c.lower(): c for c in rows[0]}
    path_col = next((cols[c] for c in ("relative_path", "path", "filepath", "file_path") if c in cols), None)
    sha_col = next((orig for low, orig in cols.items() if "sha256" in low), None)
    if path_col is None or sha_col is None:
        return {
            "status": f"unrecognised-columns {sorted(cols)}",
            "checked": 0,
            "missing": [],
            "mismatched": [],
        }

    missing: list[str] = []
    mismatched: list[str] = []
    checked = 0
    for row in rows[:limit] if limit else rows:
        rel = row[path_col].replace("\\", "/").lstrip("/")
        candidates = [root / rel, root.parent / rel]
        if rel.startswith(root.name + "/"):
            candidates.append(root / rel[len(root.name) + 1 :])
        target = next((c for c in candidates if c.is_file()), None)
        if target is None:
            missing.append(rel)
            continue
        checked += 1
        if sha256_file(target) != row[sha_col].strip().lower():
            mismatched.append(rel)
    status = "ok" if not missing and not mismatched else "problems"
    return {"status": status, "checked": checked, "missing": missing, "mismatched": mismatched}
