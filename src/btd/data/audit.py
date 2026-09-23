"""Dataset audit: class balance, image properties, exact/near duplicates and train↔test leakage.

Works on BRISC (``--layout brisc``) and on the common Kaggle folder layout ``<root>/<split>/<class>/*.jpg``
(``--layout folders``), e.g. the "Brain Tumor MRI Dataset" with ``Training/`` and ``Testing/``.

Why it matters: public brain-MRI classification sets are stitched together from older collections and contain
duplicated slices. When a test image has a (near-)copy in the training split, test accuracy measures memorisation,
not generalisation — which is how 99%+ accuracies end up in so many notebooks.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from btd.constants import IMAGE_EXTENSIONS, canonical_label
from btd.data.hashing import UnionFind, dhash, hamming_pairs, phash, thumbnail_vector
from btd.utils import imread_gray, utc_now, write_json

LOGGER = logging.getLogger(__name__)

TRAIN_NAMES = {"train", "training"}
TEST_NAMES = {"test", "testing"}


@dataclass(frozen=True)
class Item:
    path: Path
    split: str
    label: str


@dataclass
class Entry:
    item: Item
    width: int
    height: int
    nbytes: int
    file_md5: str
    pixel_md5: str
    phash: np.uint64
    dhash: np.uint64
    thumb: np.ndarray


def normalise_split(name: str) -> str:
    low = name.lower()
    if low in TRAIN_NAMES:
        return "train"
    if low in TEST_NAMES:
        return "test"
    if low in {"val", "valid", "validation"}:
        return "val"
    return low


def scan_folders(root: str | Path) -> list[Item]:
    """``root/<split>/<class>/*`` or ``root/<class>/*`` (single split called ``all``)."""
    root = Path(root)
    items: list[Item] = []
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    has_splits = any(canonical_label(d.name) is None for d in subdirs)
    split_dirs = [(normalise_split(d.name), d) for d in subdirs] if has_splits else [("all", root)]
    for split, sdir in split_dirs:
        for cdir in sorted(p for p in sdir.iterdir() if p.is_dir()):
            label = canonical_label(cdir.name) or cdir.name.lower()
            items.extend(
                Item(path=p, split=split, label=label)
                for p in sorted(cdir.rglob("*"))
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
    if not items:
        raise FileNotFoundError(f"No images found under {root} (expected <split>/<class>/*.jpg)")
    return items


def scan_brisc(root: str | Path) -> list[Item]:
    from btd.data.brisc import index_brisc

    return [Item(path=r.image_path, split=r.split, label=r.label) for r in index_brisc(root).records]


def _entry(item: Item) -> Entry:
    raw = item.path.read_bytes()
    gray = imread_gray(item.path)
    return Entry(
        item=item,
        width=int(gray.shape[1]),
        height=int(gray.shape[0]),
        nbytes=len(raw),
        file_md5=hashlib.md5(raw, usedforsecurity=False).hexdigest(),
        pixel_md5=hashlib.md5(gray.tobytes(), usedforsecurity=False).hexdigest(),
        phash=phash(gray),
        dhash=dhash(gray),
        thumb=thumbnail_vector(gray),
    )


def compute_entries(items: list[Item], workers: int = 8) -> list[Entry]:
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(_entry, items))


def _pair_kind(a: Item, b: Item) -> str:
    return "same-split" if a.split == b.split else f"cross-split:{'-'.join(sorted((a.split, b.split)))}"


def audit(
    entries: list[Entry],
    max_hamming: int = 10,
    min_corr: float = 0.95,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (report, pair rows). Pairs are exact (pixel-identical) or verified near-duplicates."""
    n = len(entries)
    items = [e.item for e in entries]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for it in items:
        counts[it.split][it.label] += 1

    # ---- exact duplicates (identical decoded pixels) --------------------------------------------------------
    by_pixels: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        by_pixels[e.pixel_md5].append(i)
    exact_groups = [g for g in by_pixels.values() if len(g) > 1]

    # ---- near duplicates: pHash candidates verified by thumbnail correlation ---------------------------------
    hashes = np.array([e.phash for e in entries], dtype=np.uint64)
    thumbs = np.stack([e.thumb for e in entries]) if n else np.zeros((0, 1024), np.float32)
    exact_pairs = {(min(a, b), max(a, b)) for g in exact_groups for a in g for b in g if a < b}
    pairs: list[dict[str, Any]] = []
    uf = UnionFind(n)
    for i, j, d in hamming_pairs(hashes, max_hamming):
        corr = float(thumbs[i] @ thumbs[j])
        exact = (i, j) in exact_pairs
        if exact or corr >= min_corr:
            uf.union(i, j)
            pairs.append(
                {
                    "a": items[i].path.as_posix(),
                    "b": items[j].path.as_posix(),
                    "split_a": items[i].split,
                    "split_b": items[j].split,
                    "label_a": items[i].label,
                    "label_b": items[j].label,
                    "kind": "exact" if exact else "near",
                    "relation": _pair_kind(items[i], items[j]),
                    "phash_distance": d,
                    "correlation": round(corr, 4),
                }
            )
    for g in exact_groups:  # exact duplicates can exceed the pHash radius only in pathological cases
        for a in g[1:]:
            uf.union(g[0], a)

    groups = [g for g in uf.groups() if len(g) > 1]
    label_conflicts = [g for g in groups if len({items[i].label for i in g}) > 1]

    # ---- leakage: evaluation images with a (near-)copy in train ------------------------------------------------
    split_names = sorted(counts)
    leakage: dict[str, Any] = {}
    train_like = "train" if "train" in counts else None
    group_of = {i: gi for gi, g in enumerate(groups) for i in g}
    if train_like:
        for split in split_names:
            if split == train_like:
                continue
            idx = [i for i, it in enumerate(items) if it.split == split]
            leaked = [
                i
                for i in idx
                if i in group_of and any(items[j].split == train_like for j in groups[group_of[i]] if j != i)
            ]
            exact_leaked = [
                i
                for i in idx
                if any(items[j].split == train_like for j in by_pixels[entries[i].pixel_md5] if j != i)
            ]
            per_label = Counter(items[i].label for i in leaked)
            leakage[split] = {
                "images": len(idx),
                "with_exact_copy_in_train": len(exact_leaked),
                "with_exact_or_near_copy_in_train": len(leaked),
                "percent_leaked": round(100.0 * len(leaked) / max(len(idx), 1), 2),
                "leaked_by_label": dict(sorted(per_label.items())),
            }

    sizes = Counter((e.width, e.height) for e in entries)
    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "images": n,
        "counts": {s: dict(sorted(c.items())) for s, c in sorted(counts.items())},
        "image_sizes": {
            "unique": len(sizes),
            "most_common": [{"width": w, "height": h, "count": c} for (w, h), c in sizes.most_common(8)],
            "min_side": min((min(e.width, e.height) for e in entries), default=0),
            "max_side": max((max(e.width, e.height) for e in entries), default=0),
        },
        "exact_duplicates": {
            "groups": len(exact_groups),
            "images_involved": sum(len(g) for g in exact_groups),
            "redundant_copies": sum(len(g) - 1 for g in exact_groups),
        },
        "near_duplicates": {
            "criteria": {"phash_max_hamming": max_hamming, "min_thumbnail_correlation": min_corr},
            "pairs": len(pairs),
            "pairs_cross_split": sum(p["relation"] != "same-split" for p in pairs),
            "clusters": len(groups),
            "images_in_clusters": sum(len(g) for g in groups),
        },
        "label_conflicts": {
            "clusters": len(label_conflicts),
            "examples": [[items[i].path.as_posix() for i in g][:4] for g in label_conflicts[:10]],
        },
        "leakage": leakage,
    }
    return report, pairs


def render_markdown(report: dict[str, Any], title: str) -> str:
    labels = sorted({k for c in report["counts"].values() for k in c})
    lines = [
        f"# Dataset audit — {title}",
        "",
        f"Generated {report['generated_at']} · {report['images']} images",
        "",
    ]
    lines += [
        "## Class balance",
        "",
        "| split | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for split, c in report["counts"].items():
        lines.append(f"| {split} | " + " | ".join(str(c.get(lbl, 0)) for lbl in labels) + " |")
    ex, nd, lc = report["exact_duplicates"], report["near_duplicates"], report["label_conflicts"]
    lines += [
        "",
        "## Duplicates",
        "",
        f"- Exact (pixel-identical) duplicate groups: **{ex['groups']}** "
        f"({ex['redundant_copies']} redundant copies)",
        f"- Verified near-duplicate pairs: **{nd['pairs']}** ({nd['pairs_cross_split']} across splits), "
        f"forming {nd['clusters']} clusters",
        f"- Clusters whose members carry **different labels**: **{lc['clusters']}**",
        "",
        "## Train → evaluation leakage",
        "",
    ]
    if report["leakage"]:
        lines += [
            "| split | images | exact copy in train | exact or near copy in train | % leaked |",
            "|---|---|---|---|---|",
        ]
        for split, lk in report["leakage"].items():
            lines.append(
                f"| {split} | {lk['images']} | {lk['with_exact_copy_in_train']} | "
                f"{lk['with_exact_or_near_copy_in_train']} | {lk['percent_leaked']}% |"
            )
    else:
        lines.append("_No train split found — leakage not computed._")
    sz = report["image_sizes"]
    lines += [
        "",
        "## Image sizes",
        "",
        f"{sz['unique']} distinct sizes (sides {sz['min_side']}–{sz['max_side']} px). Most common: "
        + ", ".join(f"{s['width']}×{s['height']} ({s['count']})" for s in sz["most_common"][:5]),
        "",
    ]
    return "\n".join(lines)


def run_audit(
    data: str | Path,
    out_dir: str | Path,
    layout: str = "auto",
    max_hamming: int = 10,
    min_corr: float = 0.95,
    workers: int = 8,
) -> dict[str, Any]:
    data = Path(data)
    if layout == "auto":
        try:
            from btd.data.brisc import find_brisc_root

            find_brisc_root(data)
            layout = "brisc"
        except FileNotFoundError:
            layout = "folders"
    items = scan_brisc(data) if layout == "brisc" else scan_folders(data)
    LOGGER.info("Auditing %d images (%s layout) from %s", len(items), layout, data)
    entries = compute_entries(items, workers=workers)
    report, pairs = audit(entries, max_hamming=max_hamming, min_corr=min_corr)
    report["source"] = {"path": data.as_posix(), "layout": layout}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit.json", report)
    (out / "audit.md").write_text(render_markdown(report, data.name), encoding="utf-8")
    with (out / "duplicate_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "kind",
            "relation",
            "phash_distance",
            "correlation",
            "split_a",
            "label_a",
            "a",
            "split_b",
            "label_b",
            "b",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(pairs, key=lambda p: (p["relation"] == "same-split", -p["correlation"])))
    return report
