"""End-to-end export: choose the head, export ONNX, quantise, tune the operating threshold, write ``model.json``.

``model.json`` is the contract between training and serving: class names, input size, head layout, the tuned
operating threshold and a SHA-256 for every artefact (the server refuses a model whose checksum does not match).
"""

from __future__ import annotations

import csv
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

from btd import __version__
from btd.constants import BRISC_DOI, DISCLAIMER, NO_TUMOR, TUMOR_CLASSES
from btd.export.quantize import convert_fp16, quantize_int8, set_metadata
from btd.utils import read_json, sha256_file, utc_now, write_json

LOGGER = logging.getLogger(__name__)
ARTIFACT_NAMES = {"fp32": "model_fp32.onnx", "fp16": "model_fp16.onnx", "int8": "model_int8.onnx"}


def load_export_config(path: str | Path | None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "imgsz": 640,
        "head": "auto",
        "precisions": ["fp32", "fp16", "int8"],
        "int8": {
            "calibration_images": 256,
            "method": "minmax",
            "per_channel": True,
            "keep_head_fp32": False,
            "target": "cpu",
        },
        "tensorrt": {"enabled": False, "precision": "fp16", "workspace_gb": 2},
    }
    if path:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(defaults.get(k), dict):
                defaults[k].update(v)
            else:
                defaults[k] = v
    return defaults


def calibration_images(dataset_dir: str | Path, n: int, seed: int = 0) -> list[Path]:
    """Stratified sample (equal share per label incl. no-tumour) from the TRAIN split only."""
    root = Path(dataset_dir)
    with (root / "splits.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"]
    by_label: dict[str, list[Path]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(root / r["image"])
    rng = random.Random(seed)
    per_label = max(1, n // max(len(by_label), 1))
    picked: list[Path] = []
    for label in sorted(by_label):
        paths = sorted(by_label[label])
        rng.shuffle(paths)
        picked.extend(paths[:per_label])
    return picked


def has_end2end_head(weights: str | Path) -> bool:
    from ultralytics import YOLO

    net: Any = YOLO(str(weights)).model
    return getattr(net.model[-1], "one2one_cv2", None) is not None


def checkpoint_architecture(weights: str | Path) -> str | None:
    """Architecture the checkpoint was built from (e.g. ``yolo26s-seg``), read from its own model config.

    ``run_summary.json`` and the checkpoint's ``train_args`` only know the file training was started from, which is
    ``.../last.pt`` for a resumed run.
    """
    from ultralytics import YOLO

    yaml_file = (getattr(YOLO(str(weights)).model, "yaml", None) or {}).get("yaml_file")
    return Path(str(yaml_file)).stem if yaml_file else None


def choose_head(weights: str | Path, data_yaml: str | Path, imgsz: int, device: Any = None) -> dict[str, Any]:
    """Score both YOLO26 heads on the validation split and keep the better mask mAP50-95."""
    from btd.training.evaluate import ultralytics_metrics

    if not has_end2end_head(weights):
        return {"layout": "raw", "reason": "model has no NMS-free head (YOLOv8/YOLO11)"}
    raw = ultralytics_metrics(weights, data_yaml, "val", imgsz, nms=None, device=device)
    e2e = ultralytics_metrics(weights, data_yaml, "val", imgsz, nms=False, device=device)
    layout = "end2end" if e2e["mask_map50_95"] >= raw["mask_map50_95"] - 0.002 else "raw"
    LOGGER.info(
        "Head selection on val - NMS-free: %.4f, one-to-many+NMS: %.4f mask mAP50-95 -> %s",
        e2e["mask_map50_95"],
        raw["mask_map50_95"],
        layout,
    )
    return {
        "layout": layout,
        "val_end2end": e2e,
        "val_raw": raw,
        "reason": "higher val mask mAP50-95 (ties -> NMS-free)",
    }


def export_onnx_fp32(weights: str | Path, out_file: Path, imgsz: int, layout: str) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    produced = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            nms=False if layout == "end2end" else None,
            simplify=True,
            device="cpu",
        )
    )
    out_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), out_file)
    set_metadata(out_file, {"btd_precision": "fp32", "btd_layout": layout})
    return out_file


def export_tensorrt(
    weights: str | Path, data_yaml: str | Path, out_dir: Path, imgsz: int, cfg: dict[str, Any]
) -> Path:
    """GPU-specific TensorRT engine (FP16 or INT8 with Ultralytics' calibrator). Not portable across GPUs."""
    from ultralytics import YOLO

    precision = cfg.get("precision", "fp16")
    kwargs: dict[str, Any] = {
        "format": "engine",
        "imgsz": imgsz,
        "device": 0,
        "workspace": cfg.get("workspace_gb", 2),
        "quantize": 8 if precision == "int8" else 16,
    }
    if precision == "int8":
        kwargs["data"] = str(data_yaml)
    produced = Path(YOLO(str(weights)).export(**kwargs))
    dst = out_dir / f"model_{precision}.engine"
    shutil.move(str(produced), dst)
    return dst


def export_model(
    weights: str | Path,
    dataset_dir: str | Path,
    out_dir: str | Path,
    config_path: str | Path | None = None,
    name: str | None = None,
    version: str | None = None,
    device: Any = None,
    imgsz: int | None = None,
) -> dict[str, Any]:
    from btd.training.evaluate import evaluate_engine

    cfg = load_export_config(config_path)
    if imgsz:
        cfg["imgsz"] = imgsz
    weights, dataset_dir, out_dir = Path(weights), Path(dataset_dir), Path(out_dir)
    data_yaml = dataset_dir / "data.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    imgsz = int(cfg["imgsz"])

    head = (
        choose_head(weights, data_yaml, imgsz, device)
        if cfg["head"] == "auto"
        else {"layout": cfg["head"], "reason": "set in config"}
    )
    layout = head["layout"]

    fp32 = export_onnx_fp32(weights, out_dir / ARTIFACT_NAMES["fp32"], imgsz, layout)
    artifacts: dict[str, Path] = {"fp32": fp32}
    if "fp16" in cfg["precisions"]:
        artifacts["fp16"] = convert_fp16(fp32, out_dir / ARTIFACT_NAMES["fp16"])
    if "int8" in cfg["precisions"]:
        q = cfg["int8"]
        calib = calibration_images(dataset_dir, int(q["calibration_images"]))
        artifacts["int8"] = quantize_int8(
            fp32,
            out_dir / ARTIFACT_NAMES["int8"],
            calib,
            per_channel=bool(q["per_channel"]),
            method=str(q["method"]),
            keep_head_fp32=bool(q["keep_head_fp32"]),
            target=str(q["target"]),
        )

    # Operating threshold: tuned ONCE on validation with the FP32 model, shared by every precision.
    LOGGER.info("Tuning the image-level operating threshold on the validation split...")
    val_report = evaluate_engine(fp32, dataset_dir, split="val", tune=True, providers="cpu")
    threshold = float(val_report["threshold"])

    engines: dict[str, Any] = {}
    if cfg["tensorrt"].get("enabled"):
        try:
            engines["tensorrt"] = export_tensorrt(weights, data_yaml, out_dir, imgsz, cfg["tensorrt"]).name
        except Exception as exc:  # TensorRT is optional and hardware-specific
            LOGGER.warning("TensorRT export skipped: %s", exc)

    run_summary_file = weights.parent.parent / "run_summary.json"
    run_summary = read_json(run_summary_file) if run_summary_file.is_file() else {}
    arch = checkpoint_architecture(weights) or Path(str(run_summary.get("model", weights.stem))).stem
    meta = {
        "schema_version": 1,
        "name": name or f"brisc-{arch}",
        "version": version or __version__,
        "created_at": utc_now(),
        "task": "segment",
        "architecture": arch,
        "class_names": list(TUMOR_CLASSES),
        "no_tumor_label": NO_TUMOR,
        "imgsz": [imgsz, imgsz],
        "layout": layout,
        "head_selection": head,
        "thresholds": {
            "conf": threshold,
            "iou": 0.7,
            "tuned_on": "val",
            "objective": "macro-F1 (4 image-level classes)",
        },
        "artifacts": {
            p: {"file": f.name, "sha256": sha256_file(f), "bytes": f.stat().st_size}
            for p, f in artifacts.items()
        },
        "engines": engines,
        "validation": {
            "classification": val_report["classification"],
            "screening": val_report["screening"],
            "segmentation": {k: val_report["segmentation"][k] for k in ("mean_dice", "weighted_dataset_iou")},
        },
        "training": {
            k: run_summary.get(k)
            for k in (
                "model",
                "epochs_completed",
                "best_fitness",
                "peak_gpu_memory_reserved_gb",
                "git_commit",
            )
            if k in run_summary
        },
        "dataset": {"name": "BRISC 2025", "doi": BRISC_DOI, "license": "CC BY 4.0"},
        "intended_use": DISCLAIMER,
    }
    write_json(out_dir / "model.json", meta)
    write_json(out_dir / "val_threshold_sweep.json", val_report.get("threshold_sweep", []))
    LOGGER.info("Exported %s to %s (threshold %.2f)", ", ".join(artifacts), out_dir, threshold)
    return meta
