"""Command-line interface: ``btd <command>`` (run ``btd -h``).

Heavy dependencies (torch, ultralytics) are imported lazily inside each command so the serving container, which
has neither, can still run ``btd serve`` / ``btd predict``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from btd import __version__
from btd.utils import setup_logging

LOGGER = logging.getLogger("btd.cli")
DEFAULT_RAW = "data/raw"
DEFAULT_DATASET = "data/processed/brisc-yolo"
DEFAULT_MODEL_DIR = "artifacts/model"


# ---------------------------------------------------------------------------------------------------------- env
def cmd_env(args: argparse.Namespace) -> int:
    from btd.training.train import environment_report

    info = environment_report()
    hints: list[str] = []
    cuda = str(info.get("torch_cuda") or "")
    py_minor = sys.version_info[:2]
    if info.get("torch") and not info.get("cuda_available"):
        hints.append(
            "torch cannot see a CUDA GPU: training will be CPU-only. Check the NVIDIA driver / torch build."
        )
    providers = info.get("onnxruntime_providers") or []
    if info.get("cuda_available") and "CUDAExecutionProvider" not in providers:
        if cuda.startswith("12"):
            pin = "onnxruntime-gpu>=1.20,<1.27" if py_minor >= (3, 11) else "onnxruntime-gpu>=1.20,<1.24"
        else:
            pin = "onnxruntime-gpu>=1.27"
        hints.append(
            f'GPU inference: pip uninstall -y onnxruntime && pip install "{pin}"  (matches torch CUDA {cuda})'
        )
    vram = info.get("gpu_vram_gb")
    if vram and vram < 7:
        hints.append(f"{vram} GB VRAM: keep batch<=16 for yolo26s-seg @640 (batch 8 for yolo26m-seg).")
    info["hints"] = hints
    print(json.dumps(info, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------------------------------------------- data
def cmd_data_download(args: argparse.Namespace) -> int:
    from btd.data.download import download_brisc

    root = download_brisc(args.dest, zip_path=args.zip, keep_zip=not args.delete_zip)
    LOGGER.info("BRISC extracted under %s - next: btd data prepare --src %s", root, root)
    return 0


def cmd_data_synthetic(args: argparse.Namespace) -> int:
    from btd.data.synthetic import generate

    root = generate(args.out, args.train_per_class, args.test_per_class, args.size, args.seed)
    LOGGER.info("Synthetic BRISC-shaped dataset written to %s", root)
    return 0


def cmd_data_audit(args: argparse.Namespace) -> int:
    from btd.data.audit import run_audit

    report = run_audit(args.data, args.out, args.layout, args.max_hamming, args.min_corr, args.workers)
    summary = {
        "images": report["images"],
        "counts": report["counts"],
        "exact_duplicate_groups": report["exact_duplicates"]["groups"],
        "near_duplicate_pairs": report["near_duplicates"]["pairs"],
        "cross_split_pairs": report["near_duplicates"]["pairs_cross_split"],
        "label_conflict_clusters": report["label_conflicts"]["clusters"],
        "leakage": report["leakage"],
    }
    print(json.dumps(summary, indent=2))
    LOGGER.info("Full report: %s", Path(args.out) / "audit.md")
    return 0


def cmd_data_prepare(args: argparse.Namespace) -> int:
    from btd.data.prepare import prepare_dataset

    report = prepare_dataset(
        args.src,
        args.out,
        val_fraction=args.val_fraction,
        seed=args.seed,
        overwrite=args.overwrite,
        verify_checksums=not args.no_verify,
        workers=args.workers,
    )
    print(json.dumps({k: report[k] for k in ("counts", "polygon_fidelity_iou", "manifest_check")}, indent=2))
    return 0


# ---------------------------------------------------------------------------------------------------------- train
def cmd_train(args: argparse.Namespace) -> int:
    from btd.training.train import load_config, train

    summary = train(load_config(args.config, args.overrides))
    print(
        json.dumps(
            {k: summary[k] for k in ("best_weights", "epochs_completed", "peak_gpu_memory_reserved_gb")},
            indent=2,
        )
    )
    return 0


def latest_weights(runs_dir: str | Path = "runs/segment") -> Path:
    """Most recently written ``best.pt`` under ``runs_dir`` (Ultralytics names re-runs name2, name3, ...).

    The search is recursive: runs made before `btd train` pinned `project` to an absolute path landed in a nested
    ``runs/segment/runs/segment/<name>/`` folder.
    """
    candidates = sorted(Path(runs_dir).rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No */weights/best.pt under {runs_dir} - train first or pass --weights")
    return candidates[-1]


def cmd_export(args: argparse.Namespace) -> int:
    from btd.export.pipeline import export_model

    if args.weights is None:
        args.weights = latest_weights()
        LOGGER.info("Using the newest checkpoint: %s", args.weights)
    meta = export_model(
        args.weights, args.data, args.out, args.config, args.name, args.version, args.device, args.imgsz
    )
    print(json.dumps({k: meta[k] for k in ("name", "layout", "thresholds", "artifacts")}, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from btd.training.evaluate import (
        evaluate_engine,
        plot_confusion_matrix,
        render_markdown,
        ultralytics_metrics,
    )
    from btd.utils import read_json, write_json

    model_dir, out = Path(args.model_dir), Path(args.out)
    meta_file = model_dir / "model.json"
    meta = read_json(meta_file)
    reports: list[dict[str, Any]] = []
    for precision, art in meta["artifacts"].items():
        if args.precisions and precision not in args.precisions:
            continue
        rep = evaluate_engine(model_dir / art["file"], args.data, split=args.split, providers=args.providers)
        if args.map:
            rep["map"] = ultralytics_metrics(
                model_dir / art["file"], Path(args.data) / "data.yaml", args.split, meta["imgsz"][0]
            )
        reports.append(rep)
        plot_confusion_matrix(rep, out / f"confusion_{args.split}_{precision}.png")
    write_json(out / f"evaluation_{args.split}.json", reports)
    (out / f"evaluation_{args.split}.md").write_text(
        render_markdown(reports, f"Evaluation — {meta['name']} on {Path(args.data).name} ({args.split})"),
        encoding="utf-8",
    )
    if args.split == "test":
        meta["metrics"] = {
            "test": {
                r["precision"]: {
                    "accuracy": r["classification"]["accuracy"],
                    "macro_f1": r["classification"]["macro_f1"],
                    "sensitivity": r["screening"]["sensitivity"],
                    "specificity": r["screening"]["specificity"],
                    "mean_dice": r["segmentation"]["mean_dice"],
                    "weighted_dataset_iou": r["segmentation"]["weighted_dataset_iou"],
                    **({"mask_map50_95": r["map"]["mask_map50_95"]} if "map" in r else {}),
                }
                for r in reports
            }
        }
        write_json(meta_file, meta)
    print((out / f"evaluation_{args.split}.md").read_text(encoding="utf-8"))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    from btd.export.benchmark import run_benchmark

    report = run_benchmark(
        args.model_dir,
        args.data,
        providers=tuple(args.providers),
        precisions=tuple(args.precisions) if args.precisions else None,
        runs=args.runs,
        warmup=args.warmup,
        out_dir=args.out,
    )
    print((Path(args.out) / "benchmark.md").read_text(encoding="utf-8"))
    return 0 if any(r.get("status") == "ok" for r in report["rows"]) else 1


def cmd_package(args: argparse.Namespace) -> int:
    """Copy one precision + a trimmed model.json into ``models/`` (what the Docker image serves)."""
    from btd.utils import read_json, sha256_file, write_json

    src, dest = Path(args.model_dir), Path(args.dest)
    meta = read_json(src / "model.json")
    art = meta["artifacts"].get(args.precision)
    if art is None:
        LOGGER.error("Precision %s not exported (have: %s)", args.precision, ", ".join(meta["artifacts"]))
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "model.onnx"
    shutil.copyfile(src / art["file"], target)
    meta["artifacts"] = {args.precision: {**art, "file": target.name, "sha256": sha256_file(target)}}
    write_json(dest / "model.json", meta)
    LOGGER.info("Packaged %s (%s) -> %s", art["file"], args.precision, target)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from btd.inference.engine import SegmentationEngine
    from btd.inference.quality import input_warnings
    from btd.inference.visualize import draw_overlay
    from btd.utils import imread, imwrite

    engine = SegmentationEngine(args.model, providers=args.providers)
    for path in args.images:
        img = imread(path)
        pred = engine.predict(img, conf=args.conf)
        result = {
            "image": str(path),
            "decision": pred.decision.label,
            "score": round(pred.decision.score, 4),
            "detections": [
                {
                    "class": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "box": [round(v, 1) for v in d.box],
                    "area_px": d.area_px,
                }
                for d in pred.detections
            ],
            "timings_ms": {k: round(v, 1) for k, v in pred.timings_ms.items()},
            "warnings": [{"code": w.code, "message": w.message} for w in input_warnings(img)],
        }
        print(json.dumps(result))
        if args.save_dir:
            imwrite(Path(args.save_dir) / f"{Path(path).stem}_overlay.png", draw_overlay(img, pred))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import os

    import uvicorn

    if args.model:
        os.environ["BTD_MODEL_PATH"] = str(args.model)
    uvicorn.run(
        "btd.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="warning",
        proxy_headers=True,
        forwarded_allow_ips=args.forwarded_allow_ips,
    )
    return 0


# ---------------------------------------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="btd", description="Brain tumour detection: data -> train -> quantise -> serve"
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="report torch/CUDA/GPU/onnxruntime setup + fixes").set_defaults(func=cmd_env)

    data = sub.add_parser("data", help="dataset commands").add_subparsers(dest="data_command", required=True)
    d = data.add_parser("download", help="download + verify + extract BRISC 2025 from Zenodo")
    d.add_argument("--dest", default=DEFAULT_RAW)
    d.add_argument("--zip", help="use an already downloaded brisc2025.zip (e.g. from Kaggle)")
    d.add_argument("--delete-zip", action="store_true")
    d.set_defaults(func=cmd_data_download)

    s = data.add_parser("synthetic", help="write a tiny synthetic BRISC-shaped dataset (CI / smoke tests)")
    s.add_argument("--out", default="data/raw/synthetic")
    s.add_argument("--train-per-class", type=int, default=24)
    s.add_argument("--test-per-class", type=int, default=8)
    s.add_argument("--size", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_data_synthetic)

    a = data.add_parser("audit", help="duplicates, near-duplicates and train/test leakage report")
    a.add_argument("--data", required=True, help="BRISC root or <split>/<class>/ folder dataset")
    a.add_argument("--layout", choices=["auto", "brisc", "folders"], default="auto")
    a.add_argument("--out", default="reports/audit")
    a.add_argument("--max-hamming", type=int, default=10, help="pHash candidate radius (bits of 64)")
    a.add_argument(
        "--min-corr", type=float, default=0.95, help="thumbnail correlation to confirm a near-duplicate"
    )
    a.add_argument("--workers", type=int, default=8)
    a.set_defaults(func=cmd_data_audit)

    pr = data.add_parser("prepare", help="BRISC -> YOLO-seg dataset with leakage-safe val split")
    pr.add_argument("--src", default=DEFAULT_RAW)
    pr.add_argument("--out", default=DEFAULT_DATASET)
    pr.add_argument("--val-fraction", type=float, default=0.15)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--overwrite", action="store_true")
    pr.add_argument("--no-verify", action="store_true", help="skip manifest SHA-256 verification")
    pr.add_argument("--workers", type=int, default=8)
    pr.set_defaults(func=cmd_data_prepare)

    t = sub.add_parser("train", help="train YOLO-seg (extra key=value args override the config)")
    t.add_argument("--config", default="configs/train.yaml")
    t.add_argument("overrides", nargs="*", help="e.g. epochs=50 batch=8 model=yolo26n-seg.pt")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("export", help="ONNX FP32/FP16/INT8 (+TensorRT) export, threshold tuning, model.json")
    e.add_argument(
        "--weights", default=None, help="best.pt to export (default: newest runs/segment/*/weights/best.pt)"
    )
    e.add_argument("--data", default=DEFAULT_DATASET)
    e.add_argument("--out", default=DEFAULT_MODEL_DIR)
    e.add_argument("--config", default="configs/export.yaml")
    e.add_argument("--name")
    e.add_argument("--version")
    e.add_argument("--device", default=None, help="device for head selection val, e.g. 0 or cpu")
    e.add_argument(
        "--imgsz", type=int, default=None, help="override configs/export.yaml imgsz (use the train size)"
    )
    e.set_defaults(func=cmd_export)

    ev = sub.add_parser("evaluate", help="score exported models on a split (default: test)")
    ev.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ev.add_argument("--data", default=DEFAULT_DATASET)
    ev.add_argument("--split", default="test", choices=["val", "test"])
    ev.add_argument("--precisions", nargs="*", choices=["fp32", "fp16", "int8"])
    ev.add_argument("--providers", default="cpu")
    ev.add_argument(
        "--map", action="store_true", help="also compute Ultralytics box/mask mAP for each ONNX file"
    )
    ev.add_argument("--out", default="reports")
    ev.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("benchmark", help="latency/size/GPU memory per precision and provider")
    b.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    b.add_argument("--data", default=DEFAULT_DATASET, help="dataset dir for real test images ('' = random)")
    b.add_argument("--providers", nargs="+", default=["cpu"], help="cpu cuda tensorrt")
    b.add_argument("--precisions", nargs="*", choices=["fp32", "fp16", "int8"])
    b.add_argument("--runs", type=int, default=100)
    b.add_argument("--warmup", type=int, default=10)
    b.add_argument("--out", default="reports")
    b.set_defaults(func=cmd_benchmark)

    pk = sub.add_parser("package", help="copy one exported precision into models/ for the Docker image")
    pk.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    pk.add_argument("--precision", default="int8", choices=["fp32", "fp16", "int8"])
    pk.add_argument("--dest", default="models")
    pk.set_defaults(func=cmd_package)

    pd = sub.add_parser("predict", help="run the ONNX engine on image files")
    pd.add_argument("--model", required=True)
    pd.add_argument("images", nargs="+")
    pd.add_argument("--conf", type=float, default=None)
    pd.add_argument("--providers", default="auto")
    pd.add_argument("--save-dir", default=None, help="write overlay PNGs here")
    pd.set_defaults(func=cmd_predict)

    sv = sub.add_parser("serve", help="start the FastAPI service (settings via BTD_* env vars)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--workers", type=int, default=1)
    sv.add_argument("--model", default=None, help="shortcut for BTD_MODEL_PATH")
    sv.add_argument("--forwarded-allow-ips", default="127.0.0.1")
    sv.set_defaults(func=cmd_serve)
    return p


def _safe_stdio() -> None:
    """Never crash on characters the console code page can't encode (e.g. cp1252 when output is redirected)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def main(argv: list[str] | None = None) -> int:
    _safe_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    if getattr(args, "data", None) == "":
        args.data = None
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
