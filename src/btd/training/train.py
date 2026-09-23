"""Thin, reproducible wrapper around Ultralytics training.

Adds what a bare ``model.train()`` call lacks for a portfolio/production pipeline: config files with CLI
overrides, environment + GPU memory reporting (to prove the 6 GB budget), dataset provenance, and a
machine-readable ``run_summary.json`` that the export step picks up.
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any

import yaml

from btd.utils import git_commit, read_json, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def parse_overrides(overrides: list[str]) -> dict[str, Any]:
    """``["epochs=50", "batch=8", "cache=false"]`` → typed dict (values parsed as YAML scalars)."""
    out: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key=value")
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg.update(parse_overrides(overrides or []))
    return cfg


def environment_report() -> dict[str, Any]:
    """Versions and GPU details (safe to call without CUDA)."""
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_vram_gb"] = round(props.total_memory / 1024**3, 2)
            info["gpu_compute_capability"] = f"{props.major}.{props.minor}"
            info["cudnn"] = torch.backends.cudnn.version()
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime"] = ort.__version__
        info["onnxruntime_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime"] = None
    try:
        import tensorrt

        info["tensorrt"] = tensorrt.__version__
    except ImportError:
        info["tensorrt"] = None
    return info


def train(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    cfg = dict(config)
    model_name = cfg.pop("model")
    data_yaml = Path(cfg["data"])
    if not data_yaml.is_file():
        raise FileNotFoundError(f"{data_yaml} not found - run `btd data prepare` first")
    cfg["data"] = str(data_yaml.resolve())
    project = cfg.get("project")
    if project and not Path(project).is_absolute():
        # Ultralytics >=8.4 nests a relative `project` under its own runs_dir/<task>/ (runs/segment/runs/...).
        cfg["project"] = str(Path(project).resolve())

    env = environment_report()
    LOGGER.info("Environment: %s", {k: env.get(k) for k in ("torch", "torch_cuda", "gpu", "gpu_vram_gb")})
    if str(cfg.get("device", "")) not in {"cpu", "mps"} and not env.get("cuda_available"):
        LOGGER.warning(
            "CUDA is not available in this environment - training on CPU (slow). Is torch CUDA-enabled?"
        )
        cfg["device"] = "cpu"
        cfg["amp"] = False

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(model_name)
    started = utc_now()
    results = model.train(**cfg)
    trainer = model.trainer
    assert trainer is not None
    save_dir = Path(trainer.save_dir)
    best = save_dir / "weights" / "best.pt"

    peak_vram = round(torch.cuda.max_memory_reserved() / 1024**3, 2) if torch.cuda.is_available() else None
    metrics = getattr(results, "results_dict", None) or getattr(trainer, "metrics", {}) or {}
    prepare_report = data_yaml.parent / "prepare_report.json"
    summary = {
        "started_at": started,
        "finished_at": utc_now(),
        "model": model_name,
        "run_dir": save_dir.as_posix(),
        "best_weights": best.as_posix(),
        "last_weights": (save_dir / "weights" / "last.pt").as_posix(),
        "epochs_completed": int(getattr(trainer, "epoch", -1)) + 1,
        "best_fitness": float(getattr(trainer, "best_fitness", 0.0) or 0.0),
        "val_metrics": {
            k: round(float(v), 5) for k, v in dict(metrics).items() if isinstance(v, int | float)
        },
        "peak_gpu_memory_reserved_gb": peak_vram,
        "config": {"model": model_name, **cfg},
        "dataset": read_json(prepare_report) if prepare_report.is_file() else {"data_yaml": cfg["data"]},
        "environment": env,
        "git_commit": git_commit(),
    }
    write_json(save_dir / "run_summary.json", summary)
    LOGGER.info("Training finished -> %s (peak VRAM reserved: %s GB)", best, peak_vram)
    return summary
