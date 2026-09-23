"""Latency / size / GPU-memory benchmark across precisions and execution providers.

Every row runs the *serving* engine (letterbox → ONNX Runtime → numpy decode + masks) on real test slices, so
the numbers are end-to-end request latencies minus HTTP overhead. Rows whose requested provider is not
actually active (ORT silently falls back to CPU) are reported as unavailable instead of producing misleading
numbers. Optional ``.engine`` files (TensorRT, built by ``btd export`` with tensorrt.enabled) are timed through
Ultralytics' runtime.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from btd.inference.engine import PROVIDER_ALIASES, SegmentationEngine
from btd.training.metrics import latency_summary
from btd.utils import imread, read_json, utc_now, write_json

LOGGER = logging.getLogger(__name__)


def _gpu_used_mb() -> float | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        used = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
        pynvml.nvmlShutdown()
        return float(used)
    except Exception:
        return None


def _sample_images(dataset_dir: str | Path | None, n: int) -> list[np.ndarray]:
    if dataset_dir:
        from btd.training.evaluate import load_split

        samples = load_split(dataset_dir, "test")
        step = max(1, len(samples) // n)
        return [imread(s.image) for s in samples[::step][:n]]
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (512, 512, 3), dtype=np.uint8) for _ in range(n)]


def time_engine(
    engine: SegmentationEngine, images: list[np.ndarray], runs: int, warmup: int
) -> dict[str, Any]:
    for i in range(warmup):
        engine.predict(images[i % len(images)])
    total, infer = [], []
    t0 = time.perf_counter()
    for i in range(runs):
        pred = engine.predict(images[i % len(images)])
        total.append(pred.timings_ms["total_ms"])
        infer.append(pred.timings_ms["inference_ms"])
    wall = time.perf_counter() - t0
    return {
        "total": latency_summary(total),
        "inference": latency_summary(infer),
        "throughput_img_s": round(runs / wall, 2),
    }


def time_ultralytics(model_file: Path, images: list[np.ndarray], runs: int, warmup: int) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO(str(model_file), task="segment")
    for i in range(warmup):
        model.predict(images[i % len(images)], verbose=False)
    total: list[float] = []
    infer: list[float] = []
    for i in range(runs):
        results: Any = model.predict(images[i % len(images)], verbose=False, retina_masks=True)
        speed: dict[str, float] = results[0].speed
        total.append(float(sum(v or 0.0 for v in speed.values())))
        infer.append(float(speed["inference"]))
    return {"total": latency_summary(total), "inference": latency_summary(infer)}


def run_benchmark(
    model_dir: str | Path,
    dataset_dir: str | Path | None = None,
    providers: tuple[str, ...] = ("cpu",),
    precisions: tuple[str, ...] | None = None,
    runs: int = 100,
    warmup: int = 10,
    n_images: int = 20,
    out_dir: str | Path = "reports",
) -> dict[str, Any]:
    model_dir = Path(model_dir)
    meta = read_json(model_dir / "model.json")
    images = _sample_images(dataset_dir, n_images)
    rows: list[dict[str, Any]] = []
    for precision, art in meta["artifacts"].items():
        if precisions and precision not in precisions:
            continue
        path = model_dir / art["file"]
        for prov in providers:
            wanted = PROVIDER_ALIASES.get(prov.lower(), prov)
            row: dict[str, Any] = {
                "precision": precision,
                "provider": prov,
                "size_mb": round(art["bytes"] / 1e6, 2),
            }
            before = _gpu_used_mb() if wanted != "CPUExecutionProvider" else None
            try:
                engine = SegmentationEngine(path, providers=prov)
            except Exception as exc:
                rows.append({**row, "status": f"load failed: {exc}"})
                continue
            if engine.providers[0] != wanted:
                rows.append({**row, "status": f"unavailable (active: {engine.providers[0]})"})
                continue
            LOGGER.info("Benchmarking %s on %s ...", art["file"], engine.providers[0])
            row.update(time_engine(engine, images, runs, warmup))
            after = _gpu_used_mb() if before is not None else None
            if before is not None and after is not None:
                row["gpu_memory_mb"] = round(after - before, 1)
            row["status"] = "ok"
            rows.append(row)
            del engine
    for engine_file in sorted(model_dir.glob("*.engine")):
        try:
            LOGGER.info("Benchmarking TensorRT engine %s ...", engine_file.name)
            rows.append(
                {
                    "precision": engine_file.stem.replace("model_", "") + " (TensorRT)",
                    "provider": "tensorrt-engine",
                    "size_mb": round(engine_file.stat().st_size / 1e6, 2),
                    **time_ultralytics(engine_file, images, runs, warmup),
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {"precision": engine_file.name, "provider": "tensorrt-engine", "status": f"failed: {exc}"}
            )

    report = {"generated_at": utc_now(), "model": meta.get("name"), "runs": runs, "rows": rows}
    out = Path(out_dir)
    write_json(out / "benchmark.json", report)
    (out / "benchmark.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Inference benchmark — {report.get('model')}",
        "",
        f"{report['runs']} timed requests per row, end-to-end (pre-process + inference + decode + masks).",
        "",
        "| precision | provider | size (MB) | p50 ms | p95 ms | inference p50 ms | img/s | GPU mem (MB) | status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in report["rows"]:
        t, i = r.get("total", {}), r.get("inference", {})
        lines.append(
            f"| {r['precision']} | {r['provider']} | {r.get('size_mb', '')} | {t.get('p50_ms', '')} | "
            f"{t.get('p95_ms', '')} | {i.get('p50_ms', '')} | {r.get('throughput_img_s', '')} | "
            f"{r.get('gpu_memory_mb', '')} | {r.get('status', '')} |"
        )
    return "\n".join(lines) + "\n"
