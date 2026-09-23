"""ONNX Runtime inference engine for YOLO segmentation models (CPU, CUDA or TensorRT execution providers)."""

from __future__ import annotations

import ast
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from btd.constants import TUMOR_CLASSES
from btd.inference.postprocess import (
    LAYOUT_END2END,
    LAYOUT_RAW,
    DecodedDetections,
    decode_end2end,
    decode_raw,
    infer_layout,
    instance_masks,
    nms,
    scale_boxes,
)
from btd.inference.preprocess import preprocess
from btd.inference.types import Detection, ImageDecision, Prediction
from btd.utils import read_json, sha256_file, timer

LOGGER = logging.getLogger(__name__)

PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7


class ModelLoadError(RuntimeError):
    """Raised when a model cannot be loaded or fails its integrity check."""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    file: str
    precision: str
    sha256: str | None
    class_names: tuple[str, ...]
    input_hw: tuple[int, int]
    layout: str
    conf_threshold: float
    iou_threshold: float
    providers: tuple[str, ...]
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "file": self.file,
            "precision": self.precision,
            "sha256": self.sha256,
            "class_names": list(self.class_names),
            "input_size": list(self.input_hw),
            "layout": self.layout,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "providers": list(self.providers),
            "extra": dict(self.extra),
        }


def resolve_providers(spec: str | Sequence[str], available: Sequence[str]) -> list[str]:
    """Turn ``"auto"`` / ``"cuda,cpu"`` / ``["tensorrt", "cuda"]`` into available ORT provider names.

    The CPU provider is always appended as the final fallback.
    """
    if isinstance(spec, str):
        items = (
            ["cuda", "cpu"] if spec.strip().lower() == "auto" else [s for s in spec.split(",") if s.strip()]
        )
    else:
        items = list(spec)
    wanted: list[str] = []
    for item in items:
        name = PROVIDER_ALIASES.get(item.strip().lower(), item.strip())
        if name in available and name not in wanted:
            wanted.append(name)
        elif name not in available:
            LOGGER.info("Execution provider %s not available (have: %s)", name, ", ".join(available))
    if "CPUExecutionProvider" not in wanted:
        wanted.append("CPUExecutionProvider")
    return wanted


def preload_cuda_libraries() -> None:
    """Make CUDA/cuDNN shared libraries visible to onnxruntime-gpu.

    In a PyTorch environment (e.g. the conda env used for training) importing torch loads its bundled CUDA/cuDNN
    libraries into the process, which ORT then reuses. Otherwise (slim GPU container) ``onnxruntime.preload_dlls``
    loads them from the ``nvidia-*`` pip packages.
    """
    try:
        import torch

        if torch.cuda.is_available():
            LOGGER.debug("CUDA libraries loaded via torch %s", torch.__version__)
            return
    except ImportError:
        pass
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception as exc:  # pragma: no cover - depends on host CUDA install
        LOGGER.debug("onnxruntime.preload_dlls failed: %s", exc)


def _parse_names(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, dict):
        return tuple(str(parsed[k]) for k in sorted(parsed))
    if isinstance(parsed, list | tuple):
        return tuple(str(v) for v in parsed)
    return None


def _guess_precision(filename: str) -> str:
    low = filename.lower()
    if "int8" in low:
        return "int8"
    if "fp16" in low or "half" in low:
        return "fp16"
    return "fp32"


class SegmentationEngine:
    """Thread-safe predictor: ``engine.predict(bgr_image) -> Prediction``."""

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        meta_path: str | os.PathLike[str] | None = None,
        providers: str | Sequence[str] = "auto",
        conf: float | None = None,
        iou: float | None = None,
        max_det: int = 100,
        agnostic: bool = True,
        intra_op_threads: int = 0,
        gpu_mem_limit_mb: int | None = None,
        verify_checksum: bool = True,
        trt_cache_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise ModelLoadError(f"Model file not found: {self.model_path}")

        meta_file = Path(meta_path) if meta_path else self.model_path.with_name("model.json")
        self.meta: dict[str, Any] = read_json(meta_file) if meta_file.is_file() else {}
        artifact = self._artifact_entry()

        self.sha256: str | None = None
        expected = artifact.get("sha256")
        if verify_checksum and expected:
            self.sha256 = sha256_file(self.model_path)
            if self.sha256 != expected:
                raise ModelLoadError(
                    f"Checksum mismatch for {self.model_path.name}: expected {expected[:12]}..., "
                    f"got {self.sha256[:12]}... (corrupted or replaced model file)"
                )

        available = ort.get_available_providers()
        chosen = resolve_providers(providers, available)
        if "CUDAExecutionProvider" in chosen or "TensorrtExecutionProvider" in chosen:
            preload_cuda_libraries()
        provider_cfg: list[Any] = []
        for p in chosen:
            if p == "CUDAExecutionProvider":
                opts: dict[str, Any] = {"arena_extend_strategy": "kSameAsRequested"}
                if gpu_mem_limit_mb:
                    opts["gpu_mem_limit"] = int(gpu_mem_limit_mb) * 1024 * 1024
                provider_cfg.append((p, opts))
            elif p == "TensorrtExecutionProvider":
                cache = Path(trt_cache_dir or self.model_path.parent / "trt_cache")
                cache.mkdir(parents=True, exist_ok=True)
                provider_cfg.append(
                    (
                        p,
                        {
                            "trt_fp16_enable": True,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": str(cache),
                        },
                    )
                )
            else:
                provider_cfg.append(p)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if intra_op_threads > 0:
            so.intra_op_num_threads = intra_op_threads
        try:
            self.session = ort.InferenceSession(str(self.model_path), sess_options=so, providers=provider_cfg)
        except Exception as exc:
            raise ModelLoadError(f"onnxruntime could not load {self.model_path}: {exc}") from exc

        onnx_meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.class_names: tuple[str, ...] = tuple(
            self.meta.get("class_names") or _parse_names(onnx_meta.get("names")) or TUMOR_CLASSES
        )
        self.nc = len(self.class_names)

        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_dtype = np.float16 if "float16" in inp.type else np.float32
        h, w = inp.shape[2], inp.shape[3]
        if not (isinstance(h, int) and isinstance(w, int)):
            size = self.meta.get("imgsz") or ast.literal_eval(onnx_meta.get("imgsz", "[640, 640]"))
            h, w = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
        self.input_hw: tuple[int, int] = (int(h), int(w))

        outputs = self.session.get_outputs()
        by_rank = {len(o.shape): o for o in outputs}
        if 3 not in by_rank or 4 not in by_rank:
            raise ModelLoadError(
                f"Expected a segmentation model with (B,*,*) detections and (B,nm,H,W) prototypes; "
                f"got outputs {[(o.name, o.shape) for o in outputs]}"
            )
        self._det_name, self._proto_name = by_rank[3].name, by_rank[4].name
        self.nm = int(by_rank[4].shape[1])
        e2e_flag = str(self.meta.get("layout") or onnx_meta.get("end2end", "")).lower()
        if e2e_flag in {"true", LAYOUT_END2END}:
            self.layout = LAYOUT_END2END
        elif e2e_flag in {"false", LAYOUT_RAW}:
            self.layout = LAYOUT_RAW
        else:
            self.layout = infer_layout(tuple(by_rank[3].shape), self.nc, self.nm)

        thresholds = self.meta.get("thresholds", {})
        self.conf_threshold = float(conf if conf is not None else thresholds.get("conf", DEFAULT_CONF))
        self.iou_threshold = float(iou if iou is not None else thresholds.get("iou", DEFAULT_IOU))
        self.max_det = int(max_det)
        self.agnostic = bool(agnostic)
        self.precision = str(
            artifact.get("precision")
            or onnx_meta.get("btd_precision")
            or _guess_precision(self.model_path.name)
        )
        self.providers = tuple(self.session.get_providers())
        if self.precision == "int8" and self.providers[0] != "CPUExecutionProvider":
            LOGGER.warning(
                "INT8 (QDQ) models are accelerated on CPU; on %s prefer the FP16 model.", self.providers[0]
            )
        LOGGER.info(
            "Loaded %s (%s, %s layout, input %sx%s) on %s",
            self.model_path.name,
            self.precision,
            self.layout,
            *self.input_hw,
            ", ".join(self.providers),
        )

    # ------------------------------------------------------------------------------------------------ helpers
    def _artifact_entry(self) -> dict[str, Any]:
        artifacts = self.meta.get("artifacts", {})
        for precision, entry in artifacts.items():
            if entry.get("file") == self.model_path.name:
                return {"precision": precision, **entry}
        return {}

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name=str(self.meta.get("name", self.model_path.stem)),
            version=str(self.meta.get("version", "unversioned")),
            file=self.model_path.name,
            precision=self.precision,
            sha256=self.sha256,
            class_names=self.class_names,
            input_hw=self.input_hw,
            layout=self.layout,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            providers=self.providers,
            extra={k: self.meta[k] for k in ("architecture", "dataset", "metrics") if k in self.meta},
        )

    def warmup(self, runs: int = 2) -> None:
        dummy = np.full((*self.input_hw, 3), 114, dtype=np.uint8)
        for _ in range(runs):
            self.predict(dummy)

    def _decode(self, det_out: np.ndarray, conf: float, iou: float) -> DecodedDetections:
        if self.layout == LAYOUT_END2END:
            dets = decode_end2end(det_out, conf, self.nm, self.max_det)
            if self.agnostic and len(dets) > 1:
                dets = dets.select(nms(dets.boxes, dets.scores, iou))
            return dets
        return decode_raw(det_out, conf, iou, self.nc, self.nm, self.max_det, agnostic=self.agnostic)

    # ------------------------------------------------------------------------------------------------ API
    def predict(
        self,
        img_bgr: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
        with_masks: bool = True,
    ) -> Prediction:
        """Run the full pipeline on one BGR (or grayscale) uint8 image."""
        timings: dict[str, float] = {}
        display_conf = self.conf_threshold if conf is None else float(conf)
        decode_conf = min(display_conf, self.conf_threshold)
        iou_thr = self.iou_threshold if iou is None else float(iou)

        with timer(timings, "preprocess_ms"):
            x, lb = preprocess(img_bgr, self.input_hw)
            if x.dtype != self.input_dtype:
                x = x.astype(self.input_dtype)
        with timer(timings, "inference_ms"):
            det_out, protos = self.session.run([self._det_name, self._proto_name], {self.input_name: x})
        with timer(timings, "postprocess_ms"):
            dets = self._decode(np.asarray(det_out, dtype=np.float32), decode_conf, iou_thr)
            boxes = scale_boxes(dets.boxes, self.input_hw, lb.orig_hw)
            masks = (
                instance_masks(np.asarray(protos[0], dtype=np.float32), dets.coeffs, boxes, lb.orig_hw)
                if with_masks
                else [((0, 0), None)] * len(dets)
            )
            detections: list[Detection] = []
            for i in range(len(dets)):
                origin, mask = masks[i]
                if with_masks and (mask is None or not mask.any()):
                    continue  # Ultralytics drops detections whose mask is empty
                cid = int(dets.class_ids[i])
                detections.append(
                    Detection(
                        class_id=cid,
                        class_name=self.class_names[cid] if 0 <= cid < self.nc else str(cid),
                        confidence=float(dets.scores[i]),
                        box=(float(boxes[i, 0]), float(boxes[i, 1]), float(boxes[i, 2]), float(boxes[i, 3])),
                        mask=mask,
                        mask_origin=origin,
                    )
                )
        decision = ImageDecision.from_detections(detections, self.conf_threshold)
        shown = [d for d in detections if d.confidence >= display_conf]
        timings["total_ms"] = sum(timings.values())
        return Prediction(decision=decision, detections=shown, image_hw=lb.orig_hw, timings_ms=timings)
