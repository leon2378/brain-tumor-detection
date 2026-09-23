"""Post-training quantisation of YOLO-seg ONNX models: FP16 conversion and INT8 static (QDQ) quantisation.

INT8 uses ONNX Runtime static quantisation with calibration images pre-processed by the *same* letterbox code the
server uses, so calibration statistics match production inputs exactly.

Design choices (measured, see docs/QUANTIZATION.md):
* **U8S8** (uint8 activations, int8 per-channel weights) by default. On x86 ONNX Runtime only fuses QDQ convs into
  ``QLinearConv`` for U8S8; the S8S8 default of recent ORT releases (and Ultralytics' built-in INT8 ONNX export)
  leaves most convs un-fused and runs *slower* than FP32 on CPU.
* The whole backbone/neck/head is quantised, but everything *downstream* of the head's output convolutions
  (box decoding, sigmoid scores, TopK/Gather, mask-coefficient plumbing) stays FP32: a single INT8 scale cannot
  cover box pixels (0-640) and probabilities (0-1) at once.
* ``keep_head_fp32`` additionally keeps those final head convolutions in FP32 (most quantisation-sensitive layers).
* ``target="tensorrt"`` switches to symmetric int8 activations, which TensorRT's explicit-quantisation path needs.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from btd.inference.preprocess import preprocess
from btd.utils import imread

LOGGER = logging.getLogger(__name__)


def _load_onnx(path: str | Path) -> Any:
    import onnx

    return onnx.load(str(path))


def copy_metadata(src: Any, dst: Any, extra: dict[str, str] | None = None) -> None:
    """Copy ``metadata_props`` (class names, stride, imgsz, end2end…) and add/override ``extra`` keys."""
    merged = {p.key: p.value for p in src.metadata_props}
    merged.update({p.key: p.value for p in dst.metadata_props})
    merged.update(extra or {})
    del dst.metadata_props[:]
    for k, v in merged.items():
        prop = dst.metadata_props.add()
        prop.key, prop.value = k, str(v)


def set_metadata(path: str | Path, extra: dict[str, str]) -> None:
    import onnx

    model = onnx.load(str(path))
    copy_metadata(model, model, extra)
    onnx.save(model, str(path))


def head_output_convs(model: Any) -> list[str]:
    """Names of the last Conv node on every path to a graph output (walk backwards, stop at the first Conv)."""
    graph = model.graph
    producer = {out: node for node in graph.node for out in node.output}
    found: list[str] = []
    seen: set[str] = set()
    stack = [o.name for o in graph.output]
    while stack:
        tensor = stack.pop()
        node = producer.get(tensor)
        if node is None or node.name in seen:
            continue
        seen.add(node.name)
        if node.op_type == "Conv":
            found.append(node.name)
            continue
        stack.extend(node.input)
    return sorted(found)


def convert_fp16(src: str | Path, dst: str | Path) -> Path:
    """FP16 weights/activations with FP32 graph inputs/outputs (drop-in replacement for the FP32 model)."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(src))
    fp16 = convert_float_to_float16(model, keep_io_types=True)
    copy_metadata(model, fp16, {"btd_precision": "fp16"})
    onnx.save(fp16, str(dst))
    return Path(dst)


class LetterboxCalibrationReader:
    """ONNX Runtime ``CalibrationDataReader`` feeding letterboxed images one at a time."""

    def __init__(self, image_paths: Sequence[str | Path], input_name: str, input_hw: tuple[int, int]) -> None:
        if not image_paths:
            raise ValueError("INT8 calibration needs at least one image")
        self.image_paths = [Path(p) for p in image_paths]
        self.input_name = input_name
        self.input_hw = input_hw
        self._it: Iterator[Path] = iter(self.image_paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        path = next(self._it, None)
        if path is None:
            return None
        x, _ = preprocess(imread(path), self.input_hw)
        return {self.input_name: x}

    def rewind(self) -> None:
        self._it = iter(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)


def downstream_nodes(model: Any, start_nodes: set[str]) -> set[str]:
    """All nodes reachable from the outputs of ``start_nodes`` (i.e. the post-head decode sub-graph)."""
    graph = model.graph
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    found: set[str] = set()
    stack = [t for node in graph.node if node.name in start_nodes for t in node.output]
    while stack:
        for node in consumers.get(stack.pop(), []):
            if node.name not in found:
                found.add(node.name)
                stack.extend(node.output)
    return found


def quantize_int8(
    fp32_path: str | Path,
    int8_path: str | Path,
    calibration_images: Sequence[str | Path],
    per_channel: bool = True,
    method: str = "minmax",
    keep_head_fp32: bool = False,
    target: str = "cpu",
    reduce_range: bool = False,
) -> Path:
    """Static INT8 (QDQ) quantisation with ONNX Runtime. ``target``: ``"cpu"`` (U8S8) or ``"tensorrt"`` (S8S8
    symmetric). ``reduce_range`` (7-bit weights) avoids VPMADDUBSW saturation on old AVX2-only CPUs."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }
    if method not in methods:
        raise ValueError(f"Unknown calibration method {method!r}; choose from {sorted(methods)}")
    if target not in {"cpu", "tensorrt"}:
        raise ValueError(f"Unknown INT8 target {target!r}; choose 'cpu' or 'tensorrt'")

    fp32_model = _load_onnx(fp32_path)
    input_name = fp32_model.graph.input[0].name
    dims = fp32_model.graph.input[0].type.tensor_type.shape.dim
    input_hw = (int(dims[2].dim_value or 640), int(dims[3].dim_value or 640))

    tmp_dir = Path(tempfile.mkdtemp(prefix="btd-quant-"))
    try:
        pre = tmp_dir / "pre.onnx"
        try:
            quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
        except Exception as exc:  # pragma: no cover - depends on ORT version/model
            LOGGER.warning("quant_pre_process failed (%s); quantising the original graph", exc)
            shutil.copyfile(fp32_path, pre)
        pre_model = _load_onnx(pre)
        head = set(head_output_convs(pre_model))
        exclude = downstream_nodes(pre_model, head)
        if keep_head_fp32:
            LOGGER.info("Keeping %d head output conv(s) in FP32", len(head))
            exclude |= head
        LOGGER.info("Excluding %d decode node(s) from quantisation", len(exclude))

        class _Reader(CalibrationDataReader):  # adapt to ORT's abstract base class
            def __init__(self, inner: LetterboxCalibrationReader) -> None:
                self.inner = inner

            def get_next(self) -> dict[str, np.ndarray] | None:
                return self.inner.get_next()

            def rewind(self) -> None:
                self.inner.rewind()

        reader = _Reader(LetterboxCalibrationReader(calibration_images, input_name, input_hw))
        tensorrt = target == "tensorrt"
        extra: dict[str, Any] = {"ActivationSymmetric": tensorrt, "WeightSymmetric": True}
        if method == "percentile":
            extra["CalibPercentile"] = 99.999
        LOGGER.info(
            "INT8 static quantisation: %d calibration images, method=%s, per_channel=%s, target=%s",
            len(calibration_images),
            method,
            per_channel,
            target,
        )
        quantize_static(
            str(pre),
            str(int8_path),
            reader,
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            reduce_range=reduce_range,
            activation_type=QuantType.QInt8 if tensorrt else QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=methods[method],
            nodes_to_exclude=sorted(exclude),
            extra_options=extra,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    import onnx

    q = onnx.load(str(int8_path))
    copy_metadata(
        fp32_model,
        q,
        {
            "btd_precision": "int8",
            "btd_int8_target": target,
            "btd_int8_method": method,
            "btd_int8_per_channel": str(per_channel),
            "btd_int8_keep_head_fp32": str(keep_head_fp32),
            "btd_int8_calibration_images": str(len(calibration_images)),
        },
    )
    onnx.save(q, str(int8_path))
    return Path(int8_path)
