from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from btd.export.quantize import convert_fp16, downstream_nodes, head_output_convs, quantize_int8  # noqa: E402
from btd.testing import make_conv_model  # noqa: E402
from btd.utils import imwrite  # noqa: E402


@pytest.fixture
def conv_model(tmp_path: Path) -> Path:
    return make_conv_model(tmp_path / "conv.onnx", imgsz=64)


@pytest.fixture
def calib_images(tmp_path: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    paths = []
    for i in range(8):
        p = tmp_path / "calib" / f"{i}.png"
        imwrite(p, rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        paths.append(p)
    return paths


def _run(path: Path, x: np.ndarray) -> list[np.ndarray]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"images": x})


def test_graph_analysis(conv_model: Path) -> None:
    model = onnx.load(str(conv_model))
    head = head_output_convs(model)
    assert head == ["head_det", "head_proto"]
    assert downstream_nodes(model, set(head)) == {"flatten", "to_bnc"}


def test_fp16_conversion_keeps_io_and_metadata(conv_model: Path, tmp_path: Path) -> None:
    out = convert_fp16(conv_model, tmp_path / "fp16.onnx")
    m = onnx.load(str(out))
    meta = {p.key: p.value for p in m.metadata_props}
    assert meta["btd_precision"] == "fp16" and "names" in meta
    assert m.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT  # FP32 I/O
    x = np.random.default_rng(1).random((1, 3, 64, 64), dtype=np.float32)
    ref, got = _run(conv_model, x), _run(out, x)
    np.testing.assert_allclose(got[0], ref[0], atol=2e-2, rtol=2e-2)
    assert out.stat().st_size < conv_model.stat().st_size


@pytest.mark.parametrize(("target", "keep_head"), [("cpu", False), ("tensorrt", True)])
def test_int8_quantisation(
    conv_model: Path, calib_images: list[Path], tmp_path: Path, target: str, keep_head: bool
) -> None:
    out = quantize_int8(
        conv_model, tmp_path / "int8.onnx", calib_images, target=target, keep_head_fp32=keep_head
    )
    m = onnx.load(str(out))
    ops = [n.op_type for n in m.graph.node]
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    meta = {p.key: p.value for p in m.metadata_props}
    assert meta["btd_precision"] == "int8" and meta["btd_int8_target"] == target
    # Decode nodes stay float: nothing downstream of the head convs is quantised.
    q_inputs = {n.input[0] for n in m.graph.node if n.op_type == "QuantizeLinear"}
    assert not any(name.startswith("c2f") for name in q_inputs)
    x = np.random.default_rng(2).random((1, 3, 64, 64), dtype=np.float32)
    ref, got = _run(conv_model, x), _run(out, x)
    corr = np.corrcoef(ref[0].ravel(), got[0].ravel())[0, 1]
    assert corr > 0.98


def test_int8_rejects_bad_options(conv_model: Path, calib_images: list[Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="calibration method"):
        quantize_int8(conv_model, tmp_path / "x.onnx", calib_images, method="magic")
    with pytest.raises(ValueError, match="target"):
        quantize_int8(conv_model, tmp_path / "x.onnx", calib_images, target="tpu")
