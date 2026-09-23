"""Tiny hand-built ONNX models with the exact I/O contract of YOLO-seg exports.

Used by the unit/API tests and by CI's container smoke test, so the serving stack can be exercised without
torch, Ultralytics or a trained checkpoint. The dummy "detects" a meningioma (class 1) in the centre of the image
whenever the mean pixel intensity is above ~0.2 — so a black image yields ``no_tumor`` and a bright one a tumour.

Requires the ``onnx`` package (dev/train extras).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from btd.constants import NO_TUMOR, TUMOR_CLASSES
from btd.utils import sha256_file, write_json

NM = 32


def _score_subgraph(nodes: list[Any], inits: list[Any]) -> str:
    """s = sigmoid(30 * (mean(images) - 0.2)) with shape [1, 1, 1]."""
    from onnx import helper, numpy_helper

    inits += [
        numpy_helper.from_array(np.array([0.2], np.float32), "thr"),
        numpy_helper.from_array(np.array([30.0], np.float32), "gain"),
        numpy_helper.from_array(np.array([1, 1, 1], np.int64), "shape111"),
    ]
    nodes += [
        helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=1),
        helper.make_node("Sub", ["mean", "thr"], ["centered"]),
        helper.make_node("Mul", ["centered", "gain"], ["logit"]),
        helper.make_node("Sigmoid", ["logit"], ["score4d"]),
        helper.make_node("Reshape", ["score4d", "shape111"], ["score"]),
    ]
    return "score"


def _prototypes(size: int) -> np.ndarray:
    protos = np.full((1, NM, size, size), -1.0, np.float32)
    yy, xx = np.mgrid[:size, :size]
    c, r = size / 2, size * 0.15
    protos[0, 0][(yy - c) ** 2 + (xx - c) ** 2 <= r**2] = 1.0
    protos[0, 1:] = 0.0
    return protos


def make_dummy_model(
    out_dir: str | Path,
    imgsz: int = 320,
    layout: str = "end2end",
    filename: str = "model.onnx",
    threshold: float = 0.5,
) -> Path:
    """Write ``<out_dir>/<filename>`` and a matching ``model.json``; return the model path."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    nodes: list[Any] = []
    inits: list[Any] = []
    score = _score_subgraph(nodes, inits)
    ps = imgsz // 4
    box = np.array([imgsz * 0.3, imgsz * 0.3, imgsz * 0.7, imgsz * 0.7], np.float32)
    nc = len(TUMOR_CLASSES)

    if layout == "end2end":
        max_det = 300
        rest = np.zeros((1, max_det, 6 + NM), np.float32)
        rest[0, 0, :4] = box
        rest[0, 0, 5] = 1.0  # class id: meningioma
        rest[0, 0, 6] = 4.0  # mask coefficient on prototype 0
        onehot = np.zeros((1, max_det, 6 + NM), np.float32)
        onehot[0, 0, 4] = 1.0
        det_shape = [1, max_det, 6 + NM]
    elif layout == "raw":
        anchors = sum((imgsz // s) ** 2 for s in (8, 16, 32))
        rest = np.zeros((1, 4 + nc + NM, anchors), np.float32)
        onehot = np.zeros_like(rest)
        cx, cy, w, h = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0], box[3] - box[1]
        for a, (dx, peak) in enumerate([(0.0, 1.0), (2.0, 0.9), (-imgsz * 0.25, 0.3)]):  # 2nd overlaps → NMS
            rest[0, :4, a] = [cx + dx, cy + dx, w, h]
            onehot[0, 4 + 1, a] = peak  # class scores scale with the image score
            rest[0, 4 + nc, a] = 4.0
        det_shape = [1, 4 + nc + NM, anchors]
    else:
        raise ValueError(f"layout must be 'end2end' or 'raw', got {layout!r}")

    inits += [
        numpy_helper.from_array(rest, "det_rest"),
        numpy_helper.from_array(onehot, "det_scale"),
        numpy_helper.from_array(_prototypes(ps), "protos_const"),
        numpy_helper.from_array(np.zeros((1, 1, 1, 1), np.float32), "zero4"),
    ]
    nodes += [
        helper.make_node("Mul", ["det_scale", score], ["det_scored"]),
        helper.make_node("Add", ["det_rest", "det_scored"], ["output0"]),
        # tie output1 to the input so the graph has no dangling constant output
        helper.make_node("ReduceMean", ["images"], ["mean_again"], keepdims=1),
        helper.make_node("Mul", ["mean_again", "zero4"], ["zero_scalar"]),
        helper.make_node("Add", ["protos_const", "zero_scalar"], ["output1"]),
    ]
    graph = helper.make_graph(
        nodes,
        "dummy-yolo-seg",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])],
        [
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, det_shape),
            helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, NM, ps, ps]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="btd.testing")
    model.ir_version = 8
    meta = {
        "names": str(dict(enumerate(TUMOR_CLASSES))),
        "imgsz": str([imgsz, imgsz]),
        "end2end": str(layout == "end2end"),
        "task": "segment",
        "btd_precision": "fp32",
    }
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(model)
    path = out / filename
    onnx.save(model, str(path))
    write_json(
        out / "model.json",
        {
            "schema_version": 1,
            "name": f"dummy-{layout}",
            "version": "0.0.0-test",
            "task": "segment",
            "architecture": "dummy",
            "class_names": list(TUMOR_CLASSES),
            "no_tumor_label": NO_TUMOR,
            "imgsz": [imgsz, imgsz],
            "layout": layout,
            "thresholds": {"conf": threshold, "iou": 0.7, "tuned_on": "none"},
            "artifacts": {
                "fp32": {"file": filename, "sha256": sha256_file(path), "bytes": path.stat().st_size}
            },
        },
    )
    return path


def make_conv_model(path: str | Path, imgsz: int = 64) -> Path:
    """A small real-conv network with YOLO-seg shaped outputs (for quantisation tests)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    g = imgsz // 8
    w1 = rng.normal(0, 0.2, (16, 3, 3, 3)).astype(np.float32)
    w2 = rng.normal(0, 0.2, (6 + NM, 16, 1, 1)).astype(np.float32)
    w3 = rng.normal(0, 0.2, (NM, 16, 1, 1)).astype(np.float32)
    inits = [
        numpy_helper.from_array(w1, "w1"),
        numpy_helper.from_array(np.zeros(16, np.float32), "b1"),
        numpy_helper.from_array(w2, "w2"),
        numpy_helper.from_array(np.zeros(6 + NM, np.float32), "b2"),
        numpy_helper.from_array(w3, "w3"),
        numpy_helper.from_array(np.zeros(NM, np.float32), "b3"),
        numpy_helper.from_array(np.array([1, 6 + NM, g * g], np.int64), "flat"),
    ]
    nodes = [
        helper.make_node(
            "Conv", ["images", "w1", "b1"], ["c1"], name="stem", strides=[8, 8], pads=[1, 1, 1, 1]
        ),
        helper.make_node("Relu", ["c1"], ["r1"], name="act"),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], name="head_det"),
        helper.make_node("Reshape", ["c2", "flat"], ["c2f"], name="flatten"),
        helper.make_node("Transpose", ["c2f"], ["output0"], name="to_bnc", perm=[0, 2, 1]),
        helper.make_node("Conv", ["r1", "w3", "b3"], ["output1"], name="head_proto"),
    ]
    graph = helper.make_graph(
        nodes,
        "conv-yolo-seg",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, imgsz, imgsz])],
        [
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, g * g, 6 + NM]),
            helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, NM, g, g]),
        ],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    p = model.metadata_props.add()
    p.key, p.value = "names", str(dict(enumerate(TUMOR_CLASSES)))
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return Path(path)
