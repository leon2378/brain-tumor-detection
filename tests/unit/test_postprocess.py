from __future__ import annotations

import numpy as np
import pytest

from btd.inference.postprocess import (
    LAYOUT_END2END,
    LAYOUT_RAW,
    crop_proto_padding,
    decode_end2end,
    decode_raw,
    infer_layout,
    instance_masks,
    mask_to_polygons,
    nms,
    roi_mask,
    scale_boxes,
    xywh2xyxy,
)


def test_infer_layout() -> None:
    assert infer_layout((1, 300, 38), nc=3, nm=32) == LAYOUT_END2END
    assert infer_layout((1, 39, 8400), nc=3, nm=32) == LAYOUT_RAW
    assert infer_layout(("batch", 39, "anchors"), nc=3, nm=32) == LAYOUT_RAW
    with pytest.raises(ValueError):
        infer_layout((1, 10, 10), nc=3, nm=32)


def test_xywh2xyxy() -> None:
    out = xywh2xyxy(np.array([[10.0, 20.0, 4.0, 6.0]]))
    np.testing.assert_allclose(out, [[8, 17, 12, 23]])


def test_nms_matches_bruteforce() -> None:
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, (60, 2))
    wh = rng.uniform(5, 40, (60, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
    scores = rng.uniform(0, 1, 60).astype(np.float32)
    keep = nms(boxes, scores, 0.5)

    def iou(a: np.ndarray, b: np.ndarray) -> float:
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

    expected: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        if all(iou(boxes[i], boxes[j]) <= 0.5 for j in expected):
            expected.append(int(i))
    assert keep.tolist() == expected


def test_nms_empty() -> None:
    assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.5).size == 0


def test_decode_end2end_filters_by_conf_and_max_det() -> None:
    out = np.zeros((1, 5, 6 + 2), np.float32)
    out[0, :, 4] = [0.9, 0.2, 0.6, 0.0, 0.8]
    out[0, :, 5] = [0, 1, 2, 0, 1]
    dets = decode_end2end(out, conf=0.5, nm=2, max_det=2)
    assert len(dets) == 2
    assert dets.scores.tolist() == pytest.approx([0.9, 0.6])
    assert dets.class_ids.tolist() == [0, 2]


def test_decode_raw_applies_class_aware_nms() -> None:
    nc, nm = 3, 2
    out = np.zeros((1, 4 + nc + nm, 3), np.float32)
    out[0, :4, 0] = [50, 50, 20, 20]
    out[0, :4, 1] = [51, 51, 20, 20]  # overlaps box 0
    out[0, :4, 2] = [51, 51, 20, 20]  # overlaps too but different class
    out[0, 4 + 0, 0] = 0.9
    out[0, 4 + 0, 1] = 0.8
    out[0, 4 + 1, 2] = 0.7
    dets = decode_raw(out, conf=0.25, iou=0.5, nc=nc, nm=nm, max_det=10)
    assert sorted(dets.class_ids.tolist()) == [0, 1]
    agnostic = decode_raw(out, conf=0.25, iou=0.5, nc=nc, nm=nm, max_det=10, agnostic=True)
    assert agnostic.class_ids.tolist() == [0]


def test_scale_boxes_roundtrip_with_letterbox() -> None:
    from btd.inference.preprocess import letterbox

    img = np.zeros((300, 500, 3), np.uint8)
    _, info = letterbox(img, (640, 640))
    box_orig = np.array([[100, 50, 200, 150]], np.float32)
    box_in = box_orig * info.gain + [info.pad_left, info.pad_top, info.pad_left, info.pad_top]
    back = scale_boxes(box_in, (640, 640), (300, 500))
    np.testing.assert_allclose(back, box_orig, atol=1e-3)


def test_roi_mask_equals_full_resize_then_crop() -> None:
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1, (37, 53)).astype(np.float32)
    orig_hw = (211, 307)
    box = np.array([20.4, 11.0, 250.7, 190.2], np.float32)
    origin, roi = roi_mask(logits, box, orig_hw)
    full = F.interpolate(torch.from_numpy(logits)[None, None], orig_hw, mode="bilinear")[0, 0].numpy() > 0
    x1, y1 = origin
    expected = full[y1 : y1 + roi.shape[0], x1 : x1 + roi.shape[1]]
    assert roi.shape == (int(np.ceil(190.2)) - 11, int(np.ceil(250.7)) - 21)
    assert (roi == expected).mean() > 0.999


def test_instance_masks_and_polygons() -> None:
    nm, ps = 4, 40
    protos = np.full((nm, ps, ps), -1.0, np.float32)
    protos[0, 10:30, 10:30] = 1.0  # square in proto space (input 160x160 letterboxed from 160x160)
    coeffs = np.array([[1.0, 0, 0, 0]], np.float32)
    boxes = np.array([[0, 0, 160, 160]], np.float32)
    ((origin, mask),) = instance_masks(protos, coeffs, boxes, (160, 160))
    assert origin == (0, 0)
    assert mask.shape == (160, 160)
    assert abs(int(mask.sum()) - 80 * 80) < 400
    polys = mask_to_polygons(mask, origin)
    assert len(polys) == 1 and len(polys[0]) >= 4
    xs = [p[0] for p in polys[0]]
    assert min(xs) >= 36 and max(xs) <= 124


def test_crop_proto_padding_removes_letterbox_bands() -> None:
    protos = np.zeros((2, 160, 160), np.float32)
    cropped = crop_proto_padding(protos, (320, 640))  # wide image → bands at top/bottom
    assert cropped.shape == (2, 80, 160)


def test_empty_inputs() -> None:
    assert (
        instance_masks(
            np.zeros((2, 8, 8), np.float32), np.zeros((0, 2), np.float32), np.zeros((0, 4)), (8, 8)
        )
        == []
    )
    assert mask_to_polygons(np.zeros((5, 5), bool)) == []
    _, m = roi_mask(np.zeros((8, 8), np.float32), np.array([5, 5, 5, 5], np.float32), (8, 8))
    assert m.size == 0
