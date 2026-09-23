"""Decode YOLO segmentation ONNX outputs with numpy only (no torch in the serving image).

Two head layouts are supported and auto-detected:

* ``end2end`` — NMS-free one-to-one head (YOLO26 default when exported with ``nms=False``):
  ``output0`` is ``(B, max_det, 6 + nm)`` = ``x1, y1, x2, y2, score, class, mask_coeffs...``.
* ``raw`` — classic one-to-many head (YOLOv8/YOLO11, or YOLO26 exported with ``nms=None``):
  ``output0`` is ``(B, 4 + nc + nm, anchors)`` = ``cx, cy, w, h, class_scores..., mask_coeffs...``; needs NMS.

``output1`` holds the mask prototypes ``(B, nm, H/4, W/4)``. Mask assembly mirrors Ultralytics'
``process_mask_native`` (retina masks): prototypes are cropped to the un-padded region, bilinearly upsampled to the
original image size (align_corners=False semantics), thresholded at 0 and cropped to the box — but evaluated only
inside each box, so memory stays O(box area) instead of O(n * H * W).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

LAYOUT_END2END = "end2end"
LAYOUT_RAW = "raw"


@dataclass
class DecodedDetections:
    boxes: np.ndarray  # (n, 4) xyxy in letterboxed-input pixels
    scores: np.ndarray  # (n,)
    class_ids: np.ndarray  # (n,) int64
    coeffs: np.ndarray  # (n, nm)

    def __len__(self) -> int:
        return int(self.scores.shape[0])

    @classmethod
    def empty(cls, nm: int) -> DecodedDetections:
        return cls(
            boxes=np.zeros((0, 4), np.float32),
            scores=np.zeros((0,), np.float32),
            class_ids=np.zeros((0,), np.int64),
            coeffs=np.zeros((0, nm), np.float32),
        )

    def select(self, idx: np.ndarray) -> DecodedDetections:
        return DecodedDetections(self.boxes[idx], self.scores[idx], self.class_ids[idx], self.coeffs[idx])


def infer_layout(output0_shape: tuple[int | str | None, ...], nc: int, nm: int) -> str:
    """Work out the head layout from the (possibly symbolic) shape of ``output0``."""
    if len(output0_shape) != 3:
        raise ValueError(f"Unexpected output0 rank {len(output0_shape)}; expected 3")
    _, d1, d2 = output0_shape
    if d1 == 4 + nc + nm:
        return LAYOUT_RAW
    if d2 == 6 + nm:
        return LAYOUT_END2END
    raise ValueError(
        f"Cannot infer head layout from output0 shape {output0_shape} with nc={nc}, nm={nm}. "
        "Is this a YOLO segmentation model trained on the same classes?"
    )


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.empty_like(x)
    half_wh = x[..., 2:4] / 2
    y[..., :2] = x[..., :2] - half_wh
    y[..., 2:4] = x[..., :2] + half_wh
    return y


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against many."""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedy NMS with torchvision semantics (drop boxes with IoU > threshold). Returns kept indices."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        order = rest[box_iou(boxes[i], boxes[rest]) <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def decode_end2end(output0: np.ndarray, conf: float, nm: int, max_det: int) -> DecodedDetections:
    pred = output0[0]
    pred = pred[pred[:, 4] > conf][:max_det]
    if pred.shape[0] == 0:
        return DecodedDetections.empty(nm)
    return DecodedDetections(
        boxes=pred[:, :4].astype(np.float32, copy=True),
        scores=pred[:, 4].astype(np.float32, copy=True),
        class_ids=pred[:, 5].round().astype(np.int64),
        coeffs=pred[:, 6 : 6 + nm].astype(np.float32, copy=True),
    )


def decode_raw(
    output0: np.ndarray,
    conf: float,
    iou: float,
    nc: int,
    nm: int,
    max_det: int,
    agnostic: bool = False,
    max_nms: int = 30000,
    max_wh: int = 7680,
) -> DecodedDetections:
    pred = output0[0].T  # (anchors, 4 + nc + nm)
    cls_scores = pred[:, 4 : 4 + nc]
    scores = cls_scores.max(axis=1)
    keep = scores > conf
    if not keep.any():
        return DecodedDetections.empty(nm)
    pred, scores = pred[keep], scores[keep]
    class_ids = cls_scores[keep].argmax(axis=1).astype(np.int64)
    if scores.shape[0] > max_nms:
        top = np.argsort(-scores, kind="stable")[:max_nms]
        pred, scores, class_ids = pred[top], scores[top], class_ids[top]
    boxes = xywh2xyxy(pred[:, :4].astype(np.float32))
    offsets = 0.0 if agnostic else class_ids[:, None].astype(np.float32) * max_wh
    kept = nms(boxes + offsets, scores, iou)[:max_det]
    return DecodedDetections(
        boxes=boxes[kept],
        scores=scores[kept].astype(np.float32),
        class_ids=class_ids[kept],
        coeffs=pred[kept, 4 + nc : 4 + nc + nm].astype(np.float32),
    )


def scale_boxes(boxes: np.ndarray, input_hw: tuple[int, int], orig_hw: tuple[int, int]) -> np.ndarray:
    """Map xyxy boxes from letterboxed-input pixels to original-image pixels (Ultralytics rounding rules)."""
    ih, iw = input_hw
    h0, w0 = orig_hw
    gain = min(ih / h0, iw / w0)
    pad_x = round((iw - round(w0 * gain)) / 2 - 0.1)
    pad_y = round((ih - round(h0 * gain)) / 2 - 0.1)
    out = boxes.astype(np.float32, copy=True)
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= gain
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w0)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h0)
    return out


def crop_proto_padding(protos: np.ndarray, orig_hw: tuple[int, int]) -> np.ndarray:
    """Remove the letterbox padding from prototype maps ``(nm, mh, mw)``."""
    _, mh, mw = protos.shape
    h0, w0 = orig_hw
    gain = min(mh / h0, mw / w0)
    pad_w = (mw - round(w0 * gain)) / 2
    pad_h = (mh - round(h0 * gain)) / 2
    top, left = round(pad_h - 0.1), round(pad_w - 0.1)
    bottom, right = mh - round(pad_h + 0.1), mw - round(pad_w + 0.1)
    return protos[:, top:bottom, left:right]


def _linear_taps(dst: np.ndarray, in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source indices/weights for 1-D linear resize with PyTorch ``align_corners=False`` semantics."""
    scale = in_size / out_size
    src = (dst.astype(np.float64) + 0.5) * scale - 0.5
    src = np.maximum(src, 0.0)
    i0 = np.floor(src).astype(np.int64)
    i0 = np.minimum(i0, in_size - 1)
    i1 = np.minimum(i0 + 1, in_size - 1)
    w1 = (src - i0).astype(np.float32)
    return i0, i1, w1


def roi_mask(
    logits: np.ndarray, box: np.ndarray, orig_hw: tuple[int, int]
) -> tuple[tuple[int, int], np.ndarray]:
    """Upsample prototype-resolution mask logits ``(hc, wc)`` only inside ``box`` and threshold at 0.

    Equivalent to resizing the whole map to ``orig_hw`` then cropping ``[ceil(x1), ceil(x2))`` x
    ``[ceil(y1), ceil(y2))`` — the exact region Ultralytics' ``crop_mask`` keeps.
    """
    h0, w0 = orig_hw
    hc, wc = logits.shape
    x1, y1, x2, y2 = (float(v) for v in box)
    ix1, iy1 = max(0, math.ceil(x1)), max(0, math.ceil(y1))
    ix2, iy2 = min(w0, math.ceil(x2)), min(h0, math.ceil(y2))
    if ix2 <= ix1 or iy2 <= iy1 or hc == 0 or wc == 0:
        return (ix1, iy1), np.zeros((0, 0), dtype=bool)
    cx0, cx1, wx = _linear_taps(np.arange(ix1, ix2), wc, w0)
    ry0, ry1, wy = _linear_taps(np.arange(iy1, iy2), hc, h0)
    cols = logits[:, cx0] * (1.0 - wx) + logits[:, cx1] * wx  # (hc, w_roi)
    vals = cols[ry0] * (1.0 - wy)[:, None] + cols[ry1] * wy[:, None]  # (h_roi, w_roi)
    return (ix1, iy1), vals > 0.0


def instance_masks(
    protos: np.ndarray,
    coeffs: np.ndarray,
    boxes_orig: np.ndarray,
    orig_hw: tuple[int, int],
) -> list[tuple[tuple[int, int], np.ndarray]]:
    """ROI masks for each detection. ``protos``: (nm, mh, mw); ``boxes_orig`` in original pixels."""
    if coeffs.shape[0] == 0:
        return []
    cropped = crop_proto_padding(protos, orig_hw)
    nm, hc, wc = cropped.shape
    logits = (coeffs @ cropped.reshape(nm, -1)).reshape(-1, hc, wc)
    return [roi_mask(logits[i], boxes_orig[i], orig_hw) for i in range(coeffs.shape[0])]


def mask_to_polygons(
    mask: np.ndarray,
    origin: tuple[int, int] = (0, 0),
    epsilon: float = 1.0,
    min_area: float = 4.0,
) -> list[list[list[int]]]:
    """External contours of a boolean mask as integer polygons in image coordinates (largest first)."""
    if mask.size == 0 or not mask.any():
        return []
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: list[tuple[float, list[list[int]]]] = []
    ox, oy = origin
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        simplified = cv2.approxPolyDP(contour, epsilon, True) if epsilon > 0 else contour
        pts = simplified.reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        polys.append((area, [[int(x) + ox, int(y) + oy] for x, y in pts]))
    polys.sort(key=lambda t: -t[0])
    return [p for _, p in polys]
