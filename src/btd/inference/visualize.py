"""Overlay rendering for predictions (used by the CLI and the /v1/predict/overlay endpoint)."""

from __future__ import annotations

import cv2
import numpy as np

from btd.constants import DISCLAIMER
from btd.inference.types import Prediction

# BGR colours, colour-blind friendly (Okabe-Ito).
PALETTE: dict[str, tuple[int, int, int]] = {
    "glioma": (0, 159, 230),  # orange
    "meningioma": (233, 180, 86),  # sky blue
    "pituitary": (115, 158, 0),  # bluish green
    "no_tumor": (160, 160, 160),
}
_FALLBACK = (167, 121, 204)


def draw_overlay(
    img_bgr: np.ndarray, prediction: Prediction, alpha: float = 0.45, show_disclaimer: bool = True
) -> np.ndarray:
    """Return a copy of the image with masks, boxes and the image-level decision drawn on it."""
    out = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR) if img_bgr.ndim == 2 else img_bgr.copy()
    h, w = out.shape[:2]
    thickness = max(1, round(min(h, w) / 300))
    font_scale = max(0.4, min(h, w) / 900)

    for det in prediction.detections:
        color = PALETTE.get(det.class_name, _FALLBACK)
        if det.mask is not None and det.mask.size:
            x0, y0 = det.mask_origin
            mh, mw = det.mask.shape
            roi = out[y0 : y0 + mh, x0 : x0 + mw]
            tint = np.empty_like(roi)
            tint[:] = color
            blended = cv2.addWeighted(roi, 1 - alpha, tint, alpha, 0)
            roi[det.mask] = blended[det.mask]
            contours, _ = cv2.findContours(
                det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(roi, contours, -1, color, thickness)
        x1, y1, x2, y2 = (round(v) for v in det.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        ty = max(th + base, y1)
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw, ty), color, -1)
        cv2.putText(
            out,
            label,
            (x1, ty - base),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    d = prediction.decision
    banner = f"{d.label}  (score {d.score:.2f}, threshold {d.threshold:.2f})"
    _put_fitted(
        out, banner, (8, 8), font_scale * 1.1, PALETTE.get(d.label, _FALLBACK), thickness + 1, top=True
    )
    if show_disclaimer:
        _put_fitted(out, DISCLAIMER, (8, h - 8), font_scale * 0.7, (200, 200, 200), 1, top=False)
    return out


def _put_fitted(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
    top: bool,
) -> None:
    """Draw text, shrinking the font so it never runs past the right edge."""
    max_w = img.shape[1] - 2 * origin[0]
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if tw > max_w > 0:
        scale *= max_w / tw
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = origin
    y = y + th if top else y
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
