"""Letterbox pre-processing that reproduces Ultralytics' ``LetterBox`` bit-for-bit (numpy + OpenCV only)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114


@dataclass(frozen=True)
class LetterboxInfo:
    orig_hw: tuple[int, int]
    input_hw: tuple[int, int]
    gain: float
    pad_top: int
    pad_left: int


def letterbox(
    img: np.ndarray,
    new_hw: tuple[int, int] = (640, 640),
    pad_value: int = PAD_VALUE,
    scaleup: bool = True,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio and pad to ``new_hw`` (centered), exactly like Ultralytics.

    Python's ``round`` (banker's rounding) is used on purpose: it is what Ultralytics uses, and matching it
    keeps the ONNX pipeline pixel-identical to the training/validation pipeline.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h0, w0 = img.shape[:2]
    nh, nw = new_hw
    r = min(nh / h0, nw / w0)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (round(w0 * r), round(h0 * r))  # (w, h)
    dw, dh = (nw - new_unpad[0]) / 2, (nh - new_unpad[1]) / 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
    return img, LetterboxInfo(orig_hw=(h0, w0), input_hw=(nh, nw), gain=r, pad_top=top, pad_left=left)


def to_tensor(img_bgr: np.ndarray) -> np.ndarray:
    """HWC BGR uint8 -> NCHW RGB float32 in [0, 1]."""
    x = img_bgr[..., ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32)
    x /= 255.0  # division (not *1/255) to match Ultralytics' float32 values exactly
    return x[None]


def preprocess(img_bgr: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, LetterboxInfo]:
    padded, info = letterbox(img_bgr, input_hw)
    return to_tensor(padded), info
