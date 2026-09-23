"""Small helpers shared by tests."""

from __future__ import annotations

import cv2
import numpy as np


def encode(img: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()
