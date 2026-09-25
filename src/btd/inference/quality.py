"""Cheap sanity checks on an input image, reported next to the prediction.

The model only knows 2-D T1 contrast-enhanced brain MRI slices, and it answers just as confidently on anything else:
a landscape photo came back as "glioma 0.81". These checks catch obvious mismatches. Passing them does not mean an
image is in distribution: a greyscale photo, or a T2 or FLAIR slice, still gets through (see MODEL_CARD.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A pixel counts as coloured when its channels differ by more than CHROMA_LEVEL (0-255). Measured share of coloured
# pixels: 0 for all 5,198 BRISC 2025 slices, at most 0.041 across the 7,200-slice Kaggle MRI set (some carry coloured
# annotations), and 0.27-0.98 for photos. MAX_COLOURFUL_FRACTION sits well clear of both.
CHROMA_LEVEL = 40
MAX_COLOURFUL_FRACTION = 0.10


@dataclass(frozen=True)
class InputWarning:
    code: str
    message: str


NOT_GREYSCALE = InputWarning(
    "not_greyscale",
    "This image is in colour, but MRI slices are greyscale. It may not be an MRI slice, so the result is unreliable.",
)


def colourful_fraction(img: np.ndarray) -> float:
    """Share of pixels whose colour channels clearly differ (0 for a greyscale image)."""
    if img.ndim == 2 or img.shape[2] == 1:
        return 0.0
    step = max(1, max(img.shape[:2]) // 256)  # a ~256 px sample is plenty and keeps large uploads cheap
    sample = img[::step, ::step, :3].astype(np.int16)
    chroma = sample.max(axis=2) - sample.min(axis=2)
    return float((chroma > CHROMA_LEVEL).mean())


def input_warnings(img: np.ndarray) -> list[InputWarning]:
    """Reasons to distrust a prediction on this image; empty when nothing looks off."""
    return [NOT_GREYSCALE] if colourful_fraction(img) > MAX_COLOURFUL_FRACTION else []
