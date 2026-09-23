"""Typed results returned by the inference engine (framework-free)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from btd.constants import NO_TUMOR


@dataclass(frozen=True)
class Detection:
    """One tumour instance in *original image* pixel coordinates.

    ``mask`` is stored only inside the box region (like Mask R-CNN ROI masks) to keep memory bounded for
    large uploads: ``mask[r, c]`` corresponds to pixel ``(mask_origin[1] + r, mask_origin[0] + c)``.
    """

    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    mask: np.ndarray | None = None  # bool (h_roi, w_roi)
    mask_origin: tuple[int, int] = (0, 0)  # (x, y) of mask[0, 0]

    @property
    def area_px(self) -> int:
        if self.mask is not None:
            return int(self.mask.sum())
        x1, y1, x2, y2 = self.box
        return int(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    def full_mask(self, image_hw: tuple[int, int]) -> np.ndarray:
        """Paste the ROI mask into a full-size boolean canvas."""
        canvas = np.zeros(image_hw, dtype=bool)
        if self.mask is None or self.mask.size == 0:
            return canvas
        x0, y0 = self.mask_origin
        h, w = self.mask.shape
        canvas[y0 : y0 + h, x0 : x0 + w] = self.mask
        return canvas


@dataclass(frozen=True)
class ImageDecision:
    """Image-level classification derived from detections.

    A slice is labelled with the class of its most confident detection when that confidence reaches the
    operating threshold (tuned on the validation split); otherwise it is ``no_tumor``.
    """

    label: str
    score: float
    threshold: float

    @property
    def tumor_detected(self) -> bool:
        return self.label != NO_TUMOR

    @classmethod
    def from_detections(cls, detections: list[Detection], threshold: float) -> ImageDecision:
        if not detections:
            return cls(label=NO_TUMOR, score=0.0, threshold=threshold)
        top = max(detections, key=lambda d: d.confidence)
        if top.confidence >= threshold:
            return cls(label=top.class_name, score=top.confidence, threshold=threshold)
        return cls(label=NO_TUMOR, score=top.confidence, threshold=threshold)


@dataclass
class Prediction:
    decision: ImageDecision
    detections: list[Detection]
    image_hw: tuple[int, int]
    timings_ms: dict[str, float] = field(default_factory=dict)

    def union_mask(self) -> np.ndarray:
        """All instance masks merged into one full-size boolean mask."""
        out = np.zeros(self.image_hw, dtype=bool)
        for d in self.detections:
            if d.mask is not None and d.mask.size:
                x0, y0 = d.mask_origin
                h, w = d.mask.shape
                out[y0 : y0 + h, x0 : x0 + w] |= d.mask
        return out
