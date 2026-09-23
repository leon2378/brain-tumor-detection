"""Pydantic response models (they also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionOut(BaseModel):
    class_id: int
    class_name: str = Field(examples=["glioma"])
    confidence: float = Field(ge=0.0, le=1.0)
    box: Box
    area_px: int = Field(description="tumour pixels in the predicted mask")
    area_fraction: float = Field(description="tumour pixels / image pixels")
    polygons: list[list[list[int]]] | None = Field(
        None, description="mask outline(s) as [[x, y], ...] in image pixels, largest first"
    )


class DecisionOut(BaseModel):
    label: str = Field(examples=["glioma", "no_tumor"])
    score: float = Field(description="confidence of the top detection (0 when none)")
    threshold: float = Field(description="operating threshold tuned on the validation split")
    tumor_detected: bool


class ImageMeta(BaseModel):
    width: int
    height: int


class ModelSummary(BaseModel):
    name: str
    version: str
    precision: str
    provider: str


class PredictResponse(BaseModel):
    request_id: str
    model: ModelSummary
    image: ImageMeta
    decision: DecisionOut
    detections: list[DetectionOut]
    mask_png_base64: str | None = Field(None, description="union mask (PNG, 0/255) when include_mask=true")
    timings_ms: dict[str, float]
    disclaimer: str


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    model: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    file: str
    precision: str
    sha256: str | None
    class_names: list[str]
    input_size: list[int]
    layout: str
    conf_threshold: float
    iou_threshold: float
    providers: list[str]
    extra: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
