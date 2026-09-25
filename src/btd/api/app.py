"""FastAPI inference service.

Endpoints
---------
``GET  /``                    web page for browsers (``BTD_UI``); a small JSON index for API clients
``GET  /ui/...``              the page's script, styles and sample slices
``GET  /health``              liveness (process is up)
``GET  /ready``               readiness (model loaded + warmed up) — 503 otherwise
``GET  /v1/model``            model card: version, precision, classes, threshold, checksum, provider
``POST /v1/predict``          multipart image → decision + detections (+ polygons / mask)
``POST /v1/predict/overlay``  multipart image → PNG with masks, boxes and decision drawn
``GET  /metrics``             Prometheus metrics
``GET  /ping``, ``POST /invocations``  SageMaker's container contract (only with ``BTD_SAGEMAKER=true``)

Production concerns handled here: request IDs, JSON access logs, upload size/type/pixel limits,
bounded inference concurrency (inference runs in a worker thread, never on the event loop),
optional API-key auth, CORS for frontends hosted elsewhere, and no stack traces leaked to clients.
"""

import asyncio
import base64
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from btd import __version__
from btd.api.schemas import (
    Box,
    DecisionOut,
    DetectionOut,
    ErrorResponse,
    HealthResponse,
    ImageMeta,
    InputWarningOut,
    ModelInfoResponse,
    ModelSummary,
    PredictResponse,
    ReadyResponse,
)
from btd.api.settings import Settings
from btd.constants import DISCLAIMER
from btd.inference.engine import ModelLoadError, SegmentationEngine
from btd.inference.postprocess import mask_to_polygons
from btd.inference.quality import InputWarning, input_warnings
from btd.inference.types import Prediction
from btd.inference.visualize import draw_overlay
from btd.utils import setup_logging

LOGGER = logging.getLogger("btd.api")
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/bmp",
    "image/x-ms-bmp",
    "image/tiff",
    "image/webp",
    "application/x-image",  # what AWS suggests for images sent to a SageMaker endpoint
    "application/octet-stream",
}
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
STATIC_DIR = Path(__file__).with_name("static")
# The page loads nothing from other origins and only talks to this API, so it can be locked down tightly.
UI_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; "
        "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Vary": "Accept",
}


ConfQuery = Annotated[
    float | None,
    Query(ge=0.0, le=1.0, description="show detections above this confidence (decision threshold is fixed)"),
]
Upload = Annotated[UploadFile, File(description="MRI slice (JPEG/PNG/BMP/TIFF/WebP)")]


@dataclass
class AppState:
    settings: Settings
    engine: SegmentationEngine | None = None
    load_error: str | None = None
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(4))
    metrics: Any = None


class Metrics:
    """Prometheus metrics on a private registry (safe to build several apps in one process, e.g. tests)."""

    def __init__(self) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = CollectorRegistry()
        self.requests = Counter(
            "btd_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "btd_http_request_seconds",
            "HTTP request latency",
            ["route"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=self.registry,
        )
        self.inference = Histogram(
            "btd_inference_seconds",
            "Model pipeline latency (pre + inference + post)",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
            registry=self.registry,
        )
        self.predictions = Counter(
            "btd_predictions_total", "Image-level decisions", ["label"], registry=self.registry
        )
        self.input_warnings = Counter(
            "btd_input_warnings_total",
            "Inputs flagged as unlikely MRI slices",
            ["code"],
            registry=self.registry,
        )
        self.inflight = Gauge("btd_inflight_inferences", "Inferences in progress", registry=self.registry)
        self.model_info = Gauge(
            "btd_model_info",
            "Loaded model",
            ["name", "version", "precision", "provider"],
            registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def _load_engine(settings: Settings) -> SegmentationEngine:
    engine = SegmentationEngine(
        settings.model_path,
        meta_path=settings.model_meta,
        providers=settings.providers,
        conf=settings.conf_threshold,
        iou=settings.iou_threshold,
        max_det=settings.max_det,
        intra_op_threads=settings.intra_op_threads,
        gpu_mem_limit_mb=settings.gpu_mem_limit_mb,
        verify_checksum=settings.verify_checksum,
    )
    if settings.warmup_runs:
        engine.warmup(settings.warmup_runs)
    return engine


def create_app(settings: Settings | None = None, engine: SegmentationEngine | None = None) -> FastAPI:
    """Application factory (``uvicorn btd.api.app:create_app --factory``)."""
    settings = settings or Settings()
    setup_logging(settings.log_level, settings.log_json)
    # OpenCV refuses to decode images above this many pixels (decompression-bomb guard).
    os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(settings.max_image_pixels))
    state = AppState(settings=settings, semaphore=asyncio.Semaphore(settings.max_concurrency))
    state.metrics = Metrics() if settings.enable_metrics else None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if engine is not None:
            state.engine = engine
        else:
            try:
                state.engine = await run_in_threadpool(_load_engine, settings)
            except ModelLoadError as exc:
                state.load_error = str(exc)
                LOGGER.error("Model not loaded: %s", exc)
                if settings.require_model:
                    raise
        if state.engine is not None and state.metrics is not None:
            info = state.engine.info
            state.metrics.model_info.labels(info.name, info.version, info.precision, info.providers[0]).set(1)
        yield

    app = FastAPI(
        title="Brain Tumour Detection API",
        version=__version__,
        description=(
            "YOLO26 instance segmentation of glioma, meningioma and pituitary tumours on T1 MRI slices "
            f"(BRISC 2025), served from a quantised ONNX model. {DISCLAIMER}"
        ),
        lifespan=lifespan,
        responses={503: {"model": ErrorResponse}},
    )
    app.state.btd = state

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + 64 * 1024:
            response: Response = JSONResponse(
                {"detail": f"Upload exceeds {settings.max_upload_mb:g} MB", "request_id": request_id},
                status_code=413,
            )
        else:
            try:
                response = await call_next(request)
            except Exception:  # last-resort guard: log with request id, never leak internals
                LOGGER.exception("Unhandled error", extra={"request_id": request_id})
                response = JSONResponse(
                    {"detail": "Internal server error", "request_id": request_id},
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )
        elapsed = time.perf_counter() - start
        route = getattr(request.scope.get("route"), "path", "unmatched")
        response.headers["X-Request-ID"] = request_id
        response.headers["Server-Timing"] = f"app;dur={elapsed * 1000:.1f}"
        if state.metrics is not None and route != "/metrics":
            state.metrics.requests.labels(request.method, route, str(response.status_code)).inc()
            state.metrics.latency.labels(route).observe(elapsed)
        if route not in {"/health", "/metrics"}:
            LOGGER.info(
                "request",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 1),
                },
            )
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {
                "detail": jsonable_encoder(exc.errors()),
                "request_id": getattr(request.state, "request_id", None),
            },
            status_code=422,
        )

    # ------------------------------------------------------------------------------------------ dependencies
    async def require_api_key(request: Request) -> None:
        if settings.api_key is None:
            return
        supplied = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(supplied.encode(), settings.api_key.get_secret_value().encode()):
            raise HTTPException(
                status.HTTP_401_UNAUTHORIZED,
                "Missing or invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    def require_engine() -> SegmentationEngine:
        if state.engine is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, state.load_error or "Model not loaded")
        return state.engine

    def check_content_type(content_type: str | None) -> None:
        ctype = (content_type or "application/octet-stream").split(";")[0].strip().lower()
        if ctype not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Unsupported content type {ctype}")

    async def read_image(file: UploadFile) -> np.ndarray:
        check_content_type(file.content_type)
        return decode_image(await file.read(settings.max_upload_bytes + 1))

    async def read_body_image(request: Request) -> np.ndarray:
        """The raw request body as an image (SageMaker sends the file itself, not a form)."""
        check_content_type(request.headers.get("content-type"))
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > settings.max_upload_bytes:
                break
        return decode_image(bytes(data))

    def decode_image(data: bytes) -> np.ndarray:
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb:g} MB")
        if not data:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty upload")
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "File is not a decodable image (JPEG/PNG/BMP/TIFF/WebP)"
            )
        h, w = img.shape[:2]
        if h * w > settings.max_image_pixels:
            raise HTTPException(413, f"Image has {h * w} pixels; limit is {settings.max_image_pixels}")
        if min(h, w) < 32:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Image is too small (min side 32 px)")
        return img

    def check_input(img: np.ndarray) -> list[InputWarning]:
        found = input_warnings(img)
        if state.metrics is not None:
            for warning in found:
                state.metrics.input_warnings.labels(warning.code).inc()
        return found

    async def infer(engine: SegmentationEngine, img: np.ndarray, conf: float | None) -> Prediction:
        async with state.semaphore:
            if state.metrics is not None:
                state.metrics.inflight.inc()
            t0 = time.perf_counter()
            try:
                pred = await run_in_threadpool(engine.predict, img, conf)
            finally:
                if state.metrics is not None:
                    state.metrics.inflight.dec()
                    state.metrics.inference.observe(time.perf_counter() - t0)
        if state.metrics is not None:
            state.metrics.predictions.labels(pred.decision.label).inc()
        return pred

    def prediction_response(
        request: Request,
        engine: SegmentationEngine,
        pred: Prediction,
        warnings: list[InputWarning],
        include_polygons: bool = True,
        include_mask: bool = False,
    ) -> PredictResponse:
        h, w = pred.image_hw
        info = engine.info
        detections = [
            DetectionOut(
                class_id=d.class_id,
                class_name=d.class_name,
                confidence=round(d.confidence, 4),
                box=Box(
                    x1=round(d.box[0], 1), y1=round(d.box[1], 1), x2=round(d.box[2], 1), y2=round(d.box[3], 1)
                ),
                area_px=d.area_px,
                area_fraction=round(d.area_px / float(h * w), 6),
                polygons=mask_to_polygons(d.mask, d.mask_origin)
                if include_polygons and d.mask is not None
                else None,
            )
            for d in pred.detections
        ]
        mask_b64 = None
        if include_mask:
            ok, buf = cv2.imencode(".png", pred.union_mask().astype(np.uint8) * 255)
            mask_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None
        return PredictResponse(
            request_id=request.state.request_id,
            model=ModelSummary(
                name=info.name, version=info.version, precision=info.precision, provider=info.providers[0]
            ),
            image=ImageMeta(width=w, height=h),
            decision=DecisionOut(
                label=pred.decision.label,
                score=round(pred.decision.score, 4),
                threshold=pred.decision.threshold,
                tumor_detected=pred.decision.tumor_detected,
            ),
            detections=detections,
            mask_png_base64=mask_b64,
            timings_ms={k: round(v, 2) for k, v in pred.timings_ms.items()},
            warnings=[InputWarningOut(code=w.code, message=w.message) for w in warnings],
            disclaimer=DISCLAIMER,
        )

    # ------------------------------------------------------------------------------------------ routes
    @app.get("/", include_in_schema=False)
    async def root(request: Request) -> Response:
        if settings.ui and "text/html" in request.headers.get("accept", ""):
            return FileResponse(STATIC_DIR / "index.html", headers=UI_HEADERS)
        return JSONResponse(
            {"name": "brain-tumor-detection", "version": __version__, "docs": "/docs"},
            headers={"Vary": "Accept"},
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"], responses={503: {"model": ReadyResponse}})
    async def ready() -> Response:
        if state.engine is None:
            body = ReadyResponse(ready=False, detail=state.load_error or "loading")
            return JSONResponse(body.model_dump(), status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
        return JSONResponse(ReadyResponse(ready=True, model=state.engine.info.name).model_dump())

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["model"])
    async def model_info(engine: Annotated[SegmentationEngine, Depends(require_engine)]) -> dict[str, Any]:
        return engine.info.as_dict()

    @app.post(
        "/v1/predict",
        response_model=PredictResponse,
        tags=["inference"],
        dependencies=[Depends(require_api_key)],
        responses={
            400: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
        },
    )
    async def predict(
        request: Request,
        engine: Annotated[SegmentationEngine, Depends(require_engine)],
        file: Upload,
        conf: ConfQuery = None,
        include_polygons: Annotated[bool, Query(description="return mask outlines")] = True,
        include_mask: Annotated[bool, Query(description="return the union mask as base64 PNG")] = False,
    ) -> PredictResponse:
        img = await read_image(file)
        warnings = check_input(img)
        pred = await infer(engine, img, conf)
        return prediction_response(request, engine, pred, warnings, include_polygons, include_mask)

    @app.post(
        "/v1/predict/overlay",
        tags=["inference"],
        dependencies=[Depends(require_api_key)],
        response_class=Response,
        responses={200: {"content": {"image/png": {}}}, 400: {"model": ErrorResponse}},
    )
    async def predict_overlay(
        engine: Annotated[SegmentationEngine, Depends(require_engine)],
        file: Upload,
        conf: ConfQuery = None,
    ) -> Response:
        img = await read_image(file)
        warnings = check_input(img)
        pred = await infer(engine, img, conf)
        ok, buf = cv2.imencode(".png", draw_overlay(img, pred))
        if not ok:  # pragma: no cover
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not encode overlay")
        headers = {"X-Decision": pred.decision.label, "X-Decision-Score": f"{pred.decision.score:.4f}"}
        if warnings:
            headers["X-Input-Warnings"] = ",".join(w.code for w in warnings)
        return Response(content=buf.tobytes(), media_type="image/png", headers=headers)

    if settings.sagemaker:
        # SageMaker's container contract: GET /ping for health checks and POST /invocations with the raw image as
        # the request body. Only added when asked for: on SageMaker, AWS authenticates every call before it reaches
        # the container, but anywhere else /invocations would just be a second way in.
        @app.get("/ping", include_in_schema=False)
        async def ping() -> Response:
            return Response(status_code=status.HTTP_200_OK if state.engine is not None else 503)

        @app.post(
            "/invocations",
            response_model=PredictResponse,
            tags=["sagemaker"],
            dependencies=[Depends(require_api_key)],
            responses={
                400: {"model": ErrorResponse},
                413: {"model": ErrorResponse},
                415: {"model": ErrorResponse},
            },
        )
        async def invocations(
            request: Request, engine: Annotated[SegmentationEngine, Depends(require_engine)]
        ) -> PredictResponse:
            img = await read_body_image(request)
            warnings = check_input(img)
            pred = await infer(engine, img, None)
            return prediction_response(request, engine, pred, warnings)

    if settings.ui:
        app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="ui")

    if settings.enable_metrics:

        @app.get("/metrics", include_in_schema=False)
        async def metrics() -> Response:
            payload, ctype = state.metrics.render()
            return Response(payload, media_type=ctype)

    return app
