from __future__ import annotations

import base64
import re
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from btd.api.app import create_app
from btd.api.settings import Settings
from tests.helpers import encode


def _client(model: Path | None, **overrides: object) -> TestClient:
    settings = Settings(
        model_path=model or Path("missing/model.onnx"),
        providers="cpu",
        log_json=False,
        warmup_runs=1,
        **overrides,  # type: ignore[arg-type]
    )
    return TestClient(create_app(settings))


@pytest.fixture
def client(dummy_model: Path) -> Iterator[TestClient]:
    with _client(dummy_model) as c:
        yield c


def test_health_ready_model(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json() == {
        "ready": True,
        "model": "dummy-end2end",
        "detail": None,
    }
    info = client.get("/v1/model").json()
    assert info["class_names"] == ["glioma", "meningioma", "pituitary"]
    assert info["precision"] == "fp32" and info["sha256"]
    assert client.get("/").json()["docs"] == "/docs"
    assert client.get("/openapi.json").status_code == 200


def test_predict_tumour(client: TestClient, bright_image: np.ndarray) -> None:
    r = client.post(
        "/v1/predict?include_mask=true",
        files={"file": ("slice.png", encode(bright_image), "image/png")},
        headers={"X-Request-ID": "abc-123"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert r.headers["x-request-id"] == "abc-123" == body["request_id"]
    assert body["decision"]["label"] == "meningioma" and body["decision"]["tumor_detected"]
    assert body["image"] == {"width": 320, "height": 240}
    det = body["detections"][0]
    assert det["class_name"] == "meningioma" and 0 < det["area_fraction"] < 1
    assert det["polygons"] and len(det["polygons"][0]) >= 3
    mask = cv2.imdecode(
        np.frombuffer(base64.b64decode(body["mask_png_base64"]), np.uint8), cv2.IMREAD_GRAYSCALE
    )
    assert mask.shape == (240, 320) and set(np.unique(mask)) <= {0, 255}
    assert "Not a medical device" in body["disclaimer"]
    assert body["timings_ms"]["total_ms"] > 0


def test_predict_healthy_jpeg(client: TestClient, black_image: np.ndarray) -> None:
    r = client.post("/v1/predict", files={"file": ("x.jpg", encode(black_image, ".jpg"), "image/jpeg")})
    assert r.status_code == 200
    assert r.json()["decision"]["label"] == "no_tumor"
    assert r.json()["detections"] == []


def test_overlay_png(client: TestClient, bright_image: np.ndarray) -> None:
    r = client.post("/v1/predict/overlay", files={"file": ("x.png", encode(bright_image), "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-decision"] == "meningioma"
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (240, 320, 3)


@pytest.mark.parametrize(
    ("payload", "ctype", "status"),
    [
        (b"definitely not an image", "image/png", 400),
        (b"", "image/png", 400),
        (b"%PDF-1.7", "application/pdf", 415),
    ],
)
def test_bad_uploads(client: TestClient, payload: bytes, ctype: str, status: int) -> None:
    r = client.post("/v1/predict", files={"file": ("f", payload, ctype)})
    assert r.status_code == status
    assert r.json()["request_id"]


def test_tiny_and_oversized_images(dummy_model: Path) -> None:
    with _client(dummy_model, max_upload_mb=0.01, max_image_pixels=200 * 200) as c:
        tiny = c.post(
            "/v1/predict", files={"file": ("t.png", encode(np.zeros((8, 8, 3), np.uint8)), "image/png")}
        )
        assert tiny.status_code == 400
        noisy = np.random.default_rng(0).integers(0, 255, (300, 300, 3), dtype=np.uint8)
        big = c.post("/v1/predict", files={"file": ("b.png", encode(noisy), "image/png")})
        assert big.status_code == 413


def test_validation_error_has_request_id(client: TestClient, bright_image: np.ndarray) -> None:
    r = client.post("/v1/predict?conf=7", files={"file": ("x.png", encode(bright_image), "image/png")})
    assert r.status_code == 422 and r.json()["request_id"]
    assert client.get("/nope").status_code == 404


def test_api_key(dummy_model: Path, bright_image: np.ndarray) -> None:
    with _client(dummy_model, api_key="s3cret") as c:
        files = {"file": ("x.png", encode(bright_image), "image/png")}
        assert c.post("/v1/predict", files=files).status_code == 401
        assert c.post("/v1/predict", files=files, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/v1/predict", files=files, headers={"X-API-Key": "s3cret"}).status_code == 200
        assert c.get("/health").status_code == 200  # ops endpoints stay open for probes


def test_metrics(client: TestClient, bright_image: np.ndarray) -> None:
    client.post("/v1/predict", files={"file": ("x.png", encode(bright_image), "image/png")})
    text = client.get("/metrics").text
    assert 'btd_predictions_total{label="meningioma"} 1.0' in text
    assert "btd_http_requests_total" in text and "btd_model_info" in text


def test_missing_model_reports_not_ready() -> None:
    with _client(None) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/ready")
        assert r.status_code == 503 and "not found" in r.json()["detail"]
        p = c.post(
            "/v1/predict", files={"file": ("x.png", encode(np.zeros((64, 64, 3), np.uint8)), "image/png")}
        )
        assert p.status_code == 503


def test_require_model_fails_fast() -> None:
    with pytest.raises(Exception, match="not found"), _client(None, require_model=True):
        pass


def test_web_page(client: TestClient) -> None:
    page = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    assert "<title>Brain tumour detection</title>" in page.text
    assert "script-src 'self'" in page.headers["content-security-policy"]
    assert "Not a medical device" in page.text
    # every asset the page references is served
    for path in re.findall(r'(?:src|href)="(/ui/[^"]+)"', page.text):
        assert client.get(path).status_code == 200, path

    samples = client.get("/ui/samples/samples.json").json()["samples"]
    assert {s["label"] for s in samples} == {"glioma", "meningioma", "pituitary", "no_tumor"}
    for s in samples:
        img = client.get(f"/ui/samples/{s['file']}")
        assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
        assert bool(s["expert_polygons"]) == (s["label"] != "no_tumor")


def test_web_page_can_be_turned_off(dummy_model: Path) -> None:
    with _client(dummy_model, ui=False) as c:
        assert c.get("/", headers={"Accept": "text/html"}).json()["docs"] == "/docs"
        assert c.get("/ui/app.js").status_code == 404


def test_cors(dummy_model: Path) -> None:
    with _client(dummy_model, cors_origins=["http://localhost:5173"]) as c:
        r = c.options(
            "/v1/predict",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
        )
        assert r.headers["access-control-allow-origin"] == "http://localhost:5173"
