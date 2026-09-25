"""Black-box smoke test for a running API (stdlib only — used by CI against the Docker container).

    python scripts/smoke_test.py http://localhost:8000 [--expect-dummy] [--api-key KEY] [--sagemaker]

Checks /health, /ready, /v1/model, /metrics, the web page and a /v1/predict round-trip with generated PNGs.
With --expect-dummy it also asserts the decisions of the btd.testing dummy model (bright → tumour, black → none).
With --sagemaker it also checks SageMaker's /ping and a raw-body /invocations call (container started with
BTD_SAGEMAKER=true).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib


def png(width: int, height: int, value: int) -> bytes:
    """Minimal grayscale PNG encoder (no numpy/opencv needed)."""
    raw = b"".join(b"\x00" + bytes([value]) * width for _ in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    )


def request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()


def post_image(base: str, image: bytes, api_key: str | None) -> tuple[int, dict]:
    boundary = uuid.uuid4().hex
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="slice.png"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + image
        + f"\r\n--{boundary}--\r\n".encode()
    )
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["X-API-Key"] = api_key
    status, payload = request(f"{base}/v1/predict", body, headers)
    return status, json.loads(payload or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("--expect-dummy", action="store_true")
    ap.add_argument("--api-key")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for /ready")
    ap.add_argument("--sagemaker", action="store_true", help="also check /ping and /invocations")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    deadline = time.time() + args.wait
    while True:
        try:
            status, body = request(f"{base}/ready")
        except (urllib.error.URLError, ConnectionError):
            status, body = 0, b""
        if status == 200:
            break
        if time.time() > deadline:
            print(f"FAIL: /ready never became 200 (last: {status} {body[:200]!r})")
            return 1
        time.sleep(1)

    checks: list[tuple[str, bool, str]] = []
    status, body = request(f"{base}/health")
    checks.append(("GET /health", status == 200, body.decode()[:80]))
    status, body = request(f"{base}/v1/model")
    info = json.loads(body or b"{}")
    checks.append(("GET /v1/model", status == 200 and bool(info.get("class_names")), str(info.get("name"))))

    status, pred = post_image(base, png(256, 200, 140), args.api_key)
    ok = status == 200 and "decision" in pred and pred["image"] == {"width": 256, "height": 200}
    if args.expect_dummy:
        ok = ok and pred["decision"]["label"] == "meningioma" and len(pred["detections"]) == 1
    checks.append(("POST /v1/predict (bright)", ok, json.dumps(pred.get("decision"))))

    status, pred = post_image(base, png(128, 128, 0), args.api_key)
    ok = status == 200 and (not args.expect_dummy or pred["decision"]["label"] == "no_tumor")
    checks.append(("POST /v1/predict (black)", ok, json.dumps(pred.get("decision"))))

    status, body = request(f"{base}/v1/predict", b"not multipart", {"Content-Type": "text/plain"})
    checks.append(("POST /v1/predict (bad body) → 4xx", 400 <= status < 500, str(status)))

    status, body = request(f"{base}/", headers={"Accept": "text/html"})
    checks.append(
        ("GET / (browser) → web page", status == 200 and b"Brain tumour detection" in body, str(status))
    )
    status, body = request(f"{base}/ui/app.js")
    checks.append(("GET /ui/app.js", status == 200 and b"/v1/predict" in body, f"{len(body)} bytes"))

    if args.sagemaker:
        status, _ = request(f"{base}/ping")
        checks.append(("GET /ping (SageMaker)", status == 200, str(status)))
        status, body = request(f"{base}/invocations", png(256, 200, 140), {"Content-Type": "image/png"})
        pred = json.loads(body or b"{}")
        ok = status == 200 and pred.get("image") == {"width": 256, "height": 200}
        if args.expect_dummy:
            ok = ok and pred["decision"]["label"] == "meningioma"
        checks.append(("POST /invocations (raw PNG)", ok, json.dumps(pred.get("decision"))))

    status, body = request(f"{base}/metrics")
    checks.append(
        ("GET /metrics", status == 200 and b"btd_http_requests_total" in body, f"{len(body)} bytes")
    )

    width = max(len(c[0]) for c in checks)
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
    return 0 if all(c[1] for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
