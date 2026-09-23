from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from btd.data.download import DownloadError, download_brisc, safe_extract


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


class _Server:
    def __init__(self, payload: bytes, md5: str | None = None) -> None:
        self.payload = payload
        self.md5 = md5 or hashlib.md5(payload).hexdigest()
        self.range_requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # silence
                pass

            def do_GET(self) -> None:
                if self.path.startswith("/api/records/"):
                    body = json.dumps(
                        {
                            "files": [
                                {
                                    "key": "brisc2025.zip",
                                    "size": len(outer.payload),
                                    "checksum": f"md5:{outer.md5}",
                                    "links": {"self": f"{outer.base}/files/brisc2025.zip"},
                                }
                            ]
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                data, status = outer.payload, 200
                rng = self.headers.get("Range")
                if rng:
                    outer.range_requests += 1
                    start = int(rng.split("=")[1].split("-")[0])
                    data, status = outer.payload[start:], 206
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def api(self) -> str:
        return self.base + "/api/records/{record}"


@pytest.fixture
def server() -> Iterator[_Server]:
    srv = _Server(
        _zip_bytes({"brisc2025/README.md": b"hello", "brisc2025/segmentation_task/x.txt": b"x" * 5000})
    )
    srv.thread.start()
    yield srv
    srv.httpd.shutdown()


def test_download_verify_extract(server: _Server, tmp_path: Path) -> None:
    out = download_brisc(tmp_path, api=server.api)
    assert (out / "brisc2025" / "README.md").read_text() == "hello"
    assert (tmp_path / "brisc2025.zip").is_file()
    # second call re-uses the verified archive (no download)
    download_brisc(tmp_path, api=server.api)


def test_resume_partial_download(server: _Server, tmp_path: Path) -> None:
    (tmp_path / "brisc2025.zip.part").write_bytes(server.payload[: len(server.payload) // 2])
    download_brisc(tmp_path, api=server.api)
    assert server.range_requests == 1
    assert (tmp_path / "brisc2025" / "README.md").exists()


def test_md5_mismatch_removes_file(tmp_path: Path) -> None:
    srv = _Server(_zip_bytes({"a.txt": b"a"}), md5="0" * 32)
    srv.thread.start()
    try:
        with pytest.raises(DownloadError, match="MD5 mismatch"):
            download_brisc(tmp_path, api=srv.api)
        assert not (tmp_path / "brisc2025.zip").exists()
    finally:
        srv.httpd.shutdown()


def test_manual_zip_and_zip_slip(tmp_path: Path) -> None:
    good = tmp_path / "good.zip"
    good.write_bytes(_zip_bytes({"brisc2025/manifest.csv": b"relative_path,sha256\n"}))
    download_brisc(tmp_path / "out", zip_path=good)
    assert (tmp_path / "out" / "brisc2025" / "manifest.csv").exists()

    evil = tmp_path / "evil.zip"
    evil.write_bytes(_zip_bytes({"../../escaped.txt": b"pwned"}))
    with pytest.raises(DownloadError, match="Unsafe path"):
        safe_extract(evil, tmp_path / "x")
    assert not (tmp_path / "escaped.txt").exists()

    with pytest.raises(DownloadError, match="not found"):
        download_brisc(tmp_path, zip_path=tmp_path / "missing.zip")


def test_unreachable_zenodo_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="--zip"):
        download_brisc(tmp_path, api="http://127.0.0.1:9/api/records/{record}")
