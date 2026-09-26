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


MANIFEST = b"relative_path,sha256\nclassification_task/test/glioma/a.jpg,00\n"


class _Server:
    def __init__(
        self,
        payload: bytes,
        md5: str | None = None,
        manifest: bytes | None = MANIFEST,
        manifest_md5: str | None = None,
    ) -> None:
        self.payload = payload
        self.md5 = md5 or hashlib.md5(payload).hexdigest()
        self.manifest = manifest
        self.manifest_md5 = manifest_md5 or hashlib.md5(manifest or b"").hexdigest()
        self.range_requests = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # silence
                pass

            def do_GET(self) -> None:
                if self.path.startswith("/api/records/"):
                    files = [
                        {
                            "key": "brisc2025.zip",
                            "size": len(outer.payload),
                            "checksum": f"md5:{outer.md5}",
                            "links": {"self": f"{outer.base}/files/brisc2025.zip"},
                        }
                    ]
                    if outer.manifest is not None:
                        files.append(
                            {
                                "key": "manifest.csv",
                                "size": len(outer.manifest),
                                "checksum": f"md5:{outer.manifest_md5}",
                                "links": {"self": f"{outer.base}/files/manifest.csv"},
                            }
                        )
                    body = json.dumps({"files": files}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.endswith("/manifest.csv") and outer.manifest is not None:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(outer.manifest)))
                    self.end_headers()
                    self.wfile.write(outer.manifest)
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
    # the manifest is published next to the zip, not inside it
    assert (tmp_path / "manifest.csv").read_bytes() == MANIFEST
    # second call re-uses the verified archive and manifest (no download)
    download_brisc(tmp_path, api=server.api)
    assert (tmp_path / "manifest.csv").read_bytes() == MANIFEST


def test_manifest_md5_mismatch_removes_file(tmp_path: Path) -> None:
    srv = _Server(_zip_bytes({"a.txt": b"a"}), manifest_md5="0" * 32)
    srv.thread.start()
    try:
        with pytest.raises(DownloadError, match=r"MD5 mismatch for manifest\.csv"):
            download_brisc(tmp_path, api=srv.api)
        assert not (tmp_path / "manifest.csv").exists()
    finally:
        srv.httpd.shutdown()


def test_record_without_manifest_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    srv = _Server(_zip_bytes({"a.txt": b"a"}), manifest=None)
    srv.thread.start()
    try:
        download_brisc(tmp_path, api=srv.api)
    finally:
        srv.httpd.shutdown()
    assert (tmp_path / "a.txt").exists()
    assert not (tmp_path / "manifest.csv").exists()
    assert "can't check the images' SHA-256" in caplog.text


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


def test_manual_zip_and_zip_slip(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good = tmp_path / "good.zip"
    good.write_bytes(_zip_bytes({"brisc2025/manifest.csv": b"relative_path,sha256\n"}))
    # offline (unreachable Zenodo): the manual zip is still extracted, with a warning about the manifest
    download_brisc(tmp_path / "out", zip_path=good, api="http://127.0.0.1:9/api/records/{record}")
    assert (tmp_path / "out" / "brisc2025" / "manifest.csv").exists()
    assert "Could not fetch manifest.csv" in caplog.text

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
