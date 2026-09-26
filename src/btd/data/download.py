"""Download BRISC 2025 from Zenodo with checksum verification, resume support and safe extraction.

Zenodo publishes an MD5 for every file in its records API; the archive is verified before extraction, and every
member path is checked so a malicious archive cannot write outside the destination ("zip slip"). The record's
``manifest.csv`` (a SHA-256 for every image) sits next to the archive, not inside it, so it is fetched separately
for ``btd data prepare`` to check each file against.
If Zenodo is unreachable, download ``brisc2025.zip`` manually (Zenodo or Kaggle) and pass ``--zip``.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from btd.constants import BRISC_MANIFEST_NAME, BRISC_ZENODO_RECORD, BRISC_ZIP_NAME
from btd.utils import md5_file

LOGGER = logging.getLogger(__name__)
ZENODO_API = "https://zenodo.org/api/records/{record}"
ZENODO_FILE = "https://zenodo.org/records/{record}/files/{name}?download=1"
USER_AGENT = "brain-tumor-detection/0.1 (+https://github.com/leon2378/brain-tumor-detection)"


class DownloadError(RuntimeError):
    pass


def _request(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0) -> Any:
    if not url.startswith("https://") and not url.startswith("http://127.0.0.1"):
        raise DownloadError(f"Refusing non-HTTPS URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})  # noqa: S310
    return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310


def zenodo_file_info(
    record: str = BRISC_ZENODO_RECORD, name: str = BRISC_ZIP_NAME, api: str = ZENODO_API
) -> dict:
    """Return ``{"url", "size", "md5"}`` for a file of a Zenodo record (InvenioRDM and legacy JSON shapes)."""
    with _request(api.format(record=record)) as resp:
        meta = json.load(resp)
    files = meta.get("files", [])
    if isinstance(files, dict):  # /api/records/<id>/files style
        files = files.get("entries", [])
    for f in files:
        if f.get("key") == name or f.get("filename") == name:
            checksum = str(f.get("checksum", ""))
            md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
            links = f.get("links", {})
            url = links.get("content") or links.get("self") or links.get("download")
            if url and not url.endswith("/content") and "/api/" in url:
                url = url.rstrip("/") + "/content"
            return {
                "url": url or ZENODO_FILE.format(record=record, name=name),
                "size": f.get("size"),
                "md5": md5,
            }
    raise DownloadError(f"{name} not found in Zenodo record {record}")


def download_file(url: str, dest: Path, expected_size: int | None = None, retries: int = 5) -> Path:
    """Stream to ``dest.part`` with HTTP Range resume and exponential back-off, then atomically rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _request(url, headers=headers, timeout=120) as resp:
                status = getattr(resp, "status", 200)
                if have and status != 206:  # server ignored Range → restart
                    have = 0
                total = expected_size or (int(resp.headers.get("Content-Length", 0)) + have) or None
                mode = "ab" if have else "wb"
                t0, done, last_log = time.time(), have, 0.0
                with part.open(mode) as f:
                    while chunk := resp.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last_log > 5:
                            last_log = time.time()
                            rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                            pct = f"{100 * done / total:5.1f}%" if total else f"{done / 1e6:.0f} MB"
                            LOGGER.info("  %s  (%.1f MB/s)", pct, rate)
            if expected_size and part.stat().st_size != expected_size:
                raise DownloadError(f"size mismatch: {part.stat().st_size} != {expected_size}")
            part.replace(dest)
            return dest
        except (urllib.error.URLError, TimeoutError, ConnectionError, DownloadError) as exc:
            if attempt == retries:
                raise DownloadError(f"Download failed after {retries} attempts: {exc}") from exc
            wait = min(60, 2**attempt)
            LOGGER.warning(
                "Download interrupted (%s); retrying in %ss (attempt %d/%d)", exc, wait, attempt, retries
            )
            time.sleep(wait)
    raise DownloadError("unreachable")  # pragma: no cover


def safe_extract(zip_path: Path, dest: Path) -> Path:
    """Extract a zip after verifying that no member escapes ``dest``."""
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if dest != target and dest not in target.parents:
                raise DownloadError(f"Unsafe path in archive: {member.filename}")
        zf.extractall(dest)
    return dest


def fetch_manifest(dest_dir: Path, record: str = BRISC_ZENODO_RECORD, api: str = ZENODO_API) -> Path | None:
    """Fetch and MD5-verify ``manifest.csv`` into ``dest_dir``, where ``btd data prepare`` looks for it.

    Returns None with a warning when Zenodo can't be reached (e.g. offline with ``--zip``): the data is still
    usable, only not checked file by file.
    """
    target = dest_dir / BRISC_MANIFEST_NAME
    try:
        info = zenodo_file_info(record, BRISC_MANIFEST_NAME, api)
        if not (target.is_file() and info["md5"] and md5_file(target) == info["md5"]):
            download_file(info["url"], target, info["size"])
    except (DownloadError, urllib.error.URLError, OSError) as exc:
        LOGGER.warning(
            "Could not fetch %s (%s), so `btd data prepare` can't check the images' SHA-256. "
            "Download it from https://zenodo.org/records/%s into %s",
            BRISC_MANIFEST_NAME,
            exc,
            record,
            dest_dir,
        )
        return None
    if info["md5"]:
        got = md5_file(target)
        if got != info["md5"]:
            target.unlink(missing_ok=True)
            raise DownloadError(
                f"MD5 mismatch for {BRISC_MANIFEST_NAME} ({got} != {info['md5']}); corrupted download removed"
            )
    LOGGER.info("Manifest ready: %s", target)
    return target


def download_brisc(
    dest_dir: str | Path = "data/raw",
    zip_path: str | Path | None = None,
    record: str = BRISC_ZENODO_RECORD,
    api: str = ZENODO_API,
    keep_zip: bool = True,
) -> Path:
    """Fetch (or reuse) ``brisc2025.zip``, verify it, extract it and fetch its manifest. Returns ``dest_dir``."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = Path(zip_path) if zip_path else dest_dir / BRISC_ZIP_NAME
    expected_md5: str | None = None

    if zip_path is None:
        try:
            info = zenodo_file_info(record, BRISC_ZIP_NAME, api)
            expected_md5 = info["md5"]
            if archive.exists() and expected_md5 and md5_file(archive) == expected_md5:
                LOGGER.info("Found verified %s - skipping download", archive)
            else:
                LOGGER.info(
                    "Downloading %s (%.0f MB) from Zenodo...", BRISC_ZIP_NAME, (info["size"] or 0) / 1e6
                )
                download_file(info["url"], archive, info["size"])
        except (DownloadError, urllib.error.URLError, OSError) as exc:
            raise DownloadError(
                f"Could not download BRISC from Zenodo ({exc}). Download brisc2025.zip manually from "
                f"https://zenodo.org/records/{record} or https://www.kaggle.com/datasets/briscdataset/brisc2025 "
                "and run: btd data download --zip <path-to-zip>"
            ) from exc
        if expected_md5:
            got = md5_file(archive)
            if got != expected_md5:
                archive.unlink(missing_ok=True)
                raise DownloadError(f"MD5 mismatch ({got} != {expected_md5}); corrupted download removed")
            LOGGER.info("MD5 verified: %s", expected_md5)
    elif not archive.is_file():
        raise DownloadError(f"Zip not found: {archive}")

    LOGGER.info("Extracting %s -> %s", archive, dest_dir)
    safe_extract(archive, dest_dir)
    if zip_path is None and not keep_zip:
        archive.unlink(missing_ok=True)
    fetch_manifest(dest_dir, record, api)
    return dest_dir
