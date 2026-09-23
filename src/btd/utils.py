"""Small shared helpers: logging, hashing, JSON I/O, image I/O and timing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger("btd")


class JsonFormatter(logging.Formatter):
    """One JSON object per line — easy to ship to Loki/ELK/CloudWatch."""

    _RESERVED = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None)).keys()) | {
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extras = {k: v for k, v in vars(record).items() if k not in self._RESERVED and not k.startswith("_")}
        payload.update(extras)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure the ``btd`` logger once (idempotent)."""
    handler = logging.StreamHandler(sys.stderr)
    fmt: logging.Formatter
    if json_logs:
        fmt = JsonFormatter()
    else:
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    handler.setFormatter(fmt)
    root = logging.getLogger("btd")
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    root.propagate = False


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    """Streaming MD5 (used only to match checksums published by Zenodo, not for security)."""
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | os.PathLike[str], data: Any) -> Path:
    """Write pretty JSON (numpy-aware) and create parent folders."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return p


def read_json(path: str | os.PathLike[str]) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def imread(path: str | os.PathLike[str]) -> np.ndarray:
    """Read an image as 3-channel BGR uint8. Works with non-ASCII / spaced Windows paths (unlike cv2.imread)."""
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def imread_gray(path: str | os.PathLike[str]) -> np.ndarray:
    import cv2

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def imwrite(path: str | os.PathLike[str], img: np.ndarray, params: list[int] | None = None) -> None:
    """Write an image with cv2.imencode so non-ASCII / spaced paths work on Windows."""
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", img, params or [])
    if not ok:
        raise ValueError(f"Could not encode image for {p}")
    buf.tofile(str(p))


@contextmanager
def timer(store: dict[str, float], key: str) -> Iterator[None]:
    """Accumulate elapsed milliseconds into ``store[key]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (time.perf_counter() - t0) * 1000.0


def git_commit(cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Best-effort current git commit (for run provenance)."""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
