"""Regenerate the hash-locked serving requirements with uv.

The Docker images install requirements/serve-*.txt with ``pip --require-hashes --no-deps``, so each lock file must be a
complete, mutually compatible set of pins. Change them only through this script (or ``make lock``), never by hand or
through Dependabot: bumping one package on its own (e.g. pydantic-core without pydantic) breaks the image at start-up.

    python scripts/lock.py             # after editing pyproject.toml: re-lock, keeping existing pins where possible
    python scripts/lock.py --upgrade   # move every pin to the newest compatible release
    python scripts/lock.py --check     # exit 1 if a lock file is out of date with pyproject.toml (CI runs this)

Needs uv (``pip install uv``). The locks target Linux x86_64 and Python 3.12, the platform of docker/Dockerfile*.
"""

from __future__ import annotations

import argparse
import difflib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = ["--python-version", "3.12", "--python-platform", "x86_64-manylinux_2_28"]
LOCKS: dict[str, list[str]] = {
    "requirements/serve-cpu.txt": [
        "pyproject.toml",
        "--extra",
        "serve",
        "--extra",
        "cpu",
        "--extra",
        "headless",
    ],
    "requirements/serve-gpu.txt": [
        "pyproject.toml",
        "requirements/gpu.in",
        "--extra",
        "serve",
        "--extra",
        "headless",
    ],
}


def uv_command() -> list[str]:
    exe = shutil.which("uv")
    if exe:
        return [exe]
    try:
        from uv import find_uv_bin  # the `uv` wheel ships the binary plus this locator
    except ImportError:
        sys.exit("uv is not installed - run `pip install uv`")
    return [find_uv_bin()]


def compile_lock(inputs: list[str], out: Path, upgrade: bool) -> None:
    cmd = [
        *uv_command(),
        "pip",
        "compile",
        *inputs,
        *TARGET,
        "--generate-hashes",
        "--quiet",
        "--custom-compile-command",
        "make lock",
        "-o",
        str(out),
    ]
    if upgrade:
        cmd.append("--upgrade")
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--upgrade", action="store_true", help="move every pin to the newest compatible release"
    )
    mode.add_argument("--check", action="store_true", help="fail if a lock file is stale (writes nothing)")
    args = ap.parse_args()

    stale: list[str] = []
    for rel, inputs in LOCKS.items():
        lock = REPO / rel
        if not args.check:
            compile_lock(inputs, lock, upgrade=args.upgrade)
            print(f"wrote {rel}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            fresh = Path(tmp) / lock.name
            shutil.copyfile(lock, fresh)  # uv keeps the pins already present in the output file
            compile_lock(inputs, fresh, upgrade=False)
            old = lock.read_text(encoding="utf-8").splitlines(keepends=True)
            new = fresh.read_text(encoding="utf-8").splitlines(keepends=True)
        if old != new:
            stale.append(rel)
            sys.stdout.writelines(difflib.unified_diff(old, new, rel, f"{rel} (re-locked)", n=1))

    if stale:
        print(
            f"\nOut of date with pyproject.toml: {', '.join(stale)}. Run `make lock` (or `python scripts/lock.py`)."
        )
        return 1
    if args.check:
        print("Lock files are up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
