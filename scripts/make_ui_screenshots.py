"""Take the README screenshots of the web page with headless Chrome or Edge.

    btd serve --model models/model.onnx          # in another terminal, with the released model
    python scripts/make_ui_screenshots.py [--url http://127.0.0.1:8000] [--browser PATH]

Each shot opens a sample slice through a link such as /?sample=glioma&expert=1, so the images show real predictions
from whichever model the server is running. The browser is driven over the Chrome DevTools Protocol, which sets the
screen size, the phone emulation and light or dark mode exactly, and waits until the prediction is on the page.
A throwaway browser profile is used, never your own. Needs the `websockets` package (part of the `serve` extra).
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "images"
# file name, link parameters, CSS width, colour scheme, phone
SHOTS = [
    ("web-page.png", "?sample=glioma&expert=1", 1180, "light", False),
    ("web-page-dark.png", "?sample=meningioma", 1180, "dark", False),
    ("web-page-phone.png", "?sample=pituitary", 390, "light", True),
]
CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
]
# True once the page shows a finished prediction (not the placeholder, not "Analysing…").
READY_JS = (
    "(() => { const l = document.getElementById('label').textContent;"
    " return l !== '\u2014' && !l.startsWith('Analysing') && document.getElementById('timing').textContent !== ''; })()"
)


def find_browser(explicit: str | None) -> str:
    for candidate in [explicit] if explicit else CANDIDATES:
        found = shutil.which(candidate) or (candidate if candidate and Path(candidate).is_file() else None)
        if found:
            return found
    sys.exit("No Chrome or Edge found - pass --browser PATH")


class DevTools:
    """Minimal synchronous Chrome DevTools Protocol client for one page."""

    def __init__(self, ws_url: str) -> None:
        self.ws = connect(ws_url, max_size=None)
        self.ids = itertools.count(1)

    def send(self, method: str, **params: Any) -> dict[str, Any]:
        msg_id = next(self.ids)
        self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            reply = json.loads(self.ws.recv())
            if reply.get("id") == msg_id:
                if "error" in reply:
                    raise RuntimeError(f"{method}: {reply['error']}")
                return reply.get("result", {})

    def wait_for(self, expression: str, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.send("Runtime.evaluate", expression=expression, returnByValue=True)["result"].get(
                "value"
            ):
                return
            time.sleep(0.25)
        raise TimeoutError(f"page never satisfied: {expression}")


def start_browser(browser: str, profile: str) -> tuple[subprocess.Popen[bytes], str]:
    proc = subprocess.Popen(
        [
            browser,
            "--headless=new",
            f"--user-data-dir={profile}",
            "--remote-debugging-port=0",
            "--no-first-run",
            "--disable-extensions",
            "--hide-scrollbars",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    port_file = Path(profile) / "DevToolsActivePort"
    for _ in range(100):
        if port_file.is_file() and port_file.read_text().strip():
            port = port_file.read_text().splitlines()[0]
            break
        time.sleep(0.1)
    else:
        proc.kill()
        sys.exit("The browser didn't open its DevTools port")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=10) as resp:
        page = next(t for t in json.load(resp) if t["type"] == "page")
    return proc, page["webSocketDebuggerUrl"]


def shoot(tools: DevTools, url: str, width: int, scheme: str, phone: bool, out: Path) -> None:
    scale = 2 if phone else 1
    tools.send("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": scheme}])
    # Start short: the page's own height (measured below) then sets the screenshot height, with no empty band.
    tools.send(
        "Emulation.setDeviceMetricsOverride", width=width, height=400, deviceScaleFactor=scale, mobile=phone
    )
    tools.send("Page.navigate", url=url)
    # Match the URL too, so the previous shot's page (already complete) can't satisfy the checks.
    tools.wait_for(f"location.href === {json.dumps(url)} && document.readyState === 'complete'")
    tools.wait_for(READY_JS)
    time.sleep(0.5)  # let the confidence bar finish its transition
    height = tools.send(
        "Runtime.evaluate", expression="document.documentElement.scrollHeight", returnByValue=True
    )
    full = int(height["result"]["value"])
    tools.send(
        "Emulation.setDeviceMetricsOverride", width=width, height=full, deviceScaleFactor=scale, mobile=phone
    )
    time.sleep(0.3)
    shot = tools.send("Page.captureScreenshot", format="png")
    out.write_bytes(base64.b64decode(shot["data"]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--browser")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        proc, ws_url = start_browser(find_browser(args.browser), profile)
        try:
            tools = DevTools(ws_url)
            tools.send("Page.enable")
            for name, query, width, scheme, phone in SHOTS:
                shoot(tools, f"{args.url.rstrip('/')}/{query}", width, scheme, phone, OUT / name)
                print(f"wrote {OUT / name}")
            tools.ws.close()
        finally:
            proc.terminate()
            proc.wait(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
