"""Start the server: python -m backend [--port 8000] [--gpu]

--gpu also opens the app in its own browser window on the high-performance GPU. A browser
already running ignores launch flags, so that window uses a separate profile.
"""
import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn

from backend.config import get_settings

log = logging.getLogger(__name__)
GPU_FLAGS = ["--force_high_performance_gpu", "--use-angle=d3d11", "--ignore-gpu-blocklist"]
PROFILE = Path(tempfile.gettempdir()) / "field-service-navigator-browser"
START_TIMEOUT = 30
WINDOWS_BROWSERS = ["Microsoft/Edge/Application/msedge.exe", "Google/Chrome/Application/chrome.exe"]


def find_browser() -> str | None:
    folders = [os.environ.get(name) for name in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    found = [Path(folder) / name for folder in folders if folder for name in WINDOWS_BROWSERS]
    for path in found:
        if path.exists():
            return str(path)
    return next(filter(None, map(shutil.which, ("msedge", "google-chrome", "chromium", "chrome"))), None)


def browser_command(browser: str, url: str) -> list[str]:
    return [browser, *GPU_FLAGS, f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check", "--start-maximized", f"--app={url}"]


def open_when_ready(url: str):
    browser = find_browser()
    if browser is None:
        log.warning("no Edge or Chrome found, open %s yourself", url)
        return
    deadline = time.time() + START_TIMEOUT
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}api/health", timeout=1)
            subprocess.Popen(browser_command(browser, url))
            return
        except OSError:
            time.sleep(0.2)
    log.warning("server did not answer in %s s, browser not opened", START_TIMEOUT)


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Field-Service Navigator server")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--gpu", action="store_true", help="open the app in a browser window on the high-performance GPU")
    args = parser.parse_args()
    if args.gpu:
        threading.Thread(target=open_when_ready, args=(f"http://127.0.0.1:{args.port}/",), daemon=True).start()
    uvicorn.run("backend.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
