"""Start CampusBot and open the local app in the browser."""

from __future__ import annotations

import sys
import threading
import webbrowser
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


def open_local_app() -> None:
    url = "http://127.0.0.1:8000/"
    print(f"Opening CampusBot in your browser: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    timer = threading.Timer(1.5, open_local_app)
    timer.daemon = True
    timer.start()
    print("Starting CampusBot on http://127.0.0.1:8000/")
    uvicorn.run(app, host="127.0.0.1", port=8000)
