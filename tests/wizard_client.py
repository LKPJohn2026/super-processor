"""Requests that carry the wizard's token, as every wizard route needs."""

from __future__ import annotations

from http.client import HTTPResponse
from pathlib import Path
from typing import cast
from urllib.request import Request, urlopen

from super_processor.review import TOKEN_HEADER


def auth(token: str | None, **headers: str) -> dict[str, str]:
    """``headers`` plus the token header (urllib keeps it across redirects)."""
    return {TOKEN_HEADER: token or "", **headers}


def get(url: str, token: str | None, **headers: str) -> HTTPResponse:
    """``urlopen`` of ``url`` with the token; use it as a context manager."""
    return cast(HTTPResponse, urlopen(Request(url, headers=auth(token, **headers))))


# A tiny stand-in for the built web app (`npm run build:package`), shaped
# like Vite's output, so tests do not depend on a real build being there.
FAKE_INDEX = (
    "<!doctype html><html><head>"
    '<script type="module" crossorigin src="/assets/index-test.js"></script>'
    '<link rel="stylesheet" crossorigin href="/assets/index-test.css">'
    '</head><body><div id="root"></div></body></html>'
)
FAKE_ASSETS = {
    "index-test.js": "document.title = 'app';\n",
    "index-test.css": "body { margin: 0 }\n",
    "font-test.woff2": "wOF2",
}


def fake_web_app(root: Path) -> Path:
    """Write the stand-in app into ``root`` and return it."""
    (root / "assets").mkdir(parents=True, exist_ok=True)
    # Bytes, so Windows does not turn "\n" into "\r\n".
    (root / "index.html").write_bytes(FAKE_INDEX.encode())
    for name, text in FAKE_ASSETS.items():
        (root / "assets" / name).write_bytes(text.encode())
    return root
