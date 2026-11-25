"""The web app built into the package is served whole and under the CSP.

Runs only with SUPER_PROCESSOR_REQUIRE_WEB_DIST=1, which CI and the release
set where a real build is in place: the cibuildwheel test step runs it
against each installed wheel on Linux, macOS and Windows.
"""

from __future__ import annotations

import os
import re
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from super_processor import review
from super_processor.review import ASSET_TYPES, CONTENT_SECURITY_POLICY, WizardServer

pytestmark = pytest.mark.skipif(
    os.environ.get("SUPER_PROCESSOR_REQUIRE_WEB_DIST") != "1",
    reason="needs the built web app (SUPER_PROCESSOR_REQUIRE_WEB_DIST=1)",
)


def _get(
    base: str, path: str, headers: dict[str, str]
) -> tuple[int, dict[str, str], bytes]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    try:
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def test_the_installed_package_has_the_web_app() -> None:
    root = review.web_dist_dir()
    assert (root / "index.html").is_file(), root
    names = sorted(path.name for path in (root / "assets").iterdir())
    assert any(name.endswith(".js") for name in names), names
    assert any(name.endswith(".css") for name in names), names
    for name in names:
        # Everything built is a file the server will hand out.
        assert review.app_asset(f"/assets/{name}") is not None, name


def test_the_built_app_is_served_with_its_types_and_the_csp(tmp_path: Path) -> None:
    root = review.web_dist_dir()
    server = WizardServer(tmp_path / "jobs")
    base = server.start()
    try:
        status, headers, _body = _get(base, f"/?token={server.token}", {})
        assert status == 303 and headers["Location"] == "/"
        browser = {
            "Cookie": headers["Set-Cookie"].split(";", 1)[0],
            "Sec-Fetch-Site": "same-origin",
        }
        status, headers, page = _get(base, "/", browser)
        assert status == 200
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        assert page == (root / "index.html").read_bytes()
        html = page.decode("utf-8")
        assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "inline script"
        assert "<style" not in html
        referenced = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
        assert any(ref.endswith(".js") for ref in referenced), referenced
        for path in (root / "assets").iterdir():
            status, headers, body = _get(base, f"/assets/{path.name}", browser)
            assert status == 200, path.name
            assert headers["Content-Type"] == ASSET_TYPES[path.suffix], path.name
            assert headers["X-Content-Type-Options"] == "nosniff"
            assert body == path.read_bytes(), path.name
            if path.suffix == ".css":
                assert "url(data:" not in body.decode("utf-8")
        for ref in referenced:
            assert _get(base, ref, browser)[0] == 200, ref
    finally:
        server.stop()
