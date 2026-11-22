"""Every wizard route needs the run's token; the launch URL sets a cookie.

Also the safety headers on every response, the cross-site media refusal and
the socket timeout and connection cap that stop idle connections pinning
threads.
"""

from __future__ import annotations

import json
import socket
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest

from super_processor import review
from super_processor.review import (
    ReviewServer,
    WizardServer,
    launch_url,
    request_authenticated,
    wizard_cookie_name,
)
from super_processor.segments import TimelineSegment, write_gray_still, write_segments
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)

TOKEN = "auth-test-token"


def _request(
    base: str,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> tuple[int, dict[str, str], bytes]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    try:
        sent = dict(headers or {})
        if body:
            sent.setdefault("Content-Length", str(len(body)))
        connection.request(method, path, body=body, headers=sent)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _port(base: str) -> int:
    return int(urlparse(base).port or 0)


def _cookie(base: str) -> dict[str, str]:
    return {"Cookie": f"{wizard_cookie_name(_port(base))}={TOKEN}"}


def _wizard_with_output(tmp_path: Path) -> tuple[WizardController, str]:
    controller = WizardController(tmp_path / "jobs")
    source = tmp_path / "private.mp4"
    source.write_bytes(b"x")
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    (job_dir / "output.mp4").write_bytes(b"SECRET-VIDEO")
    (job_dir / "shot_stills").mkdir()
    (job_dir / "shot_stills" / "000000000.jpg").write_bytes(b"SECRET-STILL")
    save_session_state(
        controller.jobs_dir, WizardState(step=WizardStep.RESULT, job_id=job_id)
    )
    return controller, job_id


_GUARDED_GETS = (
    "/",
    "/api/state",
    "/api/jobs",
    "/output.mp4",
    "/shot_stills/000000000.jpg",
)


def test_no_route_answers_without_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_id = _wizard_with_output(tmp_path)
    stored: list[str] = []
    monkeypatch.setattr(
        "super_processor.wizard.store_gemini_api_key", lambda key: stored.append(key)
    )
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    try:
        paths = (
            *_GUARDED_GETS,
            f"/jobs/{job_id}/output.mp4",
            f"/jobs/{job_id}/thumb.jpg",
            # A media signature is no key to the API.
            f"/api/state?t={TOKEN}",
        )
        for path in paths:
            status, _headers, body = _request(base, "GET", path)
            assert status == 401, path
            assert TOKEN.encode() not in body and b"SECRET" not in body, path
            assert b"super-processor review" in body, path
        status, _headers, body = _request(base, "HEAD", "/output.mp4")
        assert status == 401 and body == b""
        # A wrong token, or a cookie for another port, is no token.
        for headers in (
            {"X-Wizard-Token": "guess"},
            {"Cookie": f"{wizard_cookie_name(1)}={TOKEN}"},
            {"Cookie": f"{wizard_cookie_name(_port(base))}=guess"},
        ):
            assert _request(base, "GET", "/api/state", headers)[0] == 401
        status, _headers, body = _request(
            base,
            "POST",
            "/api/setup",
            {"Content-Type": "application/json"},
            json.dumps({"api_key": "attacker"}).encode(),
        )
        assert status == 401 and TOKEN.encode() not in body
        assert stored == []
        # The header or the launch cookie opens every route.
        for headers in ({"X-Wizard-Token": TOKEN}, _cookie(base)):
            status, _headers, body = _request(base, "GET", "/api/state", headers)
            assert status == 200 and json.loads(body)["token"] == TOKEN
            for path in (*_GUARDED_GETS, f"/jobs/{job_id}/output.mp4"):
                assert _request(base, "GET", path, headers)[0] == 200, path
    finally:
        server.stop()


def test_the_launch_url_sets_the_cookie_and_drops_the_token(tmp_path: Path) -> None:
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        assert server.launch_url == f"{base}/?token={TOKEN}"
        assert launch_url(base, "a b") == f"{base}/?token=a%20b"
        status, headers, _body = _request(base, "GET", f"/?token={TOKEN}")
        assert status == 303
        assert headers["Location"] == "/"
        parts = [part.strip() for part in headers["Set-Cookie"].split(";")]
        assert parts[0] == f"{wizard_cookie_name(_port(base))}={TOKEN}"
        assert {"HttpOnly", "SameSite=Strict", "Path=/"} <= set(parts[1:])
        # Other query fields stay; the token never lands in the address.
        status, headers, _body = _request(base, "GET", f"/api/state?a=1&token={TOKEN}")
        assert (status, headers["Location"]) == (303, "/api/state?a=1")
        # Never a redirect to another host.
        for path in (
            f"//evil.example/?token={TOKEN}",
            f"/\\evil.example?token={TOKEN}",
        ):
            status, headers, _body = _request(base, "GET", path)
            assert status == 303, path
            assert headers["Location"].startswith("/"), path
            assert headers["Location"][1:2] not in {"/", "\\"}, path
        assert review._same_path("//evil.example") == "/"
        assert review._same_path("/\\evil.example") == "/"
        assert review._same_path("evil") == "/"
        assert review._same_path("/jobs/a") == "/jobs/a"
        # A wrong token sets nothing.
        status, headers, body = _request(base, "GET", "/?token=guess")
        assert status == 401 and "Set-Cookie" not in headers
        assert TOKEN.encode() not in body
    finally:
        server.stop()


def test_the_cookie_counts_only_on_a_same_origin_request() -> None:
    from email.message import Message

    def headers(**values: str) -> Message:
        message = Message()
        for key, value in values.items():
            message[key.replace("_", "-")] = value
        return message

    cookie = f"other=1; {wizard_cookie_name(9)}={TOKEN}"
    check: dict[str, Any] = {"token": TOKEN, "port": 9}
    assert request_authenticated(headers(Cookie=cookie), **check) == "cookie"
    for site in ("same-origin", "none"):
        sent = headers(Cookie=cookie, Sec_Fetch_Site=site)
        assert request_authenticated(sent, **check) == "cookie"
    # Cookies go to every port of a host: another local page must not ride on it.
    for site in ("same-site", "cross-site"):
        sent = headers(Cookie=cookie, Sec_Fetch_Site=site)
        assert request_authenticated(sent, **check) is None
        assert request_authenticated(sent, **check, media_token=TOKEN) == "media"
    assert request_authenticated(headers(X_Wizard_Token=TOKEN), **check) == "header"
    assert request_authenticated(headers(), **check, media_token="guess") is None


def test_every_response_carries_the_safety_headers(tmp_path: Path) -> None:
    controller, _job_id = _wizard_with_output(tmp_path)
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    auth = {"X-Wizard-Token": TOKEN}
    try:
        for path, headers in (
            ("/", auth),
            ("/api/state", auth),
            ("/output.mp4", auth),
            ("/api/state", {}),
            ("/", {}),
        ):
            _status, sent, _body = _request(base, "GET", path, headers)
            assert sent["X-Content-Type-Options"] == "nosniff", path
            assert sent["X-Frame-Options"] == "DENY", path
            assert "frame-ancestors 'none'" in sent["Content-Security-Policy"], path
            assert sent["Referrer-Policy"] == "same-origin", path
            assert sent["Cache-Control"] == "no-store", path
            assert sent["Cross-Origin-Resource-Policy"] == "same-site", path
    finally:
        server.stop()


def test_media_refuses_a_cross_site_embed_unless_signed(tmp_path: Path) -> None:
    controller, job_id = _wizard_with_output(tmp_path)
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    embed = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Dest": "video"}
    embed["Sec-Fetch-Mode"] = "no-cors"
    embed["Referer"] = "https://evil.example/"
    try:
        for path in (
            "/output.mp4",
            "/shot_stills/000000000.jpg",
            f"/jobs/{job_id}/output.mp4",
        ):
            for method in ("GET", "HEAD"):
                # A cross-site page's <video> sends the cookie (it is not
                # SameSite to a top-level load) but that never counts.
                status, _sent, body = _request(
                    base, method, path, {**embed, **_cookie(base)}
                )
                assert status == 401 and b"SECRET" not in body, (method, path)
            # Only ?t= may load media cross-site, not even the header.
            status, _sent, body = _request(
                base, "GET", path, {**embed, "X-Wizard-Token": TOKEN}
            )
            assert status == 403 and b"SECRET" not in body, path
            # The React shell on another origin signs its media URLs.
            status, sent, body = _request(base, "GET", f"{path}?t={TOKEN}", embed)
            assert status == 200 and body.startswith(b"SECRET"), path
            assert sent["Cross-Origin-Resource-Policy"] == "cross-origin"
            status, sent, _body = _request(
                base, "GET", f"{path}?t={TOKEN}", {**embed, "Range": "bytes=0-1"}
            )
            assert status == 206, path
            assert _request(base, "GET", f"{path}?t=guess", embed)[0] == 401
            # Same origin (the wizard's own pages, the Vite proxy) is fine.
            same = {"Sec-Fetch-Site": "same-origin", **_cookie(base)}
            status, sent, _body = _request(base, "GET", path, same)
            assert status == 200, path
            assert sent["Cross-Origin-Resource-Policy"] == "same-site"
    finally:
        server.stop()


def _legacy_job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "job"
    still = "segment_stills/seg_00.ppm"
    write_gray_still(job_dir / still, bytes([40]) * (160 * 90))
    write_segments(
        job_dir,
        [TimelineSegment(0, 0.0, 60.0, "indoor", "noisy", 10.0, 0, still)],
    )
    return job_dir


def test_the_split_review_page_needs_the_token_and_cannot_be_framed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    applied: list[dict[str, object]] = []
    monkeypatch.setattr(
        review,
        "apply_review_action",
        lambda jobs_dir, job_id, **kwargs: applied.append(kwargs),
    )
    server = ReviewServer(_legacy_job(tmp_path), jobs_dir=tmp_path, job_id="j")
    assert server.controller is None and server.token
    token = server.token
    base = server.start()
    try:
        for path in ("/", "/segments", "/segment_stills/seg_00.bmp"):
            assert _request(base, "GET", path)[0] == 401, path
        assert _request(base, "POST", "/accept", body=b"x=1")[0] == 401
        status, headers, _body = _request(base, "GET", f"/?token={token}")
        assert status == 303
        cookie = {"Cookie": headers["Set-Cookie"].split(";", 1)[0]}
        status, headers, page = _request(base, "GET", "/", cookie)
        assert status == 200
        assert headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert f'name="token" value="{token}"'.encode() in page
        # The cookie comes with a framed click too: the form's token must as well.
        origin = {**cookie, "Origin": base}
        assert _request(base, "POST", "/accept", origin, b"x=1")[0] == 403
        assert applied == []
        body = f"token={token}".encode()
        assert _request(base, "POST", "/accept", origin, body)[0] == 303
        assert applied == [{"note": None, "accept": True}]
    finally:
        server.stop()


def test_reopened_wizard_jobs_need_the_token(tmp_path: Path) -> None:
    controller, job_id = _wizard_with_output(tmp_path)
    job_dir = controller.store.job_dir(job_id)
    save_job_wizard_state(job_dir, WizardState(step=WizardStep.RESULT, job_id=job_id))
    server = ReviewServer(job_dir, jobs_dir=controller.jobs_dir, job_id=job_id)
    base = server.start()
    try:
        for path in ("/api/state", "/output.mp4"):
            assert _request(base, "GET", path)[0] == 401, path
            status = _request(base, "GET", path, {"X-Wizard-Token": server.token or ""})
            assert status[0] == 200, path
    finally:
        server.stop()


def test_an_idle_connection_is_closed_after_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(review, "SOCKET_TIMEOUT_S", 0.3)
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        with socket.create_connection(("127.0.0.1", _port(base))) as idle:
            idle.settimeout(5)
            # Half a request line, then nothing.
            idle.sendall(b"GET / HTTP/1.1\r\n")
            started = time.monotonic()
            assert idle.recv(1024) == b""
            assert time.monotonic() - started < 4
        # A whole request still gets its answer.
        assert _request(base, "GET", "/api/state", {"X-Wizard-Token": TOKEN})[0] == 200
    finally:
        server.stop()


def test_connections_past_the_cap_are_closed_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(review, "MAX_CONNECTIONS", 1)
    monkeypatch.setattr(review, "SOCKET_TIMEOUT_S", 5.0)
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        with socket.create_connection(("127.0.0.1", _port(base))) as held:
            held.sendall(b"GET / HTTP/1.1\r\n")
            time.sleep(0.2)
            with socket.create_connection(("127.0.0.1", _port(base))) as extra:
                extra.settimeout(3)
                extra.sendall(b"GET /api/state HTTP/1.0\r\n\r\n")
                try:
                    reply = extra.recv(1024)
                except ConnectionResetError:
                    reply = b""
                assert reply == b""
        # The slot is free again once the held connection ends.
        deadline = time.monotonic() + 5
        while True:
            try:
                status = _request(base, "GET", "/api/state", {"X-Wizard-Token": TOKEN})
                break
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.05)
        assert status[0] == 200
    finally:
        server.stop()
