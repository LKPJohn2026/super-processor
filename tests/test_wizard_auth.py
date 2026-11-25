"""Every wizard route needs the run's token; the launch URL sets a cookie.

Also the safety headers on every response, the cross-site media refusal and
the socket timeout and connection cap that stop idle connections pinning
threads.
"""

from __future__ import annotations

import json
import os
import socket
import time
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
from wizard_client import FAKE_ASSETS, FAKE_INDEX

from super_processor import review
from super_processor.review import (
    CONTENT_SECURITY_POLICY,
    WizardServer,
    launch_url,
    missing_app_page,
    request_authenticated,
    wizard_cookie_name,
)
from super_processor.wizard import (
    WizardController,
    WizardError,
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
    "/index.html",
    "/assets/index-test.js",
    "/assets/index-test.css",
    "/api/state",
    "/api/jobs",
    "/output.mp4",
    "/shot_stills/000000000.jpg",
)


def test_no_route_answers_without_the_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, web_app: Path
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


def test_a_decoy_cookie_from_another_port_does_not_lock_the_app_out() -> None:
    from email.message import Message

    name = wizard_cookie_name(9)
    check: dict[str, Any] = {"token": TOKEN, "port": 9}
    # Another local page set a same-named cookie with a longer path, so the
    # browser sends it first; the real one still counts.
    message = Message()
    message["Cookie"] = f"{name}=junk; {name}={TOKEN}"
    message["Sec-Fetch-Site"] = "same-origin"
    assert request_authenticated(message, **check) == "cookie"
    only_junk = Message()
    only_junk["Cookie"] = f"{name}=junk; {name}=more-junk"
    assert request_authenticated(only_junk, **check) is None


def test_the_host_header_is_parsed_strictly(tmp_path: Path) -> None:
    from email.message import Message

    from super_processor.review import request_rejection

    def host(*values: str) -> Message:
        message = Message()
        for value in values:
            message["Host"] = value
        return message

    for good in (
        "127.0.0.1:9000",
        "localhost:9000",
        "[::1]:9000",
        "LOCALHOST:9000",
        "127.0.0.1",
    ):
        assert request_rejection(host(good), port=9000, state_changing=True) is None, (
            good
        )
    for bad in (
        "127.0.0.1:9000, evil.test",
        "127.0.0.1:9001",
        "127.0.0.1:",
        "127.0.0.1:9000:1",
        "127.0.0.1:x",
        "127.0.0.1:\u0669",
        "[::1",
        "[::1]9000",
        "[::1]:9000x",
        "::1",
        "evil.test:9000",
        ":9000",
    ):
        assert request_rejection(host(bad), port=9000, state_changing=False) == (
            "unexpected Host header"
        ), bad
    # Two Host headers: refused, whichever comes first.
    twice = host("127.0.0.1:9000", "evil.test")
    assert request_rejection(twice, port=9000, state_changing=False) == (
        "unexpected Host header"
    )
    # Over the wire too.
    server = WizardServer(tmp_path / "jobs", token=TOKEN)
    base = server.start()
    port = _port(base)
    try:
        cookie = {"Cookie": f"{wizard_cookie_name(port)}={TOKEN}"}
        raw = (
            f"GET /api/state HTTP/1.1\r\nHost: 127.0.0.1:{port}, evil.test\r\n"
            f"Cookie: {cookie['Cookie']}\r\nConnection: close\r\n\r\n"
        ).encode()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            sock.sendall(raw)
            reply = b""
            while chunk := sock.recv(65536):
                reply += chunk
        assert reply.startswith(b"HTTP/1.0 403") or reply.startswith(b"HTTP/1.1 403"), (
            reply[:40]
        )
        assert TOKEN.encode() not in reply
    finally:
        server.stop()


def test_every_response_carries_the_safety_headers(
    tmp_path: Path, web_app: Path
) -> None:
    controller, _job_id = _wizard_with_output(tmp_path)
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    auth = {"X-Wizard-Token": TOKEN}
    try:
        for path, headers in (
            ("/", auth),
            ("/assets/index-test.js", auth),
            ("/assets/missing.js", auth),
            ("/api/state", auth),
            ("/output.mp4", auth),
            ("/api/state", {}),
            ("/", {}),
        ):
            _status, sent, _body = _request(base, "GET", path, headers)
            assert sent["X-Content-Type-Options"] == "nosniff", path
            assert sent["X-Frame-Options"] == "DENY", path
            assert sent["Content-Security-Policy"] == CONTENT_SECURITY_POLICY, path
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


def test_the_csp_lets_only_this_server_run_and_load() -> None:
    rules = dict(
        rule.strip().split(" ", 1) for rule in CONTENT_SECURITY_POLICY.split(";")
    )
    assert rules == {
        "default-src": "'none'",
        "script-src": "'self'",
        "style-src": "'self'",
        "font-src": "'self'",
        "img-src": "'self' blob:",
        "media-src": "'self' blob:",
        "connect-src": "'self'",
        "manifest-src": "'self'",
        "object-src": "'none'",
        "base-uri": "'none'",
        "form-action": "'self'",
        "frame-ancestors": "'none'",
    }
    assert "unsafe" not in CONTENT_SECURITY_POLICY
    assert "data:" not in CONTENT_SECURITY_POLICY


def _launched(base: str) -> dict[str, str]:
    """The cookie a browser holds after opening the printed address."""
    status, headers, _body = _request(base, "GET", f"/?token={TOKEN}")
    assert status == 303 and headers["Location"] == "/"
    cookie = headers["Set-Cookie"].split(";", 1)[0]
    return {"Cookie": cookie, "Sec-Fetch-Site": "same-origin"}


def test_the_launch_opens_the_app_which_gets_its_token_from_the_state(
    tmp_path: Path, web_app: Path
) -> None:
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        browser = _launched(base)
        status, headers, page = _request(base, "GET", "/", browser)
        assert status == 200 and page.decode() == FAKE_INDEX
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert TOKEN.encode() not in page
        types = {
            "index-test.js": "text/javascript; charset=utf-8",
            "index-test.css": "text/css; charset=utf-8",
            "font-test.woff2": "font/woff2",
        }
        for name, kind in types.items():
            # A crossorigin module script or stylesheet: same origin, cookie sent.
            sent = {**browser, "Sec-Fetch-Mode": "cors", "Origin": base}
            status, headers, body = _request(base, "GET", f"/assets/{name}", sent)
            assert status == 200, name
            assert headers["Content-Type"] == kind, name
            assert body.decode() == FAKE_ASSETS[name], name
            assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        status, headers, body = _request(base, "HEAD", "/assets/index-test.js", browser)
        assert status == 200 and body == b""
        assert headers["Content-Length"] == str(len(FAKE_ASSETS["index-test.js"]))
        status, _headers, body = _request(base, "HEAD", "/", browser)
        assert status == 200 and body == b""
        # The app's first call: the cookie gets it the token for its header.
        status, _headers, body = _request(base, "GET", "/api/state", browser)
        assert status == 200 and json.loads(body)["token"] == TOKEN
    finally:
        server.stop()


def test_the_cookie_never_opens_the_app_from_another_site_or_port(
    tmp_path: Path, web_app: Path
) -> None:
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        cookie = {"Cookie": _launched(base)["Cookie"]}
        # Another localhost port is same-site; a page elsewhere is cross-site.
        for site in ("same-site", "cross-site"):
            sent = {**cookie, "Sec-Fetch-Site": site}
            for path in ("/", "/assets/index-test.js", "/api/state"):
                status, _headers, body = _request(base, "GET", path, sent)
                assert status == 401, (site, path)
                assert TOKEN.encode() not in body
                assert b"in a new tab" in body
    finally:
        server.stop()


def test_only_built_app_files_are_served(tmp_path: Path, web_app: Path) -> None:
    (web_app / "secret.txt").write_text("SECRET", encoding="utf-8")
    (web_app / "assets" / "notes.txt").write_text("SECRET", encoding="utf-8")
    (web_app / "assets" / "sub").mkdir()
    (web_app / "assets" / "sub" / "x.js").write_text("SECRET", encoding="utf-8")
    outside = tmp_path / "outside.js"
    outside.write_text("SECRET", encoding="utf-8")
    linked = False
    try:
        os.symlink(outside, web_app / "assets" / "link.js")
        linked = True
    except (OSError, NotImplementedError):
        pass  # Windows without the symlink privilege.
    server = WizardServer(tmp_path / "jobs", token=TOKEN)
    base = server.start()
    auth = {"X-Wizard-Token": TOKEN}
    try:
        for path in (
            "/secret.txt",
            "/assets/notes.txt",
            "/assets/sub/x.js",
            "/assets/../secret.txt",
            "/assets/%2e%2e/secret.txt",
            "/assets/..%2fsecret.txt",
            "/assets/.hidden.js",
            "/assets/",
            "/assets",
            "/assets/missing.js",
            "/assets/link.js",
            "/index.htm",
            # Windows path tricks: backslash, alternate stream, trailing dot.
            "/assets/..\\secret.txt",
            "/assets/index-test.js::$DATA",
            "/assets/index-test.js.",
            "/assets/index-test.js%00.png",
        ):
            for method in ("GET", "HEAD"):
                status, _headers, body = _request(base, method, path, auth)
                assert status == 404, (method, path)
                assert b"SECRET" not in body, path
        assert review.app_asset("/assets/link.js") is None
        for trick in (
            "/assets/..\\secret.txt",
            "/assets/index-test.js::$DATA",
            "/assets/index-test.js.",
            "/assets/c:secret.js",
        ):
            assert review.app_asset(trick) is None, trick
        # Served only on a case-blind file system, and then from assets/.
        upper = review.app_asset("/assets/INDEX-TEST.JS")
        if upper is not None:
            assert upper[0].parent == web_app / "assets"
            assert upper[1].startswith("text/javascript")
        assert review.app_asset("/assets/index-test.js") is not None
        if linked:
            assert (web_app / "assets" / "link.js").is_symlink()
    finally:
        server.stop()


def test_without_a_built_app_the_page_says_how_to_get_one(
    tmp_path: Path, no_web_app: Path
) -> None:
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        assert _request(base, "GET", "/")[0] == 401
        browser = _launched(base)
        status, headers, page = _request(base, "GET", "/", browser)
        assert status == 200
        assert headers["Content-Type"] == "text/html; charset=utf-8"
        assert headers["Content-Security-Policy"] == CONTENT_SECURITY_POLICY
        text = page.decode()
        assert "npm run build:package" in text
        assert f"VITE_PROXY_TARGET=http://127.0.0.1:{_port(base)}" in text
        assert "http://localhost:5173/#token=" in text
        # Plain and static: no script or style, and no token.
        assert "<script" not in text and "<style" not in text and "style=" not in text
        assert TOKEN not in text
        assert _request(base, "GET", "/assets/index-test.js", browser)[0] == 404
        status, _headers, body = _request(base, "HEAD", "/", browser)
        assert status == 200 and body == b""
        # The API still works for the Vite dev server.
        status, _headers, body = _request(base, "GET", "/api/state", browser)
        assert status == 200 and json.loads(body)["step"] == "intro"
    finally:
        server.stop()
    assert "&lt;x&gt;" in missing_app_page("<x>")


def test_review_of_a_job_needs_the_token_and_cannot_be_framed(
    tmp_path: Path, web_app: Path
) -> None:
    controller, job_id = _wizard_with_output(tmp_path)
    job_dir = controller.store.job_dir(job_id)
    save_job_wizard_state(job_dir, WizardState(step=WizardStep.RESULT, job_id=job_id))
    server = WizardServer(controller.jobs_dir, job_id=job_id, token=TOKEN)
    base = server.start()
    try:
        for path in ("/", "/assets/index-test.js", "/api/state", "/output.mp4"):
            assert _request(base, "GET", path)[0] == 401, path
        assert _request(base, "POST", "/api/intro", body=b"{}")[0] == 401
        browser = _launched(base)
        status, headers, page = _request(base, "GET", "/", browser)
        assert status == 200 and page.decode() == FAKE_INDEX
        assert headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        # The app opens on that job.
        status, _headers, body = _request(base, "GET", "/api/state", browser)
        view = json.loads(body)
        assert (view["job_id"], view["step"]) == (job_id, "result")
        # The cookie comes with a framed click too: it never changes state.
        origin = {**browser, "Origin": base, "Content-Type": "application/json"}
        assert _request(base, "POST", "/api/new", origin, b"{}")[0] == 403
        assert server.controller.current_state().step is WizardStep.RESULT
    finally:
        server.stop()


def test_review_of_a_job_the_wizard_did_not_make_is_refused(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    source = tmp_path / "cli.mp4"
    source.write_bytes(b"x")
    job_id = controller.store.create(source).job_id
    with pytest.raises(WizardError, match="no wizard state"):
        WizardServer(controller.jobs_dir, job_id=job_id)


def test_reopened_wizard_jobs_need_the_token(tmp_path: Path) -> None:
    controller, job_id = _wizard_with_output(tmp_path)
    job_dir = controller.store.job_dir(job_id)
    save_job_wizard_state(job_dir, WizardState(step=WizardStep.RESULT, job_id=job_id))
    server = WizardServer(controller.jobs_dir, job_id=job_id)
    base = server.start()
    try:
        for path in ("/api/state", "/output.mp4"):
            assert _request(base, "GET", path)[0] == 401, path
            status = _request(base, "GET", path, {"X-Wizard-Token": server.token})
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
