"""Hardening: CSRF surface, run token, body size, pick/symlink, stills."""

from __future__ import annotations

import json
import shutil
import subprocess
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from super_processor.gemini import GeminiClient, GeminiError, _jpeg_part
from super_processor.review import (
    ALLOWED_ORIGINS_ENV,
    DEFAULT_ALLOWED_ORIGINS,
    MAX_BODY_BYTES,
    WizardServer,
    allowed_origins,
    with_wizard_token,
)
from super_processor.wizard import WizardController, WizardStep

TOKEN = "test-token"


def _request(
    base: str,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> tuple[int, bytes]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    sent = dict(headers or {})
    if body or "Content-Length" in sent:
        sent.setdefault("Content-Length", str(len(body)))
    connection.request(method, path, body=body, headers=sent)
    response = connection.getresponse()
    data = response.read()
    return response.status, data


def test_pages_origin_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "https://lkpjohn2026.github.io" not in DEFAULT_ALLOWED_ORIGINS
    assert "https://lkpjohn2026.github.io" not in allowed_origins(9000)
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://lkpjohn2026.github.io")
    assert "https://lkpjohn2026.github.io" in allowed_origins(9000)


def test_env_origins_reject_wildcards_and_junk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        ALLOWED_ORIGINS_ENV,
        "*, null, file://tmp, https://user:pass@evil.example, "
        "https://evil.example/path, not-a-url",
    )
    allowed = allowed_origins(9000)
    assert "*" not in allowed
    assert "null" not in allowed
    assert "file://tmp" not in allowed
    assert "https://evil.example/path" not in allowed
    assert "https://user:pass@evil.example" not in allowed


def test_pages_cannot_csrf_setup_by_default(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        status, body = _request(
            base,
            "POST",
            "/api/setup",
            headers={
                "Origin": "https://lkpjohn2026.github.io",
                "Content-Type": "application/json",
            },
            body=json.dumps({"api_key": "attacker-key"}).encode(),
        )
        assert status == 403
        assert b"origin not allowed" in body
        assert controller.current_state().step is WizardStep.INTRO
    finally:
        server.stop()


def test_oversized_post_is_rejected_without_reading(
    tmp_path: Path,
) -> None:
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, controller=controller, token=TOKEN)
    base = server.start()
    try:
        status, body = _request(
            base,
            "POST",
            "/api/setup",
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(MAX_BODY_BYTES + 1),
            },
            body=b"{}",  # smaller than the claimed length; must not hang
        )
        assert status == 413
        assert b"body too large" in body
        assert controller.current_state().step is WizardStep.INTRO
    finally:
        server.stop()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_pick_follows_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Footage linked in from another drive is normal; the link is followed.
    monkeypatch.setattr("super_processor.wizard.resolve_gemini_api_key", lambda: "k")
    real = tmp_path / "footage" / "clip.mp4"
    real.parent.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x36:rate=10:duration=1",
            "-pix_fmt",
            "yuv420p",
            str(real),
        ],
        check=True,
    )
    link = tmp_path / "jobs" / "clip.mp4"
    link.parent.mkdir()
    link.symlink_to(real)
    controller = WizardController(tmp_path / "jobs")
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    controller.handle_post("/setup", {"skip": ["1"]})
    controller.handle_post("/pick", {"path": [str(link)]})
    state = controller.current_state()
    assert state.error is None
    assert state.step is WizardStep.FINDING_SHOTS


def test_bad_content_length_is_400(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, controller=controller, token=TOKEN)
    base = server.start()
    try:
        for value in ("abc", "-1"):
            status, body = _request(
                base,
                "POST",
                "/api/intro",
                headers={
                    "Content-Type": "application/json",
                    "Content-Length": value,
                    "X-Wizard-Token": TOKEN,
                },
            )
            assert status == 400
            assert b"bad Content-Length" in body
        assert controller.current_state().step is WizardStep.INTRO
    finally:
        server.stop()


def test_state_changes_need_the_run_token(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, controller=controller, token=TOKEN)
    base = server.start()
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        # No Origin and no Sec-Fetch-Site: only the token stops these.
        assert _request(base, "POST", "/intro", headers=form, body=b"x=1")[0] == 403
        wrong = {**form, "X-Wizard-Token": "guess"}
        assert _request(base, "POST", "/intro", headers=wrong, body=b"x=1")[0] == 403
        status, body = _request(base, "GET", "/render?run=1")
        assert status == 403 and b"wizard token" in body
        assert controller.current_state().step is WizardStep.INTRO

        # The page embeds the token in its forms; a form field is enough.
        status, page = _request(base, "GET", "/")
        assert status == 200
        connection = HTTPConnection("127.0.0.1", urlparse(base).port)
        connection.request("GET", "/")
        framed = connection.getresponse()
        framed.read()
        assert framed.getheader("X-Frame-Options") == "DENY"
        assert f'name="token" value="{TOKEN}"'.encode() in page
        field = f"token={TOKEN}".encode()
        assert _request(base, "POST", "/intro", headers=form, body=field)[0] == 303
        assert controller.current_state().step is WizardStep.LLM_CHOICE

        # The API hands the token to the shell, which sends it as a header.
        status, body = _request(base, "GET", "/api/state")
        assert json.loads(body)["token"] == TOKEN
        status, _body = _request(
            base,
            "POST",
            "/api/llm",
            headers={"Content-Type": "application/json", "X-Wizard-Token": TOKEN},
            body=json.dumps({"choice": "gemini"}).encode(),
        )
        assert status == 200
        assert controller.current_state().step is WizardStep.SETUP
    finally:
        server.stop()


def test_pages_carry_the_token_into_run_links() -> None:
    page = (
        '<meta http-equiv="refresh" content="2;url=/render?run=1">'
        '<form method="post" action="/looks"><button>Go</button></form>'
        '<form method="get" action="/"></form>'
    )
    marked = with_wizard_token(page, "abc")
    assert "url=/render?run=1&amp;token=abc" in marked
    assert marked.count('name="token" value="abc"') == 1
    assert '<form method="get" action="/"></form>' in marked


def test_each_server_makes_its_own_token(tmp_path: Path) -> None:
    first = WizardServer(tmp_path)
    second = WizardServer(tmp_path)
    first.start()
    second.start()
    try:
        assert first.token and second.token and first.token != second.token
        assert len(first.token) >= 32
    finally:
        first.stop()
        second.stop()


def test_jpeg_part_refuses_still_outside_job(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    inside = job / "shot.jpg"
    inside.write_bytes(b"\xff\xd8\xff")
    outside = tmp_path / "leak.jpg"
    outside.write_bytes(b"\xff\xd8secret")
    part = _jpeg_part(inside, under=job)
    assert part["inlineData"]["mimeType"] == "image/jpeg"
    with pytest.raises(GeminiError, match="outside the job"):
        _jpeg_part(outside, under=job)


def test_label_shots_refuses_outside_still(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    outside = tmp_path / "leak.jpg"
    outside.write_bytes(b"\xff\xd8secret")

    class _Boom:
        def generate(self, **_kwargs: object) -> dict[str, object]:
            raise AssertionError("must not call Gemini")

    client = GeminiClient(api_key="k", transport=_Boom())
    with pytest.raises(GeminiError, match="outside the job"):
        client.label_shots(
            shots=[{"start_s": 0, "end_s": 1, "metrics": {}, "hints": []}],
            stills=[outside],
            issues=("soft",),
            contains=("faces",),
            job_dir=job,
        )
