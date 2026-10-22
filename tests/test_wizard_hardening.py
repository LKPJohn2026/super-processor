"""Hardening: CSRF surface, body size, pick/symlink, stills, key replace."""

from __future__ import annotations

import json
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
)
from super_processor.wizard import WizardController, WizardStep


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
    server = WizardServer(tmp_path, controller=controller)
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


def test_pick_refuses_a_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("super_processor.wizard.resolve_gemini_api_key", lambda: "k")
    real = tmp_path / "secret.mp4"
    real.write_bytes(b"not-really-video")
    link = tmp_path / "jobs" / "clip.mp4"
    link.parent.mkdir()
    link.symlink_to(real)
    controller = WizardController(tmp_path / "jobs")
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    controller.handle_post("/setup", {"skip": ["1"]})
    controller.handle_post("/pick", {"path": [str(link)]})
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error is not None and "symbolic link" in state.error


def test_setup_needs_replace_to_overwrite_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    stored: dict[str, str] = {}

    def fake_resolve() -> str | None:
        return stored.get("key")

    def fake_store(key: str) -> None:
        stored["key"] = key.strip()

    monkeypatch.setattr("super_processor.wizard.resolve_gemini_api_key", fake_resolve)
    monkeypatch.setattr("super_processor.wizard.store_gemini_api_key", fake_store)
    controller = WizardController(tmp_path)
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    controller.handle_post("/setup", {"api_key": ["first-key"]})
    assert controller.current_state().step is WizardStep.PICK_FILE
    assert stored["key"] == "first-key"

    # Key exists: overwrite without replace=1 must stay on setup.
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    controller.handle_post("/setup", {"api_key": ["attacker-key"]})
    state = controller.current_state()
    assert state.step is WizardStep.SETUP
    assert state.error is not None and "replace=1" in state.error
    assert stored["key"] == "first-key"

    controller.handle_post(
        "/setup", {"api_key": ["second-key"], "replace": ["1"]}
    )
    assert controller.current_state().step is WizardStep.PICK_FILE
    assert stored["key"] == "second-key"


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

    client = GeminiClient(api_key="k", transport=_Boom())  # type: ignore[arg-type]
    with pytest.raises(GeminiError, match="outside the job"):
        client.label_shots(
            shots=[{"start_s": 0, "end_s": 1, "metrics": {}, "hints": []}],
            stills=[outside],
            issues=("soft",),
            contains=("faces",),
            job_dir=job,
        )
