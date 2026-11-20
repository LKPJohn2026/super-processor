"""JSON wizard API used by the static UI shell."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest

from super_processor.gemini import GeminiClient
from super_processor.review import (
    MAX_UPLOAD_BYTES,
    WizardServer,
    form_fields_from_body,
)
from super_processor.upscale import FakeUpscaleEngine
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    save_session_state,
)

TOKEN = "test-token"


def test_json_body_parser_rejects_non_objects() -> None:
    assert form_fields_from_body(b"", "application/json") == {}
    assert form_fields_from_body(b"choice=gemini", "application/x-www-form-urlencoded")[
        "choice"
    ] == ["gemini"]
    parsed = form_fields_from_body(
        b'{"skip": true, "api_key": null, "note": "keep"}',
        "application/json",
    )
    assert parsed == {"skip": ["1"], "note": ["keep"]}
    with pytest.raises(WizardError, match="invalid JSON"):
        form_fields_from_body(b"{", "application/json")
    with pytest.raises(WizardError, match="object"):
        form_fields_from_body(b"[1]", "application/json")


def _connection(base: str) -> HTTPConnection:
    parsed = urlparse(base)
    return HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)


def _json_request(
    base: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    origin: str | None = None,
) -> tuple[int, dict[str, Any], str | None]:
    connection = _connection(base)
    body = b"" if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json", "X-Wizard-Token": TOKEN}
    if payload is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    if origin:
        headers["Origin"] = origin
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    allow = response.getheader("Access-Control-Allow-Origin")
    data: dict[str, Any] = {}
    if raw:
        loaded = json.loads(raw.decode())
        assert isinstance(loaded, dict)
        data = loaded
    return response.status, data, allow


def test_api_state_and_cors_before_a_job(tmp_path: Path) -> None:
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        status, payload, allow = _json_request(
            base, "GET", "/api/state", origin="http://127.0.0.1:5173"
        )
        assert status == 200
        assert payload["step"] == "intro"
        assert payload["output_ready"] is False
        assert payload["run"] is None
        assert allow == "http://127.0.0.1:5173"
        connection = _connection(base)
        connection.request(
            "OPTIONS",
            "/api/state",
            headers={"Origin": "http://localhost:5173"},
        )
        options = connection.getresponse()
        assert options.status == 204
        options.read()
        missing = _connection(base)
        missing.request("OPTIONS", "/")
        assert missing.getresponse().status == 404
        bad, error, _origin = _json_request(base, "POST", "/api/llm", {"nope": 1})
        assert bad == 200
        assert error["step"] == "llm_choice"
        assert error["error"]
        invalid, body, _origin = _json_request(base, "POST", "/api/intro", None)
        assert invalid == 200
        assert body["step"] == "llm_choice"
        broken = _connection(base)
        raw = b"{"
        broken.request(
            "POST",
            "/api/setup",
            body=raw,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(raw)),
            },
        )
        bad_json = broken.getresponse()
        assert bad_json.status == 400
        assert "invalid JSON" in bad_json.read().decode()
    finally:
        server.stop()


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "https://evil.example",
        "null",
        None,
    ],
)
def test_upload_preflight(tmp_path: Path, origin: str | None) -> None:
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    connection = _connection(base)
    headers = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": ("content-type,x-wizard-token,x-file-name"),
    }
    if origin is not None:
        headers["Origin"] = origin
    try:
        connection.request("OPTIONS", "/api/upload", headers=headers)
        response = connection.getresponse()
        response.read()
        if origin is None or origin in {"https://evil.example", "null"}:
            assert response.status == 403
            assert response.getheader("Access-Control-Allow-Origin") is None
            assert response.getheader("Access-Control-Allow-Headers") is None
        else:
            assert response.status == 204
            assert response.getheader("Access-Control-Allow-Origin") == origin
            assert response.getheader("Vary") == "Origin"
            allowed = {
                header.strip().lower()
                for header in (
                    response.getheader("Access-Control-Allow-Headers") or ""
                ).split(",")
            }
            assert {"content-type", "x-wizard-token", "x-file-name"} <= allowed
            assert "POST" in (response.getheader("Access-Control-Allow-Methods") or "")
    finally:
        connection.close()
        server.stop()


def _tiny_clip(path: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:duration=8:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_json_api_drives_upscale_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "ci-api-key")
    clip = _tiny_clip(tmp_path / "clip.mp4")
    controller = WizardController(
        tmp_path,
        gemini=GeminiClient(
            api_key="ci-api-key",
            transport=_RangeTransport(),
        ),
        upscale_engine=FakeUpscaleEngine(),
    )
    server = WizardServer(tmp_path, token=TOKEN, controller=controller)
    base = server.start()
    try:
        status, intro, _origin = _json_request(base, "POST", "/api/intro", {})
        assert status == 200
        assert intro["step"] == "llm_choice"
        assert intro["has_gemini_key"] is True
        _status, setup, _origin = _json_request(
            base, "POST", "/api/llm", {"choice": "gemini"}
        )
        assert setup["step"] == "setup"
        _status, pick, _origin = _json_request(
            base, "POST", "/api/setup", {"skip": True}
        )
        assert pick["step"] == "pick_file"
        _status, waiting, _origin = _json_request(
            base, "POST", "/api/pick", {"path": str(clip)}
        )
        assert waiting["step"] == "sampling"
        _json_request(base, "GET", "/api/segment?run=1")
        assert controller.wait_idle(timeout=60)
        _status, found, _origin = _json_request(base, "GET", "/api/state")
        assert found["step"] == "segments"
        # This transport only knows upscale params, so the whole clip is one.
        assert len(found["segments"]["segments"]) == 1
        assert "could not propose segments" in found["segments"]["error"]
        _status, approved, _origin = _json_request(
            base, "POST", "/api/segments", {"action": "approve"}
        )
        assert approved["step"] == "analyzing"
        _json_request(base, "GET", "/api/analyze?run=1")
        assert controller.wait_idle(timeout=60)
        _status, looks, _origin = _json_request(base, "GET", "/api/state")
        assert looks["step"] == "preview_segment"
        view = looks["segment_preview"]
        shot = view["shot"]
        assert (view["index"], view["count"], view["scale"]) == (0, 1, 2)
        # This transport cannot plan either, so the measurements set the look.
        assert "could not propose" in view["error"]
        assert shot["check"]["ok"] is None
        with urlopen(base + shot["after_url"]) as response:
            assert response.headers.get_content_type() == "video/mp4"
        with urlopen(base + shot["before_url"]) as response:
            assert response.headers.get_content_type() == "video/mp4"
        for bad in (
            "/previews/work/shot_000_in.mp4",
            "/previews/p.mp4",
            "/shot_stills/../shots.json",
            "/shot_stills/1.jpg",
        ):
            with pytest.raises(HTTPError) as missing:
                urlopen(base + bad)
            assert missing.value.code == 404
            missing.value.close()
        _status, ready, _origin = _json_request(
            base, "POST", "/api/preview", {"action": "approve"}
        )
        assert ready["step"] == "ready_to_render"
        assert len(ready["looks"]["shots"]) == 1
        _status, approved, _origin = _json_request(
            base, "POST", "/api/ready", {"action": "render"}
        )
        assert approved["step"] == "rendering"
        _status, started, _origin = _json_request(base, "GET", "/api/render?run=1")
        assert started["step"] == "rendering"
        assert controller.wait_idle(timeout=60)
        _status, result, _origin = _json_request(base, "GET", "/api/state")
        assert result["step"] == "result"
        assert result["busy"] is False
        assert result["output_ready"] is True
        assert result["output_url"].startswith("/output.mp4?v=")
        assert result["scale"] == 2
        assert result["strength"] == 0.5
        assert result["strength_summary"] == "strength 0.50"
        with urlopen(base + "/output.mp4") as response:
            assert response.headers.get_content_type() == "video/mp4"
        _status, again, _origin = _json_request(
            base,
            "POST",
            "/api/result",
            {"note": "reduce artificial detail from 1s to 3s"},
        )
        assert again["step"] == "rendering"
        _json_request(base, "GET", "/api/render?run=1")
        assert controller.wait_idle(timeout=60)
        _status, revised, _origin = _json_request(base, "GET", "/api/state")
        assert revised["step"] == "result"
        assert revised["strength"] == 0.2
        _status, done, _origin = _json_request(
            base, "POST", "/api/result", {"mood": "happy"}
        )
        assert done["step"] == "done"
        assert done["output_ready"] is True
        assert controller.current_state().step is WizardStep.DONE
        page = urlopen(base + "/")
        html = page.read().decode()
        assert "<h1>Done</h1>" in html
        unknown, err, _origin = _json_request(base, "POST", "/api/nope", {})
        assert unknown == 400
        assert "unknown action" in err["error"]
    finally:
        server.stop()


class _RangeTransport:
    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {"start_s": 1, "end_s": 3, "scale": 2, "strength": 0.2}
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


def test_html_post_still_redirects(tmp_path: Path) -> None:
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        request = Request(
            base + "/intro", data=b"", headers={"X-Wizard-Token": TOKEN}, method="POST"
        )
        with urlopen(request) as response:
            html = response.read().decode()
        assert "What type of LLM" in html
    finally:
        server.stop()


def test_run_keeps_its_id_across_polls(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )
    entered = threading.Event()
    release = threading.Event()

    def _slow_render() -> None:
        entered.set()
        release.wait(5)
        save_session_state(tmp_path, WizardState(step=WizardStep.PICK_FILE))

    controller._run_render = _slow_render  # type: ignore[method-assign]
    server = WizardServer(tmp_path, token=TOKEN, controller=controller)
    base = server.start()
    try:
        status, started, _origin = _json_request(base, "GET", "/api/render?run=1")
        assert status == 200
        assert entered.wait(5)
        _status, first, _origin = _json_request(base, "GET", "/api/state")
        _status, second, _origin = _json_request(base, "GET", "/api/state")
        assert first["busy"] is True
        assert first["run"]["outcome"] == "running"
        assert first["run"]["kind"] == "render"
        assert first["run"]["job_id"] == "job00001"
        assert first["run"]["finished_at"] is None
        assert second["run"]["id"] == first["run"]["id"]
        release.set()
        assert controller.wait_idle(timeout=5)
        _status, after, _origin = _json_request(base, "GET", "/api/state")
        assert after["busy"] is False
        assert after["run"]["id"] == first["run"]["id"]
        assert after["run"]["outcome"] == "done"
        assert after["run"]["finished_at"] >= after["run"]["started_at"]
    finally:
        release.set()
        server.stop()


def test_a_post_for_a_job_no_longer_open_is_409(tmp_path: Path) -> None:
    (tmp_path / "job00002").mkdir()
    save_session_state(tmp_path, WizardState(step=WizardStep.INTRO, job_id="job00002"))
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        status, body, _origin = _json_request(
            base, "POST", "/api/intro", {"job_id": "job00001"}
        )
        assert status == 409
        assert body["error"] == "this job is no longer open in this window"
        status, body, _origin = _json_request(
            base, "POST", "/api/intro", {"job_id": "job00002"}
        )
        assert status == 200 and body["step"] == "llm_choice"
        html = Request(
            base + "/intro",
            data=b"job_id=job00001",
            headers={
                "X-Wizard-Token": TOKEN,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as refused:
            urlopen(html)
        assert refused.value.code == 409
        refused.value.close()
    finally:
        server.stop()


def _upload(
    base: str, headers: dict[str, str], body: bytes = b""
) -> tuple[int, dict[str, Any]]:
    connection = _connection(base)
    connection.request("POST", "/api/upload", body=body, headers=headers)
    response = connection.getresponse()
    return response.status, json.loads(response.read().decode())


def test_upload_refusals_save_nothing(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, token=TOKEN, controller=controller)
    base = server.start()
    named = {"X-Wizard-Token": TOKEN, "X-File-Name": "clip.mp4"}
    uploads = tmp_path / "uploads"
    try:
        status, data = _upload(base, {**named, "Content-Length": "0"})
        assert (status, data["error"]) == (411, "the upload needs a Content-Length")
        # Refused from the header alone: the body is never read.
        too_big = {**named, "Content-Length": str(MAX_UPLOAD_BYTES + 1)}
        status, data = _upload(base, too_big)
        assert (status, data["error"]) == (413, "the file is too large to upload")
        # The token must be a header: a dropped file has no form to carry it.
        status, _data = _upload(base, {"X-File-Name": "clip.mp4"}, b"x")
        assert status == 403
        with controller._work_lock:
            status, data = _upload(base, named, b"x")
        assert (status, data["error"]) == (409, "the last step is still running")

        # A connection that ends early leaves no half file behind.
        address = _connection(base)
        with socket.create_connection((address.host, address.port)) as raw:
            raw.sendall(
                b"POST /api/upload HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                + f"X-Wizard-Token: {TOKEN}\r\n".encode()
                + b"X-File-Name: clip.mp4\r\n"
                b"Content-Length: 1000\r\n\r\n"
                b"only ten b"
            )
            raw.shutdown(socket.SHUT_WR)
            reply = b""
            while chunk := raw.recv(65536):
                reply += chunk
        assert reply.split(b"\r\n", 1)[0].endswith(b" 400 Bad Request")
        assert b"the upload stopped before the end" in reply
        assert not uploads.exists() or list(uploads.iterdir()) == []
        assert controller.current_state().step is WizardStep.INTRO
    finally:
        server.stop()
