"""JSON wizard API used by the static UI shell."""

from __future__ import annotations

import json
import shutil
import subprocess
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest

from super_processor.gemini import GeminiClient
from super_processor.review import WizardServer, form_fields_from_body
from super_processor.upscale import FakeUpscaleEngine
from super_processor.wizard import WizardController, WizardError, WizardStep


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
    headers = {"Accept": "application/json"}
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
    server = WizardServer(tmp_path)
    base = server.start()
    try:
        status, payload, allow = _json_request(
            base, "GET", "/api/state", origin="http://127.0.0.1:5173"
        )
        assert status == 200
        assert payload["step"] == "intro"
        assert payload["output_ready"] is False
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
    server = WizardServer(tmp_path, controller=controller)
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
        assert waiting["step"] == "rendering"
        _status, result, _origin = _json_request(base, "GET", "/api/render?run=1")
        assert result["step"] == "result"
        assert result["output_ready"] is True
        assert result["output_url"] == "/output.mp4"
        assert result["scale"] == 2
        assert result["strength"] == 0.5
        with urlopen(base + "/output.mp4") as response:
            assert response.headers.get_content_type() == "video/mp4"
        _status, again, _origin = _json_request(
            base,
            "POST",
            "/api/result",
            {"note": "reduce artificial detail from 1s to 3s"},
        )
        assert again["step"] == "rendering"
        _status, revised, _origin = _json_request(base, "GET", "/api/render?run=1")
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
    server = WizardServer(tmp_path)
    base = server.start()
    try:
        request = Request(base + "/intro", data=b"", method="POST")
        with urlopen(request) as response:
            html = response.read().decode()
        assert "What type of LLM" in html
    finally:
        server.stop()
