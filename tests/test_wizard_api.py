"""JSON wizard API used by the static UI shell."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
from dataclasses import replace
from http.client import HTTPConnection
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest
from wizard_client import get

from super_processor.gemini import GeminiClient
from super_processor.review import (
    MAX_UPLOAD_BYTES,
    WizardServer,
    form_fields_from_body,
)
from super_processor.segment_loop import LoopSegment, SegmentList, save_segment_list
from super_processor.shot_plan import finish, new_plan, save_plan
from super_processor.shots import Shot, ShotList, save_shots
from super_processor.upscale import FakeUpscaleEngine
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    save_job_wizard_state,
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
                "X-Wizard-Token": TOKEN,
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
        with get(base + shot["after_url"], TOKEN) as response:
            assert response.headers.get_content_type() == "video/mp4"
        with get(base + shot["before_url"], TOKEN) as response:
            assert response.headers.get_content_type() == "video/mp4"
        for bad in (
            "/previews/work/shot_000_in.mp4",
            "/previews/p.mp4",
            "/shot_stills/../shots.json",
            "/shot_stills/1.jpg",
        ):
            with pytest.raises(HTTPError) as missing:
                get(base + bad, TOKEN)
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
        job_id = result["job_id"]
        assert result["output_url"].startswith(f"/jobs/{job_id}/output.mp4?v=")
        assert result["scale"] == 2
        assert result["strength"] == 0.5
        assert result["strength_summary"] == "strength 0.50"
        with get(base + "/output.mp4", TOKEN) as response:
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
        unknown, err, _origin = _json_request(base, "POST", "/api/nope", {})
        assert unknown == 404
        assert "unknown endpoint /api/nope" in err["error"]
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
        # Left on the render by an earlier server: it waits to be resumed.
        _status, waiting, _origin = _json_request(base, "GET", "/api/render?run=1")
        assert waiting["interrupted"] is True and waiting["busy"] is False
        assert not entered.is_set()
        _json_request(base, "POST", "/api/resume", {})
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
        # A form-encoded body is read the same way.
        html = Request(
            base + "/api/intro",
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
        # Without it the upload is refused before a byte of it is read.
        status, _data = _upload(base, {"X-File-Name": "clip.mp4"}, b"x")
        assert status == 401
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


def test_a_refused_post_reads_its_body_before_closing(tmp_path: Path) -> None:
    # Closing with the body unread would reset the connection on Windows,
    # and the reset can wipe the refusal out before the client reads it.
    server = WizardServer(tmp_path, token=TOKEN, controller=WizardController(tmp_path))
    base = server.start()
    try:
        address = _connection(base)
        with socket.create_connection((address.host, address.port)) as raw:
            raw.settimeout(5)
            raw.sendall(
                b"POST /api/upload HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"X-Wizard-Token: wrong\r\n"
                b"X-File-Name: clip.mp4\r\n"
                b"Content-Length: 20\r\n\r\n"
                b"first half"
            )
            reply = b""
            while not reply.endswith(b"}"):
                chunk = raw.recv(65536)
                assert chunk, f"closed before the reply ended: {reply!r}"
                reply += chunk
            assert reply.split(b"\r\n", 1)[0].endswith(b" 401 Unauthorized")
            # The server is still reading the body, so the connection stays.
            raw.settimeout(0.3)
            with pytest.raises(TimeoutError):
                raw.recv(1)
            raw.settimeout(5)
            raw.sendall(b"other half")
            assert raw.recv(1) == b""  # then a clean close, not a reset
    finally:
        server.stop()


def test_a_stalled_refused_body_is_dropped_at_the_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from super_processor import review

    monkeypatch.setattr(review, "_DISCARD_TIMEOUT_S", 0.2)
    server = WizardServer(tmp_path, token=TOKEN, controller=WizardController(tmp_path))
    base = server.start()
    try:
        address = _connection(base)
        with socket.create_connection((address.host, address.port)) as raw:
            raw.settimeout(5)
            raw.sendall(
                b"POST /api/upload HTTP/1.1\r\n"
                b"Host: 127.0.0.1\r\n"
                b"X-Wizard-Token: wrong\r\n"
                b"Content-Length: 1000\r\n\r\n"
                b"a trickle"
            )
            reply = b""
            # The rest never comes: the server gives up and closes anyway.
            while chunk := raw.recv(65536):
                reply += chunk
            assert reply.split(b"\r\n", 1)[0].endswith(b" 401 Unauthorized")
    finally:
        server.stop()


def _job_on(tmp_path: Path, step: WizardStep, **state: Any) -> tuple[str, Path]:
    """A job with two approved, previewed scenes and the wizard on ``step``.

    The source is stand-in bytes, so this needs no FFmpeg.
    """
    source = tmp_path / "0123456789ab-My_Trailer.mp4"
    source.write_bytes(b"not-a-real-video")
    controller = WizardController(tmp_path)
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    save_segment_list(
        job_dir,
        SegmentList(
            segments=[LoopSegment(0.0, 3.0, "a"), LoopSegment(3.0, 6.0, "b")],
            approved=True,
        ),
    )
    shots = ShotList([Shot(0.0, 3.0, label="a"), Shot(3.0, 6.0, label="b")])
    save_shots(job_dir, shots)
    plan = finish(new_plan(shots, 2), [0, 1])
    plan.recipes = [replace(recipe, rev=1) for recipe in plan.recipes]
    save_plan(job_dir, plan)
    saved = WizardState(step=step, job_id=job_id, **state)
    save_job_wizard_state(job_dir, saved)
    save_session_state(tmp_path, saved)
    return job_id, job_dir


def _stages(job: dict[str, Any]) -> dict[str, str]:
    return {stage["key"]: stage["status"] for stage in job["stages"]}


def test_the_job_summary_drives_the_stepper_and_scene_strip(tmp_path: Path) -> None:
    job_id, _job_dir = _job_on(
        tmp_path, WizardStep.PREVIEW_SEGMENT, segment_index=1, approved_count=1
    )
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        status, view, _origin = _json_request(base, "GET", "/api/state")
        assert status == 200
        job = view["job"]
        assert set(job) == {
            "job_id",
            "title",
            "stage",
            "stages",
            "scene_count",
            "approved_count",
            "current_index",
            "scenes",
            "error",
        }
        # The upload prefix is not part of the title.
        assert (job["job_id"], job["title"]) == (job_id, "My_Trailer.mp4")
        assert (job["stage"], job["scene_count"], job["approved_count"]) == (
            "scenes",
            2,
            1,
        )
        assert job["current_index"] == 1 and job["error"] is None
        assert _stages(job) == {
            "file": "done",
            "segments": "done",
            "scenes": "current",
            "render": "pending",
            "result": "pending",
        }
        first, second = job["scenes"]
        assert set(first) == {
            "index",
            "start_s",
            "end_s",
            "label",
            "still_url",
            "status",
            "previewed",
            "reopenable",
        }
        assert (first["index"], first["start_s"], first["end_s"]) == (0, 0.0, 3.0)
        assert (first["label"], first["still_url"]) == ("a", None)
        assert (first["status"], first["previewed"], first["reopenable"]) == (
            "approved",
            True,
            True,
        )
        assert (second["status"], second["reopenable"]) == ("current", False)

        status, view, _origin = _json_request(
            base, "POST", "/api/scene", {"index": "0"}
        )
        assert status == 200
        assert (view["step"], view["segment_index"]) == ("preview_segment", 0)
        assert view["job"]["current_index"] == 0
        status, refused, _origin = _json_request(
            base, "POST", "/api/scene", {"index": "5"}
        )
        assert status == 400 and "cannot be opened yet" in refused["error"]

        # Back on the segments the stepper follows, and the strip is hidden.
        status, view, _origin = _json_request(
            base, "POST", "/api/goto", {"to": "segments"}
        )
        assert status == 200 and view["step"] == "segments"
        job = view["job"]
        assert job["stage"] == "segments" and job["scenes"] == []
        assert _stages(job) == {
            "file": "done",
            "segments": "current",
            "scenes": "pending",
            "render": "pending",
            "result": "pending",
        }
    finally:
        server.stop()


def test_the_job_summary_marks_a_finished_job(tmp_path: Path) -> None:
    _job_id, job_dir = _job_on(
        tmp_path, WizardStep.RESULT, segment_index=2, approved_count=2
    )
    (job_dir / "output.mp4").write_bytes(b"video")
    job = WizardController(tmp_path).api_view()["job"]
    assert _stages(job) == {
        "file": "done",
        "segments": "done",
        "scenes": "done",
        "render": "done",
        "result": "current",
    }
    assert [scene["status"] for scene in job["scenes"]] == ["approved"] * 2
    # Only a scene preview and "Ready to render" can reopen a scene.
    assert not any(scene["reopenable"] for scene in job["scenes"])


def test_no_job_summary_before_a_job(tmp_path: Path) -> None:
    assert WizardController(tmp_path).api_view()["job"] is None


@pytest.mark.parametrize(
    "step", [WizardStep.PREVIEW_SEGMENT, WizardStep.READY_TO_RENDER]
)
def test_a_half_written_shot_plan_still_gives_the_state(
    tmp_path: Path, step: WizardStep
) -> None:
    _job_id, job_dir = _job_on(tmp_path, step, segment_index=1, approved_count=1)
    (job_dir / "shot_plan.json").write_text('{"scale": 2, "reci', encoding="utf-8")
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        status, view, _origin = _json_request(base, "GET", "/api/state")
        assert status == 200 and view["step"] == step.value
        job = view["job"]
        assert "shot_plan.json is not valid JSON" in job["error"]
        # The scenes fall back to the approved segments, none reopenable.
        assert [scene["previewed"] for scene in job["scenes"]] == [False, False]
        assert not any(scene["reopenable"] for scene in job["scenes"])
        looks = view["segment_preview"] or view["looks"]
        assert "shot_plan.json is not valid JSON" in looks["error"]
    finally:
        server.stop()


def test_reopening_a_scene_needs_the_token(tmp_path: Path) -> None:
    _job_on(tmp_path, WizardStep.READY_TO_RENDER, segment_index=2, approved_count=2)
    server = WizardServer(tmp_path, token=TOKEN)
    base = server.start()
    try:
        connection = _connection(base)
        body = json.dumps({"index": "0"}).encode()
        connection.request(
            "POST",
            "/api/scene",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-Wizard-Token": "wrong",
            },
        )
        response = connection.getresponse()
        # A wrong token is no token: refused before anything is read.
        assert response.status == 401
        assert "token" in json.loads(response.read().decode())["error"]
        state = WizardController(tmp_path).current_state()
        assert (state.step, state.segment_index) == (WizardStep.READY_TO_RENDER, 2)
    finally:
        server.stop()
