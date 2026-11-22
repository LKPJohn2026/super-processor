"""Hardening: CSRF surface, run token, body size, pick/symlink, stills."""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import threading
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
from super_processor.segment_loop import (
    LoopSegment,
    SegmentList,
    load_segment_list,
    save_segment_list,
)
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)

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
                "X-Wizard-Token": TOKEN,
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
    assert state.step is WizardStep.SAMPLING


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
    port = urlparse(base).port
    form = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        # No Origin and no Sec-Fetch-Site: only the token stops these.
        assert _request(base, "POST", "/intro", headers=form, body=b"x=1")[0] == 401
        wrong = {**form, "X-Wizard-Token": "guess"}
        assert _request(base, "POST", "/intro", headers=wrong, body=b"x=1")[0] == 401
        status, body = _request(base, "GET", "/render?run=1")
        assert status == 403 and b"wizard token" in body
        # The page itself (which embeds the token) needs it too.
        assert _request(base, "GET", "/")[0] == 401
        assert controller.current_state().step is WizardStep.INTRO

        # The printed launch URL swaps the token for a cookie.
        connection = HTTPConnection("127.0.0.1", port)
        connection.request("GET", f"/?token={TOKEN}")
        launched = connection.getresponse()
        launched.read()
        assert launched.status == 303
        cookie = (launched.getheader("Set-Cookie") or "").split(";", 1)[0]
        assert cookie == f"sp_wizard_{port}={TOKEN}"
        browser = {**form, "Cookie": cookie}
        status, page = _request(base, "GET", "/", headers=browser)
        assert status == 200
        connection = HTTPConnection("127.0.0.1", port)
        connection.request("GET", "/", headers=browser)
        framed = connection.getresponse()
        framed.read()
        assert framed.getheader("X-Frame-Options") == "DENY"
        assert f'name="token" value="{TOKEN}"'.encode() in page
        # The cookie comes along by itself, so a post also needs the
        # page's own token field.
        assert _request(base, "POST", "/intro", headers=browser, body=b"x=1")[0] == 403
        field = f"token={TOKEN}".encode()
        assert _request(base, "POST", "/intro", headers=browser, body=field)[0] == 303
        assert controller.current_state().step is WizardStep.LLM_CHOICE

        # The API hands the token to the shell, which sends it as a header.
        status, body = _request(base, "GET", "/api/state", headers=browser)
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
        '<form method="post" action="/preview"><button>Go</button></form>'
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


def test_segment_list_save_and_load_retry_sharing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """segments.json is written and read while a page refresh may hold it."""
    monkeypatch.setattr("super_processor.atomic_io.RETRY_S", 0)
    saved = SegmentList(
        segments=[LoopSegment(0.0, 4.0, label="kitchen"), LoopSegment(4.0, 9.5)],
        approved=True,
    )
    real_replace = Path.replace
    real_read = Path.read_text
    calls = {"replace": 0, "read": 0}

    def flaky_replace(self: Path, target: Path) -> Path:
        calls["replace"] += 1
        if calls["replace"] < 3:
            raise PermissionError(13, "file is open in another thread")
        return real_replace(self, target)

    def flaky_read(self: Path, *args: object, **kwargs: object) -> str:
        calls["read"] += 1
        if calls["read"] < 3:
            raise PermissionError(13, "file is being replaced")
        return real_read(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "replace", flaky_replace)
    save_segment_list(tmp_path, saved)
    monkeypatch.setattr(Path, "read_text", flaky_read)
    loaded = load_segment_list(tmp_path)
    assert loaded is not None
    assert loaded.to_dict() == saved.to_dict()
    assert calls == {"replace": 3, "read": 3}
    assert list(tmp_path.glob("*.tmp")) == []


def test_stopping_a_run_needs_the_token_and_an_allowed_origin(
    tmp_path: Path,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    controller = WizardController(tmp_path / "jobs")
    job_id = controller.store.create(source).job_id
    state = WizardState(step=WizardStep.RENDERING, job_id=job_id)
    save_job_wizard_state(controller.store.job_dir(job_id), state)
    save_session_state(controller.jobs_dir, state)
    server = WizardServer(controller.jobs_dir, controller=controller, token=TOKEN)
    base = server.start()
    stop = json.dumps({"job_id": job_id}).encode()
    try:
        plain = {"Content-Type": "application/json"}
        assert _request(base, "POST", "/api/cancel", headers=plain, body=stop)[0] == 401
        foreign = {
            **plain,
            "Origin": "https://evil.example",
            "X-Wizard-Token": TOKEN,
        }
        status, body = _request(base, "POST", "/api/cancel", headers=foreign, body=stop)
        assert status == 403 and b"origin not allowed" in body
        assert controller.current_state().step is WizardStep.RENDERING

        status, body = _request(
            base,
            "POST",
            "/api/cancel",
            headers={**plain, "X-Wizard-Token": TOKEN},
            body=stop,
        )
        assert status == 200
        view = json.loads(body)
        assert view["step"] == "ready_to_render" and view["stopped"]
        assert view["run"]["outcome"] == "stopped"
        # Nothing is left to stop now.
        status, body = _request(
            base,
            "POST",
            "/api/cancel",
            headers={**plain, "X-Wizard-Token": TOKEN},
            body=stop,
        )
        assert status == 400 and b"nothing running to stop" in body
    finally:
        server.stop()


def test_starting_a_prefetch_needs_the_token_and_the_run_link(
    tmp_path: Path,
) -> None:
    controller = WizardController(tmp_path, prefetch=True)
    started: list[str] = []

    def _start() -> bool:
        started.append("prefetch")
        return False

    controller.start_prefetch = _start  # type: ignore[method-assign]
    server = WizardServer(tmp_path, controller=controller, token=TOKEN)
    base = server.start()
    try:
        status, body = _request(base, "GET", "/api/prefetch?run=1")
        assert status == 403 and b"wizard token" in body
        status, _body = _request(base, "GET", "/api/prefetch?run=1&token=guess")
        assert status == 403
        # Without the run link it is not an endpoint at all.
        status, _body = _request(
            base, "GET", "/api/prefetch", headers={"X-Wizard-Token": TOKEN}
        )
        assert status == 404
        assert started == []
        status, body = _request(base, "GET", f"/api/prefetch?run=1&token={TOKEN}")
        assert status == 200 and json.loads(body)["step"] == "intro"
        assert started == ["prefetch"]
    finally:
        server.stop()


def test_two_edits_at_once_apply_one_and_refuse_the_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two windows change the same scenes at the same moment, both from the
    list they show (revision 0): one change lands, the other is told the
    scenes changed, and nothing is half-saved."""
    from super_processor import wizard as wizard_module

    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    controller = WizardController(tmp_path / "jobs")
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    save_segment_list(
        job_dir,
        SegmentList(segments=[LoopSegment(0.0, 6.0), LoopSegment(6.0, 12.0)]),
    )
    controller._save(WizardState(step=WizardStep.SEGMENTS, job_id=job_id))
    real_load = load_segment_list
    both_in = threading.Barrier(2, timeout=0.5)

    def slow_load(path: Path) -> SegmentList | None:
        # Each POST reads the list; wait until both have tried to start, so
        # without the POST lock both would read revision 0.
        with contextlib.suppress(threading.BrokenBarrierError):
            both_in.wait()
        return real_load(path)

    monkeypatch.setattr(wizard_module, "load_segment_list", slow_load)
    outcomes: list[str] = []

    def edit(at: str) -> None:
        try:
            controller.handle_post(
                "/segments",
                {"action": ["move"], "index": ["0"], "at": [at], "revision": ["0"]},
            )
            outcomes.append(f"applied {at}")
        except WizardError as exc:
            outcomes.append(str(exc))

    threads = [threading.Thread(target=edit, args=(at,)) for at in ("4", "8")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert len(outcomes) == 2
    applied = [o for o in outcomes if o.startswith("applied")]
    assert len(applied) == 1
    assert any("scenes changed" in o for o in outcomes)
    found = real_load(job_dir)
    assert found is not None and found.revision == 1
    assert found.segments[0].end_s == float(applied[0].split()[1])
