"""Request guard and single-worker rendering for the localhost wizard."""

from __future__ import annotations

import json
import threading
from email.message import Message
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from super_processor.review import (
    ALLOWED_ORIGINS_ENV,
    WizardServer,
    request_rejection,
)
from super_processor.wizard import (
    WizardConflict,
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    save_session_state,
)


def _headers(**values: str) -> Message:
    message = Message()
    for key, value in values.items():
        message[key.replace("_", "-")] = value
    return message


def test_request_rejection_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    ok = _headers(Host="127.0.0.1:9000")
    assert request_rejection(ok, port=9000, state_changing=True) is None
    assert (
        request_rejection(
            _headers(Host="localhost:9000"), port=9000, state_changing=True
        )
        is None
    )
    assert (
        request_rejection(_headers(Host="[::1]:9000"), port=9000, state_changing=True)
        is None
    )
    assert request_rejection(_headers(), port=9000, state_changing=False) == (
        "missing Host header"
    )
    rebinding = _headers(Host="evil.example:9000")
    assert request_rejection(rebinding, port=9000, state_changing=False) == (
        "unexpected Host header"
    )
    foreign = _headers(Host="127.0.0.1:9000", Origin="https://evil.example")
    assert request_rejection(foreign, port=9000, state_changing=False) == (
        "origin not allowed"
    )
    pages = _headers(Host="127.0.0.1:9000", Origin="https://lkpjohn2026.github.io")
    assert request_rejection(pages, port=9000, state_changing=True) == (
        "origin not allowed"
    )
    own = _headers(Host="127.0.0.1:9000", Origin="http://127.0.0.1:9000")
    assert request_rejection(own, port=9000, state_changing=True) is None
    vite = _headers(Host="127.0.0.1:9000", Origin="http://localhost:5173")
    assert request_rejection(vite, port=9000, state_changing=True) is None
    image_tag = _headers(Host="127.0.0.1:9000", Sec_Fetch_Site="cross-site")
    assert request_rejection(image_tag, port=9000, state_changing=True) == (
        "cross-site request refused"
    )
    assert request_rejection(image_tag, port=9000, state_changing=False) is None
    refresh = _headers(Host="127.0.0.1:9000", Sec_Fetch_Site="same-origin")
    assert request_rejection(refresh, port=9000, state_changing=True) is None
    monkeypatch.setenv(
        ALLOWED_ORIGINS_ENV,
        "https://shell.example/, *, null, https://lkpjohn2026.github.io",
    )
    extra = _headers(Host="127.0.0.1:9000", Origin="https://shell.example")
    assert request_rejection(extra, port=9000, state_changing=True) is None
    assert (
        request_rejection(
            _headers(Host="127.0.0.1:9000", Origin="https://lkpjohn2026.github.io"),
            port=9000,
            state_changing=True,
        )
        is None
    )
    assert request_rejection(
        _headers(Host="127.0.0.1:9000", Origin="null"),
        port=9000,
        state_changing=True,
    ) == ("origin not allowed")
    assert request_rejection(
        _headers(Host="127.0.0.1:9000", Origin="*"),
        port=9000,
        state_changing=True,
    ) == ("origin not allowed")


def _request(
    base: str,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> tuple[int, bytes, str | None]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    sent = dict(headers or {})
    if body:
        sent.setdefault("Content-Length", str(len(body)))
    connection.request(method, path, body=body, headers=sent)
    response = connection.getresponse()
    data = response.read()
    return response.status, data, response.getheader("Access-Control-Allow-Origin")


def test_server_refuses_other_sites(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    evil = {"Origin": "https://evil.example"}
    try:
        payload = json.dumps({"api_key": "attacker"}).encode()
        status, _body, allow = _request(
            base,
            "POST",
            "/api/setup",
            headers={**evil, "Content-Type": "application/json"},
            body=payload,
        )
        assert status == 403
        assert allow is None
        status, _body, _allow = _request(
            base,
            "POST",
            "/intro",
            headers={**evil, "Content-Type": "application/x-www-form-urlencoded"},
            body=b"x=1",
        )
        assert status == 403
        assert controller.current_state().step is WizardStep.INTRO
        status, _body, allow = _request(base, "GET", "/api/state", headers=evil)
        assert status == 403
        assert allow is None
        status, _body, _allow = _request(base, "OPTIONS", "/api/setup", headers=evil)
        assert status == 403
        status, _body, _allow = _request(
            base, "GET", "/render?run=1", headers={"Sec-Fetch-Site": "cross-site"}
        )
        assert status == 403
        status, _body, _allow = _request(
            base, "GET", "/", headers={"Host": "rebound.example"}
        )
        assert status == 403
        status, _body, allow = _request(
            base, "GET", "/api/state", headers={"Origin": "http://localhost:5173"}
        )
        assert status == 200
        assert allow == "http://localhost:5173"
        status, _body, _allow = _request(
            base,
            "POST",
            "/intro",
            headers={"Origin": base, "X-Wizard-Token": server.token},
            body=b"x=1",
        )
        assert status == 303
        assert controller.current_state().step is WizardStep.LLM_CHOICE
    finally:
        server.stop()


def test_one_render_at_a_time(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )
    release = threading.Event()
    entered = threading.Event()
    calls: list[int] = []

    def _slow_render() -> None:
        calls.append(1)
        entered.set()
        release.wait(5)
        save_session_state(
            tmp_path, WizardState(step=WizardStep.RESULT, job_id="job00001")
        )

    controller._run_render = _slow_render  # type: ignore[method-assign]
    assert controller.start_render() is True
    assert entered.wait(5)
    assert controller.busy()
    assert controller.start_render() is False
    controller.run_render()  # a direct call while busy is a no-op
    assert controller.api_view()["busy"] is True
    with pytest.raises(WizardError, match="still running"):
        controller.handle_post("/result", {"mood": ["happy"]})
    release.set()
    assert controller.wait_idle(timeout=5)
    assert calls == [1]
    assert controller.current_state().step is WizardStep.RESULT
    assert controller.start_render() is False


def _boom() -> None:
    raise RuntimeError("gpu fell over")


def test_worker_error_without_output_goes_back_to_pick(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )
    controller._run_render = _boom  # type: ignore[method-assign]
    assert controller.start_render() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.job_id is None
    assert state.error is not None and "gpu fell over" in state.error
    assert not controller.busy()
    run = controller.api_view()["run"]
    assert run is not None
    assert (run["kind"], run["job_id"], run["outcome"]) == (
        "render",
        "job00001",
        "failed",
    )
    assert run["finished_at"] >= run["started_at"]


def test_worker_error_with_output_stays_on_result(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    (tmp_path / "job00001" / "output.mp4").write_bytes(b"earlier result")
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )
    controller._run_render = _boom  # type: ignore[method-assign]
    assert controller.start_render() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.job_id == "job00001"
    assert state.error == "gpu fell over"
    run = controller._public_run(state)
    assert run is not None and run["outcome"] == "failed"


def test_a_handled_render_failure_is_reported_as_failed(tmp_path: Path) -> None:
    """A render that catches its own error still ends as failed, not done."""
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )

    def _caught() -> None:
        controller._render_failed(controller.current_state(), "no GPU")

    controller._run_render = _caught  # type: ignore[method-assign]
    assert controller.start_render() is True
    assert controller.wait_idle(timeout=5)
    assert controller.current_state().step is WizardStep.PICK_FILE
    run = controller.api_view()["run"]
    assert run is not None and run["outcome"] == "failed"


def test_a_failing_failure_screen_still_ends_the_run(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )
    controller._run_render = _boom  # type: ignore[method-assign]

    def _broken(state: WizardState, detail: str) -> None:
        raise OSError("disk full")

    controller._render_failed = _broken  # type: ignore[method-assign]
    assert controller.start_render() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error is not None
    assert "gpu fell over" in state.error and "disk full" in state.error
    assert not controller.busy()
    run = controller.api_view()["run"]
    assert run is not None and run["outcome"] == "failed"


def test_a_run_off_its_step_records_nothing(tmp_path: Path) -> None:
    """Calling run_render on the result screen keeps the last run as it was."""
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )

    def _done() -> None:
        save_session_state(
            tmp_path, WizardState(step=WizardStep.RESULT, job_id="job00001")
        )

    controller._run_render = _done  # type: ignore[method-assign]
    assert controller.start_render() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    first = controller._public_run(state)
    assert first is not None and first["outcome"] == "done"
    controller.run_render()
    controller.run_analyze()
    controller.run_segmenting()
    assert controller._public_run(controller.current_state()) == first
    assert not controller.busy()


def test_no_run_is_recorded_before_any_work(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    save_session_state(tmp_path, WizardState(step=WizardStep.RENDERING))
    controller.run_render()  # no job: nothing to render
    assert controller.api_view()["run"] is None


def test_the_result_is_saved_while_the_run_still_counts_as_busy(
    tmp_path: Path,
) -> None:
    """A page that reads the result screen sees busy, and the run as done;
    once busy is false, the run never reads as still running."""
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )
    seen: list[tuple[bool, str | None]] = []
    real_save = controller._save

    def _hooked(state: WizardState) -> None:
        real_save(state)
        if state.step is WizardStep.RESULT:
            run = controller._public_run(controller.current_state())
            seen.append((controller.busy(), None if run is None else run["outcome"]))

    controller._save = _hooked  # type: ignore[method-assign]

    def _render() -> None:
        controller._save(WizardState(step=WizardStep.RESULT, job_id="job00001"))

    controller._run_render = _render  # type: ignore[method-assign]
    snapshots: list[tuple[bool, str | None]] = []
    stop = threading.Event()

    def _poll() -> None:
        while not stop.is_set():
            busy = controller.busy()
            run = controller._public_run(controller.current_state())
            snapshots.append((busy, None if run is None else run["outcome"]))

    poller = threading.Thread(target=_poll)
    poller.start()
    try:
        assert controller.start_render() is True
        assert controller.wait_idle(timeout=5)
    finally:
        stop.set()
        poller.join(5)
    assert seen == [(True, "done")]
    assert (False, "running") not in snapshots
    final = controller._public_run(controller.current_state())
    assert final is not None and final["outcome"] == "done"


def test_a_worker_never_starts_inside_a_post(tmp_path: Path) -> None:
    """start_render waits for a POST that is changing the state, then sees
    the state that POST saved."""
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )
    inside = threading.Event()
    release = threading.Event()

    def _slow_goto(state: WizardState, target: str) -> None:
        inside.set()
        release.wait(5)
        controller._save(WizardState(step=WizardStep.PICK_FILE, job_id="job00001"))

    controller._goto = _slow_goto  # type: ignore[method-assign]
    controller._run_render = lambda: None  # type: ignore[method-assign]
    post = threading.Thread(
        target=controller.handle_post, args=("/goto", {"to": ["file"]})
    )
    post.start()
    assert inside.wait(5)
    started: list[bool] = []
    starter = threading.Thread(target=lambda: started.append(controller.start_render()))
    starter.start()
    starter.join(0.3)
    assert starter.is_alive(), "start_render must wait for the POST"
    release.set()
    post.join(5)
    starter.join(5)
    assert started == [False]
    assert controller.current_state().step is WizardStep.PICK_FILE
    assert controller._public_run(controller.current_state()) is None


def test_a_post_is_refused_until_the_worker_thread_ends(tmp_path: Path) -> None:
    """The worker's last progress clean-up runs after the work lock is free;
    a POST must not land before it."""
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )
    cleaning = threading.Event()
    release = threading.Event()
    real_clear = controller._clear_progress

    def _slow_clear() -> None:
        cleaning.set()
        release.wait(5)
        real_clear()

    controller._clear_progress = _slow_clear  # type: ignore[method-assign]
    controller._run_render = lambda: None  # type: ignore[method-assign]
    assert controller.start_render() is True
    assert cleaning.wait(5)
    assert not controller._work_lock.locked()
    assert controller.busy()
    with pytest.raises(WizardError, match="still running"):
        controller.handle_post("/new", {})
    release.set()
    assert controller.wait_idle(timeout=5)
    controller.handle_post("/new", {})


def test_a_post_for_another_job_is_a_conflict(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    (tmp_path / "job00002").mkdir()
    save_session_state(tmp_path, WizardState(step=WizardStep.SETUP, job_id="job00002"))
    with pytest.raises(WizardConflict, match="no longer open in this window"):
        controller.handle_post("/intro", {"job_id": ["job00001"]})
    assert controller.current_state().step is WizardStep.SETUP
    controller.handle_post("/intro", {"job_id": ["job00002"]})
    assert controller.current_state().step is WizardStep.LLM_CHOICE
    controller.handle_post("/intro", {"job_id": [""]})  # no job sent: no check


def test_state_files_retry_windows_sharing_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page read racing the worker's save must not fail the request."""
    monkeypatch.setattr("super_processor.atomic_io.RETRY_S", 0)
    save_session_state(tmp_path, WizardState(step=WizardStep.PICK_FILE))

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
    save_session_state(tmp_path, WizardState(step=WizardStep.DONE))
    monkeypatch.setattr(Path, "read_text", flaky_read)
    controller = WizardController(tmp_path)
    assert controller.current_state().step is WizardStep.DONE
    assert calls == {"replace": 3, "read": 3}
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_state_saves_do_not_collide(tmp_path: Path) -> None:
    errors: list[BaseException] = []

    def hammer(step: WizardStep) -> None:
        try:
            for _ in range(200):
                save_session_state(tmp_path, WizardState(step=step))
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [
        threading.Thread(target=hammer, args=(step,))
        for step in (WizardStep.RENDERING, WizardStep.RESULT)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert WizardController(tmp_path).current_state().step in {
        WizardStep.RENDERING,
        WizardStep.RESULT,
    }
    assert list(tmp_path.glob("*.tmp")) == []


def test_a_save_that_never_gets_the_file_keeps_the_old_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retries are bounded: a file held open for good fails the save, and
    neither the old state nor a stray temp file is left broken behind it."""
    monkeypatch.setattr("super_processor.atomic_io.RETRY_S", 0)
    save_session_state(tmp_path, WizardState(step=WizardStep.PICK_FILE))
    real_replace = Path.replace
    attempts = {"replace": 0}
    held = [True]

    def locked(self: Path, target: Path) -> Path:
        if not held[0]:
            return real_replace(self, target)
        attempts["replace"] += 1
        raise PermissionError(13, "file is open in another program")

    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(PermissionError):
        save_session_state(tmp_path, WizardState(step=WizardStep.DONE))
    held[0] = False
    assert attempts["replace"] == 20
    assert list(tmp_path.glob("*.tmp")) == []
    assert WizardController(tmp_path).current_state().step is WizardStep.PICK_FILE
