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
    rebinding = _headers(Host="evil.example:9000")
    assert request_rejection(rebinding, port=9000, state_changing=False) == (
        "unexpected Host header"
    )
    foreign = _headers(Host="127.0.0.1:9000", Origin="https://evil.example")
    assert request_rejection(foreign, port=9000, state_changing=False) == (
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
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://shell.example/, ")
    extra = _headers(Host="127.0.0.1:9000", Origin="https://shell.example")
    assert request_rejection(extra, port=9000, state_changing=True) is None


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
            base, "POST", "/intro", headers={"Origin": base}, body=b"x=1"
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


def test_worker_error_lands_on_the_result_step(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    (tmp_path / "job00001").mkdir()
    save_session_state(
        tmp_path, WizardState(step=WizardStep.RENDERING, job_id="job00001")
    )

    def _boom() -> None:
        raise RuntimeError("gpu fell over")

    controller._run_render = _boom  # type: ignore[method-assign]
    assert controller.start_render() is True
    assert controller.wait_idle(timeout=5)
    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert state.error == "gpu fell over"
    assert not controller.busy()
