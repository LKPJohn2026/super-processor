"""CLI and remaining wizard route coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

from super_processor.cli import run_review_command
from super_processor.gemini import store_gemini_api_key
from super_processor.jobs import JobStore
from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)


class _FakeTransport:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        self.calls += 1
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps(self.payload)}]}}]
        }


def test_store_gemini_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    store_gemini_api_key("abc123")
    assert os.environ["GEMINI_API_KEY"] == "abc123"


def test_run_review_starts_wizard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path)
    store.ensure_root()

    class _Immediate:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.started = False

        def start(self) -> str:
            self.started = True
            return "http://127.0.0.1:9"

        def wait(self) -> None:
            return

        def stop(self) -> None:
            return

    monkeypatch.setattr("super_processor.cli.WizardServer", _Immediate)
    code = run_review_command(argparse.Namespace(jobs_dir=tmp_path, job_id=None))
    assert code == 0


def test_run_review_with_job_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JobStore(tmp_path)
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")
    manifest = store.create(source)

    class _Immediate:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def start(self) -> str:
            return "http://127.0.0.1:9"

        def wait(self) -> None:
            return

        def stop(self) -> None:
            return

    monkeypatch.setattr("super_processor.cli.ReviewServer", _Immediate)
    code = run_review_command(
        argparse.Namespace(jobs_dir=tmp_path, job_id=manifest.job_id)
    )
    assert code == 0


def test_analyze_and_render_get_triggers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from super_processor.gemini import GeminiClient

    payload = {
        "highlights": [],
        "layouts": [
            {
                "segment_count": 1,
                "summary": "one",
                "segments": [
                    {
                        "start_s": 0,
                        "end_s": 10,
                        "label": "x",
                        "issues": ["noisy"],
                    }
                ],
            }
        ],
    }
    transport = _FakeTransport(payload)
    gemini = GeminiClient(api_key="k", transport=transport)
    controller = WizardController(tmp_path, gemini=gemini)

    def _noop_analyze() -> None:
        state = WizardState(step=WizardStep.PICK_FILE)
        save_session_state(tmp_path, state)

    def _noop_render() -> None:
        state = WizardState(step=WizardStep.DONE)
        save_session_state(tmp_path, state)

    monkeypatch.setattr(controller, "run_analyze", _noop_analyze)
    monkeypatch.setattr(controller, "run_render", _noop_render)
    state = WizardState(step=WizardStep.ANALYZING)
    save_session_state(tmp_path, state)
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        with urlopen(base + "/analyze?run=1") as response:
            body = response.read().decode()
        assert "Pick" in body
        state = WizardState(step=WizardStep.RENDERING)
        save_session_state(tmp_path, state)
        with urlopen(base + "/render?run=1") as response:
            assert "Done" in response.read().decode()
    finally:
        server.stop()


def test_result_revise_without_note_stays_on_result(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    source = tmp_path / "x.mp4"
    source.write_bytes(b"x")
    manifest = controller.store.create(source)
    job_dir = controller.store.job_dir(manifest.job_id)
    state = WizardState(step=WizardStep.RESULT, job_id=manifest.job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    controller.handle_post("/result", {"mood": ["revise"]})
    assert controller.current_state().step is WizardStep.RESULT
    assert controller.current_state().error
