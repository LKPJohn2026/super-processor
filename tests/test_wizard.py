"""Tests for the Gemini wizard controller and pages."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardStep,
    load_session_state,
)
from super_processor.wizard_pages import render_intro


class _FakeTransport:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)

    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        payload = self.payloads.pop(0)
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


def _tiny_clip(path: Path, seconds: int = 20) -> Path:
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
            f"color=c=gray:duration={seconds}:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_intro_page_mentions_enhancement() -> None:
    html = render_intro()
    assert "Super Processor" in html
    assert "do not invent" in html.lower() or "Enhance" in html


def test_wizard_server_serves_intro(tmp_path: Path) -> None:
    server = WizardServer(tmp_path)
    base = server.start()
    try:
        with urlopen(base + "/") as response:
            body = response.read().decode()
    finally:
        server.stop()
    assert "Super Processor" in body
    assert "Continue" in body


def test_wizard_setup_and_pick_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = WizardController(tmp_path)
    assert controller.current_state().step is WizardStep.INTRO
    controller.handle_post("/intro", {})
    assert controller.current_state().step is WizardStep.LLM_CHOICE
    controller.handle_post("/llm", {"choice": ["local"]})
    assert controller.current_state().step is WizardStep.LOCAL_LLM_STUB
    assert "Gemini" in controller.render()
    controller.handle_post("/local-llm", {"action": ["gemini"]})
    assert controller.current_state().step is WizardStep.SETUP
    controller.handle_post("/setup", {"api_key": ["unit-test-key"]})
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert load_session_state(tmp_path).step is WizardStep.PICK_FILE


def test_wizard_llm_choice_gemini_direct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = WizardController(tmp_path)
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    assert controller.current_state().step is WizardStep.SETUP
