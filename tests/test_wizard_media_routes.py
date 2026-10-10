"""Serve preview media from the wizard server."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.wizard import (
    WizardController,
    WizardStep,
)


class _FakeTransport:
    def generate(
        self, *, model: str, api_key: str, body: dict[str, object]
    ) -> dict[str, object]:
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}


def test_setup_rejects_empty_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("super_processor.gemini.resolve_secret", lambda _env: None)
    controller = WizardController(tmp_path)
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    controller.handle_post("/setup", {"api_key": ["  "]})
    assert controller.current_state().step is WizardStep.SETUP
    assert controller.current_state().error
