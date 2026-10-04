"""Serve preview media from the wizard server."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pytest

from super_processor.gemini import GeminiClient
from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)


class _FakeTransport:
    def generate(
        self, *, model: str, api_key: str, body: dict[str, object]
    ) -> dict[str, object]:
        return {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]}


def test_serve_preview_mp4(tmp_path: Path) -> None:
    controller = WizardController(
        tmp_path, gemini=GeminiClient(api_key="k", transport=_FakeTransport())
    )
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")
    manifest = controller.store.create(source)
    job_dir = controller.store.job_dir(manifest.job_id)
    preview = job_dir / "previews" / "seg_00_wizard.A.mp4"
    preview.parent.mkdir(parents=True)
    preview.write_bytes(b"fake-mp4")
    state = WizardState(step=WizardStep.ENHANCE, job_id=manifest.job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        with urlopen(base + "/previews/seg_00_wizard.A.mp4") as response:
            assert response.read() == b"fake-mp4"
    finally:
        server.stop()


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
