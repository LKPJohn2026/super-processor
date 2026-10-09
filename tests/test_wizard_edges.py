"""Edge paths for wizard coverage."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from super_processor.gemini import GeminiClient
from super_processor.jobs import JobState
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)


class _FakeTransport:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)
        self.calls = 0
        self.bodies: list[dict[str, Any]] = []

    def generate(
        self, *, model: str, api_key: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        del model, api_key
        self.calls += 1
        self.bodies.append(body)
        payload = self.payloads.pop(0)
        return {"candidates": [{"content": {"parts": [{"text": json.dumps(payload)}]}}]}


def _clip(path: Path, seconds: int = 12) -> Path:
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
            f"color=c=green:duration={seconds}:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def test_pick_missing_file(tmp_path: Path) -> None:
    controller = WizardController(tmp_path)
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": ["gemini"]})
    controller.handle_post("/setup", {"api_key": ["k"]})
    controller.handle_post("/pick", {"path": [str(tmp_path / "missing.mp4")]})
    assert controller.current_state().step is WizardStep.PICK_FILE
    assert controller.current_state().error


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_overview_then_artificial_note_cannot_raise_knobs(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "c.mp4")
    raised = {
        "restore_strength": 0.3,
        "scale": 2,
        "vsr_quality": "HIGH",
    }
    gemini = GeminiClient(api_key="k", transport=_FakeTransport([raised]))
    controller = WizardController(tmp_path, gemini=gemini)
    manifest = controller.store.create(clip)
    controller.store.transition(manifest.job_id, JobState.PROBED)
    controller.store.transition(manifest.job_id, JobState.SPLIT_PROPOSED)
    job_dir = controller.store.job_dir(manifest.job_id)
    state = WizardState(
        step=WizardStep.OVERVIEW, job_id=manifest.job_id, highlights=["g"]
    )
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    assert "Quick" in controller.render()
    controller.handle_post("/overview", {})
    assert controller.current_state().step is WizardStep.RENDERING
    state = WizardState(
        step=WizardStep.RESULT,
        job_id=manifest.job_id,
        enhance_cache=dict(controller.current_state().enhance_cache),
    )
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    controller.handle_post(
        "/result",
        {"mood": ["revise"], "note": ["too artificial and plastic"]},
    )
    current = controller.current_state()
    assert current.step is WizardStep.RESULT
    assert current.error
    assert current.enhance_cache["upscale"]["restore_strength"] == 0.15
    assert current.enhance_cache["upscale"]["vsr_quality"] == "MEDIUM"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_soft_note_can_raise_knobs_inside_the_cap(tmp_path: Path) -> None:
    """A sharpness note may raise knobs, and the hard cap still applies."""
    clip = _clip(tmp_path / "c.mp4")
    inside = {
        "restore_strength": 0.3,
        "scale": 3,
        "vsr_quality": "HIGH",
    }
    over = {
        "restore_strength": 0.5,
        "scale": 2,
        "vsr_quality": "HIGH",
    }
    transport = _FakeTransport([inside, over])
    gemini = GeminiClient(api_key="k", transport=transport)
    controller = WizardController(tmp_path, gemini=gemini)
    manifest = controller.store.create(clip)
    controller.store.transition(manifest.job_id, JobState.PROBED)
    controller.store.transition(manifest.job_id, JobState.SPLIT_PROPOSED)
    job_dir = controller.store.job_dir(manifest.job_id)
    state = WizardState(
        step=WizardStep.RESULT,
        job_id=manifest.job_id,
        enhance_cache={
            "upscale": {
                "restore_strength": 0.15,
                "scale": 2,
                "vsr_quality": "MEDIUM",
            }
        },
    )
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    controller.handle_post(
        "/result",
        {"mood": ["revise"], "note": ["too soft, more sharpness please"]},
    )
    current = controller.current_state()
    assert current.step is WizardStep.RENDERING
    assert current.enhance_cache["upscale"]["restore_strength"] == 0.3
    assert current.enhance_cache["upscale"]["vsr_quality"] == "HIGH"
    assert current.enhance_cache["upscale"]["scale"] == 3
    user_text = transport.bodies[-1]["contents"][-1]["parts"][0]["text"]
    assert "too soft" in user_text
    assert "entire clip" in user_text

    current.step = WizardStep.RESULT
    save_job_wizard_state(job_dir, current)
    save_session_state(tmp_path, current)
    controller.handle_post(
        "/result",
        {"mood": ["revise"], "note": ["still too soft, push sharpness further"]},
    )
    held = controller.current_state()
    assert held.step is WizardStep.RESULT
    assert held.error
    assert held.enhance_cache["upscale"]["restore_strength"] == 0.3
