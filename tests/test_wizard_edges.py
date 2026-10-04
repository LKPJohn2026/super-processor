"""Edge paths for wizard coverage."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from super_processor.gemini import GeminiClient, SplitProposal
from super_processor.jobs import JobState
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
    write_split_layouts,
)


class _FakeTransport:
    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        self.payloads = list(payloads)

    def generate(
        self, *, model: str, api_key: str, body: dict[str, Any]
    ) -> dict[str, Any]:
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
    controller.handle_post("/setup", {"api_key": ["k"]})
    controller.handle_post("/pick", {"path": [str(tmp_path / "missing.mp4")]})
    assert controller.current_state().step is WizardStep.PICK_FILE
    assert controller.current_state().error


def test_enhance_note_and_overview_render(tmp_path: Path) -> None:
    clip = _clip(tmp_path / "c.mp4")
    enhance = {
        "issues": ["noisy"],
        "options": [
            {
                "id": "A",
                "label": "a",
                "ops": [{"op": "denoise", "params": {"strength": 0.2}}],
            },
            {
                "id": "B",
                "label": "b",
                "ops": [
                    {
                        "op": "contrast",
                        "params": {"contrast": 1.1, "brightness": 0.0, "gamma": 1.0},
                    }
                ],
            },
            {
                "id": "C",
                "label": "c",
                "ops": [
                    {"op": "sharpen", "params": {"luma_amount": 0.3, "luma_size": 5}}
                ],
            },
        ],
    }
    split = {
        "highlights": ["g"],
        "layouts": [
            {
                "segment_count": 1,
                "summary": "one",
                "segments": [
                    {"start_s": 0, "end_s": 12, "label": "x", "issues": ["noisy"]}
                ],
            }
        ],
    }
    # revise enhance consumes one; initial ensure consumes one
    gemini = GeminiClient(api_key="k", transport=_FakeTransport([enhance, enhance]))
    controller = WizardController(tmp_path, gemini=gemini)
    manifest = controller.store.create(clip)
    controller.store.transition(manifest.job_id, JobState.PROBED)
    controller.store.transition(manifest.job_id, JobState.SPLIT_PROPOSED)
    job_dir = controller.store.job_dir(manifest.job_id)
    write_split_layouts(job_dir, SplitProposal.from_dict(split))
    state = WizardState(
        step=WizardStep.OVERVIEW, job_id=manifest.job_id, highlights=["g"]
    )
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    assert (
        "Duration" in controller.render()
        or "overview" in controller.render().lower()
        or "Quick" in controller.render()
    )
    controller.handle_post("/overview", {})
    controller.handle_post("/split", {"layout": ["0"]})
    assert controller.current_state().step is WizardStep.ENHANCE
    controller.handle_post("/segment", {"note": ["make it warmer"]})
    assert "0" in controller.current_state().enhance_cache
