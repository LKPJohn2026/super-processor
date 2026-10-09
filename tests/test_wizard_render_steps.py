"""Drive WizardController.render for each persisted step."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

from super_processor.gemini import GeminiClient, SplitProposal
from super_processor.jobs import JobState
from super_processor.review import WizardServer
from super_processor.segments import TimelineSegment, write_gray_still, write_segments
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
    write_split_layouts,
)


class _FakeTransport:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.bodies: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        del model, api_key
        self.bodies.append(body)
        return {
            "candidates": [{"content": {"parts": [{"text": json.dumps(self.payload)}]}}]
        }


def _seed_job(tmp_path: Path) -> tuple[WizardController, str, Path]:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-a-real-video")
    # Job store only checks is_file; probe is avoided by seeding states.
    controller = WizardController(
        tmp_path,
        gemini=GeminiClient(
            api_key="k",
            transport=_FakeTransport(
                {
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
                                    "params": {
                                        "contrast": 1.1,
                                        "brightness": 0.0,
                                        "gamma": 1.0,
                                    },
                                }
                            ],
                        },
                        {
                            "id": "C",
                            "label": "c",
                            "ops": [
                                {
                                    "op": "sharpen",
                                    "params": {"luma_amount": 0.3, "luma_size": 5},
                                }
                            ],
                        },
                    ],
                }
            ),
        ),
    )
    # Bypass create() file media checks by writing manifest manually via create
    # with a real tiny file if needed — use empty bytes file is enough for create.
    manifest = controller.store.create(source)
    job_dir = controller.store.job_dir(manifest.job_id)
    return controller, manifest.job_id, job_dir


def test_render_each_wizard_step(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    still = job_dir / "segment_stills" / "seg_00.ppm"
    write_gray_still(still, bytes([20]) * (160 * 90))
    write_segments(
        job_dir,
        [
            TimelineSegment(
                0, 0.0, 10.0, "mixed", "noisy", 5.0, 0, "segment_stills/seg_00.ppm"
            )
        ],
    )
    write_split_layouts(
        job_dir,
        SplitProposal.from_dict(
            {
                "highlights": ["h"],
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
        ),
    )
    steps = [
        (WizardStep.INTRO, "Super Processor"),
        (WizardStep.LLM_CHOICE, "local LLM"),
        (WizardStep.LOCAL_LLM_STUB, "Local LLM"),
        (WizardStep.SETUP, "Gemini"),
        (WizardStep.PICK_FILE, "Pick"),
        (WizardStep.ANALYZING, "Analyzing"),
        (WizardStep.RENDERING, "Rendering"),
        (WizardStep.RESULT, "Result"),
        (WizardStep.DONE, "Done"),
    ]
    for step, needle in steps:
        state = WizardState(step=step, job_id=job_id, highlights=["h"])
        save_job_wizard_state(job_dir, state)
        save_session_state(tmp_path, state)
        html = controller.render()
        assert needle in html

    state = WizardState(
        step=WizardStep.CHOOSE_SPLIT,
        job_id=job_id,
        highlights=["h"],
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
    assert "Restore strength" in controller.render()

    state = WizardState(
        step=WizardStep.ENHANCE,
        job_id=job_id,
        segment_index=0,
        enhance_cache={
            "upscale": {
                "restore_strength": 0.1,
                "scale": 3,
                "vsr_quality": "LOW",
            }
        },
    )
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    page = controller.render()
    assert "0.10" in page
    assert "LOW" in page


def test_wizard_server_serves_still_bmp(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    still = job_dir / "segment_stills" / "seg_00.ppm"
    write_gray_still(still, bytes([90]) * (160 * 90))
    write_segments(
        job_dir,
        [
            TimelineSegment(
                0, 0.0, 10.0, "mixed", "noisy", 5.0, 0, "segment_stills/seg_00.ppm"
            )
        ],
    )
    (job_dir / "output.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    state = WizardState(step=WizardStep.RESULT, job_id=job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        with urlopen(base + "/segment_stills/seg_00.bmp") as response:
            assert response.headers.get_content_type() == "image/bmp"
            assert response.read()[:2] == b"BM"
        with urlopen(base + "/output.mp4") as response:
            assert response.read()[:4] == b"\x00\x00\x00\x18"
    finally:
        server.stop()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_upscale_note_time_range_is_prompt_context(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    # Replace source with a real tiny media so revise can probe.
    import subprocess

    clip = tmp_path / "real.mp4"
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
            "color=c=blue:duration=12:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    # Point manifest at real clip by rewriting source via notes isn't supported;
    # create a fresh job instead.
    new = controller.store.create(clip)
    job_id = new.job_id
    job_dir = controller.store.job_dir(job_id)
    controller.store.transition(job_id, JobState.PROBED)
    controller.store.transition(job_id, JobState.SPLIT_PROPOSED)
    write_split_layouts(
        job_dir,
        SplitProposal.from_dict(
            {
                "highlights": [],
                "layouts": [
                    {
                        "segment_count": 1,
                        "summary": "old",
                        "segments": [
                            {
                                "start_s": 0,
                                "end_s": 12,
                                "label": "old",
                                "issues": ["noisy"],
                            }
                        ],
                    }
                ],
            }
        ),
    )
    upscale_payload = {
        "restore_strength": 0.2,
        "scale": 2,
        "vsr_quality": "HIGH",
    }
    transport = _FakeTransport(upscale_payload)
    controller._gemini = GeminiClient(api_key="k", transport=transport)
    state = WizardState(
        step=WizardStep.RESULT,
        job_id=job_id,
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
    note = "too soft from 1s to 3s"
    controller.handle_post("/result", {"mood": ["revise"], "note": [note]})
    current = controller.current_state()
    assert current.step is WizardStep.RENDERING
    assert current.enhance_cache["upscale"]["restore_strength"] == 0.2
    assert current.enhance_cache["upscale_note"] == note
    text = transport.bodies[-1]["contents"][-1]["parts"][0]["text"]
    assert "1s to 3s" in text
    assert "entire clip" in text
