"""Tests for the Gemini wizard controller and pages."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pytest

from super_processor.gemini import (
    GeminiClient,
    SegmentEnhanceResult,
    SplitProposal,
    StructuredOp,
)
from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardStep,
    layout_to_segments,
    load_session_state,
    treatment_from_ops,
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


def test_treatment_from_ops_orders_steps() -> None:
    treatment = treatment_from_ops(
        [
            StructuredOp("sharpen", {"luma_amount": 0.4, "luma_size": 5}),
            StructuredOp(
                "contrast", {"contrast": 1.2, "brightness": 0.0, "gamma": 1.0}
            ),
        ],
        treatment_id="wizard.A",
        problem="low_contrast",
    )
    assert treatment.steps[0].op.value == "contrast"
    assert treatment.steps[1].op.value == "sharpen"


def test_layout_to_segments_maps_issues() -> None:
    proposal = SplitProposal.from_dict(
        {
            "highlights": [],
            "layouts": [
                {
                    "segment_count": 2,
                    "summary": "two",
                    "segments": [
                        {
                            "start_s": 0,
                            "end_s": 40,
                            "label": "dark room",
                            "issues": ["low light"],
                        },
                        {
                            "start_s": 40,
                            "end_s": 80,
                            "label": "outside",
                            "issues": ["too warm"],
                        },
                    ],
                }
            ],
        }
    )
    segments = layout_to_segments(proposal.layouts[0])
    assert segments[0].problem == "low_light"
    assert segments[1].problem == "too_warm"
    assert segments[0].end_s == 40


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
    assert controller.current_state().step is WizardStep.SETUP
    controller.handle_post("/setup", {"api_key": ["unit-test-key"]})
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert load_session_state(tmp_path).step is WizardStep.PICK_FILE


@pytest.mark.skipif(
    subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0,
    reason="ffmpeg required",
)
def test_wizard_analyze_with_fake_gemini(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-key")
    clip = _tiny_clip(tmp_path / "clip.mp4", seconds=20)
    split_payload = {
        "highlights": ["gray clip"],
        "layouts": [
            {
                "segment_count": 2,
                "summary": "halves",
                "segments": [
                    {
                        "start_s": 0,
                        "end_s": 10,
                        "label": "first",
                        "issues": ["noisy"],
                    },
                    {
                        "start_s": 10,
                        "end_s": 20,
                        "label": "second",
                        "issues": ["low_contrast"],
                    },
                ],
            },
            {
                "segment_count": 1,
                "summary": "whole",
                "segments": [
                    {
                        "start_s": 0,
                        "end_s": 20,
                        "label": "all",
                        "issues": ["noisy"],
                    }
                ],
            },
        ],
    }
    enhance_payload = {
        "issues": ["noisy"],
        "options": [
            {
                "id": "A",
                "label": "light denoise",
                "ops": [{"op": "denoise", "params": {"strength": 0.3}}],
            },
            {
                "id": "B",
                "label": "contrast",
                "ops": [
                    {
                        "op": "contrast",
                        "params": {"contrast": 1.2, "brightness": 0.0, "gamma": 1.0},
                    }
                ],
            },
            {
                "id": "C",
                "label": "sharpen",
                "ops": [
                    {"op": "sharpen", "params": {"luma_amount": 0.4, "luma_size": 5}}
                ],
            },
        ],
    }
    transport = _FakeTransport([split_payload, enhance_payload])
    gemini = GeminiClient(api_key="unit-test-key", transport=transport)
    controller = WizardController(tmp_path, gemini=gemini)
    controller.handle_post("/intro", {})
    controller.handle_post("/setup", {"skip": ["1"]})
    controller.handle_post("/pick", {"path": [str(clip)]})
    assert controller.current_state().step is WizardStep.ANALYZING
    controller.run_analyze()
    state = controller.current_state()
    assert state.step is WizardStep.OVERVIEW
    assert state.highlights == ["gray clip"]
    controller.handle_post("/overview", {})
    controller.handle_post("/split", {"layout": ["0"]})
    state = controller.current_state()
    assert state.step is WizardStep.ENHANCE
    assert state.segment_index == 0
    raw = state.enhance_cache["0"]
    result = SegmentEnhanceResult.from_dict(raw)
    assert len(result.options) == 3
