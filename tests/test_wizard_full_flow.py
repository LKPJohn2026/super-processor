"""CI coverage: full localhost wizard website from intro through done."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from super_processor.gemini import GeminiClient
from super_processor.review import WizardServer
from super_processor.wizard import WizardController, WizardStep


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


def _tiny_clip(path: Path, seconds: int = 12) -> Path:
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
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


def _get(base: str, path: str = "/") -> str:
    with urlopen(base + path) as response:
        return str(response.read().decode())


def _post(base: str, path: str, fields: dict[str, str]) -> str:
    req = Request(
        base + path,
        data=urlencode(fields).encode(),
        method="POST",
    )
    with urlopen(req) as response:
        return str(response.read().decode())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_wizard_website_full_flow_intro_to_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive every website screen over HTTP and assert UI updates."""
    monkeypatch.setenv("GEMINI_API_KEY", "ci-full-flow-key")
    clip = _tiny_clip(tmp_path / "clip.mp4")
    split_payload = {
        "highlights": ["gray CI clip"],
        "layouts": [
            {
                "segment_count": 1,
                "summary": "whole clip",
                "segments": [
                    {
                        "start_s": 0,
                        "end_s": 12,
                        "label": "all",
                        "issues": ["noisy"],
                    }
                ],
            },
            {
                "segment_count": 2,
                "summary": "halves",
                "segments": [
                    {
                        "start_s": 0,
                        "end_s": 6,
                        "label": "first",
                        "issues": ["noisy"],
                    },
                    {
                        "start_s": 6,
                        "end_s": 12,
                        "label": "second",
                        "issues": ["low_contrast"],
                    },
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
                "label": "contrast lift",
                "ops": [
                    {
                        "op": "contrast",
                        "params": {"contrast": 1.15, "brightness": 0.0, "gamma": 1.0},
                    }
                ],
            },
            {
                "id": "C",
                "label": "sharpen",
                "ops": [
                    {"op": "sharpen", "params": {"luma_amount": 0.35, "luma_size": 5}}
                ],
            },
        ],
    }
    gemini = GeminiClient(
        api_key="ci-full-flow-key",
        transport=_FakeTransport([split_payload, enhance_payload]),
    )
    controller = WizardController(tmp_path, gemini=gemini)
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        intro = _get(base)
        assert "brand-hero" in intro
        assert "Super Processor" in intro
        assert "Continue" in intro
        assert "hero-main" in intro

        llm = _post(base, "/intro", {})
        assert "What type of LLM" in llm
        assert "I don’t have a local LLM" in llm
        assert 'class="steps"' in llm
        assert "Setup" in llm

        setup = _post(base, "/llm", {"choice": "gemini"})
        assert "Gemini API setup" in setup
        assert "Google AI Studio" in setup
        assert "Paste it here" in setup

        pick = _post(base, "/setup", {"skip": "1"})
        assert "Pick a video file" in pick
        assert "1–3 minutes" in pick
        assert 'class="step active">File' in pick

        analyzing = _post(base, "/pick", {"path": str(clip)})
        assert "<h1>Analyzing</h1>" in analyzing
        assert "1–3 minutes" in analyzing
        assert controller.current_state().step is WizardStep.ANALYZING

        overview = _get(base, "/analyze?run=1")
        assert "Quick overview" in overview
        assert "gray CI clip" in overview
        assert "Enhance" in overview
        assert controller.current_state().step is WizardStep.OVERVIEW

        def _fake_upscale(
            source: Path, destination: Path, params: object, **kwargs: object
        ) -> Path:
            del params, kwargs
            destination.write_bytes(source.read_bytes())
            return destination

        monkeypatch.setattr("super_processor.wizard.run_two_pass", _fake_upscale)

        rendering = _post(base, "/overview", {})
        assert "<h1>Rendering</h1>" in rendering
        assert "Restore strength" in rendering
        assert "0.15" in rendering
        assert "MEDIUM" in rendering
        assert "5–10 minutes" in rendering
        assert 'class="step active">Enhance' in rendering
        assert controller.current_state().step is WizardStep.RENDERING

        result = _get(base, "/render?run=1")
        assert "<h1>Result</h1>" in result
        assert "A. I am happy" in result
        assert "Please tell me something else" in result
        assert "<video" in result
        assert 'class="step active">Result' in result
        assert controller.current_state().step is WizardStep.RESULT
        job_id = controller.current_state().job_id
        assert job_id
        output = controller.store.job_dir(job_id) / "output.mp4"
        assert output.is_file()
        with urlopen(base + "/output.mp4") as response:
            assert response.headers.get_content_type() == "video/mp4"
            assert len(response.read()) > 0

        done = _post(base, "/result", {"mood": "happy"})
        assert "<h1>Done</h1>" in done
        assert "output.mp4" in done
        assert controller.current_state().step is WizardStep.DONE
    finally:
        server.stop()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_wizard_website_local_llm_stub_then_gemini_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LLM B route shows deferred stub UI, then returns to Gemini setup."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        _post(base, "/intro", {})
        stub = _post(base, "/llm", {"choice": "local"})
        assert "Local LLM setup" in stub
        assert "not wired yet" in stub
        assert "Use Gemini instead" in stub
        assert controller.current_state().step is WizardStep.LOCAL_LLM_STUB

        setup = _post(base, "/local-llm", {"action": "gemini"})
        assert "Gemini API setup" in setup
        assert controller.current_state().step is WizardStep.SETUP
    finally:
        server.stop()
