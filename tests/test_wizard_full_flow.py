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
from super_processor.upscale import FakeUpscaleEngine
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
    """Drive pick → fake upscale → range revise → happy over HTTP."""
    monkeypatch.setenv("GEMINI_API_KEY", "ci-full-flow-key")
    clip = _tiny_clip(tmp_path / "clip.mp4")
    gemini = GeminiClient(
        api_key="ci-full-flow-key",
        transport=_FakeTransport(
            [
                {
                    "shots": [
                        {
                            "index": 0,
                            "label": "flat gray card",
                            "issues": ["flat", "made_up"],
                            "contains": [],
                        }
                    ]
                },
                {
                    "scale": 2,
                    "shots": [
                        {
                            "index": 0,
                            "strength": 0.4,
                            "look": {
                                "deblock": 0.2,
                                "denoise": 0,
                                "contrast": 1.05,
                                "brightness": 0,
                                "saturation": 1,
                                "gamma": 1,
                                "grain": 0,
                            },
                            "reason": "Flat card; light deblock, a touch of contrast.",
                        }
                    ],
                },
                {
                    "shots": [
                        {
                            "index": 0,
                            "ok": True,
                            "problems": [],
                            "note": "Cleaner, nothing invented.",
                            "strength": 0.4,
                            "look": {
                                "deblock": 0.2,
                                "denoise": 0,
                                "contrast": 1.05,
                                "brightness": 0,
                                "saturation": 1,
                                "gamma": 1,
                                "grain": 0,
                            },
                        }
                    ]
                },
                {
                    "start_s": 2,
                    "end_s": 5,
                    "scale": 2,
                    "strength": 0.2,
                },
            ]
        ),
    )
    controller = WizardController(
        tmp_path,
        gemini=gemini,
        upscale_engine=FakeUpscaleEngine(),
    )
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        intro = _get(base)
        assert "brand-hero" in intro
        assert "Super Processor" in intro
        assert "Continue" in intro

        llm = _post(base, "/intro", {})
        assert "What type of LLM" in llm
        assert "I don’t have a local LLM" in llm

        setup = _post(base, "/llm", {"choice": "gemini"})
        assert "Gemini API setup" in setup

        pick = _post(base, "/setup", {"skip": "1"})
        assert "Pick a video file" in pick

        scanning = _post(base, "/pick", {"path": str(clip)})
        assert "Finding the shots" in scanning
        assert controller.current_state().step is WizardStep.FINDING_SHOTS
        _get(base, "/shots?run=1")
        assert controller.wait_idle(timeout=60)
        shots = _get(base)
        assert "<h1>Shots</h1>" in shots
        assert "flat gray card" in shots
        assert "Problems: flat" in shots
        assert "made_up" not in shots

        planning = _post(base, "/shots", {"action": "approve"})
        assert "Planning each shot" in planning
        _get(base, "/looks?run=1")
        assert controller.wait_idle(timeout=60)
        looks = _get(base)
        assert "<h1>Looks</h1>" in looks
        assert "strength 0.40 · deblock 0.20 · contrast 1.05" in looks
        assert "Flat card; light deblock" in looks
        assert "Check passed. Cleaner, nothing invented." in looks
        assert 'src="/previews/shot_000_r1_after.mp4"' in looks
        assert 'poster="/previews/shot_000_r1_after.jpg"' in looks
        with urlopen(base + "/previews/shot_000_r1_after.jpg") as response:
            assert response.headers.get_content_type() == "image/jpeg"

        waiting = _post(base, "/looks", {"action": "approve"})
        assert "<h1>Upscaling</h1>" in waiting
        assert controller.current_state().step is WizardStep.RENDERING

        started = _get(base, "/render?run=1")
        assert "<h1>Upscaling</h1>" in started
        assert controller.wait_idle(timeout=60)
        result = _get(base)
        assert "<h1>Result</h1>" in result
        assert "A. I am happy" in result
        assert "Scale 2" in result
        assert "strength 0.40" in result
        assert controller.current_state().step is WizardStep.RESULT
        job_id = controller.current_state().job_id
        assert job_id
        output = controller.store.job_dir(job_id) / "output.mp4"
        assert output.is_file()
        with urlopen(base + "/output.mp4") as response:
            assert response.headers.get_content_type() == "video/mp4"
            assert len(response.read()) > 0

        revising = _post(
            base,
            "/result",
            {"note": "reduce artificial detail from 2s to 5s"},
        )
        assert "<h1>Upscaling</h1>" in revising
        assert controller.current_state().step is WizardStep.RENDERING

        _get(base, "/render?run=1")
        assert controller.wait_idle(timeout=60)
        revised = _get(base)
        assert "<h1>Result</h1>" in revised
        assert "strength 0.20" in revised
        assert (controller.store.job_dir(job_id) / "output.mp4").is_file()

        done = _post(base, "/result", {"mood": "happy"})
        assert "<h1>Done</h1>" in done
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
