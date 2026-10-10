"""Edge paths for wizard coverage."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from super_processor.wizard import (
    WizardController,
    WizardStep,
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


def _enhance_payload(label: str) -> dict[str, Any]:
    return {
        "issues": ["noisy"],
        "options": [
            {
                "id": "A",
                "label": label,
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
