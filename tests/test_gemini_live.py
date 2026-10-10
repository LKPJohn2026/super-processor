"""Live Gemini API checks using repository secret ``Gemini_API_Test``.

CI maps ``secrets.Gemini_API_Test`` → ``GEMINI_API_KEY`` in the
``gemini-live`` job only. Unit-test matrix jobs never receive the key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from super_processor.gemini import (
    GeminiClient,
    GeminiError,
    model_accepts_text_or_video,
    model_has_usable_rpm,
)
from super_processor.upscale import UpscaleSpan

pytestmark = [
    pytest.mark.gemini_live,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set (CI: secrets.Gemini_API_Test)",
    ),
]


def test_live_gemini_lists_multimodal_rpm_candidates() -> None:
    """Discover generateContent models that accept text/video with RPM != 0."""
    client = GeminiClient()
    infos = client.list_model_infos()
    assert infos, "models.list returned no models"
    usable = [
        info
        for info in infos
        if model_accepts_text_or_video(info) and model_has_usable_rpm(info)
    ]
    assert usable, "no multimodal models with usable RPM (limit != 0)"
    candidates = client.candidate_models(refresh=True)
    assert candidates


def test_live_gemini_revise_upscale_params(tmp_path: Path) -> None:
    """A result note becomes a bounded range, scale, and strength."""
    client = GeminiClient()
    try:
        data = client.revise_upscale_params(
            note="the faces from 4s to 9s look waxy, use less artificial detail",
            prior_scale=2,
            prior_strength=0.5,
            prior_start_s=0.0,
            prior_end_s=20.0,
            duration_s=20.0,
            job_dir=tmp_path,
        )
    except GeminiError as exc:
        pytest.fail(f"live Gemini upscale note failed: {exc}")
    span = UpscaleSpan.from_dict(data)
    assert span.scale in {2, 4}
    assert 0.0 <= span.strength < 0.5
    assert 0.0 <= span.start_s < span.end_s <= 20.0
    assert (tmp_path / "gemini_chat.json").is_file()
