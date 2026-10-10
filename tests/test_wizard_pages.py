"""Cover wizard HTML helpers."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from super_processor.review import WizardServer
from super_processor.wizard_pages import (
    render_done,
    render_llm_choice,
    render_local_llm_stub,
    render_pick,
    render_rendering,
    render_result,
    render_setup,
)


def test_render_helpers_smoke() -> None:
    assert "local LLM" in render_llm_choice()
    assert (
        "coming later" in render_local_llm_stub().lower()
        or "Gemini" in render_local_llm_stub()
    )
    assert "Gemini" in render_setup(has_key=False)
    assert "existing" in render_setup(has_key=True).lower()
    assert "Pick" in render_pick()
    assert "1080p and 30 minutes" in render_pick()
    assert "Upscaling" in render_rendering()
    assert "Done" in render_done()
    assert "happy" in render_result(output_url="/output.mp4").lower()


def test_step_rail_has_no_split_or_enhance() -> None:
    page = render_rendering()
    for label in ("Setup", "File", "Upscale", "Result"):
        assert f">{label}</span>" in page
    assert ">Split<" not in page
    assert ">Enhance<" not in page
    assert '<span class="step active">Upscale</span>' in page


def test_wizard_server_post_intro(tmp_path: Path) -> None:
    server = WizardServer(tmp_path)
    base = server.start()
    try:
        req = Request(
            base + "/intro",
            data=urlencode({}).encode(),
            method="POST",
        )
        with urlopen(req) as response:
            # 303 redirects; urlopen follows to /
            body = response.read().decode()
        assert "Gemini" in body or "setup" in body.lower() or "API" in body
    finally:
        server.stop()
