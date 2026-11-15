"""The wizard's HTML pages: what each one offers, and that text stays text."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from super_processor.review import WizardServer
from super_processor.wizard import WizardController, WizardStep
from super_processor.wizard_pages import (
    render_done,
    render_llm_choice,
    render_local_llm_stub,
    render_pick,
    render_preview_segment,
    render_ready,
    render_rendering,
    render_result,
    render_segments,
    render_setup,
)

TOKEN = "test-token"


def test_each_page_offers_its_own_actions() -> None:
    llm = render_llm_choice()
    assert 'name="choice" value="gemini"' in llm
    assert 'name="choice" value="local"' in llm
    stub = render_local_llm_stub()
    assert "not wired yet" in stub and 'action="/local-llm"' in stub
    # Without a key there is only the paste form; with one, a skip as well.
    fresh = render_setup(has_key=False)
    assert 'name="api_key"' in fresh and 'name="skip"' not in fresh
    saved = render_setup(has_key=True, key_source="/cfg/credentials.json")
    assert 'name="skip" value="1"' in saved
    assert "it comes from /cfg/credentials.json" in saved
    pick = render_pick()
    assert 'action="/browse"' in pick and 'action="/pick"' in pick
    assert 'id="dropzone"' in pick and "Up to 1080p and 30 minutes." in pick
    assert 'content="2;url=/render?run=1"' in render_rendering()
    result = render_result(output_url="/output.mp4?v=7")
    assert '<video controls src="/output.mp4?v=7">' in result
    assert 'name="mood" value="happy"' in result
    assert 'name="restart" value="1"' in result
    assert 'action="/new"' in result and 'action="/new"' in render_done()


def test_pick_page_states_only_the_real_input_limit() -> None:
    assert "1280 px" not in render_pick()


def test_pick_page_limit_matches_the_wizard_input_check() -> None:
    # The page's one stated limit must be the one the wizard enforces.
    from super_processor.segments import MAX_DURATION_S
    from super_processor.wizard import MAX_INPUT_LONG_EDGE, MAX_INPUT_SHORT_EDGE

    assert (MAX_INPUT_LONG_EDGE, MAX_INPUT_SHORT_EDGE) == (1920, 1080)
    assert MAX_DURATION_S == 30 * 60
    page = render_pick()
    assert page.count(" px") == 0
    assert "Up to 1080p and 30 minutes." in page


def test_step_rail_has_no_split_enhance_or_upscale() -> None:
    page = render_rendering()
    for label in ("Setup", "File", "Segments", "Previews", "Result"):
        assert f">{label}</span>" in page
    for gone in (">Split<", ">Enhance<", ">Upscale<"):
        assert gone not in page
    assert '<span class="step active" data-phase="result">Result</span>' in page
    assert '<span class="step done" data-phase="previews">Previews</span>' in page


def test_errors_and_model_text_are_escaped_on_every_page() -> None:
    # Errors carry file names, and labels and reasons come from Gemini: both
    # must reach the page as text, never as markup.
    bad = '<img src=x onerror="alert(1)">'
    shot = {
        "index": 0,
        "start_s": 0,
        "end_s": 4,
        "label": bad,
        "reason": bad,
        "strength": 0.4,
        "look": {},
        "check": {"ok": False, "note": bad},
        "after_url": "/previews/shot_000_r1_after.mp4",
        "before_url": "/previews/shot_000_r1_before.mp4",
    }
    pages = [
        render_pick(error=bad),
        render_setup(has_key=False, error=bad),
        render_llm_choice(error=bad),
        render_segments(
            {"segments": [{"start_s": 0, "end_s": 4, "label": bad}], "error": bad},
            error=bad,
        ),
        render_preview_segment(
            {"index": 0, "count": 1, "scale": 2, "error": bad, "shot": shot},
            error=bad,
        ),
        render_ready({"shots": [shot], "scales": [1, 2], "scale": 2}, error=bad),
        render_result(output_url="/output.mp4", error=bad, scale=2, strength=bad),
    ]
    for page in pages:
        assert bad not in page
        assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in page


def test_ready_page_offers_both_sizes_with_the_plan_scale_chosen() -> None:
    view = {"shots": [], "scales": [1, 2], "scale": 1, "source_size": [640, 360]}
    page = render_ready(view)
    assert '<input type="radio" name="scale" value="1" checked>' in page
    assert '<input type="radio" name="scale" value="2">' in page
    assert "Upscale 2×</strong> · 1280×720" in page
    assert "Don't upscale</strong> · 640×360" in page


def test_wizard_server_post_intro_with_the_form_token(tmp_path: Path) -> None:
    # A plain HTML form sends the token as a field, not a header.
    controller = WizardController(tmp_path)
    server = WizardServer(tmp_path, token=TOKEN, controller=controller)
    base = server.start()
    try:
        req = Request(
            base + "/intro",
            data=urlencode({"token": TOKEN}).encode(),
            method="POST",
        )
        with urlopen(req) as response:
            # 303 redirects; urlopen follows to /
            body = response.read().decode()
        assert "<h1>What type of LLM do you have now?</h1>" in body
        assert controller.current_state().step is WizardStep.LLM_CHOICE
    finally:
        server.stop()
