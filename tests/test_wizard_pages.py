"""Cover wizard HTML helpers and deeper controller paths."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from super_processor.gemini import (
    EnhanceOption,
    SplitProposal,
    StructuredOp,
)
from super_processor.probe import MediaFacts, StreamFacts
from super_processor.review import WizardServer
from super_processor.segments import TimelineSegment, write_gray_still, write_segments
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    extract_wizard_frames,
    load_split_layouts,
    save_job_wizard_state,
    save_segment_choice,
    save_session_state,
    write_split_layouts,
)
from super_processor.wizard_pages import (
    render_analyzing,
    render_done,
    render_enhance,
    render_llm_choice,
    render_local_llm_stub,
    render_overview,
    render_pick,
    render_rendering,
    render_result,
    render_setup,
    render_split_choice,
)


def _facts() -> MediaFacts:
    return MediaFacts(
        schema_version=1,
        source_path="/tmp/x.mp4",
        format_name="mp4",
        format_long_name="MP4",
        duration_s=12.0,
        size_bytes=1000,
        bit_rate=1000,
        streams=[
            StreamFacts(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=160,
                height=90,
                avg_frame_rate="10/1",
            )
        ],
        has_video=True,
        has_audio=False,
        is_vfr=False,
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
    assert "Analyzing" in render_analyzing()
    assert "Rendering" in render_rendering()
    assert "Done" in render_done()
    assert "happy" in render_result(output_url="/output.mp4").lower()
    assert "Duration" in render_overview(facts=_facts(), highlights=["one"])
    assert "Setup" in render_pick()  # step rail
    proposal = SplitProposal.from_dict(
        {
            "highlights": [],
            "layouts": [
                {
                    "segment_count": 1,
                    "summary": "all",
                    "segments": [
                        {
                            "start_s": 0,
                            "end_s": 12,
                            "label": "all",
                            "issues": ["noisy"],
                        }
                    ],
                }
            ],
        }
    )
    assert "A. 1 segment" in render_split_choice(proposal=proposal)
    html = render_enhance(restore_strength=0.15, scale=2, vsr_quality="MEDIUM")
    assert "Restore strength" in html
    assert "0.15" in html
    assert "MEDIUM" in html
    assert "Setup" in html
    assert "Split" not in html


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg required",
)
def test_extract_wizard_frames(tmp_path: Path) -> None:
    clip = tmp_path / "c.mp4"
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
            "color=c=red:duration=3:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    paths = extract_wizard_frames(clip, tmp_path, duration_s=3.0)
    assert paths
    assert paths[0].suffix == ".jpg"


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg required",
)
def test_wizard_overview_renders_two_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    clip = tmp_path / "clip.mp4"
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
            "color=c=gray:duration=12:size=160x90:rate=10",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=12",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(clip),
        ],
        check=True,
    )
    controller = WizardController(tmp_path)
    from super_processor.jobs import JobState

    manifest = controller.store.create(clip)
    controller.store.transition(manifest.job_id, JobState.PROBED)
    controller.store.transition(manifest.job_id, JobState.SPLIT_PROPOSED)
    job_dir = controller.store.job_dir(manifest.job_id)
    state = WizardState(
        step=WizardStep.OVERVIEW,
        job_id=manifest.job_id,
        highlights=["gray"],
    )
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)

    controller.handle_post("/overview", {})
    assert controller.current_state().step is WizardStep.RENDERING
    page = controller.render()
    assert "Restore strength" in page
    assert "0.15" in page
    assert "MEDIUM" in page
    assert 'class="step active">Enhance' in page

    def _fake_upscale(
        source: Path,
        destination: Path,
        params: object,
        **kwargs: object,
    ) -> Path:
        del source, params, kwargs
        destination.write_bytes(b"\x00\x00\x00\x18ftyp")
        return destination

    monkeypatch.setattr("super_processor.wizard.run_two_pass", _fake_upscale)
    controller.run_render()
    assert controller.current_state().step is WizardStep.RESULT
    assert (job_dir / "output.mp4").is_file()
    controller.handle_post("/result", {"mood": ["happy"]})
    assert controller.current_state().step is WizardStep.DONE
    html = controller.render()
    assert "Done" in html


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


def test_save_and_load_layouts(tmp_path: Path) -> None:
    proposal = SplitProposal.from_dict(
        {
            "highlights": ["h"],
            "layouts": [
                {
                    "segment_count": 1,
                    "summary": "s",
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
    )
    write_split_layouts(tmp_path, proposal)
    loaded = load_split_layouts(tmp_path)
    assert loaded.highlights == ["h"]
    option = EnhanceOption(
        id="A",
        label="lift",
        ops=[
            StructuredOp("contrast", {"contrast": 1.2, "brightness": 0.0, "gamma": 1.0})
        ],
    )
    save_segment_choice(tmp_path, 0, option)
    still = tmp_path / "segment_stills" / "seg_00.ppm"
    write_gray_still(still, bytes([40]) * (160 * 90))
    write_segments(
        tmp_path,
        [
            TimelineSegment(
                0, 0.0, 10.0, "mixed", "noisy", 5.0, 0, "segment_stills/seg_00.ppm"
            )
        ],
    )
