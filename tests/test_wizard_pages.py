"""Cover wizard HTML helpers and deeper controller paths."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from super_processor.gemini import (
    EnhanceOption,
    GeminiClient,
    SegmentEnhanceResult,
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
    render_overview,
    render_pick,
    render_rendering,
    render_result,
    render_setup,
    render_split_choice,
)


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
    assert "Gemini" in render_setup(has_key=False)
    assert "existing" in render_setup(has_key=True).lower()
    assert "Pick" in render_pick()
    assert "Analyzing" in render_analyzing()
    assert "Rendering" in render_rendering()
    assert "Done" in render_done()
    assert "happy" in render_result(output_url="/output.mp4")
    assert "Duration" in render_overview(facts=_facts(), highlights=["one"])
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
    assert "Option 1" in render_split_choice(proposal=proposal)
    segment = TimelineSegment(
        0, 0.0, 12.0, "mixed", "noisy", 6.0, 0, "segment_stills/a.ppm"
    )
    result = SegmentEnhanceResult.from_dict(
        {
            "issues": ["noisy"],
            "options": [
                {
                    "id": "A",
                    "label": "denoise",
                    "ops": [{"op": "denoise", "params": {"strength": 0.2}}],
                },
                {
                    "id": "B",
                    "label": "contrast",
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
                    "label": "sharpen",
                    "ops": [
                        {
                            "op": "sharpen",
                            "params": {"luma_amount": 0.3, "luma_size": 5},
                        }
                    ],
                },
            ],
        }
    )
    html = render_enhance(
        segment=segment,
        result=result,
        preview_url="/previews/x.mp4",
    )
    assert "Accept" in html


@pytest.mark.skipif(
    subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0,
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
    subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0,
    reason="ffmpeg required",
)
def test_wizard_enhance_preview_accept_and_render(
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
    enhance = {
        "issues": ["noisy"],
        "options": [
            {
                "id": "A",
                "label": "denoise",
                "ops": [{"op": "denoise", "params": {"strength": 0.25}}],
            },
            {
                "id": "B",
                "label": "contrast",
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
    # Enough enhance payloads for two segments + any revise.
    payloads = [enhance, enhance, enhance, enhance]
    gemini = GeminiClient(api_key="k", transport=_FakeTransport(payloads))
    controller = WizardController(tmp_path, gemini=gemini)
    from super_processor.jobs import JobState

    manifest = controller.store.create(clip)
    controller.store.transition(manifest.job_id, JobState.PROBED)
    controller.store.transition(manifest.job_id, JobState.SPLIT_PROPOSED)
    job_dir = controller.store.job_dir(manifest.job_id)
    proposal = SplitProposal.from_dict(
        {
            "highlights": ["gray"],
            "layouts": [
                {
                    "segment_count": 2,
                    "summary": "two",
                    "segments": [
                        {
                            "start_s": 0,
                            "end_s": 6,
                            "label": "a",
                            "issues": ["noisy"],
                        },
                        {
                            "start_s": 6,
                            "end_s": 12,
                            "label": "b",
                            "issues": ["noisy"],
                        },
                    ],
                }
            ],
        }
    )
    write_split_layouts(job_dir, proposal)
    state = WizardState(
        step=WizardStep.CHOOSE_SPLIT,
        job_id=manifest.job_id,
        highlights=["gray"],
    )
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)

    controller.handle_post("/split", {"layout": ["0"]})
    assert controller.current_state().step is WizardStep.ENHANCE
    controller.handle_post("/segment", {"option": ["A"]})
    state = controller.current_state()
    assert (
        f"{state.segment_index}:preview" in state.enhance_cache
        or "0:preview" in state.enhance_cache
    )
    controller.handle_post("/segment", {"accept": ["1"]})
    assert controller.current_state().segment_index == 1
    controller.handle_post("/segment", {"option": ["B"]})
    controller.handle_post("/segment", {"accept": ["1"]})
    assert controller.current_state().step is WizardStep.RENDERING
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
