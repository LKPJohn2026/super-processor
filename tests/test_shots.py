"""Loop 1: shot detection, measurement, labels, and the review actions."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from super_processor.gemini import GeminiClient, GeminiError
from super_processor.shots import (
    CONTENTS,
    ISSUES,
    Shot,
    ShotError,
    ShotList,
    ShotMetrics,
    apply_labels,
    detect_cuts,
    find_shots,
    load_shots,
    merge_with_next,
    parse_metadata_log,
    save_shots,
    shot_ranges,
    split_at,
)
from super_processor.upscale import (
    FakeUpscaleEngine,
    UpscaleRequest,
    load_upscale_plan,
    render_shots,
)
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    parse_time,
    save_session_state,
)
from super_processor.wizard_pages import format_time

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


def test_parse_metadata_log() -> None:
    log = "\n".join(
        [
            "[Parsed_metadata_3 @ 0x1] frame:0    pts:0       pts_time:0",
            "[Parsed_metadata_3 @ 0x1] lavfi.block=12.5",
            "[Parsed_metadata_3 @ 0x1] lavfi.blur=nan",
            "unrelated line pts_time:9",
            "[Parsed_metadata_3 @ 0x1] frame:1    pts:512     pts_time:0.5",
            "[Parsed_metadata_3 @ 0x1] lavfi.signalstats.YAVG=100",
            "[Parsed_metadata_3 @ 0x1] lavfi.note=text",
        ]
    )
    assert parse_metadata_log(log) == [
        (0.0, {"lavfi.block": 12.5}),
        (0.5, {"lavfi.signalstats.YAVG": 100.0}),
    ]


def test_shot_ranges_fold_short_shots_and_cap_the_count() -> None:
    assert shot_ranges([], 10) == [(0.0, 10)]
    assert shot_ranges([2.0, 5.0], 7.0) == [(0.0, 2.0), (2.0, 5.0), (5.0, 7.0)]
    # A 0.4s flash joins the shot before it; a short first shot joins the next.
    assert shot_ranges([0.5, 3.0, 3.4, 8.0], 10) == [
        (0.0, 3.4),
        (3.4, 8.0),
        (8.0, 10),
    ]
    capped = shot_ranges([float(t) for t in range(2, 40, 2)], 40, max_shots=5)
    assert len(capped) == 5
    assert capped[0][0] == 0.0 and capped[-1][1] == 40
    assert all(a[1] == b[0] for a, b in zip(capped, capped[1:], strict=False))
    with pytest.raises(ShotError):
        shot_ranges([], 0)


def test_metric_hints() -> None:
    assert ShotMetrics().hints() == ()
    dark = ShotMetrics(blockiness=30, noise=3, blur=7, brightness=0.1, contrast=0.1)
    assert dark.hints() == ("blocky", "noisy", "soft", "dark", "flat")
    assert ShotMetrics(brightness=0.9).hints() == ("overexposed",)
    assert ShotMetrics.from_dict({"noise": "x", "blur": True}) == ShotMetrics()


def _three() -> ShotList:
    return ShotList(
        shots=[
            Shot(0, 2, ShotMetrics(noise=1.0), label="a", issues=("noisy",)),
            Shot(2, 6, ShotMetrics(noise=4.0), label="b", issues=("dark",)),
            Shot(6, 7, label="c", contains=("faces",)),
        ]
    )


def test_merge_labels_and_storage(tmp_path: Path) -> None:
    merged = merge_with_next(_three(), 0)
    assert [(s.start_s, s.end_s) for s in merged.shots] == [(0, 6), (6, 7)]
    assert merged.shots[0].issues == ("noisy", "dark")
    assert merged.shots[0].metrics.noise == pytest.approx(3.0)
    with pytest.raises(ShotError):
        merge_with_next(_three(), 2)
    labelled = apply_labels(
        _three(),
        [
            {"index": 0, "label": "  a   street  ", "issues": [], "contains": ["text"]},
            {"index": 1, "label": "", "issues": ["Noisy", "bogus"]},
            {"index": 9, "label": "out of range"},
            {"index": True, "label": "bool index"},
            "not a dict",  # type: ignore[list-item]
        ],
    )
    assert labelled.shots[0].label == "a street"
    assert labelled.shots[0].issues == ()  # Gemini saw no problem.
    assert labelled.shots[0].contains == ("text",)
    assert labelled.shots[1].label == "b"
    assert labelled.shots[1].issues == ("noisy",)
    assert labelled.shots[2].label == "c"
    save_shots(tmp_path, labelled)
    loaded = load_shots(tmp_path)
    assert loaded is not None and loaded.shots == labelled.shots
    plan = loaded.to_plan()
    assert [(s.start_s, s.end_s) for s in plan.spans] == [(0, 2), (2, 6), (6, 7)]
    (tmp_path / "shots.json").write_text("{", encoding="utf-8")
    with pytest.raises(ShotError):
        load_shots(tmp_path)
    with pytest.raises(ShotError):
        ShotList(shots=[]).to_plan()


def test_times_read_and_print() -> None:
    assert parse_time("75") == 75
    assert parse_time("1:15.5") == 75.5
    assert parse_time("0:01:15") == 75
    for bad in ("", "x", "1:2:3:4", "-5"):
        with pytest.raises(ValueError):
            parse_time(bad)
    assert format_time(75) == "1:15"
    assert format_time(5.5) == "0:05.5"
    assert format_time(59.97) == "1:00"


def _cuts_clip(path: Path) -> Path:
    """2s test pattern, 3s dark noise, 2s bright pattern: cuts at 2s and 5s."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=24:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x303030:size=320x180:rate=24:duration=3",
            "-f",
            "lavfi",
            "-i",
            "mandelbrot=size=320x180:rate=24",
            "-filter_complex",
            "[1:v]noise=alls=40:allf=t[n];"
            "[2:v]trim=duration=2,setpts=PTS-STARTPTS[m];"
            "[0:v][n][m]concat=n=3:v=1[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


@needs_ffmpeg
def test_find_shots_detects_measures_and_stills(tmp_path: Path) -> None:
    clip = _cuts_clip(tmp_path / "cuts.mp4")
    cuts = detect_cuts(clip)
    assert [round(t, 1) for t in cuts] == [2.0, 5.0]
    shots = find_shots(clip, tmp_path, 7.0)
    assert [(s.start_s, s.end_s) for s in shots.shots] == [
        (0.0, 2.0),
        (2.0, 5.0),
        (5.0, 7.0),
    ]
    dark = shots.shots[1]
    assert dark.metrics.brightness < 0.25
    assert "dark" in dark.issues
    assert shots.shots[0].metrics.brightness > dark.metrics.brightness
    for shot in shots.shots:
        assert (tmp_path / shot.still).is_file()
    split = split_at(shots, 1, 3.5, source=clip, job_dir=tmp_path)
    assert [(s.start_s, s.end_s) for s in split.shots[1:3]] == [(2.0, 3.5), (3.5, 5.0)]
    assert split.shots[2].still != split.shots[1].still
    assert (tmp_path / split.shots[2].still).is_file()
    for bad in (2.1, 4.9):
        with pytest.raises(ShotError, match="split inside"):
            split_at(shots, 1, bad, source=clip, job_dir=tmp_path)


class _Transport:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.bodies: list[dict[str, Any]] = []

    def generate(
        self, *, model: str, api_key: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        self.bodies.append(body)
        text = json.dumps(self.payload)
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


def test_gemini_label_shots_sends_one_still_per_shot(tmp_path: Path) -> None:
    stills = []
    for index in range(2):
        still = tmp_path / f"{index}.jpg"
        still.write_bytes(b"\xff\xd8jpeg")
        stills.append(still)
    reply = {"shots": [{"index": 0, "label": "x", "issues": [], "contains": []}]}
    transport = _Transport(reply)
    client = GeminiClient(api_key="k", transport=transport)
    shots = [{"start_s": 0, "end_s": 1}, {"start_s": 1, "end_s": 2}]
    labels = client.label_shots(
        shots=shots, stills=stills, issues=ISSUES, contains=CONTENTS, job_dir=tmp_path
    )
    assert labels == reply["shots"]
    parts = transport.bodies[0]["contents"][0]["parts"]
    images = [part for part in parts if "inlineData" in part]
    assert len(images) == 2
    assert images[0]["inlineData"]["mimeType"] == "image/jpeg"
    assert "faces" in parts[0]["text"] and "blocky" in parts[0]["text"]
    assert (tmp_path / "gemini_chat.json").is_file()
    with pytest.raises(GeminiError, match="one still per shot"):
        client.label_shots(shots=shots, stills=stills[:1], issues=(), contains=())
    with pytest.raises(GeminiError, match="could not read"):
        client.label_shots(
            shots=shots[:1], stills=[tmp_path / "missing.jpg"], issues=(), contains=()
        )
    bad = GeminiClient(api_key="k", transport=_Transport({"nope": 1}))
    with pytest.raises(GeminiError, match="no shot labels"):
        bad.label_shots(shots=shots[:1], stills=stills[:1], issues=(), contains=())


class _Labels:
    def label_shots(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"index": i, "label": f"shot {i}", "issues": ["soft"], "contains": []}
            for i in range(len(kwargs["shots"]))
        ]

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        # The same settings for every shot.
        return {
            "scale": 2,
            "shots": [
                {"index": shot["index"], "strength": 0.5, "look": {}, "reason": "ok"}
                for shot in kwargs["shots"]
            ],
        }

    def check_previews(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"index": shot["index"], "ok": True} for shot in kwargs["shots"]]


class _CountingEngine(FakeUpscaleEngine):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def upscale(self, request: UpscaleRequest) -> Path:
        self.calls += 1
        return super().upscale(request)


@needs_ffmpeg
def test_wizard_review_actions_then_render(tmp_path: Path) -> None:
    clip = _cuts_clip(tmp_path / "cuts.mp4")
    engine = _CountingEngine()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Labels(),  # type: ignore[arg-type]
        upscale_engine=engine,
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    with pytest.raises(WizardError, match="no shot list"):
        controller.handle_post("/shots", {"action": ["approve"]})
    controller.handle_post("/pick", {"path": [str(clip)]})
    controller.run_scan()
    state = controller.current_state()
    assert state.step is WizardStep.SHOTS
    assert state.job_id
    job_dir = controller.store.job_dir(state.job_id)
    view = controller.api_view()
    assert [shot["label"] for shot in view["shots"]] == ["shot 0", "shot 1", "shot 2"]
    assert view["label_error"] is None
    assert "shot 1" in controller.render()

    controller.handle_post("/shots", {"action": ["merge"], "index": ["1"]})
    assert len(controller.api_view()["shots"]) == 2
    controller.handle_post("/shots", {"action": ["split"], "index": ["0"], "at": ["1"]})
    assert len(controller.api_view()["shots"]) == 3
    for bad in (
        {"action": ["split"], "index": ["0"], "at": ["0:00.1"]},
        {"action": ["split"], "index": ["0"], "at": ["soon"]},
        {"action": ["merge"], "index": ["x"]},
        {"action": ["merge"], "index": ["2"]},
        {"action": ["rotate"], "index": ["0"]},
    ):
        controller.handle_post("/shots", bad)
        state = controller.current_state()
        assert state.step is WizardStep.SHOTS
        assert state.error
    controller.handle_post("/shots", {"action": ["approve"]})
    assert controller.current_state().step is WizardStep.PLANNING
    controller.run_plan()
    assert controller.current_state().step is WizardStep.LOOKS
    assert engine.calls == 3  # one preview per shot
    controller.handle_post("/looks", {"action": ["approve"]})
    assert controller.current_state().step is WizardStep.RENDERING
    engine.calls = 0
    plan = load_upscale_plan(job_dir)
    assert plan is not None
    assert [(s.start_s, s.end_s) for s in plan.spans] == [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 7.0),
    ]
    controller.run_render()
    assert controller.current_state().step is WizardStep.RESULT
    # Every shot has the same settings, so they render as one part.
    assert engine.calls == 1


@needs_ffmpeg
def test_scan_failure_returns_to_pick(tmp_path: Path) -> None:
    clip = tmp_path / "not_video.mp4"
    _cuts_clip(clip)
    controller = WizardController(
        tmp_path / "jobs",
        gemini=_Labels(),  # type: ignore[arg-type]
        ffmpeg_bin=str(tmp_path / "no-ffmpeg"),
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(clip)]})
    controller.run_scan()
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error is not None and "Finding the shots failed" in state.error


@needs_ffmpeg
def test_render_shots_keeps_different_settings_apart(tmp_path: Path) -> None:
    from super_processor.look import ShotLook
    from super_processor.upscale import UpscaleSpan

    clip = _cuts_clip(tmp_path / "cuts.mp4")
    engine = _CountingEngine()
    spans = (
        UpscaleSpan(0, 2),
        UpscaleSpan(2, 5),
        UpscaleSpan(5, 7.0, look=ShotLook(denoise=0.5)),
    )
    render_shots(clip, tmp_path / "out.mp4", spans, engine=engine, duration_s=7.0)
    assert engine.calls == 2
