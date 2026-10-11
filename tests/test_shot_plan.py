"""Loop 2: per-shot recipes, previews, the self-check, and editor redo."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from super_processor.gemini import GeminiClient, GeminiError
from super_processor.look import LOOK_BOUNDS, ShotLook
from super_processor.shot_plan import (
    CHECK_PROBLEMS,
    PROTECTED_STRENGTH,
    ShotCheck,
    ShotPlan,
    ShotPlanError,
    ShotRecipe,
    allowed_scales,
    apply_checks,
    apply_proposals,
    fallback_recipe,
    finish,
    load_plan,
    look_from_model,
    new_plan,
    preview_window,
    render_previews,
    request_redo,
    save_plan,
    shots_payload,
)
from super_processor.shots import Shot, ShotList, ShotMetrics
from super_processor.upscale import (
    FakeUpscaleEngine,
    FlashVsrEngine,
    UpscaleError,
    UpscaleRequest,
    load_upscale_plan,
    video_stream_facts,
)
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_session_state,
)
from super_processor.wizard_pages import check_summary, settings_summary

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


def _shots() -> ShotList:
    return ShotList(
        shots=[
            Shot(0, 2, ShotMetrics(), label="street", issues=("blocky", "noisy")),
            Shot(2, 6, label="interview", issues=("dark",), contains=("faces",)),
            Shot(6, 7, label="sign", issues=("soft",), contains=("text",)),
        ]
    )


def test_model_looks_are_clamped() -> None:
    look = look_from_model(
        {"deblock": 3, "contrast": 0.1, "grain": "x", "gamma": math.nan, "sharpen": 1}
    )
    assert look == ShotLook(deblock=1.0, contrast=LOOK_BOUNDS["contrast"][0])
    assert look_from_model(None) == ShotLook()
    assert allowed_scales(960, 540) == (2, 4)
    assert allowed_scales(540, 960) == (2, 4)
    assert allowed_scales(1280, 720) == (2,)


def test_fallback_recipes_follow_the_measurements() -> None:
    street, interview, sign = _shots().shots
    damaged = fallback_recipe(street)
    assert damaged.look.deblock > 0 and damaged.look.denoise > 0
    assert damaged.strength == 0.45
    assert "blocky" in damaged.reason
    dark = fallback_recipe(interview)
    assert dark.look.brightness > 0 and dark.look.gamma > 1
    soft = fallback_recipe(sign)
    assert soft.strength == 0.6
    plain = fallback_recipe(Shot(0, 1))
    assert plain.look.is_neutral() and "no clear problem" in plain.reason


def test_proposals_are_merged_capped_and_filled_in() -> None:
    shots = _shots()
    plan = new_plan(shots, 2)
    merged = apply_proposals(
        plan,
        shots,
        [
            {
                "index": 0,
                "strength": 0.9,
                "look": {"deblock": 0.4},
                "reason": "  a  b ",
            },
            {"index": 1, "strength": 0.95, "look": {}, "reason": "faces"},
            {"index": True, "strength": 0.1},
            "junk",  # type: ignore[list-item]
        ],
        [0, 1, 2],
    )
    assert merged.recipes[0].strength == 0.9
    assert merged.recipes[0].look.deblock == 0.4
    assert merged.recipes[0].reason == "a b"
    assert merged.recipes[1].strength == PROTECTED_STRENGTH
    assert "measurements" in merged.recipes[2].reason
    assert merged.error is not None and "shot 3" in merged.error
    upscale = merged.to_upscale_plan(shots)
    assert [span.strength for span in upscale.spans][:2] == [0.9, PROTECTED_STRENGTH]
    with pytest.raises(ShotPlanError, match="no longer match"):
        merged.to_upscale_plan(ShotList(shots=shots.shots[:1]))


def test_checks_fix_only_failed_shots() -> None:
    shots = _shots()
    plan = ShotPlan(scale=2, recipes=[ShotRecipe(strength=0.5) for _ in range(3)])
    checked, changed = apply_checks(
        plan,
        shots,
        [
            {"index": 0, "ok": True, "problems": [], "note": "fine"},
            {
                "index": 1,
                "ok": False,
                "problems": ["waxy_skin", "made_up"],
                "note": "skin",
                "strength": 0.3,
                "look": {"denoise": 0.2},
            },
            {"index": 2, "ok": False, "problems": ["halos"], "strength": 0.5},
        ],
        [0, 1, 2],
    )
    assert changed == [1]
    assert checked.recipes[0].check == ShotCheck(ok=True, note="fine")
    assert checked.recipes[1].check.adjusted is True
    assert checked.recipes[2].check.adjusted is False
    assert checked.recipes[1].strength == 0.3
    assert checked.recipes[1].look.denoise == 0.2
    assert checked.recipes[1].check.problems == ("waxy_skin",)
    # Same settings back: nothing to redo, but the problem is still shown.
    assert checked.recipes[2].check.problems == ("halos",)


def test_redo_marks_one_shot_and_storage_round_trips(tmp_path: Path) -> None:
    shots = _shots()
    plan = finish(new_plan(shots, 4), [0, 1, 2])
    assert plan.stale_indexes() == []
    with pytest.raises(ShotPlanError, match="say what"):
        request_redo(plan, 1, "   ")
    with pytest.raises(ShotPlanError, match="no such"):
        request_redo(plan, 7, "softer")
    redo = request_redo(plan, 1, "  skin   too smooth ")
    assert redo.stale_indexes() == [1]
    assert redo.recipes[1].note == "skin too smooth"
    payload = shots_payload(shots, redo, [1])
    assert payload[0]["editor_note"] == "skin too smooth"
    assert payload[0]["contains"] == ["faces"]
    save_plan(tmp_path, redo)
    loaded = load_plan(tmp_path)
    assert loaded is not None
    assert loaded.to_dict() == redo.to_dict()
    assert finish(loaded, [1]).recipes[1].note == ""
    (tmp_path / "shot_plan.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ShotPlanError):
        load_plan(tmp_path)
    assert load_plan(tmp_path / "missing") is None


def test_preview_window_and_summaries() -> None:
    assert preview_window(Shot(0, 2)) == (0, 2)
    assert preview_window(Shot(10, 20)) == (13.5, 16.5)
    assert settings_summary(0.4, {"deblock": 0.2, "gamma": 1.0}) == (
        "strength 0.40 · deblock 0.20"
    )
    assert check_summary({"ok": None}) == "Not checked."
    assert check_summary({"ok": True, "note": "Clean."}) == "Check passed. Clean."
    failed = check_summary({"ok": False, "problems": ["garbled_text"], "note": "Sign."})
    assert failed == "Check found: garbled text. Sign."
    adjusted = check_summary(
        {"ok": False, "problems": [], "note": "soft", "adjusted": True}
    )
    assert adjusted == (
        "Check found: a problem. soft. Settings were adjusted and the preview redone."
    )


def _clip(path: Path, seconds: int = 6) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=160x90:rate=10:duration={seconds}",
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
def test_previews_are_versioned_and_replaced(tmp_path: Path) -> None:
    source = _clip(tmp_path / "in.mp4")
    shots = ShotList(shots=[Shot(0, 2), Shot(2, 6)])
    plan = new_plan(shots, 2)
    engine = FakeUpscaleEngine()
    plan = render_previews(source, tmp_path, shots, plan, [0, 1], engine=engine)
    folder = tmp_path / "previews"
    assert (folder / "shot_001_r1_after.mp4").is_file()
    assert (folder / "shot_001_r1_before.jpg").is_file()
    assert video_stream_facts(folder / "shot_001_r1_after.mp4")[0] == 30
    assert video_stream_facts(folder / "shot_000_r1_after.mp4")[0] == 20
    plan = render_previews(source, tmp_path, shots, plan, [1], engine=engine)
    assert plan.recipes[1].rev == 2 and plan.recipes[0].rev == 1
    assert not (folder / "shot_001_r1_after.mp4").exists()
    assert not (folder / "shot_001_r1_after.mp4.delivery.json").exists()
    assert (folder / "shot_001_r2_after.mp4.delivery.json").is_file()
    assert (folder / "shot_001_r2_after.mp4").is_file()
    assert (folder / "shot_000_r1_after.mp4").is_file()


def _three_shots(path: Path) -> Path:
    """Three 2s shots of different pictures, so scdet finds two cuts."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x303030:size=320x180:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "mandelbrot=size=320x180:rate=10",
            "-filter_complex",
            "[2:v]trim=duration=2,setpts=PTS-STARTPTS[m];"
            "[0:v][1:v][m]concat=n=3:v=1[v]",
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


def _flashvsr_tree(tmp_path: Path, script: str) -> tuple[Path, Path]:
    home = tmp_path / "FlashVSR"
    wan = home / "examples" / "WanVSR"
    weights = wan / "FlashVSR-v1.1"
    weights.mkdir(parents=True)
    for name in (
        "diffusion_pytorch_model_streaming_dmd.safetensors",
        "LQ_proj_in.ckpt",
        "TCDecoder.ckpt",
    ):
        (weights / name).write_bytes(b"w")
    infer = wan / "infer_flashvsr_v1.1_tiny.py"
    infer.write_text(script, encoding="utf-8")
    return home, infer


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


def test_gemini_propose_and_check_requests(tmp_path: Path) -> None:
    jpg = tmp_path / "s.jpg"
    jpg.write_bytes(b"\xff\xd8")
    reply = {"scale": 2, "shots": [{"index": 0, "strength": 0.4, "look": {}}]}
    transport = _Transport(reply)
    client = GeminiClient(api_key="k", transport=transport)
    out = client.propose_looks(
        shots=[{"index": 0}],
        stills=[jpg],
        bounds=LOOK_BOUNDS,
        scales=(2, 4),
        protected_strength=0.6,
        job_dir=tmp_path,
    )
    assert out == reply
    body = transport.bodies[0]
    text = body["contents"][0]["parts"][0]["text"]
    assert "deblock 0 to 1" in text and "[2, 4]" in text and "0.6" in text
    schema = body["generationConfig"]["responseSchema"]
    assert (
        "grain"
        in schema["properties"]["shots"]["items"]["properties"]["look"]["properties"]
    )
    with pytest.raises(GeminiError, match="one still per shot"):
        client.propose_looks(
            shots=[{}, {}],
            stills=[jpg],
            bounds=LOOK_BOUNDS,
            scales=(2,),
            protected_strength=0.6,
        )
    checker = GeminiClient(
        api_key="k", transport=_Transport({"shots": [{"index": 0, "ok": True}, 3]})
    )
    verdicts = checker.check_previews(
        shots=[{"index": 0}], pairs=[(jpg, jpg)], problems=CHECK_PROBLEMS
    )
    assert verdicts == [{"index": 0, "ok": True}]
    with pytest.raises(GeminiError, match="one still pair"):
        checker.check_previews(shots=[{}], pairs=[], problems=())
    empty = GeminiClient(api_key="k", transport=_Transport({"nope": 1}))
    with pytest.raises(GeminiError, match="no shot recipes"):
        empty.propose_looks(
            shots=[{}], stills=[jpg], bounds={}, scales=(2,), protected_strength=0.6
        )
    with pytest.raises(GeminiError, match="no preview checks"):
        empty.check_previews(shots=[{}], pairs=[(jpg, jpg)], problems=())


class _Planner:
    """Labels, then plans; fails the check on shot 1 once."""

    def __init__(self) -> None:
        self.proposals: list[list[dict[str, Any]]] = []
        self.scales: list[tuple[int, ...]] = []
        self.checks = 0

    def label_shots(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"index": i, "label": f"shot {i}", "issues": [], "contains": []}
            for i in range(len(kwargs["shots"]))
        ]

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        self.proposals.append(kwargs["shots"])
        self.scales.append(kwargs["scales"])
        return {
            "scale": 4,
            "shots": [
                {
                    "index": shot["index"],
                    "strength": 0.3 if "editor_note" in shot else 0.7,
                    "look": {"denoise": 0.1 * shot["index"]},
                    "reason": f"plan {shot['index']}",
                }
                for shot in kwargs["shots"]
            ],
        }

    def check_previews(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.checks += 1
        out = []
        for shot in kwargs["shots"]:
            bad = shot["index"] == 1 and self.checks == 1
            out.append(
                {
                    "index": shot["index"],
                    "ok": not bad,
                    "problems": ["fake_texture"] if bad else [],
                    "note": "grain turned to worms" if bad else "",
                    "strength": 0.4 if bad else shot["strength"],
                    "look": shot["look"],
                }
            )
        for before, after in kwargs["pairs"]:
            assert before.is_file() and after.is_file()
        return out


@needs_ffmpeg
def test_wizard_plans_checks_and_redoes_one_shot(tmp_path: Path) -> None:
    clip = _three_shots(tmp_path / "cuts.mp4")  # 320x180: 4x is allowed
    planner = _Planner()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=planner,  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(clip)]})
    controller.run_scan()
    controller.handle_post("/shots", {"action": ["approve"]})
    assert controller.start_plan() is True
    assert controller.wait_idle(timeout=60)
    state = controller.current_state()
    assert state.step is WizardStep.LOOKS, state.error
    assert planner.scales == [(2, 4)]
    looks = controller.api_view()["looks"]
    assert looks["scale"] == 4
    assert looks["error"] is None
    first, second, third = looks["shots"]
    assert first["strength"] == 0.7 and first["check"]["ok"] is True
    # The check failed shot 2: its settings changed and its preview was redone.
    assert second["strength"] == 0.4
    assert second["check"]["problems"] == ["fake_texture"]
    assert second["after_url"].endswith("shot_001_r2_after.mp4")
    assert third["after_url"].endswith("shot_002_r1_after.mp4")
    page = controller.render()
    assert "Check found: fake texture. grain turned to worms." in page
    assert "upscaled 4×" in page

    controller.handle_post("/looks", {"action": ["redo"], "index": ["2"], "note": [""]})
    assert controller.current_state().error
    assert controller.current_state().step is WizardStep.LOOKS
    controller.handle_post(
        "/looks", {"action": ["redo"], "index": ["2"], "note": ["keep the sign crisp"]}
    )
    assert controller.current_state().step is WizardStep.PLANNING
    controller.run_plan()
    assert controller.current_state().step is WizardStep.LOOKS
    assert planner.scales[-1] == (4,)
    assert [shot["index"] for shot in planner.proposals[-1]] == [2]
    assert planner.proposals[-1][0]["editor_note"] == "keep the sign crisp"
    after = controller.api_view()["looks"]["shots"]
    assert after[2]["strength"] == 0.3
    assert after[2]["after_url"].endswith("shot_002_r2_after.mp4")
    assert after[0]["after_url"].endswith("shot_000_r1_after.mp4")

    controller.handle_post("/looks", {"action": ["dance"]})
    assert "unknown action" in (controller.current_state().error or "")
    controller.handle_post("/looks", {"action": ["approve"]})
    assert controller.current_state().step is WizardStep.RENDERING
    state = controller.current_state()
    assert state.job_id
    plan = load_upscale_plan(controller.store.job_dir(state.job_id))
    assert plan is not None
    assert [span.scale for span in plan.spans] == [4, 4, 4]
    assert [span.strength for span in plan.spans] == [0.7, 0.4, 0.3]
    assert plan.spans[2].look.denoise == pytest.approx(0.2)


def test_looks_post_needs_the_looks_step(tmp_path: Path) -> None:
    from super_processor.wizard import WizardError

    controller = WizardController(tmp_path)
    with pytest.raises(WizardError, match="no shot settings"):
        controller.handle_post("/looks", {"action": ["approve"]})


_COUNTING_INFER = """
import os
import subprocess
from pathlib import Path

LOADS = Path(os.environ["FAKE_LOADS"])


class _Torch:
    bfloat16 = "bf16"


torch = _Torch()
_SOURCE = {}


def init_pipeline():
    LOADS.write_text(str(int(LOADS.read_text() or "0") + 1))
    return lambda **kwargs: kwargs["LQ_video"]


def prepare_input_tensor(path, scale, dtype, device):
    _SOURCE["path"] = path
    _SOURCE["scale"] = int(scale)
    return ("lq", 8, 8, 1, 10)


def tensor2video(video):
    return [video]


def save_video(pictures, save_path, fps, quality):
    scale = _SOURCE["scale"]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", _SOURCE["path"], "-vf", f"scale=iw*{scale}:ih*{scale}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", save_path],
        check=True,
    )
"""


@needs_ffmpeg
def test_flashvsr_batches_previews_with_one_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, _script = _flashvsr_tree(tmp_path, _COUNTING_INFER)
    loads = tmp_path / "loads.txt"
    loads.write_text("0")
    monkeypatch.setenv("FAKE_LOADS", str(loads))
    monkeypatch.setattr("super_processor.upscale.cuda_available", lambda: True)
    monkeypatch.setattr("super_processor.upscale.flashvsr_home", lambda: home)
    clips = [_clip(tmp_path / f"c{i}.mp4", seconds=2) for i in range(3)]
    requests = [
        UpscaleRequest(
            clip, tmp_path / "out" / f"{clip.stem}.mp4", look=ShotLook(denoise=0.5)
        )
        for clip in clips
    ]
    engine = FlashVsrEngine()
    outputs = engine.upscale_many(requests)
    assert loads.read_text() == "1"
    assert all(video_stream_facts(out)[0] == 20 for out in outputs)
    assert (tmp_path / "out" / "c0.clean.mp4").is_file()
    assert engine.upscale_many([]) == []
    assert str(home) not in sys.path
    long = _clip(tmp_path / "long.mp4", seconds=10)
    with pytest.raises(UpscaleError, match="at most"):
        engine.upscale_many([UpscaleRequest(long, tmp_path / "out" / "long.mp4")])
