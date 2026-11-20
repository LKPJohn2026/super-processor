"""Loop 2: per-shot recipes, previews, the self-check, and editor redo."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from loop_driver import (
    analyze,
    approve_preview,
    approve_segments,
    pick_and_segment,
    preview,
    segments,
)

from super_processor.gemini import GeminiClient, GeminiError
from super_processor.look import LOOK_BOUNDS, ShotLook
from super_processor.shot_plan import (
    CHECK_PROBLEMS,
    PROTECTED_STRENGTH,
    ShotCheck,
    ShotPlan,
    ShotPlanError,
    ShotRecipe,
    apply_checks,
    apply_proposals,
    fallback_recipe,
    finish,
    keep_editor_choices,
    load_plan,
    look_from_model,
    new_plan,
    preview_window,
    random_preview_start,
    remove_previews,
    render_previews,
    request_redo,
    save_plan,
    set_upscale,
    shots_payload,
)
from super_processor.shots import Shot, ShotError, ShotList, ShotMetrics
from super_processor.upscale import (
    FakeUpscaleEngine,
    load_upscale_plan,
    video_stream_facts,
)
from super_processor.wizard import (
    WizardController,
    WizardStep,
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
    plan = finish(new_plan(shots, 2), [0, 1, 2])
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
    """Three 3s shots of different pictures, so scdet finds two cuts."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=10:duration=3",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x303030:size=320x180:rate=10:duration=3",
            "-f",
            "lavfi",
            "-i",
            "mandelbrot=size=320x180:rate=10",
            "-filter_complex",
            "[2:v]trim=duration=3,setpts=PTS-STARTPTS[m];"
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
        scales=(1, 2),
        protected_strength=0.6,
        job_dir=tmp_path,
    )
    assert out == reply
    body = transport.bodies[0]
    text = body["contents"][0]["parts"][0]["text"]
    assert "deblock 0 to 1" in text and "[1, 2]" in text and "0.6" in text
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
    """Proposes three segments, then plans; fails the check on segment 1 once."""

    def __init__(self) -> None:
        self.proposals: list[list[dict[str, Any]]] = []
        self.scales: list[tuple[int, ...]] = []
        self.checks = 0
        self.failed: set[int] = set()

    def propose_segments(self, **kwargs: Any) -> list[dict[str, Any]]:
        return segments(
            (0.0, 3.0, "shot 0", []), (3.0, 6.0, "shot 1", []), (6.0, 9.0, "shot 2", [])
        )

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
        failed_before = 1 in self.failed
        out = []
        for shot in kwargs["shots"]:
            bad = shot["index"] == 1 and not failed_before
            if bad:
                self.failed.add(1)
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
def test_wizard_plans_checks_and_redoes_one_segment(tmp_path: Path) -> None:
    clip = _three_shots(tmp_path / "cuts.mp4")
    planner = _Planner()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=planner,  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    pick_and_segment(controller, clip)
    state = approve_segments(controller)
    assert state.step is WizardStep.PREVIEW_SEGMENT, state.error
    # Previews are at the plan's 2x; the planner's 4x is no longer a scale.
    assert planner.scales == [(2,)]
    first = preview(controller)
    assert first["scale"] == 2 and first["error"] is None
    assert first["shot"]["strength"] == 0.7 and first["shot"]["check"]["ok"] is True
    assert "upscaled 2×" in controller.render()

    approve_preview(controller)
    # The check failed segment 2: its settings changed and its preview was redone.
    second = preview(controller)["shot"]
    assert second["strength"] == 0.4
    assert second["check"]["problems"] == ["fake_texture"]
    assert second["after_url"].partition("?")[0].endswith("shot_001_r2_after.mp4")
    assert "Check found: fake texture. grain turned to worms." in controller.render()

    approve_preview(controller)
    third = preview(controller)["shot"]
    assert third["after_url"].partition("?")[0].endswith("shot_002_r1_after.mp4")
    controller.handle_post("/preview", {"action": ["redo"], "note": [""]})
    assert controller.current_state().error
    assert controller.current_state().step is WizardStep.PREVIEW_SEGMENT
    controller.handle_post(
        "/preview", {"action": ["redo"], "note": ["keep the sign crisp"]}
    )
    assert analyze(controller).step is WizardStep.PREVIEW_SEGMENT
    assert planner.scales[-1] == (2,)
    assert [shot["index"] for shot in planner.proposals[-1]] == [2]
    assert planner.proposals[-1][0]["editor_note"] == "keep the sign crisp"
    third = preview(controller)["shot"]
    assert third["strength"] == 0.3
    assert third["after_url"].partition("?")[0].endswith("shot_002_r2_after.mp4")

    controller.handle_post("/preview", {"action": ["dance"]})
    assert "unknown action" in (controller.current_state().error or "")
    assert approve_preview(controller).step is WizardStep.READY_TO_RENDER
    controller.handle_post("/ready", {"action": ["render"]})
    assert controller.current_state().step is WizardStep.RENDERING
    state = controller.current_state()
    assert state.job_id
    plan = load_upscale_plan(controller.store.job_dir(state.job_id))
    assert plan is not None
    assert [span.scale for span in plan.spans] == [2, 2, 2]
    assert [span.strength for span in plan.spans] == [0.7, 0.4, 0.3]
    assert plan.spans[2].look.denoise == pytest.approx(0.2)


def test_preview_post_needs_the_preview_step(tmp_path: Path) -> None:
    from super_processor.wizard import WizardError

    controller = WizardController(tmp_path)
    with pytest.raises(WizardError, match="no segment preview"):
        controller.handle_post("/preview", {"action": ["approve"]})
    with pytest.raises(WizardError, match="unknown action /looks"):
        controller.handle_post("/looks", {"action": ["approve"]})


def test_proposals_carry_use_upscale() -> None:
    shots = _shots()
    plan = new_plan(shots, 2)
    merged = apply_proposals(
        plan,
        shots,
        [
            {"index": 0, "strength": 0.4, "look": {}, "use_upscale": False},
            {"index": 1, "strength": 0.4, "look": {}, "use_upscale": "no"},
            {"index": 2, "strength": 0.4, "look": {}},
        ],
        [0, 1, 2],
    )
    assert [r.use_upscale for r in merged.recipes] == [False, True, True]
    data = merged.to_dict()
    assert data["recipes"][0]["use_upscale"] is False
    assert ShotRecipe.from_dict(data["recipes"][0]).use_upscale is False
    assert ShotRecipe.from_dict({"strength": 0.5}).use_upscale is True


def test_random_preview_window_is_seeded_and_inside_the_shot() -> None:
    shot = Shot(10.0, 30.0)
    first = random_preview_start(shot, "job:0:1")
    assert first is not None
    assert first == random_preview_start(shot, "job:0:1")
    starts = {random_preview_start(shot, f"job:0:{rev}") for rev in range(8)}
    assert len(starts) > 1
    for start in starts:
        assert start is not None
        begin, end = preview_window(shot, start)
        assert begin >= 10.0 and end <= 30.0 and end - begin == pytest.approx(3.0)
    assert random_preview_start(Shot(0.0, 2.0), "x") is None
    assert preview_window(shot, 29.0) == (27.0, 30.0)
    assert preview_window(shot, 1.0) == (10.0, 13.0)


def test_redo_can_resample_the_preview_window() -> None:
    shots = _shots()
    plan = new_plan(shots, 2)
    plan.recipes[0] = replace(plan.recipes[0], preview_start=4.0)
    kept = request_redo(plan, 0, "less sharp")
    assert kept.recipes[0].preview_start == 4.0
    fresh = request_redo(plan, 0, "show another part", resample=True)
    assert fresh.recipes[0].preview_start is None
    assert ShotRecipe.from_dict(kept.recipes[0].to_dict()).preview_start == 4.0


def test_a_plan_saved_at_4x_renders_at_2x() -> None:
    # 4x was removed; plans saved before that come back at 2x instead of failing.
    old = ShotPlan.from_dict({"scale": 4, "recipes": [ShotRecipe().to_dict()]})
    assert old.scale == 2
    assert ShotPlan.from_dict({"scale": 1, "recipes": []}).scale == 1


def test_editor_choices_survive_a_fresh_recipe() -> None:
    old = ShotRecipe(
        strength=0.2,
        rev=3,
        note="less grain",
        preview_start=12.5,
        upscale_override=False,
        use_upscale=False,
    )
    fresh = ShotRecipe(strength=0.8, look=ShotLook(denoise=0.3), reason="new")
    kept = keep_editor_choices(fresh, old)
    # The new settings come in; the editor's count, window, note and choice stay.
    assert (kept.strength, kept.look, kept.reason) == (0.8, fresh.look, "new")
    assert (kept.rev, kept.note, kept.preview_start) == (3, "less grain", 12.5)
    assert kept.use_upscale is False and kept.upscale_override is False
    # Without an editor choice, the fresh recipe's own upscale decision wins.
    free = keep_editor_choices(replace(fresh, use_upscale=False), ShotRecipe())
    assert free.use_upscale is False and free.upscale_override is None


def test_the_editors_upscale_choice_beats_gemini_and_the_fallback() -> None:
    shots = _shots()
    plan = new_plan(shots, 2)
    plan = set_upscale(plan, 0, False)
    plan = set_upscale(plan, 2, False)
    merged = apply_proposals(
        plan,
        shots,
        # Gemini asks to upscale shot 0 and says nothing about shot 2.
        [{"index": 0, "strength": 0.4, "look": {}, "use_upscale": True}],
        [0, 2],
    )
    assert merged.recipes[0].use_upscale is False
    assert merged.recipes[0].strength == 0.4
    assert merged.recipes[2].use_upscale is False
    assert "measurements" in merged.recipes[2].reason


def test_set_upscale_redoes_one_shot_and_clears_its_note() -> None:
    shots = _shots()
    plan = finish(new_plan(shots, 2), [0, 1, 2])
    plan = request_redo(plan, 1, "too sharp")
    plan.error = "old error"
    toggled = set_upscale(plan, 1, False)
    recipe = toggled.recipes[1]
    assert (recipe.use_upscale, recipe.upscale_override) == (False, False)
    assert recipe.stale is True and recipe.note == ""
    assert toggled.stale_indexes() == [1]
    assert toggled.error is None
    restored = ShotRecipe.from_dict(recipe.to_dict())
    assert restored.upscale_override is False and restored.use_upscale is False
    with pytest.raises(ShotPlanError, match="no such shot"):
        set_upscale(plan, 3, True)


def test_a_failed_check_keeps_the_look_it_did_not_change_and_the_face_cap() -> None:
    shots = _shots()
    look = ShotLook(denoise=0.4)
    plan = ShotPlan(scale=2, recipes=[ShotRecipe(strength=0.5, look=look)] * 3)
    checked, changed = apply_checks(
        plan,
        shots,
        # Shot 1 has unboxed faces: a higher strength from the check is capped.
        [{"index": 1, "ok": False, "problems": ["too_soft"], "strength": 0.9}],
        [0, 1, 2],
    )
    assert changed == [1]
    assert checked.recipes[1].strength == PROTECTED_STRENGTH
    assert checked.recipes[1].look == look
    # Shots without a verdict are left as they were.
    assert checked.recipes[0] == plan.recipes[0]
    assert checked.recipes[2] == plan.recipes[2]


def test_a_check_with_a_boolean_index_is_ignored() -> None:
    shots = _shots()
    plan = ShotPlan(scale=2, recipes=[ShotRecipe(strength=0.5) for _ in range(3)])
    checked, changed = apply_checks(
        plan, shots, [{"index": True, "ok": False, "strength": 0.1}], [0, 1, 2]
    )
    assert changed == []
    assert checked.recipes[1] == plan.recipes[1]


def test_saved_booleans_are_not_read_as_a_revision_or_a_scale() -> None:
    # bool is an int: a JSON true must not load as rev 1 or as 1x.
    assert ShotRecipe.from_dict({"rev": True}).rev == 0
    assert ShotRecipe.from_dict({"rev": "3"}).rev == 0
    assert ShotRecipe.from_dict({"rev": -2}).rev == 0
    assert ShotRecipe.from_dict({"rev": 3}).rev == 3
    plan = ShotPlan.from_dict({"scale": True, "recipes": []})
    assert plan.scale == 2 and type(plan.scale) is int
    assert type(ShotPlan.from_dict({"scale": 1.0, "recipes": []}).scale) is int


def test_saved_recipes_load_defensively() -> None:
    with pytest.raises(ShotPlanError, match="must be an object"):
        ShotRecipe.from_dict(["strength", 0.5])
    loaded = ShotRecipe.from_dict(
        {
            "strength": 3.0,
            "look": {"denoise": 9, "sharpen": 1},
            "shimmer": True,
            "preview_start": "4",
            "upscale_override": "no",
            "check": {"ok": "yes", "problems": ["halos", "nope"], "adjusted": 1},
        }
    )
    assert loaded.strength == 1.0
    assert loaded.look == ShotLook(denoise=1.0)
    assert loaded.shimmer is None and loaded.preview_start is None
    assert loaded.upscale_override is None
    # Only a real boolean counts as a verdict, and only True as adjusted.
    assert loaded.check == ShotCheck(ok=None, problems=("halos",), adjusted=False)
    assert ShotRecipe.from_dict({"strength": math.nan}).strength == 0.5
    assert ShotCheck.from_dict("passed") == ShotCheck()


def test_payload_shows_current_settings_only_once_a_shot_has_a_recipe() -> None:
    shots = _shots()
    plan = new_plan(shots, 2)
    assert "current" not in shots_payload(shots, plan, [0])[0]
    planned = apply_proposals(
        plan, shots, [{"index": 0, "strength": 0.3, "look": {}, "reason": "r"}], [0]
    )
    item = shots_payload(shots, planned, [0])[0]
    assert item["current"] == {"strength": 0.3, "look": ShotLook().to_dict()}
    assert item["protected_boxes"] == []
    assert shots_payload(shots, None, [2])[0]["contains"] == ["text"]
    with pytest.raises(ShotError, match="no shots"):
        new_plan(ShotList(shots=[]), 2)


def test_save_plan_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A save racing a reader on Windows retries, and a save that never gets
    the file leaves the old plan whole with no temp file behind."""
    monkeypatch.setattr("super_processor.atomic_io.RETRY_S", 0)
    old = ShotPlan(scale=2, recipes=[ShotRecipe(strength=0.5) for _ in range(2)])
    save_plan(tmp_path, old)
    real_replace = Path.replace
    held = {"left": 2}

    def flaky_replace(self: Path, target: Path) -> Path:
        if held["left"] > 0:
            held["left"] -= 1
            raise PermissionError(13, "file is open in another thread")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    new = ShotPlan(scale=1, recipes=[ShotRecipe(strength=0.2) for _ in range(3)])
    save_plan(tmp_path, new)
    loaded = load_plan(tmp_path)
    assert loaded is not None
    assert loaded.to_dict() == new.to_dict()

    held["left"] = 1000
    with pytest.raises(PermissionError):
        save_plan(tmp_path, old)
    loaded = load_plan(tmp_path)
    assert loaded is not None
    assert loaded.to_dict() == new.to_dict()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shot_plan.json"]


def test_each_new_plan_has_its_own_generation_and_copies_keep_it(
    tmp_path: Path,
) -> None:
    shots = ShotList(shots=[Shot(0.0, 4.0), Shot(4.0, 8.0)])
    first, second = new_plan(shots, 2), new_plan(shots, 2)
    assert first.generation and second.generation
    assert first.generation != second.generation
    save_plan(tmp_path, first)
    loaded = load_plan(tmp_path)
    assert loaded is not None and loaded.generation == first.generation
    # Changing a scene's settings is the same plan, so its URLs stay put.
    assert set_upscale(first, 0, False).generation == first.generation
    assert request_redo(first, 1, "softer").generation == first.generation
    # It goes into a URL: anything but the hex it was made as is dropped.
    for bad in ("../x", "a&b=1", True, 7, "A" * 3, "f" * 65):
        data = first.to_dict()
        data["generation"] = bad
        assert ShotPlan.from_dict(data).generation == ""
    data.pop("generation")
    assert ShotPlan.from_dict(data).generation == ""


def test_remove_previews_leaves_what_the_disk_will_not_let_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "previews"
    (folder / "work").mkdir(parents=True)
    names = [
        "shot_000_r1_before.mp4",
        "shot_000_r1_after.mp4",
        "shot_000_r1_after.mp4.delivery.json",
        "shot_001_r2_after.jpg",
    ]
    for name in names:
        (folder / name).write_bytes(b"x")
    (folder / "notes.txt").write_bytes(b"keep")
    real_unlink = Path.unlink

    def held(self: Path, missing_ok: bool = False) -> None:
        if self.name == "shot_000_r1_after.mp4":
            raise PermissionError(13, "a browser is playing it")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", held)
    remove_previews(tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == [
        "notes.txt",
        "shot_000_r1_after.mp4",
        "work",
    ]
    remove_previews(tmp_path / "no job")
