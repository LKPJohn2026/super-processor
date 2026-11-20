"""Shimmer and seam checks."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from loop_driver import approve_segments, pick_and_segment, preview, segments

from super_processor import stability
from super_processor.look import ShotLook
from super_processor.regions import Region
from super_processor.shot_plan import (
    SHIMMER_STEP,
    ShotCheck,
    ShotPlan,
    ShotRecipe,
    guard_shimmer,
    measure_shimmer,
)
from super_processor.stability import (
    MIN_DETAIL,
    SEAM_LIMIT,
    SEAM_MIN_JUMP,
    SHIMMER_LIMIT,
    StabilityError,
    TextureStats,
    find_seams,
    seam_candidates,
    seam_score,
    shimmer_index,
    texture_stats,
    video_size,
)
from super_processor.upscale import (
    STRUCTURE_SIGMA,
    FakeUpscaleEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
    load_upscale_plan,
    render_plan,
    save_upscale_plan,
)
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    load_render_report,
    save_job_wizard_state,
    save_session_state,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


def _encode(path: Path, *args: str) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *args, "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def _source(tmp_path: Path, seconds: int = 3) -> Path:
    return _encode(
        tmp_path / "src.mp4",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=160x90:rate=10:duration={seconds}",
        "-c:v",
        "libx264",
    )


def _restored(tmp_path: Path, source: Path, name: str, extra: str) -> Path:
    chain = "scale=320:180:flags=lanczos" + (f",{extra}" if extra else "")
    return _encode(
        tmp_path / f"{name}.mp4",
        "-i",
        str(source),
        "-vf",
        chain,
        "-c:v",
        "libx264",
        "-crf",
        "16",
    )


STATIC = (
    "geq=lum='lum(X,Y)+20*(mod(floor(X/2)+floor(Y/2),2)*2-1)':cb='cb(X,Y)':cr='cr(X,Y)'"
)
CRAWL = "noise=alls=20:allf=t"


@needs_ffmpeg
def test_shimmer_tells_crawling_texture_from_stable_texture(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plain = _restored(tmp_path, source, "plain", "")
    static = _restored(tmp_path, source, "static", STATIC)
    crawl = _restored(tmp_path, source, "crawl", CRAWL)
    assert video_size(plain) == (320, 180)

    def index(after: Path) -> float | None:
        return shimmer_index(source, after, scale=2, after_size=(320, 180))

    plain_index = index(plain)
    assert plain_index is not None and abs(plain_index - 1) < 0.2
    static_index = index(static)
    assert static_index is not None and static_index < SHIMMER_LIMIT
    crawl_index = index(crawl)
    assert crawl_index is not None and crawl_index > SHIMMER_LIMIT


@needs_ffmpeg
def test_flat_clips_are_not_judged(tmp_path: Path) -> None:
    flat = _encode(
        tmp_path / "flat.mp4",
        "-f",
        "lavfi",
        "-i",
        "color=c=gray:size=320x180:rate=10:duration=2",
        "-c:v",
        "libx264",
    )
    assert shimmer_index(flat, flat, scale=2, after_size=(320, 180)) is None
    stats = texture_stats(flat, sigma=3)
    assert stats.detail < 1
    with pytest.raises(StabilityError):
        texture_stats(tmp_path / "missing.mp4", sigma=3)
    with pytest.raises(StabilityError):
        video_size(tmp_path / "missing.mp4")


@needs_ffmpeg
def test_seam_score_finds_a_jump_the_source_does_not_have(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plain = _restored(tmp_path, source, "plain", "")
    jump = _restored(
        tmp_path, source, "jump", "eq=brightness='if(gte(t,1.5),0.12,0)':eval=frame"
    )
    assert seam_score(jump, source, 1.5) > SEAM_LIMIT
    assert seam_score(jump, source, 1.0) < 1.5
    assert seam_score(plain, source, 1.5) < 1.5
    assert find_seams(jump, source, [1.0, 1.5]) == [
        {"time_s": 1.5, "score": pytest.approx(seam_score(jump, source, 1.5), abs=0.01)}
    ]


def test_seam_candidates_cover_restarts() -> None:
    face = Region("faces", 0.1, 0.1, 0.3, 0.3)
    plan = UpscalePlan(
        spans=(
            UpscaleSpan(0, 20),
            UpscaleSpan(20, 25, look=ShotLook(denoise=0.2)),
            UpscaleSpan(25, 30, look=ShotLook(denoise=0.2)),
            UpscaleSpan(30, 40, look=ShotLook(denoise=0.2), regions=(face,)),
        ),
        pending=UpscaleSpan(33.2, 36.8),
    )
    times = seam_candidates(plan, 40, fps=25, total_frames=1000)
    # Settings change at 20 and 30; 25 is the same settings, so one part. SPAN
    # works frame by frame, so the chunks inside a part add no joins.
    assert times == [20, 30]
    # A pending request alone does not establish that a splice happened.
    assert 33.2 not in times and 36.8 not in times
    revised = seam_candidates(
        plan, 40, fps=25, total_frames=1000, revised_range=(33, 37)
    )
    assert revised == [33, 37]
    gappy = UpscalePlan(spans=(UpscaleSpan(0, 4), UpscaleSpan(6, 10, strength=0.2)))
    with pytest.raises(UpscaleError, match="gaps or overlaps"):
        seam_candidates(gappy, 10, fps=25, total_frames=250)


@needs_ffmpeg
def test_candidates_detect_a_visible_jump_where_the_recipe_changes(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, seconds=10)
    jump = _restored(
        tmp_path, source, "jump5", "eq=brightness='if(gte(t,5),0.3,0)':eval=frame"
    )
    plan = UpscalePlan(spans=(UpscaleSpan(0, 5), UpscaleSpan(5, 10, strength=0.2)))
    candidates = seam_candidates(plan, 10, fps=10, total_frames=100)
    assert candidates == [5]
    assert seam_score(jump, source, 4.5) < SEAM_LIMIT
    assert [seam["time_s"] for seam in find_seams(jump, source, candidates)] == [5]
    # Two equal spans are one render part: no join to check.
    same = UpscalePlan(spans=(UpscaleSpan(0, 5), UpscaleSpan(5, 10)))
    assert seam_candidates(same, 10, fps=10, total_frames=100) == []


@pytest.mark.parametrize("fps", [24.0, 25.0, 30000 / 1001])
def test_candidates_share_frame_snapping(fps: float) -> None:
    from super_processor.upscale import render_shot_parts

    total = round(31.7 * fps)
    # One long part: its upscale chunks are not joins.
    plan = UpscalePlan(spans=(UpscaleSpan(0, 31.7),))
    assert seam_candidates(plan, 31.7, fps=fps, total_frames=total) == []
    split = UpscalePlan(
        spans=(UpscaleSpan(0, 5.919), UpscaleSpan(5.919, 31.7, strength=0.2))
    )
    parts = render_shot_parts(split.spans, 31.7, total, fps)
    assert parts[0][2] == parts[1][1] == round(5.919 * fps)
    assert seam_candidates(split, 31.7, fps=fps, total_frames=total) == [
        round(5.919 * fps) / fps
    ]


def test_joins_too_close_to_either_end_are_not_checked() -> None:
    # A seam window is half a second each side; a join in the first or last
    # half second has no room around it.
    plan = UpscalePlan(
        spans=(
            UpscaleSpan(0, 0.4, strength=0.2),
            UpscaleSpan(0.4, 0.6),
            UpscaleSpan(0.6, 9.4, strength=0.2),
            UpscaleSpan(9.4, 9.6),
            UpscaleSpan(9.6, 10, strength=0.2),
        )
    )
    # Joins at 0.4, 0.6, 9.4, and 9.6 s; the outer two are dropped.
    assert seam_candidates(plan, 10, fps=25, total_frames=250) == [0.6, 9.4]


@pytest.mark.parametrize("scale,use", [(1, True), (2, False)])
def test_candidates_without_upscaling(scale: int, use: bool) -> None:
    plan = UpscalePlan(spans=(UpscaleSpan(0, 20, scale=scale, use_upscale=use),))
    assert seam_candidates(plan, 20, fps=25, total_frames=500) == []


def test_upscale_toggle_splits_otherwise_equal_spans() -> None:
    plan = UpscalePlan(
        spans=(
            UpscaleSpan(0, 10, use_upscale=False),
            UpscaleSpan(10, 20),
            UpscaleSpan(20, 30, use_upscale=False),
        )
    )
    assert seam_candidates(plan, 30, fps=25, total_frames=750) == [10, 20]


@needs_ffmpeg
@pytest.mark.parametrize("legacy", [False, True])
def test_wizard_checks_actual_revision_edges_before_clearing_pending(
    tmp_path: Path, legacy: bool
) -> None:
    source = _source(tmp_path, seconds=6)
    controller = WizardController(tmp_path / "jobs", upscale_engine=FakeUpscaleEngine())
    manifest = controller.store.create(source)
    job_dir = controller.store.job_dir(manifest.job_id)
    output = job_dir / "output.mp4"
    plan = UpscalePlan(spans=(UpscaleSpan(0, 6),))
    render_plan(source, output, plan, engine=FakeUpscaleEngine(), duration_s=6)
    if legacy:
        output.with_suffix(".mp4.delivery.json").unlink()
    save_upscale_plan(
        job_dir,
        UpscalePlan(spans=plan.spans, pending=UpscaleSpan(2.2, 4.8)),
    )
    state = WizardState(step=WizardStep.RENDERING, job_id=manifest.job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(controller.jobs_dir, state)
    controller.run_render()
    assert controller.current_state().step is WizardStep.RESULT
    report = json.loads((job_dir / "render_report.json").read_text())
    assert report["checked"] == [2, 5]
    saved = load_upscale_plan(job_dir)
    assert saved is not None and saved.pending is None


def test_full_render_ignores_pending_edges_and_drops_zero_frame_parts() -> None:
    # A pending request does not imply a splice (no existing output, or a
    # whole-timeline render). A sub-frame part also introduces no retained join.
    plan = UpscalePlan(
        spans=(
            UpscaleSpan(0, 0.01, strength=0.2),
            UpscaleSpan(0.01, 10),
        ),
        pending=UpscaleSpan(2.2, 4.8),
    )
    assert seam_candidates(plan, 10, fps=25, total_frames=250) == []


def test_revision_joins_are_the_actual_splice_not_requested_edges() -> None:
    plan = UpscalePlan(spans=(UpscaleSpan(0, 30),), pending=UpscaleSpan(3.2, 16.8))
    assert seam_candidates(
        plan, 30, fps=25, total_frames=750, revised_range=(3, 17)
    ) == [3, 17]
    with pytest.raises(ValueError, match="pending"):
        seam_candidates(
            UpscalePlan(spans=plan.spans),
            30,
            fps=25,
            total_frames=750,
            revised_range=(3, 17),
        )


def _plan(*shimmers: float | None, strength: float = 0.5) -> ShotPlan:
    return ShotPlan(
        scale=2,
        recipes=[ShotRecipe(strength=strength, shimmer=s) for s in shimmers],
    )


def test_shimmer_guard_lowers_strength_and_flags_flicker() -> None:
    plan = _plan(2.4, 1.1, None, 3.0, 2.0)
    plan.recipes[3] = ShotRecipe(
        strength=0.4,
        shimmer=3.0,
        check=ShotCheck(ok=False, problems=("halos",), note="halo", adjusted=True),
    )
    plan.recipes[4] = ShotRecipe(strength=0.0, shimmer=2.0)
    guarded, changed = guard_shimmer(plan, [0, 1, 2, 3, 4], already=[3])
    assert changed == [0]
    first = guarded.recipes[0]
    assert first.strength == pytest.approx(0.5 - SHIMMER_STEP)
    assert first.check.ok is False and first.check.adjusted
    assert first.check.problems == ("flicker",)
    assert "crawls" in first.check.note
    assert guarded.recipes[1] == plan.recipes[1]
    assert guarded.recipes[2] == plan.recipes[2]
    # Gemini already changed shot 4's settings: add the problem, keep its note.
    assert guarded.recipes[3].check.problems == ("halos", "flicker")
    assert guarded.recipes[3].check.note == "halo"
    assert guarded.recipes[3].strength == 0.4
    # Nothing left to lower.
    assert guarded.recipes[4].strength == 0.0
    assert "flicker" in guarded.recipes[4].check.problems


def test_recipe_stores_shimmer(tmp_path: Path) -> None:
    recipe = ShotRecipe(shimmer=1.7)
    assert ShotRecipe.from_dict(recipe.to_dict()).shimmer == 1.7
    assert ShotRecipe.from_dict({"shimmer": "high"}).shimmer is None
    measured = measure_shimmer(tmp_path, _plan(None), [0])
    assert measured.recipes[0].shimmer is None  # no preview files


class _CrawlEngine(FakeUpscaleEngine):
    """Upscales and adds texture that changes every frame."""

    def upscale(self, request: UpscaleRequest) -> Path:
        _encode(
            request.output,
            "-i",
            str(request.source),
            "-vf",
            f"scale=iw*{request.scale}:ih*{request.scale},{CRAWL}",
            "-c:v",
            "libx264",
        )
        return request.output


class _Gemini:
    """Labels nothing special and passes every preview."""

    def propose_segments(self, **kwargs: Any) -> dict[str, Any]:
        found = segments((0.0, kwargs["duration_s"], "street", []))
        return {"segments": found, "scale": None, "scale_reason": ""}

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        self.shimmer_seen: list[Any] = []
        return {
            "scale": 2,
            "shots": [
                {"index": s["index"], "strength": 0.8, "look": {}, "reason": "r"}
                for s in kwargs["shots"]
            ],
        }

    def check_previews(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.shimmer_seen = [shot["shimmer"] for shot in kwargs["shots"]]
        return [{"index": shot["index"], "ok": True} for shot in kwargs["shots"]]


@needs_ffmpeg
def test_planning_fails_a_crawling_preview_even_when_gemini_passes_it(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    gemini = _Gemini()
    controller = WizardController(
        tmp_path / "jobs",
        gemini=gemini,  # type: ignore[arg-type]
        upscale_engine=_CrawlEngine(),
    )
    pick_and_segment(controller, source)
    assert approve_segments(controller).step is WizardStep.PREVIEW_SEGMENT
    shot = preview(controller)["shot"]
    assert gemini.shimmer_seen and gemini.shimmer_seen[0] > SHIMMER_LIMIT
    assert shot["check"]["ok"] is False
    assert "flicker" in shot["check"]["problems"]
    assert shot["check"]["adjusted"] is True
    assert shot["strength"] == pytest.approx(0.8 - SHIMMER_STEP)
    assert shot["after_url"].partition("?")[0].endswith("_r2_after.mp4")
    assert shot["shimmer"] is not None
    page = controller.render()
    assert "Check found: flicker." in page
    assert "Texture stability:" in page


class _TextureEngine:
    """A fine static checkerboard over the upscale, so strength shows."""

    def upscale(self, request: UpscaleRequest) -> Path:
        return _encode(
            request.output,
            "-i",
            str(request.source),
            "-vf",
            f"scale=iw*{request.scale}:ih*{request.scale}:flags=lanczos,{STATIC}",
            "-c:v",
            "libx264",
        )


@needs_ffmpeg
def test_render_reports_a_seam_between_unlike_shots(tmp_path: Path) -> None:
    source = _source(tmp_path)
    controller = WizardController(tmp_path / "jobs", upscale_engine=_TextureEngine())
    manifest = controller.store.create(source)
    job_dir = controller.store.job_dir(manifest.job_id)
    # Full detail, then none: the texture switches off at 1.5 s.
    save_upscale_plan(
        job_dir,
        UpscalePlan(
            spans=(UpscaleSpan(0, 1.5, strength=1.0), UpscaleSpan(1.5, 3, strength=0.0))
        ),
    )
    state = WizardState(step=WizardStep.RENDERING, job_id=manifest.job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(controller.jobs_dir, state)
    controller.run_render()
    assert controller.current_state().step is WizardStep.RESULT
    report = load_render_report(job_dir)
    assert report["error"] is None
    assert [seam["time_s"] for seam in report["seams"]] == [1.5]
    view = controller.api_view()
    assert view["seams"][0]["time_s"] == 1.5
    page = controller.render()
    assert "Possible seam at 0:01.5" in page
    assert "smooth 0:00.5 to 0:02.5" in page


def test_render_report_tolerates_bad_files(tmp_path: Path) -> None:
    assert load_render_report(tmp_path) == {"seams": [], "error": None}
    (tmp_path / "render_report.json").write_text("{", encoding="utf-8")
    assert load_render_report(tmp_path)["seams"] == []
    (tmp_path / "render_report.json").write_text(
        '{"seams": [{"time_s": 2}, {"time_s": "x"}, 4], "error": "no"}',
        encoding="utf-8",
    )
    assert load_render_report(tmp_path) == {"seams": [{"time_s": 2}], "error": "no"}


def _frames(*levels: int, size: int = 4) -> list[bytes]:
    return [bytes([level]) * size for level in levels]


def _fake_frames(
    monkeypatch: pytest.MonkeyPatch, output: list[bytes], source: list[bytes]
) -> None:
    """``seam_score`` reads ``output`` and ``source`` from these grey frames."""
    by_name = {"out.mp4": output, "src.mp4": source}

    def gray(
        video: Path, start_s: float, seconds: float, *, ffmpeg_bin: str
    ) -> tuple[list[bytes], int]:
        assert seconds == pytest.approx(2 * stability.SEAM_WINDOW_S)
        frames = by_name[video.name]
        return frames, len(frames[0])

    monkeypatch.setattr(stability, "_gray_frames", gray)


STEADY = _frames(10, 11, 12, 13, 14, 15, 16)


def test_a_jump_the_source_also_has_is_no_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    cut = _frames(10, 11, 12, 60, 61, 62, 63)
    _fake_frames(monkeypatch, output=cut, source=STEADY)
    seam = seam_score(Path("out.mp4"), Path("src.mp4"), 1.0)
    # Jumps 1,1,48,1,1,1: the peak over the median plus 0.5.
    assert seam == pytest.approx(48 / 1.5)
    # The source cuts at the same frame: the output only follows it.
    _fake_frames(monkeypatch, output=cut, source=cut)
    assert seam_score(Path("out.mp4"), Path("src.mp4"), 1.0) == pytest.approx(1.0)


def test_small_or_short_changes_are_no_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    # The peak is many times its neighbours, but under SEAM_MIN_JUMP grey levels.
    faint = _frames(10, 10, 10, 11, 11, 11, 11)
    assert SEAM_MIN_JUMP > 1
    _fake_frames(monkeypatch, output=faint, source=STEADY)
    assert seam_score(Path("out.mp4"), Path("src.mp4"), 1.0) == 0.0
    # Three frames give two jumps: too few to tell a peak from its neighbours.
    _fake_frames(monkeypatch, output=_frames(0, 0, 200), source=STEADY)
    assert seam_score(Path("out.mp4"), Path("src.mp4"), 1.0) == 0.0


def test_find_seams_keeps_scores_at_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    scores = {1.0: SEAM_LIMIT - 0.01, 2.0: SEAM_LIMIT, 3.12345: 9.876}
    monkeypatch.setattr(
        stability,
        "seam_score",
        lambda output, source, time_s, *, ffmpeg_bin: scores[time_s],
    )
    found = find_seams(Path("o"), Path("s"), list(scores))
    assert found == [
        {"time_s": 2.0, "score": SEAM_LIMIT},
        {"time_s": 3.123, "score": 9.88},
    ]


def test_shimmer_compares_like_with_like(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float, tuple[int, int] | None]] = []
    stats = {
        "after.mp4": TextureStats(detail=8.0, change=4.0),
        "before.mp4": TextureStats(detail=4.0, change=1.0),
    }

    def measure(
        video: Path,
        *,
        sigma: float,
        size: tuple[int, int] | None = None,
        ffmpeg_bin: str,
    ) -> TextureStats:
        calls.append((video.name, sigma, size))
        return stats[video.name]

    monkeypatch.setattr(stability, "texture_stats", measure)
    index = shimmer_index(
        Path("before.mp4"), Path("after.mp4"), scale=2, after_size=(640, 360)
    )
    # (4 / 8) / (1 / 4): the restored texture changes twice as much.
    assert index == pytest.approx(2.0)
    # The source is measured at the restored size, with the same blur.
    sigma = STRUCTURE_SIGMA * 2
    assert calls == [("after.mp4", sigma, None), ("before.mp4", sigma, (640, 360))]
    # A restored clip with almost no texture is not judged, nor the source read.
    calls.clear()
    stats["after.mp4"] = TextureStats(detail=MIN_DETAIL - 0.01, change=4.0)
    assert (
        shimmer_index(
            Path("before.mp4"), Path("after.mp4"), scale=2, after_size=(640, 360)
        )
        is None
    )
    assert [name for name, _s, _z in calls] == ["after.mp4"]


def test_a_missing_ffmpeg_is_a_stability_error(tmp_path: Path) -> None:
    missing = str(tmp_path / "no-ffmpeg")
    clip = tmp_path / "clip.mp4"
    with pytest.raises(StabilityError, match="failed to start ffmpeg"):
        texture_stats(clip, sigma=3, ffmpeg_bin=missing)
    with pytest.raises(StabilityError, match="failed to start ffmpeg"):
        seam_score(clip, clip, 1.0, ffmpeg_bin=missing)


@needs_ffmpeg
def test_unreadable_or_one_frame_clips_are_stability_errors(tmp_path: Path) -> None:
    with pytest.raises(StabilityError, match="could not read missing.mp4"):
        seam_score(tmp_path / "missing.mp4", tmp_path / "missing.mp4", 1.0)
    still = _encode(
        tmp_path / "still.mp4",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=160x90:rate=10",
        "-frames:v",
        "1",
        "-c:v",
        "libx264",
    )
    # One frame has detail but no frame-to-frame change to compare.
    with pytest.raises(StabilityError, match="too short to measure shimmer"):
        texture_stats(still, sigma=3)
    sound = tmp_path / "sound.m4a"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=duration=1"]
        + ["-c:a", "aac", str(sound)],
        check=True,
    )
    with pytest.raises(StabilityError, match="could not read the size of sound.m4a"):
        video_size(sound)
