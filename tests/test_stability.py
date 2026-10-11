"""Shimmer and seam checks."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

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
    SEAM_LIMIT,
    SHIMMER_LIMIT,
    StabilityError,
    find_seams,
    seam_candidates,
    seam_score,
    shimmer_index,
    texture_stats,
    video_size,
)
from super_processor.upscale import (
    FakeUpscaleEngine,
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
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
    times = seam_candidates(plan, 40)
    # Chunk restarts inside the first part, every 7.5 s.
    assert 7.5 in times and 15.0 in times
    # Settings change at 20 and 30; 25 is the same settings, so one part.
    assert 20 in times and 30 in times and 25 not in times
    # The revise edges.
    assert 33.2 in times and 36.8 in times
    assert all(0.5 < t < 39.5 for t in times)
    assert times == sorted(times)
    gappy = UpscalePlan(spans=(UpscaleSpan(0, 4), UpscaleSpan(6, 10, strength=0.2)))
    assert seam_candidates(gappy, 10) == [6]


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

    def label_shots(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            {"index": i, "label": "street", "issues": [], "contains": []}
            for i in range(len(kwargs["shots"]))
        ]

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
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(source)]})
    controller.run_scan()
    controller.handle_post("/shots", {"action": ["approve"]})
    controller.run_plan()
    assert controller.current_state().step is WizardStep.LOOKS
    shot = controller.api_view()["looks"]["shots"][0]
    assert gemini.shimmer_seen and gemini.shimmer_seen[0] > SHIMMER_LIMIT
    assert shot["check"]["ok"] is False
    assert "flicker" in shot["check"]["problems"]
    assert shot["check"]["adjusted"] is True
    assert shot["strength"] == pytest.approx(0.8 - SHIMMER_STEP)
    assert shot["after_url"].endswith("_r2_after.mp4")
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
