"""Protected regions and the structure-from-source blend."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from loop_driver import (
    approve_preview,
    approve_segments,
    pick_and_segment,
    preview,
    segments,
)

from super_processor.gemini import GeminiClient, GeminiError
from super_processor.regions import (
    FEATHER,
    MAX_REGIONS,
    PAD,
    REGION_STRENGTH,
    Region,
    RegionError,
    region_from_model,
    regions_from_dicts,
    strength_expr,
)
from super_processor.shot_plan import (
    PROTECTED_STRENGTH,
    apply_checks,
    apply_proposals,
    apply_regions,
    cap_strength,
    locate_stills,
    locate_targets,
    new_plan,
)
from super_processor.shots import Shot, ShotList, load_shots
from super_processor.upscale import (
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
    apply_range_revise,
    blend_filter,
    load_upscale_plan,
    span_at,
)
from super_processor.wizard import (
    WizardController,
    WizardStep,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)

FACE = Region("faces", 0.1, 0.1, 0.4, 0.5)


def test_region_shapes() -> None:
    assert Region.from_dict(FACE.to_dict()) == FACE
    for bad in (
        {"kind": "cats", "x0": 0, "y0": 0, "x1": 1, "y1": 1},
        {"kind": "faces", "x0": 0.5, "y0": 0, "x1": 0.4, "y1": 1},
        {"kind": "faces", "x0": 0, "y0": 0, "x1": 1.2, "y1": 1},
        {"kind": "faces", "x0": True, "y0": 0, "x1": 1, "y1": 1},
        {"kind": "faces", "x0": 0, "y0": 0, "x1": 1},
        "box",
    ):
        with pytest.raises(RegionError):
            Region.from_dict(bad)
    many = [FACE.to_dict()] * (MAX_REGIONS + 2) + [{"kind": "x"}, 3]
    assert regions_from_dicts(many) == (FACE,) * MAX_REGIONS
    assert regions_from_dicts(None) == ()


def test_region_from_model_pads_and_clamps() -> None:
    region = region_from_model("Faces", [100, 200, 500, 400])
    assert region == Region("faces", 0.2 - PAD, 0.1 - PAD, 0.4 + PAD, 0.5 + PAD)
    edge = region_from_model("text", [-50, 990, 300, 2000])
    assert edge is not None and edge.x1 == 1.0 and edge.y0 == 0.0
    swapped = region_from_model("hands", [500, 400, 100, 200])
    assert swapped == Region("hands", 0.2 - PAD, 0.1 - PAD, 0.4 + PAD, 0.5 + PAD)
    for kind, box in (
        ("cats", [0, 0, 100, 100]),
        ("faces", [0, 0, 100]),
        ("faces", [0, 0, 1, 1]),
        ("faces", [0, 0, True, 100]),
        ("faces", "0,0,10,10"),
    ):
        assert region_from_model(kind, box) is None


def _share(expr: str, x: float, y: float) -> float:
    """Evaluate a strength expression the way FFmpeg would at one pixel."""

    def clip(value: float, low: float, high: float) -> float:
        return min(high, max(low, value))

    names = {"X": x * 100, "Y": y * 100, "W": 100, "H": 100}
    return float(eval(expr, {"clip": clip, "min": min, "max": max}, names))  # noqa: S307


def test_strength_expression_lowers_inside_boxes() -> None:
    assert strength_expr(0.5, ()) == "0.5000"
    # A ceiling above the shot's strength changes nothing.
    assert strength_expr(0.2, (FACE,)) == "0.2000"
    expr = strength_expr(0.8, (FACE,))
    assert _share(expr, 0.25, 0.3) == pytest.approx(REGION_STRENGTH["faces"])
    assert _share(expr, 0.9, 0.9) == pytest.approx(0.8)
    halfway = _share(expr, 0.4 + FEATHER / 2, 0.3)
    assert 0.3 < halfway < 0.8
    sign = Region("text", 0.3, 0.2, 0.6, 0.4)
    both = strength_expr(0.8, (FACE, sign, Region("faces", 0.7, 0.7, 0.9, 0.9)))
    # Where a face and a sign overlap, the lower ceiling wins.
    assert _share(both, 0.35, 0.3) == pytest.approx(REGION_STRENGTH["text"])
    assert _share(both, 0.8, 0.8) == pytest.approx(REGION_STRENGTH["faces"])


def test_blend_filter_swaps_in_the_source_structure() -> None:
    plain = blend_filter(320, 180, 0.5)
    assert "gblur" not in plain
    graph = blend_filter(320, 180, 0.5, regions=(FACE,), sigma=3.0)
    assert graph.count("gblur=sigma=3.000") == 2
    assert "x-y+128" in graph and "y+x-128" in graph
    # Per-pixel work is a lookup table or a mask drawn once, never an
    # expression evaluated for every pixel of every frame.
    assert "blend=" not in graph and "all_expr" not in graph
    # The boxes are drawn into one-frame masks, in each plane's own scale.
    assert graph.count("r=1:d=1") == 3 and "X/(W*SW)" in graph
    # Each lookup keeps the restored branch as its main input, and each
    # picture leads its own merge so it keeps its timestamps.
    assert graph.count("eof_action=pass") == 3
    assert "[rebuilt][zeror][keepr]maskedmerge" in graph
    assert "[basemix][zerob][keepb]maskedmerge" in graph
    # Without boxes the mix is one lookup table.
    assert "maskedmerge" not in blend_filter(320, 180, 0.5, sigma=3.0)


def _mixed_frames(graph: str, restored: str, base: str, frames: int) -> list[bytes]:
    """Run ``graph`` on two lavfi sources; the luma plane of each output frame."""
    out = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            restored,
            "-f",
            "lavfi",
            "-i",
            base,
            "-filter_complex",
            graph,
            "-map",
            "[v]",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    size = 64 * 48
    step = size * 3 // 2
    assert len(out) == frames * step
    return [out[i * step : i * step + size] for i in range(frames)]


@needs_ffmpeg
@pytest.mark.parametrize(("restored", "base"), [(36, 12), (12, 36)])
def test_boxes_mix_each_frame_with_its_own_base(restored: int, base: int) -> None:
    box = (Region("faces", 0.25, 0.25, 0.75, 0.75),)
    graph = blend_filter(64, 48, 0.8, regions=box)
    # Flat pictures: the restored one at 200, the base at 50.
    flat = _mixed_frames(
        graph,
        f"color=c=0xC8C8C8:s=64x48:r=30000/1001,trim=end_frame={restored}",
        f"color=c=0x323232:s=32x24:r=30000/1001,trim=end_frame={base}",
        restored,
    )
    # The restored picture sets the length, whichever input is longer.
    assert len(flat) == restored
    row = flat[0][24 * 64 : 25 * 64]

    def luma(level: int) -> float:
        return 16 + level * 219 / 255

    share = REGION_STRENGTH["faces"]
    assert abs(row[32] - (luma(50) + (luma(200) - luma(50)) * share)) <= 2
    assert abs(row[2] - (luma(50) + (luma(200) - luma(50)) * 0.8)) <= 2
    # Moving pictures: every frame matches the same mix done per pixel by
    # an expression, so no frame is paired with a neighbour's base.
    moving = (
        f"testsrc2=s=64x48:r=30000/1001,trim=end_frame={restored}",
        f"testsrc=s=32x24:r=30000/1001,trim=end_frame={base}",
    )
    fast = _mixed_frames(graph, *moving, restored)
    share_expr = strength_expr(0.8, box)
    slow_graph = (
        "[0:v]format=yuv420p[a];"
        "[1:v]scale=64:48:flags=lanczos,format=yuv420p,"
        "tpad=stop_mode=clone:stop_duration=1[b];"
        f"[a][b]blend=all_expr='B+(A-B)*({share_expr})':eof_action=pass[v]"
    )
    slow = _mixed_frames(slow_graph, *moving, restored)
    for got, want in zip(fast, slow, strict=True):
        assert max(abs(g - w) for g, w in zip(got, want, strict=True)) <= 2


def test_caps_targets_and_regions_from_gemini() -> None:
    face_shot = Shot(0, 2, contains=("faces", "text"))
    boxed = Shot(0, 2, contains=("faces",), regions=(FACE,))
    assert cap_strength(0.9, face_shot) == PROTECTED_STRENGTH
    assert cap_strength(0.9, boxed) == 0.9
    assert cap_strength(0.9, Shot(0, 2, contains=("fine_pattern",))) == 0.9
    shots = ShotList(shots=[Shot(0, 2), face_shot, Shot(4, 6, contains=("hands",))])
    assert locate_targets(shots) == [1, 2]
    found = [
        {
            "index": 1,
            "regions": [
                {"kind": "faces", "box_2d": [100, 100, 400, 300]},
                {"kind": "hands", "box_2d": [0, 0, 500, 500]},  # not in this shot
                {"kind": "text", "box_2d": [1, 2]},
                "junk",
            ],
        },
        {"index": True, "regions": []},
    ]
    updated = apply_regions(shots, found, [1, 2])
    assert [r.kind for r in updated.shots[1].regions] == ["faces"]
    assert updated.shots[2].regions == ()
    assert updated.shots[0] == shots.shots[0]


@pytest.mark.parametrize("text_box", [None, [1, 2], [0, 0, 1, 1]])
def test_partial_boxes_keep_the_global_cap(text_box: list[int] | None) -> None:
    shots = ShotList(shots=[Shot(0, 2, contains=("faces", "text"))])
    boxes: list[dict[str, Any]] = [{"kind": "faces", "box_2d": [100, 100, 400, 300]}]
    if text_box is not None:
        boxes.append({"kind": "text", "box_2d": text_box})
    shots = apply_regions(shots, [{"index": 0, "regions": boxes}], [0])
    assert [region.kind for region in shots.shots[0].regions] == ["faces"]
    assert cap_strength(0.9, shots.shots[0]) == PROTECTED_STRENGTH
    assert cap_strength(0.4, shots.shots[0]) == 0.4
    plan = apply_proposals(
        new_plan(shots, 2), shots, [{"index": 0, "strength": 0.9}], [0]
    )
    assert plan.recipes[0].strength == PROTECTED_STRENGTH
    plan, _ = apply_checks(
        plan, shots, [{"index": 0, "ok": False, "strength": 0.95}], [0]
    )
    assert plan.recipes[0].strength == PROTECTED_STRENGTH


@pytest.mark.parametrize("missing_kind", [None, "faces", "hands", "text"])
def test_global_cap_requires_boxes_for_every_protected_kind(
    missing_kind: str | None,
) -> None:
    kinds = ("faces", "hands", "text")
    shots = ShotList(shots=[Shot(0, 2, contains=(*kinds, "fine_pattern"))])
    boxes = [
        {"kind": kind, "box_2d": [100, 100, 400, 300]}
        for kind in kinds
        if kind != missing_kind
    ]
    shots = apply_regions(shots, [{"index": 0, "regions": boxes}], [0])
    strength = 0.9 if missing_kind is None else PROTECTED_STRENGTH
    assert cap_strength(0.9, shots.shots[0]) == strength
    plan = apply_proposals(
        new_plan(shots, 2), shots, [{"index": 0, "strength": 0.9}], [0]
    )
    assert plan.recipes[0].strength == strength


def test_spans_store_and_find_regions() -> None:
    span = UpscaleSpan(0, 4, regions=(FACE,))
    assert UpscaleSpan.from_dict(span.to_dict()) == span
    assert UpscaleSpan.from_dict({"start_s": 0, "end_s": 1}).regions == ()
    plan = UpscalePlan(spans=(UpscaleSpan(0, 2), span.__class__(2, 4, regions=(FACE,))))
    found = span_at(plan, 3)
    assert found is not None and found.regions == (FACE,)
    assert span_at(plan, 9) is None


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


def test_gemini_locate_regions_request(tmp_path: Path) -> None:
    jpg = tmp_path / "s.jpg"
    jpg.write_bytes(b"\xff\xd8")
    transport = _Transport({"shots": [{"index": 0, "regions": []}, 5]})
    client = GeminiClient(api_key="k", transport=transport)
    out = client.locate_regions(
        shots=[{"index": 0}],
        stills=[[jpg, jpg, jpg]],
        kinds=("faces", "text"),
        max_regions=6,
        job_dir=tmp_path,
    )
    assert out == [{"index": 0, "regions": []}]
    parts = transport.bodies[0]["contents"][0]["parts"]
    assert sum("inlineData" in part for part in parts) == 3
    assert "box_2d" in parts[0]["text"] and "faces, text" in parts[0]["text"]
    with pytest.raises(GeminiError, match="every shot"):
        client.locate_regions(shots=[{}], stills=[], kinds=(), max_regions=6)
    empty = GeminiClient(api_key="k", transport=_Transport({}))
    with pytest.raises(GeminiError, match="no regions"):
        empty.locate_regions(shots=[{}], stills=[[jpg]], kinds=(), max_regions=6)


def _gray(path: Path, seconds: int = 2) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:duration={seconds}:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


class _TextureEngine:
    """A brighter picture covered in a fine checkerboard, at the asked scale."""

    def upscale(self, request: UpscaleRequest) -> Path:
        width, height = 160 * request.scale, 90 * request.scale
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"nullsrc=size={width}x{height}:rate=10",
                "-vf",
                "geq=lum='188+50*(mod(floor(X/2)+floor(Y/2),2)*2-1)':cb=128:cr=128",
                "-frames:v",
                "20",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(request.output),
            ],
            check=True,
        )
        return request.output


def _half_detail(path: Path) -> tuple[float, float]:
    """Mean neighbour difference in the left and right thirds, first frame."""
    width = 320
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            "format=gray",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    rows = [raw[i : i + width] for i in range(0, len(raw), width)]

    def detail(lo: int, hi: int) -> float:
        diffs = [abs(row[x + 1] - row[x]) for row in rows for x in range(lo, hi)]
        return sum(diffs) / len(diffs)

    return detail(0, width // 3), detail(2 * width // 3, width - 1)


@needs_ffmpeg
def test_text_box_holds_flashvsr_down_in_a_real_render(tmp_path: Path) -> None:
    source = _gray(tmp_path / "in.mp4")
    output = tmp_path / "out.mp4"
    sign = Region("text", 0.0, 0.0, 0.4, 1.0)
    apply_range_revise(
        source,
        output,
        UpscaleSpan(0, 2, strength=1.0, regions=(sign,)),
        output,
        engine=_TextureEngine(),
        duration_s=2,
    )
    protected, free = _half_detail(output)
    assert free > 30
    assert protected / free == pytest.approx(REGION_STRENGTH["text"], abs=0.07)


class _Planner:
    def __init__(self, *, fail_locate: bool = False) -> None:
        self.fail_locate = fail_locate
        self.proposed: list[dict[str, Any]] = []

    def propose_segments(self, **kwargs: Any) -> list[dict[str, Any]]:
        return segments((0.0, kwargs["duration_s"], "anchor", ["faces"]))

    def locate_regions(self, **kwargs: Any) -> list[dict[str, Any]]:
        if self.fail_locate:
            raise GeminiError("quota")
        for frames in kwargs["stills"]:
            assert len(frames) == 3 and all(frame.is_file() for frame in frames)
        return [
            {
                "index": shot["index"],
                "regions": [{"kind": "faces", "box_2d": [100, 100, 600, 400]}],
            }
            for shot in kwargs["shots"]
        ]

    def propose_looks(self, **kwargs: Any) -> dict[str, Any]:
        self.proposed.extend(kwargs["shots"])
        return {
            "scale": 2,
            "shots": [
                {"index": shot["index"], "strength": 0.9, "look": {}, "reason": "r"}
                for shot in kwargs["shots"]
            ],
        }

    def check_previews(self, **kwargs: Any) -> list[dict[str, Any]]:
        return [{"index": shot["index"], "ok": True} for shot in kwargs["shots"]]


def _plan_job(tmp_path: Path, planner: _Planner) -> WizardController:
    from super_processor.upscale import FakeUpscaleEngine

    clip = _gray(tmp_path / "clip.mp4", seconds=3)
    controller = WizardController(
        tmp_path / "jobs",
        gemini=planner,  # type: ignore[arg-type]
        upscale_engine=FakeUpscaleEngine(),
    )
    pick_and_segment(controller, clip)
    assert approve_segments(controller).step is WizardStep.PREVIEW_SEGMENT
    return controller


@needs_ffmpeg
def test_wizard_boxes_faces_before_planning(tmp_path: Path) -> None:
    planner = _Planner()
    controller = _plan_job(tmp_path, planner)
    state = controller.current_state()
    assert state.job_id
    job_dir = controller.store.job_dir(state.job_id)
    shots = load_shots(job_dir)
    assert shots is not None
    assert [r.kind for r in shots.shots[0].regions] == ["faces"]
    assert planner.proposed[0]["protected_boxes"] == ["faces"]
    view = preview(controller)["shot"]
    # Boxed, so the shot keeps the strength Gemini asked for.
    assert view["strength"] == 0.9
    assert view["regions"][0]["kind"] == "faces"
    page = controller.render()
    assert "Protected inside the boxes: faces." in page
    assert 'class="box faces"' in page
    assert approve_preview(controller).step is WizardStep.READY_TO_RENDER
    controller.handle_post("/ready", {"action": ["render"]})
    plan = load_upscale_plan(job_dir)
    assert plan is not None and plan.spans[0].regions == shots.shots[0].regions


@needs_ffmpeg
def test_without_boxes_the_whole_shot_is_capped(tmp_path: Path) -> None:
    controller = _plan_job(tmp_path, _Planner(fail_locate=True))
    looks = preview(controller)
    assert looks["shot"]["strength"] == PROTECTED_STRENGTH
    assert "could not box" in looks["error"]
    assert "no boxes were placed" in controller.render()


@needs_ffmpeg
def test_locate_stills_cover_the_shot(tmp_path: Path) -> None:
    source = _gray(tmp_path / "in.mp4", seconds=2)
    shots = ShotList(shots=[Shot(0, 2, contains=("text",))])
    stills = locate_stills(source, tmp_path, shots, [0])
    assert [p.name for p in stills[0]] == [
        "locate_000_0.jpg",
        "locate_000_1.jpg",
        "locate_000_2.jpg",
    ]
    assert all(p.is_file() for p in stills[0])
