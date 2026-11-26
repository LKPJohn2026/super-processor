"""DPIR: only the editor's note turns it on, per scene, and the weights it
loads are exactly the pinned file.

The streaming tests use the stand-in GPU block of ``test_upscale`` (FFmpeg
in place of torch), so everything around DRUNet (weights, decoder, threads,
the delivery encode, the markers) is the real code. The last tests run the
real DRUNet on CUDA and are skipped without it.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import subprocess
import urllib.error
from pathlib import Path
from typing import Any

import pytest
from test_upscale import _moving, fake_span

from super_processor import dpir, upscale
from super_processor.dpir import (
    DPIR_URL,
    MAX_LEVEL,
    DpirError,
    ffmpeg_look,
    gpu_look,
    level_of,
    note_level,
)
from super_processor.look import ShotLook
from super_processor.shot_plan import (
    ShotPlan,
    ShotRecipe,
    apply_proposals,
    keep_editor_choices,
    request_redo,
    shots_payload,
)
from super_processor.shots import Shot, ShotList
from super_processor.upscale import (
    FakeUpscaleEngine,
    SpanEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
    encode_delivery,
    plain_delivery,
    replace_overlapping,
    video_stream_facts,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg required"
)


# -- what a note asks for ---------------------------------------------------


@pytest.mark.parametrize(
    ("note", "current", "want"),
    [
        ("more denoise", 0, 1),
        ("More denoising please", 1, 2),
        ("denoise more", 0, 1),
        ("stronger denoise", 2, 3),
        ("it's still noisy", 0, 1),
        ("way too grainy in the dark", 0, 1),
        ("too much noise on the wall", 0, 1),
        ("less noise", 0, 1),
        # Grain is also the look's own finishing grain: Gemini's to lower.
        ("less grain", 0, 0),
        ("fewer grain on the faces", 1, 1),
        ("reduce the noise", 0, 1),
        ("get rid of noise", 0, 1),
        ("clean up the speckles", 0, 1),
        ("more noise reduction", 0, 1),
        ("use DPIR", 0, 1),
        ("dpir on", 0, 1),
        # Capped at the top, floored at zero.
        ("more denoise", MAX_LEVEL, MAX_LEVEL),
        ("less denoise", 0, 0),
        ("less denoise", 2, 1),
        ("too much denoise, faces look waxy", 2, 1),
        ("too smooth", 3, 2),
        ("over-denoised", 1, 0),
        ("no DPIR", 3, 0),
        ("no denoise at all", 2, 0),
        ("turn off dpir", 1, 0),
        ("dpir off", 1, 0),
        ("don't denoise this one", 2, 0),
        ("without the denoise", 2, 0),
        # "No more" keeps the level where it is.
        ("no more denoise", 2, 2),
        # Negated asks change nothing.
        ("it's not noisy", 1, 1),
        ("the grass isn't too grainy", 0, 0),
        ("don't make it noisy", 0, 0),
        # Unrelated notes keep it.
        ("make it brighter", 2, 2),
        ("warmer skin and 2x", 0, 0),
        ("", 1, 1),
        # Each clause counts; the last one wins for on/off.
        ("still noisy, and more denoise", 0, 2),
        ("more denoise. actually no dpir", 1, 0),
        ("no dpir; still noisy", 2, 1),
        # A curly apostrophe reads as a straight one.
        ("don’t denoise", 2, 0),
    ],
)
def test_note_level_follows_the_editor_words(
    note: str, current: int, want: int
) -> None:
    assert note_level(note, current) == want


@pytest.mark.parametrize("raw", [True, False, None, -1, 4, 99, 1.0, 2.5, "2", [1], {}])
def test_a_stored_level_that_is_not_a_level_is_off(raw: Any) -> None:
    assert level_of(raw) == 0
    # And a note works from 0 on it, not from the bad value.
    assert note_level("more denoise", raw) == 1


def test_levels_map_to_drunet_noise_levels() -> None:
    sigmas = [dpir.sigma(level) for level in range(1, MAX_LEVEL + 1)]
    assert sigmas == sorted(sigmas)
    assert sigmas[1] == pytest.approx(15 / 255)
    for bad in (0, MAX_LEVEL + 1, -1):
        with pytest.raises(DpirError):
            dpir.sigma(bad)


def test_the_look_stand_ins() -> None:
    look = ShotLook(deblock=0.4, denoise=0.3, contrast=1.1)
    assert ffmpeg_look(look, 0) == look
    assert gpu_look(look, 0) == look
    assert ffmpeg_look(look, 2).denoise == dpir.FALLBACK_DENOISE
    assert "hqdn3d=6.000" in ffmpeg_look(look, 2).clean_filter()
    on_gpu = gpu_look(look, 1)
    assert on_gpu.denoise == 0.0
    assert on_gpu.clean_filter().startswith("deblock=")
    assert "hqdn3d" not in on_gpu.clean_filter()
    assert (on_gpu.deblock, on_gpu.contrast) == (look.deblock, look.contrast)


def test_batches_follow_the_frame_size() -> None:
    assert dpir.batch_size(1440, 1080) == 2
    assert dpir.batch_size(3840, 2160) == 1
    assert dpir.batch_size(64, 48) == 4
    assert dpir.batch_size(0, 0) == 4


# -- the weights ------------------------------------------------------------

_FAKE = b"pretend DRUNet weights" * 64


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pin the fake bytes as the weights; returns the models directory."""
    monkeypatch.setattr(dpir, "DPIR_SHA256", hashlib.sha256(_FAKE).hexdigest())
    monkeypatch.setattr(dpir, "DPIR_SIZE", len(_FAKE))
    monkeypatch.delenv(dpir.DPIR_WEIGHTS_ENV, raising=False)
    monkeypatch.delenv(dpir.DPIR_DOWNLOAD_ENV, raising=False)
    return tmp_path / "models"


class _Response(io.BytesIO):
    def __init__(self, data: bytes, fail_after: int | None = None) -> None:
        super().__init__(data)
        self.fail_after = fail_after
        self.closed_by_us = False

    def read(self, size: int | None = -1) -> bytes:
        if self.fail_after is not None and self.tell() >= self.fail_after:
            raise ConnectionResetError("connection reset")
        return super().read(size)

    def close(self) -> None:
        self.closed_by_us = True
        super().close()


def _opener(data: bytes, seen: list[str], **kwargs: Any) -> Any:
    def open_url(url: str) -> _Response:
        seen.append(url)
        return _Response(data, **kwargs)

    return open_url


def _leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []


def test_missing_weights_are_fetched_from_the_pinned_url(pinned: Path) -> None:
    seen: list[str] = []
    path = dpir.resolve_weights(pinned, opener=_opener(_FAKE, seen))
    assert seen == [DPIR_URL]
    assert path == pinned / "DPIR" / "drunet_color.pth"
    assert path.read_bytes() == _FAKE
    assert _leftovers(path.parent) == ["drunet_color.pth"]
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # Present and right: no second download.
    assert dpir.resolve_weights(pinned, opener=_opener(b"", seen)) == path
    assert seen == [DPIR_URL]


def test_the_pinned_url_is_github_over_https() -> None:
    assert DPIR_URL.startswith("https://github.com/cszn/KAIR/releases/download/")
    assert len(dpir.DPIR_SHA256) == 64


@pytest.mark.parametrize(
    "data",
    [_FAKE[:-1], _FAKE[:-1] + b"X", b""],
    ids=["short", "wrong-bytes", "empty"],
)
def test_a_wrong_download_is_never_kept(pinned: Path, data: bytes) -> None:
    with pytest.raises(DpirError, match="not the expected file"):
        dpir.resolve_weights(pinned, opener=_opener(data, []))
    assert _leftovers(pinned / "DPIR") == []


def test_an_oversized_download_stops_at_the_pinned_size(pinned: Path) -> None:
    response = _Response(_FAKE + b"more and more")
    with pytest.raises(DpirError, match="larger than"):
        dpir.resolve_weights(pinned, opener=lambda url: response)
    assert response.closed_by_us
    assert _leftovers(pinned / "DPIR") == []


@pytest.mark.parametrize(
    "error",
    [urllib.error.URLError("no route"), TimeoutError("timed out"), OSError("dns")],
)
def test_an_offline_machine_is_told_where_to_put_the_file(
    pinned: Path, error: BaseException
) -> None:
    def fails(url: str) -> Any:
        raise error

    with pytest.raises(DpirError) as caught:
        dpir.resolve_weights(pinned, opener=fails)
    message = str(caught.value)
    assert DPIR_URL in message and dpir.DPIR_WEIGHTS_ENV in message
    assert _leftovers(pinned / "DPIR") == []


def test_a_download_cut_short_keeps_nothing(pinned: Path) -> None:
    opener = _opener(_FAKE, [], fail_after=len(_FAKE) // 2)
    with pytest.raises(DpirError, match="could not download"):
        dpir.resolve_weights(pinned, opener=opener)
    assert _leftovers(pinned / "DPIR") == []


def test_a_damaged_file_is_fetched_again(pinned: Path) -> None:
    path = pinned / "DPIR" / "drunet_color.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"truncated")
    seen: list[str] = []
    assert dpir.resolve_weights(pinned, opener=_opener(_FAKE, seen)) == path
    assert seen and path.read_bytes() == _FAKE


def test_a_file_named_by_the_env_is_never_downloaded_over(
    pinned: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mine = tmp_path / "mine.pth"
    monkeypatch.setenv(dpir.DPIR_WEIGHTS_ENV, str(mine))
    seen: list[str] = []
    with pytest.raises(DpirError, match="not found"):
        dpir.resolve_weights(pinned, opener=_opener(_FAKE, seen))
    mine.write_bytes(b"something else")
    with pytest.raises(DpirError, match="not the expected"):
        dpir.resolve_weights(pinned, opener=_opener(_FAKE, seen))
    assert seen == []
    assert mine.read_bytes() == b"something else"
    mine.write_bytes(_FAKE)
    assert dpir.resolve_weights(pinned, opener=_opener(_FAKE, seen)) == mine


@pytest.mark.parametrize("off", ["0", "false", "No", "off"])
def test_the_download_can_be_turned_off(
    pinned: Path, monkeypatch: pytest.MonkeyPatch, off: str
) -> None:
    monkeypatch.setenv(dpir.DPIR_DOWNLOAD_ENV, off)
    seen: list[str] = []
    with pytest.raises(DpirError, match="turns the download off"):
        dpir.resolve_weights(pinned, opener=_opener(_FAKE, seen))
    path = pinned / "DPIR" / "drunet_color.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bad")
    with pytest.raises(DpirError, match="not the expected"):
        dpir.resolve_weights(pinned, opener=_opener(_FAKE, seen))
    assert seen == []


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth",
        "https://evil.example/drunet_color.pth",
        "https://github.com.evil.example/x",
        "https://user@evil.example/x",
        "file:///etc/passwd",
        "ftp://objects.githubusercontent.com/x",
        "https:///nohost",
    ],
)
def test_only_githubs_hosts_over_https_are_fetched(url: str) -> None:
    with pytest.raises(DpirError, match="only GitHub over HTTPS"):
        dpir._require_github(url)
    with pytest.raises(DpirError, match="only GitHub over HTTPS"):
        dpir._GitHubOnly().redirect_request(None, None, 302, "Found", {}, url)


@pytest.mark.parametrize(
    "url",
    [
        DPIR_URL,
        "https://release-assets.githubusercontent.com/github-production-release-asset/1/x",
        "https://objects.githubusercontent.com/x",
    ],
)
def test_githubs_own_hosts_are_allowed(url: str) -> None:
    dpir._require_github(url)


def test_the_bytes_loaded_are_the_bytes_checked(pinned: Path, tmp_path: Path) -> None:
    path = tmp_path / "w.pth"
    path.write_bytes(_FAKE)
    assert dpir.read_verified(path) == _FAKE
    path.write_bytes(_FAKE[::-1])
    with pytest.raises(DpirError, match="not the expected"):
        dpir.read_verified(path)
    with pytest.raises(DpirError, match="cannot read"):
        dpir.read_verified(tmp_path / "missing.pth")


def test_weights_load_as_tensors_only(
    pinned: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    buffer = io.BytesIO()
    torch.save({"w": torch.ones(2)}, buffer)
    data = buffer.getvalue()
    monkeypatch.setattr(dpir, "DPIR_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(dpir, "DPIR_SIZE", len(data))
    path = tmp_path / "w.pth"
    path.write_bytes(data)
    state = dpir.load_state(torch, path)
    assert torch.equal(state["w"], torch.ones(2))

    class Payload:
        def __reduce__(self) -> Any:
            return (print, ("ran code",))

    hostile = io.BytesIO()
    torch.save({"x": Payload()}, hostile)
    data = hostile.getvalue()
    monkeypatch.setattr(dpir, "DPIR_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(dpir, "DPIR_SIZE", len(data))
    path.write_bytes(data)
    with pytest.raises(Exception, match="(?i)weights only|unsupported|global"):
        dpir.load_state(torch, path)


def test_missing_gpu_modules_name_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_torch(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(dpir.importlib, "import_module", no_torch)
    with pytest.raises(DpirError, match=r"ancient-processor\[gpu\]"):
        dpir.import_gpu_modules()


# -- the recipe -------------------------------------------------------------


def _plan(*recipes: ShotRecipe) -> ShotPlan:
    return ShotPlan(scale=2, recipes=list(recipes), generation="ab")


def _shots(count: int) -> ShotList:
    return ShotList(shots=[Shot(float(i * 4), float(i * 4 + 4)) for i in range(count)])


def test_a_redo_note_sets_only_its_own_scene() -> None:
    plan = _plan(ShotRecipe(stale=False), ShotRecipe(stale=False))
    plan = request_redo(plan, 1, "  it's still noisy  ")
    assert [r.dpir for r in plan.recipes] == [0, 1]
    plan = request_redo(plan, 1, "more denoise")
    assert plan.recipes[1].dpir == 2
    plan = request_redo(plan, 1, "brighter please")
    assert plan.recipes[1].dpir == 2
    plan = request_redo(plan, 1, "no dpir")
    assert plan.recipes[1].dpir == 0
    assert plan.recipes[0].dpir == 0


def test_gemini_cannot_set_or_clear_dpir() -> None:
    shots = _shots(2)
    plan = _plan(ShotRecipe(dpir=2), ShotRecipe())
    proposals = [
        {"index": 0, "strength": 0.4, "look": {"denoise": 0.9}, "dpir": 0},
        {"index": 1, "strength": 0.4, "look": {"dpir": 3}, "dpir": 3},
    ]
    plan = apply_proposals(plan, shots, proposals, [0, 1])
    assert [r.dpir for r in plan.recipes] == [2, 0]
    # A proposal that is missing falls back to the measurements, keeping it.
    plan = apply_proposals(plan, shots, [], [0])
    assert plan.recipes[0].dpir == 2
    assert keep_editor_choices(ShotRecipe(), ShotRecipe(dpir=3)).dpir == 3


def test_the_level_is_saved_and_read_back_safely() -> None:
    recipe = ShotRecipe(dpir=2)
    assert ShotRecipe.from_dict(recipe.to_dict()).dpir == 2
    for bad in (True, 7, -2, "3", 1.5, None):
        data = {**recipe.to_dict(), "dpir": bad}
        assert ShotRecipe.from_dict(data).dpir == 0


def test_gemini_is_told_a_scene_has_dpir() -> None:
    shots = _shots(2)
    plan = _plan(ShotRecipe(dpir=1, reason="x"), ShotRecipe(reason="y"))
    items = shots_payload(shots, plan, [0, 1])
    assert items[0]["dpir"] == 1
    assert "dpir" not in items[1]


def test_the_span_plan_carries_each_scenes_level() -> None:
    shots = _shots(2)
    plan = _plan(ShotRecipe(dpir=3), ShotRecipe())
    spans = plan.to_upscale_plan(shots).spans
    assert [span.dpir for span in spans] == [3, 0]
    again = UpscalePlan.from_dict(UpscalePlan(spans=spans).to_dict())
    assert [span.dpir for span in again.spans] == [3, 0]


@pytest.mark.parametrize("bad", [True, 7, -1, "2", 2.0, None])
def test_spans_and_requests_refuse_a_bad_level(bad: Any, tmp_path: Path) -> None:
    with pytest.raises(UpscaleError, match="dpir"):
        UpscaleSpan(0.0, 1.0, dpir=bad)
    with pytest.raises(UpscaleError, match="dpir"):
        UpscaleRequest(tmp_path / "a.mp4", tmp_path / "b.mp4", dpir=bad)
    # Read from a saved plan, a bad level is off instead of an error.
    data = {**UpscaleSpan(0.0, 1.0).to_dict(), "dpir": bad}
    assert UpscaleSpan.from_dict(data).dpir == 0


def test_a_result_note_sets_dpir_in_its_range_only() -> None:
    plan = UpscalePlan(
        spans=(
            UpscaleSpan(0.0, 4.0, dpir=1),
            UpscaleSpan(4.0, 8.0),
            UpscaleSpan(8.0, 12.0),
        )
    )
    patch = UpscaleSpan(4.0, 8.0, strength=0.4)
    updated = replace_overlapping(
        plan, patch, 12.0, preserve_shot_settings=True, note="too noisy here"
    )
    by_start = {span.start_s: span.dpir for span in updated.spans}
    assert by_start == {0.0: 1, 4.0: 1, 8.0: 0}
    # Without the words, the levels stay.
    kept = replace_overlapping(plan, patch, 12.0, preserve_shot_settings=True)
    assert {span.start_s: span.dpir for span in kept.spans} == {0.0: 1, 4.0: 0, 8.0: 0}


# -- the stream -------------------------------------------------------------


def _first_frame(source: Path, *vf: str) -> bytes:
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), *vf, "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
        check=True,
        capture_output=True,
    ).stdout  # fmt: skip


@pytest.fixture
def weights_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[int]:
    """Stand in for the weights check; counts the times it was asked."""
    asked: list[int] = []

    def resolve(models: Path, **kwargs: Any) -> Path:
        asked.append(1)
        return tmp_path / "drunet_color.pth"

    monkeypatch.setattr(dpir, "resolve_weights", resolve)
    return asked


@needs_ffmpeg
def test_dpir_replaces_hqdn3d_before_span(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights_ok: list[int]
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=1)
    look = ShotLook(deblock=0.5, denoise=0.5)
    output = tmp_path / "out" / "clip.mp4"
    SpanEngine().upscale(UpscaleRequest(source, output, look=look, dpir=2))
    assert video_stream_facts(output)[0] == 24
    assert fake.dpir_loads == 1 and weights_ok == [1]
    assert {call[:2] for call in fake.calls} == {(2, 2)}
    # The decoder only deblocked: DRUNet, not hqdn3d, does the denoising.
    assert fake.firsts[0] == _first_frame(
        source, "-vf", gpu_look(look, 2).clean_filter()
    )
    assert fake.firsts[0] != _first_frame(source, "-vf", look.clean_filter())
    marker = json.loads(upscale.delivery_marker_path(output).read_text("utf-8"))
    assert marker["blend"]["dpir"] == 2


@needs_ffmpeg
def test_previews_load_dpir_only_when_a_scene_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights_ok: list[int]
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=1)
    out = tmp_path / "out"
    SpanEngine().upscale_many(
        [UpscaleRequest(source, out / "a.mp4"), UpscaleRequest(source, out / "b.mp4")]
    )
    assert (fake.loads, fake.dpir_loads, weights_ok) == (1, 0, [])
    SpanEngine().upscale_many(
        [
            UpscaleRequest(source, out / "c.mp4"),
            UpscaleRequest(source, out / "d.mp4", dpir=3),
        ]
    )
    assert (fake.loads, fake.dpir_loads, weights_ok) == (2, 1, [1])
    levels = [call[0] for call in fake.calls]
    assert 0 in levels and 3 in levels


@needs_ffmpeg
def test_a_missing_download_is_a_clear_render_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = fake_span(tmp_path, monkeypatch)

    def offline(models: Path, **kwargs: Any) -> Path:
        raise DpirError("could not download the DPIR weights (no route)")

    monkeypatch.setattr(dpir, "resolve_weights", offline)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=1)
    with pytest.raises(UpscaleError, match="could not download"):
        SpanEngine().upscale(UpscaleRequest(source, tmp_path / "o.mp4", dpir=1))
    assert fake.loads == 0


@needs_ffmpeg
@pytest.mark.parametrize("scale", [1, 2])
def test_a_pass_through_scene_is_cleaned_by_dpir_on_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights_ok: list[int], scale: int
) -> None:
    fake = fake_span(tmp_path, monkeypatch)
    monkeypatch.setattr(upscale, "dpir_runs_here", lambda: True)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=1)
    dest = tmp_path / "out" / "after.mp4"
    request = UpscaleRequest(source, tmp_path / "unused.mp4", scale=scale, dpir=1)
    assert plain_delivery(request, dest) == dest
    assert upscale._frame_size(dest) == (160 * scale, 90 * scale)
    assert video_stream_facts(dest)[0] == 24
    # No SPAN was loaded or run: DRUNet and the resize only.
    assert fake.loads == 0 and fake.dpir_loads == 1
    assert {call for call in fake.calls} == {(1, scale, None)}
    assert upscale.is_delivery(dest)
    marker = json.loads(upscale.delivery_marker_path(dest).read_text("utf-8"))
    assert marker["blend"]["dpir"] == 1 and marker["blend"]["strength"] == 0.0


@needs_ffmpeg
def test_without_a_gpu_a_pass_through_scene_gets_the_strongest_hqdn3d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights_ok: list[int]
) -> None:
    monkeypatch.setattr(upscale, "dpir_runs_here", lambda: False)
    commands: list[list[str]] = []
    real = upscale._run_ffmpeg

    def spy(command: list[str], **kwargs: Any) -> Any:
        commands.append(command)
        return real(command, **kwargs)

    monkeypatch.setattr(upscale, "_run_ffmpeg", spy)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=1)
    dest = tmp_path / "after.mp4"
    look = ShotLook(denoise=0.2)
    plain_delivery(UpscaleRequest(source, dest, scale=1, look=look, dpir=3), dest)
    picture = commands[-1][commands[-1].index("-vf") + 1]
    assert "hqdn3d=6.000:4.500:6.000:4.500" in picture
    assert weights_ok == []
    marker = json.loads(upscale.delivery_marker_path(dest).read_text("utf-8"))
    assert marker["look"]["denoise"] == dpir.FALLBACK_DENOISE


@needs_ffmpeg
def test_the_fake_engine_stands_in_with_hqdn3d(tmp_path: Path) -> None:
    source = tmp_path / "in.mp4"
    _moving(source, seconds=1)
    look = ShotLook(denoise=0.1)
    plain = FakeUpscaleEngine().upscale(
        UpscaleRequest(source, tmp_path / "plain.mp4", look=look)
    )
    strong = FakeUpscaleEngine().upscale(
        UpscaleRequest(source, tmp_path / "strong.mp4", look=look, dpir=1)
    )
    assert plain.read_bytes() != strong.read_bytes()


@needs_ffmpeg
def test_a_file_made_without_dpir_is_never_reused_for_one_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, weights_ok: list[int]
) -> None:
    fake_span(tmp_path, monkeypatch)
    source = tmp_path / "in.mp4"
    _moving(source, seconds=1)
    made = SpanEngine().upscale(UpscaleRequest(source, tmp_path / "up.mp4"))
    before = made.read_bytes()
    dest = tmp_path / "delivery.mp4"
    encode_delivery(made, dest, blend_with=source, strength=0.5, dpir=2)
    # Encoded again (a blend with the source), not renamed.
    assert made.exists() and made.read_bytes() == before
    marker = json.loads(upscale.delivery_marker_path(dest).read_text("utf-8"))
    assert marker["blend"]["dpir"] == 2


# -- the real network --------------------------------------------------------


def _real_weights() -> Path:
    path = dpir.weights_path(upscale.models_dir())
    if not path.is_file():
        pytest.skip(f"no DPIR weights at {path}")
    return path


@pytest.mark.gpu
def test_real_drunet_removes_noise_and_keeps_the_size() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    model = dpir.load_dpir(_real_weights())
    gen = torch.Generator(device="cuda").manual_seed(3)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, 77, device="cuda"),
        torch.linspace(0, 1, 101, device="cuda"),
        indexing="ij",
    )
    clean = torch.stack([xx, yy, (xx + yy) / 2]).unsqueeze(0).repeat(2, 1, 1, 1)
    noise = torch.randn(clean.shape, generator=gen, device="cuda") * (15 / 255)
    noisy = (clean + noise).clamp(0, 1)
    with torch.inference_mode():
        out = dpir.denoise(model, noisy, 2)
    assert out.shape == clean.shape and out.dtype == torch.float32
    assert out.min().item() >= 0.0 and out.max().item() <= 1.0

    def error(x: Any) -> float:
        return float(((x - clean) ** 2).mean().sqrt().item())

    assert error(out) < error(noisy) / 4
    with torch.inference_mode():
        strongest = dpir.denoise(model, noisy, 3)
    # A higher level smooths more.
    assert strongest.diff(dim=-1).abs().mean() <= out.diff(dim=-1).abs().mean()


def test_a_stop_ends_the_download_and_keeps_nothing(
    pinned: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Stopped(Exception):
        pass

    calls: list[int] = []

    def stop_on_third() -> None:
        calls.append(1)
        if len(calls) == 3:
            raise Stopped

    big = _FAKE * 3000  # several chunks
    monkeypatch.setattr(dpir, "DPIR_SHA256", hashlib.sha256(big).hexdigest())
    monkeypatch.setattr(dpir, "DPIR_SIZE", len(big))
    with pytest.raises(Stopped):
        dpir.resolve_weights(pinned, opener=_opener(big, []), should_stop=stop_on_third)
    assert _leftovers(pinned / "DPIR") == []


def test_a_download_past_its_deadline_keeps_nothing(
    pinned: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(dpir, "_DOWNLOAD_DEADLINE_S", -1.0)
    with pytest.raises(DpirError, match="took too long"):
        dpir.resolve_weights(pinned, opener=_opener(_FAKE, []))
    assert _leftovers(pinned / "DPIR") == []


def test_neighbouring_scenes_with_different_dpir_render_apart() -> None:
    spans = [
        UpscaleSpan(0.0, 5.0),
        UpscaleSpan(5.0, 10.0, dpir=2),
        UpscaleSpan(10.0, 15.0, dpir=2),
        UpscaleSpan(15.0, 20.0),
    ]
    merged = upscale._coalesce(spans)
    assert [(s.start_s, s.end_s, s.dpir) for s in merged] == [
        (0.0, 5.0, 0),
        (5.0, 15.0, 2),
        (15.0, 20.0, 0),
    ]


def test_a_broken_gpu_install_means_no_dpir(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> bool:
        raise OSError("libcudart.so: cannot open shared object file")

    monkeypatch.setattr(upscale, "cuda_available", broken)
    assert upscale.dpir_runs_here() is False
    monkeypatch.setattr(upscale, "cuda_available", lambda: True)

    def no_spandrel(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(upscale.importlib, "import_module", no_spandrel)
    assert upscale.dpir_runs_here() is False
