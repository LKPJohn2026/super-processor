"""Tests for Cython estimator primitives and look sampling."""

from __future__ import annotations

import math
import random
import shutil
import subprocess
from array import array
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from super_processor import (
    histogram_u8,
    mean_luma,
    percentile_u8,
    sad_u8,
    variance_u8,
)
from super_processor.estimators import (
    SHAKE_HEIGHT,
    SHAKE_RUN,
    SHAKE_WIDTH,
    EstimatorError,
    LookEstimates,
    _global_shift,
    _noise_residual,
    _run_shake,
    choose_sample_windows,
    ensure_decoder,
    estimate_look,
    extract_gray_frame,
    extract_gray_run,
    extract_rgb_means,
    load_estimates,
    shake_stride,
    write_estimates,
)
from super_processor.probe import MediaFacts, StreamFacts


def test_histogram_and_percentile() -> None:
    samples = bytes([0, 0, 64, 128, 255, 255])
    hist = array("Q", [0]) * 256
    histogram_u8(samples, memoryview(hist))
    assert hist[0] == 2
    assert hist[255] == 2
    assert percentile_u8(samples, 0.0) == 0.0
    assert percentile_u8(samples, 100.0) == 255.0
    assert percentile_u8(samples, 50.0) in {64.0, 128.0}


def test_variance_sad_and_mean() -> None:
    left = bytes([10, 20, 30, 40])
    right = bytes([10, 25, 25, 50])
    assert variance_u8(left) == pytest.approx(125.0)
    assert sad_u8(left, right) == 20
    with pytest.raises(ValueError):
        sad_u8(left, bytes([1, 2]))
    assert mean_luma(bytes([16, 64, 128, 235])) == pytest.approx(110.75)


def _facts(
    duration_s: float | None = 30.0, *, source: str = "/tmp/x.mp4", video: bool = True
) -> MediaFacts:
    return MediaFacts(
        schema_version=1,
        source_path=source,
        format_name="mp4",
        format_long_name="MP4",
        duration_s=duration_s,
        size_bytes=1,
        bit_rate=1,
        streams=[],
        has_video=video,
        has_audio=False,
        is_vfr=False,
    )


def test_sample_windows_sit_at_head_middle_and_tail() -> None:
    windows = choose_sample_windows(_facts(30.0))
    assert [(w.label, w.start_s, w.duration_s) for w in windows] == [
        ("head", 3.0, 0.4),
        ("mid", 15.0, 0.4),
        ("tail", 25.5, 0.4),
    ]
    assert [w.label for w in choose_sample_windows(_facts(30.0), count=1)] == ["head"]


def test_sample_windows_of_a_very_short_clip_stay_inside_it() -> None:
    # A 0.3s clip is treated as 0.5s; every window must still end in time.
    windows = choose_sample_windows(_facts(0.3))
    for window in windows:
        assert window.duration_s == 0.2
        assert 0.0 <= window.start_s <= 0.5 - 0.2 - 0.05 + 1e-9
    # No duration at all is treated as one second.
    assert choose_sample_windows(_facts(None))[1].start_s == pytest.approx(0.5)


PIXELS = 160 * 90


def _run_estimates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gray: Callable[[int], bytes],
    rgb: tuple[float, float, float] = (120.0, 120.0, 120.0),
    runs: Callable[[int], list[bytes]] | None = None,
) -> LookEstimates:
    """Estimate from fake samples; ``gray`` and ``runs`` get the call
    number. Runs default to a still picture."""
    calls = iter(range(1000))
    run_calls = iter(range(1000))
    monkeypatch.setattr(
        "super_processor.estimators.extract_gray_frame",
        lambda *_args, **_kwargs: gray(next(calls)),
    )
    still = _shaky_run([(0, 0)] * SHAKE_RUN)
    monkeypatch.setattr(
        "super_processor.estimators.extract_gray_run",
        lambda *_args, **_kwargs: (runs or (lambda _n: still))(next(run_calls)),
    )
    monkeypatch.setattr(
        "super_processor.estimators.extract_rgb_means",
        lambda *_args, **_kwargs: rgb,
    )
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    return estimate_look(source, _facts(30.0, source=str(source)))


def _spread() -> bytes:
    """A frame using the whole video range evenly, mid-grey on average."""
    return bytes(16 + (i * 220) // PIXELS for i in range(PIXELS))


def test_a_well_exposed_still_picture_needs_no_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _spread()
    estimates = _run_estimates(tmp_path, monkeypatch, lambda _n: frame)
    assert not estimates.contrast.enabled
    assert estimates.contrast.params == {
        "contrast": 1.0,
        "brightness": 0.0,
        "gamma": 1.0,
    }
    assert not estimates.white_balance.enabled
    assert estimates.white_balance.params == {"temperature": 6500.0, "tint": 0.0}
    # A still picture: no motion, so no stabilising.
    assert not estimates.stabilize.enabled
    assert estimates.metrics["shake_jitter_pct"] == 0.0
    assert len(estimates.windows) == 3


def test_a_compressed_dark_picture_asks_for_more_contrast_and_light(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Half the pixels at 40, half at 60: a narrow, dark span.
    frame = bytes([40, 60]) * (PIXELS // 2)
    estimates = _run_estimates(tmp_path, monkeypatch, lambda _n: frame)
    assert estimates.contrast.enabled
    assert estimates.contrast.params["contrast"] == 2.0  # clamped from 9.0
    assert estimates.contrast.params["brightness"] == pytest.approx(
        (128 - 50) / 255, abs=1e-3
    )
    assert estimates.metrics["luma_mean"] == pytest.approx(50.0)


def test_a_warm_cast_gets_a_cooling_white_balance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _spread()
    estimates = _run_estimates(
        tmp_path, monkeypatch, lambda _n: frame, rgb=(180.0, 120.0, 80.0)
    )
    wb = estimates.white_balance
    # Too much red over blue: correct by raising the temperature, like the
    # too_warm treatments do.
    assert wb.enabled
    assert wb.params["temperature"] > 7500.0
    assert estimates.metrics["wb_cast"] < 0


def test_a_clean_gradient_is_not_mistaken_for_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _spread()
    estimates = _run_estimates(tmp_path, monkeypatch, lambda _n: frame)
    assert not estimates.denoise.enabled


def test_grainy_frames_ask_for_denoise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    noise = random.Random(7).randbytes(PIXELS)
    estimates = _run_estimates(tmp_path, monkeypatch, lambda _n: noise)
    assert estimates.denoise.enabled
    assert estimates.denoise.params["strength"] == 1.0
    flat = bytes([128]) * PIXELS
    quiet = _run_estimates(tmp_path, monkeypatch, lambda _n: flat)
    assert not quiet.denoise.enabled
    assert quiet.denoise.params == {"strength": 0.0}


def _sized(width: int, height: int, source: Path) -> MediaFacts:
    facts = _facts(30.0, source=str(source))
    facts.streams.append(
        StreamFacts(index=0, codec_type="video", width=width, height=height)
    )
    return facts


def _grain_frame(side: int, sigma: float, seed: int) -> bytes:
    """A soft old-film picture (a dark-to-light ramp) with Gaussian grain."""
    rng = random.Random(seed)
    return bytes(
        max(0, min(255, round(40 + 170 * (i % side) / side + rng.gauss(0, sigma))))
        for i in range(side * side)
    )


def _grain_estimates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
    sigma: float,
) -> tuple[LookEstimates, list[dict[str, Any]]]:
    crops: list[dict[str, Any]] = []
    calls = iter(range(1000))

    def extract(*_args: Any, **kwargs: Any) -> bytes:
        n = next(calls)
        if kwargs.get("crop"):
            crops.append(kwargs)
            return _grain_frame(kwargs["width"], sigma, n)
        # The scaled 160x90 frames other estimates use are shrunk so far
        # that grain averages out; they must not decide denoise.
        return _spread()

    monkeypatch.setattr("super_processor.estimators.extract_gray_frame", extract)
    monkeypatch.setattr(
        "super_processor.estimators.extract_gray_run",
        lambda *_args, **_kwargs: _shaky_run([(0, 0)] * SHAKE_RUN),
    )
    monkeypatch.setattr(
        "super_processor.estimators.extract_rgb_means",
        lambda *_args, **_kwargs: (120.0, 120.0, 120.0),
    )
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    return estimate_look(source, _sized(*size, source)), crops


@pytest.mark.parametrize(
    ("size", "side"),
    [((640, 480), 256), ((320, 240), 240), ((480, 360), 256), ((176, 144), 144)],
)
def test_grain_is_measured_on_a_crop_at_the_source_s_own_pixels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size: tuple[int, int],
    side: int,
) -> None:
    # SD film transfers, phone and web video, and QCIF: none is scaled.
    _estimates, crops = _grain_estimates(tmp_path, monkeypatch, size, 0.0)
    assert len(crops) == 3
    assert all(c["width"] == c["height"] == side for c in crops)


@pytest.mark.parametrize("size", [(640, 480), (854, 480), (320, 240)])
def test_heavy_grain_on_old_or_low_resolution_video_asks_for_denoise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    estimates, _ = _grain_estimates(tmp_path, monkeypatch, size, 8.0)
    assert estimates.denoise.enabled
    assert 0.6 < estimates.denoise.params["strength"] < 1.0
    assert 7.0 < estimates.metrics["noise_residual"] ** 0.5 < 9.0


@pytest.mark.parametrize("sigma", [0.0, 1.0, 2.0])
def test_soft_or_lightly_grained_old_video_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sigma: float
) -> None:
    # Compressed SD transfers measure sigma 0.6-1.2 on real archive films;
    # light grain is part of an old film's look.
    estimates, _ = _grain_estimates(tmp_path, monkeypatch, (640, 480), sigma)
    assert not estimates.denoise.enabled
    assert estimates.denoise.params == {"strength": 0.0}


def test_noise_residual_ignores_the_picture_s_edges() -> None:
    # Hard vertical bars: every 8th neighbour pair is a 200-level edge.
    side = 64
    bars = bytes(220 if (i % side) // 8 % 2 else 20 for i in range(side * side))
    assert _noise_residual(bars, side) == 0.0
    # A frame too small to have neighbours both ways has no measure.
    assert _noise_residual(bytes(10), 10) == 0.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize(("noise", "grainy"), [(0, False), (40, True)])
def test_a_real_sd_clip_s_grain_decides_denoise(
    tmp_path: Path, noise: int, grainy: bool
) -> None:
    source = tmp_path / "sd.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i",
         "gradients=size=640x480:rate=10:duration=2:speed=0.01,format=gray,"
         f"noise=alls={noise}:allf=t",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )  # fmt: skip
    facts = _sized(640, 480, source)
    facts.duration_s = 2.0
    estimates = estimate_look(source, facts)
    assert estimates.denoise.enabled is grainy


_W, _H = SHAKE_WIDTH, SHAKE_HEIGHT
_PAD = 24


def _scene(seed: int) -> bytes:
    """A textured picture larger than a frame, to cut shifted frames from."""
    rng = random.Random(seed)
    width, height = _W + 2 * _PAD, _H + 2 * _PAD
    stride = width // 4 + 2
    coarse = [rng.randrange(256) for _ in range(stride * (height // 4 + 2))]
    # Blocky but not flat, like film: 4x4 cells plus a little grain.
    return bytes(
        max(0, min(255, coarse[(y // 4) * stride + x // 4] + rng.randrange(-3, 4)))
        for y in range(height)
        for x in range(width)
    )


_SCENES: dict[int, bytes] = {}


def _frame_at(dx: int, dy: int, seed: int = 1) -> bytes:
    scene = _SCENES.setdefault(seed, _scene(seed))
    width = _W + 2 * _PAD
    rows = []
    for y in range(_H):
        start = (y + _PAD + dy) * width + _PAD + dx
        rows.append(scene[start : start + _W])
    return b"".join(rows)


def _shaky_run(offsets: list[tuple[int, int]], seed: int = 1) -> list[bytes]:
    """Frames of one picture, each with the camera at ``offsets[i]``."""
    return [_frame_at(dx, dy, seed) for dx, dy in offsets]


# A handheld camera: the aim wobbles back and forth from frame to frame.
_HANDHELD = [(0, 0), (5, -3), (-2, 4), (6, 1), (-4, -4), (3, 5), (-5, 0), (4, -2)]
# A pan: 3 px per frame, every frame.
_PAN = [(3 * k - 12, 0) for k in range(SHAKE_RUN)]


def test_whole_frame_shift_is_found_to_the_pixel() -> None:
    # The camera moves 5 px right and 3 px up, so the picture moves the
    # other way.
    a, b = _frame_at(0, 0), _frame_at(5, -3)
    dx, dy, left = _global_shift(a, b, _W, _H)
    assert (round(dx), round(dy)) == (-5, 3)
    assert abs(dx + 5) < 0.3 and abs(dy - 3) < 0.3
    assert left < 3.0
    # A flat picture gives nothing to align: it has not moved.
    flat = bytes([90]) * (_W * _H)
    assert _global_shift(flat, flat, _W, _H)[:2] == (0.0, 0.0)


def test_a_handheld_camera_asks_for_stabilising_sized_to_the_shake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _shaky_run(_HANDHELD)
    estimates = _run_estimates(
        tmp_path, monkeypatch, lambda _n: _spread(), runs=lambda _n: run
    )
    assert estimates.stabilize.enabled
    assert estimates.metrics["shake_jitter_pct"] > 2.0
    params = estimates.stabilize.params
    assert 6.0 <= params["shakiness"] <= 10.0
    # The crop follows how far the picture strays, not a fixed 30%.
    assert 3.0 <= params["max_crop_pct"] <= 15.0
    assert 10.0 <= params["smoothing"] <= 30.0


def test_light_shake_asks_for_a_lighter_touch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    light = _shaky_run([(dx // 3, dy // 3) for dx, dy in _HANDHELD])
    strong = _shaky_run(_HANDHELD)
    a = _run_estimates(
        tmp_path, monkeypatch, lambda _n: _spread(), runs=lambda _n: light
    )
    b = _run_estimates(
        tmp_path, monkeypatch, lambda _n: _spread(), runs=lambda _n: strong
    )
    assert a.stabilize.enabled
    assert a.stabilize.params["shakiness"] < b.stabilize.params["shakiness"]
    assert a.stabilize.params["max_crop_pct"] < b.stabilize.params["max_crop_pct"]


def test_a_steady_pan_is_not_shake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _shaky_run(_PAN)
    estimates = _run_estimates(
        tmp_path, monkeypatch, lambda _n: _spread(), runs=lambda _n: run
    )
    assert not estimates.stabilize.enabled
    assert estimates.metrics["shake_jitter_pct"] < 0.2
    assert estimates.stabilize.params == {
        "shakiness": 5.0,
        "smoothing": 10.0,
        "max_crop_pct": 10.0,
    }


def test_a_cut_is_not_shake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A still shot, then a cut to another: the old measure called every cut
    # and every moving subject maximum shake.
    run = _shaky_run([(0, 0)] * 4) + _shaky_run([(0, 0)] * 4, seed=2)
    estimates = _run_estimates(
        tmp_path, monkeypatch, lambda _n: _spread(), runs=lambda _n: run
    )
    assert not estimates.stabilize.enabled
    assert estimates.metrics["shake_jitter_pct"] == 0.0


def test_frames_with_nothing_in_common_are_not_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = [_frame_at(0, 0, seed) for seed in range(10, 10 + SHAKE_RUN)]
    estimates = _run_estimates(
        tmp_path, monkeypatch, lambda _n: _spread(), runs=lambda _n: run
    )
    assert not estimates.stabilize.enabled
    assert estimates.stabilize.confidence == 0.0
    assert "no steady stretch" in estimates.stabilize.reason


def test_one_shaky_sample_of_three_does_not_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = [
        _shaky_run(_HANDHELD),
        _shaky_run(_PAN),
        _shaky_run([(0, 0)] * SHAKE_RUN),
    ]
    estimates = _run_estimates(
        tmp_path, monkeypatch, lambda _n: _spread(), runs=lambda n: runs[n]
    )
    assert not estimates.stabilize.enabled


def _walk(fps: float, frames: int = SHAKE_RUN) -> list[tuple[int, int]]:
    """A camera bobbing 2.5 times a second, sampled at ``fps``."""
    return [
        (round(9 * math.sin(2 * math.pi * 2.5 * k / fps)), 0) for k in range(frames)
    ]


def test_the_same_shake_reads_the_same_at_any_frame_rate() -> None:
    # Between frames 1/24 s apart a wobble changes more than between frames
    # 1/60 s apart; the jitter is scaled back to 30 fps.
    readings = {}
    for fps in (24.0, 30.0, 48.0):
        run = _shaky_run(_walk(fps))
        measured = _run_shake(run, _W, _H, fps)
        assert measured is not None
        readings[fps] = measured[0]
    assert max(readings.values()) < 1.6 * min(readings.values())


@pytest.mark.parametrize(
    ("fps", "stride"),
    [(None, 1), (23.976, 1), (30.0, 1), (50.0, 2), (59.94, 2), (120.0, 4)],
)
def test_fast_footage_is_read_every_few_frames(fps: float | None, stride: int) -> None:
    assert shake_stride(fps) == stride


def test_repeated_frames_do_not_make_a_pan_look_shaky() -> None:
    # A 24p pan carried at 30p repeats every fourth frame: each repeat is a
    # move of nothing between two moves of 6 px.
    pan = [(6 * k - 21, 0) for k in range(7)]
    carried = [pan[0], pan[1], pan[2], pan[3], pan[3], pan[4], pan[5], pan[6]]
    measured = _run_shake(_shaky_run(carried), _W, _H, 30.0)
    assert measured is not None
    assert measured[0] < 0.2


def test_shake_beyond_the_old_search_is_measured() -> None:
    # Moves of up to 14 px a frame (4% of the width) used to be dropped as
    # cuts, so the worst shake got the smallest crop, then none.
    wild = [(0, 0), (12, -6), (-2, 6), (13, 2), (-8, -7), (6, 9), (-10, 0), (8, -4)]
    measured = _run_shake(_shaky_run(wild, seed=3), _W, _H)
    assert measured is not None
    assert measured[0] > 5.0


def test_with_two_samples_the_calmer_decides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One window is a cut and cannot be measured: of the two left, the still
    # one decides, not the handheld insert.
    nothing = [_frame_at(0, 0, seed) for seed in range(20, 20 + SHAKE_RUN)]
    runs = [_shaky_run([(0, 0)] * SHAKE_RUN), nothing, _shaky_run(_HANDHELD)]
    estimates = _run_estimates(
        tmp_path, monkeypatch, lambda _n: _spread(), runs=lambda n: runs[n]
    )
    assert not estimates.stabilize.enabled


def test_a_run_of_frames_is_one_ffmpeg_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    size = 320 * 180

    def run(cmd: list[str], **_kwargs: Any) -> _Completed:
        seen.append(cmd)
        return _Completed(0, bytes(size * 8))

    monkeypatch.setattr("super_processor.estimators.subprocess.run", run)
    frames = extract_gray_run(Path("clip.mp4"), at_s=2.0, ffmpeg_bin="ffmpeg")
    assert len(frames) == 8 and all(len(f) == size for f in frames)
    assert seen[0][seen[0].index("-frames:v") + 1] == "8"
    assert seen[0][seen[0].index("-vf") + 1] == "scale=320:180,format=gray"
    # Each decoded frame once: FFmpeg must not repeat one to fill time.
    assert seen[0][seen[0].index("-fps_mode") + 1] == "passthrough"
    extract_gray_run(Path("clip.mp4"), at_s=2.0, ffmpeg_bin="ffmpeg", stride=2)
    assert seen[1][seen[1].index("-vf") + 1] == (
        "select=not(mod(n\\,2)),scale=320:180,format=gray"
    )
    # Near the end of a clip fewer frames come back; a part frame is dropped.
    monkeypatch.setattr(
        "super_processor.estimators.subprocess.run",
        lambda *_a, **_k: _Completed(0, bytes(size * 3 + 100)),
    )
    assert len(extract_gray_run(Path("clip.mp4"), at_s=2.0, ffmpeg_bin="ffmpeg")) == 3
    # One frame cannot show movement.
    monkeypatch.setattr(
        "super_processor.estimators.subprocess.run",
        lambda *_a, **_k: _Completed(0, bytes(size)),
    )
    with pytest.raises(EstimatorError, match="short frame buffer"):
        extract_gray_run(Path("clip.mp4"), at_s=2.0, ffmpeg_bin="ffmpeg")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.parametrize(
    ("crop", "shaky"),
    [
        ("crop=600:450:x='20+9*sin(n*2.1)':y='15+7*cos(n*1.9)'", True),
        ("crop=600:450:x='mod(n*2,40)':y=15", False),
        ("crop=600:450:x=20:y=15", False),
    ],
    ids=["handheld", "pan", "still"],
)
def test_a_real_sd_clip_s_shake_decides_stabilising(
    tmp_path: Path, crop: str, shaky: bool
) -> None:
    source = tmp_path / "sd.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=640x480:rate=30:duration=3",
         "-vf", f"{crop.replace(',', chr(92) + ',')},scale=640:480",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(source)],
        check=True,
    )  # fmt: skip
    facts = _sized(640, 480, source)
    facts.duration_s = 3.0
    assert estimate_look(source, facts).stabilize.enabled is shaky


def test_estimation_needs_a_readable_video(tmp_path: Path) -> None:
    with pytest.raises(EstimatorError, match="not a readable file"):
        estimate_look(tmp_path / "missing.mp4", _facts())
    audio = tmp_path / "audio.m4a"
    audio.write_bytes(b"audio")
    with pytest.raises(EstimatorError, match="requires a video stream"):
        estimate_look(audio, _facts(video=False))


class _Completed:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_rgb_means_average_each_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "super_processor.estimators.subprocess.run",
        lambda *_args, **_kwargs: _Completed(0, bytes([10, 20, 30, 30, 40, 50])),
    )
    assert extract_rgb_means(Path("clip.mp4"), at_s=1.0, ffmpeg_bin="ffmpeg") == (
        20.0,
        30.0,
        40.0,
    )


def test_rgb_means_retry_a_failed_device_in_software(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> _Completed:
        commands.append(cmd)
        if "-hwaccel" in cmd:
            return _Completed(1, b"", b"no device")
        return _Completed(0, bytes([90, 90, 90]))

    monkeypatch.setattr("super_processor.estimators.subprocess.run", run)
    means = extract_rgb_means(
        Path("clip.mp4"), at_s=1.0, ffmpeg_bin="ffmpeg", decoder="hevc_cuvid"
    )
    assert means == (90.0, 90.0, 90.0)
    assert len(commands) == 2


def test_failed_samples_raise_estimator_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "super_processor.estimators.subprocess.run",
        lambda *_args, **_kwargs: _Completed(1, b"", b"moov atom not found"),
    )
    with pytest.raises(EstimatorError, match="rgb extract failed at 2s"):
        extract_rgb_means(Path("clip.mp4"), at_s=2.0, ffmpeg_bin="ffmpeg")
    # The FFmpeg message reaches the error, so the user sees why.
    with pytest.raises(EstimatorError, match="moov atom not found"):
        extract_gray_frame(Path("clip.mp4"), at_s=2.0, ffmpeg_bin="ffmpeg")
    monkeypatch.setattr(
        "super_processor.estimators.subprocess.run",
        lambda *_args, **_kwargs: _Completed(0, bytes(PIXELS - 1)),
    )
    with pytest.raises(EstimatorError, match="short frame buffer"):
        extract_gray_frame(Path("clip.mp4"), at_s=2.0, ffmpeg_bin="ffmpeg")
    monkeypatch.setattr("super_processor.estimators.which", lambda _name: None)
    with pytest.raises(EstimatorError, match="ffmpeg not found"):
        extract_gray_frame(Path("clip.mp4"), at_s=2.0)


@pytest.mark.parametrize(
    ("crop", "expected"), [(False, "scale=160:90"), (True, "crop=160:90")]
)
def test_a_grain_sample_is_cut_out_not_scaled(
    monkeypatch: pytest.MonkeyPatch, crop: bool, expected: str
) -> None:
    seen: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: Any) -> _Completed:
        seen.append(cmd)
        return _Completed(0, bytes(PIXELS))

    monkeypatch.setattr("super_processor.estimators.subprocess.run", run)
    extract_gray_frame(Path("clip.mp4"), at_s=2.0, ffmpeg_bin="ffmpeg", crop=crop)
    assert seen[0][seen[0].index("-vf") + 1] == f"{expected},format=gray"


def test_an_unknown_decoder_is_refused_before_ffmpeg_runs() -> None:
    with pytest.raises(EstimatorError, match="unsupported decoder"):
        ensure_decoder("magic_cuvid")


def test_missing_or_damaged_estimates_are_refused(tmp_path: Path) -> None:
    with pytest.raises(EstimatorError, match="not found"):
        load_estimates(tmp_path)
    (tmp_path / "estimates.json").write_text("[]", encoding="utf-8")
    with pytest.raises(EstimatorError, match="must be an object"):
        load_estimates(tmp_path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_estimate_look_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1.2:size=320x240:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    facts = MediaFacts(
        schema_version=1,
        source_path=str(source),
        format_name="mp4",
        format_long_name="MP4",
        duration_s=1.2,
        size_bytes=source.stat().st_size,
        bit_rate=1000,
        streams=[
            StreamFacts(
                index=0,
                codec_type="video",
                codec_name="h264",
                width=320,
                height=240,
            )
        ],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )
    estimates = estimate_look(source, facts)
    assert isinstance(estimates, LookEstimates)
    assert set(estimates.contrast.params) == {"contrast", "brightness", "gamma"}
    assert {"luma_mean", "wb_cast", "noise_residual", "shake_jitter_pct"} <= set(
        estimates.metrics
    )
    # testsrc is full of colour bars, nowhere near black or white on average.
    assert 30.0 < estimates.metrics["luma_mean"] < 220.0
    write_estimates(tmp_path, estimates)
    loaded = load_estimates(tmp_path)
    assert loaded.to_dict() == estimates.to_dict()
