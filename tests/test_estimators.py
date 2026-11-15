"""Tests for Cython estimator primitives and look sampling."""

from __future__ import annotations

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
    EstimatorError,
    LookEstimates,
    _noise_residual,
    choose_sample_windows,
    ensure_decoder,
    estimate_look,
    extract_gray_frame,
    extract_rgb_means,
    load_estimates,
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
) -> LookEstimates:
    """Estimate from fake samples; ``gray`` gets the call number."""
    calls = iter(range(1000))
    monkeypatch.setattr(
        "super_processor.estimators.extract_gray_frame",
        lambda *_args, **_kwargs: gray(next(calls)),
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
    # Identical frames in each pair: no motion, so no stabilising.
    assert not estimates.stabilize.enabled
    assert estimates.metrics["shake_sad_norm"] == 0.0
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


def test_frames_that_jump_within_a_window_ask_for_stabilising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each window samples a frame and one a moment later: make them differ.
    frames = (bytes([60]) * PIXELS, bytes([120]) * PIXELS)
    estimates = _run_estimates(tmp_path, monkeypatch, lambda n: frames[n % 2])
    assert estimates.stabilize.enabled
    assert estimates.metrics["shake_sad_norm"] == pytest.approx(60.0)
    assert estimates.stabilize.params == {
        "shakiness": 10.0,
        "smoothing": 50.0,
        "max_crop_pct": 30.0,
    }


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
    assert {"luma_mean", "wb_cast", "noise_residual", "shake_sad_norm"} <= set(
        estimates.metrics
    )
    # testsrc is full of colour bars, nowhere near black or white on average.
    assert 30.0 < estimates.metrics["luma_mean"] < 220.0
    write_estimates(tmp_path, estimates)
    loaded = load_estimates(tmp_path)
    assert loaded.to_dict() == estimates.to_dict()
