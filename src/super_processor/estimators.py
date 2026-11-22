"""Classical look/motion estimators from sampled media windows."""

from __future__ import annotations

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import mean_luma, percentile_u8, sad_u8
from .doctor import check_ffmpeg_capability, which
from .encoders import HARDWARE_DECODERS, decoder_input_args
from .probe import MediaFacts, parse_frame_rate
from .tools import FILE_INPUT, resolve_argv

ESTIMATES_FILE_NAME = "estimates.json"
ESTIMATES_SCHEMA_VERSION = 1


class EstimatorError(RuntimeError):
    """Raised when look/motion estimation cannot complete."""


def ensure_decoder(decoder: str) -> None:
    """Fail before sampling when this ffmpeg has no such decoder."""
    if decoder not in HARDWARE_DECODERS:
        raise EstimatorError(f"unsupported decoder {decoder}")
    result = check_ffmpeg_capability(
        f"decoder:{decoder}",
        kind="decoders",
        token=decoder,
        required=True,
        hint="install ffmpeg with this decoder or sample in software",
    )
    if not result.ok:
        raise EstimatorError(result.detail)


@dataclass(slots=True)
class SampleWindow:
    """One stratified sample location in the source timeline."""

    label: str
    start_s: float
    duration_s: float = 0.4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OpSuggestion:
    """Suggested recipe op enablement and parameters from CV priors."""

    enabled: bool
    confidence: float
    params: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "confidence": self.confidence,
            "params": dict(self.params),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpSuggestion:
        return cls(
            enabled=bool(data.get("enabled", False)),
            confidence=float(data.get("confidence", 0.0)),
            params={
                str(k): float(v) for k, v in dict(data.get("params") or {}).items()
            },
            reason=str(data.get("reason", "")),
        )


@dataclass(slots=True)
class LookEstimates:
    """CV priors for the four classical look/motion operations."""

    schema_version: int
    source_path: str
    windows: list[SampleWindow]
    contrast: OpSuggestion
    white_balance: OpSuggestion
    denoise: OpSuggestion
    stabilize: OpSuggestion
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "windows": [window.to_dict() for window in self.windows],
            "contrast": self.contrast.to_dict(),
            "white_balance": self.white_balance.to_dict(),
            "denoise": self.denoise.to_dict(),
            "stabilize": self.stabilize.to_dict(),
            "metrics": dict(self.metrics),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LookEstimates:
        return cls(
            schema_version=int(data.get("schema_version", ESTIMATES_SCHEMA_VERSION)),
            source_path=str(data["source_path"]),
            windows=[
                SampleWindow(
                    label=str(item["label"]),
                    start_s=float(item["start_s"]),
                    duration_s=float(item.get("duration_s", 0.4)),
                )
                for item in list(data.get("windows") or [])
            ],
            contrast=OpSuggestion.from_dict(dict(data.get("contrast") or {})),
            white_balance=OpSuggestion.from_dict(dict(data.get("white_balance") or {})),
            denoise=OpSuggestion.from_dict(dict(data.get("denoise") or {})),
            stabilize=OpSuggestion.from_dict(dict(data.get("stabilize") or {})),
            metrics={
                str(key): float(value)
                for key, value in dict(data.get("metrics") or {}).items()
            },
        )


def choose_sample_windows(facts: MediaFacts, *, count: int = 3) -> list[SampleWindow]:
    """Pick stratified head/mid/tail windows from media duration."""
    duration = facts.duration_s or 1.0
    duration = max(duration, 0.5)
    span = max(0.2, min(0.4, duration / 8.0))
    anchors = [0.1, 0.5, 0.85][: max(1, count)]
    windows: list[SampleWindow] = []
    labels = ("head", "mid", "tail")
    for index, fraction in enumerate(anchors):
        start = max(0.0, min(max(0.0, duration - span - 0.05), duration * fraction))
        windows.append(
            SampleWindow(
                label=labels[index] if index < len(labels) else f"w{index}",
                start_s=start,
                duration_s=span,
            )
        )
    return windows


def _ffmpeg_bin() -> str:
    path = which("ffmpeg")
    if path is None:
        raise EstimatorError("ffmpeg not found on PATH")
    return path


def _decoder_attempts(decoder: str | None) -> list[str | None]:
    """Try the selected device, then the software decoder."""
    if decoder is None:
        return [None]
    return [decoder, None]


def extract_gray_frame(
    source: Path,
    *,
    at_s: float,
    width: int = 160,
    height: int = 90,
    ffmpeg_bin: str | None = None,
    decoder: str | None = None,
    crop: bool = False,
) -> bytes:
    """Decode one grayscale frame near ``at_s`` as packed 8-bit luma.

    The frame is scaled to ``width`` x ``height``, or with ``crop`` the
    centre ``width`` x ``height`` is cut out at the source's own pixels.
    """
    return _gray_frames(
        source,
        at_s=at_s,
        width=width,
        height=height,
        count=1,
        least=1,
        ffmpeg_bin=ffmpeg_bin,
        decoder=decoder,
        crop=crop,
    )


def extract_gray_run(
    source: Path,
    *,
    at_s: float,
    count: int = 8,
    width: int = 320,
    height: int = 180,
    ffmpeg_bin: str | None = None,
    decoder: str | None = None,
    stride: int = 1,
) -> list[bytes]:
    """Decode up to ``count`` consecutive grayscale frames from ``at_s``,
    every ``stride``-th one, scaled to ``width`` x ``height``. Near the end
    of a clip fewer may come back, but never fewer than two."""
    frame = width * height
    data = _gray_frames(
        source,
        at_s=at_s,
        width=width,
        height=height,
        count=count,
        least=2,
        ffmpeg_bin=ffmpeg_bin,
        decoder=decoder,
        crop=False,
        stride=stride,
    )
    return [data[i : i + frame] for i in range(0, len(data), frame)]


def _gray_frames(
    source: Path,
    *,
    at_s: float,
    width: int,
    height: int,
    count: int,
    least: int,
    ffmpeg_bin: str | None,
    decoder: str | None,
    crop: bool,
    stride: int = 1,
) -> bytes:
    """Up to ``count`` whole frames, packed; at least ``least`` of them."""
    binary = ffmpeg_bin or _ffmpeg_bin()
    frame = width * height
    expected = frame * least
    last_detail = ""
    for attempt in _decoder_attempts(decoder):
        cmd = [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            *decoder_input_args(attempt),
            "-ss",
            f"{at_s:g}",
            *FILE_INPUT,
            "-i",
            str(source),
            "-frames:v",
            str(count),
            "-vf",
            (f"select=not(mod(n\\,{stride}))," if stride > 1 else "")
            + f"{'crop' if crop else 'scale'}={width}:{height},format=gray",
            # Each decoded frame once: no repeats to fill gaps in time.
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                resolve_argv(cmd),
                check=False,
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            if attempt is not None:
                continue
            raise EstimatorError(f"failed to sample frame at {at_s:g}s: {exc}") from exc
        if completed.returncode != 0 or not completed.stdout:
            last_detail = (
                (completed.stderr or b"").decode("utf-8", errors="replace").strip()
            )
            if attempt is not None:
                continue
            raise EstimatorError(
                f"ffmpeg frame extract failed at {at_s:g}s"
                + (f": {last_detail}" if last_detail else "")
            )
        if len(completed.stdout) < expected:
            if attempt is not None:
                continue
            raise EstimatorError(
                f"short frame buffer at {at_s:g}s "
                f"({len(completed.stdout)} < {expected} bytes)"
            )
        whole = min(count, len(completed.stdout) // frame)
        return completed.stdout[: whole * frame]
    raise EstimatorError(
        f"ffmpeg frame extract failed at {at_s:g}s"
        + (f": {last_detail}" if last_detail else "")
    )


def extract_rgb_means(
    source: Path,
    *,
    at_s: float,
    width: int = 80,
    height: int = 45,
    ffmpeg_bin: str | None = None,
    decoder: str | None = None,
) -> tuple[float, float, float]:
    """Return mean R/G/B for a downscaled frame (white-balance prior)."""
    binary = ffmpeg_bin or _ffmpeg_bin()
    raw = b""
    for attempt in _decoder_attempts(decoder):
        cmd = [
            binary,
            "-hide_banner",
            "-loglevel",
            "error",
            *decoder_input_args(attempt),
            "-ss",
            f"{at_s:g}",
            *FILE_INPUT,
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:{height},format=rgb24",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                resolve_argv(cmd), check=False, capture_output=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            if attempt is not None:
                continue
            raise EstimatorError(f"ffmpeg rgb extract failed at {at_s:g}s") from None
        if completed.returncode != 0 or not completed.stdout:
            if attempt is not None:
                continue
            raise EstimatorError(f"ffmpeg rgb extract failed at {at_s:g}s")
        raw = completed.stdout
        break
    if not raw:
        raise EstimatorError(f"ffmpeg rgb extract failed at {at_s:g}s")
    pixels = len(raw) // 3
    if pixels == 0:
        raise EstimatorError("empty rgb frame")
    r_total = g_total = b_total = 0
    for index in range(0, pixels * 3, 3):
        r_total += raw[index]
        g_total += raw[index + 1]
        b_total += raw[index + 2]
    scale = float(pixels)
    return r_total / scale, g_total / scale, b_total / scale


# Contrast is judged on frames spread over the whole video, so that one
# dark or bright scene does not decide it for the rest.
CONTRAST_SAMPLES = 9
# A frame's black and white points: its 1st and 99th percentile, so a few
# stray pixels (a logo, a scratch, a speck of dust) do not count.
_BLACK_PCT = 1.0
_WHITE_PCT = 99.0
# A frame spanning fewer levels than this is a fade or a blank card.
_BLANK_SPAN = 24.0
# Rows and columns at the edge no brighter than this in every sampled frame
# are letterbox or pillarbox bars, not picture. The scaled picture's own
# edge bleeds into the bar, so 2 more pixels are dropped there.
_BAR_LEVEL = 12
_BAR_BLEED = 2
# Where a faded picture's black and white are put back (8-bit grey, where
# video black is 0 and video white 255). No sampled frame's white is
# pushed past _CLIP_WHITE.
_TARGET_BLACK = 4.0
_TARGET_WHITE = 240.0
_CLIP_WHITE = 250.0
# Black at or above this level in nearly every frame is a faded print or a
# lifted transfer. Clean and well-kept video measures 0-18 here.
_LIFTED_BLACK = 20.0
# Fading pulls the whole picture toward a grey: faded = a*picture +
# (1-a)*grey, which lifts black by (1-a)*grey and squeezes the range by a.
# Whites are stretched only as far as the black's lift explains, so a
# barely lifted black does not bring a large boost to dim whites, which may
# be the scene's own.
_FADE_GREY = 128.0
_MAX_STRETCH = 2.0
# FFmpeg's eq works on the stored levels. On limited-range video (black 16,
# white 235) a grey frame stretches those over 0-255; on full-range video
# (yuvj, MJPEG, some phones) grey is the stored level itself. Its contrast
# turns about level 127.5 + 0.5/(c-1) and its brightness adds 255.5 levels
# per 1.0.
_GRAY_PER_LEVEL = 255.0 / 219.0


def contrast_sample_times(
    duration_s: float, count: int = CONTRAST_SAMPLES
) -> list[float]:
    """Evenly spread times to judge contrast at, clear of both ends: a seek
    into the last frame's own time finds no frame to decode."""
    duration = max(0.5, duration_s)
    last = max(0.0, duration - 0.25)
    return [min(last, duration * (index + 0.5) / count) for index in range(count)]


def _eq_brightness(
    contrast: float, black: float, target: float, *, full_range: bool = False
) -> float:
    """The eq brightness that puts grey level ``black`` at ``target`` after
    eq scales it by ``contrast``."""
    if full_range:
        return (target - contrast * (black - 128.0) - 127.5) / 255.5
    level = (target - contrast * black) / _GRAY_PER_LEVEL
    return (level - 111.5 + 112.0 * contrast) / 255.5


def is_full_range(facts: MediaFacts) -> bool:
    """Whether the video stores full-range levels (0-255, not 16-235)."""
    video = facts.primary_video()
    if video is None:
        return False
    return video.color_range == "pc" or (video.pix_fmt or "").startswith("yuvj")


def media_duration(facts: MediaFacts) -> float | None:
    """The video's length: the container's, else the video stream's, else
    its frame count over its frame rate. None when nothing says."""
    if facts.duration_s:
        return facts.duration_s
    video = facts.primary_video()
    if video is None:
        return None
    if video.duration_s:
        return video.duration_s
    rate = parse_frame_rate(video.avg_frame_rate) or parse_frame_rate(
        video.r_frame_rate
    )
    if video.nb_frames and rate:
        return video.nb_frames / rate
    return None


def _centred(near: int, far: int) -> bool:
    """Whether bars of ``near`` and ``far`` pixels are a matched pair."""
    return near > 0 and far > 0 and abs(near - far) <= _BAR_BLEED


def _picture_area(frames: list[bytes], width: int) -> tuple[int, int, int, int] | None:
    """The rows and columns (top, bottom, left, right, inclusive) inside any
    letterbox or pillarbox bars, or None when there are none."""
    height = len(frames[0]) // width if frames and width else 0
    if height < 4 or width < 4:
        return None
    row_max = [0] * height
    col_max = [0] * width
    for frame in frames:
        for y in range(height):
            row_max[y] = max(row_max[y], max(frame[y * width : (y + 1) * width]))
        for x in range(width):
            col_max[x] = max(col_max[x], max(frame[x : height * width : width]))
    rows = [y for y, level in enumerate(row_max) if level > _BAR_LEVEL]
    cols = [x for x, level in enumerate(col_max) if level > _BAR_LEVEL]
    if not rows or not cols:
        return None
    top, bottom, left, right = rows[0], rows[-1], cols[0], cols[-1]
    # Bars are centred: a dark band along one edge only is part of the
    # picture (a night sky, a shadowed floor), and so is its true black.
    if not _centred(top, height - 1 - bottom):
        top, bottom = 0, height - 1
    if not _centred(left, width - 1 - right):
        left, right = 0, width - 1
    if (top, bottom, left, right) == (0, height - 1, 0, width - 1):
        return None
    top += _BAR_BLEED if top else 0
    bottom -= _BAR_BLEED if bottom < height - 1 else 0
    left += _BAR_BLEED if left else 0
    right -= _BAR_BLEED if right < width - 1 else 0
    # Bars that leave less than a quarter of either side are not bars.
    if bottom - top + 1 < height // 4 or right - left + 1 < width // 4:
        return None
    return top, bottom, left, right


def _no_contrast(
    reason: str, metrics: dict[str, float]
) -> tuple[OpSuggestion, dict[str, float]]:
    return (
        OpSuggestion(
            enabled=False,
            confidence=0.0,
            params={"contrast": 1.0, "brightness": 0.0, "gamma": 1.0},
            reason=reason,
        ),
        metrics,
    )


def _estimate_contrast(
    frames: list[bytes], width: int = 160, *, full_range: bool = False
) -> tuple[OpSuggestion, dict[str, float]]:
    """Restore the black of a faded picture, and leave every other alone.

    How dark or bright a picture is belongs to its scene: a night shot, an
    interview lit against black, a snowfield. A fault in the print or the
    transfer instead shows in every frame, the dark ones too: a faded print
    has no true black anywhere. So black is the second darkest frame's black
    point, and white the second brightest frame's white point, which a
    single title card or flash cannot move. Letterbox and pillarbox bars
    are left out first: they are black whatever the print.

    The black goes back to video black. The white is stretched only as far
    as the black's lift explains, never ends up dimmer than it was, and no
    sampled frame is pushed into clipping: from the picture alone, a dim
    transfer cannot be told from a film shot dark on purpose.
    """
    area = _picture_area(frames, width)
    if area is not None:
        top, bottom, left, right = area
        frames = [
            b"".join(
                frame[y * width + left : y * width + right + 1]
                for y in range(top, bottom + 1)
            )
            for frame in frames
        ]
    blacks: list[float] = []
    whites: list[float] = []
    means: list[float] = []
    for frame in frames:
        low = percentile_u8(frame, _BLACK_PCT)
        high = percentile_u8(frame, _WHITE_PCT)
        means.append(mean_luma(frame))
        if high - low >= _BLANK_SPAN:
            blacks.append(low)
            whites.append(high)
    mean = sum(means) / len(means) if means else 0.0
    metrics = {"luma_mean": mean}
    if len(blacks) < 2:
        return _no_contrast("too few frames with a picture to judge contrast", metrics)

    black = sorted(blacks)[1]
    white = sorted(whites)[-2]
    metrics.update({"luma_black": black, "luma_white": white})
    lifted = black >= _LIFTED_BLACK
    contrast = 1.0
    brightness = 0.0
    if lifted:
        span = max(1.0, white - black)
        full = (_TARGET_WHITE - _TARGET_BLACK) / span
        lift = min(_FADE_GREY - 1.0, black - _TARGET_BLACK)
        explained = _FADE_GREY / (_FADE_GREY - lift)
        contrast = min(full, explained)
        # Pulling black down never leaves the whites dimmer than they were.
        contrast = max(contrast, (white - _TARGET_BLACK) / span)
        brightest = max(1.0, max(whites) - black)
        contrast = min(contrast, (_CLIP_WHITE - _TARGET_BLACK) / brightest)
        contrast = max(1.0, min(_MAX_STRETCH, contrast))
        brightness = max(
            -0.5,
            min(
                0.5,
                _eq_brightness(contrast, black, _TARGET_BLACK, full_range=full_range),
            ),
        )
    confidence = (
        min(1.0, (black - _LIFTED_BLACK + 10.0) / 30.0 + (contrast - 1.0))
        if lifted
        else 0.0
    )
    state = "lifted" if lifted else "true black"
    bars = " inside bars" if area is not None else ""
    suggestion = OpSuggestion(
        enabled=lifted,
        confidence=confidence,
        params={
            "contrast": round(contrast, 3),
            "brightness": round(brightness, 3),
            "gamma": 1.0,
        },
        reason=(
            f"black={black:.0f} ({state}) white={white:.0f} "
            f"over {len(blacks)} frames{bars}, mean={mean:.0f}"
        ),
    )
    return suggestion, metrics


def _estimate_white_balance(
    rgb_means: list[tuple[float, float, float]],
) -> tuple[OpSuggestion, dict[str, float]]:
    r = sum(item[0] for item in rgb_means) / len(rgb_means)
    g = sum(item[1] for item in rgb_means) / len(rgb_means)
    b = sum(item[2] for item in rgb_means) / len(rgb_means)
    # Positive cast => too blue => lower temperature; negative => too warm.
    cast = (b - r) / max(1.0, (r + g + b) / 3.0)
    temperature = max(2000.0, min(10000.0, 6500.0 - cast * 1800.0))
    tint = max(-100.0, min(100.0, ((g - (r + b) / 2.0) / 255.0) * 100.0))
    enabled = abs(temperature - 6500.0) > 250.0 or abs(tint) > 8.0
    confidence = min(
        1.0,
        abs(temperature - 6500.0) / 1500.0 + abs(tint) / 40.0,
    )
    suggestion = OpSuggestion(
        enabled=enabled,
        confidence=confidence,
        params={
            "temperature": round(temperature if enabled else 6500.0, 1),
            "tint": round(tint if enabled else 0.0, 2),
        },
        reason=f"mean_rgb=({r:.1f},{g:.1f},{b:.1f}) cast={cast:.3f}",
    )
    return suggestion, {"wb_r": r, "wb_g": g, "wb_b": b, "wb_cast": cast}


def _noise_residual(frame: bytes, width: int = 160) -> float:
    """Estimate the grain in one packed 8-bit luma frame, ``width`` pixels
    per row, on the same scale as a pixel variance: about 0 for a clean
    picture however much contrast it has, and about the noise variance for
    a grainy one.

    The whole-frame variance measures contrast, so a clean gradient looks
    as noisy as grain. Grain differs from neighbour to neighbour; a picture
    mostly does not.

    Uses the median of the absolute differences to the right and lower
    neighbours, so the picture's edges, a minority of pixels, do not count.
    For grain of variance s², a neighbour difference has variance 2s², and
    its absolute value has median 0.6745·s·√2.
    """
    height = len(frame) // width
    if width < 2 or height < 2:
        return 0.0
    diffs: list[int] = []
    for row in range(height - 1):
        start = row * width
        line = frame[start : start + width]
        below = frame[start + width : start + 2 * width]
        diffs.extend(abs(a - b) for a, b in zip(line, line[1:], strict=False))
        diffs.extend(abs(a - b) for a, b in zip(line, below, strict=True))
    diffs.sort()
    median = diffs[len(diffs) // 2]
    return (median / (0.6745 * math.sqrt(2.0))) ** 2


# Side of the square cut from the middle of the picture to measure grain.
# Scaling down first would average the grain away.
GRAIN_CROP = 256


def grain_crop_side(facts: MediaFacts) -> int | None:
    """The grain crop's side for this source, or None when its size is not
    known. Square, so a rotated picture fits it too."""
    video = facts.primary_video()
    if video is None or not video.width or not video.height:
        return None
    return min(GRAIN_CROP, video.width, video.height)


def _estimate_denoise(
    frames: list[bytes], width: int = 160
) -> tuple[OpSuggestion, dict[str, float]]:
    residuals = [_noise_residual(frame, width) for frame in frames]
    noise = sum(residuals) / len(residuals)
    # Grain sigma at the source's own pixels. Clean and compressed footage,
    # old SD transfers included, measures sigma 0-2; visible grain or phone
    # sensor noise measures 5 and up.
    strength = max(0.0, min(1.0, (math.sqrt(noise) - 2.5) / 6.0))
    enabled = strength >= 0.18
    suggestion = OpSuggestion(
        enabled=enabled,
        confidence=min(1.0, strength * 1.4),
        params={"strength": round(strength if enabled else 0.0, 3)},
        reason=f"noise_residual={noise:.1f}",
    )
    return suggestion, {"noise_residual": noise, "denoise_strength": strength}


# Shake is measured on runs of consecutive frames, scaled to this size.
SHAKE_RUN = 8
SHAKE_WIDTH = 320
SHAKE_HEIGHT = 180
# Shake is compared at this frame rate: faster footage reads every second
# (or third) frame, and the jitter is scaled to what it would be at 30 fps.
SHAKE_FPS = 30.0
# Largest whole-frame shift searched between neighbouring frames, in pixels
# of the scaled frame (5% of the width).
_SHIFT_RADIUS = 16
# Mean absolute difference, after the best alignment, above which two
# neighbouring frames are not one picture moved: a cut or a flash.
_CUT_COST = 12.0
# Below this difference, with no move, the frame is a repeat (a 24p film
# carried at 30p, or a frame FFmpeg duplicated) and is skipped.
_REPEAT_COST = 0.5
# Jitter, in % of the frame per frame at 30 fps, from which stabilising is
# suggested.
_SHAKE_ON_PCT = 0.5


def shake_stride(fps: float | None) -> int:
    """Read every n-th frame so the run's frame rate is near 30 fps."""
    if not fps or fps <= 0:
        return 1
    return max(1, round(fps / SHAKE_FPS))


def _match_cost(
    a: bytes, b: bytes, width: int, height: int, dx: int, dy: int, margin: int
) -> float:
    """Mean absolute difference between ``a`` and ``b`` moved by (dx, dy),
    over the middle of the frame."""
    span = width - 2 * margin
    total = 0
    for y in range(margin, height - margin):
        row = y * width + margin
        other = (y + dy) * width + margin + dx
        total += sad_u8(a[row : row + span], b[other : other + span])
    return total / (span * (height - 2 * margin))


def _vertex(left: float, mid: float, right: float) -> float:
    """Where a parabola through three equally spaced costs bottoms out,
    from -0.5 to 0.5 around the middle one."""
    curve = left - 2.0 * mid + right
    if curve <= 0.0:
        return 0.0
    return max(-0.5, min(0.5, (left - right) / (2.0 * curve)))


def _global_shift(
    a: bytes, b: bytes, width: int, height: int, radius: int = _SHIFT_RADIUS
) -> tuple[float, float, float]:
    """How far the picture moved from ``a`` to ``b`` as (dx, dy) in pixels,
    to a fraction of a pixel, and the mean difference left after lining
    them up."""
    margin = radius + 2
    costs: dict[tuple[int, int], float] = {}

    def cost(dx: int, dy: int) -> float:
        if (dx, dy) not in costs:
            costs[dx, dy] = _match_cost(a, b, width, height, dx, dy, margin)
        return costs[dx, dy]

    def rank(shift: tuple[int, int]) -> tuple[float, int]:
        # On a tie, the smaller shift: a flat picture has not moved.
        return cost(*shift), abs(shift[0]) + abs(shift[1])

    coarse = range(-radius, radius + 1, 2)
    bx, by = min(((dx, dy) for dy in coarse for dx in coarse), key=rank)
    bx, by = min(((bx + i, by + j) for j in (-1, 0, 1) for i in (-1, 0, 1)), key=rank)
    mid = cost(bx, by)
    fx = _vertex(cost(bx - 1, by), mid, cost(bx + 1, by))
    fy = _vertex(cost(bx, by - 1), mid, cost(bx, by + 1))
    return bx + fx, by + fy, mid


def _run_moves(
    run: list[bytes], width: int, height: int
) -> tuple[list[tuple[float, float] | None], int]:
    """The picture's move between each pair of frames in % of the frame,
    None across a cut, with repeated frames left out; and how many repeats
    there were."""
    moves: list[tuple[float, float] | None] = []
    repeats = 0
    for a, b in zip(run, run[1:], strict=False):
        dx, dy, left = _global_shift(a, b, width, height)
        moved = round(dx) != 0 or round(dy) != 0
        if not moved and left <= _REPEAT_COST:
            repeats += 1
        elif left > _CUT_COST:
            moves.append(None)
        else:
            moves.append((100.0 * dx / width, 100.0 * dy / height))
    return moves, repeats


def _run_shake(
    run: list[bytes], width: int, height: int, fps: float = SHAKE_FPS
) -> tuple[float, float] | None:
    """Jitter and peak excursion of one run of frames, each in % of the
    frame, or None when no two neighbouring moves can be compared.

    Jitter is how much the picture's move changes from one frame to the
    next, scaled to 30 fps: a steady pan moves the same each frame and has
    none. The excursion is how far the picture strays from a steady path,
    which is what a stabiliser has to crop away. ``fps`` is the run's own
    frame rate."""
    moves, repeats = _run_moves(run, width, height)
    if not moves and repeats:
        return 0.0, 0.0  # every frame a repeat: a still picture
    steps = [
        math.hypot(m2[0] - m1[0], m2[1] - m1[1])
        for m1, m2 in zip(moves, moves[1:], strict=False)
        if m1 is not None and m2 is not None
    ]
    if not steps:
        return None
    # The move changes more between frames further apart, by the square of
    # the time between them; repeated frames were dropped, so the frames
    # left are further apart.
    pairs = len(run) - 1
    rate = fps * (pairs - repeats) / pairs
    jitter = sum(steps) / len(steps) * (rate / SHAKE_FPS) ** 2
    # The longest stretch without a cut, as positions from its first frame.
    best: list[tuple[float, float]] = []
    stretch: list[tuple[float, float]] = []
    for move in [*moves, None]:
        if move is None:
            best = max(best, stretch, key=len)
            stretch = []
        else:
            stretch.append(move)
    x = y = 0.0
    path = [(0.0, 0.0)]
    for mx, my in best:
        x, y = x + mx, y + my
        path.append((x, y))
    n = len(path) - 1
    excursion = max(
        math.hypot(px - x * k / n, py - y * k / n) for k, (px, py) in enumerate(path)
    )
    return jitter, excursion


def _estimate_shake(
    runs: list[list[bytes]],
    width: int = SHAKE_WIDTH,
    height: int = SHAKE_HEIGHT,
    fps: float = SHAKE_FPS,
) -> tuple[OpSuggestion, dict[str, float]]:
    measured = [m for m in (_run_shake(run, width, height, fps) for run in runs) if m]
    if not measured:
        suggestion = OpSuggestion(
            enabled=False,
            confidence=0.0,
            params={"shakiness": 5.0, "smoothing": 10.0, "max_crop_pct": 10.0},
            reason="no steady stretch of frames to measure shake on",
        )
        return suggestion, {"shake_jitter_pct": 0.0, "shake_excursion_pct": 0.0}

    # The middle window, or the calmer of the middle two, so one handheld
    # insert or one cut does not decide.
    jitters = sorted(j for j, _ in measured)
    jitter = jitters[(len(jitters) - 1) // 2]
    excursion = max(e for _, e in measured)
    enabled = jitter >= _SHAKE_ON_PCT
    shakiness = max(1.0, min(10.0, jitter * 3.0))
    smoothing = max(10.0, min(30.0, 10.0 + 2.0 * shakiness))
    # Crop away the stray on both sides, with half again for the parts of
    # the shot the sample did not see.
    max_crop = max(1.0, min(15.0, 3.0 * excursion))
    suggestion = OpSuggestion(
        enabled=enabled,
        confidence=min(1.0, jitter / (2.0 * _SHAKE_ON_PCT)) * len(measured) / len(runs),
        params={
            "shakiness": round(shakiness if enabled else 5.0, 2),
            "smoothing": round(smoothing if enabled else 10.0, 2),
            "max_crop_pct": round(max_crop if enabled else 10.0, 2),
        },
        reason=f"jitter={jitter:.2f}% excursion={excursion:.2f}% of the frame",
    )
    return suggestion, {
        "shake_jitter_pct": jitter,
        "shake_excursion_pct": excursion,
    }


def estimate_look(
    source: Path,
    facts: MediaFacts,
    *,
    ffmpeg_bin: str | None = None,
) -> LookEstimates:
    """Sample windows and produce classical look/motion priors."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise EstimatorError(f"source is not a readable file: {source}")
    if not facts.has_video:
        raise EstimatorError("look estimation requires a video stream")

    windows = choose_sample_windows(facts)
    gray_frames: list[bytes] = []
    rgb_means: list[tuple[float, float, float]] = []
    runs: list[list[bytes]] = []

    side = grain_crop_side(facts)
    video = facts.primary_video()
    fps = None
    if video is not None:
        fps = parse_frame_rate(video.avg_frame_rate) or parse_frame_rate(
            video.r_frame_rate
        )
    stride = shake_stride(fps)
    grain_frames: list[bytes] = []
    for window in windows:
        frame_a = extract_gray_frame(source, at_s=window.start_s, ffmpeg_bin=ffmpeg_bin)
        if side is not None:
            grain_frames.append(
                extract_gray_frame(
                    source,
                    at_s=window.start_s,
                    width=side,
                    height=side,
                    ffmpeg_bin=ffmpeg_bin,
                    crop=True,
                )
            )
        runs.append(
            extract_gray_run(
                source, at_s=window.start_s, ffmpeg_bin=ffmpeg_bin, stride=stride
            )
        )
        gray_frames.append(frame_a)
        rgb_means.append(
            extract_rgb_means(source, at_s=window.start_s, ffmpeg_bin=ffmpeg_bin)
        )

    # Each frame is its own ffmpeg process, so they decode side by side:
    # nine seeks into 1080p video take 0.6 s instead of 1.6 s.
    # A frame that will not decode is left out, as a blank one would be.
    def contrast_frame(at_s: float) -> bytes | None:
        try:
            return extract_gray_frame(source, at_s=at_s, ffmpeg_bin=ffmpeg_bin)
        except EstimatorError:
            return None

    length = media_duration(facts)
    if length is None:
        contrast, contrast_metrics = _no_contrast(
            "duration unknown: cannot spread contrast samples", {}
        )
    else:
        with ThreadPoolExecutor(max_workers=4) as pool:
            contrast_frames = [
                frame
                for frame in pool.map(contrast_frame, contrast_sample_times(length))
                if frame is not None
            ]
        contrast, contrast_metrics = _estimate_contrast(
            contrast_frames, full_range=is_full_range(facts)
        )
    white_balance, wb_metrics = _estimate_white_balance(rgb_means)
    if side is None:
        denoise, denoise_metrics = _estimate_denoise(gray_frames)
    else:
        denoise, denoise_metrics = _estimate_denoise(grain_frames, side)
    stabilize, shake_metrics = _estimate_shake(
        runs, fps=(fps / stride) if fps else SHAKE_FPS
    )

    metrics = {}
    metrics.update(contrast_metrics)
    metrics.update(wb_metrics)
    metrics.update(denoise_metrics)
    metrics.update(shake_metrics)

    return LookEstimates(
        schema_version=ESTIMATES_SCHEMA_VERSION,
        source_path=str(source),
        windows=windows,
        contrast=contrast,
        white_balance=white_balance,
        denoise=denoise,
        stabilize=stabilize,
        metrics=metrics,
    )


def estimates_path(job_dir: Path) -> Path:
    """Return the on-disk estimates path for a job."""
    return job_dir / ESTIMATES_FILE_NAME


def write_estimates(job_dir: Path, estimates: LookEstimates) -> Path:
    """Atomically write look estimates for a job."""
    path = estimates_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(estimates.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def load_estimates(job_dir: Path) -> LookEstimates:
    """Load look estimates from a job directory."""
    path = estimates_path(job_dir)
    if not path.is_file():
        raise EstimatorError(f"estimates not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise EstimatorError("estimates root must be an object")
    return LookEstimates.from_dict(data)
