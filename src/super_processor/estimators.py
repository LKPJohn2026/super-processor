"""Classical look/motion estimators from sampled media windows."""

from __future__ import annotations

import json
import math
import subprocess
from array import array
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import histogram_u8, mean_luma, percentile_u8, sad_u8, variance_u8
from .doctor import which
from .probe import MediaFacts

ESTIMATES_FILE_NAME = "estimates.json"
ESTIMATES_SCHEMA_VERSION = 1


class EstimatorError(RuntimeError):
    """Raised when look/motion estimation cannot complete."""


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


def extract_gray_frame(
    source: Path,
    *,
    at_s: float,
    width: int = 160,
    height: int = 90,
    ffmpeg_bin: str | None = None,
) -> bytes:
    """Decode one grayscale frame near ``at_s`` as packed 8-bit luma."""
    binary = ffmpeg_bin or _ffmpeg_bin()
    cmd = [
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{at_s:g}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:{height},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EstimatorError(f"failed to sample frame at {at_s:g}s: {exc}") from exc
    if completed.returncode != 0 or not completed.stdout:
        detail = (completed.stderr or b"").decode("utf-8", errors="replace").strip()
        raise EstimatorError(
            f"ffmpeg frame extract failed at {at_s:g}s"
            + (f": {detail}" if detail else "")
        )
    expected = width * height
    if len(completed.stdout) < expected:
        raise EstimatorError(
            f"short frame buffer at {at_s:g}s "
            f"({len(completed.stdout)} < {expected} bytes)"
        )
    return completed.stdout[:expected]


def extract_rgb_means(
    source: Path,
    *,
    at_s: float,
    width: int = 80,
    height: int = 45,
    ffmpeg_bin: str | None = None,
) -> tuple[float, float, float]:
    """Return mean R/G/B for a downscaled frame (white-balance prior)."""
    binary = ffmpeg_bin or _ffmpeg_bin()
    cmd = [
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{at_s:g}",
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
    completed = subprocess.run(cmd, check=False, capture_output=True, timeout=30)
    if completed.returncode != 0 or not completed.stdout:
        raise EstimatorError(f"ffmpeg rgb extract failed at {at_s:g}s")
    raw = completed.stdout
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


def _estimate_contrast(frames: list[bytes]) -> tuple[OpSuggestion, dict[str, float]]:
    p05_vals: list[float] = []
    p95_vals: list[float] = []
    means: list[float] = []
    clip_lows: list[float] = []
    clip_highs: list[float] = []
    hist = array("Q", [0]) * 256

    for frame in frames:
        histogram_u8(frame, memoryview(hist))
        total = float(sum(hist))
        p05_vals.append(percentile_u8(frame, 5.0))
        p95_vals.append(percentile_u8(frame, 95.0))
        means.append(mean_luma(frame))
        clip_lows.append(hist[0] / total)
        clip_highs.append(hist[255] / total)

    p05 = sum(p05_vals) / len(p05_vals)
    p95 = sum(p95_vals) / len(p95_vals)
    mean = sum(means) / len(means)
    span = max(1.0, p95 - p05)
    # Map compressed spans toward a mild contrast boost.
    contrast = max(0.5, min(2.0, 180.0 / span))
    brightness = max(-0.5, min(0.5, (128.0 - mean) / 255.0))
    clip = (sum(clip_lows) + sum(clip_highs)) / len(frames)
    enabled = contrast > 1.08 or abs(brightness) > 0.04 or clip > 0.02
    confidence = min(
        1.0, abs(contrast - 1.0) * 2.0 + abs(brightness) * 2.0 + clip * 5.0
    )
    suggestion = OpSuggestion(
        enabled=enabled,
        confidence=confidence,
        params={
            "contrast": round(contrast if enabled else 1.0, 3),
            "brightness": round(brightness if enabled else 0.0, 3),
            "gamma": 1.0,
        },
        reason=(f"luma p05={p05:.1f} p95={p95:.1f} mean={mean:.1f} clip={clip:.3f}"),
    )
    metrics = {
        "luma_p05": p05,
        "luma_p95": p95,
        "luma_mean": mean,
        "clip_fraction": clip,
    }
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


def _estimate_denoise(frames: list[bytes]) -> tuple[OpSuggestion, dict[str, float]]:
    # High-frequency proxy: variance of local luma in flat-ish frames.
    variances = [variance_u8(frame) for frame in frames]
    noise = sum(variances) / len(variances)
    # Map variance onto [0,1] strength; quiet studio ~ <80, noisy phone >> 400.
    strength = max(0.0, min(1.0, (math.sqrt(noise) - 8.0) / 25.0))
    enabled = strength >= 0.18
    suggestion = OpSuggestion(
        enabled=enabled,
        confidence=min(1.0, strength * 1.4),
        params={"strength": round(strength if enabled else 0.0, 3)},
        reason=f"luma_variance={noise:.1f}",
    )
    return suggestion, {"luma_variance": noise, "denoise_strength": strength}


def _estimate_shake(
    frame_pairs: list[tuple[bytes, bytes]],
) -> tuple[OpSuggestion, dict[str, float]]:
    if not frame_pairs:
        suggestion = OpSuggestion(
            enabled=False,
            confidence=0.0,
            params={"shakiness": 5.0, "smoothing": 10.0, "max_crop_pct": 10.0},
            reason="insufficient frames for shake estimate",
        )
        return suggestion, {"shake_sad_norm": 0.0}

    norms: list[float] = []
    for left, right in frame_pairs:
        sad = float(sad_u8(left, right))
        norms.append(sad / max(1.0, float(len(left))))
    shake = sum(norms) / len(norms)
    # Normalized SAD per pixel; still content motion, so keep threshold moderate.
    shakiness = max(1.0, min(10.0, shake * 12.0))
    smoothing = max(1.0, min(50.0, 8.0 + shake * 40.0))
    max_crop = max(0.0, min(30.0, 5.0 + shake * 20.0))
    enabled = shake >= 0.35
    suggestion = OpSuggestion(
        enabled=enabled,
        confidence=min(1.0, shake / 1.2),
        params={
            "shakiness": round(shakiness if enabled else 5.0, 2),
            "smoothing": round(smoothing if enabled else 10.0, 2),
            "max_crop_pct": round(max_crop if enabled else 10.0, 2),
        },
        reason=f"mean_sad_norm={shake:.3f}",
    )
    return suggestion, {"shake_sad_norm": shake}


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
    duration = max(facts.duration_s or 1.0, 0.5)
    gray_frames: list[bytes] = []
    rgb_means: list[tuple[float, float, float]] = []
    pair_frames: list[tuple[bytes, bytes]] = []

    for window in windows:
        frame_a = extract_gray_frame(source, at_s=window.start_s, ffmpeg_bin=ffmpeg_bin)
        pair_at = min(
            max(0.0, duration - 0.05),
            window.start_s + min(0.12, window.duration_s),
        )
        if abs(pair_at - window.start_s) < 0.03:
            pair_at = min(duration - 0.05, window.start_s + 0.08)
        frame_b = extract_gray_frame(source, at_s=pair_at, ffmpeg_bin=ffmpeg_bin)
        gray_frames.append(frame_a)
        pair_frames.append((frame_a, frame_b))
        rgb_means.append(
            extract_rgb_means(source, at_s=window.start_s, ffmpeg_bin=ffmpeg_bin)
        )

    contrast, contrast_metrics = _estimate_contrast(gray_frames)
    white_balance, wb_metrics = _estimate_white_balance(rgb_means)
    denoise, denoise_metrics = _estimate_denoise(gray_frames)
    stabilize, shake_metrics = _estimate_shake(pair_frames)

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
