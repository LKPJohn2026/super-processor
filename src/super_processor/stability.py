"""Shimmer and seam checks: the two failures a still frame cannot show.

**Shimmer.** Invented texture tends to crawl: it is redrawn a little
differently every frame. For a clip, measure the fine-detail layer (a frame
minus its Gaussian blur) two ways: how much there is, and how much it changes
from one frame to the next. Their ratio is how unstable the texture is.
:func:`shimmer_index` divides the restored clip's ratio by the source's, so
motion that moves both cancels out. Well above 1 means the restoration added
texture that is less stable than the footage.

**Seams.** FlashVSR starts again at every chunk, shot part, and revised
range, and its output can jump there. :func:`seam_score` looks at a short
window around such a time and compares the frame-to-frame change at that
frame with the change around it, in the output and in the source. A jump the
source does not have is a seam.

Both are rough measurements with fixed thresholds; they flag, and the editor
and Gemini decide.
"""

from __future__ import annotations

import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .probe import ProbeError, probe_file
from .shots import ShotError, parse_metadata_log
from .upscale import (
    CHUNK_OVERLAP_S,
    CHUNK_SECONDS,
    STRUCTURE_SIGMA,
    UpscalePlan,
    shot_timeline,
)

# A restored clip whose texture is this much less stable than the source's
# is flagged as flicker. Below MIN_DETAIL there is too little texture to
# shimmer.
SHIMMER_LIMIT = 1.6
MIN_DETAIL = 1.0
# A frame-to-frame jump this many times larger than its neighbours, beyond
# what the source does there, is a seam; tiny jumps never are.
SEAM_LIMIT = 2.5
SEAM_MIN_JUMP = 2.0
SEAM_WINDOW_S = 0.5
SEAM_WIDTH = 320


class StabilityError(RuntimeError):
    """Raised when a clip cannot be measured."""


def _metadata_means(
    video: Path, graph: str, key: str, *, ffmpeg_bin: str
) -> list[float]:
    try:
        completed = subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-nostats",
                "-loglevel",
                "info",
                "-i",
                str(video),
                "-an",
                "-filter_complex",
                graph + ",signalstats,metadata=mode=print",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StabilityError(f"failed to start ffmpeg: {exc}") from exc
    log = (completed.stderr or b"").decode("utf-8", errors="replace")
    if completed.returncode != 0:
        tail = " ".join(log.strip().splitlines()[-2:])
        raise StabilityError(f"ffmpeg could not measure {video.name}: {tail}")
    try:
        frames = parse_metadata_log(log)
    except ShotError as exc:  # pragma: no cover - parse never raises today
        raise StabilityError(str(exc)) from exc
    return [values[key] for _time, values in frames if key in values]


@dataclass(frozen=True, slots=True)
class TextureStats:
    """How much fine detail a clip has, and how much it changes per frame."""

    detail: float
    change: float

    @property
    def instability(self) -> float:
        return self.change / max(self.detail, 0.5)


def texture_stats(
    video: Path,
    *,
    sigma: float,
    size: tuple[int, int] | None = None,
    ffmpeg_bin: str = "ffmpeg",
) -> TextureStats:
    """Median fine-detail level and frame-to-frame detail change of ``video``.

    ``size`` first resizes the clip (lanczos), so a source can be measured at
    the restored clip's size.
    """
    resize = f"scale={size[0]}:{size[1]}:flags=lanczos," if size else ""
    layer = (
        f"[0:v]{resize}format=gray,split[a][b];[b]gblur=sigma={sigma:.3f}[l];"
        "[a][l]blend=all_expr="
    )
    detail = _metadata_means(
        video,
        layer + "'abs(A-B)'",
        "lavfi.signalstats.YAVG",
        ffmpeg_bin=ffmpeg_bin,
    )
    change = _metadata_means(
        video,
        layer + "'A-B+128',tblend=all_mode=difference",
        "lavfi.signalstats.YAVG",
        ffmpeg_bin=ffmpeg_bin,
    )
    if not detail or not change:
        raise StabilityError(f"{video.name} is too short to measure shimmer")
    return TextureStats(statistics.median(detail), statistics.median(change))


def shimmer_index(
    before: Path,
    after: Path,
    *,
    scale: int,
    after_size: tuple[int, int],
    ffmpeg_bin: str = "ffmpeg",
) -> float | None:
    """How much less stable the restored texture is than the source's.

    ``None`` when the restored clip has too little texture to judge.
    """
    sigma = STRUCTURE_SIGMA * scale
    restored = texture_stats(after, sigma=sigma, ffmpeg_bin=ffmpeg_bin)
    if restored.detail < MIN_DETAIL:
        return None
    source = texture_stats(before, sigma=sigma, size=after_size, ffmpeg_bin=ffmpeg_bin)
    return restored.instability / max(source.instability, 0.05)


def seam_candidates(plan: UpscalePlan, duration_s: float) -> list[float]:
    """Times where a render may have restarted FlashVSR.

    Shot parts with different settings start fresh, a part longer than one
    chunk restarts every ``CHUNK_SECONDS - CHUNK_OVERLAP_S``, and a revise
    splices at its edges. Checking a time that turns out clean costs a second
    of decoding.
    """
    times: set[float] = set()
    try:
        ordered = shot_timeline(plan.spans, duration_s)
    except Exception:  # noqa: BLE001 - a plan that no longer tiles: check edges
        ordered = sorted(plan.spans, key=lambda span: span.start_s)
    step = CHUNK_SECONDS - CHUNK_OVERLAP_S
    previous = None
    for span in ordered:
        settings = (span.scale, span.strength, span.look, span.regions)
        if previous is not None and settings != previous:
            times.add(span.start_s)
        previous = settings
        cursor = span.start_s + step
        while span.end_s - cursor > 0.5:
            times.add(cursor)
            cursor += step
    pending = plan.pending
    if pending is not None:
        times.update((pending.start_s, pending.end_s))
    return sorted(t for t in times if 0.5 < t < duration_s - 0.5)


def video_size(video: Path) -> tuple[int, int]:
    try:
        stream = probe_file(video).primary_video()
    except ProbeError as exc:
        raise StabilityError(f"could not read {video.name}: {exc}") from exc
    if stream is None or not stream.width or not stream.height:
        raise StabilityError(f"could not read the size of {video.name}")
    return stream.width, stream.height


def _gray_frames(
    video: Path, start_s: float, seconds: float, *, ffmpeg_bin: str
) -> tuple[list[bytes], int]:
    try:
        completed = subprocess.run(
            [
                ffmpeg_bin,
                "-v",
                "error",
                "-ss",
                f"{max(0.0, start_s):.6f}",
                "-t",
                f"{seconds:.6f}",
                "-i",
                str(video),
                "-an",
                "-vf",
                f"scale={SEAM_WIDTH}:-2:flags=area,format=gray",
                "-f",
                "rawvideo",
                "-",
            ],
            check=False,
            capture_output=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StabilityError(f"failed to start ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or b"").decode("utf-8", errors="replace")
        raise StabilityError(f"ffmpeg could not read {video.name}: {detail.strip()}")
    raw = completed.stdout
    width, height = video_size(video)
    rows = round(SEAM_WIDTH * height / width / 2) * 2
    size = SEAM_WIDTH * rows
    frames = [raw[i : i + size] for i in range(0, len(raw) - size + 1, size)]
    return frames, size


def _jumps(frames: list[bytes], size: int) -> list[float]:
    return [
        sum(abs(a - b) for a, b in zip(first, second, strict=True)) / size
        for first, second in zip(frames, frames[1:], strict=False)
    ]


def seam_score(
    output: Path, source: Path, time_s: float, *, ffmpeg_bin: str = "ffmpeg"
) -> float:
    """How much more the output jumps at ``time_s`` than the source does.

    The jump is the largest frame-to-frame change in a window around the
    time, relative to the median change in that window. Below
    :data:`SEAM_MIN_JUMP` (mean grey levels) the output jump counts as none.
    """
    start = time_s - SEAM_WINDOW_S
    out_frames, out_size = _gray_frames(
        output, start, 2 * SEAM_WINDOW_S, ffmpeg_bin=ffmpeg_bin
    )
    src_frames, src_size = _gray_frames(
        source, start, 2 * SEAM_WINDOW_S, ffmpeg_bin=ffmpeg_bin
    )
    out_jumps = _jumps(out_frames, out_size)
    src_jumps = _jumps(src_frames, src_size)
    if len(out_jumps) < 3 or len(src_jumps) < 3:
        return 0.0
    peak = max(range(len(out_jumps)), key=out_jumps.__getitem__)
    if out_jumps[peak] < SEAM_MIN_JUMP:
        return 0.0
    out_rest = statistics.median(out_jumps[:peak] + out_jumps[peak + 1 :])
    out_ratio = out_jumps[peak] / (out_rest + 0.5)
    src_peak = src_jumps[min(peak, len(src_jumps) - 1)]
    src_rest = statistics.median(src_jumps)
    src_ratio = src_peak / (src_rest + 0.5)
    return out_ratio / max(src_ratio, 1.0)


def find_seams(
    output: Path,
    source: Path,
    times: list[float],
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> list[dict[str, float]]:
    """The candidate times whose seam score is over :data:`SEAM_LIMIT`."""
    found: list[dict[str, float]] = []
    for time_s in times:
        score = seam_score(output, source, time_s, ffmpeg_bin=ffmpeg_bin)
        if score >= SEAM_LIMIT:
            found.append({"time_s": round(time_s, 3), "score": round(score, 2)})
    return found
