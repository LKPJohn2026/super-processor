"""Timeline segmentation bounds for the review loops."""

from __future__ import annotations

import json
import math
import re
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from . import histogram_u8, mean_luma, percentile_u8, sad_u8, variance_u8
from .estimators import extract_gray_frame, extract_rgb_means

MIN_SEGMENT_S = 5.0
MAX_SEGMENT_S = 120.0
MAX_SEGMENTS = 15
MAX_DURATION_S = 30.0 * 60.0
SAMPLES_FILE_NAME = "samples.json"
SAMPLES_SCHEMA_VERSION = 1
SEGMENTS_FILE_NAME = "segments.json"
SEGMENTS_SCHEMA_VERSION = 1
STILLS_DIR_NAME = "segment_stills"
SAMPLE_WIDTH = 160
SAMPLE_HEIGHT = 90

FrameReader = Callable[[float], tuple[bytes, tuple[float, float, float]]]


class SegmentError(ValueError):
    """Raised when a timeline cannot be segmented inside the bounds."""


@dataclass(slots=True)
class SampleRow:
    """One one-hertz feature row used to find photographic changes."""

    time_s: float
    luma_mean: float
    luma_p05: float
    luma_p95: float
    clip_low: float
    clip_high: float
    rb_cast: float
    variance: float
    motion: float
    subject_x: float
    upper_luma: float = 128.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SampleRow:
        def num(key: str, default: float | None = None) -> float:
            if key not in data:
                if default is None:
                    raise SegmentError(f"sample missing {key}")
                return default
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise SegmentError(f"sample {key} must be numeric")
            return float(value)

        return cls(
            time_s=num("time_s"),
            luma_mean=num("luma_mean"),
            luma_p05=num("luma_p05"),
            luma_p95=num("luma_p95"),
            clip_low=num("clip_low"),
            clip_high=num("clip_high"),
            rb_cast=num("rb_cast"),
            variance=num("variance"),
            motion=num("motion"),
            subject_x=num("subject_x"),
            upper_luma=num("upper_luma", 128.0),
        )


def change_score(before: SampleRow, after: SampleRow) -> float:
    """Weighted distance between two neighboring sample rows.

    Components are scaled into roughly unit ranges: luma and motion by 255,
    variance by 400, cast and subject position left as stored.
    """
    luma = abs(after.luma_mean - before.luma_mean) / 255.0
    cast = abs(after.rb_cast - before.rb_cast)
    variance = abs(after.variance - before.variance) / 400.0
    motion = abs(after.motion - before.motion) / 255.0
    subject = abs(after.subject_x - before.subject_x)
    return luma + (1.2 * cast) + (0.8 * variance) + motion + (0.6 * subject)


PROBLEM_NAMES: tuple[str, ...] = (
    "low_light",
    "low_contrast",
    "silhouette",
    "too_warm",
    "noisy",
    "shaky",
    "off_center",
)


def problem_scores(row: SampleRow) -> dict[str, float]:
    """Map one sample onto the photographic problem axes, each roughly 0–1+."""
    span = max(0.0, row.luma_p95 - row.luma_p05)
    subject_dark = max(0.0, (80.0 - row.luma_p05) / 80.0)
    background_bright = max(0.0, (row.upper_luma - 140.0) / 115.0)
    return {
        "low_light": max(0.0, (90.0 - row.luma_mean) / 90.0),
        "low_contrast": max(0.0, (80.0 - span) / 80.0),
        "silhouette": subject_dark * background_bright,
        "too_warm": max(0.0, row.rb_cast),
        "noisy": max(0.0, (row.variance - 150.0) / 250.0),
        "shaky": max(0.0, row.motion / 40.0),
        "off_center": min(1.0, abs(row.subject_x - 0.5) * 2.0),
    }


def _mean_problem_scores(rows: list[SampleRow]) -> dict[str, float]:
    totals = dict.fromkeys(PROBLEM_NAMES, 0.0)
    for row in rows:
        for name, score in problem_scores(row).items():
            totals[name] += score
    count = float(len(rows))
    return {name: totals[name] / count for name in PROBLEM_NAMES}


def primary_problem(rows: list[SampleRow]) -> str:
    """Return the strongest mean problem label for these samples."""
    if not rows:
        raise SegmentError("cannot label an empty sample list")
    means = _mean_problem_scores(rows)
    return max(PROBLEM_NAMES, key=lambda name: means[name])


def context_label(rows: list[SampleRow]) -> str:
    """Weak indoor/outdoor guess from average brightness and the upper frame."""
    if not rows:
        raise SegmentError("cannot label an empty sample list")
    mean_luma = sum(row.luma_mean for row in rows) / len(rows)
    upper = sum(row.upper_luma for row in rows) / len(rows)
    if upper >= 160.0 and mean_luma >= 100.0:
        return "outdoor"
    if mean_luma < 90.0:
        return "indoor"
    return "mixed"


def keyframe_time(rows: list[SampleRow]) -> float:
    """Return the timestamp whose primary problem is the strongest."""
    if not rows:
        raise SegmentError("cannot pick a key frame from an empty sample list")
    problem = primary_problem(rows)
    chosen = max(rows, key=lambda row: problem_scores(row)[problem])
    return chosen.time_s


def look_group_ids(labels: list[tuple[str, str]]) -> list[int]:
    """Share an id across neighboring segments with the same context and problem."""
    if not labels:
        return []
    ids: list[int] = []
    current = 0
    previous: tuple[str, str] | None = None
    for label in labels:
        if previous is not None and label != previous:
            current += 1
        ids.append(current)
        previous = label
    return ids


def legal_segment_counts(duration_s: float) -> tuple[int, int]:
    """Return the inclusive ``(fewest, most)`` segment counts for ``duration_s``.

    A segment is 5–120 seconds, and a timeline holds at most 15 of them.
    Durations longer than 30 minutes, or too short to hold one segment, are
    refused.
    """
    if duration_s <= 0:
        raise SegmentError("duration must be positive")
    if duration_s > MAX_DURATION_S:
        raise SegmentError(
            f"duration {duration_s:.1f}s exceeds the {MAX_DURATION_S:.0f}s limit"
        )
    fewest = math.ceil(duration_s / MAX_SEGMENT_S)
    most = min(MAX_SEGMENTS, math.floor(duration_s / MIN_SEGMENT_S))
    if fewest > most or most < 1:
        raise SegmentError(
            f"duration {duration_s:.1f}s cannot be split into "
            f"{MIN_SEGMENT_S:.0f}–{MAX_SEGMENT_S:.0f}s segments"
        )
    return fewest, most


def _equal_pieces(start: float, end: float) -> list[tuple[float, float]]:
    """Cut ``[start, end]`` into equal pieces of at most ``MAX_SEGMENT_S``."""
    length = end - start
    count = max(1, math.ceil(length / MAX_SEGMENT_S - 1e-9))
    step = length / count
    pieces: list[tuple[float, float]] = []
    cursor = start
    for index in range(count):
        nxt = end if index == count - 1 else cursor + step
        pieces.append((cursor, nxt))
        cursor = nxt
    return pieces


def _strongest_interior_cut(
    start: float,
    end: float,
    scores: list[float],
) -> float | None:
    """Return the integer second with the strongest positive change score."""
    best_time: float | None = None
    best_score = 0.0
    time_s = int(math.floor(start)) + 1
    while time_s < end:
        index = time_s - 1
        if 0 <= index < len(scores) and scores[index] > best_score:
            best_score = scores[index]
            best_time = float(time_s)
        time_s += 1
    return best_time


def split_oversized(
    intervals: list[tuple[float, float]],
    scores: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Break every interval longer than ``MAX_SEGMENT_S``.

    A flat interval (no positive interior score) is divided into equal pieces.
    Otherwise the cut falls on the strongest interior score, and both sides are
    repaired the same way.
    """
    repaired: list[tuple[float, float]] = []
    for start, end in intervals:
        if end < start:
            raise SegmentError("interval end is before its start")
        length = end - start
        if length <= MAX_SEGMENT_S + 1e-9:
            repaired.append((start, end))
            continue
        cut = None if scores is None else _strongest_interior_cut(start, end, scores)
        if cut is None:
            repaired.extend(_equal_pieces(start, end))
            continue
        repaired.extend(split_oversized([(start, cut), (cut, end)], scores))
    return repaired


def _boundary_score(time_s: float, scores: list[float] | None) -> float:
    if not scores:
        return 0.0
    index = int(round(time_s)) - 1
    if index < 0 or index >= len(scores):
        return 0.0
    return scores[index]


def _merge_pair(
    intervals: list[tuple[float, float]],
    index: int,
) -> list[tuple[float, float]]:
    start = intervals[index][0]
    end = intervals[index + 1][1]
    return [*intervals[:index], (start, end), *intervals[index + 2 :]]


def _merge_shorter_than_min(
    intervals: list[tuple[float, float]],
    scores: list[float] | None,
) -> list[tuple[float, float]]:
    """Absorb pieces shorter than ``MIN_SEGMENT_S`` into a neighbor."""
    pieces = list(intervals)
    guard = 0
    while guard < len(pieces) + 2:
        guard += 1
        short_index = next(
            (
                index
                for index, (start, end) in enumerate(pieces)
                if end - start < MIN_SEGMENT_S - 1e-6
            ),
            None,
        )
        if short_index is None or len(pieces) == 1:
            return pieces
        left = short_index - 1 if short_index > 0 else None
        right = short_index if short_index < len(pieces) - 1 else None
        if left is None and right is None:
            return pieces
        if left is None:
            assert right is not None
            pieces = _merge_pair(pieces, right)
            continue
        if right is None:
            pieces = _merge_pair(pieces, left)
            continue
        left_score = _boundary_score(pieces[short_index][0], scores)
        right_score = _boundary_score(pieces[short_index][1], scores)
        # Equal scores absorb into the previous piece, which keeps a short tail stable.
        pieces = _merge_pair(pieces, left if left_score <= right_score else right)
    return pieces


def merge_short_and_cap(
    intervals: list[tuple[float, float]],
    scores: list[float] | None = None,
) -> list[tuple[float, float]]:
    """Absorb sub-5s pieces, then merge down to ``MAX_SEGMENTS``.

    The count cap merges the weakest boundary first and skips a merge that
    would exceed ``MAX_SEGMENT_S``. A merge of a short piece that overshoots
    the cap is split again so the result still respects both bounds.
    """
    pieces = _merge_shorter_than_min(list(intervals), scores)
    pieces = split_oversized(pieces, scores)
    while len(pieces) > MAX_SEGMENTS:
        candidates: list[tuple[float, int]] = []
        for index in range(len(pieces) - 1):
            combined = pieces[index + 1][1] - pieces[index][0]
            if combined > MAX_SEGMENT_S + 1e-6:
                continue
            boundary = pieces[index][1]
            candidates.append((_boundary_score(boundary, scores), index))
        if not candidates:
            raise SegmentError(
                f"cannot merge {len(pieces)} segments without exceeding "
                f"{MAX_SEGMENT_S:.0f}s"
            )
        _, index = min(candidates, key=lambda item: (item[0], item[1]))
        pieces = _merge_pair(pieces, index)
    return pieces


def samples_path(job_dir: Path) -> Path:
    """Return the on-disk sample list for a job."""
    return job_dir / SAMPLES_FILE_NAME


def write_samples(job_dir: Path, rows: list[SampleRow]) -> Path:
    """Atomically write one-hertz sample rows."""
    path = samples_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SAMPLES_SCHEMA_VERSION,
        "samples": [row.to_dict() for row in rows],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def load_samples(job_dir: Path) -> list[SampleRow]:
    """Load one-hertz sample rows from a job directory."""
    path = samples_path(job_dir)
    if not path.is_file():
        raise SegmentError(f"samples not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SegmentError("samples root must be an object")
    raw_rows = data.get("samples")
    if not isinstance(raw_rows, list):
        raise SegmentError("samples list is missing")
    rows: list[SampleRow] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            raise SegmentError("each sample must be an object")
        rows.append(SampleRow.from_dict(item))
    return rows


def _row_from_frame(
    time_s: float,
    frame: bytes,
    rgb: tuple[float, float, float],
    previous: bytes | None,
) -> SampleRow:
    hist = array("Q", [0]) * 256
    histogram_u8(frame, memoryview(hist))
    total = float(sum(hist)) or 1.0
    upper_rows = SAMPLE_HEIGHT // 3
    upper = frame[: SAMPLE_WIDTH * upper_rows]
    red, green, blue = rgb
    cast = (blue - red) / max(1.0, (red + green + blue) / 3.0)
    motion = 0.0 if previous is None else sad_u8(previous, frame) / float(len(frame))
    return SampleRow(
        time_s=time_s,
        luma_mean=mean_luma(frame),
        luma_p05=percentile_u8(frame, 5.0),
        luma_p95=percentile_u8(frame, 95.0),
        clip_low=hist[0] / total,
        clip_high=hist[255] / total,
        rb_cast=cast,
        variance=variance_u8(frame),
        motion=motion,
        subject_x=0.5,
        upper_luma=mean_luma(upper) if upper else 0.0,
    )


def sample_from_reader(duration_s: float, read_frame: FrameReader) -> list[SampleRow]:
    """Walk one second at a time and store feature rows.

    ``read_frame`` returns a packed gray frame of ``SAMPLE_WIDTH`` by
    ``SAMPLE_HEIGHT`` plus mean red, green, and blue. Duration limits are
    enforced before any frame is read.
    """
    legal_segment_counts(duration_s)
    count = max(1, int(math.floor(duration_s)))
    pixels = SAMPLE_WIDTH * SAMPLE_HEIGHT
    rows: list[SampleRow] = []
    previous: bytes | None = None
    for index in range(count):
        time_s = float(index)
        gray, rgb = read_frame(time_s)
        if len(gray) < pixels:
            raise SegmentError(
                f"short frame at {time_s:.0f}s ({len(gray)} < {pixels} bytes)"
            )
        frame = gray[:pixels]
        rows.append(_row_from_frame(time_s, frame, rgb, previous))
        previous = frame
    return rows


def ffmpeg_frame_reader(source: Path, *, ffmpeg_bin: str | None = None) -> FrameReader:
    """Return a reader that decodes one gray frame and RGB means per second."""

    def read_frame(time_s: float) -> tuple[bytes, tuple[float, float, float]]:
        gray = extract_gray_frame(
            source,
            at_s=time_s,
            width=SAMPLE_WIDTH,
            height=SAMPLE_HEIGHT,
            ffmpeg_bin=ffmpeg_bin,
        )
        rgb = extract_rgb_means(source, at_s=time_s, ffmpeg_bin=ffmpeg_bin)
        return gray, rgb

    return read_frame


def sample_media(
    source: Path, duration_s: float, *, ffmpeg_bin: str | None = None
) -> list[SampleRow]:
    """Sample one feature row per second from a media file."""
    return sample_from_reader(
        duration_s, ffmpeg_frame_reader(source, ffmpeg_bin=ffmpeg_bin)
    )


@dataclass(slots=True)
class TimelineSegment:
    """One labeled span of the timeline, plus the still that represents it."""

    index: int
    start_s: float
    end_s: float
    context: str
    problem: str
    keyframe_s: float
    look_group: int
    still_path: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TimelineSegment:
        def num(key: str) -> float:
            value = data[key]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise SegmentError(f"segment {key} must be numeric")
            return float(value)

        def text(key: str) -> str:
            value = data.get(key, "")
            if not isinstance(value, str):
                raise SegmentError(f"segment {key} must be text")
            return value

        return cls(
            index=int(num("index")),
            start_s=num("start_s"),
            end_s=num("end_s"),
            context=text("context"),
            problem=text("problem"),
            keyframe_s=num("keyframe_s"),
            look_group=int(num("look_group")),
            still_path=text("still_path"),
        )


def boundary_scores(rows: list[SampleRow]) -> list[float]:
    """Change score for the boundary after each sample except the last."""
    return [
        change_score(rows[index], rows[index + 1]) for index in range(len(rows) - 1)
    ]


def _rows_in_span(rows: list[SampleRow], start: float, end: float) -> list[SampleRow]:
    return [row for row in rows if start <= row.time_s < end]


def propose_segments(rows: list[SampleRow]) -> list[TimelineSegment]:
    """Split sample rows into labeled segments inside the duration bounds."""
    if not rows:
        raise SegmentError("cannot segment an empty sample list")
    duration = rows[-1].time_s + 1.0
    legal_segment_counts(duration)
    intervals = merge_short_and_cap([(0.0, duration)], boundary_scores(rows))
    spans = [_rows_in_span(rows, start, end) for start, end in intervals]
    if any(not span for span in spans):
        raise SegmentError("a segment contains no sample rows")
    labels = [(context_label(span), primary_problem(span)) for span in spans]
    groups = look_group_ids(labels)
    segments: list[TimelineSegment] = []
    for index, ((start, end), span, (context, problem), group) in enumerate(
        zip(intervals, spans, labels, groups, strict=True)
    ):
        segments.append(
            TimelineSegment(
                index=index,
                start_s=start,
                end_s=end,
                context=context,
                problem=problem,
                keyframe_s=keyframe_time(span),
                look_group=group,
            )
        )
    return segments


def write_gray_still(path: Path, gray: bytes) -> None:
    """Write a packed gray frame as a binary PPM still."""
    pixels = SAMPLE_WIDTH * SAMPLE_HEIGHT
    if len(gray) < pixels:
        raise SegmentError(f"short still frame ({len(gray)} < {pixels} bytes)")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"P5\n{SAMPLE_WIDTH} {SAMPLE_HEIGHT}\n255\n".encode()
    path.write_bytes(header + gray[:pixels])


def segments_path(job_dir: Path) -> Path:
    """Return the on-disk segment list for a job."""
    return job_dir / SEGMENTS_FILE_NAME


def write_segments(job_dir: Path, segments: list[TimelineSegment]) -> Path:
    """Atomically write the proposed segment list."""
    path = segments_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SEGMENTS_SCHEMA_VERSION,
        "segments": [segment.to_dict() for segment in segments],
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    return path


def load_segments(job_dir: Path) -> list[TimelineSegment]:
    """Load a proposed segment list from a job directory."""
    path = segments_path(job_dir)
    if not path.is_file():
        raise SegmentError(f"segments not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SegmentError("segments root must be an object")
    raw_rows = data.get("segments")
    if not isinstance(raw_rows, list):
        raise SegmentError("segments list is missing")
    segments: list[TimelineSegment] = []
    for item in raw_rows:
        if not isinstance(item, dict):
            raise SegmentError("each segment must be an object")
        segments.append(TimelineSegment.from_dict(item))
    return segments


def write_segment_review(
    job_dir: Path,
    rows: list[SampleRow],
    read_frame: FrameReader | None = None,
) -> list[TimelineSegment]:
    """Propose segments, store one still per key frame, and write segments.json."""
    segments = propose_segments(rows)
    if read_frame is not None:
        for segment in segments:
            gray, _rgb = read_frame(segment.keyframe_s)
            relative = f"{STILLS_DIR_NAME}/seg_{segment.index:02d}.ppm"
            write_gray_still(job_dir / relative, gray)
            segment.still_path = relative
    write_segments(job_dir, segments)
    return segments


_MOVE_NOTE = re.compile(
    r"(?:segment\s+(?P<index>\d+)\s+)?"
    r"(?:the\s+)?(?:(?P<cue>warm|dark|bright)\s+part\s+)?"
    r"starts\s+(?:(?P<amount>\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\s+)?"
    r"(?P<direction>later|earlier)\b"
)
_RELABEL_NOTE = re.compile(
    r"(?:segment\s+(?P<index>\d+)\s+)?(?:this\s+is|call\s+this)\s+(?P<label>.+)$"
)
_NOTE_CONTEXTS = {"indoor": "indoor", "outdoor": "outdoor", "mixed": "mixed"}
_NOTE_PROBLEMS = {
    "low light": "low_light",
    "low contrast": "low_contrast",
    "silhouette": "silhouette",
    "too warm": "too_warm",
    "warm": "too_warm",
    "noisy": "noisy",
    "shaky": "shaky",
    "off center": "off_center",
    "off-center": "off_center",
}
_NOTE_CUES = {"warm": "too_warm", "dark": "low_light", "bright": "low_contrast"}


@dataclass(frozen=True, slots=True)
class SplitNote:
    """A structured correction the split loop already knows how to apply."""

    intent: str
    segment_index: int | None = None
    delta_s: float | None = None
    context: str | None = None
    problem: str | None = None


def _note_index(raw: str | None) -> int | None:
    if raw is None:
        return None
    return int(raw)


def parse_split_note(text: str) -> SplitNote:
    """Turn a short split note into one known intent.

    Unknown wording raises ``SegmentError`` so the caller can ask for a rephrase.
    A note that matches more than one intent is also refused.
    """
    cleaned = " ".join(text.strip().lower().split())
    if not cleaned:
        raise SegmentError("rephrase the note")
    matched: list[SplitNote] = []
    if "too many" in cleaned:
        matched.append(SplitNote(intent="too_many"))
    if "too few" in cleaned or "not enough" in cleaned:
        matched.append(SplitNote(intent="too_few"))
    move = _MOVE_NOTE.search(cleaned)
    if move is not None:
        amount = move.group("amount")
        delta = 5.0 if amount is None else float(amount)
        if move.group("direction") == "earlier":
            delta = -delta
        cue = move.group("cue")
        matched.append(
            SplitNote(
                intent="move_boundary",
                segment_index=_note_index(move.group("index")),
                delta_s=delta,
                problem=None if cue is None else _NOTE_CUES[cue],
            )
        )
    relabel = _RELABEL_NOTE.search(cleaned)
    if relabel is not None:
        label = relabel.group("label").strip(" .")
        context = _NOTE_CONTEXTS.get(label)
        problem = _NOTE_PROBLEMS.get(label)
        if context is None and problem is None:
            raise SegmentError("rephrase the note")
        matched.append(
            SplitNote(
                intent="relabel",
                segment_index=_note_index(relabel.group("index")),
                context=context,
                problem=problem,
            )
        )
    if len(matched) != 1:
        raise SegmentError("rephrase the note")
    return matched[0]
