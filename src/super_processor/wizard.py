"""Localhost wizard: setup → pick → segments → previews → render → result."""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, TypeVar

from .atomic_io import (
    read_text_retrying,
    rmtree_quietly,
    unlink_quietly,
    unlink_retrying,
    write_text_atomic,
)
from .cancel import Cancelled
from .file_dialog import VIDEO_EXTENSIONS, FileDialogError, choose_video
from .gemini import (
    ChatTurn,
    GeminiClient,
    GeminiError,
    GeminiVideo,
    append_chat_turn,
    gemini_api_key_source,
    resolve_gemini_api_key,
    store_gemini_api_key,
)
from .gpu_heat import GpuHeat, SlowdownWatch, read_gpu_heat
from .jobs import JobError, JobManifest, JobState, JobStore, utc_now_iso
from .look import LOOK_BOUNDS
from .probe import MediaFacts, ProbeError, probe_file
from .progress import StepClock, overall_pct
from .regions import MAX_REGIONS, REGION_STRENGTH
from .segment_loop import (
    GEMINI_COPY_NAME,
    MAX_LOOP_SEGMENTS,
    MIN_LOOP_SEGMENT_S,
    LoopSegment,
    SegmentList,
    SegmentLoopError,
    SegmentStill,
    extract_segment_stills,
    load_segment_list,
    load_segment_stills,
    make_gemini_copy,
    merge_segments,
    move_cut,
    repair_coverage,
    rounded_segments,
    sample_times,
    save_segment_list,
    shots_from_segments,
    split_segment,
    whole_video,
)
from .segments import MAX_DURATION_S
from .shot_plan import (
    CHECK_PROBLEMS,
    PROTECTED_STRENGTH,
    UPSCALE_SCALES,
    ShotPlan,
    ShotPlanError,
    apply_checks,
    apply_proposals,
    apply_regions,
    check_stills,
    default_scale,
    fallback_recipe,
    finish,
    guard_shimmer,
    keep_editor_choices,
    load_plan,
    locate_stills,
    locate_targets,
    measure_shimmer,
    new_plan,
    note_scale,
    plan_path,
    random_preview_start,
    remove_previews,
    render_previews,
    reply_scale,
    request_redo,
    save_plan,
    set_upscale,
    shots_payload,
)
from .shots import (
    CONTENTS,
    Shot,
    ShotError,
    ShotList,
    load_shots,
    save_shots,
    shots_path,
)
from .stability import StabilityError, find_seams, seam_candidates
from .upscale import (
    UPSCALE_PLAN_FILE,
    SpanEngine,
    UpscaleEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleSpan,
    _ffprobe_for,
    default_span,
    load_upscale_plan,
    render_control,
    render_plan,
    rendered_scale,
    replace_overlapping,
    require_supported_timing,
    save_upscale_plan,
    video_stream_facts,
)
from .wizard_pages import (
    render_analyzing,
    render_done,
    render_interrupted,
    render_intro,
    render_llm_choice,
    render_local_llm_stub,
    render_pick,
    render_preview_segment,
    render_ready,
    render_rendering,
    render_result,
    render_sampling,
    render_segments,
    render_setup,
    with_navigation,
    with_notice,
)

WIZARD_STATE_FILE = "wizard_state.json"
WIZARD_SCHEMA_VERSION = 3
# Inputs are limited to 1080p (either orientation) and 30 minutes. SPAN
# at 2x already makes 4K from 1080p; larger inputs exceed what one GPU pass
# and the job directory are sized for.
MAX_INPUT_LONG_EDGE = 1920
MAX_INPUT_SHORT_EDGE = 1080
# The upload Gemini watches, and its File API record (see _gemini_video).
GEMINI_VIDEO_FILE = "gemini_video.json"
# A render's time left is saved at least this often, so the page's estimate
# stays fresh between whole-percent steps.
_PROGRESS_SAVE_S = 5.0
# How often a render reads the GPU's heat (see super_processor.gpu_heat).
HEAT_SAMPLE_S = 10.0


class WizardError(RuntimeError):
    """Raised when a wizard step cannot complete."""


class WizardConflict(WizardError):
    """A request was meant for a job this window no longer has open.

    The page sent the ``job_id`` it was showing, and another window (or a
    new pick) has since moved the wizard to a different job. The server
    answers 409 so the page reloads instead of changing the wrong job.
    """


class WizardStep(str, Enum):
    """Persisted wizard screen."""

    INTRO = "intro"
    LLM_CHOICE = "llm_choice"
    LOCAL_LLM_STUB = "local_llm_stub"
    SETUP = "setup"
    PICK_FILE = "pick_file"
    # Segment loop: sample stills, Gemini proposes segments, the editor
    # approves them, then each segment is analyzed and previewed in turn.
    SAMPLING = "sampling"
    PROPOSING = "proposing"
    SEGMENTS = "segments"
    ANALYZING = "analyzing"
    PREVIEW_SEGMENT = "preview_segment"
    READY_TO_RENDER = "ready_to_render"
    RENDERING = "rendering"
    RESULT = "result"
    DONE = "done"


# Steps of the removed Gemini split / FFmpeg enhance flow. A state file saved
# on one of them is sent back to the pick step. That flow also had an
# "analyzing" step; the segment loop reuses the name from schema 3 on, so
# only an older file on it is retired.
_RETIRED_STEPS = frozenset({"overview", "choose_split", "enhance"})
_REUSED_STEPS = frozenset({"analyzing"})
_REUSED_SINCE = 3
# The shot flow the segment loop replaced in 3.2.
_SHOT_FLOW_STEPS = frozenset({"finding_shots", "shots", "planning", "looks"})
_SHOT_FLOW_MESSAGE = (
    "this job used the shot flow, which the segment loop replaced; pick the file again"
)
_RETIRED_MESSAGE = (
    "this job used the removed split and enhance flow; pick the file again"
)


_SEGMENTING_STEPS = frozenset({WizardStep.SAMPLING, WizardStep.PROPOSING})

RunKind = Literal["segmenting", "analyze", "render"]
RunOutcome = Literal["running", "done", "failed", "stopped"]
# The steps each kind of run works on. A run starts only on one of them.
_RUN_STEPS: dict[RunKind, frozenset[WizardStep]] = {
    "segmenting": _SEGMENTING_STEPS,
    "analyze": frozenset({WizardStep.ANALYZING}),
    "render": frozenset({WizardStep.RENDERING}),
}
# The steps the editor can stop: finding the segments, and the render.
_CANCEL_STEPS = frozenset({*_SEGMENTING_STEPS, WizardStep.RENDERING})
# Left in the job folder while a stop is asked for and not yet landed, so a
# stop asked for just before the server went away still lands when the run
# starts again, instead of the work carrying on by itself.
CANCEL_MARKER = "cancel_requested"
# How often a run waiting on a Gemini call looks for a stop.
_ABANDON_POLL_S = 0.2
_STOPPED_RENDER_KEPT = (
    "You stopped the render. The video below is the one you had before, "
    "with its settings."
)
_STOPPED_RENDER_NONE = (
    "You stopped the render, so there is no result yet. Press Render to "
    "start it again; every approved scene is kept."
)
_STOPPED_SEGMENTS_KEPT = (
    "You stopped Gemini. These are the segments from before; approve them "
    "or send another note."
)
_STOPPED_SEGMENTS_NONE = (
    "You stopped finding the segments. Pick the file again to start over."
)
# Every step a run works on. Reopened on one of these (after the server
# restarted, or from the job list) a job is "interrupted": its work waits
# for the editor to resume it or stop it instead of starting by itself.
_RESUME_STEPS = frozenset(step for steps in _RUN_STEPS.values() for step in steps)
_STOPPED_ANALYZE = (
    "You stopped before this scene was looked at. Your approved segments "
    "are kept; approve them to carry on."
)
_RUNNING_ELSEWHERE = (
    "this job is being worked on by another Super Processor window on this "
    "computer; wait for it to finish there"
)
# Written in the job folder while a run works on it, so another server on
# the same jobs folder can tell the job is busy (see jobs_view).
WORKER_PID_FILE = "worker.pid"
# How long starting a worker waits for a POST in progress to finish.
_POST_WAIT_S = 30.0
_JOB_CONFLICT_MESSAGE = "this job is no longer open in this window"
# The stages of one job, as the page's stepper shows them, and the stage
# each step belongs to. Intro, the LLM choice and setup belong to none.
JOB_STAGES = ("file", "segments", "scenes", "render", "result")
_STEP_STAGES: dict[WizardStep, str] = {
    WizardStep.PICK_FILE: "file",
    WizardStep.SAMPLING: "segments",
    WizardStep.PROPOSING: "segments",
    WizardStep.SEGMENTS: "segments",
    WizardStep.ANALYZING: "scenes",
    WizardStep.PREVIEW_SEGMENT: "scenes",
    WizardStep.READY_TO_RENDER: "scenes",
    WizardStep.RENDERING: "render",
    WizardStep.RESULT: "result",
    WizardStep.DONE: "result",
}
# Steps a scene can be reopened from: not while one is being analyzed.
_REOPEN_STEPS = frozenset({WizardStep.PREVIEW_SEGMENT, WizardStep.READY_TO_RENDER})
# An uploaded file's name starts with 12 hex digits (see upload_target).
_UPLOAD_PREFIX = re.compile(r"[0-9a-f]{12}-")


_T = TypeVar("_T")


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


# What reading shots.json or shot_plan.json can raise when either is damaged
# or half-written.
_PLAN_READ_ERRORS = (ShotPlanError, ShotError, OSError, KeyError, TypeError, ValueError)


def _job_title(source_path: str) -> str:
    """The source's file name, without the prefix an upload gets."""
    name = Path(source_path).name
    found = _UPLOAD_PREFIX.match(name)
    return name[found.end() :] if found and name[found.end() :] else name


def _manifest_title(manifest: JobManifest) -> str:
    """The title the pick saved for a job, or its file name before that."""
    title = manifest.notes.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()
    return _job_title(manifest.source_path)


def _run_kind(step: WizardStep) -> RunKind | None:
    """The kind of run that works on ``step``, or ``None``."""
    for kind, steps in _RUN_STEPS.items():
        if step in steps:
            return kind
    return None


def _pid_alive(pid: int) -> bool:
    """Whether a process with this id runs on this computer."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists, owned by someone else.
        return True
    except OSError:
        return False
    return True


def _windows_pid_alive(pid: int) -> bool:  # pragma: no cover - Windows only
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    query_limited_information = 0x1000
    still_active = 259
    handle = kernel32.OpenProcess(query_limited_information, False, pid)
    if not handle:
        # Access denied means the process exists but is not ours to open.
        return bool(kernel32.GetLastError() == 5)
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return True
        return code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


def _process_start(pid: int) -> str | None:
    """When the process with this id started, as text to compare, or ``None``.

    A process id is reused once its process ends, so an id alone cannot
    tell the run that wrote it from a later, unrelated process. ``None``
    when the start cannot be read here.
    """
    if pid <= 0:
        return None
    if os.name == "nt":  # pragma: no cover - Windows only
        return _windows_process_start(pid)
    if sys.platform.startswith("linux"):
        return _linux_process_start(pid)
    return _ps_process_start(pid)  # pragma: no cover - macOS and BSD


def _linux_process_start(pid: int) -> str | None:
    """The boot and the clock tick since it that the process started at."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    # The command name sits in brackets and may hold spaces; the start
    # time is the 22nd field, the 20th after the bracket.
    fields = stat.rpartition(")")[2].split()
    if len(fields) < 20 or not fields[19].isdigit():
        return None
    return f"{boot.strip()}:{fields[19]}"


def _ps_process_start(pid: int) -> str | None:  # pragma: no cover - macOS/BSD
    try:
        done = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    started = " ".join(done.stdout.split())
    return started if done.returncode == 0 and started else None


def _windows_process_start(pid: int) -> str | None:  # pragma: no cover - Windows
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    query_limited_information = 0x1000
    handle = kernel32.OpenProcess(query_limited_information, False, pid)
    if not handle:
        return None
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        if not kernel32.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        return str(times[0].value)
    finally:
        kernel32.CloseHandle(handle)


def _worker_pid_text() -> str:
    """What a run writes to its job's pid file: its id and when it started."""
    pid = os.getpid()
    started = _process_start(pid)
    return f"{pid} {started}\n" if started else f"{pid}\n"


def worker_pid(job_dir: Path) -> int | None:
    """The process a run on this job was started in, while it is alive.

    ``None`` when no run works on the job, or the process that wrote the
    file has gone (a crash leaves the file behind). The file also says
    when that process started, so an unrelated process that was later
    given the same id (after a restart, say) does not count.
    """
    try:
        raw = (job_dir / WORKER_PID_FILE).read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError):
        return None
    if not raw or not raw[0].isdigit():
        return None
    pid = int(raw[0])
    if not _pid_alive(pid):
        return None
    if len(raw) > 1:
        started = _process_start(pid)
        if started is not None and started != " ".join(raw[1:]):
            return None
    return pid


def running_elsewhere(job_dir: Path) -> bool:
    """True while another process on this computer runs work on the job."""
    pid = worker_pid(job_dir)
    return pid is not None and pid != os.getpid()


def job_thumb(job_dir: Path) -> Path | None:
    """A still to show for the job in the job list, or ``None``.

    The sampled still in the middle of the video: the first is often a
    title card or black.
    """
    stills = load_segment_stills(job_dir)
    if not stills:
        return None
    return job_dir / stills[len(stills) // 2].path


def _updated_epoch(manifest: JobManifest, job_dir: Path) -> float:
    """When the job last changed: the manifest's time or its wizard state's."""
    times: list[float] = []
    with contextlib.suppress(ValueError):
        times.append(datetime.fromisoformat(manifest.updated_at).timestamp())
    with contextlib.suppress(OSError):
        times.append(job_state_path(job_dir).stat().st_mtime)
    return max(times, default=0.0)


def _nearest_still_url(
    stills: list[SegmentStill], start_s: float, end_s: float
) -> str | None:
    """The URL of the sampled still nearest the middle of a span."""
    middle = (start_s + end_s) / 2
    nearest = min(stills, key=lambda s: abs(s.t_s - middle), default=None)
    return f"/{nearest.path}" if nearest else None


@dataclass(slots=True)
class WizardState:
    """On-disk wizard cursor for one job (or pre-job session)."""

    step: WizardStep = WizardStep.INTRO
    job_id: str | None = None
    error: str | None = None
    # Segment loop: the segment being analyzed or previewed.
    segment_index: int = 0
    # While a worker runs: {"phase", "pct" (0 to 100), "message"}.
    progress: dict[str, Any] | None = None
    # Shown as the error once the render in progress succeeds.
    notice: str | None = None
    # How many segments the editor has approved so far. It never trails
    # segment_index, which can go back to reopen an earlier segment.
    approved_count: int = 0
    # Why the last run stopped when the editor stopped it, shown until the
    # next action.
    stopped: str | None = None
    # The result note the render in progress was asked for.
    render_note: str | None = None
    schema_version: int = WIZARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step": self.step.value,
            "job_id": self.job_id,
            "error": self.error,
            "segment_index": self.segment_index,
            "progress": self.progress,
            "notice": self.notice,
            "approved_count": self.approved_count,
            "stopped": self.stopped,
            "render_note": self.render_note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WizardState:
        raw_step = str(data.get("step", WizardStep.INTRO.value))
        version = data.get("schema_version")
        old = not isinstance(version, int) or version < _REUSED_SINCE
        if raw_step in _RETIRED_STEPS or (old and raw_step in _REUSED_STEPS):
            return cls(step=WizardStep.PICK_FILE, error=_RETIRED_MESSAGE)
        if raw_step in _SHOT_FLOW_STEPS:
            return cls(step=WizardStep.PICK_FILE, error=_SHOT_FLOW_MESSAGE)
        index = data.get("segment_index", 0)
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            index = 0
        # Files saved before approved_count existed approved every segment
        # before the current one.
        approved = data.get("approved_count")
        if not isinstance(approved, int) or isinstance(approved, bool) or approved < 0:
            approved = index
        progress = data.get("progress")
        return cls(
            step=WizardStep(raw_step),
            job_id=data.get("job_id"),
            error=data.get("error"),
            segment_index=index,
            progress=progress if isinstance(progress, dict) else None,
            notice=data.get("notice"),
            approved_count=max(approved, index),
            stopped=_optional_str(data.get("stopped")),
            render_note=_optional_str(data.get("render_note")),
        )


def input_limit_error(facts: MediaFacts) -> str | None:
    """Why the wizard refuses this file, or ``None`` when it fits the limits."""
    video = facts.primary_video()
    if not facts.has_video or video is None:
        return "this file has no video stream"
    if not video.width or not video.height:
        return "could not read the video resolution"
    long_edge = max(video.width, video.height)
    short_edge = min(video.width, video.height)
    if long_edge > MAX_INPUT_LONG_EDGE or short_edge > MAX_INPUT_SHORT_EDGE:
        return (
            f"video is {video.width}x{video.height}; the limit is 1080p "
            f"({MAX_INPUT_LONG_EDGE}x{MAX_INPUT_SHORT_EDGE} in either orientation)"
        )
    duration = float(facts.duration_s or 0.0)
    if duration <= 0:
        return "could not read the video duration"
    if duration > MAX_DURATION_S:
        minutes = duration / 60.0
        return f"video is {minutes:.1f} minutes long; the limit is 30 minutes"
    return None


def session_state_path(jobs_dir: Path) -> Path:
    """Wizard state before a job exists lives under the jobs root."""
    return jobs_dir / "wizard_session.json"


def job_state_path(job_dir: Path) -> Path:
    return job_dir / WIZARD_STATE_FILE


def _read_state_file(path: Path, what: str) -> WizardState | None:
    # Retries a Windows sharing error from a save racing this read; see
    # atomic_io.
    try:
        if not path.is_file():
            return None
        text = read_text_retrying(path)
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise WizardError(f"corrupt {what}") from exc
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise WizardError(f"corrupt {what}") from exc
    if not isinstance(data, dict):
        raise WizardError(f"corrupt {what}")
    try:
        return WizardState.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        # An unknown step, say from a newer build's state file.
        raise WizardError(f"corrupt {what}") from exc


def _write_state_file(path: Path, state: WizardState) -> None:
    write_text_atomic(path, json.dumps(state.to_dict(), indent=2) + "\n")


def load_session_state(jobs_dir: Path) -> WizardState:
    state = _read_state_file(session_state_path(jobs_dir), "wizard session")
    return state if state is not None else WizardState()


def save_session_state(jobs_dir: Path, state: WizardState) -> None:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    _write_state_file(session_state_path(jobs_dir), state)


def load_job_wizard_state(job_dir: Path) -> WizardState | None:
    return _read_state_file(job_state_path(job_dir), "wizard_state.json")


def save_job_wizard_state(job_dir: Path, state: WizardState) -> None:
    _write_state_file(job_state_path(job_dir), state)


_SEGMENT_EDITS = ("move", "merge", "split")
_REVISION = re.compile(r"\d{1,9}", re.ASCII)
_SCENE_INDEX = re.compile(r"\d{1,3}", re.ASCII)
_STALE_SEGMENTS = (
    "the scenes changed since this page showed them; look at them again "
    "before changing them"
)


def _check_revision(
    found: SegmentList, fields: dict[str, list[str]], *, required: bool
) -> None:
    """Refuse a POST made from a page that shows an older segment list."""
    raw = (fields.get("revision") or [""])[0].strip()
    if not raw and not required:
        return
    if not _REVISION.fullmatch(raw) or int(raw) != found.revision:
        raise WizardError(_STALE_SEGMENTS)


def _edit_segments(
    segments: list[LoopSegment], action: str, fields: dict[str, list[str]]
) -> list[LoopSegment]:
    """The segments after one change by hand.

    A change the rules refuse raises SegmentLoopError, shown on the page; a
    scene index the page could not have sent raises WizardError.
    """
    raw = fields.get("index", [""])[0].strip()
    if not _SCENE_INDEX.fullmatch(raw):
        raise WizardError("pick one of the scenes shown")
    index = int(raw)
    if action == "merge":
        return merge_segments(segments, index)
    try:
        at = float(fields.get("at", [""])[0].strip())
    except ValueError:
        raise SegmentLoopError("say where to cut, in seconds") from None
    if action == "move":
        return move_cut(segments, index, at)
    return split_segment(segments, index, at)


def _spans(items: Sequence[Shot | LoopSegment]) -> list[tuple[float, float]]:
    """Start and end of each shot or segment, to the millisecond saved."""
    return [(round(item.start_s, 3), round(item.end_s, 3)) for item in items]


def _fallback_plan(plan: ShotPlan, shots: ShotList, indexes: list[int]) -> ShotPlan:
    """Recipes from the measurements, keeping each shot's preview count."""
    recipes = list(plan.recipes)
    for index in indexes:
        recipes[index] = keep_editor_choices(
            fallback_recipe(shots.shots[index]), recipes[index]
        )
    return replace(plan, recipes=recipes)


RENDER_REPORT_FILE = "render_report.json"
# A full restart clears these from the job folder. ``gemini_chat.json``
# stays, so Gemini sees what the editor did not like the first time.
_RESTART_FILES = (
    "segments.json",
    "shots.json",
    "shot_plan.json",
    "upscale_plan.json",
    "output.mp4",
    "output.mp4.delivery.json",
    RENDER_REPORT_FILE,
)
_RESTART_DIRS = ("shot_stills", "previews", "shot_work", "range_work")


def restart_from_segmentation(job_dir: Path) -> None:
    """Clear the segments, plans, previews, and output of a job; keep the chat.

    The still grid is sampled again on the way back through ``sampling``.
    """
    for name in _RESTART_FILES:
        (job_dir / name).unlink(missing_ok=True)
    for name in _RESTART_DIRS:
        folder = job_dir / name
        if folder.is_symlink():
            folder.unlink()
        elif folder.is_dir():
            shutil.rmtree(folder)


def load_render_report(job_dir: Path) -> dict[str, Any]:
    """The seam check of the latest render: ``seams`` and ``error``."""
    empty: dict[str, Any] = {"seams": [], "error": None}
    path = job_dir / RENDER_REPORT_FILE
    if not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(data, dict):
        return empty
    seams = [
        seam
        for seam in data.get("seams") or []
        if isinstance(seam, dict) and isinstance(seam.get("time_s"), (int, float))
    ]
    error = data.get("error")
    return {"seams": seams, "error": str(error) if error else None}


# The plan a result note replaced, until its render shows the output holds it.
_PLAN_BEFORE_NOTE = "upscale_plan_before_note.json"
_SOURCE_SIZE_NOTE = (
    "this render keeps the source size with upscaling off, so a note cannot "
    'change the picture. Say "make it 2x" to upscale it, or start over.'
)
# 1 (do not upscale) and the upscale multiples. Every source the wizard takes
# may use them all, so the size of the source no longer matters here.
SCALE_OPTIONS = (1, *UPSCALE_SCALES)


def _output_url(job_dir: Path) -> str | None:
    """The result's URL, or ``None`` before there is one.

    A revise replaces ``output.mp4``, and a browser's media layer can reuse what
    it buffered for an unchanged URL even with ``no-store``, so the URL changes
    with the file. The URL names the job, so a page left open on one job
    never plays another job's result after the wizard moves on.
    """
    try:
        version = (job_dir / "output.mp4").stat().st_mtime_ns
    except OSError:
        return None
    return f"/jobs/{job_dir.name}/output.mp4?v={version}"


def _active_knobs(
    job_dir: Path, source: Path
) -> tuple[int | None, float | None, str | None]:
    """Scale, the latest span's strength (the JSON API's ``strength``), and a
    summary of every span for the Result line."""
    plan = load_upscale_plan(job_dir)
    if plan is None or not plan.spans:
        return None, None, None
    span = plan.pending or plan.spans[-1]
    # The scale output.mp4 has: a job rendered at 4x before 4x was retired
    # loads its plan at 2x, but the file is still 4x until it renders again.
    output = job_dir / "output.mp4"
    shown = rendered_scale(source, output) if output.is_file() else None
    # The summary reads the spans alone: they are what the output holds, and a
    # pending span never reached it.
    return shown or span.scale, span.strength, strength_summary(plan.spans)


def strength_summary(spans: tuple[UpscaleSpan, ...]) -> str:
    """Describe the strength of a whole render for the Result line.

    Each segment has its own recipe, so one span's strength misdescribes the
    rest. Spans with ``use_upscale`` False skip SPAN, so their strength
    does not apply to the picture, and neither does it at scale 1.
    """
    if all(span.scale == 1 for span in spans):
        return "upscaling off"
    restored = sorted({round(span.strength, 2) for span in spans if span.use_upscale})
    if not restored:
        return "upscaling off"
    low, high = restored[0], restored[-1]
    summary = f"strength {low:.2f}" if low == high else f"strength {low:.2f}–{high:.2f}"
    # Seconds, not a count: a range revision splits spans, so spans are not
    # segments.
    skipped = sum(span.end_s - span.start_s for span in spans if not span.use_upscale)
    if skipped:
        summary += f" · upscaling off for {skipped:.1f} s"
    return summary


def _aged(progress: dict[str, Any] | None, now: float) -> dict[str, Any] | None:
    """``progress`` as of ``now``: time left counted down, time spent counted up.

    The worker saves it every few seconds; between saves the page should
    still see the clock move.
    """
    if progress is None:
        return None
    updated = progress.get("updated_at")
    if not isinstance(updated, (int, float)) or isinstance(updated, bool):
        return progress
    age = max(0.0, now - float(updated))
    aged = dict(progress)
    eta = progress.get("eta_s")
    if isinstance(eta, (int, float)) and not isinstance(eta, bool):
        aged["eta_s"] = round(max(0.0, float(eta) - age), 1)
    elapsed = progress.get("elapsed_s")
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
        aged["elapsed_s"] = round(float(elapsed) + age, 1)
    return aged


@dataclass(slots=True)
class _Run:
    """One foreground run (segmenting, analyze or render) and how it ended.

    ``api_view`` shows it as ``run``, so a page can tell a run that finished
    from one that failed or was stopped, even after the step has moved on.
    """

    kind: RunKind
    job_id: str | None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    outcome: RunOutcome = "running"
    cancel: threading.Event = field(default_factory=threading.Event)
    # Until the run commits its result, a stop can still undo it.
    cancellable: bool = True
    committed: bool = False
    # True once a render's ``render_plan`` returned: ``output.mp4`` now holds
    # the new plan, so a later failure (the probe, the seam check, a save)
    # must keep that plan rather than restore the old one.
    rendered: bool = False
    # True once a stop has been landed (the state saved as stopped).
    landed: bool = False
    # The worker thread that owns the run; only it may mark the run failed.
    thread_id: int = field(default_factory=threading.get_ident)
    # The controller's cancel guard, so a stop and a commit never interleave.
    guard: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def check(self) -> None:
        """Raise :class:`Cancelled` once the editor has asked to stop."""
        if self.cancel.is_set():
            raise Cancelled

    def commit(self) -> None:
        """Make the run's result final: from here on it cannot be stopped.

        Raises :class:`Cancelled` instead when a stop came first.
        """
        with self.guard:
            if self.cancel.is_set():
                raise Cancelled
            self.cancellable = False
            self.committed = True

    def view(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "job_id": self.job_id,
            "started_at": round(self.started_at, 3),
            "finished_at": (
                None if self.finished_at is None else round(self.finished_at, 3)
            ),
            "outcome": self.outcome,
        }


class WizardController:
    """Drive wizard actions against a jobs directory."""

    def __init__(
        self,
        jobs_dir: Path,
        *,
        job_id: str | None = None,
        gemini: GeminiClient | None = None,
        ffmpeg_bin: str = "ffmpeg",
        upscale_engine: UpscaleEngine | None = None,
    ) -> None:
        self.store = JobStore(jobs_dir)
        self.jobs_dir = self.store.root
        # An explicitly reopened job must not follow another window's session.
        self._session_state: WizardState | None = None
        if job_id is not None:
            self._session_state = load_job_wizard_state(self.store.job_dir(job_id))
            if self._session_state is None:
                raise WizardError(f"no wizard state for job {job_id}")
        self.ffmpeg_bin = ffmpeg_bin
        self._gemini = gemini
        self._upscale = upscale_engine
        # One analyze or render at a time. The HTTP server is threaded, and a
        # reload or a second tab must not start a second GPU pass on the
        # same job directory.
        self._work_lock = threading.Lock()
        self._worker_guard = threading.Lock()
        self._worker: threading.Thread | None = None
        # Held for the whole of a POST, and while a worker starts, so a worker
        # never starts in the middle of a POST's read-modify-save.
        self._post_lock = threading.RLock()
        # A run's progress save and the landing of its failure (or stop) go
        # one at a time, so a late progress save never lands after them.
        self._state_lock = threading.RLock()
        # Guards the run record: setting it, reading it, stopping it.
        self._cancel_guard = threading.Lock()
        self._run: _Run | None = None
        self._last_run: _Run | None = None
        # On a helper thread a run hands a Gemini call to (see _abandonable):
        # the run it works for. Its progress saves check that run, not
        # whichever run is current, so a helper left behind by a stop can
        # never save over the stopped screen or over a newer run.
        self._bound = threading.local()
        # The job and kind of run a POST in this process last moved the
        # wizard to (or the editor resumed). A run step not armed here was
        # left by an earlier server, or reopened from the job list, and is
        # "interrupted": its work starts only once the editor resumes it.
        # Kept in memory on purpose, so a restart never resumes by itself.
        self._armed: tuple[str, RunKind] | None = None
        # The thread applying a POST, whose saves arm the step they move to.
        self._posting: int | None = None
        # The running worker's time-left estimate, per measurable task.
        self._clock: StepClock | None = None
        self._clock_task = ""
        self._step_started = 0.0
        self._revising = False
        # Shown under the render's progress once the render slows down while
        # the GPU runs hot; tests replace the reader.
        self._heat_note: str | None = None
        self.read_gpu_heat: Callable[[], GpuHeat | None] = read_gpu_heat

    def gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = GeminiClient()
        return self._gemini

    def upscale_engine(self) -> UpscaleEngine:
        if self._upscale is None:
            self._upscale = SpanEngine()
        return self._upscale

    def current_state(self) -> WizardState:
        """Return the active wizard state (job-scoped when a job exists)."""
        session = self._session_state or load_session_state(self.jobs_dir)
        if session.job_id:
            job_dir = self.store.job_dir(session.job_id)
            job_state = load_job_wizard_state(job_dir)
            if job_state is not None:
                return job_state
        return session

    def _state(self) -> WizardState:
        return self.current_state()

    def busy(self) -> bool:
        """True while a worker runs: from its start until its thread ends.

        This covers the moment before a new worker takes the work lock and
        the progress clean-up after it lets go, so no POST lands in between.
        """
        worker = self._worker
        return self._work_lock.locked() or (worker is not None and worker.is_alive())

    def start_render(self) -> bool:
        """Run :meth:`run_render` on a worker thread. False if not started."""
        return self._start_worker(WizardStep.RENDERING, self.run_render)

    def start_segmenting(self) -> bool:
        """Run :meth:`run_segmenting` on a worker thread. False if not started."""
        return self._start_worker(_SEGMENTING_STEPS, self.run_segmenting)

    def start_analyze(self) -> bool:
        """Run :meth:`run_analyze` on a worker thread. False if not started."""
        return self._start_worker(WizardStep.ANALYZING, self.run_analyze)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the worker finishes. True when nothing is running."""
        with self._worker_guard:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
        return not self.busy()

    def _start_worker(
        self, step: WizardStep | frozenset[WizardStep], target: Callable[[], None]
    ) -> bool:
        steps = step if isinstance(step, frozenset) else frozenset({step})
        if not self._post_lock.acquire(timeout=_POST_WAIT_S):
            return False
        try:
            with self._worker_guard:
                if self.busy():
                    return False
                state = self._state()
                current = state.step
                if current not in steps or self.interrupted(state):
                    return False
                if state.job_id and running_elsewhere(self.store.job_dir(state.job_id)):
                    return False
                worker = threading.Thread(
                    target=self._run_worker,
                    args=(target,),
                    name=f"wizard-{current.value}",
                    daemon=True,
                )
                self._worker = worker
                worker.start()
                return True
        finally:
            self._post_lock.release()

    def _begin_step(self) -> None:
        """Start the clocks for a worker step (called under the work lock)."""
        self._clock = None
        self._clock_task = ""
        self._step_started = time.time()
        self._revising = False

    def _run_worker(self, target: Callable[[], None]) -> None:
        # A run lands its own failure under the work lock (see _running).
        # This only catches a crash outside any run, such as an unreadable
        # state file; busy() stays true until this thread ends, so no POST
        # lands in between.
        try:
            target()
            self._clear_progress()
        except Exception as exc:  # noqa: BLE001 - surfaced on the page
            self._land_failure(None, exc)

    @contextlib.contextmanager
    def _running(self, kind: RunKind) -> Iterator[_Run | None]:
        """Hold the work lock for one run of ``kind`` and record how it ends.

        Yields ``None`` without doing anything when another run holds the
        lock or the wizard is not on one of the run's steps; no run is
        recorded then. Otherwise yields the new run. An exception from the
        body is landed under the lock (the step's failure screen is saved),
        and the run's outcome is set before the lock is released, so a page
        that sees ``busy`` false also sees how the run ended.
        """
        if not self._work_lock.acquire(blocking=False):
            yield None
            return
        run: _Run | None = None
        try:
            state = self._state()
            if state.job_id and state.step in _RUN_STEPS[kind]:
                run = _Run(kind=kind, job_id=state.job_id, guard=self._cancel_guard)
            with self._cancel_guard:
                self._run = run
            if run is None:
                yield None
                return
            self._begin_step()
            # Best effort: without it another server only misses that the
            # job is busy, as it did before the file existed.
            with contextlib.suppress(OSError):
                write_text_atomic(
                    self.store.job_dir(run.job_id or "") / WORKER_PID_FILE,
                    _worker_pid_text(),
                )
            try:
                if self._cancel_marker(run.job_id).is_file():
                    # A stop was asked for before this run began: while its
                    # worker was starting, or before the server restarted.
                    self._land_cancel(run)
                    yield None
                    return
                yield run
            except Cancelled:
                self._land_cancel(run)
            except Exception as exc:  # noqa: BLE001 - surfaced on the page
                self._land_failure(run, exc)
        finally:
            if run is not None:
                run.finished_at = time.time()
                if run.outcome == "running":
                    run.outcome = "done"
            with self._cancel_guard:
                if run is not None:
                    # Under the guard, so a stop asked for from here on finds
                    # no run and reads the step this run left behind.
                    self._drop_cancel_marker(run.job_id)
                    if run.job_id:
                        unlink_quietly(self.store.job_dir(run.job_id) / WORKER_PID_FILE)
                    self._last_run = replace(run)
                self._run = None
            self._work_lock.release()

    def _cancel_marker(self, job_id: str | None) -> Path:
        assert job_id
        return self.store.job_dir(job_id) / CANCEL_MARKER

    def _drop_cancel_marker(self, job_id: str | None) -> None:
        if job_id:
            unlink_quietly(self._cancel_marker(job_id))

    def _enter_running_step(self, state: WizardState, step: WizardStep) -> None:
        """Move to a step a run works on, dropping a stop left from before.

        Every POST that starts segmenting or a render goes through here, so
        a stop asked for on an earlier run never stops the new one.
        """
        self._drop_cancel_marker(state.job_id)
        state.step = step

    def cancel(self, *, job_id: str = "", run_id: str = "") -> None:
        """Stop the scan or render in progress, or one left interrupted.

        ``job_id`` and ``run_id`` are what the page showed; when the wizard
        has moved on to another job, or the run the page showed has ended,
        :class:`WizardConflict` refuses the stop, so a late click never
        stops something the editor did not see. A run stops at its next
        check (within moments; the GPU can take up to a minute) and lands
        on the step before it. A render that has started replacing its
        output, or segments being saved, can no longer be stopped.
        """
        with self._post_lock:
            state = self._state()
            if job_id and job_id != state.job_id:
                raise WizardConflict(_JOB_CONFLICT_MESSAGE)
            if state.step is WizardStep.ANALYZING and self.interrupted(state):
                # A scene's analysis cannot be stopped while it runs, but one
                # left interrupted can be dropped instead of resumed.
                self._stop_interrupted_analyze(state)
                return
            if state.step not in _CANCEL_STEPS or not state.job_id:
                raise WizardError("there is nothing running to stop")
            marker = self._cancel_marker(state.job_id)
            with self._cancel_guard:
                run = self._run
                if run_id and (run is None or run.id != run_id):
                    raise WizardConflict("that run has already ended; reload the page")
                if run is not None:
                    if not run.cancellable:
                        what = "render" if run.kind == "render" else "segmenting"
                        raise WizardError(
                            f"the {what} is finishing and can no longer be stopped"
                        )
                    # The run itself stops on the event; the marker only
                    # covers a server that goes away before it lands.
                    with contextlib.suppress(OSError):
                        marker.touch()
                    run.cancel.set()
                    return
                idle = not self.busy() and self._work_lock.acquire(blocking=False)
                if not idle:
                    # A worker is about to begin its run, and checks the
                    # marker first thing; or one has just finished, and the
                    # step it saved says so.
                    if self._state().step not in _CANCEL_STEPS:
                        raise WizardError("it has already finished")
                    try:
                        marker.touch()
                    except OSError as exc:
                        raise WizardError(f"could not ask it to stop: {exc}") from exc
                    return
            # Nothing runs (the server restarted mid-step, say): land the stop
            # here, holding the work lock so no run starts meanwhile.
            try:
                state = self._state()
                if state.step not in _CANCEL_STEPS or not state.job_id:
                    raise WizardError("there is nothing running to stop")
                if running_elsewhere(self.store.job_dir(state.job_id)):
                    raise WizardError(_RUNNING_ELSEWHERE)
                kind: RunKind = (
                    "render" if state.step is WizardStep.RENDERING else "segmenting"
                )
                stopped = _Run(kind=kind, job_id=state.job_id, guard=self._cancel_guard)
                self._land_cancel(stopped)
                stopped.finished_at = time.time()
                with self._cancel_guard:
                    self._drop_cancel_marker(stopped.job_id)
                    self._last_run = replace(stopped)
            finally:
                self._work_lock.release()

    def interrupted(self, state: WizardState, busy: bool | None = None) -> bool:
        """True when ``state`` is on a run step no POST here moved it to.

        That is a job left mid-step by an earlier server, or one reopened
        from the job list. Its work does not start by itself: ``?run=1``
        and ``start_*`` refuse it until :meth:`resume` (or a stop).
        """
        if busy is None:
            busy = self.busy()
        kind = _run_kind(state.step)
        if busy or kind is None or not state.job_id:
            return False
        return self._armed != (state.job_id, kind)

    def _arm(self, state: WizardState) -> None:
        """Remember the run step a POST here moved the wizard to."""
        kind = _run_kind(state.step)
        job_id = state.job_id
        self._armed = (job_id, kind) if kind is not None and job_id else None

    def resume(self, *, job_id: str = "") -> None:
        """Let an interrupted step's work start again (see :meth:`interrupted`).

        Only arms the step: the page then starts the work with ``?run=1``
        as for any run. A stop asked for before the server went away is
        dropped, since the editor has now chosen to carry on.
        """
        with self._post_lock:
            if self.busy():
                raise WizardError("the last step is still running; wait for it")
            state = self._state()
            if job_id and job_id != state.job_id:
                raise WizardConflict(_JOB_CONFLICT_MESSAGE)
            if not state.job_id or _run_kind(state.step) is None:
                raise WizardError("there is nothing to resume")
            if running_elsewhere(self.store.job_dir(state.job_id)):
                raise WizardError(_RUNNING_ELSEWHERE)
            self._drop_cancel_marker(state.job_id)
            self._arm(state)
            if state.error is not None or state.stopped is not None:
                state.error = None
                state.stopped = None
                self._save(state)

    def _stop_interrupted_analyze(self, state: WizardState) -> None:
        """Drop an interrupted scene analysis: back to the approved segments.

        Nothing is deleted: the segments stay approved, so approving them
        again keeps the plan and carries on from the scene the editor was on.
        """
        if not self._work_lock.acquire(blocking=False):
            raise WizardError("the last step is still running; wait for it")
        try:
            assert state.job_id
            if running_elsewhere(self.store.job_dir(state.job_id)):
                raise WizardError(_RUNNING_ELSEWHERE)
            stopped = _Run(
                kind="analyze", job_id=state.job_id, guard=self._cancel_guard
            )
            stopped.outcome = "stopped"
            stopped.landed = True
            state.error = None
            state.progress = None
            state.stopped = _STOPPED_ANALYZE
            state.step = WizardStep.SEGMENTS
            self._save(state)
            stopped.finished_at = time.time()
            with self._cancel_guard:
                self._last_run = replace(stopped)
        finally:
            self._work_lock.release()

    def open_job(self, job_id: str) -> None:
        """Show another job in this window, where it was left.

        A job left on a run step opens interrupted: its work waits for the
        editor to resume or stop it. Refused while a run works here, and
        while another server on the same folder works on that job.
        """
        with self._post_lock:
            if self.busy():
                raise WizardError("the last step is still running; wait for it")
            try:
                job_dir = self.store.job_dir(job_id)
            except JobError as exc:
                raise WizardError(str(exc)) from exc
            if running_elsewhere(job_dir):
                raise WizardError(_RUNNING_ELSEWHERE)
            state = load_job_wizard_state(job_dir)
            if state is None or not job_dir.is_dir():
                raise WizardError(f"no wizard state for job {job_id}")
            state.job_id = job_id
            # A run that died here left its last progress behind; and the
            # note a render was asked for belongs to that render alone.
            state.progress = None
            state.render_note = None
            self._armed = None
            self._save(state, arm=False)

    def jobs_view(self) -> list[dict[str, Any]]:
        """Every wizard job under the jobs folder, most recently changed first.

        Read-only: it creates nothing, and a job that cannot be read is
        listed with its ``error`` rather than breaking the list. Jobs the
        command line made without the wizard are left out.

        ``status`` is ``running`` (a run works on it here),
        ``running_elsewhere`` (another server on this folder works on it),
        ``result`` (it has a rendered video), ``failed`` or ``paused``.
        ``resumes_work`` says it was left on a run step, and ``interrupted``
        that opening it waits for the editor to resume or stop that work.
        """
        root = self.store.root
        if not root.is_dir():
            return []
        current = self._state()
        busy = self.busy()
        with self._cancel_guard:
            working = self._run.job_id if self._run is not None else None
        if busy and working is None:
            working = current.job_id
        entries: list[tuple[float, dict[str, Any]]] = []
        for child in root.iterdir():
            try:
                job_dir = self.store.job_dir(child.name)
            except JobError:
                continue
            if not job_dir.is_dir() or not job_state_path(job_dir).is_file():
                continue
            entries.append(self._job_entry(child.name, job_dir, current, working))
        entries.sort(key=lambda item: item[0], reverse=True)
        return [entry for _updated, entry in entries]

    def _job_entry(
        self,
        job_id: str,
        job_dir: Path,
        current: WizardState,
        working: str | None,
    ) -> tuple[float, dict[str, Any]]:
        """One row of :meth:`jobs_view`, with the time it is sorted by."""
        entry: dict[str, Any] = {
            "job_id": job_id,
            "title": job_id,
            "file_name": None,
            "step": None,
            "stage": None,
            "status": "paused",
            "updated_at": None,
            "resumes_work": False,
            "interrupted": False,
            "open": current.job_id == job_id,
            "thumb_url": None,
            "output_url": None,
            "error": None,
        }
        try:
            manifest = self.store.load(job_id)
            state = (
                current if current.job_id == job_id else load_job_wizard_state(job_dir)
            )
        except (WizardError, JobError, OSError, ValueError, KeyError, TypeError) as exc:
            entry["error"] = str(exc) or type(exc).__name__
            with contextlib.suppress(OSError):
                updated = job_state_path(job_dir).stat().st_mtime
                entry["updated_at"] = round(updated, 3)
                return updated, entry
            return 0.0, entry
        updated = _updated_epoch(manifest, job_dir)
        entry["title"] = _manifest_title(manifest)
        entry["file_name"] = _job_title(manifest.source_path)
        entry["updated_at"] = round(updated, 3)
        with contextlib.suppress(OSError, SegmentLoopError):
            if job_thumb(job_dir) is not None:
                entry["thumb_url"] = f"/jobs/{job_id}/thumb.jpg"
        entry["output_url"] = _output_url(job_dir)
        has_versions = False
        with contextlib.suppress(OSError):
            has_versions = any((job_dir / "outputs").glob("v*.mp4"))
        if state is None:
            return updated, entry
        entry["step"] = state.step.value
        entry["stage"] = _STEP_STAGES.get(state.step)
        entry["error"] = state.error
        resumes = _run_kind(state.step) is not None
        entry["resumes_work"] = resumes
        elsewhere = running_elsewhere(job_dir)
        if working == job_id:
            entry["status"] = "running"
        elif elsewhere:
            entry["status"] = "running_elsewhere"
        elif entry["output_url"] is not None or has_versions:
            entry["status"] = "result"
        elif manifest.state is JobState.FAILED:
            entry["status"] = "failed"
        if working != job_id and not elsewhere and resumes:
            entry["interrupted"] = (
                self.interrupted(state, busy=False)
                if current.job_id == job_id
                else True
            )
        return updated, entry

    def _land_cancel(self, run: _Run) -> None:
        """Save the screen a stopped run goes back to.

        Runs under the work lock, and under the state lock so a progress
        save from a helper the run left behind lands before it or not at
        all (see :meth:`_progress`).
        """
        with self._state_lock:
            run.outcome = "stopped"
            run.landed = True
            try:
                state = self._state()
                state.error = None
                state.progress = None
                if state.step is WizardStep.RENDERING:
                    self._render_cancelled(state)
                elif state.step in _SEGMENTING_STEPS:
                    self._segmenting_cancelled(state)
                else:
                    self._save(state)
            except Exception as exc:  # noqa: BLE001 - last resort
                message = f"Stopped, but saving that failed: {exc}"
                with contextlib.suppress(Exception):
                    self._save(WizardState(step=WizardStep.PICK_FILE, error=message))
            finally:
                self._drop_cancel_marker(run.job_id)

    def _check_stop(self) -> None:
        """Raise :class:`Cancelled` when the current run was asked to stop."""
        run = self._run
        if run is not None:
            run.check()

    def _abandonable(self, call: Callable[[], _T]) -> _T:
        """Run a Gemini call so a stop does not wait for it to answer.

        The call runs on a helper thread while this one looks for a stop
        every moment; on a stop the helper is left to finish on its own and
        its answer is dropped. Its progress saves check this run first (see
        :meth:`_progress`), so it never saves after the stop has landed.
        Without a run (a direct call in a test), the call simply runs here.
        """
        run = self._run
        if run is None:
            return call()
        answer: list[_T] = []
        failure: list[BaseException] = []
        done = threading.Event()

        def work() -> None:
            self._bound.run = run
            try:
                answer.append(call())
            except BaseException as exc:  # noqa: BLE001 - raised below
                failure.append(exc)
            finally:
                done.set()

        threading.Thread(target=work, name="wizard-gemini", daemon=True).start()
        while not done.wait(_ABANDON_POLL_S):
            run.check()
        run.check()
        if failure:
            raise failure[0]
        return answer[0]

    def _cancel_view(self, state: WizardState, busy: bool) -> tuple[bool, bool]:
        """Whether a stop can be asked for now, and whether one is pending."""
        with self._cancel_guard:
            run = self._run
            live = run is not None and run.cancellable and not run.cancel.is_set()
            asked = run is not None and run.cancel.is_set()
        on_step = state.step in _CANCEL_STEPS and bool(state.job_id)
        if not on_step and state.step is WizardStep.ANALYZING:
            # Only an interrupted analysis can be dropped (see cancel).
            on_step = self.interrupted(state, busy)
        return on_step and (not busy or live), busy and asked

    def _mark_failed(self, run: _Run | None) -> None:
        """Record that ``run`` failed. Only the run's own thread can do this,
        so a late call from elsewhere never marks a newer run."""
        if run is not None and run.thread_id == threading.get_ident():
            run.outcome = "failed"

    def _land_failure(self, run: _Run | None, exc: Exception) -> None:
        """Save the failure screen for the step a run crashed on."""
        detail = str(exc) or type(exc).__name__
        with self._state_lock:
            self._mark_failed(run)
            try:
                state = self._state()
                state.error = detail
                if state.step is WizardStep.RENDERING:
                    self._render_failed(state, detail)
                elif state.step in _SEGMENTING_STEPS:
                    self._segmenting_failed(state, detail)
                elif state.step is WizardStep.ANALYZING:
                    self._analyze_failed(state, detail)
                else:
                    self._save(state)
            except Exception as again:  # noqa: BLE001 - last resort
                # The failure screen itself failed (a bad job folder, a full
                # disk); go back to the pick screen with both errors rather
                # than leave the job looking as if it still runs.
                message = f"{detail} (and saving that failed: {again})"
                with contextlib.suppress(Exception):
                    self._save(WizardState(step=WizardStep.PICK_FILE, error=message))

    def _public_run(self, state: WizardState) -> dict[str, Any] | None:
        """The run in progress, else the last one, for ``api_view``."""
        with self._cancel_guard:
            run = self._run or self._last_run
            view = None if run is None else run.view()
        if view is None:
            return None
        # The run has finished its step (the result screen is saved) and is
        # only tidying up; the page can treat it as done.
        if view["outcome"] == "running" and state.step not in _RUN_STEPS[view["kind"]]:
            view["outcome"] = "done"
        return view

    def _progress(
        self,
        state: WizardState,
        phase: str,
        pct: float,
        message: str,
        *,
        task: str = "",
        done: float | None = None,
    ) -> None:
        """Record how far the running worker is, for ``/api/state``.

        ``pct`` is the worker's own step, 0 to 100. ``task`` names a stretch
        of work that reports ``done`` (0 to 1) as it goes, such as saving
        stills, an upload, or the render; its time left is estimated from
        how fast ``done`` moves. Without ``task`` (a Gemini call, a check)
        there is nothing to measure, and the page shows the time spent.
        """
        now = time.time()
        eta: float | None = None
        if task:
            if self._clock is None or self._clock_task != task:
                self._clock = StepClock(started_at=now)
                self._clock_task = task
            eta = self._clock.update(0.0 if done is None else done, now)
        else:
            self._clock = None
            self._clock_task = ""
        state.progress = {
            "phase": phase,
            "pct": max(0, min(100, int(pct))),
            "message": message,
            "overall": overall_pct(
                phase,
                pct,
                # A reopened scene does not move the bar back: it counts
                # from the furthest scene approved so far.
                segment_index=max(state.segment_index, state.approved_count),
                segment_count=self._segment_count(state),
                revising=self._revising,
            ),
            "eta_s": None if eta is None else round(eta, 1),
            "elapsed_s": round(max(0.0, now - self._step_started), 1),
            "updated_at": round(now, 3),
        }
        if self._heat_note:
            state.progress["note"] = self._heat_note
        with self._state_lock:
            # A helper thread checks the run it works for, which may have
            # been stopped and replaced since (see _abandonable).
            run = getattr(self._bound, "run", None) or self._run
            if run is not None:
                run.check()
            self._save(state)

    def _segment_count(self, state: WizardState) -> int:
        if not state.job_id:
            return 1
        try:
            found = load_segment_list(self.store.job_dir(state.job_id))
        except (SegmentLoopError, OSError):
            return 1
        return max(1, len(found.segments)) if found else 1

    def _task_progress(
        self,
        state: WizardState,
        phase: str,
        low: float,
        high: float,
        task: str,
        describe: Callable[[float], str],
    ) -> Callable[[float], None]:
        """Report a task's 0-1 as ``low`` to ``high`` percent of the step.

        Saved when the whole percentage changes or every few seconds, so a
        long task neither rewrites the state per chunk nor lets its time
        left go stale.
        """
        shown = [-1, 0.0]

        def report(done: float) -> None:
            done = min(max(done, 0.0), 1.0)
            pct = low + (high - low) * done
            now = time.time()
            if round(pct) == shown[0] and now - shown[1] < _PROGRESS_SAVE_S:
                if self._clock is not None and self._clock_task == task:
                    self._clock.update(done, now)
                return
            shown[0], shown[1] = round(pct), now
            self._progress(state, phase, pct, describe(done), task=task, done=done)

        return report

    def _render_progress(self, state: WizardState) -> Callable[[float], None]:
        """Show the share of the render done as 5% to 90% of the render step."""
        return self._task_progress(
            state,
            "rendering",
            5,
            90,
            "render",
            lambda done: f"Upscaling on the GPU: {round(done * 100)}% of the work",
        )

    def _clear_progress(self) -> None:
        state = self._state()
        if state.progress is not None:
            state.progress = None
            self._save(state)

    def _save(self, state: WizardState, *, arm: bool = True) -> None:
        # A POST that moves the wizard to a run step arms it, before the
        # state is on disk, so no reader ever sees it as interrupted.
        if arm and self._posting == threading.get_ident():
            self._arm(state)
        save_session_state(self.jobs_dir, state)
        if state.job_id:
            save_job_wizard_state(self.store.job_dir(state.job_id), state)
        if self._session_state is not None:
            self._session_state = state

    def render(self) -> str:
        page = with_notice(self._render_page(), self._state().stopped)
        return with_navigation(page, self.navigation())

    def navigation(self) -> dict[str, bool]:
        """Which of setup, file, and segments the editor can go to now."""
        state = self._state()
        if self.busy() or state.step in {
            WizardStep.INTRO,
            WizardStep.LLM_CHOICE,
            WizardStep.LOCAL_LLM_STUB,
        }:
            return {"setup": False, "file": False, "segments": False}
        segments = False
        if state.job_id:
            with contextlib.suppress(SegmentLoopError):
                found = load_segment_list(self.store.job_dir(state.job_id))
                segments = found is not None and bool(found.segments)
        return {
            "setup": state.step is not WizardStep.SETUP,
            "file": state.step is not WizardStep.PICK_FILE
            and resolve_gemini_api_key() is not None,
            "segments": segments and state.step is not WizardStep.SEGMENTS,
        }

    def _render_page(self) -> str:
        state = self._state()
        busy = self.busy()
        if self.interrupted(state, busy):
            assert state.job_id
            can_stop, _stopping = self._cancel_view(state, busy)
            return render_interrupted(
                state.step.value,
                can_stop=can_stop,
                elsewhere=running_elsewhere(self.store.job_dir(state.job_id)),
            )
        if state.step is WizardStep.INTRO:
            return render_intro()
        if state.step is WizardStep.LLM_CHOICE:
            return render_llm_choice(error=state.error)
        if state.step is WizardStep.LOCAL_LLM_STUB:
            return render_local_llm_stub()
        if state.step is WizardStep.SETUP:
            return render_setup(
                has_key=bool(resolve_gemini_api_key()),
                error=state.error,
                key_source=gemini_api_key_source(),
            )
        if state.step is WizardStep.PICK_FILE:
            return render_pick(error=state.error)
        if state.step in _SEGMENTING_STEPS:
            can_stop, stopping = self._cancel_view(state, self.busy())
            return render_sampling(state.progress, can_stop=can_stop, stopping=stopping)
        if state.step is WizardStep.SEGMENTS:
            assert state.job_id
            return render_segments(
                self._segments_view(self.store.job_dir(state.job_id)),
                error=state.error,
            )
        if state.step is WizardStep.ANALYZING:
            assert state.job_id
            found = self._segments_view(self.store.job_dir(state.job_id))
            count = len(found["segments"])
            return render_analyzing(state.segment_index, count, state.progress)
        if state.step is WizardStep.PREVIEW_SEGMENT:
            assert state.job_id
            return render_preview_segment(
                self._segment_preview_view(
                    self.store.job_dir(state.job_id), state.segment_index
                ),
                error=state.error,
            )
        if state.step is WizardStep.READY_TO_RENDER:
            assert state.job_id
            return render_ready(
                self._ready_view(self.store.job_dir(state.job_id), state.job_id),
                error=state.error,
            )
        if state.step is WizardStep.RENDERING:
            can_stop, stopping = self._cancel_view(state, self.busy())
            return render_rendering(
                state.progress, can_stop=can_stop, stopping=stopping
            )
        if state.step is WizardStep.RESULT:
            assert state.job_id
            job_dir = self.store.job_dir(state.job_id)
            scale, _strength, summary = _active_knobs(
                job_dir, Path(self.store.load(state.job_id).source_path)
            )
            report = load_render_report(job_dir)
            return render_result(
                output_url=_output_url(job_dir) or "/output.mp4",
                error=state.error,
                scale=scale,
                strength=summary,
                seams=report["seams"],
                seam_error=report["error"],
            )
        if state.step is WizardStep.DONE:
            return render_done()
        return render_intro()

    def api_view(self) -> dict[str, Any]:
        """JSON snapshot of the current step for a separate UI shell."""
        state = self._state()
        # Read busy before the run: a run's outcome is set before it lets go
        # of the work lock, so "not busy" is never paired with "running".
        busy = self.busy()
        run = self._public_run(state)
        cancellable, cancel_requested = self._cancel_view(state, busy)
        interrupted = self.interrupted(state, busy)
        elsewhere = False
        if interrupted and state.job_id:
            elsewhere = running_elsewhere(self.store.job_dir(state.job_id))
            cancellable = cancellable and not elsewhere
        output_ready = False
        output_url: str | None = None
        scale: int | None = None
        strength: float | None = None
        summary: str | None = None
        seams: list[dict[str, Any]] = []
        seam_error: str | None = None
        if state.job_id and state.step in {WizardStep.RESULT, WizardStep.DONE}:
            job_dir = self.store.job_dir(state.job_id)
            output_url = _output_url(job_dir)
            output_ready = output_url is not None
            if state.step is WizardStep.RESULT:
                scale, strength, summary = _active_knobs(
                    job_dir, Path(self.store.load(state.job_id).source_path)
                )
                report = load_render_report(job_dir)
                seams, seam_error = report["seams"], report["error"]
        looks: dict[str, Any] | None = None
        if state.job_id and state.step is WizardStep.READY_TO_RENDER:
            looks = self._ready_view(self.store.job_dir(state.job_id), state.job_id)
        segments: dict[str, Any] | None = None
        if state.job_id and state.step is WizardStep.SEGMENTS:
            segments = self._segments_view(self.store.job_dir(state.job_id))
        segment_preview: dict[str, Any] | None = None
        if state.job_id and state.step is WizardStep.PREVIEW_SEGMENT:
            segment_preview = self._segment_preview_view(
                self.store.job_dir(state.job_id), state.segment_index
            )
        return {
            "step": state.step.value,
            "busy": busy,
            "error": state.error,
            # A worker that died mid-run leaves its last progress behind.
            "progress": _aged(state.progress, time.time()) if busy else None,
            "run": run,
            "cancellable": cancellable,
            "cancel_requested": cancel_requested,
            "stopped": state.stopped,
            # Left mid-step: ?run=1 is refused until a POST to /resume (or
            # /cancel to stop it instead).
            "interrupted": interrupted,
            "running_elsewhere": elsewhere,
            "job": self._job_summary(state, busy),
            "navigation": self.navigation(),
            "segment_index": state.segment_index,
            "job_id": state.job_id,
            "has_gemini_key": resolve_gemini_api_key() is not None,
            "output_ready": output_ready,
            "output_url": output_url,
            "scale": scale,
            "strength": strength,
            "strength_summary": summary,
            "looks": looks,
            "segments": segments,
            "segment_preview": segment_preview,
            "seams": seams,
            "seam_error": seam_error,
        }

    def handle_post(self, path: str, fields: dict[str, list[str]]) -> None:
        """Apply one POST. POSTs run one at a time, never beside a worker.

        A page may send the ``job_id`` it shows; when the wizard has moved
        to another job since, :class:`WizardConflict` refuses the POST.
        """
        with self._post_lock:
            outer = self._posting
            self._posting = threading.get_ident()
            try:
                self._handle_post(path, fields)
            finally:
                self._posting = outer

    def _handle_post(self, path: str, fields: dict[str, list[str]]) -> None:
        sent_job = fields.get("job_id", [""])[0].strip()
        if path == "/cancel":
            # The one action meant for while a run works.
            self.cancel(job_id=sent_job, run_id=fields.get("run_id", [""])[0].strip())
            return
        if path == "/resume":
            self.resume(job_id=sent_job)
            return
        if path == "/jobs/open":
            # ``job`` is the job to open; ``job_id`` stays the one the page
            # shows, so it is not checked here: any window may switch jobs.
            self.open_job(fields.get("job", [""])[0].strip())
            return
        if self.busy():
            raise WizardError("the last step is still running; wait for it")
        state = self._state()
        if sent_job and sent_job != state.job_id:
            raise WizardConflict(_JOB_CONFLICT_MESSAGE)
        state.error = None
        state.stopped = None
        self._dispatch_post(path, fields, state)

    def _dispatch_post(
        self, path: str, fields: dict[str, list[str]], state: WizardState
    ) -> None:
        if path == "/intro":
            state.step = WizardStep.LLM_CHOICE
            self._save(state)
            return
        if path == "/llm":
            choice = fields.get("choice", [""])[0]
            if choice == "gemini":
                state.step = WizardStep.SETUP
            elif choice == "local":
                state.step = WizardStep.LOCAL_LLM_STUB
            else:
                state.error = "pick how you want to run the LLM"
                state.step = WizardStep.LLM_CHOICE
            self._save(state)
            return
        if path == "/local-llm":
            # Deferred stub: only path forward is Gemini setup.
            state.step = WizardStep.SETUP
            self._save(state)
            return
        if path == "/setup":
            if fields.get("skip", [""])[0] == "1" and resolve_gemini_api_key():
                state.step = WizardStep.PICK_FILE
                self._save(state)
                return
            key = fields.get("api_key", [""])[0]
            try:
                store_gemini_api_key(key)
            except GeminiError as exc:
                state.error = str(exc)
                state.step = WizardStep.SETUP
                self._save(state)
                return
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        if path == "/pick":
            self._pick_file(state, fields.get("path", [""])[0])
            return
        if path == "/browse":
            self._browse(state)
            return
        if path == "/goto":
            self._goto(state, fields.get("to", [""])[0])
            return
        if path == "/new":
            self.start_new_job()
            return
        if path == "/segments":
            self._segments_post(state, fields)
            return
        if path == "/preview":
            self._preview_post(state, fields)
            return
        if path == "/scene":
            self._scene_post(state, fields)
            return
        if path == "/ready":
            self._ready_post(state, fields)
            return
        if path == "/result":
            self._result_post(state, fields)
            return
        raise WizardError(f"unknown action {path}")

    def start_new_job(self) -> None:
        """Leave the current job as it is on disk and go back to picking a file.

        Setup is skipped when a Gemini key is already available.
        """
        step = WizardStep.PICK_FILE if resolve_gemini_api_key() else WizardStep.SETUP
        self._save(WizardState(step=step))

    def _render_failed(self, state: WizardState, detail: str) -> None:
        """A render failed. Keep whatever the editor already had.

        The renderer replaces ``output.mp4`` in one step, so the file on disk
        is either the old output or the new one. The rules, in order:

        1. The render finished (``render_plan`` returned) and a later step
           failed, such as the probe, the seam check or a save: the output
           holds the new plan, so the result screen shows it with the error.
        2. The render did not finish and there is an earlier output: it is
           untouched, so the plan goes back to the one it holds (the copy
           ``_ready_post`` or ``_result_post`` kept) and the result screen
           shows it with the error. A failed rescale would otherwise pin
           later notes to a size it never reached.
        3. There is no output yet, but the approved scenes are still on
           disk: back to "Ready to render" with the error, so the editor can
           try again. The job is kept as it is and nothing is deleted.
        4. Otherwise (the scenes cannot be read back) the job is marked
           failed and the editor goes back to the pick screen.
        """
        run = self._run
        self._mark_failed(run)
        assert state.job_id
        job_id = state.job_id
        job_dir = self.store.job_dir(job_id)
        before = job_dir / _PLAN_BEFORE_NOTE
        state.notice = None
        state.progress = None
        has_output = (job_dir / "output.mp4").is_file()
        if run is not None and run.rendered:
            before.unlink(missing_ok=True)
            self._settle_rendered_plan(job_dir)
        elif has_output and before.is_file():
            with contextlib.suppress(OSError):
                before.replace(job_dir / UPSCALE_PLAN_FILE)
        if has_output or (run is not None and run.rendered):
            state.error = detail
            state.step = WizardStep.RESULT
            self._save(state)
            return
        # A copy of a plan with no output to go with it means nothing.
        before.unlink(missing_ok=True)
        scenes = self._approved_scene_count(job_dir)
        if scenes:
            state.error = f"The render failed, so there is no result yet: {detail}"
            state.segment_index = state.approved_count = scenes
            state.step = WizardStep.READY_TO_RENDER
            self._save(state)
            return
        with contextlib.suppress(JobError):
            self.store.transition(job_id, JobState.FAILED)
        message = f"The upscale failed, so there is no result yet: {detail}"
        save_job_wizard_state(
            job_dir,
            WizardState(step=WizardStep.PICK_FILE, job_id=job_id, error=message),
        )
        self._save(WizardState(step=WizardStep.PICK_FILE, error=message))

    def _render_cancelled(self, state: WizardState) -> None:
        """The editor stopped a render before it replaced the output.

        The rules of :meth:`_render_failed` apply, except that a stop is
        never a failure: with an earlier output the result screen shows it
        with the plan it holds; without one the editor goes back to "Ready
        to render", even when the scenes cannot be read back, and the job is
        never marked failed. The render's work files are removed.
        """
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        before = job_dir / _PLAN_BEFORE_NOTE
        state.notice = None
        state.progress = None
        state.error = None
        for name in ("shot_work", "range_work"):
            rmtree_quietly(job_dir / name)
        unlink_quietly(job_dir / "output.encoding.mp4")
        if (job_dir / "output.mp4").is_file():
            if before.is_file():
                with contextlib.suppress(OSError):
                    before.replace(job_dir / UPSCALE_PLAN_FILE)
            state.stopped = _STOPPED_RENDER_KEPT
            state.step = WizardStep.RESULT
            self._save(state)
            return
        unlink_quietly(before)
        scenes = self._approved_scene_count(job_dir)
        if scenes:
            state.segment_index = state.approved_count = scenes
        state.stopped = _STOPPED_RENDER_NONE
        state.step = WizardStep.READY_TO_RENDER
        self._save(state)

    def _settle_rendered_plan(self, job_dir: Path) -> None:
        """Clear the plan's pending span once the output holds it.

        Best effort: a render that failed after its output was written may
        not have got as far as saving this itself.
        """
        with contextlib.suppress(OSError, UpscaleError, ValueError, TypeError):
            plan = load_upscale_plan(job_dir)
            if plan is not None and plan.pending is not None:
                save_upscale_plan(job_dir, replace(plan, pending=None))

    def _approved_scene_count(self, job_dir: Path) -> int:
        """How many approved scenes a new render can start from, or 0.

        0 when the shots or the plan are missing, unreadable, or do not
        match one to one.
        """
        try:
            shots = load_shots(job_dir)
            plan = load_plan(job_dir)
            if shots is None or plan is None:
                return 0
            count = len(plan.recipes)
            return count if count and count == len(shots.shots) else 0
        except _PLAN_READ_ERRORS:
            return 0

    def _goto(self, state: WizardState, target: str) -> None:
        """Go back (or forward again) to setup, the file pick, or the segments.

        Nothing is deleted: the job stays, and approving the same segments
        again carries on from the segment the editor was on.
        """
        if not self.navigation().get(target):
            state.error = f"you cannot go to {target or 'that step'} from here"
        elif target == "setup":
            state.step = WizardStep.SETUP
        elif target == "file":
            state.step = WizardStep.PICK_FILE
        else:
            state.step = WizardStep.SEGMENTS
        self._save(state)

    def _browse(self, state: WizardState) -> None:
        """Pick the file the editor chooses in the system file dialog."""
        try:
            chosen = choose_video()
        except FileDialogError as exc:
            state.error = str(exc)
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        if chosen is None:
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        self._pick_file(state, str(chosen))

    def upload_target(self, name: str) -> Path:
        """Where an uploaded video is kept; refuses names that are not videos."""
        base = Path(name.replace("\\", "/")).name.strip()
        if Path(base).suffix.lower() not in VIDEO_EXTENSIONS:
            raise WizardError(
                "only video files can be uploaded (" + ", ".join(VIDEO_EXTENSIONS) + ")"
            )
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)[-120:]
        folder = self.jobs_dir / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{uuid.uuid4().hex[:12]}-{safe}"

    def pick_upload(self, path: Path, name: str) -> None:
        """Pick an uploaded file; it is removed again when the pick refuses it."""
        with self._post_lock:
            outer = self._posting
            self._posting = threading.get_ident()
            try:
                state = self._state()
                state.error = None
                state.stopped = None
                self._pick_file(state, str(path), shown_name=Path(name).name)
            finally:
                self._posting = outer
            if self._state().step is not WizardStep.SAMPLING:
                path.unlink(missing_ok=True)

    def _pick_file(
        self, state: WizardState, raw_path: str, *, shown_name: str | None = None
    ) -> None:
        source = Path(raw_path.strip()).expanduser()
        if not source.is_file():
            state.error = f"file not found: {source}"
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        try:
            facts = probe_file(source)
            refusal = input_limit_error(facts)
            if refusal is not None:
                state.error = refusal
                state.step = WizardStep.PICK_FILE
                self._save(state)
                return
            require_supported_timing(source, ffmpeg_bin=self.ffmpeg_bin)
            manifest = self.store.create(source)
            # The job list's title: the name the editor knows the file by.
            title = _job_title(shown_name or source.name)
            self.store.transition(
                manifest.job_id, JobState.PROBED, notes={"title": title}
            )
        except ProbeError:
            # FFprobe's own text names demuxers and internal paths; say what
            # the editor can act on.
            name = shown_name or source.name
            state.error = (
                f"{name} could not be read as a video; it may be damaged, "
                "not a video, or still being copied"
            )
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        except (JobError, UpscaleError) as exc:
            state.error = str(exc)
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        state.job_id = manifest.job_id
        state.segment_index = state.approved_count = 0
        state.step = WizardStep.SAMPLING
        self._save(state)

    def _work_on(
        self,
        state: WizardState,
        job_dir: Path,
        source: Path,
        shots: ShotList,
        plan: ShotPlan,
        indexes: list[int],
        *,
        phase: str,
        done: WizardStep,
        failed: Callable[[WizardState, str], None],
    ) -> None:
        """Propose, preview, and check ``indexes``; then go to ``done``."""
        warnings: list[str] = []
        # The scale was chosen with the segments, watching the whole video;
        # every shot shares it.
        first = not any(recipe.reason for recipe in plan.recipes)
        if first:
            self._progress(state, phase, 10, "Boxing faces, hands, and text")
            shots = self._locate(shots, source, job_dir, warnings)
        video = self._gemini_video(state, job_dir, source, phase, 20, 30)
        self._progress(state, phase, 30, "Gemini is proposing settings")
        try:
            kwargs: dict[str, Any] = {"video": video} if video is not None else {}
            reply = self.gemini().propose_looks(
                shots=shots_payload(shots, plan, indexes),
                stills=[job_dir / shots.shots[i].still for i in indexes],
                bounds=LOOK_BOUNDS,
                scales=(plan.scale,),
                protected_strength=PROTECTED_STRENGTH,
                job_dir=job_dir,
                **kwargs,
            )
            plan = apply_proposals(plan, shots, list(reply["shots"]), indexes)
        except GeminiError as exc:
            warnings.append(f"Gemini could not propose settings ({exc})")
            plan = _fallback_plan(plan, shots, indexes)
        self._progress(state, phase, 50, "Rendering previews on the GPU")
        try:
            plan = render_previews(
                source,
                job_dir,
                shots,
                plan,
                indexes,
                engine=self.upscale_engine(),
                ffmpeg_bin=self.ffmpeg_bin,
            )
            plan = measure_shimmer(job_dir, plan, indexes, ffmpeg_bin=self.ffmpeg_bin)
            self._progress(state, phase, 80, "Checking the previews")
            plan = self._check(
                plan,
                shots,
                indexes,
                source,
                job_dir,
                warnings,
                ask_gemini=not warnings,
            )
        except (UpscaleError, ShotError, OSError) as exc:
            failed(state, str(exc))
            return
        plan = finish(plan, indexes)
        if warnings:
            plan.error = "; ".join(warnings) + "."
        save_plan(job_dir, plan)
        state.error = None
        state.step = done
        self._save(state)

    def _locate(
        self, shots: ShotList, source: Path, job_dir: Path, warnings: list[str]
    ) -> ShotList:
        """Box faces, hands, and text so the blend can hold SPAN down there."""
        targets = locate_targets(shots)
        if not targets:
            return shots
        try:
            stills = locate_stills(
                source, job_dir, shots, targets, ffmpeg_bin=self.ffmpeg_bin
            )
            found = self.gemini().locate_regions(
                shots=[
                    {
                        "index": index,
                        "label": shots.shots[index].label,
                        "contains": [
                            kind
                            for kind in shots.shots[index].contains
                            if kind in REGION_STRENGTH
                        ],
                    }
                    for index in targets
                ],
                stills=stills,
                kinds=tuple(REGION_STRENGTH),
                max_regions=MAX_REGIONS,
                job_dir=job_dir,
            )
        except (GeminiError, ShotError) as exc:
            warnings.append(
                f"Gemini could not box faces, hands, and text ({exc}); those "
                f"shots are capped at strength {PROTECTED_STRENGTH:g} overall"
            )
            return shots
        shots = apply_regions(shots, found, targets)
        save_shots(job_dir, shots)
        return shots

    def _check(
        self,
        plan: ShotPlan,
        shots: ShotList,
        indexes: list[int],
        source: Path,
        job_dir: Path,
        warnings: list[str],
        *,
        ask_gemini: bool,
    ) -> ShotPlan:
        """Check each preview; failed shots are fixed and redone once.

        Gemini compares the stills (with the measured shimmer alongside);
        then the shimmer guard fails any preview whose texture crawls,
        whatever Gemini said, since a still cannot show that.
        """
        changed: list[int] = []
        if ask_gemini:
            try:
                verdicts = self.gemini().check_previews(
                    shots=[
                        {
                            "index": index,
                            "label": shots.shots[index].label,
                            "contains": list(shots.shots[index].contains),
                            "strength": plan.recipes[index].strength,
                            "look": plan.recipes[index].look.to_dict(),
                            "shimmer": plan.recipes[index].shimmer,
                        }
                        for index in indexes
                    ],
                    pairs=[check_stills(job_dir, plan, index) for index in indexes],
                    problems=CHECK_PROBLEMS,
                    job_dir=job_dir,
                )
                plan, changed = apply_checks(plan, shots, verdicts, indexes)
            except GeminiError as exc:
                warnings.append(f"Gemini could not check the previews ({exc})")
        plan, flickering = guard_shimmer(plan, indexes, changed)
        redo = sorted({*changed, *flickering})
        if redo:
            plan = render_previews(
                source,
                job_dir,
                shots,
                plan,
                redo,
                engine=self.upscale_engine(),
                ffmpeg_bin=self.ffmpeg_bin,
            )
            plan = measure_shimmer(job_dir, plan, redo, ffmpeg_bin=self.ffmpeg_bin)
        return plan

    def _check_seams(
        self, job_dir: Path, output: Path, source: Path, candidates: list[float]
    ) -> None:
        """Record likely seams in the new output. Never fails the render."""
        report: dict[str, Any] = {"checked": candidates, "seams": [], "error": None}
        try:
            report["seams"] = find_seams(
                output, source, candidates, ffmpeg_bin=self.ffmpeg_bin
            )
        except StabilityError as exc:
            report["error"] = f"The seam check could not run: {exc}"
        (job_dir / RENDER_REPORT_FILE).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )

    def _looks_view(self, job_dir: Path) -> dict[str, Any]:
        try:
            shots = load_shots(job_dir)
            plan = load_plan(job_dir)
        except _PLAN_READ_ERRORS as exc:
            # A half-written or damaged file: show the error, not a crash.
            return {"scale": None, "error": str(exc), "shots": []}
        if shots is None or plan is None:
            return {"scale": None, "error": None, "shots": []}
        items: list[dict[str, Any]] = []
        # A new plan starts its preview names at r1 again; the generation
        # keeps a browser from showing the last plan's cached r1.
        query = f"?g={plan.generation}" if plan.generation else ""
        for index, (shot, recipe) in enumerate(
            zip(shots.shots, plan.recipes, strict=False)
        ):
            before, after = (
                f"/previews/{name}{query}"
                for name in (
                    f"shot_{index:03d}_r{recipe.rev}_before.mp4",
                    f"shot_{index:03d}_r{recipe.rev}_after.mp4",
                )
            )
            items.append(
                {
                    "index": index,
                    "start_s": shot.start_s,
                    "end_s": shot.end_s,
                    "label": shot.label,
                    "contains": list(shot.contains),
                    "regions": [region.to_dict() for region in shot.regions],
                    "strength": recipe.strength,
                    "look": recipe.look.to_dict(),
                    "reason": recipe.reason,
                    "check": recipe.check.to_dict(),
                    "shimmer": recipe.shimmer,
                    "use_upscale": recipe.use_upscale,
                    "upscale_override": recipe.upscale_override,
                    "before_url": before if recipe.rev else None,
                    "after_url": after if recipe.rev else None,
                }
            )
        return {"scale": plan.scale, "error": plan.error, "shots": items}

    # Segment loop -----------------------------------------------------------

    def run_segmenting(self) -> None:
        with self._running("segmenting") as run:
            if run is None:
                return
            state = self._state()
            if state.job_id and state.step is WizardStep.SAMPLING:
                self._sample(state)
            state = self._state()
            if state.job_id and state.step is WizardStep.PROPOSING:
                self._propose_segments(state)

    def _sample(self, state: WizardState) -> None:
        """Save the still grid Gemini cuts the video from."""
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        source = Path(self.store.load(state.job_id).source_path)
        self._progress(state, "sampling", 2, "Saving stills from the video")
        try:
            duration = float(probe_file(source).duration_s or 0.0)
            times = sample_times(duration)
            extract_segment_stills(
                source,
                job_dir,
                times,
                ffmpeg_bin=self.ffmpeg_bin,
                on_progress=self._task_progress(
                    state,
                    "sampling",
                    2,
                    25,
                    "stills",
                    lambda done: (
                        "Saving stills from the video: "
                        f"{round(done * len(times))} of {len(times)}"
                    ),
                ),
            )
        except (SegmentLoopError, ProbeError, ValueError, OSError) as exc:
            self._segmenting_failed(state, str(exc))
            return
        # A stop asked for while the last stills were saved lands here, not
        # after Gemini has been asked.
        self._check_stop()
        state.step = WizardStep.PROPOSING
        self._save(state)

    def _propose_segments(self, state: WizardState) -> None:
        """Ask Gemini for segments; one whole-video segment if it cannot."""
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        source = Path(self.store.load(state.job_id).source_path)
        try:
            duration = float(probe_file(source).duration_s or 0.0)
            earlier = load_segment_list(job_dir)
        except (SegmentLoopError, ProbeError, OSError) as exc:
            self._segmenting_failed(state, str(exc))
            return
        stills = load_segment_stills(job_dir)
        if not stills:
            self._segmenting_failed(state, "the stills are missing")
            return
        assert state.job_id
        video = self._gemini_video(state, job_dir, source, "proposing", 25, 75)
        self._progress(
            state,
            "proposing",
            75,
            "Gemini is watching the video and proposing segments"
            if video is not None
            else "Gemini is proposing segments from the stills",
        )
        error: str | None = None
        scale: int | None = None
        reason = ""
        ask: dict[str, Any] = {
            "duration_s": duration,
            "contains": CONTENTS,
            "note": earlier.note if earlier else "",
            "max_segments": MAX_LOOP_SEGMENTS,
            "min_segment_s": MIN_LOOP_SEGMENT_S,
            "job_dir": job_dir,
            "scales": SCALE_OPTIONS,
            "source_size": self._source_size(state.job_id),
        }
        stills_ask: dict[str, Any] = {
            "stills": [job_dir / still.path for still in stills],
            "times": [still.t_s for still in stills],
        }
        try:
            try:
                media: dict[str, Any] = (
                    stills_ask if video is None else {"video": video}
                )
                raw = self._abandonable(
                    lambda: self.gemini().propose_segments(**media, **ask)
                )
            except GeminiError as exc:
                if video is None:
                    raise
                # The upload is a convenience: the stills still work. The
                # upload stays usable for the next request.
                self._note_video_problem(
                    job_dir, f"watching the video failed: {exc}", video=video
                )
                raw = self._abandonable(
                    lambda: self.gemini().propose_segments(**stills_ask, **ask)
                )
            else:
                if video is not None:
                    self._save_video_record(job_dir, video)
            # Test doubles may still answer with the bare segment list.
            found = raw if isinstance(raw, list) else raw.get("segments", [])
            if isinstance(raw, dict):
                scale = raw.get("scale") if raw.get("scale") in SCALE_OPTIONS else None
                reason = str(raw.get("scale_reason") or "") if scale else ""
            segments = repair_coverage(found, duration)
        except (GeminiError, SegmentLoopError) as exc:
            error = (
                f"Gemini could not propose segments ({exc}); the whole video is one."
            )
            segments = [whole_video(duration)]
        # From here on the new segments are saved and the stop is refused.
        if self._run is not None:
            self._run.commit()
        save_segment_list(
            job_dir,
            SegmentList(
                segments=segments,
                error=error,
                scale=scale,
                scale_reason=reason,
                # A page still showing the earlier list cannot change this one.
                revision=earlier.revision + 1 if earlier else 0,
            ),
        )
        self._clear_stopped_note(state.job_id)
        state.error = None
        state.step = WizardStep.SEGMENTS
        self._save(state)

    def _gemini_video(
        self,
        state: WizardState,
        job_dir: Path,
        source: Path,
        phase: str,
        low: float,
        high: float,
    ) -> GeminiVideo | None:
        """The upload Gemini watches, made and uploaded on first use.

        An upload lives about 48 hours, so a job resumed later uploads
        again. None when the video cannot be used (no File API in this
        client, an encode or upload failure); the reason is kept for the
        Segments screen, and Gemini works from the stills instead.
        """
        try:
            client = self.gemini()
        except GeminiError:
            # No key: the callers' own fallbacks report it.
            return None
        upload = getattr(client, "upload_video", None)
        if not callable(upload):
            return None
        record = job_dir / GEMINI_VIDEO_FILE
        try:
            saved = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = {}
        known = GeminiVideo.from_dict(
            saved.get("video") if isinstance(saved, dict) else None
        )
        alive = getattr(client, "video_alive", None)
        if (
            known is not None
            and known.usable(time.time())
            and (not callable(alive) or alive(known))
        ):
            return known
        if isinstance(saved, dict) and saved.get("error") and known is None:
            # It failed for this job already; do not try on every call.
            return None
        middle = low + (high - low) * 0.4
        try:
            duration = float(probe_file(source).duration_s or 0.0)
            copy = make_gemini_copy(
                source,
                job_dir,
                duration,
                ffmpeg_bin=self.ffmpeg_bin,
                on_progress=self._task_progress(
                    state,
                    phase,
                    low,
                    middle,
                    "gemini-copy",
                    lambda done: (
                        f"Preparing a silent copy for Gemini: {round(done * 100)}%"
                    ),
                ),
            )
            self._progress(state, phase, middle, "Uploading the video to Gemini")
            on_upload = self._task_progress(
                state,
                phase,
                middle,
                high,
                "gemini-upload",
                lambda done: f"Uploading the video to Gemini: {round(done * 100)}%",
            )
            video = self._abandonable(lambda: upload(copy, on_progress=on_upload))
            if not isinstance(video, GeminiVideo):
                raise GeminiError("the upload returned no video")
        except (GeminiError, SegmentLoopError, ProbeError, OSError) as exc:
            self._note_video_problem(job_dir, str(exc))
            return None
        self._save_video_record(job_dir, video)
        return video

    def _save_video_record(
        self, job_dir: Path, video: GeminiVideo | None, error: str | None = None
    ) -> None:
        record = job_dir / GEMINI_VIDEO_FILE
        payload = {"video": video.to_dict() if video else None, "error": error}
        record.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _note_video_problem(
        self, job_dir: Path, detail: str, *, video: GeminiVideo | None = None
    ) -> None:
        """Keep ``detail`` for the Segments screen.

        With ``video`` (the upload worked but one request with it failed),
        the upload is kept so later requests use it again; without, the
        upload failed and is not tried again for this job.
        """
        self._save_video_record(job_dir, video, detail)

    def _video_problem(self, job_dir: Path) -> str:
        try:
            saved = json.loads((job_dir / GEMINI_VIDEO_FILE).read_text("utf-8"))
        except (OSError, ValueError):
            return ""
        error = saved.get("error") if isinstance(saved, dict) else None
        return error if isinstance(error, str) else ""

    def _segmenting_failed(self, state: WizardState, detail: str) -> None:
        """Sampling or proposing failed; mark the job failed and go back to pick."""
        self._mark_failed(self._run)
        assert state.job_id
        job_id = state.job_id
        with contextlib.suppress(JobError):
            self.store.transition(job_id, JobState.FAILED)
        message = f"Finding the segments failed: {detail}"
        save_job_wizard_state(
            self.store.job_dir(job_id),
            WizardState(step=WizardStep.PICK_FILE, job_id=job_id, error=message),
        )
        self._save(WizardState(step=WizardStep.PICK_FILE, error=message))

    def _segmenting_cancelled(self, state: WizardState) -> None:
        """The editor stopped sampling or proposing.

        A stop is not a failure: the job stays as it was (never marked
        failed) and its upload record keeps no error, so the next try
        watches the video again. When segments were proposed before (the
        editor sent a note back), they come back as they were; otherwise
        the editor goes back to the pick screen and the job keeps its place.
        """
        assert state.job_id
        job_id = state.job_id
        job_dir = self.store.job_dir(job_id)
        # Only the half-made copy: a finished copy may still be open in an
        # upload the stop left behind (Windows would refuse to remove it).
        unlink_quietly(job_dir / Path(GEMINI_COPY_NAME).with_suffix(".part.mp4"))
        with contextlib.suppress(JobError, OSError):
            self.store.update_notes(job_id, {"stopped": utc_now_iso()})
        try:
            found = load_segment_list(job_dir)
        except (SegmentLoopError, OSError):
            found = None
        if found is not None and found.segments:
            state.stopped = _STOPPED_SEGMENTS_KEPT
            state.step = WizardStep.SEGMENTS
            self._save(state)
            return
        save_job_wizard_state(
            job_dir,
            WizardState(
                step=WizardStep.PICK_FILE, job_id=job_id, stopped=_STOPPED_SEGMENTS_NONE
            ),
        )
        self._save(
            WizardState(step=WizardStep.PICK_FILE, stopped=_STOPPED_SEGMENTS_NONE)
        )

    def _clear_stopped_note(self, job_id: str) -> None:
        """Forget that the job was stopped once a run gets it further."""
        with contextlib.suppress(JobError, OSError):
            if self.store.load(job_id).notes.get("stopped"):
                self.store.update_notes(job_id, {"stopped": None})

    def _segments_view(self, job_dir: Path) -> dict[str, Any]:
        try:
            found = load_segment_list(job_dir)
        except SegmentLoopError as exc:
            return {"segments": [], "error": str(exc), "note": ""}
        if found is None:
            return {"segments": [], "error": None, "note": ""}
        stills = load_segment_stills(job_dir)
        items: list[dict[str, Any]] = []
        for index, segment in enumerate(found.segments):
            item = segment.to_dict()
            item["index"] = index
            # The sampled still nearest the middle stands for the segment.
            item["still_url"] = _nearest_still_url(
                stills, segment.start_s, segment.end_s
            )
            items.append(item)
        return {
            "segments": items,
            "error": found.error,
            "note": found.note,
            "scale": found.scale,
            "scale_reason": found.scale_reason,
            # Why Gemini worked from the stills, when it could not watch.
            "video_problem": self._video_problem(job_dir),
            # For editing by hand: send ``revision`` back with each change.
            "revision": found.revision,
            "approved": found.approved,
            # The segments always cover the whole video, so the last end is
            # where it ends.
            "duration_s": found.segments[-1].end_s if found.segments else None,
            "min_segment_s": MIN_LOOP_SEGMENT_S,
            "max_segments": MAX_LOOP_SEGMENTS,
        }

    def _segments_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        """Approve the segments, change them by hand, or send them back to
        Gemini with a note.

        By hand, the editor moves the cut after a scene (``move``), joins a
        scene with the next one (``merge``), or splits one (``split``), at
        ``at`` seconds. Each change sends the ``revision`` the page shows; a
        page showing an older list is refused, so two windows cannot both
        change the scenes. A change that leaves the scenes as they were
        keeps their approval.
        """
        if state.step is not WizardStep.SEGMENTS or not state.job_id:
            raise WizardError("there are no segments to change")
        job_dir = self.store.job_dir(state.job_id)
        try:
            found = load_segment_list(job_dir)
        except SegmentLoopError as exc:
            raise WizardError(str(exc)) from exc
        if found is None:
            raise WizardError("the segments are missing; pick the file again")
        action = fields.get("action", [""])[0]
        # Approve and note predate revisions, so older pages may leave it
        # out; a change by hand always says which list it changes.
        _check_revision(found, fields, required=action in _SEGMENT_EDITS)
        if action == "approve":
            if not found.approved:
                # New segments: plan them from the start.
                self._clear_old_plan(job_dir)
                found.approved = True
                save_segment_list(job_dir, found)
                state.segment_index = state.approved_count = 0
            state.step = WizardStep.ANALYZING
        elif action == "note":
            text = " ".join(fields.get("note", [""])[0].split())
            if not text:
                state.error = "say what to change about the segments"
            else:
                found.note = text[:500]
                found.approved = False
                save_segment_list(job_dir, found)
                self._enter_running_step(state, WizardStep.PROPOSING)
        elif action in _SEGMENT_EDITS:
            try:
                edited = _edit_segments(found.segments, action, fields)
            except SegmentLoopError as exc:
                state.error = str(exc)
            else:
                # Compared as saved, to the millisecond: moving a cut to
                # where it already is changes nothing.
                if edited != rounded_segments(found.segments):
                    found.segments = edited
                    found.revision += 1
                    found.approved = False
                    # Gemini's word on its proposal no longer describes it.
                    found.error = None
                    save_segment_list(job_dir, found)
        else:
            state.error = f"unknown action {action!r}"
        self._save(state)

    def _clear_old_plan(self, job_dir: Path) -> None:
        """Forget the plan of the segments approved before, before new ones are.

        The segments are still saved as not approved while this runs, so a
        stop part way leaves nothing that looks approved. The shots and the
        plan must go (the next analysis builds them from the new segments);
        old previews, the plan the output was rendered from, and the seam
        report are removed as far as the disk lets us.
        """
        try:
            unlink_retrying(shots_path(job_dir))
            unlink_retrying(plan_path(job_dir))
        except OSError as exc:
            raise WizardError(
                f"could not clear the settings of the old scenes: {exc}"
            ) from exc
        remove_previews(job_dir)
        unlink_quietly(job_dir / _PLAN_BEFORE_NOTE)
        unlink_quietly(job_dir / RENDER_REPORT_FILE)

    def run_analyze(self) -> None:
        with self._running("analyze") as run:
            if run is None:
                return
            state = self._state()
            if state.step is WizardStep.ANALYZING and state.job_id:
                self._analyze(state)

    def _analyze(self, state: WizardState) -> None:
        """Plan, preview, and check the current segment only."""
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        source = Path(self.store.load(state.job_id).source_path)
        try:
            found = load_segment_list(job_dir)
        except SegmentLoopError:
            found = None
        if found is None or not found.approved:
            message = "this job has no approved segments; pick the file again"
            self._save(WizardState(step=WizardStep.PICK_FILE, error=message))
            return
        shots = load_shots(job_dir)
        plan = load_plan(job_dir)
        if shots is not None and _spans(shots.shots) != _spans(found.segments):
            # Shots measured from other segments (an approval that could not
            # clear them): measure the approved ones again.
            shots = None
        if shots is None or plan is None:
            self._progress(state, "analyzing", 5, "Measuring the segments")
            try:
                shots = shots_from_segments(
                    source, job_dir, found.segments, ffmpeg_bin=self.ffmpeg_bin
                )
            except (SegmentLoopError, ProbeError) as exc:
                self._analyze_failed(state, str(exc))
                return
            save_shots(job_dir, shots)
            options = SCALE_OPTIONS
            scale = found.scale if found.scale in options else default_scale(options)
            plan = new_plan(shots, scale)
            save_plan(job_dir, plan)
            # A new plan has nothing approved yet.
            state.segment_index = state.approved_count = 0
        self._fit_plan_scale(job_dir, plan)
        index = state.segment_index
        if index >= len(plan.recipes):
            state.step = WizardStep.READY_TO_RENDER
            self._save(state)
            return
        recipe = plan.recipes[index]
        if recipe.preview_start is None:
            seed = f"{state.job_id}:{index}:{recipe.rev}"
            plan.recipes[index] = replace(
                recipe, preview_start=random_preview_start(shots.shots[index], seed)
            )
        if not recipe.stale and recipe.rev:
            state.step = WizardStep.PREVIEW_SEGMENT
            self._save(state)
            return
        self._work_on(
            state,
            job_dir,
            source,
            shots,
            plan,
            [index],
            phase="analyzing",
            done=WizardStep.PREVIEW_SEGMENT,
            failed=self._analyze_failed,
        )

    def _analyze_failed(self, state: WizardState, detail: str) -> None:
        """A segment preview failed; go back to the segments to try again."""
        self._mark_failed(self._run)
        state.error = f"The segment preview failed: {detail}"
        state.step = WizardStep.SEGMENTS
        self._save(state)

    def _segment_preview_view(self, job_dir: Path, index: int) -> dict[str, Any]:
        looks = self._looks_view(job_dir)
        items = looks["shots"]
        return {
            "index": index,
            "count": len(items),
            "scale": looks["scale"],
            "error": looks["error"],
            "shot": items[index] if 0 <= index < len(items) else None,
        }

    def _preview_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        """Approve this segment's preview, or redo it with a note."""
        if state.step is not WizardStep.PREVIEW_SEGMENT or not state.job_id:
            raise WizardError("there is no segment preview to change")
        job_dir = self.store.job_dir(state.job_id)
        plan = load_plan(job_dir)
        if plan is None:
            raise WizardError("the segment settings are missing; pick the file again")
        action = fields.get("action", [""])[0]
        try:
            if action == "approve":
                # Approving a reopened scene goes on from the furthest scene
                # reached, not from the one after it.
                state.approved_count = max(
                    state.approved_count, state.segment_index + 1
                )
                state.segment_index = state.approved_count
                state.step = (
                    WizardStep.READY_TO_RENDER
                    if state.segment_index >= len(plan.recipes)
                    else WizardStep.ANALYZING
                )
            elif action == "upscale":
                on = fields.get("on", [""])[0] == "1"
                save_plan(job_dir, set_upscale(plan, state.segment_index, on))
                state.step = WizardStep.ANALYZING
            elif action == "redo":
                save_plan(
                    job_dir,
                    request_redo(
                        plan,
                        state.segment_index,
                        fields.get("note", [""])[0],
                        resample=fields.get("resample", [""])[0] == "1",
                    ),
                )
                state.step = WizardStep.ANALYZING
            else:
                raise ValueError(f"unknown action {action!r}")
        except (ShotPlanError, ValueError) as exc:
            state.error = str(exc)
        self._save(state)

    def _scene_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        """Reopen an approved scene, or the next one to approve, to look again.

        Allowed from a scene preview and from "Ready to render", for any
        scene up to the first one not approved yet. Approving the reopened
        scene goes on from the furthest scene reached.
        """
        if state.step not in _REOPEN_STEPS or not state.job_id:
            raise WizardError(
                "a scene can be reopened from a scene preview or from Ready to render"
            )
        raw = fields.get("index", [""])[0].strip()
        if not re.fullmatch(r"\d{1,3}", raw, re.ASCII):
            raise WizardError("pick one of the scenes shown")
        index = int(raw)
        job_dir = self.store.job_dir(state.job_id)
        try:
            plan = load_plan(job_dir)
            shots = load_shots(job_dir)
        except _PLAN_READ_ERRORS as exc:
            raise WizardError(str(exc)) from exc
        if plan is None:
            raise WizardError("the segment settings are missing; pick the file again")
        # The same check the scene strip makes, so a scene it shows as not
        # reopenable is refused here too.
        if shots is not None and len(shots.shots) != len(plan.recipes):
            raise WizardError("the shots and their settings do not match")
        if index >= len(plan.recipes) or index > state.approved_count:
            raise WizardError(
                f"scene {index + 1} cannot be opened yet; approve the scenes "
                "before it first"
            )
        recipe = plan.recipes[index]
        state.segment_index = index
        # A scene previewed as it is now opens at once; one with changes
        # still to preview is analyzed again first.
        state.step = (
            WizardStep.PREVIEW_SEGMENT
            if recipe.rev and not recipe.stale
            else WizardStep.ANALYZING
        )
        self._save(state)

    def _job_summary(self, state: WizardState, busy: bool) -> dict[str, Any] | None:
        """The open job for the page's stepper and scene strip, or ``None``.

        Each file is read on its own, and one that cannot be read leaves its
        part empty and says why in ``error``, so a damaged file never breaks
        the whole view. Scenes are listed only from the scenes stage on.
        """
        if not state.job_id:
            return None
        job_id = state.job_id
        job_dir = self.store.job_dir(job_id)
        errors: list[str] = []
        title = job_id
        try:
            title = _manifest_title(self.store.load(job_id))
        except Exception as exc:  # noqa: BLE001 - shown in the summary
            errors.append(str(exc) or type(exc).__name__)
        found: SegmentList | None = None
        try:
            found = load_segment_list(job_dir)
        except Exception as exc:  # noqa: BLE001 - shown in the summary
            errors.append(str(exc) or type(exc).__name__)
        shots: ShotList | None = None
        plan: ShotPlan | None = None
        try:
            shots = load_shots(job_dir)
            plan = load_plan(job_dir)
        except Exception as exc:  # noqa: BLE001 - shown in the summary
            errors.append(str(exc) or type(exc).__name__)
            shots = plan = None
        if (
            shots is not None
            and plan is not None
            and len(shots.shots) != len(plan.recipes)
        ):
            errors.append("the shots and their settings do not match")
            shots = plan = None
        stills: list[SegmentStill] = []
        try:
            stills = load_segment_stills(job_dir)
        except Exception as exc:  # noqa: BLE001 - shown in the summary
            errors.append(str(exc) or type(exc).__name__)
        try:
            has_output = (job_dir / "output.mp4").is_file()
        except OSError:
            has_output = False

        spans: list[tuple[float, float, str]] = []
        if shots is not None and plan is not None:
            spans = [(shot.start_s, shot.end_s, shot.label) for shot in shots.shots]
        elif found is not None and found.approved:
            spans = [(seg.start_s, seg.end_s, seg.label) for seg in found.segments]
        count = len(spans)
        approved = min(state.approved_count, count)
        stage = _STEP_STAGES.get(state.step)
        current: int | None = None
        if (
            state.step in {WizardStep.ANALYZING, WizardStep.PREVIEW_SEGMENT}
            and state.segment_index < count
        ):
            current = state.segment_index
        done = {
            "file": True,
            "segments": found is not None and found.approved and bool(found.segments),
            "scenes": plan is not None and bool(plan.recipes) and approved == count,
            "render": has_output,
            "result": has_output,
        }
        stages = [
            {
                "key": key,
                "status": (
                    "current" if key == stage else "done" if done[key] else "pending"
                ),
            }
            for key in JOB_STAGES
        ]
        scenes: list[dict[str, Any]] = []
        # A failed pick, or the segments being redone, shows no strip.
        if stage in {"scenes", "render", "result"}:
            can_reopen = not busy and state.step in _REOPEN_STEPS and plan is not None
            for index, (start_s, end_s, label) in enumerate(spans):
                recipe = plan.recipes[index] if plan is not None else None
                status = (
                    "current"
                    if index == current
                    else "approved"
                    if index < approved
                    else "pending"
                )
                scenes.append(
                    {
                        "index": index,
                        "start_s": start_s,
                        "end_s": end_s,
                        "label": label,
                        "still_url": _nearest_still_url(stills, start_s, end_s),
                        "status": status,
                        "previewed": bool(
                            recipe is not None and recipe.rev and not recipe.stale
                        ),
                        "reopenable": can_reopen
                        and index <= state.approved_count
                        and index != current,
                    }
                )
        return {
            "job_id": job_id,
            "title": title,
            "stage": stage,
            "stages": stages,
            "scene_count": count,
            "approved_count": approved,
            "current_index": current,
            "scenes": scenes,
            "error": "; ".join(errors) or None,
        }

    def _ready_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        """Start the full render from every approved segment."""
        if state.step is not WizardStep.READY_TO_RENDER or not state.job_id:
            raise WizardError("the segments are not all approved yet")
        job_dir = self.store.job_dir(state.job_id)
        shots = load_shots(job_dir)
        plan = load_plan(job_dir)
        if shots is None or plan is None:
            raise WizardError("the segment settings are missing; pick the file again")
        if fields.get("action", [""])[0] != "render":
            state.error = "send action=render to start the full render"
            self._save(state)
            return
        stale = plan.stale_indexes()
        if stale:
            # A scene changed (or was reopened and redone) after its preview;
            # render only what the editor has seen.
            state.error = f"scene {stale[0] + 1} still has changes to preview"
            self._save(state)
            return
        raw = fields.get("scale", [""])[0]
        if raw:
            if not raw.isdigit() or int(raw) not in SCALE_OPTIONS:
                state.error = "pick one of the upscale options shown"
                self._save(state)
                return
            plan.scale = int(raw)
            save_plan(job_dir, plan)
        self._fit_plan_scale(job_dir, plan)
        try:
            upscale_plan = plan.to_upscale_plan(shots)
            self._keep_plan_before_render(job_dir)
            save_upscale_plan(job_dir, upscale_plan)
            self._enter_running_step(state, WizardStep.RENDERING)
        except (ShotPlanError, OSError) as exc:
            state.error = str(exc)
        self._save(state)

    def _keep_plan_before_render(self, job_dir: Path) -> None:
        """Keep the plan the current ``output.mp4`` holds, before a new
        render replaces it, so a render that fails or is stopped can put it
        back (see :meth:`_render_failed`). Nothing is kept without an output.
        """
        before = job_dir / _PLAN_BEFORE_NOTE
        held: UpscalePlan | None = None
        if (job_dir / "output.mp4").is_file():
            # An unreadable plan cannot be put back; the render still runs.
            with contextlib.suppress(UpscaleError, OSError, ValueError):
                held = load_upscale_plan(job_dir)
        if held is None:
            before.unlink(missing_ok=True)
            return
        write_text_atomic(before, json.dumps(held.to_dict(), indent=2) + "\n")

    def _source_size(self, job_id: str) -> tuple[int, int]:
        source = Path(self.store.load(job_id).source_path)
        try:
            video = probe_file(source).primary_video()
        except ProbeError:
            return 0, 0
        return ((video.width or 0), (video.height or 0)) if video else (0, 0)

    def _fit_plan_scale(self, job_dir: Path, plan: ShotPlan) -> None:
        """Bring a saved plan back inside the scale options.

        A plan saved at 4x, before 4x was removed, renders at 2x.
        """
        if plan.scale not in SCALE_OPTIONS:
            plan.scale = default_scale(SCALE_OPTIONS)
            save_plan(job_dir, plan)

    def _ready_view(self, job_dir: Path, job_id: str) -> dict[str, Any]:
        view = self._looks_view(job_dir)
        width, height = self._source_size(job_id)
        view["source_size"] = [width, height]
        view["scales"] = list(SCALE_OPTIONS)
        # Why the scale is what it is: Gemini's reason while its choice stands.
        try:
            found = load_segment_list(job_dir)
        except SegmentLoopError:
            found = None
        chose = (
            found is not None
            and found.scale == view.get("scale")
            and found.scale in view["scales"]
        )
        view["scale_reason"] = found.scale_reason if found and chose else ""
        return view

    def _result_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        mood = fields.get("mood", [""])[0]
        note = fields.get("note", [""])[0].strip()
        if mood == "happy":
            state.step = WizardStep.DONE
            self._save(state)
            return
        if not note:
            state.error = (
                "describe what to change, including a time range if you have one"
            )
            state.step = WizardStep.RESULT
            self._save(state)
            return
        if not state.job_id:
            state.error = "no job to revise"
            self._save(state)
            return
        job_dir = self.store.job_dir(state.job_id)
        if fields.get("restart", [""])[0] == "1":
            self._restart(state, job_dir, note)
            return
        source = Path(self.store.load(state.job_id).source_path)
        try:
            facts = probe_file(source)
            duration = float(facts.duration_s or 0.0)
            plan = load_upscale_plan(job_dir) or UpscalePlan(
                spans=(default_span(duration),)
            )
            prior = plan.latest_revision or plan.spans[-1]
            current = prior.scale
            allowed = SCALE_OPTIONS
            scale = note_scale(note, current, allowed)
            if scale == 1 and current == 1:
                raise ValueError(_SOURCE_SIZE_NOTE)
            raw = self.gemini().revise_upscale_params(
                note=note,
                prior_scale=current,
                prior_strength=prior.strength,
                prior_start_s=prior.start_s,
                prior_end_s=prior.end_s,
                duration_s=duration,
                allowed_scales=allowed,
                job_dir=job_dir,
            )
            ignored = reply_scale(raw.get("scale"))
            raw["scale"] = scale
            patch = UpscaleSpan.from_dict(raw)
            updated = replace_overlapping(
                plan, patch, duration, preserve_shot_settings=True
            )
            write_text_atomic(
                job_dir / _PLAN_BEFORE_NOTE, json.dumps(plan.to_dict(), indent=2) + "\n"
            )
            save_upscale_plan(job_dir, updated)
            state.error = None
            state.notice = None
            if scale == current and ignored not in (None, current):
                example = (
                    ignored
                    if ignored in allowed
                    else max(option for option in allowed if option != current)
                )
                state.notice = (
                    f"Kept {current}x: Gemini answered {ignored}x, but this note "
                    f'did not name a size. To change it, say "make it {example}x".'
                )
            self._enter_running_step(state, WizardStep.RENDERING)
        except (UpscaleError, GeminiError, ProbeError, OSError, ValueError) as exc:
            state.error = str(exc)
            state.step = WizardStep.RESULT
        self._save(state)

    def _restart(self, state: WizardState, job_dir: Path, note: str) -> None:
        """Start the job again from the segments, with the note in the chat."""
        text = " ".join(note.split())[:500]
        append_chat_turn(
            job_dir,
            ChatTurn(
                role="user",
                text=(
                    "The editor watched the finished video, was not happy, and "
                    f"started over from the segments. Their note: {text!r}. Do "
                    "not repeat what they disliked."
                ),
            ),
        )
        restart_from_segmentation(job_dir)
        (job_dir / _PLAN_BEFORE_NOTE).unlink(missing_ok=True)
        state.error = None
        state.notice = None
        state.segment_index = state.approved_count = 0
        state.progress = None
        self._enter_running_step(state, WizardStep.SAMPLING)
        self._save(state)

    def run_render(self) -> None:
        with self._running("render") as run:
            if run is None:
                return
            self._run_render()

    def _run_render(self) -> None:
        state = self._state()
        if state.step is not WizardStep.RENDERING or not state.job_id:
            return
        self._render_upscale(state, self.store.job_dir(state.job_id))

    def _render_upscale(self, state: WizardState, job_dir: Path) -> None:
        assert state.job_id
        source = Path(self.store.load(state.job_id).source_path)
        output = job_dir / "output.mp4"
        # A render from a result note revises the output: its percentage is
        # its own. Only a note leaves a pending span on the plan; a job
        # re-planned from the segments renders as part of the whole job.
        try:
            held = load_upscale_plan(job_dir)
        except (UpscaleError, OSError, ValueError):
            held = None
        self._revising = held is not None and held.pending is not None
        self._progress(state, "rendering", 5, "Upscaling on the GPU")
        self._heat_note = None
        watch = SlowdownWatch()
        sampled = threading.Event()
        sampler = threading.Thread(
            target=self._sample_heat,
            args=(watch, sampled),
            name="gpu-heat",
            daemon=True,
        )
        sampler.start()
        try:
            facts = probe_file(source)
            duration = float(facts.duration_s or 0.0)
            plan = load_upscale_plan(job_dir)
            if plan is None:
                plan = UpscalePlan(spans=(default_span(duration),))
                save_upscale_plan(job_dir, plan)
            revised_ranges: list[tuple[float, float]] = []
            # The run tells FFmpeg and SPAN when to stop, and the renderer
            # commits it just before it replaces the output.
            control = (
                render_control(self._run)
                if self._run is not None
                else contextlib.nullcontext()
            )
            with control:
                render_plan(
                    source,
                    output,
                    plan,
                    engine=self.upscale_engine(),
                    ffmpeg_bin=self.ffmpeg_bin,
                    duration_s=duration,
                    on_progress=self._watched(watch, self._render_progress(state)),
                    on_revise_range=lambda start, end: revised_ranges.append(
                        (start, end)
                    ),
                )
            # The output now holds this plan, so a later failure must keep it.
            if self._run is not None:
                self._run.rendered = True
            (job_dir / _PLAN_BEFORE_NOTE).unlink(missing_ok=True)
            # The renderer probes the old output's keyframes (after legacy
            # delivery conversion, if needed). Keep its actual splice edges,
            # not the requested range or the new output's different keyframes.
            total_frames, rate = video_stream_facts(
                source, ffprobe_bin=_ffprobe_for(self.ffmpeg_bin)
            )
            candidates = seam_candidates(
                plan,
                duration,
                fps=float(Fraction(rate)),
                total_frames=total_frames,
                revised_range=revised_ranges[0] if revised_ranges else None,
            )
            plan = replace(plan, pending=None)
            save_upscale_plan(job_dir, plan)
            self._progress(state, "rendering", 90, "Checking for seams")
            self._check_seams(job_dir, output, source, candidates)
            self._clear_stopped_note(state.job_id)
            state.error = state.notice
            state.notice = None
            state.step = WizardStep.RESULT
        except (UpscaleError, ProbeError, OSError, ValueError) as exc:
            self._render_failed(state, str(exc))
            return
        finally:
            sampled.set()
            sampler.join()
            self._heat_note = None
        self._save(state)

    def _sample_heat(self, watch: SlowdownWatch, stop: threading.Event) -> None:
        """Read the GPU's heat into ``watch`` until the render ends."""
        while not stop.wait(HEAT_SAMPLE_S):
            watch.heat(self.read_gpu_heat())

    def _watched(
        self, watch: SlowdownWatch, report: Callable[[float], None]
    ) -> Callable[[float], None]:
        """``report``, also feeding the render's speed to ``watch`` and
        showing its note when it has one."""

        def tracked(done: float) -> None:
            watch.progress(time.time(), done)
            note = watch.note()
            if note is not None:
                self._heat_note = note
            report(done)

        return tracked
