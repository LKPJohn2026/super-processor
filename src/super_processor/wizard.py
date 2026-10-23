"""Localhost wizard: setup → pick → shots → looks → FlashVSR upscale → result."""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

from .gemini import (
    GeminiClient,
    GeminiError,
    resolve_gemini_api_key,
    store_gemini_api_key,
)
from .jobs import JobError, JobState, JobStore
from .look import LOOK_BOUNDS
from .probe import MediaFacts, ProbeError, probe_file
from .regions import MAX_REGIONS, REGION_STRENGTH
from .segment_loop import (
    LoopSegment,
    SegmentList,
    SegmentLoopError,
    extract_segment_stills,
    load_segment_list,
    load_segment_stills,
    repair_coverage,
    sample_times,
    save_segment_list,
    shots_from_segments,
)
from .segments import MAX_DURATION_S
from .shot_plan import (
    CHECK_PROBLEMS,
    PROTECTED_STRENGTH,
    ShotPlan,
    ShotPlanError,
    allowed_scales,
    apply_checks,
    apply_proposals,
    apply_regions,
    check_stills,
    fallback_recipe,
    finish,
    guard_shimmer,
    load_plan,
    locate_stills,
    locate_targets,
    measure_shimmer,
    new_plan,
    plan_path,
    random_preview_start,
    render_previews,
    request_redo,
    save_plan,
    shots_payload,
)
from .shots import (
    CONTENTS,
    ISSUES,
    ShotError,
    ShotList,
    apply_labels,
    find_shots,
    load_shots,
    merge_with_next,
    save_shots,
    shots_path,
    split_at,
)
from .stability import StabilityError, find_seams, seam_candidates
from .upscale import (
    FlashVsrEngine,
    UpscaleEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleSpan,
    default_span,
    load_upscale_plan,
    render_plan,
    replace_overlapping,
    save_upscale_plan,
    span_at,
)
from .wizard_pages import (
    render_done,
    render_finding_shots,
    render_intro,
    render_llm_choice,
    render_local_llm_stub,
    render_looks,
    render_pick,
    render_planning,
    render_rendering,
    render_result,
    render_segment_loop_pending,
    render_setup,
    render_shots,
)

WIZARD_STATE_FILE = "wizard_state.json"
WIZARD_SCHEMA_VERSION = 3
# Inputs are limited to 1080p (either orientation) and 30 minutes. FlashVSR
# at 2x already makes 4K from 1080p; larger inputs exceed what one GPU pass
# and the job directory are sized for.
# Set to 1 to send a picked file through the segment loop instead of the
# shot flow. The loop has no page of its own yet; the JSON API drives it.
SEGMENT_LOOP_ENV = "SUPER_PROCESSOR_SEGMENT_LOOP"
MAX_INPUT_LONG_EDGE = 1920
MAX_INPUT_SHORT_EDGE = 1080


class WizardError(RuntimeError):
    """Raised when a wizard step cannot complete."""


class WizardStep(str, Enum):
    """Persisted wizard screen."""

    INTRO = "intro"
    LLM_CHOICE = "llm_choice"
    LOCAL_LLM_STUB = "local_llm_stub"
    SETUP = "setup"
    PICK_FILE = "pick_file"
    FINDING_SHOTS = "finding_shots"
    SHOTS = "shots"
    PLANNING = "planning"
    LOOKS = "looks"
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
_RETIRED_MESSAGE = (
    "this job used the removed split and enhance flow; pick the file again"
)


_SEGMENTING_STEPS = frozenset({WizardStep.SAMPLING, WizardStep.PROPOSING})
_LOOP_TITLES = {
    WizardStep.SAMPLING: "Sampling stills",
    WizardStep.PROPOSING: "Proposing segments",
    WizardStep.SEGMENTS: "Review segments",
    WizardStep.ANALYZING: "Analyzing segment",
    WizardStep.PREVIEW_SEGMENT: "Segment preview",
    WizardStep.READY_TO_RENDER: "Ready to render",
}


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
    schema_version: int = WIZARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step": self.step.value,
            "job_id": self.job_id,
            "error": self.error,
            "segment_index": self.segment_index,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WizardState:
        raw_step = str(data.get("step", WizardStep.INTRO.value))
        version = data.get("schema_version")
        old = not isinstance(version, int) or version < _REUSED_SINCE
        if raw_step in _RETIRED_STEPS or (old and raw_step in _REUSED_STEPS):
            return cls(step=WizardStep.PICK_FILE, error=_RETIRED_MESSAGE)
        index = data.get("segment_index", 0)
        progress = data.get("progress")
        return cls(
            step=WizardStep(raw_step),
            job_id=data.get("job_id"),
            error=data.get("error"),
            segment_index=index if isinstance(index, int) and index >= 0 else 0,
            progress=progress if isinstance(progress, dict) else None,
        )


def segment_loop_enabled() -> bool:
    return os.environ.get(SEGMENT_LOOP_ENV, "") == "1"


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


# On Windows, replacing a file that another thread has open (a page refresh
# reading the state while the render worker saves it) fails with
# PermissionError, and so does opening it mid-replace. Both clear within
# milliseconds, so state reads and writes retry briefly instead of failing the
# request.
_STATE_RETRIES = 20
_STATE_RETRY_S = 0.025


def _read_state_file(path: Path, what: str) -> WizardState | None:
    for attempt in range(_STATE_RETRIES):
        try:
            if not path.is_file():
                return None
            text = path.read_text(encoding="utf-8")
            break
        except PermissionError:
            if attempt == _STATE_RETRIES - 1:
                raise
            time.sleep(_STATE_RETRY_S)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise WizardError(f"corrupt {what}")
    return WizardState.from_dict(data)


def _write_state_file(path: Path, state: WizardState) -> None:
    payload = json.dumps(state.to_dict(), indent=2) + "\n"
    # One temp file per write, so two threads saving at once never share it.
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    try:
        for attempt in range(_STATE_RETRIES):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                if attempt == _STATE_RETRIES - 1:
                    raise
                time.sleep(_STATE_RETRY_S)
    finally:
        tmp.unlink(missing_ok=True)


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


def _fallback_plan(plan: ShotPlan, shots: ShotList, indexes: list[int]) -> ShotPlan:
    """Recipes from the measurements, keeping each shot's preview count."""
    recipes = list(plan.recipes)
    for index in indexes:
        recipes[index] = replace(
            fallback_recipe(shots.shots[index]), rev=recipes[index].rev
        )
    return ShotPlan(scale=plan.scale, recipes=recipes, error=plan.error)


RENDER_REPORT_FILE = "render_report.json"


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


def parse_time(raw: str) -> float:
    """Seconds from ``"75"``, ``"1:15"``, or ``"0:01:15.5"``."""
    text = raw.strip()
    if not text:
        raise ValueError("give a time to split at, like 1:15")
    total = 0.0
    try:
        for part in text.split(":"):
            total = total * 60 + float(part)
    except ValueError as exc:
        raise ValueError(f"could not read the time {text!r}") from exc
    if total < 0 or text.count(":") > 2:
        raise ValueError(f"could not read the time {text!r}")
    return total


def _active_knobs(job_dir: Path) -> tuple[int | None, float | None]:
    plan = load_upscale_plan(job_dir)
    if plan is None or not plan.spans:
        return None, None
    span = plan.pending or plan.spans[-1]
    return span.scale, span.strength


class WizardController:
    """Drive wizard actions against a jobs directory."""

    def __init__(
        self,
        jobs_dir: Path,
        *,
        gemini: GeminiClient | None = None,
        ffmpeg_bin: str = "ffmpeg",
        upscale_engine: UpscaleEngine | None = None,
    ) -> None:
        self.store = JobStore(jobs_dir)
        self.jobs_dir = self.store.root
        self.ffmpeg_bin = ffmpeg_bin
        self._gemini = gemini
        self._upscale = upscale_engine
        # One analyze or render at a time. The HTTP server is threaded, and a
        # reload or a second tab must not start a second GPU pass on the
        # same job directory.
        self._work_lock = threading.Lock()
        self._worker_guard = threading.Lock()
        self._worker: threading.Thread | None = None

    def gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = GeminiClient()
        return self._gemini

    def upscale_engine(self) -> UpscaleEngine:
        if self._upscale is None:
            self._upscale = FlashVsrEngine()
        return self._upscale

    def current_state(self) -> WizardState:
        """Return the active wizard state (job-scoped when a job exists)."""
        session = load_session_state(self.jobs_dir)
        if session.job_id:
            job_dir = self.store.job_dir(session.job_id)
            job_state = load_job_wizard_state(job_dir)
            if job_state is not None:
                return job_state
        return session

    def _state(self) -> WizardState:
        return self.current_state()

    def busy(self) -> bool:
        """True while a shot scan or a render pass is running."""
        return self._work_lock.locked()

    def start_render(self) -> bool:
        """Run :meth:`run_render` on a worker thread. False if not started."""
        return self._start_worker(WizardStep.RENDERING, self.run_render)

    def start_scan(self) -> bool:
        """Run :meth:`run_scan` on a worker thread. False if not started."""
        return self._start_worker(WizardStep.FINDING_SHOTS, self.run_scan)

    def start_plan(self) -> bool:
        """Run :meth:`run_plan` on a worker thread. False if not started."""
        return self._start_worker(WizardStep.PLANNING, self.run_plan)

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
        with self._worker_guard:
            if self._worker is not None and self._worker.is_alive():
                return False
            if self.busy():
                return False
            current = self._state().step
            if current not in steps:
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

    def _run_worker(self, target: Callable[[], None]) -> None:
        try:
            target()
            self._clear_progress()
        except Exception as exc:  # noqa: BLE001 - surfaced on the page
            state = self._state()
            state.error = str(exc) or type(exc).__name__
            if state.step is WizardStep.RENDERING:
                self._render_failed(state, str(exc) or type(exc).__name__)
                return
            if state.step is WizardStep.FINDING_SHOTS:
                self._scan_failed(state, str(exc) or type(exc).__name__)
                return
            if state.step is WizardStep.PLANNING:
                self._plan_failed(state, str(exc) or type(exc).__name__)
                return
            if state.step in _SEGMENTING_STEPS:
                self._segmenting_failed(state, str(exc) or type(exc).__name__)
                return
            if state.step is WizardStep.ANALYZING:
                self._analyze_failed(state, str(exc) or type(exc).__name__)
                return
            self._save(state)

    def _progress(self, state: WizardState, phase: str, pct: int, message: str) -> None:
        """Record how far the running worker is, for ``/api/state``."""
        state.progress = {
            "phase": phase,
            "pct": max(0, min(100, int(pct))),
            "message": message,
        }
        self._save(state)

    def _clear_progress(self) -> None:
        state = self._state()
        if state.progress is not None:
            state.progress = None
            self._save(state)

    def _save(self, state: WizardState) -> None:
        save_session_state(self.jobs_dir, state)
        if state.job_id:
            save_job_wizard_state(self.store.job_dir(state.job_id), state)

    def render(self) -> str:
        state = self._state()
        if state.step is WizardStep.INTRO:
            return render_intro()
        if state.step is WizardStep.LLM_CHOICE:
            return render_llm_choice(error=state.error)
        if state.step is WizardStep.LOCAL_LLM_STUB:
            return render_local_llm_stub()
        if state.step is WizardStep.SETUP:
            return render_setup(
                has_key=bool(resolve_gemini_api_key()), error=state.error
            )
        if state.step is WizardStep.PICK_FILE:
            return render_pick(error=state.error)
        if state.step is WizardStep.FINDING_SHOTS:
            return render_finding_shots()
        if state.step is WizardStep.SHOTS:
            assert state.job_id
            shots = load_shots(self.store.job_dir(state.job_id))
            return render_shots(
                [shot.to_dict() for shot in shots.shots] if shots else [],
                error=state.error,
                label_error=shots.label_error if shots else None,
            )
        if state.step is WizardStep.PLANNING:
            return render_planning()
        if state.step is WizardStep.LOOKS:
            assert state.job_id
            return render_looks(
                self._looks_view(self.store.job_dir(state.job_id)),
                error=state.error,
            )
        if state.step in _LOOP_TITLES:
            return render_segment_loop_pending(_LOOP_TITLES[state.step])
        if state.step is WizardStep.RENDERING:
            return render_rendering()
        if state.step is WizardStep.RESULT:
            assert state.job_id
            job_dir = self.store.job_dir(state.job_id)
            scale, strength = _active_knobs(job_dir)
            report = load_render_report(job_dir)
            return render_result(
                output_url="/output.mp4",
                error=state.error,
                scale=scale,
                strength=strength,
                seams=report["seams"],
                seam_error=report["error"],
            )
        if state.step is WizardStep.DONE:
            return render_done()
        return render_intro()

    def api_view(self) -> dict[str, Any]:
        """JSON snapshot of the current step for a separate UI shell."""
        state = self._state()
        output_ready = False
        output_url: str | None = None
        scale: int | None = None
        strength: float | None = None
        seams: list[dict[str, Any]] = []
        seam_error: str | None = None
        if state.job_id and state.step in {WizardStep.RESULT, WizardStep.DONE}:
            job_dir = self.store.job_dir(state.job_id)
            if (job_dir / "output.mp4").is_file():
                output_ready = True
                output_url = "/output.mp4"
            if state.step is WizardStep.RESULT:
                scale, strength = _active_knobs(job_dir)
                report = load_render_report(job_dir)
                seams, seam_error = report["seams"], report["error"]
        shots: list[dict[str, Any]] | None = None
        label_error: str | None = None
        if state.job_id and state.step is WizardStep.SHOTS:
            found = load_shots(self.store.job_dir(state.job_id))
            if found is not None:
                shots = [shot.to_dict() for shot in found.shots]
                label_error = found.label_error
        looks: dict[str, Any] | None = None
        if state.job_id and state.step in {
            WizardStep.LOOKS,
            WizardStep.READY_TO_RENDER,
        }:
            looks = self._looks_view(self.store.job_dir(state.job_id))
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
            "busy": self.busy(),
            "error": state.error,
            # A worker that died mid-run leaves its last progress behind.
            "progress": state.progress if self.busy() else None,
            "segment_index": state.segment_index,
            "job_id": state.job_id,
            "has_gemini_key": resolve_gemini_api_key() is not None,
            "output_ready": output_ready,
            "output_url": output_url,
            "scale": scale,
            "strength": strength,
            "shots": shots,
            "label_error": label_error,
            "looks": looks,
            "segments": segments,
            "segment_preview": segment_preview,
            "seams": seams,
            "seam_error": seam_error,
        }

    def handle_post(self, path: str, fields: dict[str, list[str]]) -> None:
        if self.busy():
            raise WizardError("the last step is still running; wait for it")
        state = self._state()
        state.error = None
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
        if path == "/new":
            self.start_new_job()
            return
        if path == "/shots":
            self._shots_post(state, fields)
            return
        if path == "/looks":
            self._looks_post(state, fields)
            return
        if path == "/segments":
            self._segments_post(state, fields)
            return
        if path == "/preview":
            self._preview_post(state, fields)
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
        """A render failed. Keep a previous result if there is one.

        When a revise fails, the earlier ``output.mp4`` is still good, so the
        user stays on the result screen with the error. When the first upscale
        fails there is nothing to show; the job is marked failed and the user
        goes back to the pick screen to fix the cause or choose another file.
        """
        assert state.job_id
        job_id = state.job_id
        job_dir = self.store.job_dir(job_id)
        if (job_dir / "output.mp4").is_file():
            state.error = detail
            state.step = WizardStep.RESULT
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

    def _pick_file(self, state: WizardState, raw_path: str) -> None:
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
            manifest = self.store.create(source)
            self.store.transition(manifest.job_id, JobState.PROBED)
        except (JobError, ProbeError) as exc:
            state.error = str(exc)
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        state.job_id = manifest.job_id
        state.step = (
            WizardStep.SAMPLING if segment_loop_enabled() else WizardStep.FINDING_SHOTS
        )
        self._save(state)

    def run_scan(self) -> None:
        if not self._work_lock.acquire(blocking=False):
            return
        try:
            state = self._state()
            if state.step is WizardStep.FINDING_SHOTS and state.job_id:
                self._scan(state)
        finally:
            self._work_lock.release()

    def _scan(self, state: WizardState) -> None:
        """Loop 1: find and measure the shots, then ask Gemini to label them.

        A labelling failure is not fatal: the shots keep the hints from the
        measurements, and the review screen says why the labels are missing.
        """
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        source = Path(self.store.load(state.job_id).source_path)
        self._progress(state, "finding_shots", 5, "Finding and measuring shots")
        try:
            duration = float(probe_file(source).duration_s or 0.0)
            shots = find_shots(source, job_dir, duration, ffmpeg_bin=self.ffmpeg_bin)
        except (ShotError, ProbeError, OSError) as exc:
            self._scan_failed(state, str(exc))
            return
        self._progress(state, "finding_shots", 60, "Gemini is labelling the shots")
        try:
            labels = self.gemini().label_shots(
                shots=[
                    {
                        "start_s": round(shot.start_s, 2),
                        "end_s": round(shot.end_s, 2),
                        "metrics": shot.metrics.to_dict(),
                        "hints": list(shot.issues),
                    }
                    for shot in shots.shots
                ],
                stills=[job_dir / shot.still for shot in shots.shots],
                issues=ISSUES,
                contains=CONTENTS,
                job_dir=job_dir,
            )
            shots = apply_labels(shots, labels)
        except GeminiError as exc:
            shots.label_error = f"Gemini could not label the shots: {exc}"
        save_shots(job_dir, shots)
        state.error = None
        state.step = WizardStep.SHOTS
        self._save(state)

    def _scan_failed(self, state: WizardState, detail: str) -> None:
        """Finding the shots failed; mark the job failed and go back to pick."""
        assert state.job_id
        job_id = state.job_id
        with contextlib.suppress(JobError):
            self.store.transition(job_id, JobState.FAILED)
        message = f"Finding the shots failed: {detail}"
        save_job_wizard_state(
            self.store.job_dir(job_id),
            WizardState(step=WizardStep.PICK_FILE, job_id=job_id, error=message),
        )
        self._save(WizardState(step=WizardStep.PICK_FILE, error=message))

    def _shots_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        """Approve the shot list, or merge or split a shot and stay on it."""
        if state.step is not WizardStep.SHOTS or not state.job_id:
            raise WizardError("there is no shot list to change")
        job_dir = self.store.job_dir(state.job_id)
        shots = load_shots(job_dir)
        if shots is None:
            raise WizardError("the shot list is missing; pick the file again")
        action = fields.get("action", [""])[0]
        try:
            if action == "approve":
                source = Path(self.store.load(state.job_id).source_path)
                video = probe_file(source).primary_video()
                scales = allowed_scales(
                    (video.width or 0) if video else 0,
                    (video.height or 0) if video else 0,
                )
                save_plan(job_dir, new_plan(shots, max(scales)))
                state.step = WizardStep.PLANNING
                self._save(state)
                return
            index = int(fields.get("index", [""])[0])
            if action == "merge":
                shots = merge_with_next(shots, index)
            elif action == "split":
                source = Path(self.store.load(state.job_id).source_path)
                shots = split_at(
                    shots,
                    index,
                    parse_time(fields.get("at", [""])[0]),
                    source=source,
                    job_dir=job_dir,
                    ffmpeg_bin=self.ffmpeg_bin,
                )
            else:
                raise ValueError(f"unknown shot action {action!r}")
            save_shots(job_dir, shots)
        except (ShotError, ValueError, UpscaleError, ProbeError) as exc:
            state.error = str(exc)
        self._save(state)

    def run_plan(self) -> None:
        if not self._work_lock.acquire(blocking=False):
            return
        try:
            state = self._state()
            if state.step is WizardStep.PLANNING and state.job_id:
                self._plan(state)
        finally:
            self._work_lock.release()

    def _plan(self, state: WizardState) -> None:
        """Loop 2: propose recipes, render previews, and self-check them.

        Only shots marked stale are worked on: all of them the first time,
        one after an editor note. Gemini failures fall back to recipes from
        the measurements and skip the check; the looks screen says so.
        """
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        source = Path(self.store.load(state.job_id).source_path)
        shots = load_shots(job_dir)
        plan = load_plan(job_dir)
        if shots is None or plan is None:
            self._plan_failed(state, "the shot list is missing; pick the file again")
            return
        indexes = plan.stale_indexes()
        if not indexes:
            state.step = WizardStep.LOOKS
            self._save(state)
            return
        self._work_on(
            state,
            job_dir,
            source,
            shots,
            plan,
            indexes,
            phase="planning",
            done=WizardStep.LOOKS,
            failed=self._plan_failed,
        )

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
        # The scale is chosen once, on the first pass; every shot shares it.
        first = not any(recipe.reason for recipe in plan.recipes)
        scales = tuple(sorted({2, plan.scale})) if first else (plan.scale,)
        if first:
            self._progress(state, phase, 10, "Boxing faces, hands, and text")
            shots = self._locate(shots, source, job_dir, warnings)
        self._progress(state, phase, 30, "Gemini is proposing settings")
        try:
            reply = self.gemini().propose_looks(
                shots=shots_payload(shots, plan, indexes),
                stills=[job_dir / shots.shots[i].still for i in indexes],
                bounds=LOOK_BOUNDS,
                scales=scales,
                protected_strength=PROTECTED_STRENGTH,
                job_dir=job_dir,
            )
            scale = reply.get("scale")
            plan.scale = scale if first and scale in scales else min(scales)
            plan = apply_proposals(plan, shots, list(reply["shots"]), indexes)
        except GeminiError as exc:
            warnings.append(f"Gemini could not propose settings ({exc})")
            plan.scale = min(scales)
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
        """Box faces, hands, and text so the blend can hold FlashVSR down there."""
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

    def _plan_failed(self, state: WizardState, detail: str) -> None:
        """Planning failed (usually the GPU); go back to the shot list."""
        state.error = f"The previews failed: {detail}"
        state.step = WizardStep.SHOTS
        self._save(state)

    def _looks_view(self, job_dir: Path) -> dict[str, Any]:
        shots = load_shots(job_dir)
        plan = load_plan(job_dir)
        if shots is None or plan is None:
            return {"scale": None, "error": None, "shots": []}
        items: list[dict[str, Any]] = []
        for index, (shot, recipe) in enumerate(
            zip(shots.shots, plan.recipes, strict=False)
        ):
            before, after = (
                f"/previews/{name}"
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
                    "before_url": before if recipe.rev else None,
                    "after_url": after if recipe.rev else None,
                }
            )
        return {"scale": plan.scale, "error": plan.error, "shots": items}

    def _looks_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        """Approve every shot's settings, or send one shot back with a note."""
        if state.step is not WizardStep.LOOKS or not state.job_id:
            raise WizardError("there are no shot settings to change")
        job_dir = self.store.job_dir(state.job_id)
        shots = load_shots(job_dir)
        plan = load_plan(job_dir)
        if shots is None or plan is None:
            raise WizardError("the shot settings are missing; pick the file again")
        action = fields.get("action", [""])[0]
        try:
            if action == "approve":
                save_upscale_plan(job_dir, plan.to_upscale_plan(shots))
                state.step = WizardStep.RENDERING
            elif action == "redo":
                index = int(fields.get("index", [""])[0])
                save_plan(
                    job_dir, request_redo(plan, index, fields.get("note", [""])[0])
                )
                state.step = WizardStep.PLANNING
            else:
                raise ValueError(f"unknown action {action!r}")
        except (ShotPlanError, ValueError) as exc:
            state.error = str(exc)
        self._save(state)

    # Segment loop -----------------------------------------------------------

    def run_segmenting(self) -> None:
        if not self._work_lock.acquire(blocking=False):
            return
        try:
            state = self._state()
            if state.job_id and state.step is WizardStep.SAMPLING:
                self._sample(state)
            state = self._state()
            if state.job_id and state.step is WizardStep.PROPOSING:
                self._propose_segments(state)
        finally:
            self._work_lock.release()

    def _sample(self, state: WizardState) -> None:
        """Save the still grid Gemini cuts the video from."""
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        source = Path(self.store.load(state.job_id).source_path)
        self._progress(state, "sampling", 5, "Saving stills from the video")
        try:
            duration = float(probe_file(source).duration_s or 0.0)
            extract_segment_stills(
                source, job_dir, sample_times(duration), ffmpeg_bin=self.ffmpeg_bin
            )
        except (SegmentLoopError, ProbeError, ValueError, OSError) as exc:
            self._segmenting_failed(state, str(exc))
            return
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
        self._progress(state, "proposing", 50, "Gemini is proposing segments")
        error: str | None = None
        try:
            raw = self.gemini().propose_segments(
                stills=[job_dir / still.path for still in stills],
                times=[still.t_s for still in stills],
                duration_s=duration,
                contains=CONTENTS,
                note=earlier.note if earlier else "",
                job_dir=job_dir,
            )
            segments = repair_coverage(raw, duration)
        except (GeminiError, SegmentLoopError) as exc:
            error = (
                f"Gemini could not propose segments ({exc}); the whole video is one."
            )
            segments = [LoopSegment(0.0, duration, label="whole video")]
        save_segment_list(job_dir, SegmentList(segments=segments, error=error))
        state.error = None
        state.step = WizardStep.SEGMENTS
        self._save(state)

    def _segmenting_failed(self, state: WizardState, detail: str) -> None:
        """Sampling or proposing failed; mark the job failed and go back to pick."""
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

    def _segments_view(self, job_dir: Path) -> dict[str, Any]:
        try:
            found = load_segment_list(job_dir)
        except SegmentLoopError as exc:
            return {"segments": [], "error": str(exc), "note": ""}
        if found is None:
            return {"segments": [], "error": None, "note": ""}
        return {
            "segments": [segment.to_dict() for segment in found.segments],
            "error": found.error,
            "note": found.note,
        }

    def _segments_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        """Approve the segments, or send them back to Gemini with a note."""
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
        if action == "approve":
            if not found.approved:
                # New segments: plan them from the start.
                found.approved = True
                save_segment_list(job_dir, found)
                shots_path(job_dir).unlink(missing_ok=True)
                plan_path(job_dir).unlink(missing_ok=True)
                state.segment_index = 0
            state.step = WizardStep.ANALYZING
        elif action == "note":
            text = " ".join(fields.get("note", [""])[0].split())
            if not text:
                state.error = "say what to change about the segments"
            else:
                found.note = text[:500]
                found.approved = False
                save_segment_list(job_dir, found)
                state.step = WizardStep.PROPOSING
        else:
            state.error = f"unknown action {action!r}"
        self._save(state)

    def run_analyze(self) -> None:
        if not self._work_lock.acquire(blocking=False):
            return
        try:
            state = self._state()
            if state.step is WizardStep.ANALYZING and state.job_id:
                self._analyze(state)
        finally:
            self._work_lock.release()

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
        if shots is None or plan is None:
            self._progress(state, "analyzing", 5, "Measuring the segments")
            try:
                shots = shots_from_segments(
                    source, job_dir, found.segments, ffmpeg_bin=self.ffmpeg_bin
                )
                video = probe_file(source).primary_video()
            except (SegmentLoopError, ProbeError) as exc:
                self._analyze_failed(state, str(exc))
                return
            save_shots(job_dir, shots)
            scales = allowed_scales(
                (video.width or 0) if video else 0,
                (video.height or 0) if video else 0,
            )
            plan = new_plan(shots, max(scales))
            save_plan(job_dir, plan)
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
                state.segment_index += 1
                state.step = (
                    WizardStep.READY_TO_RENDER
                    if state.segment_index >= len(plan.recipes)
                    else WizardStep.ANALYZING
                )
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
        try:
            save_upscale_plan(job_dir, plan.to_upscale_plan(shots))
            state.step = WizardStep.RENDERING
        except ShotPlanError as exc:
            state.error = str(exc)
        self._save(state)

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
        source = Path(self.store.load(state.job_id).source_path)
        try:
            facts = probe_file(source)
            duration = float(facts.duration_s or 0.0)
            plan = load_upscale_plan(job_dir) or UpscalePlan(
                spans=(default_span(duration),)
            )
            prior = plan.spans[-1]
            raw = self.gemini().revise_upscale_params(
                note=note,
                prior_scale=prior.scale,
                prior_strength=prior.strength,
                prior_start_s=prior.start_s,
                prior_end_s=prior.end_s,
                duration_s=duration,
                job_dir=job_dir,
            )
            patch = UpscaleSpan.from_dict(raw)
            # A note retunes scale and strength; the shot keeps its look and
            # its protected regions.
            under = span_at(plan, patch.start_s)
            if under is not None:
                patch = replace(patch, look=under.look, regions=under.regions)
            updated = replace_overlapping(plan, patch, duration)
            save_upscale_plan(job_dir, updated)
            state.error = None
            state.step = WizardStep.RENDERING
        except (UpscaleError, GeminiError, ProbeError, OSError, ValueError) as exc:
            state.error = str(exc)
            state.step = WizardStep.RESULT
        self._save(state)

    def run_render(self) -> None:
        if not self._work_lock.acquire(blocking=False):
            return
        try:
            self._run_render()
        finally:
            self._work_lock.release()

    def _run_render(self) -> None:
        state = self._state()
        if state.step is not WizardStep.RENDERING or not state.job_id:
            return
        self._render_upscale(state, self.store.job_dir(state.job_id))

    def _render_upscale(self, state: WizardState, job_dir: Path) -> None:
        assert state.job_id
        source = Path(self.store.load(state.job_id).source_path)
        output = job_dir / "output.mp4"
        self._progress(state, "rendering", 5, "Upscaling on the GPU")
        try:
            facts = probe_file(source)
            duration = float(facts.duration_s or 0.0)
            plan = load_upscale_plan(job_dir)
            if plan is None:
                plan = UpscalePlan(spans=(default_span(duration),))
                save_upscale_plan(job_dir, plan)
            render_plan(
                source,
                output,
                plan,
                engine=self.upscale_engine(),
                ffmpeg_bin=self.ffmpeg_bin,
                duration_s=duration,
            )
            # Candidates include the pending revise's edges, so read them
            # before the pending span is cleared.
            candidates = seam_candidates(plan, duration)
            plan = UpscalePlan(spans=plan.spans, pending=None)
            save_upscale_plan(job_dir, plan)
            self._progress(state, "rendering", 90, "Checking for seams")
            self._check_seams(job_dir, output, source, candidates)
            state.error = None
            state.step = WizardStep.RESULT
        except (UpscaleError, ProbeError, OSError, ValueError) as exc:
            self._render_failed(state, str(exc))
            return
        self._save(state)
