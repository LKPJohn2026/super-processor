"""Localhost wizard: setup → pick → shots → FlashVSR upscale → result → revise."""

from __future__ import annotations

import contextlib
import json
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
from .probe import MediaFacts, ProbeError, probe_file
from .segments import MAX_DURATION_S
from .shots import (
    CONTENTS,
    ISSUES,
    ShotError,
    apply_labels,
    find_shots,
    load_shots,
    merge_with_next,
    save_shots,
    split_at,
)
from .upscale import (
    FlashVsrEngine,
    UpscaleEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleSpan,
    default_span,
    load_upscale_plan,
    look_at,
    render_plan,
    replace_overlapping,
    save_upscale_plan,
)
from .wizard_pages import (
    render_done,
    render_finding_shots,
    render_intro,
    render_llm_choice,
    render_local_llm_stub,
    render_pick,
    render_rendering,
    render_result,
    render_setup,
    render_shots,
)

WIZARD_STATE_FILE = "wizard_state.json"
WIZARD_SCHEMA_VERSION = 2
# Inputs are limited to 1080p (either orientation) and 30 minutes. FlashVSR
# at 2x already makes 4K from 1080p; larger inputs exceed what one GPU pass
# and the job directory are sized for.
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
    RENDERING = "rendering"
    RESULT = "result"
    DONE = "done"


# Steps of the removed Gemini split / FFmpeg enhance flow. A state file saved
# on one of them is sent back to the pick step.
_RETIRED_STEPS = frozenset({"analyzing", "overview", "choose_split", "enhance"})
_RETIRED_MESSAGE = (
    "this job used the removed split and enhance flow; pick the file again"
)


@dataclass(slots=True)
class WizardState:
    """On-disk wizard cursor for one job (or pre-job session)."""

    step: WizardStep = WizardStep.INTRO
    job_id: str | None = None
    error: str | None = None
    schema_version: int = WIZARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step": self.step.value,
            "job_id": self.job_id,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WizardState:
        raw_step = str(data.get("step", WizardStep.INTRO.value))
        if raw_step in _RETIRED_STEPS:
            return cls(step=WizardStep.PICK_FILE, error=_RETIRED_MESSAGE)
        return cls(
            step=WizardStep(raw_step),
            job_id=data.get("job_id"),
            error=data.get("error"),
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

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until the worker finishes. True when nothing is running."""
        with self._worker_guard:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
        return not self.busy()

    def _start_worker(self, step: WizardStep, target: Callable[[], None]) -> bool:
        with self._worker_guard:
            if self._worker is not None and self._worker.is_alive():
                return False
            if self.busy():
                return False
            if self._state().step is not step:
                return False
            worker = threading.Thread(
                target=self._run_worker,
                args=(target,),
                name=f"wizard-{step.value}",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return True

    def _run_worker(self, target: Callable[[], None]) -> None:
        try:
            target()
        except Exception as exc:  # noqa: BLE001 - surfaced on the page
            state = self._state()
            state.error = str(exc) or type(exc).__name__
            if state.step is WizardStep.RENDERING:
                self._render_failed(state, str(exc) or type(exc).__name__)
                return
            if state.step is WizardStep.FINDING_SHOTS:
                self._scan_failed(state, str(exc) or type(exc).__name__)
                return
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
        if state.step is WizardStep.RENDERING:
            return render_rendering()
        if state.step is WizardStep.RESULT:
            assert state.job_id
            scale, strength = _active_knobs(self.store.job_dir(state.job_id))
            return render_result(
                output_url="/output.mp4",
                error=state.error,
                scale=scale,
                strength=strength,
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
        if state.job_id and state.step in {WizardStep.RESULT, WizardStep.DONE}:
            job_dir = self.store.job_dir(state.job_id)
            if (job_dir / "output.mp4").is_file():
                output_ready = True
                output_url = "/output.mp4"
            if state.step is WizardStep.RESULT:
                scale, strength = _active_knobs(job_dir)
        shots: list[dict[str, Any]] | None = None
        label_error: str | None = None
        if state.job_id and state.step is WizardStep.SHOTS:
            found = load_shots(self.store.job_dir(state.job_id))
            if found is not None:
                shots = [shot.to_dict() for shot in found.shots]
                label_error = found.label_error
        return {
            "step": state.step.value,
            "busy": self.busy(),
            "error": state.error,
            "job_id": state.job_id,
            "has_gemini_key": resolve_gemini_api_key() is not None,
            "output_ready": output_ready,
            "output_url": output_url,
            "scale": scale,
            "strength": strength,
            "shots": shots,
            "label_error": label_error,
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
        state.step = WizardStep.FINDING_SHOTS
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
        try:
            duration = float(probe_file(source).duration_s or 0.0)
            shots = find_shots(source, job_dir, duration, ffmpeg_bin=self.ffmpeg_bin)
        except (ShotError, ProbeError, OSError) as exc:
            self._scan_failed(state, str(exc))
            return
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
                save_upscale_plan(job_dir, shots.to_plan())
                state.step = WizardStep.RENDERING
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
        except (ShotError, ValueError, UpscaleError) as exc:
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
            # A note retunes scale and strength; the shot keeps its look.
            patch = replace(patch, look=look_at(plan, patch.start_s))
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
            plan = UpscalePlan(spans=plan.spans, pending=None)
            save_upscale_plan(job_dir, plan)
            state.error = None
            state.step = WizardStep.RESULT
        except (UpscaleError, ProbeError, OSError, ValueError) as exc:
            self._render_failed(state, str(exc))
            return
        self._save(state)
