"""Localhost Gemini wizard: setup → overview → capped restore and RTX VSR."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .gemini import (
    FRAME_CAP,
    WIZARD_FRAMES_DIR,
    EnhanceOption,
    GeminiClient,
    GeminiError,
    SplitLayout,
    SplitProposal,
    StructuredOp,
    frame_timestamps,
    resolve_gemini_api_key,
    store_gemini_api_key,
)
from .jobs import JobError, JobState, JobStore
from .probe import ProbeError, parse_frame_rate, probe_file
from .recipe import DEFAULT_OP_ORDER, OpName
from .segments import STILLS_DIR_NAME, TimelineSegment
from .treatments import Treatment, TreatmentStep
from .upscale import UpscaleError, UpscaleParams, enforce_note_direction, run_two_pass
from .wizard_pages import (
    render_analyzing,
    render_done,
    render_intro,
    render_llm_choice,
    render_local_llm_stub,
    render_overview,
    render_pick,
    render_rendering,
    render_result,
    render_setup,
)

WIZARD_STATE_FILE = "wizard_state.json"
SPLIT_LAYOUTS_FILE = "split_layouts.json"
SEGMENT_CHOICES_FILE = "segment_choices.json"
WIZARD_SCHEMA_VERSION = 1


class WizardError(RuntimeError):
    """Raised when a wizard step cannot complete."""


class WizardStep(str, Enum):
    """Persisted wizard screen."""

    INTRO = "intro"
    LLM_CHOICE = "llm_choice"
    LOCAL_LLM_STUB = "local_llm_stub"
    SETUP = "setup"
    PICK_FILE = "pick_file"
    ANALYZING = "analyzing"
    OVERVIEW = "overview"
    CHOOSE_SPLIT = "choose_split"
    ENHANCE = "enhance"
    RENDERING = "rendering"
    RESULT = "result"
    DONE = "done"


@dataclass(slots=True)
class WizardState:
    """On-disk wizard cursor for one job (or pre-job session)."""

    step: WizardStep = WizardStep.INTRO
    job_id: str | None = None
    layout_index: int | None = None
    segment_index: int = 0
    highlights: list[str] = field(default_factory=list)
    enhance_cache: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    schema_version: int = WIZARD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "step": self.step.value,
            "job_id": self.job_id,
            "layout_index": self.layout_index,
            "segment_index": self.segment_index,
            "highlights": list(self.highlights),
            "enhance_cache": dict(self.enhance_cache),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WizardState:
        return cls(
            step=WizardStep(str(data.get("step", WizardStep.INTRO.value))),
            job_id=data.get("job_id"),
            layout_index=data.get("layout_index"),
            segment_index=int(data.get("segment_index", 0)),
            highlights=[str(item) for item in (data.get("highlights") or [])],
            enhance_cache=dict(data.get("enhance_cache") or {}),
            error=data.get("error"),
            schema_version=int(data.get("schema_version", WIZARD_SCHEMA_VERSION)),
        )


def session_state_path(jobs_dir: Path) -> Path:
    """Wizard state before a job exists lives under the jobs root."""
    return jobs_dir / "wizard_session.json"


def job_state_path(job_dir: Path) -> Path:
    return job_dir / WIZARD_STATE_FILE


def load_session_state(jobs_dir: Path) -> WizardState:
    path = session_state_path(jobs_dir)
    if not path.is_file():
        return WizardState()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WizardError("corrupt wizard session")
    return WizardState.from_dict(data)


def save_session_state(jobs_dir: Path, state: WizardState) -> None:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    path = session_state_path(jobs_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_job_wizard_state(job_dir: Path) -> WizardState | None:
    path = job_state_path(job_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WizardError("corrupt wizard_state.json")
    return WizardState.from_dict(data)


def save_job_wizard_state(job_dir: Path, state: WizardState) -> None:
    path = job_state_path(job_dir)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


_ORDER = {name: index for index, name in enumerate(DEFAULT_OP_ORDER)}


def treatment_from_ops(
    ops: list[StructuredOp],
    *,
    treatment_id: str,
    problem: str,
) -> Treatment:
    """Build a Treatment from wizard structured ops."""
    steps: list[TreatmentStep] = []
    for item in ops:
        try:
            op = OpName(item.op)
        except ValueError as exc:
            raise WizardError(f"unknown op {item.op!r}") from exc
        steps.append(TreatmentStep(op, tuple(sorted(item.params.items()))))
    steps.sort(key=lambda step: _ORDER[step.op])
    return Treatment(treatment_id, problem, tuple(steps))


def _map_problem(issues: list[str]) -> str:
    aliases = {
        "low_light": "low_light",
        "lowlight": "low_light",
        "dark": "low_light",
        "low_contrast": "low_contrast",
        "flat": "low_contrast",
        "silhouette": "silhouette",
        "too_warm": "too_warm",
        "warm": "too_warm",
        "noisy": "noisy",
        "noise": "noisy",
        "grain": "noisy",
        "shaky": "shaky",
        "shake": "shaky",
        "blurry": "noisy",
        "soft": "noisy",
        "off_center": "off_center",
    }
    for issue in issues:
        key = re.sub(r"[^a-z0-9]+", "_", issue.lower()).strip("_")
        if key in aliases:
            return aliases[key]
    return "noisy"


def layout_to_segments(
    layout: SplitLayout,
    *,
    still_dir: str = STILLS_DIR_NAME,
) -> list[TimelineSegment]:
    """Convert a Gemini layout into TimelineSegment rows."""
    segments: list[TimelineSegment] = []
    for index, proposal in enumerate(layout.segments):
        problem = _map_problem(proposal.issues)
        key = (proposal.start_s + proposal.end_s) / 2.0
        still = f"{still_dir}/seg_{index:02d}.ppm"
        segments.append(
            TimelineSegment(
                index=index,
                start_s=proposal.start_s,
                end_s=proposal.end_s,
                context="mixed",
                problem=problem,
                keyframe_s=key,
                look_group=index,
                still_path=still,
            )
        )
    return segments


def extract_wizard_frames(
    source: Path,
    job_dir: Path,
    *,
    duration_s: float,
    ffmpeg_bin: str = "ffmpeg",
) -> list[Path]:
    """Extract JPEG frames at dynamic-FPS timestamps for Gemini."""
    stamps = frame_timestamps(duration_s, frame_cap=FRAME_CAP)
    out_dir = job_dir / WIZARD_FRAMES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, at in enumerate(stamps):
        dest = out_dir / f"f_{index:03d}.jpg"
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{at:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-q:v",
            "5",
            str(dest),
        ]
        completed = subprocess.run(cmd, check=False, capture_output=True)
        if completed.returncode == 0 and dest.is_file():
            paths.append(dest)
    if not paths:
        raise WizardError("could not extract sample frames for Gemini")
    return paths


def write_split_layouts(job_dir: Path, proposal: SplitProposal) -> Path:
    path = job_dir / SPLIT_LAYOUTS_FILE
    path.write_text(json.dumps(proposal.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_split_layouts(job_dir: Path) -> SplitProposal:
    path = job_dir / SPLIT_LAYOUTS_FILE
    if not path.is_file():
        raise WizardError("split layouts missing; re-run analysis")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WizardError("corrupt split_layouts.json")
    return SplitProposal.from_dict(data)


def load_segment_choices(job_dir: Path) -> dict[str, Any]:
    path = job_dir / SEGMENT_CHOICES_FILE
    if not path.is_file():
        return {"choices": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise WizardError("corrupt segment_choices.json")
    return data


def save_segment_choice(
    job_dir: Path,
    segment_index: int,
    option: EnhanceOption,
) -> None:
    data = load_segment_choices(job_dir)
    choices = data.setdefault("choices", {})
    if not isinstance(choices, dict):
        choices = {}
        data["choices"] = choices
    choices[str(segment_index)] = option.to_dict()
    path = job_dir / SEGMENT_CHOICES_FILE
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def params_from_state(state: WizardState) -> UpscaleParams:
    """Return saved upscale knobs, or the capped defaults."""
    raw = state.enhance_cache.get("upscale")
    if isinstance(raw, dict):
        return UpscaleParams.from_dict(raw)
    return UpscaleParams()


class WizardController:
    """Drive wizard actions against a jobs directory."""

    def __init__(
        self,
        jobs_dir: Path,
        *,
        gemini: GeminiClient | None = None,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        self.store = JobStore(jobs_dir)
        self.jobs_dir = self.store.root
        self.ffmpeg_bin = ffmpeg_bin
        self._gemini = gemini

    def gemini(self) -> GeminiClient:
        if self._gemini is None:
            self._gemini = GeminiClient()
        return self._gemini

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
        if state.step is WizardStep.ANALYZING:
            return render_analyzing()
        if state.step is WizardStep.OVERVIEW:
            assert state.job_id
            facts = probe_file(Path(self.store.load(state.job_id).source_path))
            return render_overview(facts=facts, highlights=state.highlights)
        if state.step in {
            WizardStep.CHOOSE_SPLIT,
            WizardStep.ENHANCE,
            WizardStep.RENDERING,
        }:
            params = params_from_state(state)
            return render_rendering(
                restore_strength=params.restore_strength,
                scale=params.scale,
                vsr_quality=params.vsr_quality,
            )
        if state.step is WizardStep.RESULT:
            assert state.job_id
            params = params_from_state(state)
            return render_result(
                output_url="/output.mp4",
                error=state.error,
                restore_strength=params.restore_strength,
                scale=params.scale,
                vsr_quality=params.vsr_quality,
            )
        if state.step is WizardStep.DONE:
            return render_done()
        return render_intro()

    def handle_post(self, path: str, fields: dict[str, list[str]]) -> None:
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
        if path == "/overview":
            if "upscale" not in state.enhance_cache:
                state.enhance_cache["upscale"] = UpscaleParams().to_dict()
            state.step = WizardStep.RENDERING
            self._save(state)
            return
        if path == "/result":
            self._result_post(state, fields)
            return
        raise WizardError(f"unknown action {path}")

    def _pick_file(self, state: WizardState, raw_path: str) -> None:
        source = Path(raw_path.strip()).expanduser()
        if not source.is_file():
            state.error = f"file not found: {source}"
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        try:
            manifest = self.store.create(source)
            probe_file(source)
            self.store.transition(manifest.job_id, JobState.PROBED)
        except (JobError, ProbeError) as exc:
            state.error = str(exc)
            state.step = WizardStep.PICK_FILE
            self._save(state)
            return
        state.job_id = manifest.job_id
        state.step = WizardStep.ANALYZING
        self._save(state)

    def run_analyze(self) -> None:
        state = self._state()
        if state.step is not WizardStep.ANALYZING or not state.job_id:
            return
        job_id = state.job_id
        job_dir = self.store.job_dir(job_id)
        source = Path(self.store.load(job_id).source_path)
        try:
            facts = probe_file(source)
            duration = float(facts.duration_s or 0.0)
            if duration <= 0:
                raise WizardError("media has no duration")
            frames = extract_wizard_frames(
                source,
                job_dir,
                duration_s=duration,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            # Cap frames sent to Gemini for cost; still store all extracted.
            send = frames[:: max(1, len(frames) // 24)][:24]
            proposal = self.gemini().propose_splits(
                frame_paths=send,
                duration_s=duration,
                job_dir=job_dir,
            )
            write_split_layouts(job_dir, proposal)
            state.highlights = proposal.highlights
            state.step = WizardStep.OVERVIEW
            if self.store.load(job_id).state is JobState.PROBED:
                self.store.transition(job_id, JobState.SPLIT_PROPOSED)
        except (GeminiError, WizardError, ProbeError, JobError, OSError) as exc:
            state.error = str(exc)
            state.step = WizardStep.PICK_FILE
        self._save(state)

    def _result_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        mood = fields.get("mood", [""])[0]
        if mood == "happy":
            state.step = WizardStep.DONE
            self._save(state)
            return
        if mood != "revise":
            state.error = "choose happy or describe a change"
            state.step = WizardStep.RESULT
            self._save(state)
            return
        note = fields.get("note", [""])[0].strip()
        if not note:
            state.error = "describe what to change"
            state.step = WizardStep.RESULT
            self._save(state)
            return
        if not state.job_id:
            raise WizardError("no job")
        prior = params_from_state(state)
        source = Path(self.store.load(state.job_id).source_path)
        try:
            facts = probe_file(source)
            duration = float(facts.duration_s or 0.0)
            raw = self.gemini().propose_upscale_params(
                prior=prior.to_dict(),
                user_note=note,
                duration_s=duration,
                job_dir=self.store.job_dir(state.job_id),
            )
            proposed = UpscaleParams.from_dict(raw)
            checked = enforce_note_direction(prior, proposed, note)
        except (GeminiError, UpscaleError, ProbeError) as exc:
            state.error = str(exc)
            state.step = WizardStep.RESULT
            self._save(state)
            return
        state.enhance_cache["upscale"] = checked.to_dict()
        state.enhance_cache["upscale_note"] = note
        state.error = None
        state.step = WizardStep.RENDERING
        self._save(state)

    def _advance_to_encoding(self, job_id: str) -> None:
        current = self.store.load(job_id).state
        order = (
            JobState.SPLIT_PROPOSED,
            JobState.SPLIT_ACCEPTED,
            JobState.PLANS_READY,
            JobState.PLAN_SELECTED,
            JobState.ENCODING,
        )
        while current in order and current is not JobState.ENCODING:
            nxt = order[order.index(current) + 1]
            self.store.transition(job_id, nxt)
            current = nxt

    def run_render(self) -> None:
        state = self._state()
        if state.step is not WizardStep.RENDERING or not state.job_id:
            return
        job_id = state.job_id
        job_dir = self.store.job_dir(job_id)
        params = params_from_state(state)
        try:
            source = Path(self.store.load(job_id).source_path)
            facts = probe_file(source)
            video = facts.primary_video()
            if video is None or not video.width or not video.height:
                raise WizardError("source has no video stream")
            fps = parse_frame_rate(video.avg_frame_rate) or 30.0
            self._advance_to_encoding(job_id)
            run_two_pass(
                source,
                job_dir / "output.mp4",
                params,
                ffmpeg_bin=self.ffmpeg_bin,
                width=video.width,
                height=video.height,
                fps=fps,
            )
            if self.store.load(job_id).state is JobState.ENCODING:
                self.store.transition(job_id, JobState.COMPLETE)
            state.error = None
            state.step = WizardStep.RESULT
        except (WizardError, UpscaleError, ProbeError, JobError, OSError) as exc:
            state.error = str(exc)
            try:
                if self.store.load(job_id).state not in {
                    JobState.COMPLETE,
                    JobState.FAILED,
                }:
                    self.store.transition(job_id, JobState.FAILED)
            except JobError:
                pass
            state.step = WizardStep.RESULT
        self._save(state)
