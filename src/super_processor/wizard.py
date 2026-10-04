"""Localhost Gemini wizard: setup → split → enhance → concat."""

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
    SegmentEnhanceResult,
    SplitLayout,
    SplitProposal,
    StructuredOp,
    frame_timestamps,
    resolve_gemini_api_key,
    store_gemini_api_key,
)
from .jobs import JobError, JobState, JobStore
from .preview import PreviewError, encode_preview
from .probe import ProbeError, probe_file
from .recipe import DEFAULT_OP_ORDER, OpName
from .render import RenderError, render_chosen_plan
from .segments import (
    STILLS_DIR_NAME,
    TimelineSegment,
    write_gray_still,
    write_segments,
)
from .treatments import Treatment, TreatmentStep
from .wizard_pages import (
    render_analyzing,
    render_done,
    render_enhance,
    render_intro,
    render_llm_choice,
    render_local_llm_stub,
    render_overview,
    render_pick,
    render_rendering,
    render_result,
    render_setup,
    render_split_choice,
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


def prepare_segment_stills(
    source: Path,
    job_dir: Path,
    segments: list[TimelineSegment],
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> None:
    """Write gray stills for the review UI from keyframes."""
    from .estimators import extract_gray_frame

    for segment in segments:
        gray = extract_gray_frame(
            source,
            at_s=segment.keyframe_s,
            ffmpeg_bin=ffmpeg_bin,
        )
        write_gray_still(job_dir / segment.still_path, gray)


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
        if state.step is WizardStep.CHOOSE_SPLIT:
            assert state.job_id
            proposal = load_split_layouts(self.store.job_dir(state.job_id))
            return render_split_choice(proposal=proposal, error=state.error)
        if state.step is WizardStep.ENHANCE:
            return self._render_enhance(state)
        if state.step is WizardStep.RENDERING:
            return render_rendering()
        if state.step is WizardStep.RESULT:
            assert state.job_id
            return render_result(output_url="/output.mp4", error=state.error)
        if state.step is WizardStep.DONE:
            return render_done()
        return render_intro()

    def _render_enhance(self, state: WizardState) -> str:
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        from .segments import load_segments

        segments = load_segments(job_dir)
        segment = segments[state.segment_index]
        cache_key = str(state.segment_index)
        raw = state.enhance_cache.get(cache_key)
        if not isinstance(raw, dict):
            raise WizardError("enhancement options missing; revise the segment")
        result = SegmentEnhanceResult.from_dict(raw)
        preview_url = None
        chosen = state.enhance_cache.get(f"{cache_key}:preview")
        if isinstance(chosen, str):
            preview_url = chosen
        return render_enhance(
            segment=segment,
            result=result,
            preview_url=preview_url,
            error=state.error,
            segment_count=len(segments),
        )

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
            state.step = WizardStep.CHOOSE_SPLIT
            self._save(state)
            return
        if path == "/split":
            self._choose_split(state, fields)
            return
        if path == "/segment":
            self._enhance_post(state, fields)
            return
        if path == "/result":
            mood = fields.get("mood", [""])[0]
            if mood == "happy":
                state.step = WizardStep.DONE
            else:
                state.step = WizardStep.CHOOSE_SPLIT
                state.segment_index = 0
                state.layout_index = None
            self._save(state)
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

    def _choose_split(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        if not state.job_id:
            raise WizardError("no job")
        job_dir = self.store.job_dir(state.job_id)
        note = fields.get("note", [""])[0].strip()
        if note:
            try:
                source = Path(self.store.load(state.job_id).source_path)
                facts = probe_file(source)
                duration = float(facts.duration_s or 0.0)
                frames = sorted((job_dir / WIZARD_FRAMES_DIR).glob("f_*.jpg"))[:24]
                proposal = self.gemini().propose_splits(
                    frame_paths=frames,
                    duration_s=duration,
                    user_text=note,
                    job_dir=job_dir,
                )
                write_split_layouts(job_dir, proposal)
                state.highlights = proposal.highlights
                state.step = WizardStep.CHOOSE_SPLIT
            except (GeminiError, ProbeError) as exc:
                state.error = str(exc)
            self._save(state)
            return
        layout_raw = fields.get("layout", [""])[0]
        if layout_raw == "":
            state.error = "pick a layout or enter a note"
            self._save(state)
            return
        proposal = load_split_layouts(job_dir)
        index = int(layout_raw)
        if index < 0 or index >= len(proposal.layouts):
            state.error = "invalid layout"
            self._save(state)
            return
        layout = proposal.layouts[index]
        segments = layout_to_segments(layout)
        source = Path(self.store.load(state.job_id).source_path)
        prepare_segment_stills(source, job_dir, segments, ffmpeg_bin=self.ffmpeg_bin)
        write_segments(job_dir, segments)
        manifest = self.store.load(state.job_id)
        if manifest.state is JobState.SPLIT_PROPOSED:
            self.store.transition(state.job_id, JobState.SPLIT_ACCEPTED)
        state.layout_index = index
        state.segment_index = 0
        state.enhance_cache = {}
        try:
            self._ensure_enhance_options(state, segments[0])
        except GeminiError as exc:
            state.error = str(exc)
            state.step = WizardStep.CHOOSE_SPLIT
            self._save(state)
            return
        state.step = WizardStep.ENHANCE
        self._save(state)

    def _ensure_enhance_options(
        self,
        state: WizardState,
        segment: TimelineSegment,
    ) -> None:
        assert state.job_id
        job_dir = self.store.job_dir(state.job_id)
        key = str(state.segment_index)
        if key in state.enhance_cache:
            return
        still = job_dir / segment.still_path
        frames = [still] if still.is_file() else []
        jpg = job_dir / WIZARD_FRAMES_DIR
        near = sorted(jpg.glob("f_*.jpg"))[:3]
        frames = near or frames
        result = self.gemini().propose_enhance(
            frame_paths=frames,
            segment_label=segment.problem,
            start_s=segment.start_s,
            end_s=segment.end_s,
            job_dir=job_dir,
        )
        state.enhance_cache[key] = result.to_dict()

    def _enhance_post(self, state: WizardState, fields: dict[str, list[str]]) -> None:
        if not state.job_id:
            raise WizardError("no job")
        job_dir = self.store.job_dir(state.job_id)
        from .segments import load_segments

        segments = load_segments(job_dir)
        segment = segments[state.segment_index]
        note = fields.get("note", [""])[0].strip()
        if note:
            # Something-else loop: combine note + prior options, re-propose.
            # Repeats until the user previews/accepts an option (no note).
            try:
                frames = sorted((job_dir / WIZARD_FRAMES_DIR).glob("f_*.jpg"))[:3]
                key = str(state.segment_index)
                raw_prior = state.enhance_cache.get(key)
                prior = (
                    SegmentEnhanceResult.from_dict(raw_prior)
                    if isinstance(raw_prior, dict)
                    else None
                )
                if prior is None:
                    result = self.gemini().propose_enhance(
                        frame_paths=frames,
                        segment_label=segment.problem,
                        start_s=segment.start_s,
                        end_s=segment.end_s,
                        user_text=note,
                        job_dir=job_dir,
                    )
                else:
                    result = self.gemini().revise_enhance(
                        prior=prior,
                        frame_paths=frames,
                        segment_label=segment.problem,
                        start_s=segment.start_s,
                        end_s=segment.end_s,
                        user_note=note,
                        job_dir=job_dir,
                    )
                revise_key = f"{key}:revise_count"
                prior_count = state.enhance_cache.get(revise_key, 0)
                try:
                    count = int(prior_count)
                except (TypeError, ValueError):
                    count = 0
                state.enhance_cache[revise_key] = count + 1
                state.enhance_cache[key] = result.to_dict()
                state.enhance_cache.pop(f"{key}:preview", None)
                state.enhance_cache.pop(f"{key}:option", None)
                state.step = WizardStep.ENHANCE
            except GeminiError as exc:
                state.error = str(exc)
            self._save(state)
            return
        if fields.get("accept", [""])[0] == "1":
            preview_key = f"{state.segment_index}:option"
            option_id = state.enhance_cache.get(preview_key)
            raw = state.enhance_cache.get(str(state.segment_index))
            if not isinstance(option_id, str) or not isinstance(raw, dict):
                state.error = "preview an option before accepting"
                self._save(state)
                return
            result = SegmentEnhanceResult.from_dict(raw)
            option = next(
                (item for item in result.options if item.id == option_id), None
            )
            if option is None:
                state.error = "selected option is gone; preview again"
                self._save(state)
                return
            save_segment_choice(job_dir, state.segment_index, option)
            if state.segment_index + 1 < len(segments):
                state.segment_index += 1
                self._ensure_enhance_options(state, segments[state.segment_index])
                state.step = WizardStep.ENHANCE
            else:
                state.step = WizardStep.RENDERING
            self._save(state)
            return
        option_id = fields.get("option", [""])[0]
        if not option_id:
            state.error = "choose an option or enter a note"
            self._save(state)
            return
        raw = state.enhance_cache.get(str(state.segment_index))
        if not isinstance(raw, dict):
            state.error = "options missing"
            self._save(state)
            return
        result = SegmentEnhanceResult.from_dict(raw)
        option = next((item for item in result.options if item.id == option_id), None)
        if option is None:
            state.error = "unknown option"
            self._save(state)
            return
        treatment = treatment_from_ops(
            option.ops,
            treatment_id=f"wizard.{option.id}",
            problem=segment.problem,
        )
        source = Path(self.store.load(state.job_id).source_path)
        try:
            clip = encode_preview(
                job_dir,
                source,
                segment,
                treatment,
                ffmpeg_bin=self.ffmpeg_bin,
            )
        except (PreviewError, WizardError) as exc:
            state.error = str(exc)
            self._save(state)
            return
        rel = clip.path.relative_to(job_dir).as_posix()
        state.enhance_cache[f"{state.segment_index}:preview"] = f"/{rel}"
        state.enhance_cache[f"{state.segment_index}:option"] = option.id
        self._save(state)

    def run_render(self) -> None:
        state = self._state()
        if state.step is not WizardStep.RENDERING or not state.job_id:
            return
        job_id = state.job_id
        job_dir = self.store.job_dir(job_id)
        from .segments import load_segments

        segments = load_segments(job_dir)
        choices = load_segment_choices(job_dir).get("choices") or {}
        treatments: list[Treatment] = []
        try:
            for segment in segments:
                raw = choices.get(str(segment.index))
                if not isinstance(raw, dict):
                    raise WizardError(f"missing choice for segment {segment.index}")
                option = EnhanceOption.from_dict(raw)
                treatments.append(
                    treatment_from_ops(
                        option.ops,
                        treatment_id=f"wizard.{option.id}",
                        problem=segment.problem,
                    )
                )
            source = Path(self.store.load(job_id).source_path)
            current = self.store.load(job_id).state
            if current is JobState.SPLIT_ACCEPTED:
                self.store.transition(job_id, JobState.PLANS_READY)
                current = JobState.PLANS_READY
            if current is JobState.PLANS_READY:
                self.store.transition(job_id, JobState.PLAN_SELECTED)
                current = JobState.PLAN_SELECTED
            if current is JobState.PLAN_SELECTED:
                self.store.transition(job_id, JobState.ENCODING)
            render_chosen_plan(
                job_dir,
                source,
                segments,
                treatments,
                ffmpeg_bin=self.ffmpeg_bin,
            )
            if self.store.load(job_id).state is JobState.ENCODING:
                self.store.transition(job_id, JobState.COMPLETE)
            state.step = WizardStep.RESULT
        except (WizardError, RenderError, JobError, KeyError, ValueError) as exc:
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
