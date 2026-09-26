"""Command-line entry point for Super Processor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__, mean_luma, sum_squares
from .diagnose import (
    DiagnosisError,
    diagnose_job_dir,
    diagnose_source,
    write_diagnosis,
)
from .doctor import (
    collect_doctor_report,
    doctor_json,
    doctor_report_as_dict,
    format_doctor_text,
    which,
)
from .estimators import EstimatorError, estimate_look, write_estimates
from .jobs import JobError, JobState, JobStore, default_jobs_root
from .models import apply_llm_plan_patch, list_models
from .plan import PlanError, plan_job
from .probe import ProbeError, load_media_facts, probe_file, write_media_facts
from .qa import analyze_preview, write_qa_report
from .recipe import RecipeError, TargetMode, empty_recipe, load_recipe, write_recipe
from .reframe import ReframeError, plan_social_export, write_reframe_plan
from .segments import (
    SegmentError,
    TimelineSegment,
    apply_split_note,
    ffmpeg_frame_reader,
    load_samples,
    load_segments,
    parse_split_note,
    sample_media,
    write_gray_still,
    write_samples,
    write_segment_review,
    write_segments,
)
from .templates import (
    PREVIEW_FILE_NAME,
    TemplateError,
    build_ffmpeg_plan,
    default_output_path,
)
from .validator import validate_job_recipe
from .worker import FFmpegWorker, WorkerError, format_progress_status


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="super-processor",
        description="Local-first AI-assisted video processing.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=None,
        help=(
            "directory used to store processing jobs "
            f"(default: {default_jobs_root()} or $SUPER_PROCESSOR_JOBS_DIR)"
        ),
    )
    parser.add_argument(
        "--safe-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="prefer CV/NL patches and skip remote planner calls (default: true)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="print the installed version")

    self_test = subparsers.add_parser(
        "self-test",
        help="verify that the compiled Cython extension is working",
    )
    self_test.add_argument(
        "--sample-size",
        type=positive_int,
        default=10_000,
        help="number of integers used by the compiled test (default: 10000)",
    )

    doctor_cmd = subparsers.add_parser(
        "doctor",
        help="check local FFmpeg, preview player, disk, and model configuration",
    )
    doctor_cmd.add_argument(
        "--json",
        action="store_true",
        help="print the doctor report as JSON",
    )

    models_cmd = subparsers.add_parser(
        "models",
        help="list configured local/BYOK model endpoints",
    )
    models_cmd.add_argument(
        "--consent-remote-frames",
        action="store_true",
        help="acknowledge that remote VLM calls may upload frames",
    )

    diagnose_cmd = subparsers.add_parser(
        "diagnose",
        help="create a job and CV diagnosis from an input media file",
    )
    diagnose_cmd.add_argument("source", type=Path, help="path to the source media file")
    diagnose_cmd.add_argument("--job-id", default=None, help="optional explicit job id")
    diagnose_cmd.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help="optional social export size cap",
    )
    diagnose_cmd.add_argument(
        "--acknowledge-size-risk",
        action="store_true",
        help="allow planning when a size cap is below the bitrate floor",
    )

    plan_cmd = subparsers.add_parser(
        "plan",
        help="build an allowlisted recipe from diagnosis (CV + optional NL)",
    )
    plan_cmd.add_argument("job_id", help="job identifier")
    plan_cmd.add_argument(
        "--instruction",
        default=None,
        help="optional natural-language patch (e.g. 'less denoise')",
    )
    plan_cmd.add_argument(
        "--acknowledge-size-risk",
        action="store_true",
        help="downgrade an infeasible size cap to a warning",
    )

    preview_cmd = subparsers.add_parser(
        "preview",
        help="encode a preview, run QA, optionally open a player",
    )
    preview_cmd.add_argument("job_id", help="job identifier")
    preview_cmd.add_argument(
        "--open",
        choices=["vlc", "ffplay", "auto"],
        default=None,
        help="open the preview in a local player",
    )
    preview_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="print the FFmpeg plan without encoding",
    )

    apply_cmd = subparsers.add_parser(
        "apply",
        help="run a full-timeline encode after successful preview + approval",
    )
    apply_cmd.add_argument("job_id", help="job identifier")
    apply_cmd.add_argument(
        "--approve",
        action="store_true",
        required=True,
        help="required explicit approval for the final encode",
    )
    apply_cmd.add_argument(
        "--acknowledge-size-risk",
        action="store_true",
        help="allow final encode when the size cap is below the bitrate floor",
    )

    show_cmd = subparsers.add_parser(
        "show",
        help="show job manifest plus available artifacts summary",
    )
    show_cmd.add_argument("job_id", help="job identifier")

    segment_cmd = subparsers.add_parser(
        "segment",
        help="propose a timeline split and print each segment",
    )
    segment_cmd.add_argument("job_id", help="job identifier")
    segment_cmd.add_argument(
        "--note",
        default=None,
        help="short correction: too many, too few, a boundary move, or a relabel",
    )

    job = subparsers.add_parser("job", help="low-level job management commands")
    job_sub = job.add_subparsers(dest="job_command", required=True)

    create = job_sub.add_parser("create", help="create a job from a local media file")
    create.add_argument("source", type=Path, help="path to the source media file")
    create.add_argument("--job-id", default=None, help="optional explicit job id")

    show_job = job_sub.add_parser("show", help="print a job manifest as JSON")
    show_job.add_argument("job_id", help="job identifier")

    probe = job_sub.add_parser("probe", help="run ffprobe for a job")
    probe.add_argument("job_id", help="job identifier")

    estimate = job_sub.add_parser("estimate", help="sample CV look/motion priors")
    estimate.add_argument("job_id", help="job identifier")

    reframe = job_sub.add_parser("reframe", help="compute vertical reframe + size-cap")
    reframe.add_argument("job_id", help="job identifier")
    reframe.add_argument("--max-height", type=positive_int, default=1920)
    reframe.add_argument("--max-size-mb", type=float, default=None)
    reframe.add_argument("--padding", type=float, default=0.0)
    reframe.add_argument(
        "--acknowledge-size-risk",
        action="store_true",
        help="exit 0 even when the size cap is below the bitrate floor",
    )

    init_recipe = job_sub.add_parser("init-recipe", help="write an empty recipe")
    init_recipe.add_argument("job_id", help="job identifier")

    validate = job_sub.add_parser("validate", help="validate a job recipe")
    validate.add_argument("job_id", help="job identifier")

    run = job_sub.add_parser("run", help="execute a recipe through FFmpeg templates")
    run.add_argument("job_id", help="job identifier")
    run.add_argument("--mode", choices=["preview", "final"], default="preview")
    run.add_argument("--approve", action="store_true")
    run.add_argument("--dry-run", action="store_true")

    job_sub.add_parser("list", help="list known jobs")
    return parser


def run_self_test(sample_size: int) -> int:
    """Exercise compiled functions and return a process exit code."""
    observed = sum_squares(sample_size)
    expected = sample_size * (sample_size - 1) * (2 * sample_size - 1) // 6
    luma = mean_luma(bytes((16, 64, 128, 235)))
    if observed != expected or luma != 110.75:
        print("Cython extension self-test failed")
        return 1
    print(
        "Cython extension OK "
        f"(sum_squares({sample_size})={observed}, mean_luma={luma:.2f})"
    )
    return 0


def _store_from_args(arguments: argparse.Namespace) -> JobStore:
    root = arguments.jobs_dir
    return JobStore(root.resolve() if root is not None else None)


def run_doctor_command(arguments: argparse.Namespace) -> int:
    """Run local toolchain diagnostics."""
    jobs_dir = arguments.jobs_dir.resolve() if arguments.jobs_dir is not None else None
    results = collect_doctor_report(jobs_dir)
    if bool(arguments.json):
        print(doctor_json(results))
    else:
        print(format_doctor_text(results))
    return 0 if doctor_report_as_dict(results)["ok"] else 1


def run_models_command(arguments: argparse.Namespace) -> int:
    """List configured model endpoints."""
    payload = list_models(
        remote_frames_consent=bool(arguments.consent_remote_frames),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_diagnose_command(arguments: argparse.Namespace) -> int:
    """Create/probe a job and write a CV diagnosis."""
    store = _store_from_args(arguments)
    source = Path(arguments.source)
    manifest = store.create(source, job_id=arguments.job_id)
    facts = probe_file(Path(manifest.source_path))
    write_media_facts(store.job_dir(manifest.job_id), facts)
    store.transition(
        manifest.job_id,
        JobState.PROBED,
        notes={"has_video": facts.has_video, "duration_s": facts.duration_s},
    )
    estimates = estimate_look(Path(manifest.source_path), facts)
    write_estimates(store.job_dir(manifest.job_id), estimates)
    max_size = None if arguments.max_size_mb is None else float(arguments.max_size_mb)
    ack = bool(getattr(arguments, "acknowledge_size_risk", False))
    social = plan_social_export(
        facts,
        max_size_mb=max_size,
        source=Path(manifest.source_path),
    )
    write_reframe_plan(store.job_dir(manifest.job_id), social)
    diagnosis = diagnose_source(
        Path(manifest.source_path),
        facts,
        max_size_mb=max_size,
        acknowledge_size_risk=ack,
    )
    write_diagnosis(store.job_dir(manifest.job_id), diagnosis)
    store.transition(
        manifest.job_id,
        JobState.DIAGNOSED,
        notes={
            "summary": diagnosis.summary,
            "subject_strategy": social.reframe.subject_strategy,
            "size_risk_acknowledged": ack,
        },
    )
    print(json.dumps(diagnosis.to_dict(), indent=2, sort_keys=True))
    return 0


def _print_segments(segments: list[TimelineSegment]) -> None:
    for segment in segments:
        print(
            f"{segment.index}  {segment.start_s:.0f}-{segment.end_s:.0f}s  "
            f"{segment.context}  {segment.problem}  "
            f"key {segment.keyframe_s:.0f}s  {segment.still_path}"
        )


def _attach_stills(
    job_dir: Path, source: Path, segments: list[TimelineSegment]
) -> None:
    read_frame = ffmpeg_frame_reader(source)
    for segment in segments:
        gray, _rgb = read_frame(segment.keyframe_s)
        relative = f"segment_stills/seg_{segment.index:02d}.ppm"
        write_gray_still(job_dir / relative, gray)
        segment.still_path = relative


def run_segment_command(arguments: argparse.Namespace) -> int:
    """Sample a probed job, write the split, and print it."""
    store = _store_from_args(arguments)
    job_id = str(arguments.job_id)
    manifest = store.load(job_id)
    if manifest.state not in {JobState.PROBED, JobState.SPLIT_PROPOSED}:
        raise JobError(
            f"job {job_id} must be probed before a split "
            f"(current: {manifest.state.value})"
        )
    job_dir = store.job_dir(job_id)
    source = Path(manifest.source_path)
    note = getattr(arguments, "note", None)
    if note:
        if manifest.state is not JobState.SPLIT_PROPOSED:
            raise JobError(
                f"job {job_id} needs a proposed split before a note "
                f"(current: {manifest.state.value})"
            )
        rows = load_samples(job_dir)
        segments = apply_split_note(
            rows,
            load_segments(job_dir),
            parse_split_note(str(note)),
        )
        _attach_stills(job_dir, source, segments)
        write_segments(job_dir, segments)
        _print_segments(segments)
        return 0
    try:
        rows = load_samples(job_dir)
    except SegmentError:
        facts = load_media_facts(job_dir)
        if facts.duration_s is None:
            raise JobError(f"job {job_id} has no duration to sample") from None
        rows = sample_media(source, facts.duration_s)
        write_samples(job_dir, rows)
    segments = write_segment_review(job_dir, rows, ffmpeg_frame_reader(source))
    if manifest.state is JobState.PROBED:
        store.transition(
            job_id,
            JobState.SPLIT_PROPOSED,
            notes={"segments": len(segments)},
        )
    _print_segments(segments)
    return 0


def run_plan_command(arguments: argparse.Namespace) -> int:
    """Plan a recipe from diagnosis with optional NL / LLM advisory patching."""
    store = _store_from_args(arguments)
    job_id = str(arguments.job_id)
    manifest = store.load(job_id)
    ack = bool(getattr(arguments, "acknowledge_size_risk", False))
    if manifest.state not in {
        JobState.DIAGNOSED,
        JobState.PLANNED,
        JobState.VALIDATED,
    }:
        # Allow planning from probed jobs by diagnosing on the fly.
        if manifest.state is JobState.PROBED:
            diagnosis = diagnose_job_dir(
                store.job_dir(job_id),
                acknowledge_size_risk=ack,
            )
            write_diagnosis(store.job_dir(job_id), diagnosis)
            store.transition(job_id, JobState.DIAGNOSED)
        else:
            raise JobError(
                f"job {job_id} must be diagnosed before plan "
                f"(current: {manifest.state.value})"
            )

    if ack:
        # Re-stamp diagnosis so planner/validator see the acknowledgment.
        diagnosis = diagnose_job_dir(
            store.job_dir(job_id),
            acknowledge_size_risk=True,
        )
        write_diagnosis(store.job_dir(job_id), diagnosis)

    recipe = plan_job(
        store.job_dir(job_id),
        job_id,
        instruction=arguments.instruction,
    )
    if arguments.instruction:
        recipe = apply_llm_plan_patch(
            recipe,
            str(arguments.instruction),
            safe_mode=bool(arguments.safe_mode),
        )
        write_recipe(store.job_dir(job_id), recipe)

    if ack:
        for op in recipe.ops:
            if op.op.value == "encode_hevc_size_cap" and op.enabled:
                op.params = dict(op.params)
                op.params["acknowledge_size_risk"] = 1.0
        write_recipe(store.job_dir(job_id), recipe)

    validation = validate_job_recipe(store.job_dir(job_id))
    if store.load(job_id).state is JobState.DIAGNOSED:
        store.transition(
            job_id,
            JobState.PLANNED,
            notes={"planned": True, "size_risk_acknowledged": ack},
        )
    if validation.ok and store.load(job_id).state is JobState.PLANNED:
        store.transition(job_id, JobState.VALIDATED, notes={"validated": True})

    payload = {
        "recipe": recipe.to_dict(),
        "validation": validation.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if validation.ok else 1


def _open_preview(path: Path, player: str) -> None:
    chosen = player
    if player == "auto":
        chosen = "vlc" if which("vlc") else "ffplay"
    exe = which(chosen)
    if exe is None:
        raise JobError(f"player not found: {chosen}")
    subprocess.Popen(  # noqa: S603 - argv list, no shell
        [exe, str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_preview_command(arguments: argparse.Namespace) -> int:
    """Encode preview, run QA, optionally open a player."""
    store = _store_from_args(arguments)
    job_id = str(arguments.job_id)
    code = run_job_run(
        store,
        job_id,
        mode="preview",
        approve=False,
        dry_run=bool(arguments.dry_run),
    )
    if bool(arguments.dry_run) or code != 0:
        return code
    preview = store.job_dir(job_id) / PREVIEW_FILE_NAME
    report = analyze_preview(preview, source_path=Path(store.load(job_id).source_path))
    write_qa_report(store.job_dir(job_id), report)
    print(json.dumps({"qa": report.to_dict()}, indent=2, sort_keys=True))
    if arguments.open:
        _open_preview(preview, str(arguments.open))
    return 0 if report.ok else 1


def run_apply_command(arguments: argparse.Namespace) -> int:
    """Full-timeline encode gated on preview + explicit approval."""
    store = _store_from_args(arguments)
    job_id = str(arguments.job_id)
    manifest = store.load(job_id)
    if manifest.state not in {JobState.PREVIEWED, JobState.APPROVED}:
        raise JobError(
            f"job {job_id} must be previewed before apply "
            f"(current: {manifest.state.value})"
        )
    if bool(getattr(arguments, "acknowledge_size_risk", False)):
        recipe = load_recipe(store.job_dir(job_id))
        for op in recipe.ops:
            if op.op.value == "encode_hevc_size_cap" and op.enabled:
                op.params = dict(op.params)
                op.params["acknowledge_size_risk"] = 1.0
        write_recipe(store.job_dir(job_id), recipe)
        store.update_notes(job_id, {"size_risk_acknowledged": True})
    if manifest.state is JobState.PREVIEWED:
        store.transition(job_id, JobState.APPROVED, notes={"approved": True})
    return run_job_run(
        store,
        job_id,
        mode="final",
        approve=True,
        dry_run=False,
    )


def run_show_command(arguments: argparse.Namespace) -> int:
    """Show manifest plus artifact presence."""
    store = _store_from_args(arguments)
    job_id = str(arguments.job_id)
    manifest = store.load(job_id)
    job_dir = store.job_dir(job_id)
    artifacts = {
        name: (job_dir / name).is_file()
        for name in (
            "media_facts.json",
            "estimates.json",
            "reframe_plan.json",
            "diagnosis.json",
            "recipe.json",
            "preview.mp4",
            "qa_report.json",
            "output.mp4",
        )
    }
    payload = {"manifest": manifest.to_dict(), "artifacts": artifacts}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_job_command(arguments: argparse.Namespace) -> int:
    """Dispatch low-level job subcommands."""
    store = _store_from_args(arguments)
    command = str(arguments.job_command)
    try:
        if command == "create":
            manifest = store.create(arguments.source, job_id=arguments.job_id)
            print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
            return 0
        if command == "show":
            return run_show_command(arguments)
        if command == "probe":
            return run_job_probe(store, str(arguments.job_id))
        if command == "estimate":
            return run_job_estimate(store, str(arguments.job_id))
        if command == "reframe":
            return run_job_reframe(
                store,
                str(arguments.job_id),
                max_height=int(arguments.max_height),
                max_size_mb=(
                    None
                    if arguments.max_size_mb is None
                    else float(arguments.max_size_mb)
                ),
                padding=float(arguments.padding),
                acknowledge_size_risk=bool(arguments.acknowledge_size_risk),
            )
        if command == "init-recipe":
            return run_job_init_recipe(store, str(arguments.job_id))
        if command == "validate":
            return run_job_validate(store, str(arguments.job_id))
        if command == "run":
            return run_job_run(
                store,
                str(arguments.job_id),
                mode=str(arguments.mode),
                approve=bool(arguments.approve),
                dry_run=bool(arguments.dry_run),
            )
        if command == "list":
            jobs = store.list_jobs()
            print(
                json.dumps([job.to_dict() for job in jobs], indent=2, sort_keys=True)
                if jobs
                else "[]"
            )
            return 0
    except (
        JobError,
        ProbeError,
        RecipeError,
        TemplateError,
        WorkerError,
        EstimatorError,
        ReframeError,
        DiagnosisError,
        PlanError,
    ) as exc:
        print(f"error: {exc}", flush=True)
        return 1
    raise AssertionError(f"unhandled job command: {command}")


def run_job_probe(store: JobStore, job_id: str) -> int:
    """Probe a job's source file and transition imported -> probed."""
    manifest = store.load(job_id)
    if manifest.state not in {JobState.IMPORTED, JobState.PROBED}:
        raise JobError(
            "job "
            f"{job_id} must be imported or probed to run probe "
            f"(current: {manifest.state.value})"
        )
    facts = probe_file(Path(manifest.source_path))
    write_media_facts(store.job_dir(job_id), facts)
    notes = {
        "has_video": facts.has_video,
        "has_audio": facts.has_audio,
        "duration_s": facts.duration_s,
    }
    if manifest.state is JobState.IMPORTED:
        store.transition(job_id, JobState.PROBED, notes=notes)
    else:
        store.update_notes(job_id, notes)
    print(json.dumps(facts.to_dict(), indent=2, sort_keys=True))
    return 0


def run_job_estimate(store: JobStore, job_id: str) -> int:
    """Sample classical look/motion priors for a probed job."""
    manifest = store.load(job_id)
    if manifest.state not in {
        JobState.PROBED,
        JobState.DIAGNOSED,
        JobState.PLANNED,
        JobState.VALIDATED,
    }:
        raise JobError(
            f"job {job_id} must be probed before estimate "
            f"(current: {manifest.state.value})"
        )
    facts = load_media_facts(store.job_dir(job_id))
    estimates = estimate_look(Path(manifest.source_path), facts)
    write_estimates(store.job_dir(job_id), estimates)
    store.update_notes(
        job_id,
        {
            "estimated": True,
            "contrast": estimates.contrast.enabled,
            "white_balance": estimates.white_balance.enabled,
            "denoise": estimates.denoise.enabled,
            "stabilize": estimates.stabilize.enabled,
        },
    )
    print(json.dumps(estimates.to_dict(), indent=2, sort_keys=True))
    return 0


def run_job_reframe(
    store: JobStore,
    job_id: str,
    *,
    max_height: int,
    max_size_mb: float | None,
    padding: float,
    acknowledge_size_risk: bool = False,
) -> int:
    """Compute vertical reframe path and optional size-cap feasibility."""
    store.load(job_id)
    facts = load_media_facts(store.job_dir(job_id))
    plan = plan_social_export(
        facts,
        max_height=max_height,
        max_size_mb=max_size_mb,
        padding=padding,
        source=Path(facts.source_path),
    )
    write_reframe_plan(store.job_dir(job_id), plan)
    notes: dict[str, object] = {
        "reframe": True,
        "subject_strategy": plan.reframe.subject_strategy,
        "subject_cx": plan.reframe.subject_cx,
    }
    if plan.size_cap is not None:
        notes["size_cap_status"] = plan.size_cap.status
        if acknowledge_size_risk and plan.size_cap.status == "infeasible":
            notes["size_risk_acknowledged"] = True
    store.update_notes(job_id, notes)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    if (
        plan.size_cap is not None
        and plan.size_cap.status == "infeasible"
        and not acknowledge_size_risk
    ):
        return 1
    return 0


def run_job_init_recipe(store: JobStore, job_id: str) -> int:
    """Write an empty allowlisted recipe for a job."""
    manifest = store.load(job_id)
    recipe = empty_recipe(manifest.job_id, manifest.source_path)
    write_recipe(store.job_dir(job_id), recipe)
    print(json.dumps(recipe.to_dict(), indent=2, sort_keys=True))
    return 0


def run_job_validate(store: JobStore, job_id: str) -> int:
    """Validate the on-disk recipe for a job."""
    manifest = store.load(job_id)
    result = validate_job_recipe(store.job_dir(job_id))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.ok:
        return 1
    if manifest.state in {JobState.PROBED, JobState.PLANNED}:
        store.transition(job_id, JobState.VALIDATED, notes={"validated": True})
    elif manifest.state is JobState.VALIDATED:
        store.update_notes(job_id, {"validated": True})
    return 0


def run_job_run(
    store: JobStore,
    job_id: str,
    *,
    mode: str,
    approve: bool,
    dry_run: bool,
) -> int:
    """Build and optionally execute an FFmpeg plan for a job recipe."""
    if mode == "final" and not approve and not dry_run:
        raise JobError("final encode requires --approve")

    manifest = store.load(job_id)
    job_dir = store.job_dir(job_id)
    validation = validate_job_recipe(job_dir)
    if not validation.ok:
        print(json.dumps(validation.to_dict(), indent=2, sort_keys=True))
        return 1

    if manifest.state in {JobState.PROBED, JobState.PLANNED}:
        manifest = store.transition(
            job_id, JobState.VALIDATED, notes={"validated": True}
        )

    recipe = load_recipe(job_dir)
    recipe.target.mode = TargetMode.PREVIEW if mode == "preview" else TargetMode.FINAL
    write_recipe(job_dir, recipe)

    duration_s: float | None = None
    try:
        facts = load_media_facts(job_dir)
        duration_s = facts.duration_s
    except Exception:
        duration_s = None

    output_path = default_output_path(job_dir, recipe)
    plan = build_ffmpeg_plan(
        recipe,
        output_path=output_path,
        work_dir=job_dir,
        duration_s=duration_s,
    )
    if dry_run:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0

    if mode == "final":
        if manifest.state not in {
            JobState.VALIDATED,
            JobState.PREVIEWED,
            JobState.APPROVED,
        }:
            raise JobError(
                f"job {job_id} must be validated before final encode "
                f"(current: {manifest.state.value})"
            )
        store.transition(job_id, JobState.ENCODING, notes={"mode": "final"})

    worker = FFmpegWorker()

    def _on_progress(progress: dict[str, str]) -> None:
        status = format_progress_status(progress, duration_s=duration_s)
        print(f"\r{status}", end="", flush=True, file=sys.stderr)

    result = worker.run_plan(plan, on_progress=_on_progress)
    print(file=sys.stderr)
    payload = result.to_dict()
    payload["validation"] = validation.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))

    if not result.ok:
        store.transition(
            job_id,
            JobState.FAILED,
            error=result.steps[-1].stderr_tail if result.steps else "ffmpeg failed",
        )
        return 1

    if mode == "preview":
        if store.load(job_id).state is JobState.VALIDATED:
            store.transition(
                job_id,
                JobState.PREVIEWED,
                notes={"preview_path": result.output_path},
            )
        else:
            store.update_notes(job_id, {"preview_path": result.output_path})
    else:
        store.transition(
            job_id,
            JobState.COMPLETE,
            notes={"output_path": result.output_path},
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    arguments = build_parser().parse_args(argv)
    command = str(arguments.command)
    try:
        if command == "version":
            print(__version__)
            return 0
        if command == "self-test":
            return run_self_test(int(arguments.sample_size))
        if command == "doctor":
            return run_doctor_command(arguments)
        if command == "models":
            return run_models_command(arguments)
        if command == "diagnose":
            return run_diagnose_command(arguments)
        if command == "plan":
            return run_plan_command(arguments)
        if command == "preview":
            return run_preview_command(arguments)
        if command == "apply":
            return run_apply_command(arguments)
        if command == "show":
            return run_show_command(arguments)
        if command == "segment":
            return run_segment_command(arguments)
        if command == "job":
            return run_job_command(arguments)
    except (
        JobError,
        ProbeError,
        RecipeError,
        TemplateError,
        WorkerError,
        EstimatorError,
        ReframeError,
        DiagnosisError,
        PlanError,
        SegmentError,
    ) as exc:
        print(f"error: {exc}", flush=True)
        return 1
    raise AssertionError(f"unhandled command: {command}")
