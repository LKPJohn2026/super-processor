"""Command-line entry point for Super Processor."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from . import __version__, mean_luma, sum_squares
from .doctor import (
    collect_doctor_report,
    doctor_json,
    doctor_report_as_dict,
    format_doctor_text,
)
from .estimators import EstimatorError, estimate_look, write_estimates
from .jobs import JobError, JobState, JobStore, default_jobs_root
from .probe import ProbeError, load_media_facts, probe_file, write_media_facts
from .recipe import RecipeError, TargetMode, empty_recipe, load_recipe, write_recipe
from .reframe import ReframeError, plan_social_export, write_reframe_plan
from .templates import TemplateError, build_ffmpeg_plan, default_output_path
from .validator import validate_job_recipe
from .worker import FFmpegWorker, WorkerError


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

    job = subparsers.add_parser("job", help="create and inspect processing jobs")
    job_sub = job.add_subparsers(dest="job_command", required=True)

    create = job_sub.add_parser("create", help="create a job from a local media file")
    create.add_argument("source", type=Path, help="path to the source media file")
    create.add_argument(
        "--job-id",
        default=None,
        help="optional explicit job id (8-32 lowercase hex/alphanumeric chars)",
    )

    show = job_sub.add_parser("show", help="print a job manifest as JSON")
    show.add_argument("job_id", help="job identifier")

    probe = job_sub.add_parser(
        "probe",
        help="run ffprobe and store versioned media facts for a job",
    )
    probe.add_argument("job_id", help="job identifier")

    estimate = job_sub.add_parser(
        "estimate",
        help="sample CV look/motion priors for a probed job",
    )
    estimate.add_argument("job_id", help="job identifier")

    reframe = job_sub.add_parser(
        "reframe",
        help="compute vertical reframe path and size-cap feasibility",
    )
    reframe.add_argument("job_id", help="job identifier")
    reframe.add_argument(
        "--max-height",
        type=positive_int,
        default=1920,
        help="target vertical height (default: 1920)",
    )
    reframe.add_argument(
        "--max-size-mb",
        type=float,
        default=None,
        help="optional delivery size cap in megabytes",
    )
    reframe.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="optional pad inset fraction in [0, 0.25]",
    )

    init_recipe = job_sub.add_parser(
        "init-recipe",
        help="write an empty allowlisted recipe for a probed or imported job",
    )
    init_recipe.add_argument("job_id", help="job identifier")

    validate = job_sub.add_parser(
        "validate",
        help="validate the job recipe against policy and media facts",
    )
    validate.add_argument("job_id", help="job identifier")

    run = job_sub.add_parser(
        "run",
        help="execute a validated recipe through local FFmpeg templates",
    )
    run.add_argument("job_id", help="job identifier")
    run.add_argument(
        "--mode",
        choices=["preview", "final"],
        default="preview",
        help="preview window encode or full-timeline final encode (default: preview)",
    )
    run.add_argument(
        "--approve",
        action="store_true",
        help="required acknowledgment for --mode final",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the FFmpeg plan as JSON without executing",
    )

    job_sub.add_parser("list", help="list known jobs")

    doctor_cmd = subparsers.add_parser(
        "doctor",
        help="check local FFmpeg, preview player, disk, and model configuration",
    )
    doctor_cmd.add_argument(
        "--json",
        action="store_true",
        help="print the doctor report as JSON",
    )
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


def run_job_command(arguments: argparse.Namespace) -> int:
    """Dispatch job subcommands."""
    store = _store_from_args(arguments)
    command = str(arguments.job_command)

    try:
        if command == "create":
            manifest = store.create(arguments.source, job_id=arguments.job_id)
            print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
            return 0
        if command == "show":
            manifest = store.load(str(arguments.job_id))
            print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
            return 0
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
            if not jobs:
                print("[]")
                return 0
            print(
                json.dumps(
                    [job.to_dict() for job in jobs],
                    indent=2,
                    sort_keys=True,
                )
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
) -> int:
    """Compute vertical reframe path and optional size-cap feasibility."""
    store.load(job_id)
    facts = load_media_facts(store.job_dir(job_id))
    plan = plan_social_export(
        facts,
        max_height=max_height,
        max_size_mb=max_size_mb,
        padding=padding,
    )
    write_reframe_plan(store.job_dir(job_id), plan)
    notes: dict[str, object] = {
        "reframe": True,
        "subject_strategy": plan.reframe.subject_strategy,
    }
    if plan.size_cap is not None:
        notes["size_cap_status"] = plan.size_cap.status
    store.update_notes(job_id, notes)
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    if plan.size_cap is not None and plan.size_cap.status == "infeasible":
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
    elif manifest.state is JobState.VALIDATED:
        # Stay validated until preview succeeds.
        pass

    worker = FFmpegWorker()
    result = worker.run_plan(plan)
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


def run_doctor_command(arguments: argparse.Namespace) -> int:
    """Run local toolchain diagnostics."""
    jobs_dir = arguments.jobs_dir.resolve() if arguments.jobs_dir is not None else None
    results = collect_doctor_report(jobs_dir)
    if bool(arguments.json):
        print(doctor_json(results))
    else:
        print(format_doctor_text(results))
    return 0 if doctor_report_as_dict(results)["ok"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    arguments = build_parser().parse_args(argv)
    command = str(arguments.command)

    if command == "version":
        print(__version__)
        return 0
    if command == "self-test":
        return run_self_test(int(arguments.sample_size))
    if command == "job":
        return run_job_command(arguments)
    if command == "doctor":
        return run_doctor_command(arguments)

    raise AssertionError(f"unhandled command: {command}")
