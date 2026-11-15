"""Tests for the FFmpeg worker and job run CLI."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call

import pytest

from super_processor import cli
from super_processor import templates as templates_mod
from super_processor import worker as worker_mod
from super_processor.jobs import JobState, JobStore
from super_processor.probe import (
    MediaFacts,
    StreamFacts,
    probe_file,
    write_media_facts,
)
from super_processor.recipe import OpName, TargetMode, empty_recipe, write_recipe
from super_processor.templates import (
    FFmpegPlan,
    FFmpegStep,
    TemplateError,
    build_ffmpeg_plan,
)
from super_processor.worker import (
    FFmpegWorker,
    RunResult,
    StepResult,
    WorkerError,
    format_progress_status,
    parse_progress_line,
)


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


requires_ffmpeg = pytest.mark.skipif(not _have_ffmpeg(), reason="ffmpeg not on PATH")


def _make_clip(path: Path, *, duration: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration:g}:size=320x240:rate=10",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration:g}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def test_parse_progress_line() -> None:
    progress: dict[str, str] = {}
    parse_progress_line("out_time_ms=1234000", progress)
    parse_progress_line("progress=continue", progress)
    parse_progress_line("ignored", progress)
    assert progress["out_time_ms"] == "1234000"
    assert progress["progress"] == "continue"


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX SIGTERM handlers")
@pytest.mark.parametrize("resist_term", [False, True], ids=["cooperative", "resistant"])
def test_worker_cancel_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resist_term: bool,
) -> None:
    monkeypatch.setattr(worker_mod, "_CANCEL_GRACE_S", 0.2)
    kill_group = Mock(wraps=worker_mod._kill_process_group)
    monkeypatch.setattr(worker_mod, "_kill_process_group", kill_group)
    output = tmp_path / "output.mp4"
    output.write_text("original")
    partial = output.with_suffix(".partial.mp4")
    later_step = tmp_path / "later-step"
    handler = "signal.SIG_IGN" if resist_term else "lambda *_: sys.exit(0)"
    script = (
        "import pathlib, signal, sys, time; "
        f"signal.signal(signal.SIGTERM, {handler}); "
        "pathlib.Path(sys.argv[1]).write_text('partial'); "
        "print('ready=1', flush=True); "
        "time.sleep(30)"
    )
    plan = FFmpegPlan(
        mode="preview",
        input_path="unused",
        output_path=str(output),
        steps=[
            FFmpegStep("encode", [sys.executable, "-c", script, str(output)]),
            FFmpegStep(
                "later",
                [sys.executable, "-c", f"open({str(later_step)!r}, 'w').close()"],
            ),
        ],
        filter_graph="",
    )
    worker = FFmpegWorker()
    ready = threading.Event()
    results: list[RunResult] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(
                worker.run_plan(plan, on_progress=lambda progress: ready.set())
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        assert ready.wait(timeout=5), "child did not install its signal handler"
        started = time.monotonic()
        worker.cancel()
        worker.cancel()  # Repeated requests must not restart the grace period.
        thread.join(timeout=3)
        assert not thread.is_alive(), "cancellation did not finish within its bound"
        assert time.monotonic() - started < 3
    finally:
        # Also bound cleanup when this regression runs against the old worker.
        if thread.is_alive():
            with worker._lock:
                proc = worker._current
            if proc is not None:
                worker_mod._kill_process_group(proc)
            thread.join(timeout=3)

    assert not errors
    result = results[0]
    assert result.cancelled is True
    assert result.ok is False
    assert len(result.steps) == 1
    assert result.steps[0].returncode == (
        -signal.SIGKILL if resist_term else -signal.SIGTERM
    )
    assert kill_group.call_count == int(resist_term)
    assert output.read_text() == "original"
    assert not partial.exists()
    assert not later_step.exists()
    assert worker._current is None

    # A completed cancellation must never leave a timer targeting a later run.
    plan.steps = [
        FFmpegStep(
            "encode",
            [
                sys.executable,
                "-c",
                "import pathlib, sys, time; time.sleep(0.3); "
                "pathlib.Path(sys.argv[1]).write_text('replacement')",
                str(output),
            ],
        )
    ]
    assert worker.run_plan(plan).ok is True
    assert output.read_text() == "replacement"
    assert kill_group.call_count == int(resist_term)


def test_worker_cancel_during_process_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = FFmpegWorker()
    popen = subprocess.Popen

    def cancel_on_start(*args: Any, **kwargs: Any) -> subprocess.Popen[str]:
        proc: subprocess.Popen[str] = popen(*args, **kwargs)
        worker.cancel()
        return proc

    monkeypatch.setattr(subprocess, "Popen", cancel_on_start)
    plan = FFmpegPlan(
        mode="preview",
        input_path="unused",
        output_path=str(tmp_path / "output.mp4"),
        steps=[
            FFmpegStep("prepare", [sys.executable, "-c", "import time; time.sleep(30)"])
        ],
        filter_graph="",
    )
    result = worker.run_plan(plan)
    assert result.cancelled is True
    assert result.ok is False
    assert result.steps[0].returncode != 0
    assert worker._current is None


def test_process_group_signals_skip_reaped_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = Mock()
    proc.poll.return_value = 0
    killpg = Mock()
    monkeypatch.setattr(worker_mod, "os", SimpleNamespace(name="posix", killpg=killpg))
    worker_mod._terminate_process_group(proc)
    worker_mod._kill_process_group(proc)
    killpg.assert_not_called()
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX signal constants")
def test_process_group_posix_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = Mock(pid=123)
    proc.poll.return_value = None
    killpg = Mock()
    monkeypatch.setattr(worker_mod, "os", SimpleNamespace(name="posix", killpg=killpg))
    worker_mod._terminate_process_group(proc)
    worker_mod._kill_process_group(proc)
    assert killpg.call_args_list == [
        call(proc.pid, signal.SIGTERM),
        call(proc.pid, signal.SIGKILL),
    ]
    proc.terminate.assert_not_called()
    proc.kill.assert_not_called()


def test_process_group_windows_signals(monkeypatch: pytest.MonkeyPatch) -> None:
    proc = Mock()
    proc.poll.return_value = None
    monkeypatch.setattr(worker_mod, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(signal, "CTRL_BREAK_EVENT", 123, raising=False)
    worker_mod._terminate_process_group(proc)
    proc.send_signal.assert_called_once_with(123)
    worker_mod._kill_process_group(proc)
    proc.kill.assert_called_once_with()

    # Console-less Windows processes may reject CTRL_BREAK_EVENT.
    proc.send_signal.side_effect = OSError("no console")
    worker_mod._terminate_process_group(proc)
    proc.terminate.assert_called_once_with()


@requires_ffmpeg
def test_worker_preview_encode_atomic(tmp_path: Path) -> None:
    source = _make_clip(tmp_path / "src.mp4")
    recipe = empty_recipe("abcd1234abcd1234", str(source))
    for op in recipe.ops:
        if op.op is OpName.CONTRAST:
            op.enabled = True
            op.params = {"contrast": 1.1, "brightness": 0.0, "gamma": 1.0}
    recipe.target.mode = TargetMode.PREVIEW
    recipe.target.preview_window.duration_s = 0.5
    recipe.encode.preset = "ultrafast"
    recipe.encode.crf = 28

    output = tmp_path / "preview.mp4"
    plan = build_ffmpeg_plan(
        recipe,
        output_path=output,
        work_dir=tmp_path / "work",
        duration_s=1.0,
    )
    updates: list[dict[str, str]] = []
    result = FFmpegWorker().run_plan(plan, on_progress=updates.append)
    assert result.ok is True
    assert output.is_file()
    assert not output.with_name(f"{output.stem}.partial{output.suffix}").exists()
    assert result.steps[0].returncode == 0
    assert updates
    progress = result.steps[0].progress
    assert int(progress["out_time_ms"]) > 0
    # Compare the two time formats emitted by an actual FFmpeg encode.
    assert format_progress_status(
        {"out_time_ms": progress["out_time_ms"]}, duration_s=0.5
    ) == format_progress_status({"out_time": progress["out_time"]}, duration_s=0.5)


@requires_ffmpeg
def test_cancelling_a_real_encode_leaves_no_output(tmp_path: Path) -> None:
    source = _make_clip(tmp_path / "src.mp4", duration=60.0)
    recipe = empty_recipe("abcd1234abcd1234", str(source.resolve()))
    recipe.target.mode = TargetMode.FINAL
    output = tmp_path / "out.mp4"
    plan = build_ffmpeg_plan(
        recipe,
        output_path=output,
        work_dir=tmp_path / "work",
        duration_s=60.0,
    )
    # "placebo" keeps the 60 s encode running for many seconds, so it is
    # still going when the first progress report arrives.
    encode = plan.steps[-1]
    encode.argv[encode.argv.index("-preset") + 1] = "placebo"

    worker = FFmpegWorker()
    # Cancel from FFmpeg's own first progress line rather than after a sleep,
    # so the cancel always lands while the encode is running.
    result = worker.run_plan(plan, on_progress=lambda _progress: worker.cancel())
    assert result.cancelled is True
    assert result.ok is False
    assert result.steps[-1].returncode != 0
    assert not output.is_file()
    assert not output.with_suffix(".partial.mp4").exists()


def test_a_failed_step_stops_the_plan_and_keeps_the_old_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.mp4"
    output.write_text("original")
    later = tmp_path / "later-step"
    plan = FFmpegPlan(
        mode="final",
        input_path="unused",
        output_path=str(output),
        steps=[
            FFmpegStep(
                "encode",
                [
                    sys.executable,
                    "-c",
                    "import pathlib, sys; "
                    "pathlib.Path(sys.argv[1]).write_text('half'); "
                    "print('Conversion failed!', file=sys.stderr); sys.exit(3)",
                    str(output),
                ],
            ),
            FFmpegStep(
                "later",
                [sys.executable, "-c", f"open({str(later)!r}, 'w').close()"],
            ),
        ],
        filter_graph="",
    )
    result = FFmpegWorker().run_plan(plan)
    assert result.ok is False
    assert result.cancelled is False
    assert [step.name for step in result.steps] == ["encode"]
    assert result.steps[0].returncode == 3
    assert result.steps[0].stderr_tail == "Conversion failed!"
    # A failed encode must not replace a good earlier output with half a file.
    assert output.read_text() == "original"
    assert not output.with_name("output.partial.mp4").exists()
    assert not later.exists()


def test_a_successful_encode_replaces_the_output_through_a_partial(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.mp4"
    output.write_text("original")
    seen: list[str] = []
    plan = FFmpegPlan(
        mode="final",
        input_path="unused",
        output_path=str(output),
        steps=[
            FFmpegStep(
                "encode",
                [
                    sys.executable,
                    "-c",
                    "import pathlib, sys; "
                    "print('target=' + sys.argv[1], flush=True); "
                    "pathlib.Path(sys.argv[1]).write_text('new')",
                    str(output),
                ],
            )
        ],
        filter_graph="",
    )
    result = FFmpegWorker().run_plan(
        plan, on_progress=lambda progress: seen.append(progress["target"])
    )
    assert result.ok is True
    # FFmpeg writes the partial name, never the final path, while it runs.
    assert seen == [str(output.with_name("output.partial.mp4"))]
    assert output.read_text() == "new"
    assert not output.with_name("output.partial.mp4").exists()


def test_a_missing_ffmpeg_binary_is_a_worker_error(tmp_path: Path) -> None:
    plan = FFmpegPlan(
        mode="preview",
        input_path="unused",
        output_path=str(tmp_path / "out.mp4"),
        steps=[FFmpegStep("encode", [str(tmp_path / "no-ffmpeg"), "out.mp4"])],
        filter_graph="",
    )
    with pytest.raises(WorkerError, match="failed to start ffmpeg for step encode"):
        FFmpegWorker().run_plan(plan)


def test_an_argv_element_with_a_newline_is_refused(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    plan = FFmpegPlan(
        mode="preview",
        input_path="unused",
        output_path=str(tmp_path / "out.mp4"),
        steps=[
            FFmpegStep(
                "prepare",
                [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()\n"],
            )
        ],
        filter_graph="",
    )
    with pytest.raises(TemplateError, match="newlines"):
        FFmpegWorker().run_plan(plan)
    assert not marker.exists()


@requires_ffmpeg
def test_job_run_dry_run_and_preview(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    source = _make_clip(tmp_path / "clip.mp4")
    store = JobStore(jobs_dir)
    manifest = store.create(source, job_id="abcd1234abcd1234")
    facts = probe_file(source)
    write_media_facts(store.job_dir(manifest.job_id), facts)
    store.transition(manifest.job_id, JobState.PROBED)

    recipe = empty_recipe(manifest.job_id, str(source.resolve()))
    for op in recipe.ops:
        if op.op is OpName.CONTRAST:
            op.enabled = True
            op.params = {"contrast": 1.05, "brightness": 0.0, "gamma": 1.0}
    recipe.target.preview_window.duration_s = 0.4
    recipe.encode.preset = "ultrafast"
    write_recipe(store.job_dir(manifest.job_id), recipe)

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "run",
                manifest.job_id,
                "--mode",
                "preview",
                "--dry-run",
            ]
        )
        == 0
    )
    dry = json.loads(capsys.readouterr().out)
    assert "eq=contrast=1.05" in dry["filter_graph"]
    encode = dry["steps"][-1]["argv"]
    assert encode[encode.index("-vf") + 1] == dry["filter_graph"]
    assert encode[encode.index("-t") + 1] == "0.4"
    # A dry run validates the recipe but encodes nothing.
    assert not (store.job_dir(manifest.job_id) / "preview.mp4").exists()
    assert store.load(manifest.job_id).state is JobState.VALIDATED

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "run",
                manifest.job_id,
                "--mode",
                "preview",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["validation"]["ok"] is True
    updated = store.load(manifest.job_id)
    assert updated.state is JobState.PREVIEWED
    assert (store.job_dir(manifest.job_id) / "preview.mp4").is_file()


def test_job_run_final_requires_approve(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The approval gate comes before any FFmpeg work, so no ffmpeg is needed.
    store, job_id = _validated_job(tmp_path)
    code = cli.main(
        ["--jobs-dir", str(store.root), "job", "run", job_id, "--mode", "final"]
    )
    assert code == 1
    assert "requires --approve" in capsys.readouterr().out
    assert store.load(job_id).state is JobState.PROBED
    assert not (store.job_dir(job_id) / "output.mp4").exists()


def _validated_job(tmp_path: Path) -> tuple[JobStore, str]:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    store = JobStore(tmp_path / "jobs")
    job_id = store.create(source, job_id="abcd1234abcd1234").job_id
    facts = MediaFacts(
        schema_version=1,
        source_path=str(source.resolve()),
        format_name="mp4",
        format_long_name="MP4",
        duration_s=10.0,
        size_bytes=1_000_000,
        bit_rate=1_000_000,
        streams=[
            StreamFacts(
                index=0, codec_type="video", codec_name="h264", width=320, height=240
            )
        ],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )
    write_media_facts(store.job_dir(job_id), facts)
    store.transition(job_id, JobState.PROBED)
    write_recipe(store.job_dir(job_id), empty_recipe(job_id, str(source.resolve())))
    return store, job_id


def test_a_failed_final_encode_marks_the_job_failed_with_ffmpeg_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _validated_job(tmp_path)

    class FailingWorker:
        def run_plan(self, plan: FFmpegPlan, **_kwargs: object) -> RunResult:
            return RunResult(
                ok=False,
                output_path=plan.output_path,
                steps=[StepResult("encode", 1, 0.1, stderr_tail="No space left")],
            )

    monkeypatch.setattr(cli, "FFmpegWorker", FailingWorker)
    # The worker is faked, so no FFmpeg runs; only the plan needs a binary name.
    # Without this the test fails on CI's wheel run, which has no ffmpeg on PATH.
    monkeypatch.setattr(templates_mod, "which", lambda _name: "/usr/bin/ffmpeg")
    argv = ["--jobs-dir", str(store.root), "job", "run", job_id]
    assert cli.main([*argv, "--mode", "final", "--approve"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False
    manifest = store.load(job_id)
    assert manifest.state is JobState.FAILED
    assert manifest.error == "No space left"


def test_an_invalid_recipe_is_not_encoded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, job_id = _validated_job(tmp_path)
    recipe = empty_recipe(job_id, str(tmp_path / "clip.mp4"))
    for op in recipe.ops:
        if op.op is OpName.DENOISE:
            op.enabled = True
            op.params = {"strength": 5.0}
    write_recipe(store.job_dir(job_id), recipe)

    class NoWorker:
        def __init__(self) -> None:
            raise AssertionError("an invalid recipe reached the worker")

    monkeypatch.setattr(cli, "FFmpegWorker", NoWorker)
    argv = ["--jobs-dir", str(store.root), "job", "run", job_id, "--mode", "preview"]
    assert cli.main(argv) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "param_bounds" in {issue["code"] for issue in report["errors"]}
    assert store.load(job_id).state is JobState.PROBED


@requires_ffmpeg
def test_job_run_final_approve(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    source = _make_clip(tmp_path / "clip.mp4", duration=0.5)
    store = JobStore(jobs_dir)
    manifest = store.create(source, job_id="abcd1234abcd1234")
    write_media_facts(store.job_dir(manifest.job_id), probe_file(source))
    store.transition(manifest.job_id, JobState.PROBED)
    recipe = empty_recipe(manifest.job_id, str(source.resolve()))
    recipe.encode.preset = "ultrafast"
    recipe.encode.crf = 28
    write_recipe(store.job_dir(manifest.job_id), recipe)

    assert (
        cli.main(
            [
                "--jobs-dir",
                str(jobs_dir),
                "job",
                "run",
                manifest.job_id,
                "--mode",
                "final",
                "--approve",
            ]
        )
        == 0
    )
    capsys.readouterr()
    updated = store.load(manifest.job_id)
    assert updated.state is JobState.COMPLETE
    assert (store.job_dir(manifest.job_id) / "output.mp4").is_file()


@requires_ffmpeg
def test_job_validate_transitions_to_validated(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    source = _make_clip(tmp_path / "clip.mp4", duration=0.5)
    store = JobStore(jobs_dir)
    manifest = store.create(source, job_id="abcd1234abcd1234")
    write_media_facts(store.job_dir(manifest.job_id), probe_file(source))
    store.transition(manifest.job_id, JobState.PROBED)
    write_recipe(
        store.job_dir(manifest.job_id),
        empty_recipe(manifest.job_id, str(source.resolve())),
    )
    assert (
        cli.main(["--jobs-dir", str(jobs_dir), "job", "validate", manifest.job_id]) == 0
    )
    capsys.readouterr()
    assert store.load(manifest.job_id).state is JobState.VALIDATED
