"""FFmpeg process runner with progress parsing, cancel, and atomic outputs."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .templates import FFmpegPlan, FFmpegStep, TemplateError


class WorkerError(RuntimeError):
    """Raised when an FFmpeg worker invocation fails."""


ProgressCallback = Callable[[dict[str, str]], None]


@dataclass(slots=True)
class StepResult:
    """Outcome of one FFmpeg step."""

    name: str
    returncode: int
    duration_s: float
    progress: dict[str, str] = field(default_factory=dict)
    stderr_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "returncode": self.returncode,
            "duration_s": self.duration_s,
            "progress": dict(self.progress),
            "stderr_tail": self.stderr_tail,
        }


@dataclass(slots=True)
class RunResult:
    """Outcome of executing a full FFmpeg plan."""

    ok: bool
    output_path: str
    steps: list[StepResult]
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "output_path": self.output_path,
            "cancelled": self.cancelled,
            "steps": [step.to_dict() for step in self.steps],
        }


def _creationflags() -> int:
    if os.name == "nt":  # pragma: no cover - exercised on Windows CI only
        return int(subprocess.CREATE_NEW_PROCESS_GROUP)  # type: ignore[attr-defined]
    return 0


def _start_new_session() -> bool:
    return os.name != "nt"


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":  # pragma: no cover
            proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        proc.terminate()


def _kill_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":  # pragma: no cover
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        proc.kill()


def parse_progress_line(line: str, progress: dict[str, str]) -> None:
    """Update ``progress`` from one ``-progress`` key=value line."""
    text = line.strip()
    if not text or "=" not in text:
        return
    key, _, value = text.partition("=")
    progress[key] = value


def format_progress_status(
    progress: dict[str, str],
    *,
    duration_s: float | None = None,
) -> str:
    """Render a short human-readable progress status for stderr."""
    out_ms = progress.get("out_time_ms")
    out_time = progress.get("out_time")
    seconds: float | None = None
    if out_ms is not None:
        with contextlib.suppress(ValueError):
            seconds = int(out_ms) / 1000.0
    if seconds is None and out_time is not None:
        parts = out_time.split(":")
        with contextlib.suppress(ValueError):
            if len(parts) == 3:
                hours, minutes, secs = parts
                seconds = int(hours) * 3600 + int(minutes) * 60 + float(secs)
    speed = progress.get("speed", "?")
    if seconds is None:
        return f"encoding… speed={speed}"
    if duration_s and duration_s > 0:
        pct = min(100.0, 100.0 * seconds / duration_s)
        return f"encoding {pct:5.1f}% ({seconds:.1f}/{duration_s:.1f}s) speed={speed}"
    return f"encoding {seconds:.1f}s speed={speed}"


class FFmpegWorker:
    """Run template plans as subprocesses without involving a shell."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: subprocess.Popen[str] | None = None
        self._cancel_requested = False

    def cancel(self) -> None:
        """Request cancellation of the active FFmpeg process group."""
        with self._lock:
            self._cancel_requested = True
            proc = self._current
        if proc is not None:
            _terminate_process_group(proc)

    def run_plan(
        self,
        plan: FFmpegPlan,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> RunResult:
        """Execute all steps; encode output is written atomically."""
        with self._lock:
            self._cancel_requested = False

        results: list[StepResult] = []
        final_output = Path(plan.output_path)

        for step in plan.steps:
            if step.name == "encode":
                result = self._run_encode_step(
                    step,
                    final_output=final_output,
                    on_progress=on_progress,
                )
            else:
                result = self._run_step(step, on_progress=None)
            results.append(result)
            if self._cancel_requested:
                return RunResult(
                    ok=False,
                    output_path=str(final_output),
                    steps=results,
                    cancelled=True,
                )
            if result.returncode != 0:
                return RunResult(
                    ok=False,
                    output_path=str(final_output),
                    steps=results,
                    cancelled=False,
                )

        return RunResult(
            ok=True,
            output_path=str(final_output),
            steps=results,
            cancelled=False,
        )

    def _run_encode_step(
        self,
        step: FFmpegStep,
        *,
        final_output: Path,
        on_progress: ProgressCallback | None,
    ) -> StepResult:
        final_output.parent.mkdir(parents=True, exist_ok=True)
        partial = final_output.with_name(
            f"{final_output.stem}.partial{final_output.suffix}"
        )
        if partial.exists():
            partial.unlink()

        argv = list(step.argv)
        if not argv:
            raise TemplateError("encode step has empty argv")
        # Last argv element is the planned output path from the template.
        argv[-1] = str(partial)

        result = self._run_argv(step.name, argv, on_progress=on_progress)
        if result.returncode == 0 and not self._cancel_requested:
            partial.replace(final_output)
        else:
            partial.unlink(missing_ok=True)
        return result

    def _run_step(
        self,
        step: FFmpegStep,
        *,
        on_progress: ProgressCallback | None,
    ) -> StepResult:
        return self._run_argv(step.name, list(step.argv), on_progress=on_progress)

    def _run_argv(
        self,
        name: str,
        argv: list[str],
        *,
        on_progress: ProgressCallback | None,
    ) -> StepResult:
        if not argv:
            raise TemplateError(f"step {name!r} has empty argv")
        if any("\n" in part or "\r" in part for part in argv):
            raise TemplateError("argv elements must not contain newlines")

        started = time.monotonic()
        progress: dict[str, str] = {}
        stderr_chunks: list[str] = []

        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=_start_new_session(),
                creationflags=_creationflags(),
            )
        except OSError as exc:
            raise WorkerError(f"failed to start ffmpeg for step {name}: {exc}") from exc

        with self._lock:
            self._current = proc
            cancel_now = self._cancel_requested

        if cancel_now:
            _terminate_process_group(proc)

        def _read_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                parse_progress_line(line, progress)
                if on_progress is not None:
                    on_progress(dict(progress))
                if progress.get("progress") == "end":
                    break

        def _read_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_chunks.append(line)
                # Keep memory bounded for long encodes.
                if len(stderr_chunks) > 200:
                    del stderr_chunks[:50]

        stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stdout_thread.start()
        stderr_thread.start()

        returncode = proc.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        if proc.stdout is not None:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()

        with self._lock:
            self._current = None
            cancelled = self._cancel_requested

        if cancelled and returncode == 0:
            # Rare race: process exited successfully as cancel arrived.
            pass
        elif cancelled and returncode not in (0, -signal.SIGTERM, -15):
            _kill_process_group(proc)

        stderr_tail = "".join(stderr_chunks[-40:]).strip()
        return StepResult(
            name=name,
            returncode=returncode if not cancelled else (returncode or -signal.SIGTERM),
            duration_s=time.monotonic() - started,
            progress=progress,
            stderr_tail=stderr_tail,
        )
