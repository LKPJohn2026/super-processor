"""The stop signal and stopping a child process on POSIX and on Windows."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Any

import pytest

from super_processor import cancel
from super_processor.cancel import Cancelled, stop_process

posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX signals")


def _sleeper(*, ignore_term: bool = False) -> subprocess.Popen[str]:
    """Start a Python child that sleeps, once it has said it is ready."""
    setup = (
        "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        if ignore_term
        else ""
    )
    proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
        [
            sys.executable,
            "-c",
            f"{setup}print('ready', flush=True); import time; time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "ready"
    return proc


def test_cancelled_passes_through_handlers_for_ordinary_errors() -> None:
    with pytest.raises(Cancelled):
        try:
            raise Cancelled
        except Exception:  # noqa: BLE001 - the point of the test
            pytest.fail("Cancelled was caught as an ordinary error")


@posix_only
def test_posix_stop_asks_politely_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cancel, "_is_windows", lambda: False)
    with _sleeper() as proc:
        started = time.monotonic()
        stop_process(proc, grace_s=5.0)
    assert time.monotonic() - started < 5.0
    assert proc.returncode == -signal.SIGTERM


@posix_only
def test_posix_stop_kills_a_child_that_ignores_terminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cancel, "_is_windows", lambda: False)
    with _sleeper(ignore_term=True) as proc:
        started = time.monotonic()
        stop_process(proc, grace_s=0.2)
    assert time.monotonic() - started < 5.0
    assert proc.returncode == -signal.SIGKILL


@posix_only
def test_stop_on_a_finished_process_only_reaps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cancel, "_is_windows", lambda: False)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603
    proc.wait()
    stop_process(proc)
    assert proc.returncode == 0


class _FakeProc:
    """A stand-in Popen that records how it was stopped."""

    pid = 4242

    def __init__(self) -> None:
        self.alive = True
        self.calls: list[str] = []

    def poll(self) -> int | None:
        return None if self.alive else 1

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")
        self.alive = False

    def wait(self, timeout: float | None = None) -> int:
        self.calls.append("wait")
        self.alive = False
        return 1


def test_windows_stop_ends_the_tree_with_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cancel, "_is_windows", lambda: True)
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    proc = _FakeProc()
    stop_process(proc)  # type: ignore[arg-type]
    assert seen == [["taskkill", "/T", "/F", "/PID", "4242"]]
    assert proc.calls == ["wait"]


@pytest.mark.parametrize(
    "outcome", ["fails", "missing"], ids=["taskkill-fails", "no-taskkill"]
)
def test_windows_stop_falls_back_to_kill(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setattr(cancel, "_is_windows", lambda: True)

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if outcome == "missing":
            raise FileNotFoundError(2, "taskkill")
        return subprocess.CompletedProcess(argv, 128, b"", b"not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    proc = _FakeProc()
    stop_process(proc)  # type: ignore[arg-type]
    assert proc.calls == ["kill", "wait"]
