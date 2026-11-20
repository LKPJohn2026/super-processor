"""The stop signal for a wizard run, and stopping the program a run started.

:class:`Cancelled` is a BaseException on purpose: the wizard catches
``GeminiError``, ``UpscaleError``, ``ShotError`` and ``OSError`` around its
steps to show the user a message, and a stop must pass through all of those
untouched until the run's own landing code catches it.

:func:`stop_process` ends an ffmpeg (or any other) child that a stopped run
leaves behind. On Windows it ends the whole process tree with ``taskkill``,
because ffmpeg started through a wrapper would otherwise keep encoding after
its parent is gone.

This module imports only the standard library, so any module may use it.
"""

from __future__ import annotations

import os
import subprocess
from contextlib import suppress
from typing import Any

TASKKILL_TIMEOUT_S = 10.0


class Cancelled(BaseException):  # noqa: N818 - a signal, not an error
    """The user stopped the run; unwind and let the run land the stop."""


def _is_windows() -> bool:
    """True on Windows. Tests patch this, never ``os.name``."""
    return os.name == "nt"


def stop_process(proc: subprocess.Popen[Any], grace_s: float = 1.0) -> None:
    """End ``proc`` and wait until it is gone.

    On POSIX it is asked to stop first (SIGTERM) and killed if it is still
    running after ``grace_s`` seconds. On Windows the process and every
    child it started are ended with ``taskkill /T /F``; if taskkill cannot
    run or fails, the process itself is killed. A process that has already
    exited is only reaped.
    """
    if proc.poll() is not None:
        return
    if _is_windows():
        _taskkill_tree(proc)
    else:
        with suppress(OSError):
            proc.terminate()
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            pass
        with suppress(OSError):
            proc.kill()
    proc.wait()


def _taskkill_tree(proc: subprocess.Popen[Any]) -> None:
    """End ``proc`` and its children on Windows, falling back to a kill."""
    try:
        completed = subprocess.run(  # noqa: S603 - argv list, no shell
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],  # noqa: S607
            capture_output=True,
            check=False,
            timeout=TASKKILL_TIMEOUT_S,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        ended = completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        ended = False
    if not ended and proc.poll() is None:
        with suppress(OSError):
            proc.kill()
