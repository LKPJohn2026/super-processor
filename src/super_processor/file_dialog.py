"""Ask the desktop for a video file with its own file dialog.

The wizard runs on the editor's machine, so the server can open the system
file chooser and get the real path back; nothing is copied. On Linux zenity
goes through the XDG desktop portal (``GTK_USE_PORTAL=1``), so the dialog
is the desktop's own; kdialog is tried next. macOS uses ``osascript`` and
Windows a PowerShell ``OpenFileDialog``. Every one is run by its full path
(System32's PowerShell, ``/usr/bin/osascript``, zenity and kdialog from
PATH), never from the current folder.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from .tools import find_tool, resolve_argv, windows_system_tool

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".ts")
DIALOG_TIMEOUT_S = 15 * 60
_TITLE = "Choose a video for Super Processor"
# Each dialog program is started by full path, never found in the current
# folder (see tools.py).
OSASCRIPT = "/usr/bin/osascript"


class FileDialogError(RuntimeError):
    """Raised when no system file dialog can be opened."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


def _commands(start: Path) -> list[tuple[list[str], dict[str, str]]]:
    patterns = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
    found: list[tuple[list[str], dict[str, str]]] = []
    if sys.platform == "darwin":
        script = (
            f'POSIX path of (choose file with prompt "{_TITLE}" '
            f'default location POSIX file "{start}")'
        )
        found.append(([OSASCRIPT, "-e", script], {}))
    elif sys.platform == "win32":
        filters = ";".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d = New-Object System.Windows.Forms.OpenFileDialog;"
            f"$d.Title = '{_TITLE}'; $d.Filter = 'Videos|{filters}';"
            f"$d.InitialDirectory = '{start}';"
            "if ($d.ShowDialog() -eq 'OK') { $d.FileName }"
        )
        powershell = windows_system_tool("WindowsPowerShell", "v1.0", "powershell.exe")
        found.append(([powershell, "-NoProfile", "-STA", "-Command", script], {}))
    else:
        zenity = find_tool("zenity")
        if zenity:
            found.append(
                (
                    [
                        zenity,
                        "--file-selection",
                        f"--title={_TITLE}",
                        f"--filename={start}/",
                        f"--file-filter=Videos | {patterns}",
                    ],
                    {"GTK_USE_PORTAL": "1"},
                )
            )
        kdialog = find_tool("kdialog")
        if kdialog:
            found.append(
                (
                    [
                        kdialog,
                        "--title",
                        _TITLE,
                        "--getopenfilename",
                        str(start),
                        f"Videos ({patterns})",
                    ],
                    {},
                )
            )
    return found


def choose_video(
    start: Path | None = None, *, run: Runner = subprocess.run
) -> Path | None:
    """The file the editor chose, or ``None`` when they cancelled."""
    start = start or Path.home()
    commands = _commands(start)
    if not commands:
        raise FileDialogError(
            "no system file dialog is available here; drop the file or type its path"
        )
    failures: list[str] = []
    for argv, extra in commands:
        try:
            done = run(
                resolve_argv(argv),
                capture_output=True,
                text=True,
                timeout=DIALOG_TIMEOUT_S,
                env={**os.environ, **extra},
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"{argv[0]}: {exc}")
            continue
        chosen = done.stdout.strip().splitlines()
        if done.returncode == 0 and chosen:
            return Path(chosen[-1])
        if done.returncode == 1 and not done.stderr.strip():
            return None  # the editor closed the dialog
        failures.append(f"{argv[0]}: {done.stderr.strip() or done.returncode}")
    raise FileDialogError("the file dialog failed (" + "; ".join(failures) + ")")
