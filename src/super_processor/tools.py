"""Find the outside programs Super Processor runs, never in the current folder.

On Windows, starting a program by its bare name (``ffmpeg``) makes Windows
look in the folder the terminal is open in before it looks on PATH, and
``shutil.which`` does the same. An ``ffmpeg.exe`` left in a downloaded
folder would then run instead of the real one. So every program the package
starts goes through :func:`resolve_argv`, which swaps a bare name for the full
path of the copy on PATH, found by :func:`find_tool`. That search reads PATH
itself and skips empty, ``.`` and relative entries on every system, so the
current folder is never searched.

Windows' own programs (PowerShell, taskkill) are started from the Windows
folder by full path (:func:`windows_system_tool`), not looked up at all.

This module imports only the standard library, so any module may use it.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import sys
from collections.abc import Sequence

# Tells Windows (CreateProcess, and shutil.which on 3.12+) to leave the
# current folder out of its program search: a backstop for a program that
# is not on PATH at all and so keeps its bare name.
NO_CWD_SEARCH_ENV = "NoDefaultCurrentDirectoryInExePath"
_DEFAULT_PATHEXT = ".COM;.EXE;.BAT;.CMD"

# Put before each ``-i`` of a file (and before an ffprobe's file): FFmpeg may
# then open nothing but local files for that input, never a URL, even when
# the "video" is really a playlist that names one.
FILE_INPUT = ("-protocol_whitelist", "file")

# Output options for anything sent to Gemini: none of the source's metadata
# (a phone video's GPS location, title, author, comments, creation time)
# and none of its chapters go into the file.
NO_METADATA = ("-map_metadata", "-1", "-map_chapters", "-1")


def _is_windows() -> bool:
    """True on Windows. Tests pass ``windows=`` instead of patching this."""
    return sys.platform == "win32"


def _absolute(folder: str, *, windows: bool) -> bool:
    """True for a PATH entry that cannot mean the current folder.

    On Windows a drive (``C:\\tools``), a share (``\\\\host\\tools``) or a
    rooted path (``\\tools``, on the current drive) is fine; ``""``, ``.``,
    ``tools`` and ``C:tools`` (relative to that drive's current folder) are
    not.
    """
    if not folder or folder == ".":
        return False
    if not windows:
        return posixpath.isabs(folder)
    rest = ntpath.splitdrive(folder)[1]
    return rest[:1] in ("\\", "/")


def find_tool(
    name: str,
    *,
    search_path: str | None = None,
    windows: bool | None = None,
    pathext: str | None = None,
) -> str | None:
    """The full path of program ``name`` on PATH, or ``None``.

    Only absolute PATH entries are searched: an empty entry, ``.`` and any
    other relative entry would mean the current folder, so they are skipped.
    On Windows each PATHEXT ending is tried (``ffmpeg`` -> ``ffmpeg.exe``).
    A ``name`` that already holds a folder is not searched for: it is
    returned when it is a file, else ``None``.

    ``search_path``, ``windows`` and ``pathext`` default to this machine's
    PATH, system and PATHEXT; tests pass them to check either system's
    rules anywhere.
    """
    windows = _is_windows() if windows is None else windows
    paths = ntpath if windows else posixpath
    if not name:
        return None
    if paths.basename(name) != name:
        return name if os.path.isfile(name) else None
    raw = os.environ.get("PATH", os.defpath) if search_path is None else search_path
    endings = [""]
    if windows:
        known = (pathext or os.environ.get("PATHEXT") or _DEFAULT_PATHEXT).split(";")
        known = [ext.lower() for ext in known if ext]
        if paths.splitext(name)[1].lower() not in known:
            endings = known
    for entry in raw.split(";" if windows else ":"):
        folder = entry.strip().strip('"') if windows else entry
        if not _absolute(folder, windows=windows):
            continue
        for ending in endings:
            candidate = os.path.join(folder, name + ending)
            if not os.path.isfile(candidate):
                continue
            if windows or os.access(candidate, os.X_OK):
                return candidate
    return None


def guard_windows_search() -> None:
    """Keep Windows from looking in the current folder for programs."""
    if _is_windows():
        os.environ.setdefault(NO_CWD_SEARCH_ENV, "1")


def tool_path(name: str) -> str:
    """``name`` as a full path from PATH when it is a bare program name.

    A name that already holds a folder is returned as it is. A bare name
    that is not on PATH is returned as it is too, so the caller's own
    "ffmpeg not found" error still reads the same; on Windows the current
    folder is then kept out of the search by :func:`guard_windows_search`.
    """
    guard_windows_search()
    if (ntpath if _is_windows() else posixpath).basename(name) != name:
        return name
    return find_tool(name) or name


def resolve_argv(argv: Sequence[str]) -> list[str]:
    """``argv`` with its program swapped for its full path (:func:`tool_path`)."""
    if not argv:
        return list(argv)
    return [tool_path(str(argv[0])), *argv[1:]]


def windows_system_tool(*parts: str) -> str:
    """A program in Windows' System32 folder, by full path."""
    root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or r"C:\Windows"
    return ntpath.join(root, "System32", *parts)
