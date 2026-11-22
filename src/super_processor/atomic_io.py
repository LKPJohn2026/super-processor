"""Small disk helpers that survive another thread or program holding a file.

On Windows, replacing a file that another thread has open (a page refresh
reading the wizard state while the render worker saves it) fails with
PermissionError, and so does opening it mid-replace. Both clear within
milliseconds, so reads and replaces retry briefly instead of failing.

Every write goes through its own temp file next to the target, named with a
fresh uuid, so two threads saving the same file at once never share one, and
a reader only ever sees the old file or the new one, never half of either.

Job folders hold the editor's own videos, Gemini's notes on them and the
wizard's state, so they are private to the user who runs the program: on
Linux and macOS folders made here are 0700 and files 0600, whatever the
umask (:func:`make_private_dir`, :func:`open_private_new`, and every
:func:`write_text_atomic`). A private folder also keeps out every file FFmpeg
writes in it, whatever that file's own mode. Windows has no such mode bits;
there a folder in the user's profile is already private to that user, so
these helpers change nothing.

This module imports only the standard library, so any module may use it.
"""

from __future__ import annotations

import os
import shutil
import stat
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

RETRIES = 20
RETRY_S = 0.025
PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
_OTHERS = 0o077  # group and other permission bits


def _posix() -> bool:
    """True where folders and files have owner/group/other mode bits."""
    return os.name == "posix"


def tighten(path: Path, mode: int = PRIVATE_DIR_MODE) -> None:
    """Set ``path`` to ``mode`` when this user owns it and others have access.

    A symlink, a path someone else owns and a path that is already private
    are left as they are, and so is everything on Windows.
    """
    if not _posix():
        return
    with suppress(OSError):
        info = os.lstat(path)
        if stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid():
            return
        if info.st_mode & _OTHERS:
            os.chmod(path, mode)


def make_private_dir(path: Path, *, exist_ok: bool = True) -> Path:
    """Create the folder ``path`` private to this user (0700).

    Parents that are missing are created too, with the usual mode: only the
    last folder is made private. A folder that is already there and that
    this user owns is tightened to 0700.
    """
    path.mkdir(mode=PRIVATE_DIR_MODE, parents=True, exist_ok=exist_ok)
    tighten(path)
    return path


def make_private_file(path: Path) -> None:
    """Make the file ``path`` readable by this user only (0600)."""
    tighten(path, PRIVATE_FILE_MODE)


def open_private_new(path: Path) -> BinaryIO:
    """Open a new file at ``path`` for writing bytes, private to this user.

    Fails with FileExistsError when ``path`` is already there.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    return os.fdopen(os.open(path, flags, PRIVATE_FILE_MODE), "wb")


def _write_private(path: Path, text: str) -> None:
    """Write ``text`` to the new file ``path``, readable by this user only.

    Text mode as in ``Path.write_text``: on Windows, newlines are written as
    CRLF by the text layer, so the descriptor itself is binary.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        with os.fdopen(
            os.open(path, flags, PRIVATE_FILE_MODE), "w", encoding="utf-8"
        ) as out:
            out.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` through a uuid temp file and one replace.

    A PermissionError on the replace is retried :data:`RETRIES` times; the
    last one is raised, with the old file untouched and no temp file left.
    The file is private to this user (0600 on Linux and macOS).
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    _write_private(tmp, text)
    try:
        for attempt in range(RETRIES):
            try:
                tmp.replace(path)
                return path
            except PermissionError:
                if attempt == RETRIES - 1:
                    raise
                time.sleep(RETRY_S)
    finally:
        tmp.unlink(missing_ok=True)
    raise AssertionError("unreachable")  # pragma: no cover


def read_text_retrying(path: Path) -> str:
    """Read ``path`` as UTF-8, retrying a PermissionError.

    The last error is raised as it is; callers turn it into their own error
    only after this gives up. Other errors (a missing file, bad UTF-8) are
    raised at once.
    """
    for attempt in range(RETRIES):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_S)
    raise AssertionError("unreachable")  # pragma: no cover


def unlink_retrying(path: Path) -> None:
    """Remove a file if it is there, retrying a PermissionError.

    The last error is raised as it is, so a file that must go and will not
    is never mistaken for one that went.
    """
    for attempt in range(RETRIES):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == RETRIES - 1:
                raise
            time.sleep(RETRY_S)


def unlink_quietly(path: Path) -> None:
    """Remove a file if it is there; a file that will not go is left."""
    with suppress(OSError):
        path.unlink(missing_ok=True)


def rmtree_quietly(path: Path) -> None:
    """Remove a folder and what is in it, as far as the disk lets us.

    A symlink is removed as a link and never followed, so a link to a folder
    elsewhere cannot empty that folder.
    """
    with suppress(OSError):
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
