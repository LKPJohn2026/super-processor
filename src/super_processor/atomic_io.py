"""Small disk helpers that survive another thread or program holding a file.

On Windows, replacing a file that another thread has open (a page refresh
reading the wizard state while the render worker saves it) fails with
PermissionError, and so does opening it mid-replace. Both clear within
milliseconds, so reads and replaces retry briefly instead of failing.

Every write goes through its own temp file next to the target, named with a
fresh uuid, so two threads saving the same file at once never share one, and
a reader only ever sees the old file or the new one, never half of either.

This module imports only the standard library, so any module may use it.
"""

from __future__ import annotations

import shutil
import time
import uuid
from contextlib import suppress
from pathlib import Path

RETRIES = 20
RETRY_S = 0.025


def write_text_atomic(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` through a uuid temp file and one replace.

    A PermissionError on the replace is retried :data:`RETRIES` times; the
    last one is raised, with the old file untouched and no temp file left.
    """
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
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
