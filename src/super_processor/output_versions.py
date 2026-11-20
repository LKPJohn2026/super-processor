"""Earlier results of a job, kept under ``outputs/`` as ``v001.mp4``, ``v002.mp4``…

Every render replaces ``output.mp4``. Before this module, a revise threw the
previous result away; now each finished render is kept as a numbered
version, so the editor can go back to any of them.

- A version is made from ``output.mp4`` through a temp file next to its
  final name (``.<uuid>.part``), then given that name without ever
  overwriting one: two archives at once (two threads, or two servers on the
  same folder) end up as two numbers, never one file.
- On POSIX the version is a hard link to the output, which costs no space
  and no time. That is safe because the renderer never writes into
  ``output.mp4``: it replaces it with a new file (see
  ``super_processor.upscale._commit_output``), so the link keeps the old
  picture. On Windows a version is a full copy (see :func:`can_link`).
- The delivery marker beside the output (``output.mp4.delivery.json``) is
  copied beside the version, never linked: it is rewritten in place.
- ``outputs/versions.json`` lists the versions with what made each one. A
  ``v*.mp4`` the index does not know (an archive that stopped between its
  rename and the index write, or a file put there by hand) is listed as
  ``recovered`` and written into the index by the next archive.
- Nothing is ever pruned: every version is kept until the editor deletes the
  job folder.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .atomic_io import read_text_retrying, unlink_quietly, write_text_atomic
from .jobs import utc_now_iso
from .upscale import delivery_marker_path

OUTPUTS_DIR = "outputs"
VERSIONS_INDEX = "versions.json"
OUTPUT_NAME = "output.mp4"
_VERSION_NAME = re.compile(r"v(\d{3,6})\.mp4")
_INDEX_VERSION = 1

VersionKind = Literal["render", "earlier", "recovered"]
_KINDS: tuple[str, ...] = ("render", "earlier", "recovered")

# One archive at a time in this process: numbering and the index write. Two
# processes are kept apart by the exclusive final name alone.
_VERSIONS_LOCK = threading.RLock()


class VersionError(RuntimeError):
    """A version could not be kept."""


@dataclass(frozen=True)
class OutputVersion:
    """One kept result.

    ``kind`` says where it came from: ``render`` (kept when its render
    finished), ``earlier`` (an output from before versions were kept, or
    one a failed step left, kept before it was replaced) or ``recovered``
    (found in ``outputs/`` without an index entry). ``size`` and
    ``mtime_ns`` are those of ``output.mp4`` when it was kept, which is how
    :func:`current_version` knows the output is this version.
    """

    number: int
    kind: VersionKind
    created_at: str
    size: int
    mtime_ns: int
    note: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return version_name(self.number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "file": self.name,
            "kind": self.kind,
            "created_at": self.created_at,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "note": self.note,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputVersion:
        number = data["number"]
        if not isinstance(number, int) or isinstance(number, bool) or number < 1:
            raise ValueError(f"bad version number {number!r}")
        kind = data.get("kind")
        note = data.get("note")
        extra = data.get("extra")
        return cls(
            number=number,
            kind=kind if kind in _KINDS else "recovered",
            created_at=str(data.get("created_at") or ""),
            size=int(data.get("size") or 0),
            mtime_ns=int(data.get("mtime_ns") or 0),
            note=note if isinstance(note, str) and note else None,
            extra=dict(extra) if isinstance(extra, dict) else {},
        )


def version_name(number: int) -> str:
    """``v001.mp4`` for 1: the file name of a version."""
    return f"v{number:03d}.mp4"


def outputs_dir(job_dir: Path) -> Path:
    return job_dir / OUTPUTS_DIR


def can_link() -> bool:
    """Whether a version can be a hard link to the output instead of a copy.

    Not on Windows: there a link shares the file's sharing rules, so a
    player holding a version open could keep the next render from
    replacing ``output.mp4``.
    """
    return os.name != "nt" and hasattr(os, "link")


def list_versions(job_dir: Path) -> list[OutputVersion]:
    """Every kept version of a job, oldest first. Read-only.

    Index entries whose file is gone are left out; files the index does not
    know are listed as ``recovered``. A missing or unreadable index lists
    the files alone.
    """
    folder = outputs_dir(job_dir)
    if not folder.is_dir():
        return []
    on_disk = _files_on_disk(folder)
    known = {
        version.number: version
        for version in _read_index(folder)
        if version.number in on_disk
    }
    for number, path in on_disk.items():
        if number not in known:
            known[number] = _recovered(number, path)
    return [known[number] for number in sorted(known)]


def current_version(job_dir: Path) -> OutputVersion | None:
    """The kept version ``output.mp4`` is, or ``None`` when it is not kept.

    The output matches a version when its size and modification time are
    the ones recorded when it was kept, or (with hard links) when it is the
    same file.
    """
    output = job_dir / OUTPUT_NAME
    try:
        stat = output.stat()
    except OSError:
        return None
    folder = outputs_dir(job_dir)
    for version in reversed(list_versions(job_dir)):
        if version.size == stat.st_size and version.mtime_ns == stat.st_mtime_ns:
            return version
        try:
            if os.path.samefile(output, folder / version.name):
                return version
        except OSError:
            continue
    return None


def archive_output(
    job_dir: Path,
    *,
    kind: VersionKind = "render",
    note: str | None = None,
    extra: dict[str, Any] | None = None,
    move: bool = False,
) -> OutputVersion | None:
    """Keep ``output.mp4`` as the next version, and return it.

    ``None`` when there is no output. An output that is already kept is not
    kept twice: its version is returned as it is. With ``move`` the output
    is renamed into ``outputs/`` (constant time, for a restart that is about
    to delete it); otherwise it is linked or copied and stays in place.

    Raises :class:`VersionError` (or the ``OSError``) when it cannot be kept;
    the output itself is never changed by a failed archive. Once the file is
    in ``outputs/`` the version counts as kept, even if its index entry or
    delivery marker could not be written.
    """
    output = job_dir / OUTPUT_NAME
    with _VERSIONS_LOCK:
        try:
            stat = output.stat()
        except FileNotFoundError:
            return None
        kept = current_version(job_dir)
        if kept is not None:
            return kept
        folder = outputs_dir(job_dir)
        folder.mkdir(exist_ok=True)
        part = folder / f".{uuid.uuid4().hex}.part"
        try:
            if move:
                os.replace(output, part)
            elif can_link():
                os.link(output, part)
            else:
                shutil.copy2(output, part)
            number = _claim_next(folder, part)
        except BaseException:
            if move and part.is_file() and not output.exists():
                # Put the output back where the restart expects to find it.
                with suppress(OSError):
                    os.replace(part, output)
            unlink_quietly(part)
            raise
        # The file is kept from here on. A marker or index that cannot be
        # written does not undo that: the version is then listed as
        # recovered, and the next archive writes it into the index.
        with suppress(OSError):
            _copy_marker(output, folder / version_name(number))
        version = OutputVersion(
            number=number,
            kind=kind,
            created_at=utc_now_iso(),
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            note=note,
            extra=dict(extra or {}),
        )
        with suppress(OSError):
            _write_index(folder, [*list_versions(job_dir), version])
        return version


def _claim_next(folder: Path, part: Path) -> int:
    """Give ``part`` the first free version name and return its number.

    The name is taken with a hard link (or, where the disk has none, an
    exclusive create), so it never replaces a version another archive just
    made; on a clash the next number is tried.
    """
    number = max(_files_on_disk(folder), default=0) + 1
    for _attempt in range(1000):
        final = folder / version_name(number)
        try:
            _place(part, final)
            return number
        except FileExistsError:
            number += 1
    raise VersionError("could not find a free version number")


def _place(part: Path, final: Path) -> None:
    """Rename ``part`` to ``final``, refusing when ``final`` exists."""
    try:
        os.link(part, final)
    except FileExistsError:
        raise
    except (OSError, NotImplementedError, AttributeError):
        # No hard links on this disk: reserve the name, then fill it.
        fd = os.open(final, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        os.replace(part, final)
        return
    part.unlink()


def _copy_marker(output: Path, version: Path) -> None:
    marker = delivery_marker_path(output)
    if marker.is_file():
        shutil.copy2(marker, delivery_marker_path(version))


def _files_on_disk(folder: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    try:
        children = list(folder.iterdir())
    except OSError:
        return found
    for child in children:
        match = _VERSION_NAME.fullmatch(child.name)
        if match is None or child.is_symlink() or not child.is_file():
            continue
        found[int(match.group(1))] = child
    return found


def _recovered(number: int, path: Path) -> OutputVersion:
    try:
        stat = path.stat()
        size, mtime_ns = stat.st_size, stat.st_mtime_ns
    except OSError:
        size, mtime_ns = 0, 0
    return OutputVersion(
        number=number, kind="recovered", created_at="", size=size, mtime_ns=mtime_ns
    )


def _read_index(folder: Path) -> list[OutputVersion]:
    path = folder / VERSIONS_INDEX
    try:
        data = json.loads(read_text_retrying(path))
    except (OSError, ValueError):
        return []
    entries = data.get("versions") if isinstance(data, dict) else None
    versions: list[OutputVersion] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            versions.append(OutputVersion.from_dict(entry))
        except (KeyError, TypeError, ValueError):
            continue
    return versions


def _write_index(folder: Path, versions: list[OutputVersion]) -> None:
    by_number = {version.number: version for version in versions}
    payload = {
        "version": _INDEX_VERSION,
        "versions": [by_number[number].to_dict() for number in sorted(by_number)],
    }
    write_text_atomic(folder / VERSIONS_INDEX, json.dumps(payload, indent=2) + "\n")
