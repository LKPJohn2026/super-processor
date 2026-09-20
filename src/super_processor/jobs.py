"""Resumable on-disk job store and legal state transitions."""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

JOB_ID_PATTERN = re.compile(r"^[a-z0-9]{8,32}$")
MANIFEST_NAME = "manifest.json"
DEFAULT_JOBS_DIR_ENV = "SUPER_PROCESSOR_JOBS_DIR"


class JobState(str, Enum):
    """Legal states for a processing job."""

    IMPORTED = "imported"
    PROBED = "probed"
    DIAGNOSED = "diagnosed"
    PLANNED = "planned"
    VALIDATED = "validated"
    PREVIEWED = "previewed"
    APPROVED = "approved"
    ENCODING = "encoding"
    COMPLETE = "complete"
    FAILED = "failed"


# Directed edges: from_state -> allowed next states (FAILED is reachable from any
# non-terminal state; COMPLETE only from ENCODING).
_TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.IMPORTED: frozenset({JobState.PROBED, JobState.FAILED}),
    JobState.PROBED: frozenset({JobState.DIAGNOSED, JobState.FAILED}),
    JobState.DIAGNOSED: frozenset({JobState.PLANNED, JobState.FAILED}),
    JobState.PLANNED: frozenset({JobState.VALIDATED, JobState.FAILED}),
    JobState.VALIDATED: frozenset({JobState.PREVIEWED, JobState.FAILED}),
    JobState.PREVIEWED: frozenset({JobState.APPROVED, JobState.FAILED}),
    JobState.APPROVED: frozenset({JobState.ENCODING, JobState.FAILED}),
    JobState.ENCODING: frozenset({JobState.COMPLETE, JobState.FAILED}),
    JobState.COMPLETE: frozenset(),
    JobState.FAILED: frozenset(),
}


class JobError(Exception):
    """Base error for job-store operations."""


class JobNotFoundError(JobError):
    """Raised when a job id does not exist."""


class InvalidTransitionError(JobError):
    """Raised when a state change is not allowed."""


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_jobs_root() -> Path:
    """Resolve the default jobs directory from env or the user home."""
    override = os.environ.get(DEFAULT_JOBS_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".super-processor" / "jobs").resolve()


def generate_job_id() -> str:
    """Create a short, filesystem-safe job identifier."""
    return secrets.token_hex(8)


def validate_job_id(job_id: str) -> str:
    """Validate and return a job id."""
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise JobError(f"invalid job id: {job_id!r}")
    return job_id


@dataclass(slots=True)
class JobManifest:
    """Versioned on-disk description of a processing job."""

    job_id: str
    state: JobState
    source_path: str
    created_at: str
    updated_at: str
    schema_version: int = 1
    error: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest for JSON storage."""
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobManifest:
        """Deserialize a manifest from JSON data."""
        return cls(
            job_id=str(data["job_id"]),
            state=JobState(str(data["state"])),
            source_path=str(data["source_path"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            schema_version=int(data.get("schema_version", 1)),
            error=data.get("error"),
            notes=dict(data.get("notes") or {}),
        )


class JobStore:
    """Filesystem-backed store for resumable processing jobs."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_jobs_root()).resolve()

    def ensure_root(self) -> None:
        """Create the jobs root directory if needed."""
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        """Return the directory path for a job id."""
        return self.root / validate_job_id(job_id)

    def manifest_path(self, job_id: str) -> Path:
        """Return the manifest path for a job id."""
        return self.job_dir(job_id) / MANIFEST_NAME

    def create(self, source_path: Path, *, job_id: str | None = None) -> JobManifest:
        """Create a new job for a local media file."""
        resolved = source_path.expanduser().resolve()
        if not resolved.is_file():
            raise JobError(f"source is not a readable file: {resolved}")

        self.ensure_root()
        new_id = validate_job_id(job_id) if job_id else generate_job_id()
        destination = self.job_dir(new_id)
        if destination.exists():
            raise JobError(f"job already exists: {new_id}")

        now = utc_now_iso()
        manifest = JobManifest(
            job_id=new_id,
            state=JobState.IMPORTED,
            source_path=str(resolved),
            created_at=now,
            updated_at=now,
        )
        destination.mkdir(parents=True, exist_ok=False)
        self._write_manifest(manifest)
        return manifest

    def load(self, job_id: str) -> JobManifest:
        """Load an existing job manifest."""
        path = self.manifest_path(job_id)
        if not path.is_file():
            raise JobNotFoundError(f"job not found: {job_id}")
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise JobError(f"corrupt manifest for job {job_id}")
        return JobManifest.from_dict(data)

    def list_jobs(self) -> list[JobManifest]:
        """List all jobs under the store root, newest first."""
        self.ensure_root()
        manifests: list[JobManifest] = []
        for child in self.root.iterdir():
            if not child.is_dir():
                continue
            try:
                manifests.append(self.load(child.name))
            except (JobError, OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
        manifests.sort(key=lambda item: item.updated_at, reverse=True)
        return manifests

    def can_transition(self, current: JobState, target: JobState) -> bool:
        """Return whether ``current`` may move to ``target``."""
        return target in _TRANSITIONS[current]

    def transition(
        self,
        job_id: str,
        target: JobState,
        *,
        error: str | None = None,
        notes: dict[str, Any] | None = None,
    ) -> JobManifest:
        """Advance a job to ``target`` if the transition is legal."""
        manifest = self.load(job_id)
        if not self.can_transition(manifest.state, target):
            raise InvalidTransitionError(
                "cannot move job "
                f"{job_id} from {manifest.state.value} to {target.value}"
            )
        manifest.state = target
        manifest.updated_at = utc_now_iso()
        if target is JobState.FAILED:
            manifest.error = error or "job failed"
        else:
            manifest.error = None
        if notes:
            manifest.notes.update(notes)
        self._write_manifest(manifest)
        return manifest

    def update_notes(self, job_id: str, notes: dict[str, Any]) -> JobManifest:
        """Merge notes into an existing job without changing its state."""
        manifest = self.load(job_id)
        manifest.notes.update(notes)
        manifest.updated_at = utc_now_iso()
        self._write_manifest(manifest)
        return manifest

    def _write_manifest(self, manifest: JobManifest) -> None:
        """Atomically write a job manifest to disk."""
        path = self.manifest_path(manifest.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(
            prefix=".manifest-",
            suffix=".tmp",
            dir=path.parent,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp_name).replace(path)
        except Exception:
            Path(tmp_name).unlink(missing_ok=True)
            raise
