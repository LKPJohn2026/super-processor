"""Local toolchain diagnostics for Super Processor."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .jobs import default_jobs_root


@dataclass(slots=True)
class CheckResult:
    """One doctor check with a machine-readable status."""

    name: str
    ok: bool
    detail: str
    path: str | None = None


def which(command: str) -> str | None:
    """Return the absolute path of ``command`` if it is on PATH."""
    found = shutil.which(command)
    return str(Path(found).resolve()) if found else None


def run_version(command: str, *args: str) -> str | None:
    """Run a version command and return trimmed stdout/stderr on success."""
    executable = which(command)
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr or "").strip()
    if completed.returncode != 0 or not output:
        return None
    return output.splitlines()[0]


def disk_free_bytes(path: Path) -> int:
    """Return free bytes available on the filesystem containing ``path``."""
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(target).free


def check_tool(name: str, command: str, *version_args: str) -> CheckResult:
    """Check whether a CLI tool is available and runnable."""
    path = which(command)
    if path is None:
        return CheckResult(name=name, ok=False, detail=f"{command} not found on PATH")
    version = run_version(command, *version_args) if version_args else path
    detail = version or f"{command} found"
    return CheckResult(name=name, ok=True, detail=detail, path=path)


def check_player() -> CheckResult:
    """Prefer VLC, then ffplay, for preview playback."""
    for command, args in (("vlc", ("--version",)), ("ffplay", ("-version",))):
        result = check_tool(f"player:{command}", command, *args)
        if result.ok:
            return CheckResult(
                name="player",
                ok=True,
                detail=f"{command}: {result.detail}",
                path=result.path,
            )
    return CheckResult(
        name="player",
        ok=False,
        detail="neither vlc nor ffplay found on PATH",
    )


def check_jobs_dir(jobs_dir: Path | None = None) -> CheckResult:
    """Verify the jobs directory is writable and report free space."""
    root = (jobs_dir or default_jobs_root()).resolve()
    try:
        root.mkdir(parents=True, exist_ok=True)
        free = disk_free_bytes(root)
    except OSError as exc:
        return CheckResult(
            name="jobs_dir",
            ok=False,
            detail=f"cannot use jobs directory {root}: {exc}",
            path=str(root),
        )
    free_gib = free / (1024**3)
    return CheckResult(
        name="jobs_dir",
        ok=True,
        detail=f"writable; {free_gib:.2f} GiB free",
        path=str(root),
    )


def check_model_endpoint(name: str, env_var: str) -> CheckResult:
    """Report configured model endpoints without contacting the network."""
    value = os.environ.get(env_var)
    if not value:
        return CheckResult(
            name=name,
            ok=True,
            detail=f"{env_var} unset (optional)",
        )
    return CheckResult(
        name=name,
        ok=True,
        detail=f"{env_var} configured",
        path=value,
    )


def collect_doctor_report(jobs_dir: Path | None = None) -> list[CheckResult]:
    """Run all doctor checks and return structured results."""
    return [
        check_tool("ffmpeg", "ffmpeg", "-version"),
        check_tool("ffprobe", "ffprobe", "-version"),
        check_player(),
        check_jobs_dir(jobs_dir),
        check_model_endpoint("local_llm", "SUPER_PROCESSOR_LLM_BASE_URL"),
        check_model_endpoint("openai_api_key", "OPENAI_API_KEY"),
        check_model_endpoint("anthropic_api_key", "ANTHROPIC_API_KEY"),
    ]


def doctor_report_as_dict(results: list[CheckResult]) -> dict[str, Any]:
    """Serialize doctor results for JSON output."""
    return {
        "ok": all(item.ok for item in results if item.name in {"ffmpeg", "ffprobe"}),
        "checks": [asdict(item) for item in results],
    }


def format_doctor_text(results: list[CheckResult]) -> str:
    """Render a human-readable doctor report."""
    lines: list[str] = []
    for item in results:
        mark = "ok" if item.ok else "FAIL"
        location = f" ({item.path})" if item.path else ""
        lines.append(f"[{mark}] {item.name}: {item.detail}{location}")
    summary = doctor_report_as_dict(results)
    lines.append(
        "required tools: " + ("ready" if summary["ok"] else "missing ffmpeg/ffprobe")
    )
    return "\n".join(lines)


def doctor_json(results: list[CheckResult]) -> str:
    """Render doctor results as JSON text."""
    return json.dumps(doctor_report_as_dict(results), indent=2, sort_keys=True)
