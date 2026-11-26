"""Local toolchain diagnostics for Super Processor."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .encoders import HARDWARE_DECODERS, HARDWARE_ENCODERS, SOFTWARE_ENCODER
from .jobs import default_jobs_root
from .tools import find_tool, resolve_argv


@dataclass(slots=True)
class CheckResult:
    """One doctor check with a machine-readable status."""

    name: str
    ok: bool
    detail: str
    path: str | None = None


def which(command: str) -> str | None:
    """Return the absolute path of ``command`` if it is on PATH.

    The search never looks in the current folder (see tools.find_tool), so
    an ``ffmpeg.exe`` left in a downloaded folder is not picked up.
    """
    found = find_tool(command)
    return str(Path(found).resolve()) if found else None


def run_version(command: str, *args: str) -> str | None:
    """Run a version command and return trimmed stdout/stderr on success."""
    executable = which(command)
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            resolve_argv([executable, *args]),
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


# Shorter values are never shown in part: four characters of them would
# give away too much.
_MIN_TAIL_LEN = 12


def mask_secret(value: str) -> str:
    """How doctor shows a key: that it is set, and at most its last 4 characters.

    The rest never appears, so a doctor report pasted into an issue or a
    chat does not give the key away.
    """
    cleaned = value.strip()
    if len(cleaned) < _MIN_TAIL_LEN:
        return "set"
    return f"set, ends in ...{cleaned[-4:]}"


def safe_url(value: str) -> str:
    """``value`` as scheme://host[:port]/path: no user name, password, query
    or fragment, which may hold a token."""
    try:
        parts = urllib.parse.urlsplit(value.strip())
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "unreadable URL"
    if not parts.scheme or not host:
        return "unreadable URL"
    if ":" in host:
        host = f"[{host}]"
    netloc = f"{host}:{port}" if port is not None else host
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def check_model_endpoint(
    name: str, env_var: str, *, secret: bool | None = None
) -> CheckResult:
    """Report configured model endpoints without contacting the network.

    A key (``secret``, the default for a ``*_API_KEY`` variable) is never
    printed: only that it is set and its last 4 characters. A URL is shown
    without credentials, query or fragment.
    """
    value = os.environ.get(env_var)
    if not value:
        return CheckResult(
            name=name,
            ok=True,
            detail=f"{env_var} unset (optional)",
        )
    if secret is None:
        secret = env_var.endswith(("_API_KEY", "_TOKEN", "_SECRET"))
    if secret:
        return CheckResult(
            name=name,
            ok=True,
            detail=f"{env_var} configured ({mask_secret(value)})",
        )
    return CheckResult(
        name=name,
        ok=True,
        detail=f"{env_var} configured",
        path=safe_url(value),
    )


def check_gemini_key() -> CheckResult:
    """Where the Gemini key comes from, never the key itself."""
    # Imported here: gemini imports models, which imports this module.
    from .gemini import gemini_api_key_source, resolve_gemini_api_key

    try:
        key = resolve_gemini_api_key()
        source = gemini_api_key_source()
    except Exception:  # noqa: BLE001 - a broken keyring must not stop doctor
        key, source = None, None
    if not key:
        return CheckResult(
            name="gemini_api_key",
            ok=True,
            detail="not set (paste one on the wizard's setup screen)",
        )
    return CheckResult(
        name="gemini_api_key",
        ok=True,
        detail=f"{mask_secret(key)}, from {source or 'an unknown place'}",
    )


def _ffmpeg_listing(kind: str) -> str:
    """Return stdout from ``ffmpeg -<filters|encoders>`` or empty string."""
    path = which("ffmpeg")
    if path is None:
        return ""
    try:
        completed = subprocess.run(
            resolve_argv([path, "-hide_banner", f"-{kind}"]),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout or ""


def check_ffmpeg_capability(
    name: str,
    *,
    kind: str,
    token: str,
    required: bool,
    hint: str,
) -> CheckResult:
    """Check whether ffmpeg exposes an encoder or filter capability."""
    listing = _ffmpeg_listing(kind)
    if not listing:
        return CheckResult(
            name=name,
            ok=not required,
            detail=(
                f"ffmpeg not found; cannot probe {token}"
                if which("ffmpeg") is None
                else f"unable to list ffmpeg {kind}"
            ),
        )
    present = any(
        parts and token in parts
        for line in listing.splitlines()
        for parts in [line.split()]
    )
    if present:
        return CheckResult(name=name, ok=True, detail=f"{token} available")
    return CheckResult(
        name=name,
        ok=not required,
        detail=f"{token} missing ({hint})",
    )


def check_gpu_video() -> list[CheckResult]:
    """Whether NVENC and NVDEC actually run, not only whether FFmpeg lists
    them (it lists them without a GPU). Optional: without them delivery is
    libx264 and decoding is FFmpeg's own, on the CPU."""
    from .upscale import cuda_decode_works, nvenc_works

    return [
        CheckResult(
            name="gpu:h264_nvenc",
            ok=True,
            detail=(
                "h264_nvenc encodes on this GPU; delivery uses it"
                if nvenc_works()
                else "h264_nvenc does not run here; delivery uses libx264 (CPU)"
            ),
        ),
        CheckResult(
            name="gpu:cuda_decode",
            ok=True,
            detail=(
                "CUDA decoding works; SPAN and pass-through shots decode on NVDEC"
                if cuda_decode_works()
                else "CUDA decoding does not run here; FFmpeg decodes on the CPU"
            ),
        ),
    ]


def check_dpir_weights() -> CheckResult:
    """Where DPIR's weights are, or that they are fetched on first use.

    Optional: only a scene whose note asks for more denoise uses DPIR. Only
    the size is checked here; the hash is checked before every load.
    """
    from . import dpir
    from .upscale import models_dir

    path = dpir.weights_path(models_dir())
    try:
        size = path.stat().st_size
    except OSError:
        size = -1
    if size == dpir.DPIR_SIZE:
        detail = f"DPIR weights are at {path}"
    elif size >= 0:
        detail = f"{path} is not the DPIR weights ({size} bytes); it is fetched again"
    elif dpir.download_allowed():
        detail = f"fetched from {dpir.DPIR_URL} the first time a note asks for DPIR"
    else:
        detail = (
            f"not at {path}, and {dpir.DPIR_DOWNLOAD_ENV} turns the download "
            f"off; download {dpir.DPIR_URL} there"
        )
    return CheckResult(name="model:dpir", ok=True, detail=detail, path=str(path))


def collect_doctor_report(jobs_dir: Path | None = None) -> list[CheckResult]:
    """Run all doctor checks and return structured results."""
    return [
        check_tool("ffmpeg", "ffmpeg", "-version"),
        check_tool("ffprobe", "ffprobe", "-version"),
        check_ffmpeg_capability(
            f"encoder:{SOFTWARE_ENCODER}",
            kind="encoders",
            token=SOFTWARE_ENCODER,
            required=True,
            hint="required for previews, delivery, and export",
        ),
        *check_gpu_video(),
        check_dpir_weights(),
        *[
            check_ffmpeg_capability(
                f"encoder:{name}",
                kind="encoders",
                token=name,
                required=False,
                hint="optional hardware encoder",
            )
            for name in HARDWARE_ENCODERS
        ],
        *[
            check_ffmpeg_capability(
                f"decoder:{name}",
                kind="decoders",
                token=name,
                required=False,
                hint="optional hardware decoder",
            )
            for name in HARDWARE_DECODERS
        ],
        check_ffmpeg_capability(
            "filter:stabilize",
            kind="filters",
            token="vidstabdetect",
            required=False,
            hint="falls back to deshake when absent",
        ),
        check_ffmpeg_capability(
            "filter:deshake",
            kind="filters",
            token="deshake",
            required=False,
            hint="soft stabilize fallback",
        ),
        check_ffmpeg_capability(
            "filter:libvmaf",
            kind="filters",
            token="libvmaf",
            required=False,
            hint="optional offline VMAF regression",
        ),
        check_player(),
        check_jobs_dir(jobs_dir),
        check_gemini_key(),
        check_model_endpoint("local_llm", "SUPER_PROCESSOR_LLM_BASE_URL"),
        check_model_endpoint("openai_api_key", "OPENAI_API_KEY"),
        check_model_endpoint("anthropic_api_key", "ANTHROPIC_API_KEY"),
    ]


def doctor_report_as_dict(results: list[CheckResult]) -> dict[str, Any]:
    """Serialize doctor results for JSON output."""
    required = {"ffmpeg", "ffprobe", f"encoder:{SOFTWARE_ENCODER}"}
    return {
        "ok": all(item.ok for item in results if item.name in required),
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
        "required tools: "
        + ("ready" if summary["ok"] else "missing ffmpeg/ffprobe/libx264")
    )
    return "\n".join(lines)


def doctor_json(results: list[CheckResult]) -> str:
    """Render doctor results as JSON text."""
    return json.dumps(doctor_report_as_dict(results), indent=2, sort_keys=True)
