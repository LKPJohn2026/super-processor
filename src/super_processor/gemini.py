"""Gemini structured-output client for the localhost wizard.

Uses the legacy Generative Language ``models.generateContent`` REST API
(``v1beta``), not the Interactions API. That API is stateless: every call
names its model in the URL and must resend ``systemInstruction``,
``generationConfig``, and the full ``contents`` history. There is no
``previous_interaction_id``. Switching models repeats that same body against
the next model. ``responseMimeType`` / ``responseSchema`` keep replies
parseable.

The wizard asks Gemini to cut the video into segments (watching an
audio-free copy uploaded through the File API, sampled at 1 fps), choose the
output scale, plan and check per-shot looks, and turn a result note into a
time range, scale, and strength. Transport is injectable for tests.
"""

from __future__ import annotations

import base64
import contextlib
import http.client
import json
import logging
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .atomic_io import write_text_atomic
from .models import resolve_secret
from .textline import one_line

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
CONFIG_DIR_ENV = "SUPER_PROCESSOR_CONFIG_DIR"
CREDENTIALS_FILE = "credentials.json"
_CREDENTIALS_KEY = "gemini_api_key"
GEMINI_MODEL_ENV = "SUPER_PROCESSOR_GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
CHAT_FILE_NAME = "gemini_chat.json"
# One process-wide lock around every read-modify-write of a chat file, so two
# threads appending at once (a foreground call and a prefetch, or a restart
# note) never drop each other's turns. Reentrant, so a holder can save.
_CHAT_LOCK = threading.RLock()
CHAT_VERSION = 1
# Production revise loops until the user picks an option; tests use this cap.
_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
# Names that cannot diagnose frames/video for the wizard.
_EXCLUDE_NAME_PARTS = (
    "embedding",
    "embed-content",
    "tts",
    "imagen",
    "image-generation",
    "-image",
    "aqa",
    "gecko",
    "learnlm",
    "customtools",
)
_RETRYABLE_STATUS = frozenset({404, 429, 500, 503})
_EXHAUSTING_STATUS = frozenset({404, 429})
_TRANSIENT_STATUS = frozenset({500, 503})
MAX_CANDIDATE_MODELS = 8
_TRANSIENT_RETRIES = 2
# Per-model read timeout. A model that stops answering is skipped for the next
# candidate, so a full failover stays close to one slow call.
_REQUEST_TIMEOUT_S = 60
# A request that carries video waits for Gemini to watch it; a 20-minute video
# at 1 fps is about 1200 frames. A timeout here ends the failover (see
# _generate_with_failover): every model would read the same video.
_VIDEO_REQUEST_TIMEOUT_S = 300
_UPLOAD_URL = "https://generativelanguage.googleapis.com/upload/v1beta/files"
_FILES_URL = "https://generativelanguage.googleapis.com/v1beta/files"
# Uploads go in pieces so the wizard can show how far they are. The File API
# wants every piece but the last to be a multiple of 256 KiB.
_UPLOAD_CHUNK_BYTES = 8 * 1024 * 1024
# How often, and for how long, to wait for an upload to become ACTIVE.
_FILE_POLL_S = 2.0
_FILE_ACTIVE_TIMEOUT_S = 600.0
# Frames per second Gemini samples from a video (its own default).
VIDEO_FPS = 1.0
# The key goes in this header, never in a URL: URLs end up in logs, proxies
# and error messages (http.client quotes the whole path in some errors).
API_KEY_HEADER = "x-goog-api-key"
# A File API name as Gemini hands it out (``files/abc-123``).
_FILE_NAME = re.compile(r"files/[A-Za-z0-9][A-Za-z0-9_-]{0,127}", re.ASCII)
# Said instead of the text of an error that may quote the request.
_INVALID_REQUEST = "invalid request (check the API key on the setup screen)"
_BAD_KEY_MESSAGE = (
    "that does not look like a Gemini API key: it has spaces, line breaks or "
    "other characters a key never has; copy it again from Google AI Studio"
)

_LOG = logging.getLogger(__name__)

# Keys this process has seen, so any text it stores or shows can have them
# taken out (see redact_secrets). Shorter strings are never treated as keys,
# so a test key such as "k" does not blank out every letter k.
_MIN_SECRET_LEN = 16
_SECRETS: set[str] = set()
_SECRETS_LOCK = threading.Lock()
# A Gemini key (AIza and 35 more characters) standing on its own, and a
# ``key=`` query parameter in a URL quoted by an error.
_KEY_LIKE = re.compile(r"(?<![0-9A-Za-z_-])AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])")
_KEY_PARAM = re.compile(r"([?&](?:key|api_key)=)[^&\s'\"]+")
REDACTED = "***"


def remember_secret(value: str | None) -> None:
    """Treat ``value`` as a secret from now on (see :func:`redact_secrets`)."""
    if not value:
        return
    with _SECRETS_LOCK:
        for item in (value, value.strip()):
            if len(item) >= _MIN_SECRET_LEN:
                _SECRETS.add(item)


def redact_secrets(text: str) -> str:
    """``text`` with every Gemini key in it replaced by ``***``.

    It takes out the keys this process has used or saved, the key in the
    environment, anything shaped like a Gemini key, and ``key=`` query
    parameters. Used on every error, progress line and page the wizard
    stores or shows, so a key never ends up in ``wizard_state.json``.
    """
    if not text:
        return text
    with _SECRETS_LOCK:
        known = set(_SECRETS)
    env = os.environ.get(GEMINI_API_KEY_ENV, "")
    for item in (env, env.strip()):
        if len(item) >= _MIN_SECRET_LEN:
            known.add(item)
    for secret in sorted(known, key=len, reverse=True):
        if secret in text:
            text = text.replace(secret, REDACTED)
    text = _KEY_LIKE.sub(REDACTED, text)
    return _KEY_PARAM.sub(lambda m: m.group(1) + REDACTED, text)


def redact_payload(value: Any) -> Any:
    """A copy of a JSON-like ``value`` with :func:`redact_secrets` applied
    to every string in it (keys of objects included)."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {
            redact_payload(key) if isinstance(key, str) else key: redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    return value


def key_looks_valid(api_key: str) -> bool:
    """True when ``api_key`` holds only the visible ASCII a key is made of.

    A key with a space or a line break in it (copied across a wrapped line)
    never works, and used to end up quoted in error messages.
    """
    return bool(api_key) and all("!" <= char <= "~" for char in api_key)


def _key_headers(api_key: str, **extra: str) -> dict[str, str]:
    if not key_looks_valid(api_key):
        raise GeminiError(_BAD_KEY_MESSAGE)
    return {API_KEY_HEADER: api_key, **extra}


def _file_id(name: str) -> str:
    """The id part of a File API name; refuses anything else."""
    if not _FILE_NAME.fullmatch(name):
        raise GeminiError("that is not a Gemini file name")
    return name.removeprefix("files/")


def finite_json_number(text: str) -> float | None:
    """``parse_float`` for model replies: a number too big for a float
    (``1e999`` would read as infinity) is read as null."""
    value = float(text)
    return value if math.isfinite(value) else None


def bounded_json_int(text: str) -> int | None:
    """``parse_int`` for model replies: no reply field needs more than 20
    digits, and a longer one could not become a float either."""
    return int(text) if len(text.lstrip("-")) <= 20 else None


def drop_json_constant(name: str) -> None:
    """``parse_constant`` for model replies: ``NaN`` and ``Infinity`` are
    not JSON, and Python's reader would otherwise accept them. They are
    read as null, which every reader of a reply already treats as a
    missing value (a span's strength then fails to validate)."""
    return None


def loads_finite(text: str) -> Any:
    """``json.loads`` that reads NaN, Infinity and numbers out of range as
    null, so no non-finite number ever reaches a plan."""
    return json.loads(
        text,
        parse_constant=drop_json_constant,
        parse_float=finite_json_number,
        parse_int=bounded_json_int,
    )


class GeminiError(RuntimeError):
    """Raised when Gemini configuration or structured parse fails."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class GeminiTransport(Protocol):
    """HTTP transport for Gemini generateContent calls."""

    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the parsed JSON response body."""


class GeminiFileUploader(Protocol):
    """Optional transport capability for the File API (video uploads)."""

    def upload_file(
        self,
        *,
        api_key: str,
        path: Path,
        mime_type: str,
        display_name: str,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        """Upload ``path`` and return the File resource."""

    def get_file(self, *, api_key: str, name: str) -> dict[str, Any]:
        """Return the File resource called ``name`` (``files/...``)."""


@dataclass(frozen=True)
class GeminiVideo:
    """A video uploaded through the File API, valid until ``expires_at``."""

    name: str
    uri: str
    mime_type: str = "video/mp4"
    # Seconds since the epoch; uploads are deleted after about 48 hours.
    expires_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "mime_type": self.mime_type,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> GeminiVideo | None:
        if not isinstance(data, dict):
            return None
        name, uri = data.get("name"), data.get("uri")
        if not isinstance(name, str) or not _FILE_NAME.fullmatch(name):
            return None
        if not isinstance(uri, str) or not _plain_uri(uri):
            return None
        return cls(
            name=name,
            uri=uri,
            mime_type=_video_mime(data.get("mime_type")),
            expires_at=_finite_seconds(data.get("expires_at")),
        )

    def usable(self, now: float, *, margin_s: float = 3600.0) -> bool:
        """True while the upload has at least ``margin_s`` left to live."""
        return self.expires_at - margin_s > now


# What Gemini is sent back about an upload: its address and type, bounded.
_MAX_URI_LEN = 2048
_VIDEO_MIME = re.compile(r"video/[a-z0-9.+-]{1,64}", re.ASCII)


def _plain_uri(raw: Any) -> bool:
    """True for a non-empty address of visible ASCII, no longer than a URL."""
    return (
        isinstance(raw, str)
        and 0 < len(raw) <= _MAX_URI_LEN
        and all("!" <= char <= "~" for char in raw)
    )


def _video_mime(raw: Any) -> str:
    """A video type as the File API gives it; video/mp4 for anything else."""
    return raw if isinstance(raw, str) and _VIDEO_MIME.fullmatch(raw) else "video/mp4"


def _finite_seconds(raw: Any) -> float:
    """A stored expiry as a finite float; 0 (expired) for anything else."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return 0.0
    try:
        value = float(raw)
    except OverflowError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def video_part(
    video: GeminiVideo,
    *,
    start_s: float | None = None,
    end_s: float | None = None,
    fps: float = VIDEO_FPS,
) -> dict[str, Any]:
    """A ``fileData`` part Gemini samples at ``fps``, clipped when asked."""
    metadata: dict[str, Any] = {"fps": fps}
    if start_s is not None:
        metadata["startOffset"] = f"{max(0.0, start_s):.3f}s"
    if end_s is not None:
        metadata["endOffset"] = f"{max(0.0, end_s):.3f}s"
    return {
        "fileData": {"mimeType": video.mime_type, "fileUri": video.uri},
        "videoMetadata": metadata,
    }


def _has_video(body: dict[str, Any]) -> bool:
    return any(
        "fileData" in part
        for content in body.get("contents", [])
        if isinstance(content, dict)
        for part in content.get("parts", [])
        if isinstance(part, dict)
    )


def _parse_expiry(raw: Any) -> float:
    """Seconds since the epoch from an RFC 3339 time; 0 when unreadable."""
    if not isinstance(raw, str) or not raw:
        return 0.0
    text = raw.replace("Z", "+00:00")
    # Python 3.10 reads at most six fractional digits.
    head, dot, rest = text.partition(".")
    if dot:
        count = len(rest) - len(rest.lstrip("0123456789"))
        digits, zone = rest[:count], rest[count:]
        # Python 3.10 reads only 3 or 6 fraction digits.
        text = f"{head}.{digits[:6].ljust(6, '0')}{zone}"
    try:
        return datetime.fromisoformat(text).timestamp()
    except (ValueError, OverflowError, OSError):
        # A naive time far from now can fall outside the platform's range
        # (OverflowError, or OSError on Windows).
        return 0.0


class GeminiModelLister(Protocol):
    """Optional transport capability for ``models.list`` discovery."""

    def list_models(self, *, api_key: str) -> list[dict[str, Any]]:
        """Return raw model objects from ``models.list``."""


@dataclass(slots=True)
class ModelInfo:
    """One Gemini model candidate for the wizard."""

    name: str
    display_name: str = ""
    description: str = ""
    methods: tuple[str, ...] = ()
    rpm_limit: int | None = None
    rpm_used: int | None = None

    @property
    def rpm_remaining(self) -> int | None:
        if self.rpm_limit is None or self.rpm_used is None:
            return None
        return max(0, self.rpm_limit - self.rpm_used)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "methods": list(self.methods),
            "rpm_limit": self.rpm_limit,
            "rpm_used": self.rpm_used,
            "rpm_remaining": self.rpm_remaining,
        }


def _grouped(errors: list[str]) -> str:
    """``model: reason`` lines with the models that share a reason joined."""
    if not errors:
        return "no candidates"
    by_reason: dict[str, list[str]] = {}
    for line in errors:
        model, _, reason = line.partition(": ")
        by_reason.setdefault(reason, []).append(model)
    text = " ".join(
        "; ".join(
            f"{', '.join(models)}: {reason}" for reason, models in by_reason.items()
        ).split()
    )
    # Shown to the editor and saved with the job: one line, kept short.
    return text[:_MAX_GROUPED_LEN] + ("…" if len(text) > _MAX_GROUPED_LEN else "")


_MAX_GROUPED_LEN = 400
# A model id as models.list names it (``gemini-2.5-flash``); anything else is
# never tried, so a strange name cannot reach a URL or an error message.
_MODEL_NAME = re.compile(r"[a-z0-9][a-z0-9.-]{0,63}", re.ASCII)


def _strip_models_prefix(name: str) -> str:
    cleaned = name.strip()
    if cleaned.startswith("models/"):
        return cleaned[len("models/") :]
    return cleaned


def _parse_rpm_fields(raw: dict[str, Any]) -> tuple[int | None, int | None]:
    """Pull RPM used/limit when the list payload exposes them (AI Studio shape)."""
    limit: int | None = None
    used: int | None = None
    for key, target in (
        ("rpmLimit", "limit"),
        ("rpm_limit", "limit"),
        ("requestsPerMinuteLimit", "limit"),
        ("rpm", "limit"),
        ("rpmUsed", "used"),
        ("rpm_used", "used"),
        ("requestsPerMinuteUsed", "used"),
    ):
        if key not in raw:
            continue
        try:
            value = int(raw[key])
        except (TypeError, ValueError, OverflowError):
            continue
        if target == "limit":
            limit = value
        else:
            used = value
    # Nested rateLimits: { "rpm": "0/15" } as on the AI Studio rate-limit page.
    nested = raw.get("rateLimits") or raw.get("rate_limits") or {}
    if isinstance(nested, dict):
        rpm = nested.get("rpm") or nested.get("RPM")
        if isinstance(rpm, str) and "/" in rpm:
            left, right = rpm.split("/", 1)
            try:
                used = int(left.strip())
                limit = int(right.strip())
            except ValueError:
                pass
        elif isinstance(rpm, dict):
            try:
                if "used" in rpm:
                    used = int(rpm["used"])
                if "limit" in rpm:
                    limit = int(rpm["limit"])
            except (TypeError, ValueError, OverflowError):
                pass
    return used, limit


def parse_model_info(raw: dict[str, Any]) -> ModelInfo:
    """Normalize a ``models.list`` / rate-limit row into ``ModelInfo``."""
    name = _strip_models_prefix(str(raw.get("name") or raw.get("model") or ""))
    methods_raw = (
        raw.get("supportedGenerationMethods") or raw.get("supported_actions") or []
    )
    methods = (
        tuple(str(item) for item in methods_raw)
        if isinstance(methods_raw, list)
        else ()
    )
    used, limit = _parse_rpm_fields(raw)
    return ModelInfo(
        name=name,
        display_name=str(raw.get("displayName") or raw.get("display_name") or name),
        description=str(raw.get("description") or ""),
        methods=methods,
        rpm_limit=limit,
        rpm_used=used,
    )


def model_accepts_text_or_video(info: ModelInfo) -> bool:
    """Return True when the model can take text and image/video inputs."""
    if not info.name:
        return False
    lowered = info.name.lower()
    if any(part in lowered for part in _EXCLUDE_NAME_PARTS):
        return False
    if info.methods and "generateContent" not in info.methods:
        return False
    blob = f"{info.name} {info.display_name} {info.description}".lower()
    # Wizard sends text + stills (and can send video later). Accept multimodal
    # Gemini chat models; reject pure audio-TTS / embedding names above.
    if "gemini" not in lowered:
        return False
    if "video" in blob or "image" in blob or "multimodal" in blob:
        return True
    # Default Gemini flash/pro family accepts text + images/video.
    return any(token in lowered for token in ("flash", "pro", "lite"))


def model_has_usable_rpm(info: ModelInfo) -> bool:
    """Match AI Studio rows like ``0/5`` / ``0/10``: limit must be non-zero.

    When remaining is known, it must also be > 0 (not exhausted like ``5/5``).
    Unknown RPM metadata is treated as usable so we can probe the model.
    """
    if info.rpm_limit is not None and info.rpm_limit <= 0:
        return False
    return not (info.rpm_remaining is not None and info.rpm_remaining <= 0)


def select_candidate_models(
    models: list[ModelInfo],
    *,
    preferred: str | None = None,
) -> list[str]:
    """Order multimodal models with usable RPM, preferred name first."""
    usable = [
        item
        for item in models
        if model_accepts_text_or_video(item) and model_has_usable_rpm(item)
    ]
    names = [_strip_models_prefix(item.name) for item in usable if item.name]
    # De-dupe while preserving order.
    ordered: list[str] = []
    seen: set[str] = set()
    pref = _strip_models_prefix(preferred) if preferred else ""
    if pref:
        ordered.append(pref)
        seen.add(pref)
    for name in names:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _is_retryable_error(exc: GeminiError) -> bool:
    if exc.status in _RETRYABLE_STATUS:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "no longer available",
            "not found",
            "resource_exhausted",
            "rate limit",
            "quota",
            "unavailable",
            "timed out",
        )
    )


def _is_exhausting_error(exc: GeminiError) -> bool:
    """Return True when the model should be skipped for the rest of the session."""
    if exc.status in _EXHAUSTING_STATUS:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "no longer available",
            "not found",
            "resource_exhausted",
            "rate limit",
            "quota",
        )
    )


def _is_transient_error(exc: GeminiError) -> bool:
    if exc.status in _TRANSIENT_STATUS:
        return True
    return (
        "unavailable" in str(exc).lower() or "service unavailable" in str(exc).lower()
    )


def _http_error_detail(exc: urllib.error.HTTPError) -> tuple[int, str]:
    """Read and close an ``HTTPError`` so pytest does not see ResourceWarnings."""
    try:
        detail = exc.read().decode("utf-8", errors="replace")
        status = int(exc.code)
    finally:
        exc.close()
    return status, detail


# Longest retry delay worth telling the editor about: one week.
_MAX_RETRY_DELAY_S = 7 * 24 * 3600.0


def _retry_text(seconds: float) -> str:
    if seconds >= 3600:
        return f"about {round(seconds / 3600)} h"
    if seconds >= 60:
        return f"about {round(seconds / 60)} min"
    return f"{round(seconds)} s"


def summarize_gemini_error(status: int, detail: str) -> str:
    """One readable line for an error body from the Gemini API.

    The API answers with a JSON object several kilobytes long (help links,
    quota metrics, retry hints). The wizard shows this text to the editor,
    so keep what they can act on: what went wrong and when to try again.
    Words the failover looks for (``quota``, ``no longer available``) stay.
    """
    try:
        error = json.loads(detail).get("error") or {}
    except (ValueError, AttributeError, RecursionError):
        error = {}
    if not isinstance(error, dict):
        error = {}
    message = " ".join(str(error.get("message") or detail).split())
    reason = str(error.get("status") or "")
    retry: str | None = None
    daily = False
    details = error.get("details")
    for item in details if isinstance(details, list) else []:
        if not isinstance(item, dict):
            continue
        delay = str(item.get("retryDelay") or "")
        if delay.endswith("s"):
            with contextlib.suppress(ValueError):
                seconds = float(delay[:-1])
                # Only a delay a person could wait for; not NaN or forever.
                if 0 <= seconds <= _MAX_RETRY_DELAY_S:
                    retry = _retry_text(seconds)
        violations = item.get("violations")
        for violation in violations if isinstance(violations, list) else []:
            if isinstance(violation, dict) and "PerDay" in str(
                violation.get("quotaId", "")
            ):
                daily = True
        if item.get("reason") == "API_KEY_INVALID":
            return "the API key was rejected; replace it on the setup screen"
    if status == 429 or reason == "RESOURCE_EXHAUSTED":
        kind = "daily quota" if daily else "quota"
        text = f"{kind} used up"
        return f"{text}; try again in {retry}" if retry else text
    if status == 404 and "no longer available" in message:
        return "this model is no longer available"
    first = message.split(". ")[0].rstrip(".")
    return first[:160] + ("…" if len(first) > 160 else "")


# models.list is read a page at a time; this many pages, rows, and token
# characters are more than the API has ever needed.
_MAX_MODEL_PAGES = 20
_MAX_MODEL_ROWS = 2000
_MAX_PAGE_TOKEN_LEN = 1024


def _refuse_constant(name: str) -> None:
    """``parse_constant`` for API answers: NaN and Infinity are not JSON."""
    raise ValueError(f"{name} is not JSON")


def _api_object(raw: bytes, what: str) -> dict[str, Any]:
    """The JSON object in an API answer; anything else is a GeminiError.

    Bytes that are not UTF-8, text that is not JSON (NaN included), nesting
    too deep to read, and a root that is not an object are all refused.
    """
    try:
        parsed = json.loads(raw.decode("utf-8"), parse_constant=_refuse_constant)
    except RecursionError:
        raise GeminiError(f"{what} could not be read: nested too deep") from None
    except ValueError:
        # Covers bad JSON and bytes that are not UTF-8.
        raise GeminiError(f"{what} could not be read: not JSON") from None
    if not isinstance(parsed, dict):
        raise GeminiError(f"{what} root must be an object")
    return parsed


class UrllibGeminiTransport:
    """Default transport using ``urllib``."""

    def generate(
        self,
        *,
        model: str,
        api_key: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:  # pragma: no cover - exercised via FakeTransport in tests
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{urllib.parse.quote(model, safe='-._')}:generateContent"
        )
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=_key_headers(api_key, **{"Content-Type": "application/json"}),
            method="POST",
        )
        timeout = _VIDEO_REQUEST_TIMEOUT_S if _has_video(body) else _REQUEST_TIMEOUT_S
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status, detail = _http_error_detail(exc)
            raise GeminiError(
                f"Gemini HTTP {status}: {summarize_gemini_error(status, detail)}",
                status=status,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GeminiError(
                f"Gemini request failed: {redact_secrets(str(exc))}"
            ) from None
        except (http.client.HTTPException, ValueError):
            # An InvalidURL or a bad header value quotes the request.
            raise GeminiError(f"Gemini request failed: {_INVALID_REQUEST}") from None
        return _api_object(raw, "Gemini response")

    def upload_file(
        self,
        *,
        api_key: str,
        path: Path,
        mime_type: str,
        display_name: str,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:  # pragma: no cover - needs the network; faked in tests
        """Resumable File API upload, in pieces, reporting the share sent."""
        size = path.stat().st_size
        start = urllib.request.Request(
            _UPLOAD_URL,
            data=json.dumps({"file": {"display_name": display_name}}).encode(),
            headers=_key_headers(
                api_key,
                **{
                    "X-Goog-Upload-Protocol": "resumable",
                    "X-Goog-Upload-Command": "start",
                    "X-Goog-Upload-Header-Content-Length": str(size),
                    "X-Goog-Upload-Header-Content-Type": mime_type,
                    "Content-Type": "application/json",
                },
            ),
            method="POST",
        )
        try:
            with urllib.request.urlopen(start, timeout=_REQUEST_TIMEOUT_S) as response:
                upload_url = response.headers.get("X-Goog-Upload-URL", "")
            if not upload_url:
                raise GeminiError("Gemini did not return an upload address")
            offset = 0
            result = b""
            with path.open("rb") as handle:
                while True:
                    piece = handle.read(_UPLOAD_CHUNK_BYTES)
                    last = offset + len(piece) >= size
                    req = urllib.request.Request(
                        upload_url,
                        data=piece,
                        headers={
                            "X-Goog-Upload-Command": (
                                "upload, finalize" if last else "upload"
                            ),
                            "X-Goog-Upload-Offset": str(offset),
                            "Content-Length": str(len(piece)),
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(
                        req, timeout=_VIDEO_REQUEST_TIMEOUT_S
                    ) as response:
                        body = response.read()
                    offset += len(piece)
                    if on_progress is not None:
                        on_progress(offset / size if size else 1.0)
                    if last:
                        result = body
                        break
        except urllib.error.HTTPError as exc:
            status, detail = _http_error_detail(exc)
            raise GeminiError(
                f"Gemini upload HTTP {status}: "
                f"{summarize_gemini_error(status, detail)}",
                status=status,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GeminiError(
                f"Gemini upload failed: {redact_secrets(str(exc))}"
            ) from None
        except (http.client.HTTPException, ValueError):
            raise GeminiError(f"Gemini upload failed: {_INVALID_REQUEST}") from None
        file = _api_object(result, "Gemini upload").get("file")
        if not isinstance(file, dict):
            raise GeminiError("Gemini upload returned no file")
        return file

    def get_file(
        self, *, api_key: str, name: str
    ) -> dict[str, Any]:  # pragma: no cover - needs the network; faked in tests
        req = urllib.request.Request(
            f"{_FILES_URL}/{_file_id(name)}",
            headers=_key_headers(api_key),
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status, detail = _http_error_detail(exc)
            raise GeminiError(
                f"Gemini HTTP {status}: {summarize_gemini_error(status, detail)}",
                status=status,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GeminiError(
                f"Gemini request failed: {redact_secrets(str(exc))}"
            ) from None
        except (http.client.HTTPException, ValueError):
            raise GeminiError(f"Gemini request failed: {_INVALID_REQUEST}") from None
        return _api_object(raw, "Gemini file response")

    def delete_file(
        self, *, api_key: str, name: str
    ) -> None:  # pragma: no cover - needs the network; faked in tests
        """Delete the File API file ``name``; one already gone counts as done."""
        req = urllib.request.Request(
            f"{_FILES_URL}/{_file_id(name)}",
            headers=_key_headers(api_key),
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as response:
                response.read()
        except urllib.error.HTTPError as exc:
            status, detail = _http_error_detail(exc)
            if status == 404:
                return
            raise GeminiError(
                f"Gemini HTTP {status}: {summarize_gemini_error(status, detail)}",
                status=status,
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise GeminiError(
                f"Gemini delete failed: {redact_secrets(str(exc))}"
            ) from None
        except (http.client.HTTPException, ValueError):
            raise GeminiError(f"Gemini delete failed: {_INVALID_REQUEST}") from None

    def list_models(
        self, *, api_key: str
    ) -> list[dict[str, Any]]:  # pragma: no cover - mocked in unit tests
        models: list[dict[str, Any]] = []
        page_token = ""
        seen_tokens: set[str] = set()
        # A server that always names a next page must not keep this looping.
        for _page in range(_MAX_MODEL_PAGES):
            url = f"{_MODELS_URL}?pageSize=100"
            if page_token:
                url += f"&pageToken={urllib.parse.quote(page_token)}"
            req = urllib.request.Request(
                url, headers=_key_headers(api_key), method="GET"
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    raw = response.read()
            except urllib.error.HTTPError as exc:
                status, detail = _http_error_detail(exc)
                raise GeminiError(
                    f"Gemini HTTP {status}: {summarize_gemini_error(status, detail)}",
                    status=status,
                ) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise GeminiError(
                    f"Gemini list models failed: {redact_secrets(str(exc))}"
                ) from None
            except (http.client.HTTPException, ValueError):
                raise GeminiError(
                    f"Gemini list models failed: {_INVALID_REQUEST}"
                ) from None
            parsed = _api_object(raw, "Gemini models.list")
            batch = parsed.get("models") or []
            if isinstance(batch, list):
                models.extend(item for item in batch if isinstance(item, dict))
            token = parsed.get("nextPageToken")
            page_token = token if isinstance(token, str) else ""
            if (
                not page_token
                or page_token in seen_tokens
                or len(page_token) > _MAX_PAGE_TOKEN_LEN
                or len(models) >= _MAX_MODEL_ROWS
            ):
                break
            seen_tokens.add(page_token)
        return models[:_MAX_MODEL_ROWS]


@dataclass(slots=True)
class ChatTurn:
    """One persisted chat turn."""

    role: str
    text: str
    frame_refs: list[str] = field(default_factory=list)
    structured: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": self.role,
            "text": self.text,
            "frame_refs": list(self.frame_refs),
        }
        if self.structured is not None:
            payload["structured"] = self.structured
        return payload

    @classmethod
    def from_dict(cls, data: Any) -> ChatTurn:
        if not isinstance(data, dict):
            raise GeminiError("chat turn must be an object")
        refs = data.get("frame_refs") or []
        if not isinstance(refs, list):
            raise GeminiError("frame_refs must be a list")
        structured = data.get("structured")
        if structured is not None and not isinstance(structured, dict):
            raise GeminiError("structured must be an object")
        if "role" not in data:
            raise GeminiError("chat turn needs a role")
        return cls(
            role=str(data["role"]),
            text=str(data.get("text", "")),
            frame_refs=[str(item) for item in refs],
            structured=structured,
        )


def config_dir() -> Path:
    """Where Super Processor keeps per-user settings, such as the Gemini key.

    ``SUPER_PROCESSOR_CONFIG_DIR`` overrides it (tests use this). Otherwise
    the platform's usual place: ``$XDG_CONFIG_HOME`` or ``~/.config`` on
    Linux, ``~/Library/Application Support`` on macOS, ``%APPDATA%`` on
    Windows.
    """
    override = os.environ.get(CONFIG_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "super-processor"


def credentials_path() -> Path:
    return config_dir() / CREDENTIALS_FILE


def _stored_gemini_api_key() -> str | None:
    """The key saved by the setup screen, or ``None`` if there is none."""
    try:
        data = json.loads(credentials_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    key = data.get(_CREDENTIALS_KEY) if isinstance(data, dict) else None
    if not isinstance(key, str) or not key.strip():
        return None
    return key.strip()


def gemini_api_key_source() -> str | None:
    """Where the key in use comes from, for the setup screen; ``None`` if none."""
    if os.environ.get(GEMINI_API_KEY_ENV):
        return f"the {GEMINI_API_KEY_ENV} environment variable"
    if _stored_gemini_api_key():
        return str(credentials_path())
    if resolve_secret(GEMINI_API_KEY_ENV):
        return "the system keyring (saved by an older version)"
    return None


def resolve_gemini_api_key() -> str | None:
    """The Gemini API key: the environment, then the saved file, then the keyring.

    The environment wins so a key set for one run is used for that run. The
    keyring is only read, for keys saved by versions before the file.
    """
    key = (
        os.environ.get(GEMINI_API_KEY_ENV, "").strip()
        or _stored_gemini_api_key()
        or (resolve_secret(GEMINI_API_KEY_ENV) or "").strip()
        or None
    )
    remember_secret(key)
    return key


def store_gemini_api_key(api_key: str) -> Path:
    """Save the key where later runs find it, and use it in this process.

    It goes to :func:`credentials_path`, readable only by the user (0600,
    in a 0700 folder), written to a temporary file and moved into place so
    a crash never leaves half a key. Returns the file's path.
    """
    cleaned = api_key.strip()
    if not cleaned:
        raise GeminiError("API key is empty")
    if not key_looks_valid(cleaned):
        raise GeminiError(_BAD_KEY_MESSAGE)
    remember_secret(cleaned)
    path = credentials_path()
    payload = json.dumps({_CREDENTIALS_KEY: cleaned}, indent=2) + "\n"
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise GeminiError(
            f"could not save the API key to {path}: {redact_secrets(str(exc))}"
        ) from None
    os.environ[GEMINI_API_KEY_ENV] = cleaned
    return path


def load_chat(job_dir: Path) -> list[ChatTurn]:
    """Load persisted Gemini chat turns for a job."""
    path = job_dir / CHAT_FILE_NAME
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except RecursionError as exc:
        raise GeminiError(f"cannot read {CHAT_FILE_NAME}: nested too deep") from exc
    except (OSError, ValueError) as exc:
        # ValueError: bad JSON, or bytes that are not UTF-8.
        raise GeminiError(f"cannot read {CHAT_FILE_NAME}: {exc}") from exc
    if not isinstance(data, dict):
        raise GeminiError(f"{CHAT_FILE_NAME} root must be an object")
    turns = data.get("turns") or []
    if not isinstance(turns, list):
        raise GeminiError(f"{CHAT_FILE_NAME}: chat turns must be a list")
    try:
        return [ChatTurn.from_dict(item) for item in turns if isinstance(item, dict)]
    except GeminiError as exc:
        # Name the file, so the editor knows what is damaged.
        raise GeminiError(f"{CHAT_FILE_NAME}: {exc}") from exc


def save_chat(job_dir: Path, turns: list[ChatTurn]) -> Path:
    """Write chat history atomically."""
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / CHAT_FILE_NAME
    payload = {"version": CHAT_VERSION, "turns": [turn.to_dict() for turn in turns]}
    with _CHAT_LOCK:
        return write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def append_chat_turn(job_dir: Path, turn: ChatTurn) -> list[ChatTurn]:
    """Append one turn and persist."""
    return append_chat_turns(job_dir, [turn])


def append_chat_turns(job_dir: Path, turns: Sequence[ChatTurn]) -> list[ChatTurn]:
    """Append several turns in one locked load-and-save, and persist.

    A question and its answer go in together, so another thread's turns can
    never land between them. Returns the whole chat after the append.
    """
    with _CHAT_LOCK:
        history = load_chat(job_dir)
        history.extend(turns)
        save_chat(job_dir, history)
        return history


def contents_from_chat(turns: list[ChatTurn]) -> list[dict[str, Any]]:
    """Turn stored chat into ``generateContent`` ``contents``.

    Roles alternate ``user`` and ``model``. Consecutive same-role turns are
    joined so a replay stays valid. Stills are not replayed; they belong on
    the new user turn. Thought signatures are not stored: these calls are
    structured JSON, not function calls, and the legacy API only requires
    signatures when a later turn continues a tool call.
    """
    contents: list[dict[str, Any]] = []
    for turn in turns:
        role = "model" if turn.role == "model" else "user"
        text = turn.text.strip()
        if not text:
            continue
        if contents and contents[-1]["role"] == role:
            previous = contents[-1]["parts"][0]["text"]
            contents[-1]["parts"][0]["text"] = f"{previous}\n{text}"
            continue
        contents.append({"role": role, "parts": [{"text": text}]})
    return contents


def _extract_text(response: dict[str, Any]) -> str:
    try:
        candidates = response["candidates"]
        content = candidates[0]["content"]
        parts = content["parts"]
        texts = [str(part["text"]) for part in parts if "text" in part]
    except (KeyError, IndexError, TypeError) as exc:
        raise GeminiError("unexpected Gemini response shape") from exc
    if not texts:
        raise GeminiError("Gemini returned no text")
    return "".join(texts)


# No reply schema nests deeper than a handful of levels.
_MAX_REPLY_DEPTH = 32


def _too_deep(value: Any, limit: int) -> bool:
    """True when ``value`` nests lists or objects more than ``limit`` deep."""
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        children = (
            item.values()
            if isinstance(item, dict)
            else item
            if isinstance(item, list)
            else None
        )
        if children is None:
            continue
        if depth > limit:
            return True
        stack.extend(
            (child, depth + 1) for child in children if isinstance(child, (dict, list))
        )
    return False


def _parse_json_object(text: str) -> dict[str, Any]:
    """The object in a model reply. NaN, Infinity and numbers too big for a
    float are read as null here, once for every reply, so none reaches a
    plan. A reply nested deeper than any schema is refused."""
    try:
        data = loads_finite(text)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"Gemini JSON parse failed: {exc}") from exc
    except RecursionError as exc:
        raise GeminiError("Gemini JSON parse failed: nested too deep") from exc
    if not isinstance(data, dict):
        raise GeminiError("Gemini structured output must be an object")
    if _too_deep(data, _MAX_REPLY_DEPTH):
        raise GeminiError("Gemini JSON parse failed: nested too deep")
    return data


_UPSCALE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "start_s": {"type": "number"},
        "end_s": {"type": "number"},
        "scale": {"type": "integer"},
        "strength": {"type": "number"},
    },
    "required": ["start_s", "end_s", "scale", "strength"],
}


_SHOT_LABELS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "label": {"type": "string"},
                    "issues": {"type": "array", "items": {"type": "string"}},
                    "contains": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "label", "issues", "contains"],
            },
        }
    },
    "required": ["shots"],
}


_LOOK_PROPERTIES: dict[str, Any] = {
    name: {"type": "number"}
    for name in (
        "deblock",
        "denoise",
        "contrast",
        "brightness",
        "saturation",
        "gamma",
        "grain",
    )
}
_LOOK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": _LOOK_PROPERTIES,
    "required": list(_LOOK_PROPERTIES),
}
_PROPOSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scale": {"type": "integer"},
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "strength": {"type": "number"},
                    "use_upscale": {"type": "boolean"},
                    "look": _LOOK_SCHEMA,
                    "reason": {"type": "string"},
                },
                "required": ["index", "strength", "look", "reason"],
            },
        },
    },
    "required": ["scale", "shots"],
}
_SEGMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_s": {"type": "number"},
                    "end_s": {"type": "number"},
                    "label": {"type": "string"},
                    "contains": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start_s", "end_s", "label", "contains"],
            },
        },
    },
    "required": ["segments"],
}


def _segments_schema(scales: tuple[int, ...]) -> dict[str, Any]:
    """The segment schema, plus one output scale when ``scales`` are offered."""
    if not scales:
        return _SEGMENTS_SCHEMA
    return {
        "type": "object",
        "properties": {
            **_SEGMENTS_SCHEMA["properties"],
            # Gemini allows enums only on strings; the reply is checked
            # against the offered scales instead.
            "scale": {"type": "integer"},
            "scale_reason": {"type": "string"},
        },
        "required": ["segments", "scale", "scale_reason"],
    }


_REGIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "regions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string"},
                                "box_2d": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                            },
                            "required": ["kind", "box_2d"],
                        },
                    },
                },
                "required": ["index", "regions"],
            },
        }
    },
    "required": ["shots"],
}
_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "ok": {"type": "boolean"},
                    "problems": {"type": "array", "items": {"type": "string"}},
                    "note": {"type": "string"},
                    "strength": {"type": "number"},
                    "look": _LOOK_SCHEMA,
                },
                "required": ["index", "ok", "problems", "note", "strength", "look"],
            },
        }
    },
    "required": ["shots"],
}


# How much of a reply the chat keeps (see _chat_copy). Text fields the
# editor reads keep their own caps; any other string is a short name.
_CHAT_TEXT_CAPS = {"label": 80, "reason": 240, "note": 240, "scale_reason": 240}
_CHAT_NAME_CAP = 32
_CHAT_LIST_CAP = 40
_CHAT_LIST_CAPS = {"regions": 8, "box_2d": 4}
_CHAT_NAMES_CAP = 16


def _chat_copy(value: Any, schema: dict[str, Any], key: str = "") -> Any:
    """The part of a reply that ``schema`` asks for, cut to size.

    Unknown keys go, a value of the wrong type becomes null, text is one
    short line, and lists keep only their first items, so a hostile reply
    cannot grow the chat that is sent again with every later call.
    """
    kind = str(schema.get("type", "")).lower()
    if kind == "object":
        if not isinstance(value, dict):
            return None
        properties: dict[str, Any] = schema.get("properties", {})
        return {
            name: _chat_copy(value[name], sub, name)
            for name, sub in properties.items()
            if name in value
        }
    if kind == "array":
        if not isinstance(value, list):
            return None
        items: dict[str, Any] = schema.get("items", {})
        cap = (
            _CHAT_NAMES_CAP
            if items.get("type") == "string"
            else _CHAT_LIST_CAPS.get(key, _CHAT_LIST_CAP)
        )
        return [_chat_copy(item, items, key) for item in value[:cap]]
    if kind == "string":
        if not isinstance(value, str):
            return None
        return one_line(value, _CHAT_TEXT_CAPS.get(key, _CHAT_NAME_CAP))
    if kind == "boolean":
        return value if isinstance(value, bool) else None
    if kind in ("number", "integer"):
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        return value if number else None
    return None


class GeminiClient:
    """Call Gemini with a structured schema for upscale result notes.

    Every call that takes ``job_dir`` replays that job's chat and then saves
    its question and answer there as one pair. Passing a list as
    ``turns_out`` still replays the chat but collects the pair in that list
    instead of saving it, for work whose result may be thrown away.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        transport: GeminiTransport | None = None,
    ) -> None:
        self.api_key = (api_key or resolve_gemini_api_key() or "").strip()
        if not self.api_key:
            raise GeminiError(
                f"set {GEMINI_API_KEY_ENV} or paste a key in the wizard setup screen"
            )
        if not key_looks_valid(self.api_key):
            raise GeminiError(_BAD_KEY_MESSAGE)
        remember_secret(self.api_key)
        self.model = model or os.environ.get(GEMINI_MODEL_ENV, DEFAULT_GEMINI_MODEL)
        self.transport: GeminiTransport = transport or UrllibGeminiTransport()
        self._candidate_models: list[str] | None = None
        self._exhausted_models: set[str] = set()

    def list_model_infos(self) -> list[ModelInfo]:
        """Fetch and normalize models from the transport when supported."""
        lister = getattr(self.transport, "list_models", None)
        if not callable(lister):
            return [
                ModelInfo(
                    name=self.model,
                    methods=("generateContent",),
                    description="multimodal image video",
                )
            ]
        raw_models = lister(api_key=self.api_key)
        if not isinstance(raw_models, list):
            raise GeminiError("Gemini models.list returned no list")
        return [parse_model_info(item) for item in raw_models if isinstance(item, dict)]

    def candidate_models(self, *, refresh: bool = False) -> list[str]:
        """Multimodal models with usable RPM (limit != 0 / remaining > 0).

        Preferred / env model is tried first. Results are cached until a
        refresh or until failover marks models exhausted.
        """
        if self._candidate_models is not None and not refresh:
            return [
                name
                for name in self._candidate_models
                if name not in self._exhausted_models
            ]
        env_extra = os.environ.get("SUPER_PROCESSOR_GEMINI_CANDIDATES", "").strip()
        extras = [
            _strip_models_prefix(part) for part in env_extra.split(",") if part.strip()
        ]
        discovered = select_candidate_models(
            self.list_model_infos(),
            preferred=self.model,
        )
        ordered: list[str] = []
        seen: set[str] = set()
        for name in [self.model, *extras, *discovered]:
            clean = _strip_models_prefix(name)
            if not clean or clean in seen or clean in self._exhausted_models:
                continue
            if not _MODEL_NAME.fullmatch(clean):
                continue
            # Skip image-generation / tool-specialist names even if env-listed.
            probe = ModelInfo(
                name=clean,
                methods=("generateContent",),
                description="multimodal image video",
            )
            if not model_accepts_text_or_video(probe):
                continue
            ordered.append(clean)
            seen.add(clean)
            if len(ordered) >= MAX_CANDIDATE_MODELS:
                break
        self._candidate_models = ordered
        return list(ordered)

    def _generate_with_failover(self, body: dict[str, Any]) -> dict[str, Any]:
        """Try candidate models, retrying transient 503s before moving on.

        Each attempt is a new ``generateContent`` call. The model name is only
        in the URL. ``body`` already holds the system instruction, schema, and
        the full conversation, so a fallback model sees the same context.
        """
        errors: list[str] = []
        # A video request is long: a busy model is skipped at once rather
        # than retried, and a timeout ends the failover, since every model
        # would read the same video and the caller can fall back to stills.
        video = _has_video(body)
        retries = 0 if video else _TRANSIENT_RETRIES
        for model in self.candidate_models():
            last_exc: GeminiError | None = None
            for attempt in range(retries + 1):
                try:
                    response = self.transport.generate(
                        model=model,
                        api_key=self.api_key,
                        body=body,
                    )
                except GeminiError as exc:
                    last_exc = exc
                    if _is_exhausting_error(exc):
                        self._exhausted_models.add(model)
                        errors.append(f"{model}: {exc}")
                        break
                    if video and "timed out" in str(exc).lower():
                        errors.append(f"{model}: {exc}")
                        raise GeminiError(
                            f"Gemini timed out watching the video: {_grouped(errors)}"
                        ) from exc
                    if _is_transient_error(exc) and attempt < retries:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    if _is_retryable_error(exc):
                        errors.append(f"{model}: {exc}")
                        break
                    raise
                self.model = model
                return response
            if last_exc is not None and not _is_retryable_error(last_exc):
                raise last_exc
        raise GeminiError(f"All candidate Gemini models failed: {_grouped(errors)}")

    def pick_working_model(self) -> str:
        """Probe candidates until one accepts a tiny structured generate call."""
        probe_schema = {
            "type": "OBJECT",
            "properties": {"ok": {"type": "BOOLEAN"}},
            "required": ["ok"],
        }
        body = {
            "systemInstruction": {
                "parts": [{"text": "Reply with structured JSON only."}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": 'Return {"ok": true}.'}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": probe_schema,
            },
        }
        self.candidate_models(refresh=True)
        try:
            response = self._generate_with_failover(body)
            _parse_json_object(_extract_text(response))
        except GeminiError as exc:
            raise GeminiError(f"No Gemini model with usable quota: {exc}") from exc
        remaining = [
            name
            for name in (self._candidate_models or [])
            if name != self.model and name not in self._exhausted_models
        ]
        self._candidate_models = [self.model, *remaining][:MAX_CANDIDATE_MODELS]
        return self.model

    def _generate(
        self,
        *,
        contents: list[dict[str, Any]],
        schema: dict[str, Any],
        system: str,
        prior: list[ChatTurn] | None = None,
    ) -> dict[str, Any]:
        # systemInstruction is request-scoped on generateContent. Resend it
        # with the replayed history; do not put it in contents.
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [*contents_from_chat(prior or []), *contents],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        response = self._generate_with_failover(body)
        return _parse_json_object(_extract_text(response))

    def label_shots(
        self,
        *,
        shots: list[dict[str, Any]],
        stills: list[Path],
        issues: tuple[str, ...],
        contains: tuple[str, ...],
        job_dir: Path | None = None,
        turns_out: list[ChatTurn] | None = None,
    ) -> list[dict[str, Any]]:
        """Label each shot from its still and measurements.

        ``shots`` are plain dicts (``start_s``, ``end_s``, ``metrics``,
        ``hints``); ``stills`` holds one JPEG per shot, in the same order.
        Returns ``[{index, label, issues, contains}]``; the caller drops any
        name outside ``issues`` and ``contains``.
        """
        if len(stills) != len(shots):
            raise GeminiError("label_shots needs one still per shot")
        intro = (
            "These are the shots of one video an editor wants restored and "
            "upscaled. Each shot has one still from its middle and FFmpeg "
            "measurements: blockiness (compression blocks, higher is worse), "
            "blur (higher is softer), noise (higher is noisier), brightness "
            "and contrast (0 to 1). 'hints' are what the numbers alone "
            "suggest; correct them from what you see. For every shot return "
            "its index, a label of at most eight words saying what is in it, "
            f"the problems it has from this list only: {', '.join(issues)}; "
            "and what it contains from this list only, where restoration "
            f"mistakes would be obvious: {', '.join(contains)}."
        )
        parts: list[dict[str, Any]] = [{"text": intro}]
        for index, (shot, still) in enumerate(zip(shots, stills, strict=True)):
            parts.append({"text": f"Shot {index}: {json.dumps(shot)}"})
            parts.append(_jpeg_part(still, under=job_dir))
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=_SHOT_LABELS_SCHEMA,
            system=(
                "You label shots for a video restoration tool. You never "
                "suggest commands or settings. Return JSON matching the schema."
            ),
            prior=load_chat(job_dir) if job_dir is not None else None,
        )
        labels = reply.get("shots")
        if not isinstance(labels, list):
            raise GeminiError("Gemini returned no shot labels")
        self._log(
            job_dir,
            intro + f" ({len(shots)} shots)",
            reply,
            turns_out,
            schema=_SHOT_LABELS_SCHEMA,
        )
        return [item for item in labels if isinstance(item, dict)]

    def upload_video(
        self,
        path: Path,
        *,
        on_progress: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> GeminiVideo:
        """Upload ``path`` through the File API and wait until Gemini can use it.

        ``on_progress`` gets the share of bytes sent, 0 to 1.
        """
        uploader = getattr(self.transport, "upload_file", None)
        getter = getattr(self.transport, "get_file", None)
        if not callable(uploader) or not callable(getter):
            raise GeminiError("this Gemini transport cannot upload video")
        file = uploader(
            api_key=self.api_key,
            path=path,
            mime_type="video/mp4",
            display_name=path.name,
            on_progress=on_progress,
        )
        deadline = clock() + _FILE_ACTIVE_TIMEOUT_S
        while True:
            if not isinstance(file, dict):
                raise GeminiError("Gemini returned an upload that is not an object")
            state = str(file.get("state") or "")
            if state == "ACTIVE":
                break
            if state == "FAILED":
                raise GeminiError("Gemini could not process the uploaded video")
            if clock() > deadline:
                raise GeminiError("Gemini took too long to process the uploaded video")
            name = file.get("name")
            if not isinstance(name, str) or not _FILE_NAME.fullmatch(name):
                raise GeminiError("Gemini returned an upload with a bad name")
            sleep(_FILE_POLL_S)
            file = getter(api_key=self.api_key, name=name)
        uri = file.get("uri")
        video = GeminiVideo(
            name=str(file.get("name") or ""),
            # Sent back to Gemini with every request that watches the video.
            uri=uri
            if isinstance(uri, str) and _plain_uri(uri) and uri.startswith("https://")
            else "",
            mime_type=_video_mime(file.get("mimeType")),
            expires_at=_parse_expiry(file.get("expirationTime")),
        )
        if not video.name or not video.uri:
            raise GeminiError("Gemini returned an upload without a name or address")
        if not _FILE_NAME.fullmatch(video.name):
            # The name goes into File API URLs later; refuse anything else.
            raise GeminiError("Gemini returned an upload with a bad name")
        return video

    def video_alive(self, video: GeminiVideo) -> bool:
        """True when the File API still has ``video`` and it is ACTIVE."""
        getter = getattr(self.transport, "get_file", None)
        if not callable(getter):
            return False
        try:
            file = getter(api_key=self.api_key, name=video.name)
        except GeminiError:
            return False
        return isinstance(file, dict) and file.get("state") == "ACTIVE"

    def delete_video(self, video: GeminiVideo) -> bool:
        """Remove ``video`` from the File API, best effort.

        True when it is gone (or the transport cannot delete, in which case
        Google removes it after about 48 hours). A failure is logged without
        the key and never raised: cleanup must not fail the job.
        """
        deleter = getattr(self.transport, "delete_file", None)
        if not callable(deleter):
            return False
        try:
            deleter(api_key=self.api_key, name=video.name)
        except Exception as exc:  # noqa: BLE001 - best effort, logged
            _LOG.warning(
                "could not delete the Gemini upload %s: %s",
                video.name,
                redact_secrets(str(exc)) or type(exc).__name__,
            )
            return False
        return True

    def propose_looks(
        self,
        *,
        shots: list[dict[str, Any]],
        stills: list[Path],
        bounds: dict[str, tuple[float, float, float]],
        scales: tuple[int, ...],
        protected_strength: float,
        job_dir: Path | None = None,
        turns_out: list[ChatTurn] | None = None,
        video: GeminiVideo | None = None,
    ) -> dict[str, Any]:
        """Propose a strength and look per shot, and one scale for the video.

        ``shots`` carry their label, problems, contents, measurements, and,
        on a redo, the current settings and the editor's note. With
        ``video``, Gemini also watches each shot's stretch of the upload.
        """
        if len(stills) != len(shots):
            raise GeminiError("propose_looks needs one still per shot")
        limits = "; ".join(
            f"{name} {low:g} to {high:g} (no change at {neutral:g})"
            for name, (low, high, neutral) in bounds.items()
        )
        intro = (
            "You plan a restoration of an editor's footage. Each shot goes "
            "through fixed FFmpeg clean-up (deblock, denoise), then a SPAN 2x "
            "upscale, whose fine detail is added to a plain upscale by "
            "'strength' (0 adds none, 1 adds all of it), then fixed FFmpeg finishing "
            "(contrast, brightness, saturation, gamma, grain). SPAN treats "
            "blocks and noise as detail and sharpens them into texture, so "
            "clean those first and do not ask for more strength than the shot "
            "needs. The goal is footage that looks well shot, not AI made: no "
            "waxy skin, no crunchy edges, no invented patterns, faces and text "
            "unchanged. Set only what the shot needs; leave the rest at its "
            f"no-change value. Ranges: {limits}. Shapes and colour always "
            "come from the source; strength only sets how much of SPAN's "
            "fine detail is added. A shot listing protected boxes is held "
            "down inside them automatically, so its strength can suit the "
            "rest of the frame; a shot that contains faces, hands, or text "
            f"without boxes is capped at strength {protected_strength:g}. "
            "Set use_upscale to false for a shot that SPAN would only "
            "harm (already clean and sharp, or fine texture it would invent "
            "over); it is then resized with a plain upscale and keeps its "
            "look. "
            f"Pick one scale for the whole video from {list(scales)}. If a "
            "shot has an editor_note, follow it; current shows its settings "
            "now. Return a short reason per shot in plain words."
        )
        if video is not None:
            intro += (
                " Each shot comes with its stretch of the video, sampled at "
                f"{VIDEO_FPS:g} frame per second, and one full-size still."
            )
        parts: list[dict[str, Any]] = [{"text": intro}]
        for shot, still in zip(shots, stills, strict=True):
            parts.append({"text": f"Shot: {json.dumps(shot)}"})
            if video is not None:
                parts.append(
                    video_part(
                        video,
                        start_s=float(shot.get("start_s", 0.0)),
                        end_s=float(shot.get("end_s", 0.0)) or None,
                    )
                )
            parts.append(_jpeg_part(still, under=job_dir))
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=_PROPOSE_SCHEMA,
            system=(
                "You choose bounded numbers for a fixed video pipeline. You "
                "never write commands. Return JSON matching the schema."
            ),
            prior=load_chat(job_dir) if job_dir is not None else None,
        )
        if not isinstance(reply.get("shots"), list):
            raise GeminiError("Gemini returned no shot recipes")
        self._log(
            job_dir,
            intro + f" ({len(shots)} shots)",
            reply,
            turns_out,
            schema=_PROPOSE_SCHEMA,
        )
        return reply

    def propose_segments(
        self,
        *,
        stills: list[Path] | None = None,
        times: list[float] | None = None,
        duration_s: float,
        contains: tuple[str, ...] = (),
        note: str = "",
        max_segments: int = 10,
        min_segment_s: float = 3.0,
        job_dir: Path | None = None,
        turns_out: list[ChatTurn] | None = None,
        video: GeminiVideo | None = None,
        scales: tuple[int, ...] = (),
        source_size: tuple[int, int] = (0, 0),
    ) -> dict[str, Any]:
        """Cut the video into segments, from the uploaded video or a still grid.

        A segment is a stretch where the picture asks for the same
        restoration. ``note`` is the editor's reason for rejecting the last
        proposal; the replayed chat shows Gemini what it proposed then. With
        ``video``, Gemini watches the upload at 1 fps; otherwise it sees
        ``stills`` with their ``times``. When ``scales`` are offered, Gemini
        also picks the output scale for the whole video and says why.
        Returns ``{segments: [{start_s, end_s, label, contains}], scale,
        scale_reason}`` (``scale`` None when none was offered); the caller
        repairs gaps and overlaps and drops any name outside ``contains``.
        """
        stills = stills or []
        times = times or []
        if video is None and (not stills or len(stills) != len(times)):
            raise GeminiError("propose_segments needs one time per still")
        seen = (
            f"This video, sampled at {VIDEO_FPS:g} frame per second, is one an "
            "editor wants restored. Times are in seconds from its start."
            if video is not None
            else "These stills are sampled from one video an editor wants "
            "restored, each with its time in seconds."
        )
        intro = (
            f"{seen} The video "
            f"is {duration_s:.2f}s long. Cut it into as few segments as the "
            "restoration needs, at most "
            f"{max_segments}; fewer is better. Each segment is reviewed on its "
            f"own, so it must be at least {min_segment_s:g} seconds long. Cut "
            "only where the photographic context changes enough to need "
            "different settings: location, light, camera, subject distance, or "
            "image quality. Keep a short shot with the similar footage around "
            "it rather than giving it its own segment. "
            "The segments run in order from 0 to the end with no gaps or "
            "overlaps. Give each a label of at most eight words saying what "
            "is in it and how it is lit, and what it contains from this list "
            "only, where restoration mistakes would be obvious: "
            f"{', '.join(contains)}."
        )
        if scales:
            intro += " " + _scale_question(scales, source_size)
        text = " ".join(note.split())
        if text:
            intro += (
                " The editor did not accept your last segments and wrote: "
                f"{text[:500]!r}. Follow that note."
            )
        # Gemini reads the request's text better after its video.
        parts: list[dict[str, Any]] = []
        if video is not None:
            parts.append(video_part(video))
            parts.append({"text": intro})
        else:
            parts.append({"text": intro})
            for t_s, still in zip(times, stills, strict=True):
                parts.append({"text": f"t={t_s:.3f}s"})
                parts.append(_jpeg_part(still, under=job_dir))
        schema = _segments_schema(scales)
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=schema,
            system=(
                "You split videos into segments for a restoration tool and "
                "choose its output size from the options given. You never "
                "suggest commands. Return JSON matching the schema."
            ),
            prior=load_chat(job_dir) if job_dir is not None else None,
        )
        segments = reply.get("segments")
        if not isinstance(segments, list) or not segments:
            raise GeminiError("Gemini returned no segments")
        seen_as = "the video" if video is not None else f"{len(stills)} stills"
        self._log(job_dir, intro + f" ({seen_as})", reply, turns_out, schema=schema)
        scale = _offered_scale(reply.get("scale"), scales)
        reason = reply.get("scale_reason")
        return {
            "segments": [item for item in segments if isinstance(item, dict)],
            "scale": scale,
            "scale_reason": one_line(reason, 240) if scale is not None else "",
        }

    def locate_regions(
        self,
        *,
        shots: list[dict[str, Any]],
        stills: list[list[Path]],
        kinds: tuple[str, ...],
        max_regions: int,
        job_dir: Path | None = None,
        turns_out: list[ChatTurn] | None = None,
    ) -> list[dict[str, Any]]:
        """Box the faces, hands, and text in each shot.

        Each shot comes with stills from its start, middle, and end; one box
        should cover where the thing is across all of them.
        """
        if len(stills) != len(shots):
            raise GeminiError("locate_regions needs stills for every shot")
        intro = (
            "For each shot below you get stills from its start, middle, and "
            "end. Draw boxes around every "
            f"{', '.join(kinds)} that is clearly visible, as kind plus box_2d "
            "[ymin, xmin, ymax, xmax] scaled 0 to 1000. Each box must cover "
            "where that thing is in all three stills, so a moving face gets "
            "one box over its whole path. Use only these kinds: "
            f"{', '.join(kinds)}. At most {max_regions} boxes per shot; "
            "merge small neighbours. Return an empty list when there are none."
        )
        parts: list[dict[str, Any]] = [{"text": intro}]
        for shot, frames in zip(shots, stills, strict=True):
            parts.append({"text": f"Shot: {json.dumps(shot)}"})
            parts.extend(_jpeg_part(frame, under=job_dir) for frame in frames)
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=_REGIONS_SCHEMA,
            system=(
                "You locate faces, hands, and text in video stills. Return "
                "JSON matching the schema."
            ),
            prior=load_chat(job_dir) if job_dir is not None else None,
        )
        found = reply.get("shots")
        if not isinstance(found, list):
            raise GeminiError("Gemini returned no regions")
        self._log(
            job_dir,
            intro + f" ({len(shots)} shots)",
            reply,
            turns_out,
            schema=_REGIONS_SCHEMA,
        )
        return [item for item in found if isinstance(item, dict)]

    def check_previews(
        self,
        *,
        shots: list[dict[str, Any]],
        pairs: list[tuple[Path, Path]],
        problems: tuple[str, ...],
        job_dir: Path | None = None,
        turns_out: list[ChatTurn] | None = None,
    ) -> list[dict[str, Any]]:
        """Compare each before/after still for restoration mistakes.

        For a shot that fails, the reply carries corrected settings.
        """
        if len(pairs) != len(shots):
            raise GeminiError("check_previews needs one still pair per shot")
        intro = (
            "Each shot below has a BEFORE still (the source) and an AFTER "
            "still (restored and upscaled with the settings shown). Compare "
            "them like a picky editor. Fail a shot only for a real mistake "
            f"from this list: {', '.join(problems)}. Look hardest at faces "
            "(eyes, irises, teeth, symmetry), hands (finger count), and text. "
            "Pass a shot that simply looks better. For a failed shot, return "
            "corrected strength and look (usually lower strength, or more "
            "deblock/denoise for fake texture); for a passed shot return its "
            "settings unchanged. Each shot's 'shimmer' is measured on the "
            "moving previews: how much less stable its added texture is than "
            "the source's (1 is the same, above 1.6 visibly crawls; null when "
            "there is too little texture to tell). Fail a high one as flicker. "
            "Keep the note to one sentence."
        )
        parts: list[dict[str, Any]] = [{"text": intro}]
        for shot, (before, after) in zip(shots, pairs, strict=True):
            parts.append({"text": f"Shot: {json.dumps(shot)}. BEFORE:"})
            parts.append(_jpeg_part(before, under=job_dir))
            parts.append({"text": "AFTER:"})
            parts.append(_jpeg_part(after, under=job_dir))
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=_CHECK_SCHEMA,
            system=(
                "You check restoration previews for visible mistakes. Return "
                "JSON matching the schema."
            ),
            prior=load_chat(job_dir) if job_dir is not None else None,
        )
        verdicts = reply.get("shots")
        if not isinstance(verdicts, list):
            raise GeminiError("Gemini returned no preview checks")
        self._log(
            job_dir,
            intro + f" ({len(shots)} shots)",
            reply,
            turns_out,
            schema=_CHECK_SCHEMA,
        )
        return [item for item in verdicts if isinstance(item, dict)]

    def _log(
        self,
        job_dir: Path | None,
        prompt: str,
        reply: dict[str, Any],
        turns_out: list[ChatTurn] | None = None,
        *,
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Keep the question and its answer as one user and model pair.

        With ``turns_out`` the pair is added to that list and the chat file
        is left alone, so the caller can append it later or drop it.
        Otherwise both turns are written to ``job_dir``'s chat in one go.
        With ``schema`` only the fields it asks for are kept, cut to size
        (see :func:`_chat_copy`), since the chat is replayed with every
        later call for the job.
        """
        if schema is not None:
            reply = _chat_copy(reply, schema)
        turns = [
            ChatTurn(role="user", text=prompt),
            ChatTurn(role="model", text=json.dumps(reply), structured=reply),
        ]
        if turns_out is not None:
            turns_out.extend(turns)
        elif job_dir is not None:
            append_chat_turns(job_dir, turns)

    def revise_upscale_params(
        self,
        *,
        note: str,
        prior_scale: int,
        prior_strength: float,
        prior_start_s: float,
        prior_end_s: float,
        duration_s: float,
        allowed_scales: tuple[int, ...] = (1, 2),
        job_dir: Path | None = None,
        turns_out: list[ChatTurn] | None = None,
    ) -> dict[str, Any]:
        """Turn a result note into a bounded scale, strength, and time range."""
        scales = ", ".join(str(scale) for scale in allowed_scales)
        prompt = (
            "We already restored and upscaled a video locally. "
            f"Duration is {duration_s:.1f} seconds. Previous params: "
            f"scale={prior_scale}, strength={prior_strength:.2f}, "
            f"applied from {prior_start_s:.1f}s to {prior_end_s:.1f}s. "
            f"The user said: {note.strip()}. "
            f"Return start_s, end_s, scale (only one of: {scales}; 1 means "
            "the video keeps its size), and strength from 0 to 1. Keep "
            f"scale={prior_scale} unless they explicitly ask for a different "
            "output size. Lower strength means less invented texture. If they "
            "name a time range, set start_s and end_s to that range. "
            "Do not invent shell commands or a new model."
        )
        data = self._generate(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            schema=_UPSCALE_SCHEMA,
            system=(
                "You only retune restoration upscale parameters. "
                "Return JSON matching the schema."
            ),
            prior=load_chat(job_dir) if job_dir is not None else None,
        )
        self._log(job_dir, prompt, data, turns_out, schema=_UPSCALE_SCHEMA)
        return data


def _scale_question(scales: tuple[int, ...], source_size: tuple[int, int]) -> str:
    """Ask for one output scale for the whole video, 2 unless there is a reason."""
    width, height = source_size
    size = f"The source is {width}x{height}. " if width and height else ""
    options = {
        1: "1 keeps the source size: clean-up and finishing only, no SPAN",
        2: "2 doubles it with SPAN, the default",
    }
    listed = "; ".join(options[scale] for scale in scales if scale in options)
    return (
        f"{size}Also choose one output scale for the whole video from "
        f"{list(scales)} ({listed}). Use 2 unless the video gives a reason: "
        "1 when it is already large and sharp enough that upscaling would add "
        "nothing but invented detail. Say "
        "why in scale_reason, one short sentence an editor understands."
    )


def _offered_scale(raw: Any, scales: tuple[int, ...]) -> int | None:
    """``raw`` as one of the offered ``scales``, or None."""
    if isinstance(raw, bool):
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value in scales and float(raw) == value else None


def _jpeg_part(path: Path, *, under: Path | None = None) -> dict[str, Any]:
    """Encode a still. When ``under`` is set, the file must sit inside that tree."""
    resolved = path.expanduser().resolve()
    if under is not None:
        root = under.expanduser().resolve()
        if resolved != root and root not in resolved.parents:
            raise GeminiError(f"still is outside the job: {path.name}")
    try:
        data = base64.b64encode(resolved.read_bytes()).decode("ascii")
    except OSError as exc:
        raise GeminiError(f"could not read {path.name}: {exc}") from exc
    return {"inlineData": {"mimeType": "image/jpeg", "data": data}}
