"""Gemini structured-output client for the localhost wizard.

Uses the Generative Language REST API with ``responseMimeType`` /
``responseSchema`` so replies stay parseable. The model emits structured ops
only; FFmpeg argv is built elsewhere. Transport is injectable for tests.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .models import resolve_secret
from .recipe import OpName

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "SUPER_PROCESSOR_GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
CHAT_FILE_NAME = "gemini_chat.json"
CHAT_VERSION = 1
MAX_SEGMENTS = 20
MAX_LAYOUTS = 5
MIN_SEGMENT_S = 5.0
FRAME_CAP = 160
WIZARD_FRAMES_DIR = "wizard_frames"
# Production revise loops until the user picks an option; tests use this cap.
MAX_ENHANCE_REVISE_ROUNDS_FOR_TESTS = 5
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

_ALLOWED_OPS = {
    OpName.WHITE_BALANCE.value,
    OpName.CONTRAST.value,
    OpName.DENOISE.value,
    OpName.SHARPEN.value,
    OpName.STABILIZE.value,
    OpName.TRIM.value,
}


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
        except (TypeError, ValueError):
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
            except (TypeError, ValueError):
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
            f"{model}:generateContent?key={api_key}"
        )
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                parsed: object = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            status, detail = _http_error_detail(exc)
            raise GeminiError(
                f"Gemini HTTP {status}: {detail}",
                status=status,
            ) from None
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise GeminiError(f"Gemini request failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise GeminiError("Gemini response root must be an object")
        return parsed

    def list_models(
        self, *, api_key: str
    ) -> list[dict[str, Any]]:  # pragma: no cover - mocked in unit tests
        models: list[dict[str, Any]] = []
        page_token = ""
        while True:
            url = f"{_MODELS_URL}?key={api_key}&pageSize=100"
            if page_token:
                url += f"&pageToken={urllib.parse.quote(page_token)}"
            req = urllib.request.Request(url, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    parsed: object = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                status, detail = _http_error_detail(exc)
                raise GeminiError(
                    f"Gemini HTTP {status}: {detail}",
                    status=status,
                ) from None
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                raise GeminiError(f"Gemini list models failed: {exc}") from exc
            if not isinstance(parsed, dict):
                raise GeminiError("Gemini models.list root must be an object")
            batch = parsed.get("models") or []
            if isinstance(batch, list):
                models.extend(item for item in batch if isinstance(item, dict))
            page_token = str(parsed.get("nextPageToken") or "")
            if not page_token:
                break
        return models


@dataclass(slots=True)
class StructuredOp:
    """One allowlisted operation and parameters."""

    op: str
    params: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StructuredOp:
        raw_params = data.get("params") or {}
        if not isinstance(raw_params, dict):
            raise GeminiError("op params must be an object")
        params: dict[str, float] = {}
        for key, value in raw_params.items():
            params[str(key)] = float(value)
        return cls(op=str(data["op"]), params=params)


@dataclass(slots=True)
class SegmentProposal:
    """One proposed timeline segment."""

    start_s: float
    end_s: float
    label: str
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentProposal:
        issues = data.get("issues") or []
        if not isinstance(issues, list):
            raise GeminiError("segment issues must be a list")
        return cls(
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            label=str(data.get("label", "")),
            issues=[str(item) for item in issues],
        )


@dataclass(slots=True)
class SplitLayout:
    """One candidate split of the timeline."""

    segment_count: int
    summary: str
    segments: list[SegmentProposal]

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_count": self.segment_count,
            "summary": self.summary,
            "segments": [item.to_dict() for item in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitLayout:
        raw = data.get("segments") or []
        if not isinstance(raw, list):
            raise GeminiError("layout segments must be a list")
        segments = [
            SegmentProposal.from_dict(item) for item in raw if isinstance(item, dict)
        ]
        return cls(
            segment_count=int(data.get("segment_count", len(segments))),
            summary=str(data.get("summary", "")),
            segments=segments,
        )


@dataclass(slots=True)
class SplitProposal:
    """Gemini response for segmentation."""

    layouts: list[SplitLayout]
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layouts": [item.to_dict() for item in self.layouts],
            "highlights": list(self.highlights),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SplitProposal:
        raw_layouts = data.get("layouts") or []
        raw_highlights = data.get("highlights") or []
        if not isinstance(raw_layouts, list):
            raise GeminiError("layouts must be a list")
        if not isinstance(raw_highlights, list):
            raise GeminiError("highlights must be a list")
        layouts = [
            SplitLayout.from_dict(item)
            for item in raw_layouts
            if isinstance(item, dict)
        ]
        return cls(
            layouts=layouts,
            highlights=[str(item) for item in raw_highlights],
        )


@dataclass(slots=True)
class EnhanceOption:
    """One enhancement choice for a segment."""

    id: str
    label: str
    ops: list[StructuredOp]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "ops": [op.to_dict() for op in self.ops],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnhanceOption:
        raw_ops = data.get("ops") or []
        if not isinstance(raw_ops, list):
            raise GeminiError("option ops must be a list")
        ops = [
            StructuredOp.from_dict(item) for item in raw_ops if isinstance(item, dict)
        ]
        return cls(id=str(data["id"]), label=str(data.get("label", "")), ops=ops)


@dataclass(slots=True)
class SegmentEnhanceResult:
    """Gemini response for per-segment enhancement options."""

    issues: list[str]
    options: list[EnhanceOption]

    def to_dict(self) -> dict[str, Any]:
        return {
            "issues": list(self.issues),
            "options": [item.to_dict() for item in self.options],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentEnhanceResult:
        raw_issues = data.get("issues") or []
        raw_options = data.get("options") or []
        if not isinstance(raw_issues, list) or not isinstance(raw_options, list):
            raise GeminiError("enhance issues/options must be lists")
        options = [
            EnhanceOption.from_dict(item)
            for item in raw_options
            if isinstance(item, dict)
        ]
        return cls(
            issues=[str(item) for item in raw_issues],
            options=options,
        )


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
    def from_dict(cls, data: dict[str, Any]) -> ChatTurn:
        refs = data.get("frame_refs") or []
        if not isinstance(refs, list):
            raise GeminiError("frame_refs must be a list")
        structured = data.get("structured")
        if structured is not None and not isinstance(structured, dict):
            raise GeminiError("structured must be an object")
        return cls(
            role=str(data["role"]),
            text=str(data.get("text", "")),
            frame_refs=[str(item) for item in refs],
            structured=structured,
        )


def resolve_gemini_api_key() -> str | None:
    """Resolve the Gemini API key from env or keyring."""
    return resolve_secret(GEMINI_API_KEY_ENV)


def store_gemini_api_key(api_key: str) -> None:
    """Persist the key in the process env and optional OS keyring.

    Env is always set so headless CI (no Secret Service / keyring backend)
    still works. Keyring persistence is best-effort.
    """
    cleaned = api_key.strip()
    if not cleaned:
        raise GeminiError("API key is empty")
    os.environ[GEMINI_API_KEY_ENV] = cleaned
    try:
        import keyring
        from keyring.errors import KeyringError
    except ImportError:
        return
    try:
        keyring.set_password("super-processor", GEMINI_API_KEY_ENV, cleaned)
    except KeyringError:
        # e.g. keyring.backends.fail.Keyring on Ubuntu runners.
        return


def sample_fps_for_duration(duration_s: float) -> float:
    """Return the target sample rate for Gemini frames."""
    if duration_s <= 0:
        return 1.0
    if duration_s <= 60.0:
        return 10.0
    if duration_s >= 600.0:
        return 1.0
    # Linear interpolate 10 → 1 between 60s and 600s.
    t = (duration_s - 60.0) / 540.0
    return 10.0 + (1.0 - 10.0) * t


def frame_timestamps(duration_s: float, *, frame_cap: int = FRAME_CAP) -> list[float]:
    """Build sample timestamps with dynamic FPS and a hard frame cap."""
    if duration_s <= 0:
        return [0.0]
    fps = sample_fps_for_duration(duration_s)
    step = 1.0 / fps
    stamps: list[float] = []
    t = 0.0
    while t < duration_s - 1e-6:
        stamps.append(round(t, 3))
        t += step
    if not stamps or stamps[-1] < duration_s - 0.5:
        stamps.append(round(max(0.0, duration_s - 0.05), 3))
    if len(stamps) <= frame_cap:
        return stamps
    stride = max(1, len(stamps) // frame_cap)
    reduced = stamps[::stride][:frame_cap]
    if reduced[-1] != stamps[-1]:
        reduced[-1] = stamps[-1]
    return reduced


def load_chat(job_dir: Path) -> list[ChatTurn]:
    """Load persisted Gemini chat turns for a job."""
    path = job_dir / CHAT_FILE_NAME
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeminiError(f"cannot read {CHAT_FILE_NAME}: {exc}") from exc
    turns = data.get("turns") or []
    if not isinstance(turns, list):
        raise GeminiError("chat turns must be a list")
    return [ChatTurn.from_dict(item) for item in turns if isinstance(item, dict)]


def save_chat(job_dir: Path, turns: list[ChatTurn]) -> Path:
    """Write chat history atomically."""
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / CHAT_FILE_NAME
    payload = {"version": CHAT_VERSION, "turns": [turn.to_dict() for turn in turns]}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def append_chat_turn(job_dir: Path, turn: ChatTurn) -> list[ChatTurn]:
    """Append one turn and persist."""
    turns = load_chat(job_dir)
    turns.append(turn)
    save_chat(job_dir, turns)
    return turns


_SPLIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "highlights": {
            "type": "array",
            "items": {"type": "string"},
        },
        "layouts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_count": {"type": "integer"},
                    "summary": {"type": "string"},
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_s": {"type": "number"},
                                "end_s": {"type": "number"},
                                "label": {"type": "string"},
                                "issues": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["start_s", "end_s", "label", "issues"],
                        },
                    },
                },
                "required": ["segment_count", "summary", "segments"],
            },
        },
    },
    "required": ["layouts", "highlights"],
}

_ENHANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issues": {"type": "array", "items": {"type": "string"}},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "label": {"type": "string"},
                    "ops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {"type": "string"},
                                "params": {"type": "object"},
                            },
                            "required": ["op", "params"],
                        },
                    },
                },
                "required": ["id", "label", "ops"],
            },
        },
    },
    "required": ["issues", "options"],
}

_UPSCALE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "restore_strength": {"type": "number"},
        "scale": {"type": "integer"},
        "vsr_quality": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH"],
        },
    },
    "required": ["restore_strength", "scale", "vsr_quality"],
}


def validate_split_proposal(
    proposal: SplitProposal,
    *,
    duration_s: float,
) -> SplitProposal:
    """Enforce layout and segment bounds for the wizard."""
    if not proposal.layouts:
        raise GeminiError("Gemini returned no split layouts")
    if len(proposal.layouts) > MAX_LAYOUTS:
        proposal = SplitProposal(
            layouts=proposal.layouts[:MAX_LAYOUTS],
            highlights=proposal.highlights,
        )
    cleaned: list[SplitLayout] = []
    for layout in proposal.layouts:
        segments = layout.segments
        if not segments:
            raise GeminiError("a layout has no segments")
        if len(segments) > MAX_SEGMENTS:
            raise GeminiError(f"layout exceeds {MAX_SEGMENTS} segments")
        prev_end = 0.0
        for index, segment in enumerate(segments):
            if segment.end_s <= segment.start_s:
                raise GeminiError("segment end must be after start")
            if segment.end_s - segment.start_s < MIN_SEGMENT_S:
                raise GeminiError(
                    f"segment shorter than {MIN_SEGMENT_S:.0f}s is not allowed"
                )
            if segment.start_s < -1e-3 or segment.end_s > duration_s + 0.5:
                raise GeminiError("segment is outside the media duration")
            if index > 0 and segment.start_s + 1e-3 < prev_end:
                raise GeminiError("segments overlap")
            prev_end = segment.end_s
        cleaned.append(
            SplitLayout(
                segment_count=len(segments),
                summary=layout.summary,
                segments=segments,
            )
        )
    return SplitProposal(layouts=cleaned, highlights=proposal.highlights)


def combine_enhance_revise_message(
    *,
    prior: SegmentEnhanceResult,
    user_note: str,
    segment_label: str,
    start_s: float,
    end_s: float,
) -> str:
    """Build the structured-revise prompt from prior options + user note.

    Used by the Something-else loop: the user is improving or changing the
    current option set, so Gemini must see both the request and the last JSON.
    """
    note = user_note.strip()
    if not note:
        raise GeminiError("revise note is empty")
    prior_json = json.dumps(prior.to_dict(), indent=2)
    return (
        f"Act as a pro photographer. Segment {segment_label!r} runs from "
        f"{start_s:.1f}s to {end_s:.1f}s.\n"
        "The user wants to improve, enhance, or change the current options "
        "(Something else).\n"
        f"User request:\n{note}\n\n"
        "Previous structured options (JSON):\n"
        f"{prior_json}\n\n"
        "Return a new set of 3 to 5 enhancement options that incorporate the "
        "request. Keep ops allowlisted: white_balance, contrast, denoise, "
        "sharpen, stabilize, trim. Do not invent shell commands."
    )


def validate_enhance_result(result: SegmentEnhanceResult) -> SegmentEnhanceResult:
    """Enforce option count and allowlisted ops."""
    if not (3 <= len(result.options) <= 5):
        raise GeminiError("Gemini must return 3 to 5 enhancement options")
    for option in result.options:
        if not option.ops:
            raise GeminiError(f"option {option.id} has no ops")
        for op in option.ops:
            if op.op not in _ALLOWED_OPS:
                raise GeminiError(
                    f"operation {op.op!r} is not allowlisted for the wizard"
                )
    return result


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


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiError(f"Gemini JSON parse failed: {exc}") from exc
    if not isinstance(data, dict):
        raise GeminiError("Gemini structured output must be an object")
    return data


def _file_part(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    if path.suffix.lower() == ".ppm":
        mime = "image/png"
    return {
        "inline_data": {
            "mime_type": mime,
            "data": base64.b64encode(raw).decode("ascii"),
        }
    }


class GeminiClient:
    """Call Gemini with structured schemas for split and enhance."""

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
        """Try candidate models, retrying transient 503s before moving on."""
        errors: list[str] = []
        for model in self.candidate_models():
            last_exc: GeminiError | None = None
            for attempt in range(_TRANSIENT_RETRIES + 1):
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
                    if _is_transient_error(exc) and attempt < _TRANSIENT_RETRIES:
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
        detail = "; ".join(errors) if errors else "no candidates"
        raise GeminiError(f"All candidate Gemini models failed: {detail}")

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
    ) -> dict[str, Any]:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
                "responseSchema": schema,
            },
        }
        response = self._generate_with_failover(body)
        return _parse_json_object(_extract_text(response))

    def propose_splits(
        self,
        *,
        frame_paths: list[Path],
        duration_s: float,
        user_text: str | None = None,
        job_dir: Path | None = None,
    ) -> SplitProposal:
        """Ask Gemini for candidate timeline layouts."""
        prompt = user_text or (
            "Act as a pro video editor. Propose how to split this video into "
            "independent enhancement segments. Return 3 to 5 alternate layouts "
            f"with different segment counts when useful. Max {MAX_SEGMENTS} "
            f"segments. Each segment must be at least {MIN_SEGMENT_S:.0f} seconds. "
            f"Media duration is {duration_s:.1f} seconds."
        )
        parts: list[dict[str, Any]] = [{"text": prompt}]
        refs: list[str] = []
        for path in frame_paths:
            if path.is_file():
                parts.append(_file_part(path))
                refs.append(path.name)
        contents: list[dict[str, Any]] = []
        if job_dir is not None:
            for turn in load_chat(job_dir):
                role = "user" if turn.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn.text}]})
        contents.append({"role": "user", "parts": parts})
        data = self._generate(
            contents=contents,
            schema=_SPLIT_SCHEMA,
            system=(
                "You split videos for enhancement, not generation. "
                "Return only structured JSON matching the schema."
            ),
        )
        proposal = validate_split_proposal(
            SplitProposal.from_dict(data),
            duration_s=duration_s,
        )
        if job_dir is not None:
            append_chat_turn(
                job_dir,
                ChatTurn(
                    role="user",
                    text=user_text if user_text else prompt,
                    frame_refs=refs,
                ),
            )
            append_chat_turn(
                job_dir,
                ChatTurn(
                    role="model",
                    text=json.dumps(proposal.to_dict()),
                    structured=proposal.to_dict(),
                ),
            )
        return proposal

    def propose_enhance(
        self,
        *,
        frame_paths: list[Path],
        segment_label: str,
        start_s: float,
        end_s: float,
        user_text: str | None = None,
        prior: SegmentEnhanceResult | None = None,
        job_dir: Path | None = None,
    ) -> SegmentEnhanceResult:
        """Ask Gemini for 3–5 allowlisted enhancement options.

        When ``user_text`` and ``prior`` are both set, the prompt combines the
        user's revise note with the previous structured options (Something else
        loop). Otherwise this is the first proposal for the segment.
        """
        if user_text and prior is not None:
            prompt = combine_enhance_revise_message(
                prior=prior,
                user_note=user_text,
                segment_label=segment_label,
                start_s=start_s,
                end_s=end_s,
            )
        elif user_text:
            prompt = (
                f"Act as a pro photographer. Segment {segment_label!r} runs from "
                f"{start_s:.1f}s to {end_s:.1f}s. The user wants different "
                f"options: {user_text.strip()}. Return 3 to 5 enhancement "
                "options. Ops must be from: white_balance, contrast, denoise, "
                "sharpen, stabilize, trim. Do not invent shell commands."
            )
        else:
            prompt = (
                f"Act as a pro photographer. Segment {segment_label!r} runs from "
                f"{start_s:.1f}s to {end_s:.1f}s. List issues and 3 to 5 "
                "enhancement options. Ops must be from: white_balance, contrast, "
                "denoise, sharpen, stabilize, trim. Do not invent shell commands."
            )
        parts: list[dict[str, Any]] = [{"text": prompt}]
        refs: list[str] = []
        for path in frame_paths:
            if path.is_file():
                parts.append(_file_part(path))
                refs.append(path.name)
        contents: list[dict[str, Any]] = []
        if job_dir is not None:
            for turn in load_chat(job_dir):
                role = "user" if turn.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn.text}]})
        contents.append({"role": "user", "parts": parts})
        data = self._generate(
            contents=contents,
            schema=_ENHANCE_SCHEMA,
            system=(
                "You enhance existing frames with structured ops only. "
                "When revising, incorporate the user's request into a fresh "
                "option set. Never propose generative edits or shell strings."
            ),
        )
        result = validate_enhance_result(SegmentEnhanceResult.from_dict(data))
        if job_dir is not None:
            append_chat_turn(
                job_dir,
                ChatTurn(
                    role="user",
                    text=prompt,
                    frame_refs=refs,
                ),
            )
            append_chat_turn(
                job_dir,
                ChatTurn(
                    role="model",
                    text=json.dumps(result.to_dict()),
                    structured=result.to_dict(),
                ),
            )
        return result

    def revise_enhance(
        self,
        *,
        prior: SegmentEnhanceResult,
        frame_paths: list[Path],
        segment_label: str,
        start_s: float,
        end_s: float,
        user_note: str,
        job_dir: Path | None = None,
    ) -> SegmentEnhanceResult:
        """Combine a Something-else note with prior options and re-propose."""
        return self.propose_enhance(
            frame_paths=frame_paths,
            segment_label=segment_label,
            start_s=start_s,
            end_s=end_s,
            user_text=user_note,
            prior=prior,
            job_dir=job_dir,
        )

    def propose_upscale_params(
        self,
        *,
        prior: dict[str, Any],
        user_note: str,
        duration_s: float,
        job_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Ask Gemini for allowlisted restore and RTX VSR knobs only.

        The model must not return a prompt, a model name, or FFmpeg argv.
        A time range in the note is context; the caller reprocesses the whole clip.
        """
        note = user_note.strip()
        if not note:
            raise GeminiError("revise note is empty")
        prompt = (
            "Revise restore and upscale settings for the whole clip. "
            f"Media duration is {duration_s:.1f} seconds. "
            "A time range in the note is context only. Reprocess the entire clip. "
            f"Current settings JSON: {json.dumps(prior)}. "
            f"User note: {note}. "
            "Return only restore_strength (0.0 to 0.35), scale (2, 3, or 4), "
            "and vsr_quality (LOW, MEDIUM, or HIGH). "
            "Do not emit a prompt, a model name, or FFmpeg argv."
        )
        contents: list[dict[str, Any]] = []
        if job_dir is not None:
            for turn in load_chat(job_dir):
                role = "user" if turn.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn.text}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        data = self._generate(
            contents=contents,
            schema=_UPSCALE_SCHEMA,
            system=(
                "You retune allowlisted restore and RTX VSR knobs. "
                "Never return a text prompt, a model name, or FFmpeg arguments."
            ),
        )
        if job_dir is not None:
            append_chat_turn(job_dir, ChatTurn(role="user", text=prompt))
            append_chat_turn(
                job_dir,
                ChatTurn(
                    role="model",
                    text=json.dumps(data),
                    structured=data,
                ),
            )
        return data
