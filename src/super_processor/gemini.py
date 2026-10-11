"""Gemini structured-output client for the localhost wizard.

Uses the Generative Language REST API with ``responseMimeType`` /
``responseSchema`` so replies stay parseable. The wizard asks Gemini to label
the shots FFmpeg found (from one still each plus the measurements), and to
turn a result note into a time range, scale, and strength for the FlashVSR
upscale. Transport is injectable for tests.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .models import resolve_secret

GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_MODEL_ENV = "SUPER_PROCESSOR_GEMINI_MODEL"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
CHAT_FILE_NAME = "gemini_chat.json"
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
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as response:
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
                    "look": _LOOK_SCHEMA,
                    "reason": {"type": "string"},
                },
                "required": ["index", "strength", "look", "reason"],
            },
        },
    },
    "required": ["scale", "shots"],
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


class GeminiClient:
    """Call Gemini with a structured schema for upscale result notes."""

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

    def label_shots(
        self,
        *,
        shots: list[dict[str, Any]],
        stills: list[Path],
        issues: tuple[str, ...],
        contains: tuple[str, ...],
        job_dir: Path | None = None,
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
            parts.append(_jpeg_part(still))
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=_SHOT_LABELS_SCHEMA,
            system=(
                "You label shots for a video restoration tool. You never "
                "suggest commands or settings. Return JSON matching the schema."
            ),
        )
        labels = reply.get("shots")
        if not isinstance(labels, list):
            raise GeminiError("Gemini returned no shot labels")
        if job_dir is not None:
            append_chat_turn(
                job_dir,
                ChatTurn(role="user", text=intro + f" ({len(shots)} shots)"),
            )
            append_chat_turn(
                job_dir,
                ChatTurn(role="model", text=json.dumps(reply), structured=reply),
            )
        return [item for item in labels if isinstance(item, dict)]

    def propose_looks(
        self,
        *,
        shots: list[dict[str, Any]],
        stills: list[Path],
        bounds: dict[str, tuple[float, float, float]],
        scales: tuple[int, ...],
        protected_strength: float,
        job_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Propose a strength and look per shot, and one scale for the video.

        ``shots`` carry their label, problems, contents, measurements, and,
        on a redo, the current settings and the editor's note.
        """
        if len(stills) != len(shots):
            raise GeminiError("propose_looks needs one still per shot")
        limits = "; ".join(
            f"{name} {low:g} to {high:g} (no change at {neutral:g})"
            for name, (low, high, neutral) in bounds.items()
        )
        intro = (
            "You plan a restoration of an editor's footage. Each shot goes "
            "through fixed FFmpeg clean-up (deblock, denoise), then FlashVSR "
            "upscaling, whose fine detail is added to a plain upscale by "
            "'strength' (0 adds none, 1 adds all of it), then fixed FFmpeg finishing "
            "(contrast, brightness, saturation, gamma, grain). FlashVSR treats "
            "blocks and noise as detail and sharpens them into texture, so "
            "clean those first and do not ask for more strength than the shot "
            "needs. The goal is footage that looks well shot, not AI made: no "
            "waxy skin, no crunchy edges, no invented patterns, faces and text "
            "unchanged. Set only what the shot needs; leave the rest at its "
            f"no-change value. Ranges: {limits}. Shapes and colour always "
            "come from the source; strength only sets how much of FlashVSR's "
            "fine detail is added. A shot listing protected boxes is held "
            "down inside them automatically, so its strength can suit the "
            "rest of the frame; a shot that contains faces, hands, or text "
            f"without boxes is capped at strength {protected_strength:g}. "
            f"Pick one scale for the whole video from {list(scales)}. If a "
            "shot has an editor_note, follow it; current shows its settings "
            "now. Return a short reason per shot in plain words."
        )
        parts: list[dict[str, Any]] = [{"text": intro}]
        for shot, still in zip(shots, stills, strict=True):
            parts.append({"text": f"Shot: {json.dumps(shot)}"})
            parts.append(_jpeg_part(still))
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=_PROPOSE_SCHEMA,
            system=(
                "You choose bounded numbers for a fixed video pipeline. You "
                "never write commands. Return JSON matching the schema."
            ),
        )
        if not isinstance(reply.get("shots"), list):
            raise GeminiError("Gemini returned no shot recipes")
        self._log(job_dir, intro + f" ({len(shots)} shots)", reply)
        return reply

    def locate_regions(
        self,
        *,
        shots: list[dict[str, Any]],
        stills: list[list[Path]],
        kinds: tuple[str, ...],
        max_regions: int,
        job_dir: Path | None = None,
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
            parts.extend(_jpeg_part(frame) for frame in frames)
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=_REGIONS_SCHEMA,
            system=(
                "You locate faces, hands, and text in video stills. Return "
                "JSON matching the schema."
            ),
        )
        found = reply.get("shots")
        if not isinstance(found, list):
            raise GeminiError("Gemini returned no regions")
        self._log(job_dir, intro + f" ({len(shots)} shots)", reply)
        return [item for item in found if isinstance(item, dict)]

    def check_previews(
        self,
        *,
        shots: list[dict[str, Any]],
        pairs: list[tuple[Path, Path]],
        problems: tuple[str, ...],
        job_dir: Path | None = None,
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
            "settings unchanged. Keep the note to one sentence."
        )
        parts: list[dict[str, Any]] = [{"text": intro}]
        for shot, (before, after) in zip(shots, pairs, strict=True):
            parts.append({"text": f"Shot: {json.dumps(shot)}. BEFORE:"})
            parts.append(_jpeg_part(before))
            parts.append({"text": "AFTER:"})
            parts.append(_jpeg_part(after))
        reply = self._generate(
            contents=[{"role": "user", "parts": parts}],
            schema=_CHECK_SCHEMA,
            system=(
                "You check restoration previews for visible mistakes. Return "
                "JSON matching the schema."
            ),
        )
        verdicts = reply.get("shots")
        if not isinstance(verdicts, list):
            raise GeminiError("Gemini returned no preview checks")
        self._log(job_dir, intro + f" ({len(shots)} shots)", reply)
        return [item for item in verdicts if isinstance(item, dict)]

    def _log(self, job_dir: Path | None, prompt: str, reply: dict[str, Any]) -> None:
        if job_dir is None:
            return
        append_chat_turn(job_dir, ChatTurn(role="user", text=prompt))
        append_chat_turn(
            job_dir, ChatTurn(role="model", text=json.dumps(reply), structured=reply)
        )

    def revise_upscale_params(
        self,
        *,
        note: str,
        prior_scale: int,
        prior_strength: float,
        prior_start_s: float,
        prior_end_s: float,
        duration_s: float,
        job_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Turn a result note into a bounded scale, strength, and time range."""
        prompt = (
            "We already restored and upscaled a video locally. "
            f"Duration is {duration_s:.1f} seconds. Previous params: "
            f"scale={prior_scale}, strength={prior_strength:.2f}, "
            f"applied from {prior_start_s:.1f}s to {prior_end_s:.1f}s. "
            f"The user said: {note.strip()}. "
            "Return start_s, end_s, scale (only 2 or 4), and strength from "
            "0 to 1. Lower strength means less invented texture. If they "
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
        )
        if job_dir is not None:
            append_chat_turn(job_dir, ChatTurn(role="user", text=prompt))
            append_chat_turn(
                job_dir,
                ChatTurn(role="model", text=json.dumps(data), structured=data),
            )
        return data


def _jpeg_part(path: Path) -> dict[str, Any]:
    try:
        data = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise GeminiError(f"could not read {path.name}: {exc}") from exc
    return {"inlineData": {"mimeType": "image/jpeg", "data": data}}
