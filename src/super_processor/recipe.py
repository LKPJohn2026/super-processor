"""Versioned Recipe JSON schema for constrained FFmpeg planning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

RECIPE_VERSION = "1"
RECIPE_FILE_NAME = "recipe.json"


class RecipeError(ValueError):
    """Raised when a recipe document is structurally invalid."""


class OpName(str, Enum):
    """Allowlisted processing operations for v1."""

    WHITE_BALANCE = "white_balance"
    CONTRAST = "contrast"
    DENOISE = "denoise"
    STABILIZE = "stabilize"
    REFRAME_VERTICAL = "reframe_vertical"
    ENCODE_HEVC_SIZE_CAP = "encode_hevc_size_cap"


class TargetMode(str, Enum):
    """Whether a recipe is intended for preview or final encode."""

    PREVIEW = "preview"
    FINAL = "final"


# Canonical processing order for DAG checks (validator enforces).
DEFAULT_OP_ORDER: tuple[OpName, ...] = (
    OpName.WHITE_BALANCE,
    OpName.CONTRAST,
    OpName.DENOISE,
    OpName.STABILIZE,
    OpName.REFRAME_VERTICAL,
    OpName.ENCODE_HEVC_SIZE_CAP,
)


@dataclass(slots=True)
class PreviewWindow:
    """Time window used for preview encodes."""

    start_s: float = 0.0
    duration_s: float = 15.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreviewWindow:
        return cls(
            start_s=float(data.get("start_s", 0.0)),
            duration_s=float(data.get("duration_s", 15.0)),
        )


@dataclass(slots=True)
class ExportTarget:
    """Social / delivery constraints for the recipe."""

    profile: str = "social_vertical"
    aspect: str = "9:16"
    codec: str = "hevc"
    max_size_mb: float | None = None
    max_height: int | None = 1920

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExportTarget:
        max_size = data.get("max_size_mb")
        max_height = data.get("max_height")
        return cls(
            profile=str(data.get("profile", "social_vertical")),
            aspect=str(data.get("aspect", "9:16")),
            codec=str(data.get("codec", "hevc")),
            max_size_mb=None if max_size is None else float(max_size),
            max_height=None if max_height is None else int(max_height),
        )


@dataclass(slots=True)
class RecipeTarget:
    """Output intent for preview or final processing."""

    mode: TargetMode = TargetMode.PREVIEW
    preview_window: PreviewWindow = field(default_factory=PreviewWindow)
    export: ExportTarget = field(default_factory=ExportTarget)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "preview_window": self.preview_window.to_dict(),
            "export": self.export.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecipeTarget:
        mode_raw = str(data.get("mode", TargetMode.PREVIEW.value))
        try:
            mode = TargetMode(mode_raw)
        except ValueError as exc:
            raise RecipeError(f"unsupported target mode: {mode_raw}") from exc
        return cls(
            mode=mode,
            preview_window=PreviewWindow.from_dict(
                dict(data.get("preview_window") or {})
            ),
            export=ExportTarget.from_dict(dict(data.get("export") or {})),
        )


@dataclass(slots=True)
class RecipeOp:
    """One allowlisted operation with parameters."""

    op: OpName
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op.value,
            "enabled": self.enabled,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecipeOp:
        if "op" not in data:
            raise RecipeError("recipe op missing 'op' field")
        try:
            op = OpName(str(data["op"]))
        except ValueError as exc:
            raise RecipeError(f"unsupported op: {data['op']!r}") from exc
        params = data.get("params") or {}
        if not isinstance(params, dict):
            raise RecipeError(f"op {op.value} params must be an object")
        return cls(op=op, enabled=bool(data.get("enabled", True)), params=dict(params))


@dataclass(slots=True)
class EncodeSettings:
    """Software encode settings for the recipe."""

    video_codec: str = "libx265"
    mode: str = "crf_or_size_cap"
    crf: int = 24
    preset: str = "medium"
    audio_action: str = "copy_or_aac"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": {
                "codec": self.video_codec,
                "mode": self.mode,
                "crf": self.crf,
                "preset": self.preset,
            },
            "audio": {"action": self.audio_action},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EncodeSettings:
        video = dict(data.get("video") or {})
        audio = dict(data.get("audio") or {})
        return cls(
            video_codec=str(video.get("codec", "libx265")),
            mode=str(video.get("mode", "crf_or_size_cap")),
            crf=int(video.get("crf", 24)),
            preset=str(video.get("preset", "medium")),
            audio_action=str(audio.get("action", "copy_or_aac")),
        )


@dataclass(slots=True)
class Recipe:
    """Constrained recipe document consumed by the FFmpeg template layer."""

    version: str
    job_id: str
    source_path: str
    facts_hash: str | None
    target: RecipeTarget
    ops: list[RecipeOp]
    encode: EncodeSettings = field(default_factory=EncodeSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "job_id": self.job_id,
            "source_path": self.source_path,
            "facts_hash": self.facts_hash,
            "target": self.target.to_dict(),
            "ops": [op.to_dict() for op in self.ops],
            "encode": self.encode.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Recipe:
        if not isinstance(data, dict):
            raise RecipeError("recipe root must be an object")
        version = str(data.get("version", ""))
        if version != RECIPE_VERSION:
            raise RecipeError(f"unsupported recipe version: {version!r}")
        if "job_id" not in data or "source_path" not in data:
            raise RecipeError("recipe requires job_id and source_path")
        ops_raw = data.get("ops")
        if not isinstance(ops_raw, list):
            raise RecipeError("recipe ops must be a list")
        ops = [RecipeOp.from_dict(item) for item in ops_raw]
        names = [op.op for op in ops]
        if len(names) != len(set(names)):
            raise RecipeError("recipe ops must be unique by name")
        return cls(
            version=version,
            job_id=str(data["job_id"]),
            source_path=str(data["source_path"]),
            facts_hash=(
                None if data.get("facts_hash") is None else str(data.get("facts_hash"))
            ),
            target=RecipeTarget.from_dict(dict(data.get("target") or {})),
            ops=ops,
            encode=EncodeSettings.from_dict(dict(data.get("encode") or {})),
        )

    def enabled_ops(self) -> list[RecipeOp]:
        """Return enabled operations in recipe order."""
        return [op for op in self.ops if op.enabled]


def empty_recipe(job_id: str, source_path: str) -> Recipe:
    """Create a disabled-ops recipe shell for a job."""
    return Recipe(
        version=RECIPE_VERSION,
        job_id=job_id,
        source_path=source_path,
        facts_hash=None,
        target=RecipeTarget(),
        ops=[RecipeOp(op=name, enabled=False) for name in DEFAULT_OP_ORDER],
    )


def recipe_path(job_dir: Path) -> Path:
    """Return the on-disk recipe path for a job."""
    return job_dir / RECIPE_FILE_NAME


def write_recipe(job_dir: Path, recipe: Recipe) -> Path:
    """Atomically write a recipe document."""
    path = recipe_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(recipe.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def load_recipe(job_dir: Path) -> Recipe:
    """Load and structurally validate a recipe from disk."""
    path = recipe_path(job_dir)
    if not path.is_file():
        raise RecipeError(f"recipe not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecipeError(f"recipe is not valid JSON: {path}") from exc
    return Recipe.from_dict(data)


def parse_recipe_json(text: str) -> Recipe:
    """Parse recipe JSON text into a Recipe."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RecipeError("recipe is not valid JSON") from exc
    return Recipe.from_dict(data)
