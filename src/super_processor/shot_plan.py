"""Loop 2: a recipe per shot, short previews, a self-check, editor approval.

Every approved shot gets a recipe: a FlashVSR ``strength`` and a
:class:`~super_processor.look.ShotLook`. The whole video shares one ``scale``.
Gemini proposes the recipes from the shot labels and measurements (or the
rules in :func:`fallback_recipe` do, without Gemini). Each shot then gets a
short before/after preview, Gemini compares the two for the usual restoration
mistakes, and may correct the recipe once. The editor approves the lot or
writes a note on one shot, which sends only that shot round again.

Model numbers are clamped into :data:`~super_processor.look.LOOK_BOUNDS`, and
shots that contain faces, hands, or text are capped at
:data:`PROTECTED_STRENGTH`, because those are where invented detail shows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .look import LOOK_BOUNDS, ShotLook
from .shots import Shot, ShotError, ShotList, extract_still
from .upscale import (
    UpscaleEngine,
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
    encode_delivery,
    trim_source,
    upscale_all,
)

PLAN_FILE = "shot_plan.json"
PLAN_SCHEMA_VERSION = 1
PREVIEWS_DIR = "previews"
# Preview length, from the middle of the shot.
PREVIEW_S = 3.0
CHECK_STILL_WIDTH = 960
# 4x only for small sources; on larger ones it is mostly invented detail.
MAX_4X_LONG_EDGE = 960
DEFAULT_STRENGTH = 0.5
PROTECTED_CONTENTS = frozenset({"faces", "hands", "text"})
PROTECTED_STRENGTH = 0.6
# What the self-check looks for. Gemini may only use these names.
CHECK_PROBLEMS = (
    "identity_change",
    "bad_anatomy",
    "garbled_text",
    "waxy_skin",
    "oversharpened",
    "halos",
    "fake_texture",
    "color_shift",
    "too_soft",
)
PREVIEW_NAME = re.compile(r"shot_(\d{3})_r(\d+)_(before|after)\.(mp4|jpg)")


class ShotPlanError(RuntimeError):
    """Raised when the per-shot plan cannot be built or edited."""


@dataclass(frozen=True, slots=True)
class ShotCheck:
    """Gemini's verdict on one preview. ``ok`` is ``None`` when not checked."""

    ok: bool | None = None
    problems: tuple[str, ...] = ()
    note: str = ""
    adjusted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "note": self.note,
            "adjusted": self.adjusted,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ShotCheck:
        if not isinstance(data, dict):
            return cls()
        ok = data.get("ok")
        return cls(
            ok=ok if isinstance(ok, bool) else None,
            problems=_known(data.get("problems"), CHECK_PROBLEMS),
            note=str(data.get("note", "")),
            adjusted=data.get("adjusted") is True,
        )


@dataclass(frozen=True, slots=True)
class ShotRecipe:
    """Settings for one shot, why they were chosen, and its preview state."""

    strength: float = DEFAULT_STRENGTH
    look: ShotLook = field(default_factory=ShotLook)
    reason: str = ""
    check: ShotCheck = field(default_factory=ShotCheck)
    rev: int = 0
    stale: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "look": self.look.to_dict(),
            "reason": self.reason,
            "check": self.check.to_dict(),
            "rev": self.rev,
            "stale": self.stale,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ShotRecipe:
        if not isinstance(data, dict):
            raise ShotPlanError("a shot recipe must be an object")
        return cls(
            strength=_clamp(data.get("strength"), 0.0, 1.0, DEFAULT_STRENGTH),
            look=look_from_model(data.get("look")),
            reason=str(data.get("reason", "")),
            check=ShotCheck.from_dict(data.get("check")),
            rev=int(data.get("rev", 0)),
            stale=bool(data.get("stale", True)),
            note=str(data.get("note", "")),
        )


@dataclass(slots=True)
class ShotPlan:
    """One recipe per approved shot, plus the scale for the whole video."""

    scale: int
    recipes: list[ShotRecipe]
    error: str | None = None
    schema_version: int = PLAN_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scale": self.scale,
            "error": self.error,
            "recipes": [recipe.to_dict() for recipe in self.recipes],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ShotPlan:
        if not isinstance(data, dict) or not isinstance(data.get("recipes"), list):
            raise ShotPlanError("shot_plan.json must hold a list of recipes")
        scale = data.get("scale")
        error = data.get("error")
        return cls(
            scale=scale if scale in (2, 4) else 2,
            recipes=[ShotRecipe.from_dict(item) for item in data["recipes"]],
            error=str(error) if error else None,
        )

    def stale_indexes(self) -> list[int]:
        return [i for i, recipe in enumerate(self.recipes) if recipe.stale]

    def to_upscale_plan(self, shots: ShotList) -> UpscalePlan:
        """One upscale span per shot with its strength and look."""
        if len(shots.shots) != len(self.recipes):
            raise ShotPlanError("the shot list and the recipes no longer match")
        return UpscalePlan(
            spans=tuple(
                UpscaleSpan(
                    shot.start_s,
                    shot.end_s,
                    scale=self.scale,
                    strength=recipe.strength,
                    look=recipe.look,
                )
                for shot, recipe in zip(shots.shots, self.recipes, strict=True)
            )
        )


def _known(raw: Any, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    seen: list[str] = []
    for item in raw:
        name = str(item).strip().lower()
        if name in allowed and name not in seen:
            seen.append(name)
    return tuple(seen)


def _clamp(raw: Any, low: float, high: float, default: float) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    value = float(raw)
    if value != value:  # NaN
        return default
    return min(high, max(low, value))


def look_from_model(raw: Any) -> ShotLook:
    """A look from model output: unknown keys dropped, values clamped."""
    if not isinstance(raw, dict):
        return ShotLook()
    values = {
        name: _clamp(raw.get(name), low, high, neutral)
        for name, (low, high, neutral) in LOOK_BOUNDS.items()
    }
    return ShotLook(**values)


def allowed_scales(width: int, height: int) -> tuple[int, ...]:
    """4x only when the long edge is small enough for it to recover detail."""
    return (2, 4) if max(width, height) <= MAX_4X_LONG_EDGE else (2,)


def cap_strength(strength: float, shot: Shot) -> float:
    """Hold strength down on shots with faces, hands, or text."""
    if PROTECTED_CONTENTS.intersection(shot.contains):
        return min(strength, PROTECTED_STRENGTH)
    return strength


def fallback_recipe(shot: Shot) -> ShotRecipe:
    """A recipe from the measurements alone, for when Gemini is unavailable."""
    issues = set(shot.issues)
    deblock = 0.6 if "blocky" in issues else 0.0
    denoise = 0.5 if "noisy" in issues else 0.0
    brightness = (
        0.05 if "dark" in issues else (-0.04 if "overexposed" in issues else 0.0)
    )
    gamma = 1.1 if "dark" in issues else 1.0
    contrast = 1.1 if "flat" in issues else 1.0
    strength = DEFAULT_STRENGTH
    if "soft" in issues:
        strength = 0.6
    if issues & {"blocky", "noisy"}:
        # Clean first, and ask FlashVSR for less: what is left of the damage
        # is what it would sharpen into texture.
        strength = min(strength, 0.45)
    look = ShotLook(
        deblock=deblock,
        denoise=denoise,
        brightness=brightness,
        gamma=gamma,
        contrast=contrast,
    )
    reasons = [name.replace("_", " ") for name in shot.issues] or ["no clear problem"]
    return ShotRecipe(
        strength=cap_strength(strength, shot),
        look=look,
        reason="Set from the measurements (" + ", ".join(reasons) + ").",
    )


def apply_proposals(
    plan: ShotPlan,
    shots: ShotList,
    proposals: list[dict[str, Any]],
    indexes: list[int],
) -> ShotPlan:
    """Merge Gemini's recipes for ``indexes``; the rest keep theirs."""
    recipes = list(plan.recipes)
    by_index = {
        item["index"]: item
        for item in proposals
        if isinstance(item, dict)
        and isinstance(item.get("index"), int)
        and not isinstance(item.get("index"), bool)
    }
    missing: list[int] = []
    for index in indexes:
        item = by_index.get(index)
        shot = shots.shots[index]
        if item is None:
            missing.append(index)
            fallback = fallback_recipe(shot)
            recipes[index] = replace(
                fallback, rev=recipes[index].rev, note=recipes[index].note
            )
            continue
        strength = _clamp(item.get("strength"), 0.0, 1.0, DEFAULT_STRENGTH)
        recipes[index] = replace(
            recipes[index],
            strength=cap_strength(strength, shot),
            look=look_from_model(item.get("look")),
            reason=" ".join(str(item.get("reason", "")).split())[:240],
        )
    error = plan.error
    if missing:
        listed = ", ".join(str(i + 1) for i in missing)
        error = f"Gemini gave no recipe for shot {listed}; those use the measurements."
    return ShotPlan(scale=plan.scale, recipes=recipes, error=error)


def apply_checks(
    plan: ShotPlan, shots: ShotList, checks: list[dict[str, Any]], indexes: list[int]
) -> tuple[ShotPlan, list[int]]:
    """Store the self-check results. Returns the shots whose recipe changed."""
    recipes = list(plan.recipes)
    changed: list[int] = []
    by_index = {
        item.get("index"): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("index"), int)
    }
    for index in indexes:
        item = by_index.get(index)
        if item is None:
            continue
        verdict = ShotCheck.from_dict(item)
        recipe = replace(recipes[index], check=verdict)
        if verdict.ok is False:
            strength = cap_strength(
                _clamp(item.get("strength"), 0.0, 1.0, recipe.strength),
                shots.shots[index],
            )
            look = look_from_model(item.get("look")) if "look" in item else recipe.look
            if (strength, look) != (recipe.strength, recipe.look):
                recipe = replace(
                    recipe,
                    strength=strength,
                    look=look,
                    check=replace(verdict, adjusted=True),
                )
                changed.append(index)
        recipes[index] = recipe
    return ShotPlan(scale=plan.scale, recipes=recipes, error=plan.error), changed


def shots_payload(
    shots: ShotList, plan: ShotPlan | None, indexes: list[int]
) -> list[dict[str, Any]]:
    """What Gemini sees about each shot it is asked to plan."""
    items: list[dict[str, Any]] = []
    for index in indexes:
        shot = shots.shots[index]
        item: dict[str, Any] = {
            "index": index,
            "start_s": round(shot.start_s, 2),
            "end_s": round(shot.end_s, 2),
            "label": shot.label,
            "issues": list(shot.issues),
            "contains": list(shot.contains),
            "metrics": shot.metrics.to_dict(),
        }
        if plan is not None and index < len(plan.recipes):
            recipe = plan.recipes[index]
            if recipe.reason:
                item["current"] = {
                    "strength": recipe.strength,
                    "look": recipe.look.to_dict(),
                }
            if recipe.note:
                item["editor_note"] = recipe.note
        items.append(item)
    return items


def plan_path(job_dir: Path) -> Path:
    return job_dir / PLAN_FILE


def save_plan(job_dir: Path, plan: ShotPlan) -> Path:
    path = plan_path(job_dir)
    path.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_plan(job_dir: Path) -> ShotPlan | None:
    path = plan_path(job_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShotPlanError(f"shot_plan.json is not valid JSON: {exc}") from exc
    return ShotPlan.from_dict(data)


def preview_window(shot: Shot) -> tuple[float, float]:
    """At most :data:`PREVIEW_S` seconds from the middle of the shot."""
    if shot.duration_s <= PREVIEW_S:
        return shot.start_s, shot.end_s
    middle = (shot.start_s + shot.end_s) / 2
    return middle - PREVIEW_S / 2, middle + PREVIEW_S / 2


def preview_name(index: int, rev: int, side: str, ext: str = "mp4") -> str:
    return f"shot_{index:03d}_r{rev}_{side}.{ext}"


def render_previews(
    source: Path,
    job_dir: Path,
    shots: ShotList,
    plan: ShotPlan,
    indexes: list[int],
    *,
    engine: UpscaleEngine,
    ffmpeg_bin: str = "ffmpeg",
) -> ShotPlan:
    """Render before/after previews and check stills for ``indexes``.

    Each shot's ``rev`` goes up by one, so its files get new names and a
    browser never shows a cached older preview. Older files are removed.
    """
    folder = job_dir / PREVIEWS_DIR
    work = folder / "work"
    work.mkdir(parents=True, exist_ok=True)
    recipes = list(plan.recipes)
    clips: dict[int, Path] = {}
    requests: list[UpscaleRequest] = []
    for index in indexes:
        recipe = replace(recipes[index], rev=recipes[index].rev + 1)
        recipes[index] = recipe
        start, end = preview_window(shots.shots[index])
        clip = trim_source(
            source,
            work / f"shot_{index:03d}_in.mp4",
            start,
            end,
            ffmpeg_bin=ffmpeg_bin,
            video_filter="setpts=PTS-STARTPTS",
        )
        clips[index] = clip
        requests.append(
            UpscaleRequest(
                clip,
                work / f"shot_{index:03d}_up.mp4",
                scale=plan.scale,
                strength=recipe.strength,
                look=recipe.look,
            )
        )
    raws = upscale_all(engine, requests)
    for index, raw in zip(indexes, raws, strict=True):
        recipe = recipes[index]
        _remove_old_previews(folder, index)
        after = folder / preview_name(index, recipe.rev, "after")
        before = folder / preview_name(index, recipe.rev, "before")
        encode_delivery(
            raw,
            after,
            ffmpeg_bin=ffmpeg_bin,
            blend_with=clips[index],
            strength=recipe.strength,
            look=recipe.look,
        )
        encode_delivery(clips[index], before, ffmpeg_bin=ffmpeg_bin)
        for side, video in (("before", before), ("after", after)):
            middle = _middle_s(shots.shots[index])
            extract_still(
                video,
                middle,
                folder / preview_name(index, recipe.rev, side, "jpg"),
                ffmpeg_bin=ffmpeg_bin,
                width=CHECK_STILL_WIDTH,
            )
    return ShotPlan(scale=plan.scale, recipes=recipes, error=plan.error)


def _middle_s(shot: Shot) -> float:
    start, end = preview_window(shot)
    return (end - start) / 2


def _remove_old_previews(folder: Path, index: int) -> None:
    for old in folder.glob(f"shot_{index:03d}_r*_*.*"):
        # The delivery encode leaves a ``.delivery.json`` marker beside each.
        if PREVIEW_NAME.fullmatch(old.name.removesuffix(".delivery.json")):
            old.unlink(missing_ok=True)


def check_stills(job_dir: Path, plan: ShotPlan, index: int) -> tuple[Path, Path]:
    rev = plan.recipes[index].rev
    folder = job_dir / PREVIEWS_DIR
    return (
        folder / preview_name(index, rev, "before", "jpg"),
        folder / preview_name(index, rev, "after", "jpg"),
    )


def finish(plan: ShotPlan, indexes: list[int]) -> ShotPlan:
    """Mark ``indexes`` as up to date and clear their editor notes."""
    recipes = list(plan.recipes)
    for index in indexes:
        recipes[index] = replace(recipes[index], stale=False, note="")
    return ShotPlan(scale=plan.scale, recipes=recipes, error=plan.error)


def request_redo(plan: ShotPlan, index: int, note: str) -> ShotPlan:
    """Send one shot round again with the editor's note."""
    if not 0 <= index < len(plan.recipes):
        raise ShotPlanError("no such shot")
    text = " ".join(note.split())
    if not text:
        raise ShotPlanError("say what to change about this shot")
    recipes = list(plan.recipes)
    recipes[index] = replace(recipes[index], stale=True, note=text[:500])
    return ShotPlan(scale=plan.scale, recipes=recipes, error=None)


def new_plan(shots: ShotList, scale: int) -> ShotPlan:
    if not shots.shots:
        raise ShotError("there are no shots to plan")
    return ShotPlan(scale=scale, recipes=[ShotRecipe() for _ in shots.shots])
