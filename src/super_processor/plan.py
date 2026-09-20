"""CV-first recipe planner with optional natural-language patches."""

from __future__ import annotations

import re
from pathlib import Path

from .diagnose import Diagnosis, load_diagnosis
from .estimators import OpSuggestion
from .probe import load_media_facts
from .recipe import (
    DEFAULT_OP_ORDER,
    OpName,
    Recipe,
    RecipeOp,
    empty_recipe,
    write_recipe,
)

# Lightweight NL patches applied on top of CV priors (LLM is a no-op until v1.0).
_PATCHES: list[tuple[re.Pattern[str], str, dict[str, float] | None]] = [
    (re.compile(r"\bless\s+denoise\b", re.I), "denoise", {"strength": 0.15}),
    (re.compile(r"\bmore\s+denoise\b", re.I), "denoise", {"strength": 0.55}),
    (re.compile(r"\bno\s+denoise\b", re.I), "denoise", None),
    (re.compile(r"\bwarmer\b", re.I), "white_balance", {"temperature": 5800.0}),
    (re.compile(r"\bcooler\b", re.I), "white_balance", {"temperature": 7200.0}),
    (re.compile(r"\bless\s+contrast\b", re.I), "contrast", {"contrast": 1.05}),
    (re.compile(r"\bmore\s+contrast\b", re.I), "contrast", {"contrast": 1.35}),
    (re.compile(r"\bno\s+stabilize\b|\bno\s+stabilisation\b", re.I), "stabilize", None),
]


class PlanError(RuntimeError):
    """Raised when a recipe cannot be planned."""


def _suggestion_to_op(name: OpName, suggestion: OpSuggestion) -> RecipeOp:
    return RecipeOp(op=name, enabled=suggestion.enabled, params=dict(suggestion.params))


def recipe_from_diagnosis(
    diagnosis: Diagnosis,
    *,
    job_id: str,
    source_path: str,
    instruction: str | None = None,
) -> Recipe:
    """Build an allowlisted recipe from CV diagnosis and optional NL patches."""
    recipe = empty_recipe(job_id, source_path)
    mapping = {
        OpName.CONTRAST: diagnosis.suggestions.get("contrast"),
        OpName.WHITE_BALANCE: diagnosis.suggestions.get("white_balance"),
        OpName.DENOISE: diagnosis.suggestions.get("denoise"),
        OpName.STABILIZE: diagnosis.suggestions.get("stabilize"),
    }
    ops: list[RecipeOp] = []
    for name in DEFAULT_OP_ORDER:
        suggestion = mapping.get(name)
        if suggestion is not None:
            ops.append(_suggestion_to_op(name, suggestion))
        else:
            # Keep disabled placeholders for reframe/encode until social plan says so.
            existing = next(op for op in recipe.ops if op.op is name)
            ops.append(existing)

    if diagnosis.reframe is not None:
        for op in ops:
            if op.op is OpName.REFRAME_VERTICAL:
                op.enabled = True
                reframe = diagnosis.reframe
                params: dict[str, float] = {
                    "padding": float(reframe.get("padding", 0.0)),
                }
                for key in ("crop_x", "crop_y", "crop_w", "crop_h", "subject_cx"):
                    if key in reframe:
                        params[key] = float(reframe[key])
                op.params = params
    if diagnosis.size_cap is not None:
        for op in ops:
            if op.op is OpName.ENCODE_HEVC_SIZE_CAP:
                op.enabled = True
                params = {
                    "max_size_mb": float(diagnosis.size_cap.get("max_size_mb", 50.0)),
                    "max_height": float(
                        (diagnosis.reframe or {}).get("output_height", 1920)
                    ),
                }
                if float(diagnosis.size_cap.get("acknowledge_size_risk", 0.0) or 0.0):
                    params["acknowledge_size_risk"] = 1.0
                op.params = params
        recipe.target.export.max_size_mb = float(
            diagnosis.size_cap.get("max_size_mb", 50.0)
        )

    recipe.ops = ops
    if instruction:
        recipe = apply_instruction_patches(recipe, instruction)
    return recipe


def apply_instruction_patches(recipe: Recipe, instruction: str) -> Recipe:
    """Apply deterministic NL patches; unknown phrases are ignored."""
    text = instruction.strip()
    if not text:
        return recipe
    ops = {op.op: op for op in recipe.ops}
    for pattern, name, params in _PATCHES:
        if not pattern.search(text):
            continue
        op_name = OpName(name)
        op = ops[op_name]
        if params is None:
            op.enabled = False
            op.params = {}
        else:
            op.enabled = True
            merged = dict(op.params)
            merged.update(params)
            op.params = merged
    recipe.ops = [ops[name] for name in DEFAULT_OP_ORDER]
    return recipe


def plan_job(
    job_dir: Path,
    job_id: str,
    *,
    instruction: str | None = None,
) -> Recipe:
    """Load diagnosis + facts and write a planned recipe."""
    try:
        diagnosis = load_diagnosis(job_dir)
    except Exception as exc:
        raise PlanError(f"diagnosis required before plan: {exc}") from exc
    facts = load_media_facts(job_dir)
    recipe = recipe_from_diagnosis(
        diagnosis,
        job_id=job_id,
        source_path=facts.source_path,
        instruction=instruction,
    )
    write_recipe(job_dir, recipe)
    return recipe
