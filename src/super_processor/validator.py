"""Policy validation for Recipe documents before FFmpeg execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .probe import MediaFacts, load_media_facts
from .recipe import (
    DEFAULT_OP_ORDER,
    OpName,
    Recipe,
    RecipeError,
    load_recipe,
)

# Minimum average video bitrate we refuse to undercut for size-capped exports.
MIN_BITRATE_KBPS = 300.0


@dataclass(slots=True)
class ValidationIssue:
    """One policy violation or warning."""

    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a recipe against policy and media facts."""

    ok: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
        }


PARAM_BOUNDS: dict[OpName, dict[str, tuple[float, float]]] = {
    OpName.WHITE_BALANCE: {
        "temperature": (2000.0, 10000.0),
        "tint": (-100.0, 100.0),
    },
    OpName.CONTRAST: {
        "contrast": (0.5, 2.0),
        "brightness": (-0.5, 0.5),
        "gamma": (0.5, 2.0),
    },
    OpName.DENOISE: {
        "strength": (0.0, 1.0),
    },
    OpName.STABILIZE: {
        "shakiness": (1.0, 10.0),
        "smoothing": (1.0, 50.0),
        "max_crop_pct": (0.0, 30.0),
    },
    OpName.REFRAME_VERTICAL: {
        "padding": (0.0, 0.25),
    },
    OpName.ENCODE_HEVC_SIZE_CAP: {
        "max_size_mb": (1.0, 10_000.0),
        "max_height": (360.0, 4320.0),
    },
}


def validate_param_bounds(recipe: Recipe) -> list[ValidationIssue]:
    """Ensure numeric params stay inside allowlisted ranges."""
    issues: list[ValidationIssue] = []
    for op in recipe.enabled_ops():
        bounds = PARAM_BOUNDS.get(op.op, {})
        for key, value in op.params.items():
            if key not in bounds:
                # Unknown keys are rejected to keep templates deterministic.
                issues.append(
                    ValidationIssue(
                        code="unknown_param",
                        message=f"{op.op.value}: unknown param {key!r}",
                    )
                )
                continue
            low, high = bounds[key]
            try:
                number = float(value)
            except (TypeError, ValueError):
                issues.append(
                    ValidationIssue(
                        code="param_type",
                        message=f"{op.op.value}.{key} must be numeric",
                    )
                )
                continue
            if number < low or number > high:
                issues.append(
                    ValidationIssue(
                        code="param_bounds",
                        message=(
                            f"{op.op.value}.{key}={number} outside "
                            f"allowed range [{low}, {high}]"
                        ),
                    )
                )
    return issues


def validate_op_order(recipe: Recipe) -> list[ValidationIssue]:
    """Ensure enabled ops follow the canonical DAG order."""
    issues: list[ValidationIssue] = []
    order_index = {name: index for index, name in enumerate(DEFAULT_OP_ORDER)}
    last = -1
    for op in recipe.enabled_ops():
        current = order_index[op.op]
        if current < last:
            issues.append(
                ValidationIssue(
                    code="op_order",
                    message=(
                        f"op {op.op.value} appears out of canonical order "
                        f"relative to earlier enabled ops"
                    ),
                )
            )
        last = max(last, current)
    return issues


def validate_encode_settings(recipe: Recipe) -> list[ValidationIssue]:
    """Check software-only encode defaults for v1."""
    issues: list[ValidationIssue] = []
    if recipe.encode.video_codec != "libx265":
        issues.append(
            ValidationIssue(
                code="encode_codec",
                message="v1 requires encode.video.codec=libx265",
            )
        )
    if recipe.encode.crf < 10 or recipe.encode.crf > 40:
        issues.append(
            ValidationIssue(
                code="encode_crf",
                message=f"encode CRF {recipe.encode.crf} outside [10, 40]",
            )
        )
    if recipe.target.export.codec != "hevc":
        issues.append(
            ValidationIssue(
                code="export_codec",
                message="export.codec must be hevc in v1",
            )
        )
    return issues


def validate_probe_compliance(
    recipe: Recipe,
    facts: MediaFacts | None,
) -> list[ValidationIssue]:
    """Reject recipes that disagree with probed media facts."""
    issues: list[ValidationIssue] = []
    if facts is None:
        issues.append(
            ValidationIssue(
                code="missing_facts",
                message="media facts are required before validation",
            )
        )
        return issues

    if Path(recipe.source_path).resolve() != Path(facts.source_path).resolve():
        issues.append(
            ValidationIssue(
                code="source_mismatch",
                message="recipe source_path does not match probed media facts",
            )
        )

    if any(
        op.op is OpName.REFRAME_VERTICAL and op.enabled for op in recipe.ops
    ) and not facts.has_video:
        issues.append(
            ValidationIssue(
                code="reframe_without_video",
                message="reframe_vertical requires a video stream",
            )
        )

    window = recipe.target.preview_window
    if window.start_s < 0 or window.duration_s <= 0:
        issues.append(
            ValidationIssue(
                code="preview_window",
                message="preview window start_s must be >= 0 and duration_s > 0",
            )
        )
    if facts.duration_s is not None and window.start_s >= facts.duration_s:
        issues.append(
            ValidationIssue(
                code="preview_window",
                message="preview start_s is past media duration",
            )
        )
    return issues


def validate_size_cap_feasibility(
    recipe: Recipe,
    facts: MediaFacts | None,
) -> tuple[list[ValidationIssue], list[ValidationIssue]]:
    """Refuse or warn when a size cap cannot meet a minimum bitrate floor."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    export = recipe.target.export
    if export.max_size_mb is None or facts is None or facts.duration_s is None:
        return errors, warnings
    if facts.duration_s <= 0:
        errors.append(
            ValidationIssue(
                code="invalid_duration",
                message="media duration must be positive for size-cap checks",
            )
        )
        return errors, warnings

    budget_bits = export.max_size_mb * 1024 * 1024 * 8
    avg_kbps = (budget_bits / facts.duration_s) / 1000.0
    if avg_kbps < MIN_BITRATE_KBPS:
        errors.append(
            ValidationIssue(
                code="size_cap_infeasible",
                message=(
                    f"max_size_mb={export.max_size_mb} implies "
                    f"~{avg_kbps:.1f} kbps over {facts.duration_s:.1f}s; "
                    f"below floor {MIN_BITRATE_KBPS:.0f} kbps"
                ),
            )
        )
    elif avg_kbps < MIN_BITRATE_KBPS * 1.5:
        warnings.append(
            ValidationIssue(
                code="size_cap_tight",
                message=(
                    f"max_size_mb={export.max_size_mb} implies ~{avg_kbps:.1f} kbps; "
                    "quality may be poor"
                ),
                severity="warning",
            )
        )
    return errors, warnings


def validate_recipe(
    recipe: Recipe,
    facts: MediaFacts | None = None,
) -> ValidationResult:
    """Validate structure-adjacent policy for a recipe."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    errors.extend(validate_param_bounds(recipe))
    errors.extend(validate_op_order(recipe))
    errors.extend(validate_encode_settings(recipe))
    errors.extend(validate_probe_compliance(recipe, facts))
    size_errors, size_warnings = validate_size_cap_feasibility(recipe, facts)
    errors.extend(size_errors)
    warnings.extend(size_warnings)

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_job_recipe(job_dir: Path) -> ValidationResult:
    """Load recipe and media facts from a job directory and validate."""
    try:
        recipe = load_recipe(job_dir)
    except RecipeError as exc:
        return ValidationResult(
            ok=False,
            errors=[ValidationIssue(code="recipe_load", message=str(exc))],
        )
    facts: MediaFacts | None
    try:
        facts = load_media_facts(job_dir)
    except Exception:
        facts = None
    return validate_recipe(recipe, facts)
