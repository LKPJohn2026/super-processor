"""CV-only diagnosis documents produced from look/motion estimates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .estimators import LookEstimates, OpSuggestion, estimate_look, load_estimates
from .probe import MediaFacts, load_media_facts
from .reframe import SocialExportPlan, load_reframe_plan, plan_social_export

DIAGNOSIS_FILE_NAME = "diagnosis.json"
DIAGNOSIS_SCHEMA_VERSION = 1


class DiagnosisError(RuntimeError):
    """Raised when diagnosis cannot be produced."""


@dataclass(slots=True)
class Diagnosis:
    """Machine-readable diagnosis used by the planner."""

    schema_version: int
    source_path: str
    summary: str
    suggestions: dict[str, OpSuggestion]
    metrics: dict[str, float] = field(default_factory=dict)
    reframe: dict[str, Any] | None = None
    size_cap: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "summary": self.summary,
            "suggestions": {
                name: suggestion.to_dict()
                for name, suggestion in self.suggestions.items()
            },
            "metrics": dict(self.metrics),
            "reframe": self.reframe,
            "size_cap": self.size_cap,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Diagnosis:
        suggestions_raw = dict(data.get("suggestions") or {})
        return cls(
            schema_version=int(data.get("schema_version", DIAGNOSIS_SCHEMA_VERSION)),
            source_path=str(data["source_path"]),
            summary=str(data.get("summary", "")),
            suggestions={
                str(name): OpSuggestion.from_dict(dict(payload))
                for name, payload in suggestions_raw.items()
            },
            metrics={
                str(key): float(value)
                for key, value in dict(data.get("metrics") or {}).items()
            },
            reframe=(
                None if data.get("reframe") is None else dict(data.get("reframe") or {})
            ),
            size_cap=(
                None
                if data.get("size_cap") is None
                else dict(data.get("size_cap") or {})
            ),
        )


def build_diagnosis(
    estimates: LookEstimates,
    social: SocialExportPlan | None = None,
) -> Diagnosis:
    """Combine look estimates and optional social-export plan into a diagnosis."""
    suggestions = {
        "contrast": estimates.contrast,
        "white_balance": estimates.white_balance,
        "denoise": estimates.denoise,
        "stabilize": estimates.stabilize,
    }
    enabled = [name for name, item in suggestions.items() if item.enabled]
    if enabled:
        summary = "Suggested ops: " + ", ".join(enabled)
    else:
        summary = "No look/motion corrections suggested"
    if social is not None and social.size_cap is not None:
        summary += f"; size-cap {social.size_cap.status}"

    return Diagnosis(
        schema_version=DIAGNOSIS_SCHEMA_VERSION,
        source_path=estimates.source_path,
        summary=summary,
        suggestions=suggestions,
        metrics=dict(estimates.metrics),
        reframe=None if social is None else social.reframe.to_dict(),
        size_cap=(
            None
            if social is None or social.size_cap is None
            else social.size_cap.to_dict()
        ),
    )


def diagnose_source(
    source: Path,
    facts: MediaFacts,
    *,
    max_size_mb: float | None = None,
    max_height: int = 1920,
    acknowledge_size_risk: bool = False,
) -> Diagnosis:
    """Run estimators (+ social plan) for a source file."""
    estimates = estimate_look(source, facts)
    social = plan_social_export(
        facts,
        max_height=max_height,
        max_size_mb=max_size_mb,
        padding=0.0,
        source=source,
    )
    diagnosis = build_diagnosis(estimates, social)
    if (
        acknowledge_size_risk
        and diagnosis.size_cap is not None
        and diagnosis.size_cap.get("status") == "infeasible"
    ):
        diagnosis.size_cap = dict(diagnosis.size_cap)
        diagnosis.size_cap["acknowledge_size_risk"] = 1.0
        diagnosis.summary += "; size risk acknowledged"
    return diagnosis


def diagnose_job_dir(
    job_dir: Path,
    *,
    max_size_mb: float | None = None,
    acknowledge_size_risk: bool = False,
) -> Diagnosis:
    """Diagnose using on-disk facts/estimates when present, else compute."""
    facts = load_media_facts(job_dir)
    try:
        estimates = load_estimates(job_dir)
    except Exception:
        estimates = estimate_look(Path(facts.source_path), facts)
    try:
        social = load_reframe_plan(job_dir)
    except Exception:
        social = plan_social_export(
            facts,
            max_size_mb=max_size_mb,
            source=Path(facts.source_path),
        )
    diagnosis = build_diagnosis(estimates, social)
    if (
        acknowledge_size_risk
        and diagnosis.size_cap is not None
        and diagnosis.size_cap.get("status") == "infeasible"
    ):
        diagnosis.size_cap = dict(diagnosis.size_cap)
        diagnosis.size_cap["acknowledge_size_risk"] = 1.0
        diagnosis.summary += "; size risk acknowledged"
    return diagnosis


def diagnosis_path(job_dir: Path) -> Path:
    """Return diagnosis.json path for a job."""
    return job_dir / DIAGNOSIS_FILE_NAME


def write_diagnosis(job_dir: Path, diagnosis: Diagnosis) -> Path:
    """Atomically write a diagnosis document."""
    path = diagnosis_path(job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(diagnosis.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path


def load_diagnosis(job_dir: Path) -> Diagnosis:
    """Load a diagnosis document."""
    path = diagnosis_path(job_dir)
    if not path.is_file():
        raise DiagnosisError(f"diagnosis not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise DiagnosisError("diagnosis root must be an object")
    return Diagnosis.from_dict(data)
