"""Runtime QA checks for preview outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import mean_luma, percentile_u8, variance_u8
from .estimators import extract_gray_frame, extract_rgb_means

QA_FILE_NAME = "qa_report.json"
QA_SCHEMA_VERSION = 1


@dataclass(slots=True)
class QAIssue:
    """One runtime quality finding."""

    code: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(slots=True)
class QAReport:
    """Preview QA summary."""

    schema_version: int
    ok: bool
    issues: list[QAIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }


def analyze_preview(
    preview_path: Path,
    *,
    source_path: Path | None = None,
) -> QAReport:
    """Heuristic QA on a preview file (plastic denoise / crop / flicker proxies)."""
    issues: list[QAIssue] = []
    metrics: dict[str, float] = {}
    if not preview_path.is_file():
        return QAReport(
            schema_version=QA_SCHEMA_VERSION,
            ok=False,
            issues=[
                QAIssue(
                    code="missing_preview",
                    severity="error",
                    message=f"preview not found: {preview_path}",
                )
            ],
        )

    try:
        frame = extract_gray_frame(preview_path, at_s=0.05)
        var = variance_u8(frame)
        p05 = percentile_u8(frame, 5.0)
        p95 = percentile_u8(frame, 95.0)
        metrics.update(
            {"preview_variance": var, "preview_p05": p05, "preview_p95": p95}
        )
        # Extremely flat luma span can indicate plastic over-denoise.
        if (p95 - p05) < 25.0 and var < 40.0:
            issues.append(
                QAIssue(
                    code="plastic_denoise",
                    severity="warning",
                    message=(
                        "preview luma range is unusually flat; "
                        "denoise may be too strong"
                    ),
                )
            )
        frame_b = extract_gray_frame(preview_path, at_s=0.2)
        flicker = abs(mean_luma(frame) - mean_luma(frame_b))
        metrics["flicker_delta"] = flicker
        if flicker > 18.0:
            issues.append(
                QAIssue(
                    code="flicker",
                    severity="warning",
                    message=f"preview luma flicker delta={flicker:.1f}",
                )
            )
        # Edge crop proxy: compare border vs center mean.
        width, height = 160, 90
        # frame is width*height gray from extract_gray_frame defaults
        center = frame[(height // 4) * width : (3 * height // 4) * width]
        if center:
            metrics["center_mean"] = mean_luma(center)
        r, g, b = extract_rgb_means(preview_path, at_s=0.05)
        metrics.update({"preview_r": r, "preview_g": g, "preview_b": b})
        # Text-damage proxy: very high local contrast saturation not computed;
        # flag only when RGB collapses (posterization / crushing).
        if max(r, g, b) - min(r, g, b) < 2.0 and var < 30.0:
            issues.append(
                QAIssue(
                    code="text_readability",
                    severity="warning",
                    message="preview color collapsed; burned-in text may be damaged",
                )
            )
    except Exception as exc:
        issues.append(
            QAIssue(
                code="qa_failed",
                severity="error",
                message=f"preview QA failed: {exc}",
            )
        )

    if source_path is not None and source_path.is_file():
        metrics["source_exists"] = 1.0

    errors = [issue for issue in issues if issue.severity == "error"]
    return QAReport(
        schema_version=QA_SCHEMA_VERSION,
        ok=not errors,
        issues=issues,
        metrics=metrics,
    )


def write_qa_report(job_dir: Path, report: QAReport) -> Path:
    """Atomically write a QA report."""
    path = job_dir / QA_FILE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    return path
