"""Drive a job through the wizard's segment loop in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_session_state,
)


def segments(*spans: tuple[float, float, str, list[str]]) -> list[dict[str, Any]]:
    """A ``propose_segments`` reply: ``(start_s, end_s, label, contains)``."""
    return [
        {"start_s": start, "end_s": end, "label": label, "contains": contains}
        for start, end, label, contains in spans
    ]


def pick_and_segment(controller: WizardController, clip: Path) -> WizardState:
    """Pick ``clip`` and run sampling and proposing; stop on the segments."""
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(clip)]})
    controller.run_segmenting()
    state = controller.current_state()
    assert state.step is WizardStep.SEGMENTS, state.error
    return state


def analyze(controller: WizardController) -> WizardState:
    """Run the analyze worker on the current segment."""
    controller.run_analyze()
    return controller.current_state()


def approve_segments(controller: WizardController) -> WizardState:
    """Approve the segments and analyze the first one."""
    controller.handle_post("/segments", {"action": ["approve"]})
    return analyze(controller)


def preview(controller: WizardController) -> dict[str, Any]:
    """The segment preview the API shows now."""
    view = controller.api_view()["segment_preview"]
    assert isinstance(view, dict)
    return view


def approve_preview(controller: WizardController) -> WizardState:
    """Approve the current preview; analyze the next segment if there is one."""
    controller.handle_post("/preview", {"action": ["approve"]})
    state = controller.current_state()
    if state.step is WizardStep.ANALYZING:
        return analyze(controller)
    return state


def to_ready(controller: WizardController, clip: Path) -> Path:
    """Pick ``clip`` and approve everything up to the full render. The job dir."""
    job_id = pick_and_segment(controller, clip).job_id
    assert job_id
    state = approve_segments(controller)
    while state.step is WizardStep.PREVIEW_SEGMENT:
        state = approve_preview(controller)
    assert state.step is WizardStep.READY_TO_RENDER, state.error
    return controller.store.job_dir(job_id)
