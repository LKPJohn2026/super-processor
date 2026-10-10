"""Drive WizardController.render for each persisted step."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    job_state_path,
    load_job_wizard_state,
    save_job_wizard_state,
    save_session_state,
    session_state_path,
)


def _seed_job(tmp_path: Path) -> tuple[WizardController, str, Path]:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-a-real-video")
    controller = WizardController(tmp_path)
    manifest = controller.store.create(source)
    return controller, manifest.job_id, controller.store.job_dir(manifest.job_id)


def test_render_each_wizard_step(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    steps = [
        (WizardStep.INTRO, "Super Processor"),
        (WizardStep.LLM_CHOICE, "local LLM"),
        (WizardStep.LOCAL_LLM_STUB, "Local LLM"),
        (WizardStep.SETUP, "Gemini"),
        (WizardStep.PICK_FILE, "Pick"),
        (WizardStep.RENDERING, "Upscaling"),
        (WizardStep.RESULT, "Result"),
        (WizardStep.DONE, "Done"),
    ]
    assert {step for step, _needle in steps} == set(WizardStep)
    for step, needle in steps:
        state = WizardState(step=step, job_id=job_id)
        save_job_wizard_state(job_dir, state)
        save_session_state(tmp_path, state)
        assert needle in controller.render()


def test_retired_steps_go_back_to_pick(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    for retired in ("analyzing", "overview", "choose_split", "enhance"):
        old: dict[str, object] = {
            "schema_version": 1,
            "step": retired,
            "job_id": job_id,
            "layout_index": 0,
            "segment_index": 1,
            "highlights": ["h"],
            "enhance_cache": {"0": {}},
            "error": None,
        }
        session_state_path(tmp_path).write_text(json.dumps(old), encoding="utf-8")
        job_state_path(job_dir).write_text(json.dumps(old), encoding="utf-8")
        state = controller.current_state()
        assert state.step is WizardStep.PICK_FILE
        assert state.job_id is None
        assert state.error is not None and "pick the file again" in state.error
        assert "Pick" in controller.render()
    loaded = load_job_wizard_state(job_dir)
    assert loaded is not None
    assert loaded.to_dict() == {
        "schema_version": 2,
        "step": "pick_file",
        "job_id": None,
        "error": loaded.error,
    }


def test_wizard_server_serves_output_only(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    (job_dir / "output.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    (job_dir / "segment_stills").mkdir()
    (job_dir / "segment_stills" / "seg_00.ppm").write_bytes(b"P5")
    state = WizardState(step=WizardStep.RESULT, job_id=job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(tmp_path, state)
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        with urlopen(base + "/output.mp4") as response:
            assert response.read()[:4] == b"\x00\x00\x00\x18"
        for retired in ("/segment_stills/seg_00.ppm", "/segment_stills/seg_00.bmp"):
            try:
                urlopen(base + retired)
            except HTTPError as exc:
                assert exc.code == 404
                exc.close()
            else:
                raise AssertionError(f"{retired} should be gone")
    finally:
        server.stop()
