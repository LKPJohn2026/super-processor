"""Drive WizardController.render for each persisted step."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from super_processor.gemini import GeminiError
from super_processor.jobs import JobState
from super_processor.review import WizardServer
from super_processor.upscale import FakeUpscaleEngine, UpscaleError
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
        (WizardStep.FINDING_SHOTS, "Finding the shots"),
        (WizardStep.SHOTS, "Approve shots"),
        (WizardStep.PLANNING, "Planning each shot"),
        (WizardStep.LOOKS, "Approve all and render"),
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


class _BrokenEngine:
    def upscale(self, request: object) -> Path:
        raise UpscaleError("FlashVSR needs an NVIDIA GPU with CUDA")


class _NoGemini:
    """Every Gemini call fails, so each loop falls back to the measurements."""

    def label_shots(self, **_kwargs: object) -> list[dict[str, object]]:
        raise GeminiError("no key in tests")

    def propose_looks(self, **_kwargs: object) -> dict[str, object]:
        raise GeminiError("no key in tests")

    def check_previews(self, **_kwargs: object) -> list[dict[str, object]]:
        raise GeminiError("no key in tests")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_failed_first_upscale_returns_to_pick(tmp_path: Path) -> None:
    clip = tmp_path / "real.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:size=160x90:rate=10:duration=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    controller = WizardController(
        tmp_path / "jobs",
        gemini=_NoGemini(),  # type: ignore[arg-type]
        upscale_engine=_BrokenEngine(),
    )
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(clip)]})
    job_id = controller.current_state().job_id
    assert job_id is not None
    assert controller.current_state().step is WizardStep.FINDING_SHOTS
    controller.run_scan()
    assert controller.current_state().step is WizardStep.SHOTS
    controller.handle_post("/shots", {"action": ["approve"]})
    assert controller.current_state().step is WizardStep.PLANNING
    # Without a GPU the previews fail, and the editor is back on the shots.
    controller.run_plan()
    state = controller.current_state()
    assert state.step is WizardStep.SHOTS
    assert state.error is not None and "previews failed" in state.error
    assert "CUDA" in controller.render()
    # With previews working, approve the looks; then the full render fails.
    controller._upscale = FakeUpscaleEngine()
    controller.handle_post("/shots", {"action": ["approve"]})
    controller.run_plan()
    assert controller.current_state().step is WizardStep.LOOKS
    assert "could not propose" in controller.render()
    controller.handle_post("/looks", {"action": ["approve"]})
    assert controller.current_state().step is WizardStep.RENDERING
    controller._upscale = _BrokenEngine()
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.job_id is None
    assert state.error is not None
    assert "no result yet" in state.error and "CUDA" in state.error
    assert controller.store.load(job_id).state is JobState.FAILED
    page = controller.render()
    assert "Pick" in page and "CUDA" in page
    # Picking again starts a fresh job.
    controller.handle_post("/pick", {"path": [str(clip)]})
    assert controller.current_state().job_id not in {None, job_id}


def test_start_new_job_from_done_and_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    (job_dir / "output.mp4").write_bytes(b"result")
    for step in (WizardStep.DONE, WizardStep.RESULT):
        state = WizardState(step=step, job_id=job_id)
        save_job_wizard_state(job_dir, state)
        save_session_state(tmp_path, state)
        assert 'action="/new"' in controller.render()
        monkeypatch.setattr(
            "super_processor.wizard.resolve_gemini_api_key", lambda: "key"
        )
        controller.handle_post("/new", {})
        fresh = controller.current_state()
        assert fresh.step is WizardStep.PICK_FILE
        assert fresh.job_id is None
        assert (job_dir / "output.mp4").read_bytes() == b"result"
    monkeypatch.setattr("super_processor.wizard.resolve_gemini_api_key", lambda: None)
    controller.handle_post("/new", {})
    assert controller.current_state().step is WizardStep.SETUP


def test_start_new_job_over_the_api(tmp_path: Path) -> None:
    controller, job_id, job_dir = _seed_job(tmp_path)
    save_session_state(tmp_path, WizardState(step=WizardStep.DONE, job_id=job_id))
    server = WizardServer(tmp_path, controller=controller)
    base = server.start()
    try:
        request = Request(
            base + "/api/new",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request) as response:
            view = json.loads(response.read().decode())
        assert view["step"] in {"pick_file", "setup"}
        assert view["job_id"] is None
    finally:
        server.stop()
