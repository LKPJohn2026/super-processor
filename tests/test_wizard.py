"""The wizard's opening steps: intro, LLM choice, Gemini setup, and the pick."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from wizard_client import FAKE_INDEX, get

from super_processor.gemini import credentials_path, resolve_gemini_api_key
from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    load_session_state,
    save_session_state,
)


def test_wizard_server_serves_the_app_on_the_intro_step(
    tmp_path: Path, web_app: Path
) -> None:
    server = WizardServer(tmp_path)
    base = server.start()
    try:
        with get(base + "/", server.token) as response:
            body = response.read().decode()
        with get(base + "/api/state", server.token) as response:
            view = json.loads(response.read())
    finally:
        server.stop()
    assert body == FAKE_INDEX
    assert view["step"] == "intro"


def test_setup_saves_the_key_and_moves_on_to_the_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real key store writes into the test's own config folder (conftest),
    # so the test proves the key is kept for later runs, not that a stub ran.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = WizardController(tmp_path)
    assert controller.current_state().step is WizardStep.INTRO
    controller.handle_post("/intro", {})
    assert controller.current_state().step is WizardStep.LLM_CHOICE
    controller.handle_post("/llm", {"choice": ["local"]})
    assert controller.current_state().step is WizardStep.LOCAL_LLM_STUB
    view = controller.api_view()
    assert view["step"] == "local_llm_stub" and view["error"] is None
    controller.handle_post("/local-llm", {})
    assert controller.current_state().step is WizardStep.SETUP
    # Without a key, "skip" cannot skip: it falls through to saving an empty
    # key, which is refused and keeps the editor on setup.
    controller.handle_post("/setup", {"skip": ["1"]})
    state = controller.current_state()
    assert state.step is WizardStep.SETUP
    assert state.error is not None and "empty" in state.error
    controller.handle_post("/setup", {"api_key": ["  unit-test-key  "]})
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE and state.error is None
    assert load_session_state(tmp_path).step is WizardStep.PICK_FILE
    assert credentials_path().is_file()
    assert resolve_gemini_api_key() == "unit-test-key"


def test_setup_can_be_skipped_once_a_key_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "already-set")
    controller = WizardController(tmp_path)
    save_session_state(tmp_path, WizardState(step=WizardStep.SETUP))
    controller.handle_post("/setup", {"skip": ["1"]})
    assert controller.current_state().step is WizardStep.PICK_FILE
    # Skipping must not overwrite the key in use with an empty one.
    assert not credentials_path().exists()
    assert resolve_gemini_api_key() == "already-set"


@pytest.mark.parametrize(
    ("choice", "step"),
    [
        ("gemini", WizardStep.SETUP),
        ("local", WizardStep.LOCAL_LLM_STUB),
        ("", WizardStep.LLM_CHOICE),
        ("openai", WizardStep.LLM_CHOICE),
    ],
)
def test_llm_choice_leads_to_its_step(
    tmp_path: Path, choice: str, step: WizardStep
) -> None:
    controller = WizardController(tmp_path)
    controller.handle_post("/intro", {})
    controller.handle_post("/llm", {"choice": [choice]})
    state = controller.current_state()
    assert state.step is step
    # Only an unknown choice is an error; it asks again.
    assert (state.error is not None) is (step is WizardStep.LLM_CHOICE)
