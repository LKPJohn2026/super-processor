"""The Gemini key saved by the setup screen, and where it is looked up."""

from __future__ import annotations

import json
import os
import stat
import sys
import types
from pathlib import Path

import pytest

from super_processor.gemini import (
    GeminiClient,
    GeminiError,
    config_dir,
    credentials_path,
    gemini_api_key_source,
    resolve_gemini_api_key,
    store_gemini_api_key,
)
from super_processor.wizard import WizardController, WizardState, WizardStep


def test_key_is_saved_to_a_private_file(
    _own_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    path = store_gemini_api_key("  AIza-real-key  ")
    assert path == _own_config_dir / "credentials.json" == credentials_path()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "gemini_api_key": "AIza-real-key"
    }
    if sys.platform != "win32":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert [p.name for p in path.parent.iterdir()] == ["credentials.json"]

    # A later run has no environment variable; the file still has the key.
    monkeypatch.delenv("GEMINI_API_KEY")
    assert resolve_gemini_api_key() == "AIza-real-key"
    assert gemini_api_key_source() == str(path)
    store_gemini_api_key("AIza-new-key")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert resolve_gemini_api_key() == "AIza-new-key"


def test_lookup_order_is_env_then_file_then_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    old = types.SimpleNamespace(get_password=lambda service, name: "from-keyring")
    monkeypatch.setitem(sys.modules, "keyring", old)
    assert resolve_gemini_api_key() == "from-keyring"
    assert gemini_api_key_source() == "the system keyring (saved by an older version)"
    store_gemini_api_key("from-file")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert resolve_gemini_api_key() == "from-file"
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert resolve_gemini_api_key() == "from-env"
    assert gemini_api_key_source() == "the GEMINI_API_KEY environment variable"


def test_no_key_anywhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert resolve_gemini_api_key() is None
    assert gemini_api_key_source() is None


@pytest.mark.parametrize(
    "content", ["{", "[]", '{"gemini_api_key": 5}', '{"gemini_api_key": "  "}']
)
def test_unusable_saved_file_is_ignored(
    _own_config_dir: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (_own_config_dir / "credentials.json").write_text(content, encoding="utf-8")
    assert resolve_gemini_api_key() is None


def test_empty_or_unsavable_key_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(GeminiError, match="empty"):
        store_gemini_api_key("   ")
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setenv("SUPER_PROCESSOR_CONFIG_DIR", str(blocker / "sub"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiError, match="could not save the API key"):
        store_gemini_api_key("AIza-key")
    assert "GEMINI_API_KEY" not in os.environ


def test_a_failed_save_keeps_the_old_key_and_no_half_file(
    _own_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    path = store_gemini_api_key("AIza-old-key")
    monkeypatch.delenv("GEMINI_API_KEY")

    def full_disk(_src: object, _dst: object) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr("super_processor.gemini.os.replace", full_disk)
    with pytest.raises(GeminiError, match="No space left"):
        store_gemini_api_key("AIza-new-key")
    # The saved key is untouched, the temporary file is gone, and this
    # process does not start using a key that was never saved.
    assert [p.name for p in _own_config_dir.iterdir()] == ["credentials.json"]
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "gemini_api_key": "AIza-old-key"
    }
    assert "GEMINI_API_KEY" not in os.environ


def test_the_client_uses_the_saved_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(GeminiError, match="paste a key"):
        GeminiClient()
    store_gemini_api_key("AIza-saved")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert GeminiClient().api_key == "AIza-saved"
    monkeypatch.setenv("GEMINI_API_KEY", " AIza-env ")
    assert GeminiClient().api_key == "AIza-env"


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("linux", {"XDG_CONFIG_HOME": "/xdg"}, Path("/xdg/super-processor")),
        ("linux", {}, Path.home() / ".config" / "super-processor"),
        (
            "darwin",
            {},
            Path.home() / "Library" / "Application Support" / "super-processor",
        ),
        ("win32", {"APPDATA": "/appdata"}, Path("/appdata/super-processor")),
        (
            "win32",
            {},
            Path.home() / "AppData" / "Roaming" / "super-processor",
        ),
    ],
)
def test_config_dir_follows_the_platform(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    env: dict[str, str],
    expected: Path,
) -> None:
    monkeypatch.delenv("SUPER_PROCESSOR_CONFIG_DIR")
    for name in ("XDG_CONFIG_HOME", "APPDATA"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", platform)
    assert config_dir() == expected


def test_setup_screen_saves_the_key_for_later_runs(
    tmp_path: Path, _own_config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    controller = WizardController(tmp_path / "jobs")
    from super_processor.wizard import save_session_state

    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.SETUP))
    view = controller.api_view()
    assert view["step"] == "setup" and view["has_gemini_key"] is False
    controller.handle_post("/setup", {"api_key": ["AIza-pasted"]})
    assert controller.current_state().step is WizardStep.PICK_FILE

    # A new server process: no environment variable, same saved file.
    monkeypatch.delenv("GEMINI_API_KEY")
    restarted = WizardController(tmp_path / "jobs")
    save_session_state(restarted.jobs_dir, WizardState(step=WizardStep.SETUP))
    assert restarted.api_view()["has_gemini_key"] is True
    # Where the saved key comes from (the old setup page named it).
    source = gemini_api_key_source()
    assert source is not None
    assert str(_own_config_dir / "credentials.json") in source
