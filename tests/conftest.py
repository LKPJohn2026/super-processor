"""Keep tests away from the user's own settings and saved keys."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _own_config_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Every test gets an empty config folder and an empty keyring.

    A saved key then never leaks into a test, and a test that stores a key
    never replaces the editor's real one (an earlier test overwrote the
    keyring entry this way). Tests that need a keyring install their own.
    """
    folder = tmp_path_factory.mktemp("config")
    monkeypatch.setenv("SUPER_PROCESSOR_CONFIG_DIR", str(folder))
    # An exported FlashVSR+ choice would send the v1.1 tests to the real model.
    for name in ("SUPER_PROCESSOR_FLASHVSR", "FLASHVSR_PLUS_HOME"):
        monkeypatch.delenv(name, raising=False)
    empty = types.SimpleNamespace(
        get_password=lambda service, name: None,
        set_password=lambda service, name, value: None,
    )
    monkeypatch.setitem(sys.modules, "keyring", empty)
    return folder


@pytest.fixture(autouse=True)
def _no_managed_memory_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clip that runs out of memory never starts a real FlashVSR child.

    Tests that check the fallback replace this with their own.
    """
    from super_processor import managed_memory
    from super_processor.upscale import UpscaleError

    def refuse(request: object, **_kwargs: object) -> Path:
        raise UpscaleError("the system-RAM fallback is off in tests")

    monkeypatch.setattr(managed_memory, "upscale_in_managed_memory", refuse)
