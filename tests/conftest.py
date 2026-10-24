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
    empty = types.SimpleNamespace(
        get_password=lambda service, name: None,
        set_password=lambda service, name, value: None,
    )
    monkeypatch.setitem(sys.modules, "keyring", empty)
    return folder
