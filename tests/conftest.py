"""Keep tests away from the user's own settings and saved keys, and the
delivery encoder fixed."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

# Windows keeps the current test id in an environment variable capped at
# 32,767 characters, so a parameter that is a huge string or a deep list
# must not become the id itself.
_MAX_ID_LEN = 60


def short_id(value: object) -> str:
    """A short, readable test id for any parameter value."""
    try:
        text = repr(value)
    except RecursionError:
        return f"{type(value).__name__}-deep"
    if len(text) <= _MAX_ID_LEN:
        return text
    return f"{text[:40]}...len{len(text)}"


def pytest_make_parametrize_id(
    config: pytest.Config, val: object, argname: str
) -> str | None:
    """Shorten only long ids; short values keep pytest's own id."""
    text = short_id(val)
    return text if len(text) < len(_safe_repr(val)) else None


def _safe_repr(value: object) -> str:
    try:
        return repr(value)
    except RecursionError:
        return "x" * (_MAX_ID_LEN + 1)


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
    # An exported weights path would send the SPAN tests to the user's file.
    monkeypatch.delenv("SUPER_PROCESSOR_SPAN_WEIGHTS", raising=False)
    empty = types.SimpleNamespace(
        get_password=lambda service, name: None,
        set_password=lambda service, name, value: None,
    )
    monkeypatch.setitem(sys.modules, "keyring", empty)
    return folder


# Set to h264_nvenc to run the whole suite against NVENC (and NVDEC, where
# FFmpeg can open CUDA) instead of libx264 and FFmpeg's own decoders.
TEST_ENCODER_ENV = "SUPER_PROCESSOR_TEST_ENCODER"


@pytest.fixture(autouse=True)
def _fixed_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """libx264 and software decoding everywhere, unless asked otherwise.

    The delivery encoder depends on the machine (NVENC where it works);
    tests that compare frames or markers must not.
    """
    from super_processor import upscale

    nvenc = os.environ.get(TEST_ENCODER_ENV, "").strip() == upscale.NVENC
    monkeypatch.setattr(upscale, "nvenc_works", lambda ffmpeg_bin="ffmpeg": nvenc)
    if not nvenc:
        monkeypatch.setattr(
            upscale, "cuda_decode_works", lambda ffmpeg_bin="ffmpeg": False
        )


@pytest.fixture
def web_app(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Serve a small stand-in web app instead of the built one (if any)."""
    from wizard_client import fake_web_app

    from super_processor import review

    root = fake_web_app(tmp_path_factory.mktemp("web_dist"))
    monkeypatch.setattr(review, "WEB_DIST_DIR", root)
    return root


@pytest.fixture
def no_web_app(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Act like a dev checkout that never ran ``npm run build:package``."""
    from super_processor import review

    root = tmp_path_factory.mktemp("no_web_dist") / "web_dist"
    monkeypatch.setattr(review, "WEB_DIST_DIR", root)
    return root
