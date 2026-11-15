"""Keep tests away from the user's own settings and saved keys, and the
delivery encoder fixed."""

from __future__ import annotations

import os
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
