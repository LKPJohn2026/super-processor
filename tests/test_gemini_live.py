"""Live Gemini API checks using repository secret ``Gemini_API_Test``.

CI maps ``secrets.Gemini_API_Test`` → ``GEMINI_API_KEY`` in the
``gemini-live`` job only. Unit-test matrix jobs never receive the key.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import pytest

from super_processor.gemini import (
    GeminiClient,
    GeminiError,
    load_chat,
    model_accepts_text_or_video,
    model_has_usable_rpm,
)

pytestmark = [
    pytest.mark.gemini_live,
    pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set (CI: secrets.Gemini_API_Test)",
    ),
]


def _tiny_png(path: Path, *, shade: int = 120) -> Path:
    """Write a valid 32×32 gray PNG without Pillow."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    width = height = 32
    level = max(0, min(255, shade))
    raw = b"".join(b"\x00" + bytes([level]) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


def test_live_gemini_lists_multimodal_rpm_candidates() -> None:
    """Discover generateContent models that accept text/video with RPM != 0."""
    client = GeminiClient()
    infos = client.list_model_infos()
    assert infos, "models.list returned no models"
    usable = [
        info
        for info in infos
        if model_accepts_text_or_video(info) and model_has_usable_rpm(info)
    ]
    assert usable, "no multimodal models with usable RPM (limit != 0)"
    candidates = client.candidate_models(refresh=True)
    assert candidates


def test_live_gemini_propose_splits_structured(tmp_path: Path) -> None:
    """Call the real API and require a parseable split layout set."""
    frames = [
        _tiny_png(tmp_path / "frame_a.png", shade=40),
        _tiny_png(tmp_path / "frame_b.png", shade=200),
    ]
    client = GeminiClient()
    try:
        # Fail over across candidates when preferred model is 404/429.
        if not client.candidate_models():
            client.pick_working_model()
        proposal = client.propose_splits(
            frame_paths=frames,
            duration_s=30.0,
            job_dir=tmp_path,
        )
    except GeminiError as exc:
        pytest.fail(f"live Gemini split call failed: {exc}")
    assert proposal.layouts
    assert 1 <= len(proposal.layouts) <= 5
    for layout in proposal.layouts:
        assert layout.segments
        assert len(layout.segments) <= 20
        for segment in layout.segments:
            assert segment.end_s - segment.start_s >= 5.0 - 1e-6
            assert 0.0 <= segment.start_s <= 30.5
            assert segment.end_s <= 30.5
    assert load_chat(tmp_path)


def test_live_gemini_propose_enhance_structured(tmp_path: Path) -> None:
    """Call the real API and require 3–5 allowlisted enhance options."""
    frame = _tiny_png(tmp_path / "frame.png", shade=35)
    client = GeminiClient()
    try:
        result = client.propose_enhance(
            frame_paths=[frame],
            segment_label="low_light",
            start_s=0.0,
            end_s=20.0,
            job_dir=tmp_path,
        )
    except GeminiError as exc:
        pytest.fail(f"live Gemini enhance call failed: {exc}")
    assert 3 <= len(result.options) <= 5
    allowed = {
        "white_balance",
        "contrast",
        "denoise",
        "sharpen",
        "stabilize",
        "trim",
    }
    for option in result.options:
        assert option.ops
        for op in option.ops:
            assert op.op in allowed, f"disallowed op {op.op!r}"


def test_live_gemini_split_multi_turn_revise(tmp_path: Path) -> None:
    """Free-text revise must return another structured layout set."""
    frames = [_tiny_png(tmp_path / "frame.png", shade=90)]
    client = GeminiClient()
    try:
        first = client.propose_splits(
            frame_paths=frames,
            duration_s=40.0,
            job_dir=tmp_path,
        )
        revised = client.propose_splits(
            frame_paths=frames,
            duration_s=40.0,
            user_text=(
                "Please use exactly 3 segments if possible, "
                "and label one of them as a darker indoor stretch."
            ),
            job_dir=tmp_path,
        )
    except GeminiError as exc:
        pytest.fail(f"live Gemini multi-turn split failed: {exc}")
    assert first.layouts
    assert revised.layouts
    assert any(layout.segments for layout in revised.layouts)
    assert len(load_chat(tmp_path)) >= 4
