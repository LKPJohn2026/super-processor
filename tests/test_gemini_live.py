"""Live Gemini API checks (skipped unless GEMINI_API_KEY is set).

CI maps the repository secret ``Gemini_API_Test`` to ``GEMINI_API_KEY``.
"""

from __future__ import annotations

import os
import struct
import zlib
from pathlib import Path

import pytest

from super_processor.gemini import GeminiClient, GeminiError

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not set (CI: secrets.Gemini_API_Test)",
)


def _tiny_png(path: Path) -> Path:
    """Write a valid 8×8 gray PNG without Pillow."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    width = height = 8
    raw = b"".join(b"\x00" + bytes([120]) * width for _ in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)
    return path


def test_live_gemini_propose_splits_structured(tmp_path: Path) -> None:
    """Call the real API and require a parseable split layout set."""
    frame = _tiny_png(tmp_path / "frame.png")
    client = GeminiClient()
    try:
        proposal = client.propose_splits(
            frame_paths=[frame],
            duration_s=30.0,
            job_dir=tmp_path,
        )
    except GeminiError as exc:
        pytest.fail(f"live Gemini split call failed: {exc}")
    assert proposal.layouts
    assert all(layout.segments for layout in proposal.layouts)
    assert all(
        segment.end_s > segment.start_s
        for layout in proposal.layouts
        for segment in layout.segments
    )


def test_live_gemini_propose_enhance_structured(tmp_path: Path) -> None:
    """Call the real API and require 3–5 allowlisted enhance options."""
    frame = _tiny_png(tmp_path / "frame.png")
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
    assert all(option.ops for option in result.options)
