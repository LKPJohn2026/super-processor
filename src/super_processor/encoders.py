"""Allowed HEVC encoders. ``libx265`` stays the default and the quality reference."""

from __future__ import annotations

SOFTWARE_ENCODER = "libx265"
HARDWARE_ENCODERS: tuple[str, ...] = (
    "hevc_nvenc",
    "hevc_qsv",
    "hevc_amf",
    "hevc_videotoolbox",
)
ALLOWED_ENCODERS: tuple[str, ...] = (SOFTWARE_ENCODER, *HARDWARE_ENCODERS)
