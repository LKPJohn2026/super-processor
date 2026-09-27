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


class EncoderError(ValueError):
    """Raised when an encoder name is outside the allowlist."""


def encoder_rate_args(encoder: str, *, crf: int = 24) -> list[str]:
    """Return rate-control flags for one encoder.

    The filter graph and the trim stay outside this list. Each encoder uses
    its own quality flag. VideoToolbox quality runs from 1 to 100, with a
    higher number closer to a low CRF.
    """
    if encoder not in ALLOWED_ENCODERS:
        raise EncoderError(f"unsupported encoder {encoder}")
    if encoder == SOFTWARE_ENCODER:
        return [
            "-c:v",
            encoder,
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    if encoder == "hevc_nvenc":
        return [
            "-c:v",
            encoder,
            "-rc",
            "constqp",
            "-qp",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    if encoder == "hevc_qsv":
        return [
            "-c:v",
            encoder,
            "-global_quality",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    if encoder == "hevc_amf":
        return [
            "-c:v",
            encoder,
            "-rc",
            "cqp",
            "-qp_i",
            str(crf),
            "-qp_p",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    quality = max(1, min(100, 100 - crf))
    return ["-c:v", encoder, "-q:v", str(quality), "-pix_fmt", "yuv420p"]
