"""Allowed H.264 encoders for the CLI export. ``libx264`` is the default.

Every output this project writes is H.264; HEVC and AV1 are not offered.
Decoders below only read sources, whatever codec they are in.
"""

from __future__ import annotations

SOFTWARE_ENCODER = "libx264"
HARDWARE_ENCODERS: tuple[str, ...] = (
    "h264_nvenc",
    "h264_qsv",
    "h264_amf",
    "h264_videotoolbox",
)
ALLOWED_ENCODERS: tuple[str, ...] = (SOFTWARE_ENCODER, *HARDWARE_ENCODERS)
# On 120 frames of a 1440x1080 film, libx264 "medium" at CRF 23 made
# 40.63 dB in 4.0 MB where the libx265 CRF 24 this replaced made 40.60 dB
# in 3.4 MB (x264 CRF 24: 39.86 dB).
DEFAULT_CRF = 23
DECODER_INPUT_ARGS: dict[str, tuple[str, ...]] = {
    "hevc_cuvid": ("-hwaccel", "cuda", "-c:v", "hevc_cuvid"),
    "hevc_qsv": ("-hwaccel", "qsv", "-c:v", "hevc_qsv"),
    "hevc_amf": ("-hwaccel", "amf", "-c:v", "hevc_amf"),
    "hevc_videotoolbox": (
        "-hwaccel",
        "videotoolbox",
    ),
}
HARDWARE_DECODERS: tuple[str, ...] = tuple(DECODER_INPUT_ARGS)


def decoder_input_args(decoder: str | None) -> list[str]:
    """Return input flags for a hardware decoder. ``None`` is software decode."""
    if decoder is None:
        return []
    try:
        return list(DECODER_INPUT_ARGS[decoder])
    except KeyError as exc:
        raise EncoderError(f"unsupported decoder {decoder}") from exc


class EncoderError(ValueError):
    """Raised when an encoder name is outside the allowlist."""


def encoder_rate_args(encoder: str, *, crf: int = DEFAULT_CRF) -> list[str]:
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
    if encoder == "h264_nvenc":
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
    if encoder == "h264_qsv":
        return [
            "-c:v",
            encoder,
            "-global_quality",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    if encoder == "h264_amf":
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
