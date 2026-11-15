"""A hardware decoder for sampling frames, and the software retry when the
device fails."""

from __future__ import annotations

from pathlib import Path

import pytest

from super_processor.cli import main
from super_processor.doctor import CheckResult
from super_processor.encoders import (
    HARDWARE_DECODERS,
    EncoderError,
    decoder_input_args,
)
from super_processor.estimators import EstimatorError, extract_gray_frame
from super_processor.jobs import JobState, JobStore

PIXELS = 160 * 90
# Written out, not read from encoders.DECODER_INPUT_ARGS, so a wrong table
# entry fails here instead of quietly turning hardware decode into software.
CUVID_FLAGS = ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]
JOB_ID = "abcd1234abcd1234"


class _Completed:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_hardware_decoder_flags_precede_the_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> _Completed:
        commands.append(cmd)
        # Bytes past one frame are cut off: the caller gets exactly one frame.
        return _Completed(0, bytes([8]) * (PIXELS + 3))

    monkeypatch.setattr("super_processor.estimators.subprocess.run", run)
    frame = extract_gray_frame(
        Path("clip.mp4"), at_s=1.0, decoder="hevc_cuvid", ffmpeg_bin="ffmpeg"
    )
    assert frame == bytes([8]) * PIXELS
    # One run: a working device is not asked twice.
    [command] = commands
    # FFmpeg applies -hwaccel and -c:v to the input that follows them.
    at = command.index("-hwaccel")
    assert command[at : at + 4] == CUVID_FLAGS
    assert at + 4 <= command.index("-i")


@pytest.mark.parametrize(
    ("decoder", "flags"),
    [
        ("hevc_cuvid", CUVID_FLAGS),
        ("hevc_qsv", ["-hwaccel", "qsv", "-c:v", "hevc_qsv"]),
        ("hevc_amf", ["-hwaccel", "amf", "-c:v", "hevc_amf"]),
        # VideoToolbox is given no -c:v; FFmpeg picks the codec for it.
        ("hevc_videotoolbox", ["-hwaccel", "videotoolbox"]),
    ],
)
def test_each_hardware_decoder_names_its_own_device_and_codec(
    decoder: str, flags: list[str]
) -> None:
    assert decoder_input_args(decoder) == flags


def test_every_offered_hardware_decoder_has_written_out_flags() -> None:
    # A decoder added to the table must be added to the case above too.
    assert set(HARDWARE_DECODERS) == {
        "hevc_cuvid",
        "hevc_qsv",
        "hevc_amf",
        "hevc_videotoolbox",
    }


def test_software_decode_adds_no_flags_and_an_unknown_decoder_is_refused() -> None:
    assert decoder_input_args(None) == []
    with pytest.raises(EncoderError, match="unsupported decoder magic_cuvid"):
        decoder_input_args("magic_cuvid")


@pytest.mark.parametrize(
    "failure",
    [
        _Completed(1, b"", b"no device"),
        _Completed(0, bytes([20]) * (PIXELS // 2)),  # a short, partial frame
        OSError("device busy"),
    ],
    ids=["exit-code", "short-frame", "oserror"],
)
def test_a_failed_device_samples_the_same_frame_in_software(
    monkeypatch: pytest.MonkeyPatch, failure: _Completed | OSError
) -> None:
    gray = bytes([20]) * PIXELS
    commands: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> _Completed:
        commands.append(cmd)
        if "-hwaccel" not in cmd:
            return _Completed(0, gray)
        if isinstance(failure, OSError):
            raise failure
        return failure

    monkeypatch.setattr("super_processor.estimators.subprocess.run", run)
    frame = extract_gray_frame(
        Path("clip.mp4"), at_s=1.0, decoder="hevc_cuvid", ffmpeg_bin="ffmpeg"
    )
    assert frame == gray
    assert len(commands) == 2
    # The retry is the same request with only the device flags left out, so
    # the sample is of the same moment at the same size.
    at = commands[0].index("-hwaccel")
    assert commands[0][at : at + 4] == CUVID_FLAGS
    del commands[0][at : at + 4]
    assert commands[0] == commands[1]


def test_a_frame_neither_decoder_can_read_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def run(cmd: list[str], **_kwargs: object) -> _Completed:
        commands.append(cmd)
        return _Completed(1, b"", b"moov atom not found")

    monkeypatch.setattr("super_processor.estimators.subprocess.run", run)
    with pytest.raises(EstimatorError, match="moov atom not found"):
        extract_gray_frame(
            Path("clip.mp4"), at_s=1.0, decoder="hevc_cuvid", ffmpeg_bin="ffmpeg"
        )
    assert ["-hwaccel" in command for command in commands] == [True, False]


def test_missing_decoder_fails_before_sampling(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake-video")
    store = JobStore(tmp_path / "jobs")
    store.create(source, job_id=JOB_ID)
    store.transition(JOB_ID, JobState.PROBED)

    def missing(*_args: object, **_kwargs: object) -> CheckResult:
        return CheckResult(
            name="decoder:hevc_cuvid",
            ok=False,
            detail=(
                "hevc_cuvid missing "
                "(install ffmpeg with this decoder or sample in software)"
            ),
        )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("sampler ran")

    monkeypatch.setattr("super_processor.estimators.check_ffmpeg_capability", missing)
    monkeypatch.setattr("super_processor.segments.sample_media", boom)
    code = main(
        ["--jobs-dir", str(store.root), "segment", JOB_ID, "--decode", "hevc_cuvid"]
    )
    assert code == 1
    assert "hevc_cuvid missing" in capsys.readouterr().out
