"""Only real video files are worked on, never playlists or scripts.

FFmpeg picks how to read a file from what is in it, not from its name. A
".mp4" that is really an ffconcat script, or a .m3u8 playlist, would make
FFmpeg read other files on the disk as if they were the video, and the
result could then be sent back to whoever made the file. The wizard and the
CLI refuse any file ffprobe does not read as a real video container, and
every FFmpeg input of a file may only open local files.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import super_processor
from super_processor.cli import main
from super_processor.jobs import JobState, JobStore
from super_processor.probe import (
    MediaFacts,
    ProbeError,
    StreamFacts,
    UnsupportedContainerError,
    container_refusal,
    require_video_container,
    run_ffprobe_json,
)
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    input_limit_error,
    save_session_state,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe required",
)

REAL_CONTAINERS = [
    "mov,mp4,m4a,3gp,3g2,mj2",
    "matroska,webm",
    "avi",
    "mpegts",
    "mpeg",
    "mp4",
]
NOT_VIDEO_FILES = [
    "concat",
    "ffconcat",
    "hls",
    "applehttp",
    "dash",
    "sdp",
    "rtp",
    "tee",
    "lavfi",
    "data",
    "image2",
    "png_pipe",
    "srt",
    "tty",
    "ffmetadata",
    "mov,mp4,concat",  # a single name that is not a container is enough
    "",
    None,
]


@pytest.mark.parametrize("format_name", REAL_CONTAINERS)
def test_real_containers_are_allowed(format_name: str) -> None:
    assert container_refusal(format_name) is None


@pytest.mark.parametrize("format_name", NOT_VIDEO_FILES)
def test_playlists_scripts_and_streams_are_refused(format_name: str | None) -> None:
    refusal = container_refusal(format_name, name="video.mp4")
    assert refusal is not None
    assert refusal.startswith("video.mp4 is ")
    assert "choose a real MP4, MOV, MKV, WebM, AVI or MPEG-TS video" in refusal


@pytest.mark.parametrize("format_name", ["concat", "hls", "dash", "tee"])
def test_a_list_of_other_files_says_so(format_name: str) -> None:
    refusal = container_refusal(format_name) or ""
    assert "a playlist or a list of other files, not a video file" in refusal


def _facts(format_name: str | None) -> MediaFacts:
    return MediaFacts(
        schema_version=1,
        source_path="video.mp4",
        format_name=format_name,
        format_long_name=None,
        duration_s=1.0,
        size_bytes=60,
        bit_rate=None,
        streams=[StreamFacts(index=0, codec_type="video", width=320, height=240)],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )


def test_the_wizard_limits_refuse_a_concat_script_that_looks_like_a_video() -> None:
    assert input_limit_error(_facts("mov,mp4,m4a,3gp,3g2,mj2")) is None
    refusal = input_limit_error(_facts("concat"), name="video.mp4")
    assert refusal is not None and refusal.startswith("video.mp4 is a playlist")


def test_require_video_container_raises_a_probe_error() -> None:
    require_video_container(_facts("matroska,webm"))
    with pytest.raises(UnsupportedContainerError, match="hls") as raised:
        require_video_container(_facts("hls"), name="list.m3u8")
    assert isinstance(raised.value, ProbeError)


def test_ffprobe_may_only_open_local_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    seen: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "{}", "")

    monkeypatch.setattr("super_processor.probe.subprocess.run", run)
    run_ffprobe_json(source, ffprobe="/usr/bin/ffprobe")
    (command,) = seen
    assert command[-3:] == ["-protocol_whitelist", "file", str(source.resolve())]


def test_every_ffmpeg_file_input_may_only_open_local_files() -> None:
    """Each ``-i <file>`` in the package has ``*FILE_INPUT`` right before it."""
    package = Path(super_processor.__file__).parent
    file_input = re.compile(r'"-i",\s*str\(')
    missing: list[str] = []
    for module in sorted(package.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        for match in file_input.finditer(text):
            before = text[: match.start()].rstrip().rstrip(",").rstrip()
            if not before.endswith("*FILE_INPUT"):
                line = text.count("\n", 0, match.start()) + 1
                missing.append(f"{module.name}:{line}")
    assert missing == []


def _clip(path: Path, *, fmt: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
            "-pix_fmt", "yuv420p", *(fmt or []), str(path),
        ],
        check=True,
    )  # fmt: skip
    return path


def _concat_disguised_as_mp4(folder: Path) -> Path:
    """A 50-byte text file named .mp4 that makes FFmpeg read sub/private.mp4."""
    _clip(folder / "sub" / "private.mp4")
    bait = folder / "video.mp4"
    bait.write_text(
        "ffconcat version 1.0\nfile sub/private.mp4\nduration 1\n", encoding="utf-8"
    )
    return bait


def _playlist(folder: Path) -> Path:
    """A .m3u8 that names another local file by its full path."""
    segment = _clip(folder / "elsewhere" / "seg.ts", fmt=["-f", "mpegts"])
    playlist = folder / "list.m3u8"
    playlist.write_text(
        f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{segment}\n#EXT-X-ENDLIST\n",
        encoding="utf-8",
    )
    return playlist


def _pick(tmp_path: Path, path: Path) -> WizardController:
    controller = WizardController(tmp_path / "jobs")
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(path)]})
    return controller


@needs_ffmpeg
def test_the_wizard_refuses_a_concat_script_named_mp4(tmp_path: Path) -> None:
    controller = _pick(tmp_path, _concat_disguised_as_mp4(tmp_path / "download"))
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error is not None
    assert state.error.startswith("video.mp4 is a playlist or a list of other files")
    assert state.job_id is None
    assert controller.store.list_jobs() == []


@needs_ffmpeg
def test_the_wizard_refuses_an_m3u8_playlist(tmp_path: Path) -> None:
    controller = _pick(tmp_path, _playlist(tmp_path / "download"))
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error is not None
    if os.name == "posix":
        assert "is a playlist" in state.error
        assert "(FFmpeg reads it as hls)" in state.error
    else:  # FFmpeg may not read a drive-letter segment path at all
        assert "list.m3u8" in state.error
    assert controller.store.list_jobs() == []


@needs_ffmpeg
def test_a_refused_upload_is_removed(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    upload = controller.upload_target("video.mp4")
    _clip(upload.parent / "sub" / "private.mp4")
    upload.write_text(
        "ffconcat version 1.0\nfile sub/private.mp4\nduration 1\n", encoding="utf-8"
    )
    controller.pick_upload(upload, "video.mp4")
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error is not None and "not a video file" in state.error
    assert not upload.exists()


@needs_ffmpeg
def test_the_cli_refuses_a_concat_script_named_mp4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bait = _concat_disguised_as_mp4(tmp_path / "download")
    jobs = tmp_path / "jobs"
    assert main(["--jobs-dir", str(jobs), "job", "create", str(bait)]) == 0
    (job,) = JobStore(jobs).list_jobs()
    capsys.readouterr()
    assert main(["--jobs-dir", str(jobs), "job", "probe", job.job_id]) == 1
    printed = capsys.readouterr().out
    assert "error: video.mp4 is a playlist or a list of other files" in printed
    assert JobStore(jobs).load(job.job_id).state is JobState.IMPORTED
    assert not (jobs / job.job_id / "media_facts.json").exists()

    assert main(["--jobs-dir", str(jobs), "diagnose", str(bait)]) == 1
    assert "is a playlist or a list of other files" in capsys.readouterr().out
