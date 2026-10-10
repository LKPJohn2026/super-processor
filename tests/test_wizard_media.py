"""Range streaming for output.mp4 and the 1080p / 30 minute input limit."""

from __future__ import annotations

import os
import shutil
import subprocess
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from super_processor.probe import MediaFacts, StreamFacts
from super_processor.review import WizardServer, parse_byte_range
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    input_limit_error,
    save_session_state,
)


def test_parse_byte_range() -> None:
    assert parse_byte_range(None, 100) is None
    assert parse_byte_range("bytes=0-9", 100) == (0, 9)
    assert parse_byte_range("bytes=90-", 100) == (90, 99)
    assert parse_byte_range("bytes=-10", 100) == (90, 99)
    assert parse_byte_range("bytes=-500", 100) == (0, 99)
    assert parse_byte_range("bytes=50-500", 100) == (50, 99)
    # Ignored, so the whole file is served.
    assert parse_byte_range("items=0-9", 100) is None
    assert parse_byte_range("bytes=0-9,20-29", 100) is None
    assert parse_byte_range("bytes=abc", 100) is None
    assert parse_byte_range("bytes=x-9", 100) is None
    for unsatisfiable in ("bytes=100-", "bytes=9-3", "bytes=-0"):
        with pytest.raises(ValueError):
            parse_byte_range(unsatisfiable, 100)


def _job_with_output(tmp_path: Path, payload: bytes) -> WizardController:
    controller = WizardController(tmp_path / "jobs")
    source = tmp_path / "src.mp4"
    source.write_bytes(b"x")
    manifest = controller.store.create(source)
    job_dir = controller.store.job_dir(manifest.job_id)
    (job_dir / "output.mp4").write_bytes(payload)
    (job_dir / "previews").mkdir()
    (job_dir / "previews" / "p.mp4").write_bytes(payload[:1000])
    save_session_state(
        controller.jobs_dir,
        WizardState(step=WizardStep.RESULT, job_id=manifest.job_id),
    )
    return controller


def _request(
    base: str, method: str, path: str, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, str], bytes]:
    parsed = urlparse(base)
    connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
    connection.request(method, path, headers=headers or {})
    response = connection.getresponse()
    body = response.read()
    return response.status, dict(response.getheaders()), body


def test_output_is_streamed_with_ranges(tmp_path: Path) -> None:
    payload = os.urandom(3 * 1024 * 1024 + 17)  # spans several stream chunks
    controller = _job_with_output(tmp_path, payload)
    server = WizardServer(controller.jobs_dir, controller=controller)
    base = server.start()
    try:
        status, headers, body = _request(base, "GET", "/output.mp4")
        assert status == 200
        assert body == payload
        assert headers["Accept-Ranges"] == "bytes"
        assert headers["Cache-Control"] == "no-store"

        status, headers, body = _request(
            base, "GET", "/output.mp4", {"Range": "bytes=1048570-1048590"}
        )
        assert status == 206
        assert body == payload[1048570:1048591]
        assert headers["Content-Range"] == f"bytes 1048570-1048590/{len(payload)}"
        assert headers["Content-Length"] == "21"

        status, headers, body = _request(
            base, "GET", "/output.mp4", {"Range": "bytes=-100"}
        )
        assert status == 206
        assert body == payload[-100:]

        status, headers, body = _request(
            base, "GET", "/output.mp4", {"Range": f"bytes={len(payload)}-"}
        )
        assert status == 416
        assert headers["Content-Range"] == f"bytes */{len(payload)}"

        status, headers, body = _request(base, "HEAD", "/output.mp4")
        assert status == 200
        assert body == b""
        assert headers["Content-Length"] == str(len(payload))

        # Previews belonged to the removed enhance step.
        status, _headers, _body = _request(base, "GET", "/previews/p.mp4")
        assert status == 404

        status, _headers, _body = _request(base, "GET", "/previews/../../src.mp4")
        assert status == 404
        status, _headers, _body = _request(base, "HEAD", "/missing.mp4")
        assert status == 404
    finally:
        server.stop()


def _facts(width: int, height: int, duration_s: float) -> MediaFacts:
    return MediaFacts(
        schema_version=1,
        source_path="clip.mp4",
        format_name="mov,mp4",
        format_long_name=None,
        duration_s=duration_s,
        size_bytes=1,
        bit_rate=None,
        streams=[StreamFacts(index=0, codec_type="video", width=width, height=height)],
        has_video=True,
        has_audio=False,
        is_vfr=False,
    )


def test_input_limit_error() -> None:
    assert input_limit_error(_facts(1920, 1080, 1800)) is None
    assert input_limit_error(_facts(1080, 1920, 600)) is None  # portrait
    assert input_limit_error(_facts(1280, 720, 60)) is None
    assert "1080p" in (input_limit_error(_facts(2560, 1440, 60)) or "")
    assert "1080p" in (input_limit_error(_facts(1920, 1200, 60)) or "")
    assert "1080p" in (input_limit_error(_facts(1200, 1920, 60)) or "")
    assert "30 minutes" in (input_limit_error(_facts(1920, 1080, 1801)) or "")
    assert "duration" in (input_limit_error(_facts(1920, 1080, 0)) or "")
    no_video = _facts(1920, 1080, 60)
    no_video.has_video = False
    no_video.streams = []
    assert input_limit_error(no_video) == "this file has no video stream"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_pick_refuses_a_file_over_1080p(tmp_path: Path) -> None:
    big = tmp_path / "big.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:size=2560x1440:rate=1:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(big),
        ],
        check=True,
    )
    controller = WizardController(tmp_path / "jobs")
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(big)]})
    state = controller.current_state()
    assert state.step is WizardStep.PICK_FILE
    assert state.error is not None and "2560x1440" in state.error
    assert state.job_id is None
    assert controller.store.list_jobs() == []


def _top_level_boxes(path: Path) -> list[str]:
    boxes: list[str] = []
    data = path.read_bytes()
    offset = 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        boxes.append(data[offset + 4 : offset + 8].decode("latin-1"))
        if size == 1:
            size = int.from_bytes(data[offset + 8 : offset + 16], "big")
        if size < 8:
            break
        offset += size
    return boxes


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg required")
def test_final_output_is_fast_start(tmp_path: Path) -> None:
    from super_processor.upscale import (
        FakeUpscaleEngine,
        apply_range_revise,
        default_span,
    )

    source = tmp_path / "in.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=160x90:rate=10:duration=3",
            "-f",
            "lavfi",
            "-i",
            "sine=duration=3",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "output.mp4"
    apply_range_revise(
        source,
        output,
        default_span(3),
        output,
        engine=FakeUpscaleEngine(),
        duration_s=3,
    )
    boxes = _top_level_boxes(output)
    assert boxes.index("moov") < boxes.index("mdat")
