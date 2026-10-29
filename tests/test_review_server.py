"""Tests for the localhost segment review route."""

from __future__ import annotations

import json
import shutil
import subprocess
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pytest

from super_processor.jobs import JobState, JobStore
from super_processor.review import ReviewServer
from super_processor.segments import (
    SampleRow,
    TimelineSegment,
    load_segments,
    write_gray_still,
    write_samples,
    write_segments,
)
from super_processor.wizard import (
    WizardController,
    WizardError,
    WizardState,
    WizardStep,
    load_job_wizard_state,
    save_job_wizard_state,
    save_session_state,
)

PIXELS = 160 * 90


def _job(tmp_path: Path) -> Path:
    job_dir = tmp_path / "job"
    still = "segment_stills/seg_00.ppm"
    write_gray_still(job_dir / still, bytes([40]) * PIXELS)
    write_segments(
        job_dir,
        [TimelineSegment(0, 0.0, 60.0, "indoor", "noisy", 10.0, 0, still)],
    )
    return job_dir


def test_segments_route_returns_the_list_and_the_still(tmp_path: Path) -> None:
    server = ReviewServer(_job(tmp_path))
    base = server.start()
    try:
        with urlopen(f"{base}/segments") as response:
            payload = json.load(response)
        assert payload["segments"][0]["problem"] == "noisy"
        still_url = payload["segments"][0]["still_url"]
        assert still_url == "/segment_stills/seg_00.ppm"
        with urlopen(f"{base}{still_url}") as response:
            body = response.read()
        assert body.startswith(b"P5\n")
        parsed = urlparse(base)
        connection = HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port)
        connection.request("GET", "/segment_stills/../segments.json")
        missing = connection.getresponse().status
    finally:
        server.stop()
    assert missing == 404


def _proposed_job(tmp_path: Path) -> tuple[JobStore, str]:
    source = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:duration=20:size=160x90:rate=10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    store = JobStore(tmp_path / "jobs")
    job_id = "abcd1234abcd1234"
    store.create(source, job_id=job_id)
    store.transition(job_id, JobState.PROBED)
    store.transition(job_id, JobState.SPLIT_PROPOSED)
    job_dir = store.job_dir(job_id)
    still = "segment_stills/seg_00.ppm"
    write_gray_still(job_dir / still, bytes([40]) * PIXELS)
    write_segments(
        job_dir,
        [
            TimelineSegment(0, 0.0, 10.0, "indoor", "noisy", 4.0, 0, still),
            TimelineSegment(1, 10.0, 20.0, "indoor", "noisy", 14.0, 0),
        ],
    )
    write_samples(
        job_dir,
        [
            SampleRow(
                time_s=float(index),
                luma_mean=40.0,
                luma_p05=20.0,
                luma_p95=80.0,
                clip_low=0.0,
                clip_high=0.0,
                rb_cast=0.0,
                variance=10.0,
                motion=1.0,
                subject_x=0.5,
            )
            for index in range(20)
        ],
    )
    return store, job_id


@pytest.mark.parametrize("session", ["other", "empty", "missing"])
@pytest.mark.parametrize("explicit_root", [True, False])
def test_reopened_wizard_uses_requested_job(
    tmp_path: Path, session: str, explicit_root: bool
) -> None:
    store = JobStore(tmp_path / "jobs")
    (tmp_path / "clip.mp4").write_bytes(b"source")
    requested_id = "aaaaaaaaaaaaaaaa"
    other_id = "bbbbbbbbbbbbbbbb"
    for job_id in (requested_id, other_id):
        store.create(tmp_path / "clip.mp4", job_id=job_id)
        job_dir = store.job_dir(job_id)
        save_job_wizard_state(
            job_dir, WizardState(step=WizardStep.RESULT, job_id=job_id)
        )
        (job_dir / "output.mp4").write_bytes(job_id.encode())
    other_state = load_job_wizard_state(store.job_dir(other_id))
    assert other_state is not None
    if session == "other":
        save_session_state(store.root, other_state)
    elif session == "empty":
        save_session_state(store.root, WizardState())

    server = ReviewServer(
        store.job_dir(requested_id),
        jobs_dir=store.root if explicit_root else None,
        job_id=requested_id if explicit_root else None,
    )
    base = server.start()
    try:
        with urlopen(base + "/api/state") as response:
            assert json.load(response)["job_id"] == requested_id
        with urlopen(base + "/output.mp4") as response:
            assert response.read() == requested_id.encode()

        # Another window selecting a different job must not redirect this one.
        save_session_state(store.root, other_state)
        request = Request(
            base + "/api/result",
            data=b'{"mood": "happy"}',
            headers={
                "Content-Type": "application/json",
                "X-Wizard-Token": server.token or "",
            },
            method="POST",
        )
        with urlopen(request) as response:
            view = json.load(response)
        assert view["job_id"] == requested_id
        assert view["step"] == "done"
        requested = load_job_wizard_state(store.job_dir(requested_id))
        assert requested is not None and requested.step is WizardStep.DONE
        assert load_job_wizard_state(store.job_dir(other_id)) == other_state
    finally:
        server.stop()


def test_reopened_controller_can_start_a_new_session(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"source")
    store = JobStore(tmp_path / "jobs")
    job_id = "aaaaaaaaaaaaaaaa"
    store.create(tmp_path / "clip.mp4", job_id=job_id)
    saved = WizardState(step=WizardStep.DONE, job_id=job_id)
    save_job_wizard_state(store.job_dir(job_id), saved)
    controller = WizardController(store.root, job_id=job_id)

    controller.start_new_job()
    fresh = controller.current_state()
    assert fresh.job_id is None
    assert fresh.step in {WizardStep.PICK_FILE, WizardStep.SETUP}
    save_session_state(store.root, saved)
    assert controller.current_state() == fresh
    assert load_job_wizard_state(store.job_dir(job_id)) == saved


def test_reopened_controller_requires_job_wizard_state(tmp_path: Path) -> None:
    (tmp_path / "clip.mp4").write_bytes(b"source")
    store = JobStore(tmp_path / "jobs")
    job_id = "aaaaaaaaaaaaaaaa"
    store.create(tmp_path / "clip.mp4", job_id=job_id)
    with pytest.raises(WizardError, match="no wizard state"):
        WizardController(store.root, job_id=job_id)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_split_page_accepts_and_applies_a_note(tmp_path: Path) -> None:
    store, job_id = _proposed_job(tmp_path)
    server = ReviewServer(
        store.job_dir(job_id),
        jobs_dir=store.root,
        job_id=job_id,
    )
    base = server.start()
    try:
        with urlopen(base + "/") as response:
            page = response.read().decode()
        assert "Accept" in page
        assert "background: #fff" in page
        assert "noisy" in page
        assert 'src="/segment_stills/seg_00.bmp"' in page
        with urlopen(base + "/segment_stills/seg_00.bmp") as response:
            assert response.read().startswith(b"BM")
        request = Request(base + "/note", data=b"note=too+many", method="POST")
        with urlopen(request) as response:
            assert response.status == 200
        assert len(load_segments(store.job_dir(job_id))) == 1
        accepted = Request(base + "/accept", data=b"", method="POST")
        with urlopen(accepted) as response:
            assert response.status == 200
        assert store.load(job_id).state is JobState.SPLIT_ACCEPTED
    finally:
        server.stop()
