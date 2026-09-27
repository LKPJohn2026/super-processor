"""Tests for the localhost segment review route."""

from __future__ import annotations

import json
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from super_processor.review import ReviewServer
from super_processor.segments import TimelineSegment, write_gray_still, write_segments

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
