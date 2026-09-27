"""Localhost review server for one job directory.

The page reads the same ``segments.json`` the CLI writes. Accept and note
call the existing ``segment`` command. The page does not split the timeline
itself.
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .estimators import EstimatorError
from .jobs import JobError
from .segments import STILLS_DIR_NAME, SegmentError, load_segments


class ReviewError(RuntimeError):
    """Raised when a review request cannot be served."""


def segment_review_payload(job_dir: Path) -> dict[str, object]:
    """Return the segment list, including a URL for each still."""
    try:
        segments = load_segments(job_dir)
    except SegmentError as exc:
        raise ReviewError(str(exc)) from exc
    items: list[dict[str, object]] = []
    for segment in segments:
        data = segment.to_dict()
        still = segment.still_path.replace("\\", "/")
        data["still_url"] = f"/{still}" if still else ""
        items.append(data)
    return {"segments": items}


def _still_file(job_dir: Path, url_path: str) -> Path | None:
    """Resolve a still URL onto a file inside the job's still directory."""
    relative = unquote(url_path).lstrip("/")
    if not relative.startswith(f"{STILLS_DIR_NAME}/"):
        return None
    root = (job_dir / STILLS_DIR_NAME).resolve()
    candidate = (job_dir / relative).resolve()
    if candidate != root and root not in candidate.parents:
        return None
    if not candidate.is_file():
        return None
    return candidate


def gray_ppm_to_bmp(ppm: bytes) -> bytes:
    """Turn a binary gray PPM into a 24-bit BMP a browser can show."""
    marker = b"\n255\n"
    if not ppm.startswith(b"P5\n") or marker not in ppm:
        raise ReviewError("still is not a gray PPM")
    header, pixels = ppm[3:].split(marker, 1)
    width_text, height_text = header.split()
    width = int(width_text)
    height = int(height_text)
    if len(pixels) < width * height:
        raise ReviewError("still is shorter than its header")
    row_stride = (width * 3 + 3) & ~3
    pixel_size = row_stride * height
    header_size = 54
    out = bytearray(b"BM")
    out += struct.pack("<IHHI", header_size + pixel_size, 0, 0, header_size)
    out += struct.pack(
        "<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_size, 0, 0, 0, 0
    )
    for y in range(height - 1, -1, -1):
        start = y * width
        row = bytearray()
        for value in pixels[start : start + width]:
            row += bytes((value, value, value))
        row += b"\x00" * (row_stride - width * 3)
        out += row
    return bytes(out)


def split_review_page(payload: dict[str, object]) -> str:
    """Render the split review page from a segment payload."""
    raw_segments = payload.get("segments")
    segments = raw_segments if isinstance(raw_segments, list) else []
    cards: list[str] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        start = item.get("start_s", "")
        end = item.get("end_s", "")
        context = escape(str(item.get("context", "")))
        problem = escape(str(item.get("problem", "")))
        still = str(item.get("still_url", ""))
        image = ""
        if still.endswith(".ppm"):
            image = f'<img src="{escape(still[:-4] + ".bmp")}" alt="segment still">'
        cards.append(
            "<article>"
            f"<p>{escape(str(start))}–{escape(str(end))}s {context} {problem}</p>"
            f"{image}"
            "</article>"
        )
    body = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Split review</title>
<style>
body {{ background: #fff; color: #111; font-family: sans-serif; }}
img {{ display: block; margin: 0.4rem 0 1.2rem; background: #ddd; }}
form {{ display: inline; margin-right: 0.6rem; }}
</style>
</head>
<body>
<h1>Split review</h1>
<form method="post" action="/accept"><button type="submit">Accept</button></form>
<form method="post" action="/note">
<label>Note <input name="note" type="text"></label>
<button type="submit">Send note</button>
</form>
{body}
</body>
</html>
"""


def apply_review_action(
    jobs_dir: Path,
    job_id: str,
    *,
    note: str | None,
    accept: bool,
) -> None:
    """Run ``segment --accept`` or ``segment --note`` for this job."""
    from .cli import run_segment_command

    try:
        code = run_segment_command(
            argparse.Namespace(
                jobs_dir=jobs_dir,
                job_id=job_id,
                note=note,
                accept=accept,
            )
        )
    except (JobError, SegmentError, EstimatorError) as exc:
        raise ReviewError(str(exc)) from exc
    if code != 0:
        raise ReviewError("the split command failed")


def _handler(
    job_dir: Path,
    *,
    jobs_dir: Path | None,
    job_id: str | None,
) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _redirect(self) -> None:
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

        def _reject(self, message: str) -> None:
            body = message.encode()
            self.send_response(400)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if jobs_dir is None or job_id is None or path not in {"/accept", "/note"}:
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            fields = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
            try:
                if path == "/accept":
                    apply_review_action(jobs_dir, job_id, note=None, accept=True)
                else:
                    note = fields.get("note", [""])[0].strip()
                    if not note:
                        raise ReviewError("a note is required")
                    apply_review_action(jobs_dir, job_id, note=note, accept=False)
            except (ReviewError, OSError) as exc:
                self._reject(str(exc))
                return
            self._redirect()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                try:
                    page = split_review_page(segment_review_payload(job_dir)).encode()
                except ReviewError as exc:
                    self._reject(str(exc))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return
            if path.endswith(".bmp"):
                still = _still_file(job_dir, path[:-4] + ".ppm")
                if still is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    data = gray_ppm_to_bmp(still.read_bytes())
                except ReviewError as exc:
                    self._reject(str(exc))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/bmp")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path == "/segments":
                try:
                    payload = segment_review_payload(job_dir)
                except ReviewError as exc:
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            still = _still_file(job_dir, path)
            if still is None:
                self.send_response(404)
                self.end_headers()
                return
            data = still.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/x-portable-pixmap")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


class ReviewServer:
    """Serve one job directory on localhost until ``stop``."""

    def __init__(
        self,
        job_dir: Path,
        *,
        jobs_dir: Path | None = None,
        job_id: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.job_dir = job_dir
        self._httpd = ThreadingHTTPServer(
            (host, port),
            _handler(job_dir, jobs_dir=jobs_dir, job_id=job_id),
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        name = host.decode() if isinstance(host, bytes) else host
        return f"http://{name}:{port}"

    def start(self) -> str:
        """Start the server and return its base URL."""
        self._thread.start()
        return self.url

    def wait(self) -> None:
        """Block until ``stop`` is called."""
        self._thread.join()

    def stop(self) -> None:
        """Stop the server."""
        self._httpd.shutdown()
        self._thread.join(timeout=5)
        self._httpd.server_close()
