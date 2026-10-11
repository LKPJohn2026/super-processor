"""Localhost review server: legacy split page and Gemini wizard."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import threading
from email.message import Message
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .estimators import EstimatorError
from .jobs import JobError
from .segments import STILLS_DIR_NAME, SegmentError, load_segments
from .shot_plan import PREVIEW_NAME, PREVIEWS_DIR
from .shots import STILLS_DIR
from .wizard import WizardController, WizardError, WizardStep, load_job_wizard_state


class ReviewError(RuntimeError):
    """Raised when a review request cannot be served."""


_STREAM_CHUNK = 1 << 20


def parse_byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Resolve a single ``Range: bytes=`` header to inclusive offsets.

    ``None`` means serve the whole file: no header, a malformed one, another
    unit, or several ranges (which a server may ignore). :class:`ValueError`
    means the range cannot be satisfied and the answer is 416.
    """
    if not header:
        return None
    unit, _, spec = header.strip().partition("=")
    first, dash, last = spec.strip().partition("-")
    if unit.strip().lower() != "bytes" or "," in spec or not dash:
        return None
    if not (first.isdigit() or not first) or not (last.isdigit() or not last):
        return None
    if not first:
        if not last or int(last) == 0 or size == 0:
            raise ValueError("range not satisfiable")
        return max(0, size - int(last)), size - 1
    start = int(first)
    end = int(last) if last else size - 1
    if start >= size or end < start:
        raise ValueError("range not satisfiable")
    return start, min(end, size - 1)


ALLOWED_ORIGINS_ENV = "SUPER_PROCESSOR_ALLOWED_ORIGINS"
# The Vite dev and preview servers, and the published React shell, may drive a
# local wizard. Add others (comma separated) with SUPER_PROCESSOR_ALLOWED_ORIGINS.
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "https://lkpjohn2026.github.io",
)
_LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_TRUSTED_FETCH_SITES = frozenset({"same-origin", "none"})


def _hostname(host_header: str) -> str:
    value = host_header.strip().lower()
    if value.startswith("["):
        return value[1 : value.find("]")] if "]" in value else value[1:]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _server_port(server: object) -> int:
    address = getattr(server, "server_address", None)
    if isinstance(address, tuple) and len(address) >= 2:
        return int(address[1])
    return 0


def allowed_origins(port: int) -> frozenset[str]:
    """Origins that may call the wizard: itself, the React shell, and env extras."""
    own = {
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
        f"http://[::1]:{port}",
    }
    extra = {
        item.strip().rstrip("/")
        for item in os.environ.get(ALLOWED_ORIGINS_ENV, "").split(",")
        if item.strip()
    }
    return frozenset({*DEFAULT_ALLOWED_ORIGINS, *own, *extra})


def request_rejection(
    headers: Message,
    *,
    port: int,
    state_changing: bool,
) -> str | None:
    """Why a request must be refused, or ``None`` when it may proceed.

    The Host check stops DNS rebinding. A browser always sends ``Origin`` on a
    cross-site POST, and ``Sec-Fetch-Site`` on a cross-site GET such as an
    image tag, so neither can start work from another page. Non-browser
    clients send neither header and are local already.
    """
    host = headers.get("Host")
    if host is not None and _hostname(host) not in _LOCAL_HOSTNAMES:
        return "unexpected Host header"
    origin = headers.get("Origin")
    if origin is not None and origin.rstrip("/") not in allowed_origins(port):
        return "origin not allowed"
    if state_changing and origin is None:
        site = headers.get("Sec-Fetch-Site")
        if site is not None and site not in _TRUSTED_FETCH_SITES:
            return "cross-site request refused"
    return None


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


def _legacy_handler(
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

        def _reject(self, message: str, status: int = 400) -> None:
            body = message.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _refused(self, *, state_changing: bool) -> bool:
            reason = request_rejection(
                self.headers,
                port=_server_port(self.server),
                state_changing=state_changing,
            )
            if reason is None:
                return False
            self._reject(reason, status=403)
            return True

        def do_POST(self) -> None:  # noqa: N802
            if self._refused(state_changing=True):
                return
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
            if self._refused(state_changing=False):
                return
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


# Shot stills are named by their start time in milliseconds.
_STILL_URL = re.compile(r"/shot_stills/(\d{9}\.jpg)")

_API_POSTS = {
    "/api/intro": "/intro",
    "/api/llm": "/llm",
    "/api/local-llm": "/local-llm",
    "/api/setup": "/setup",
    "/api/pick": "/pick",
    "/api/new": "/new",
    "/api/shots": "/shots",
    "/api/looks": "/looks",
    "/api/result": "/result",
}


def form_fields_from_body(raw: bytes, content_type: str) -> dict[str, list[str]]:
    """Parse a wizard POST body from a form or a JSON object."""
    if "application/json" in content_type:
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return {}
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise WizardError("invalid JSON") from exc
        if not isinstance(data, dict):
            raise WizardError("JSON body must be an object")
        fields: dict[str, list[str]] = {}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, bool):
                fields[str(key)] = ["1" if value else "0"]
                continue
            fields[str(key)] = [str(value)]
        return fields
    return parse_qs(raw.decode("utf-8", errors="replace"))


def _wizard_handler(controller: WizardController) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def _redirect(self, location: str = "/") -> None:
            self.send_response(303)
            self.send_header("Location", location)
            self.end_headers()

        def _html(self, page: str, status: int = 200) -> None:
            body = page.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _reject(self, message: str, status: int = 400) -> None:
            self._html(
                f"<!DOCTYPE html><html><body><p>{escape(message)}</p>"
                f'<p><a href="/">Back</a></p></body></html>',
                status=status,
            )

        def _port(self) -> int:
            return _server_port(self.server)

        def _refused(self, path: str, *, state_changing: bool) -> bool:
            reason = request_rejection(
                self.headers, port=self._port(), state_changing=state_changing
            )
            if reason is None:
                return False
            if path.startswith("/api/"):
                self._json({"error": reason}, status=403)
            else:
                self._reject(reason, status=403)
            return True

        def _json(self, payload: object, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            origin = self.headers.get("Origin")
            if origin and origin.rstrip("/") in allowed_origins(self._port()):
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str, *, head: bool) -> None:
            """Serve a file with single-range support, streamed in chunks."""
            with path.open("rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                try:
                    span = parse_byte_range(self.headers.get("Range"), size)
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start, end = span if span is not None else (0, size - 1)
                length = max(0, end - start + 1)
                self.send_response(206 if span is not None else 200)
                self.send_header("Content-Type", content_type)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(length))
                # output.mp4 is rewritten in place by a revise.
                self.send_header("Cache-Control", "no-store")
                if span is not None:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if head:
                    return
                handle.seek(start)
                remaining = length
                try:
                    while remaining > 0:
                        chunk = handle.read(min(_STREAM_CHUNK, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Players drop a range request as soon as they seek.
                    return

        def _media_file(self, path: str) -> tuple[Path, str] | None:
            """Map a media URL to a file inside the job, or ``None``."""
            state = controller.current_state()
            if not state.job_id:
                return None
            job_dir = controller.store.job_dir(state.job_id)
            if path == "/output.mp4":
                output = job_dir / "output.mp4"
                return (output, "video/mp4") if output.is_file() else None
            still = _STILL_URL.fullmatch(path)
            if still is not None:
                image = job_dir / STILLS_DIR / still.group(1)
                return (image, "image/jpeg") if image.is_file() else None
            if path.startswith("/previews/"):
                name = path.removeprefix("/previews/")
                match = PREVIEW_NAME.fullmatch(name)
                if match is None:
                    return None
                preview = job_dir / PREVIEWS_DIR / name
                kind = "video/mp4" if match.group(4) == "mp4" else "image/jpeg"
                return (preview, kind) if preview.is_file() else None
            return None

        def do_HEAD(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if self._refused(path, state_changing=False):
                return
            media = self._media_file(path)
            if media is None:
                self.send_response(404)
                self.end_headers()
                return
            self._send_file(media[0], media[1], head=True)

        def _read_fields(self) -> dict[str, list[str]]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length)
            return form_fields_from_body(raw, self.headers.get("Content-Type", ""))

        def do_OPTIONS(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if not path.startswith("/api/"):
                self.send_response(404)
                self.end_headers()
                return
            origin = self.headers.get("Origin")
            if (
                request_rejection(self.headers, port=self._port(), state_changing=False)
                or origin is None
            ):
                self.send_response(403)
                self.end_headers()
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if self._refused(path, state_changing=True):
                return
            try:
                fields = self._read_fields()
            except WizardError as exc:
                if path.startswith("/api/"):
                    self._json({"error": str(exc)}, status=400)
                    return
                self._reject(str(exc))
                return
            action = _API_POSTS.get(path, path)
            try:
                controller.handle_post(action, fields)
            except WizardError as exc:
                if path.startswith("/api/"):
                    self._json({"error": str(exc)}, status=400)
                    return
                self._reject(str(exc))
                return
            if path.startswith("/api/"):
                self._json(controller.api_view())
                return
            self._redirect("/")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            runs = query.get("run", [""])[0] == "1"
            if self._refused(path, state_changing=runs):
                return
            try:
                if path == "/api/state":
                    self._json(controller.api_view())
                    return
                # Long passes run on the controller's worker thread. These
                # GETs start one when none is running and return at once;
                # callers poll /api/state (or the page refreshes).
                if path == "/api/render" and runs:
                    controller.start_render()
                    self._json(controller.api_view())
                    return
                if path == "/render" and runs:
                    controller.start_render()
                    self._redirect("/")
                    return
                if path == "/api/scan" and runs:
                    controller.start_scan()
                    self._json(controller.api_view())
                    return
                if path == "/shots" and runs:
                    controller.start_scan()
                    self._redirect("/")
                    return
                if path == "/api/plan" and runs:
                    controller.start_plan()
                    self._json(controller.api_view())
                    return
                if path == "/looks" and runs:
                    controller.start_plan()
                    self._redirect("/")
                    return
                if path == "/":
                    self._html(controller.render())
                    return
                media = self._media_file(path)
                if media is not None:
                    self._send_file(media[0], media[1], head=False)
                    return
                self.send_response(404)
                self.end_headers()
            except (WizardError, ReviewError, OSError) as exc:
                self._reject(str(exc))

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
        wizard_state = load_job_wizard_state(job_dir)
        if wizard_state is not None and wizard_state.step is not WizardStep.INTRO:
            controller = WizardController(jobs_dir or job_dir.parent)
            handler: type[BaseHTTPRequestHandler] = _wizard_handler(controller)
        else:
            handler = _legacy_handler(job_dir, jobs_dir=jobs_dir, job_id=job_id)
        self._httpd = ThreadingHTTPServer((host, port), handler)
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


class WizardServer:
    """Serve the Gemini wizard for a jobs directory."""

    def __init__(
        self,
        jobs_dir: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        controller: WizardController | None = None,
    ) -> None:
        self.controller = controller or WizardController(jobs_dir)
        self._httpd = ThreadingHTTPServer(
            (host, port),
            _wizard_handler(self.controller),
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        name = host.decode() if isinstance(host, bytes) else host
        return f"http://{name}:{port}"

    def start(self) -> str:
        self._thread.start()
        return self.url

    def wait(self) -> None:
        self._thread.join()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=5)
        self._httpd.server_close()
