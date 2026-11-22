"""Localhost review server: legacy split page and Gemini wizard."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import secrets
import struct
import threading
from email.message import Message
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import parse_qs, parse_qsl, quote, unquote, urlencode, urlparse

from .atomic_io import open_private_new
from .estimators import EstimatorError
from .gemini import redact_payload, redact_secrets
from .jobs import JobError
from .segments import STILLS_DIR_NAME, SegmentError, load_segments
from .shot_plan import PREVIEW_NAME, PREVIEWS_DIR
from .shots import STILLS_DIR
from .wizard import (
    WizardConflict,
    WizardController,
    WizardError,
    WizardStep,
    job_thumb,
    load_job_wizard_state,
)


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
# Local Vite / preview shells may drive the wizard. Pages and any other site
# must be listed in SUPER_PROCESSOR_ALLOWED_ORIGINS (comma-separated http(s)
# origins). A published static shell without that env cannot CSRF a running
# wizard.
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
)
_LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})
_TRUSTED_FETCH_SITES = frozenset({"same-origin", "none"})
_REJECTED_ORIGIN_TOKENS = frozenset({"*", "null"})
# Wizard POSTs only carry a key, a path, or a short note.
MAX_BODY_BYTES = 256 * 1024
# A dropped video is streamed to disk; 30 minutes of high-bitrate 1080p fits.
MAX_UPLOAD_BYTES = 32 * 1024**3
_UPLOAD_CHUNK = 1024 * 1024
_UPLOAD_PATHS = frozenset({"/upload", "/api/upload"})
# Every request to the wizard proves it holds this run's token: as this
# header, as the per-port cookie the launch URL (``/?token=``) sets, or, for
# a GET of a video or still only, as ``?t=``. State-changing requests also
# carry it as this header or a ``token`` field, so the cookie alone (which a
# browser sends by itself) cannot forge one.
TOKEN_HEADER = "X-Wizard-Token"
# The launch URL's query parameter; the wizard swaps it for the cookie.
LAUNCH_TOKEN_PARAM = "token"
# Media URLs the React shell builds carry the token this way, since a
# <video> or <img> cannot send a header (and a shell on another origin has
# no cookie here).
MEDIA_TOKEN_PARAM = "t"
WIZARD_COOKIE_PREFIX = "sp_wizard_"
# A connection that sends or takes no byte for this long gives its thread
# back. It bounds each read and write, not a whole request, so a long video
# download or upload is unaffected; a paused player just asks again with a
# Range request.
SOCKET_TIMEOUT_S = 60.0
# More connections than this at once are closed straight away. It stays
# well under the usual 1024 open-file limit, so the server never runs out of
# descriptors, yet high enough that a local process holding idle sockets
# must open hundreds of them to crowd out the browser's own requests.
MAX_CONNECTIONS = 512
UNAUTHENTICATED_MESSAGE = (
    "This wizard needs its token. Open the address that super-processor review "
    "printed (it ends in ?token=...), or send the printed token as the "
    "X-Wizard-Token header."
)
_POST_FORM = re.compile(r'<form\b[^>]*\bmethod="post"[^>]*>', re.IGNORECASE)


class BodyError(WizardError):
    """A request body the wizard will not read."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


def content_length(headers: Message) -> int:
    """The declared body size, refusing junk, negatives, and big bodies."""
    raw = (headers.get("Content-Length") or "0").strip() or "0"
    try:
        length = int(raw)
    except ValueError:
        raise BodyError("bad Content-Length", 400) from None
    if length < 0:
        raise BodyError("bad Content-Length", 400)
    if length > MAX_BODY_BYTES:
        raise BodyError("body too large", 413)
    return length


def new_wizard_token() -> str:
    return secrets.token_urlsafe(24)


def with_wizard_token(page: str, token: str) -> str:
    """Put ``token`` in every POST form and every ``?run=1`` link of a page."""
    hidden = f'<input type="hidden" name="token" value="{token}">'
    page = _POST_FORM.sub(lambda match: match.group(0) + hidden, page)
    return page.replace("?run=1", f"?run=1&amp;token={token}")


def token_matches(supplied: str | None, token: str) -> bool:
    return bool(supplied) and hmac.compare_digest(
        str(supplied).encode(), token.encode()
    )


def wizard_cookie_name(port: int) -> str:
    """One cookie per wizard port, so two wizards never swap tokens."""
    return f"{WIZARD_COOKIE_PREFIX}{port}"


def launch_url(base: str, token: str) -> str:
    """The address to open: it logs the browser in, then drops the token."""
    return f"{base}/?{LAUNCH_TOKEN_PARAM}={quote(token, safe='')}"


def _cookie(headers: Message, name: str) -> str | None:
    for raw in headers.get_all("Cookie") or []:
        for part in str(raw).split(";"):
            key, sep, value = part.strip().partition("=")
            if sep and key.strip() == name:
                return value.strip().strip('"')
    return None


def request_authenticated(
    headers: Message,
    *,
    token: str,
    port: int,
    media_token: str | None = None,
) -> str | None:
    """How a request proves it holds the run's token, or ``None``.

    ``"header"`` for ``X-Wizard-Token``; ``"cookie"`` for the launch cookie,
    counted only on a same-origin request or one with no ``Sec-Fetch-Site``
    (a browser sends cookies to every port of a host, so another local page
    must not ride on it); ``"media"`` for a matching ``?t=`` that the caller
    passes for a media GET.
    """
    if token_matches(headers.get(TOKEN_HEADER), token):
        return "header"
    site = headers.get("Sec-Fetch-Site")
    same_origin = site is None or site in _TRUSTED_FETCH_SITES
    if same_origin and token_matches(_cookie(headers, wizard_cookie_name(port)), token):
        return "cookie"
    if media_token is not None and token_matches(media_token, token):
        return "media"
    return None


def _same_path(path: str) -> str:
    """A redirect target on this server; never ``//host`` or ``/\\host``."""
    if not path.startswith("/") or path[1:2] in {"/", "\\"}:
        return "/"
    return path


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


def _origin_allowed(item: str) -> bool:
    """True for a concrete http(s) origin; rejects ``*``, ``null``, and junk."""
    value = item.strip().rstrip("/")
    if not value or value.lower() in _REJECTED_ORIGIN_TOKENS:
        return False
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        return False
    if not parsed.netloc or parsed.username is not None or parsed.password is not None:
        return False
    if parsed.path not in {"", "/"}:
        return False
    return not (parsed.query or parsed.fragment)


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
        if _origin_allowed(item)
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
    image tag, so neither can start work from another page. This is only the
    browser layer: every request must also carry the run's token (see
    :func:`request_authenticated`), since any local program can leave these
    headers out.
    """
    host = headers.get("Host")
    if host is None or not str(host).strip():
        return "missing Host header"
    if _hostname(host) not in _LOCAL_HOSTNAMES:
        return "unexpected Host header"
    origin = headers.get("Origin")
    if origin is not None and origin.rstrip("/") not in allowed_origins(port):
        return "origin not allowed"
    if state_changing and origin is None:
        site = headers.get("Sec-Fetch-Site")
        if site is not None and site not in _TRUSTED_FETCH_SITES:
            return "cross-site request refused"
    return None


class _GuardedHandler(BaseHTTPRequestHandler):
    """What both servers share: the token gate and the safety headers.

    Subclasses set ``token``. ``timeout`` (``SOCKET_TIMEOUT_S`` when the
    server is made) makes an idle or trickling connection give its thread
    back instead of holding it forever.
    """

    token: ClassVar[str] = ""
    timeout = SOCKET_TIMEOUT_S

    def log_message(self, format: str, *args: object) -> None:
        return

    def setup(self) -> None:
        # Media authenticated by ?t= may be shown by a shell on another site.
        self.resource_policy = "same-site"
        super().setup()

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        # Not no-referrer: a browser then sends "Origin: null" on the pages'
        # own form posts, which the origin check refuses. same-origin still
        # sends no Referer to any other site.
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header(
            "Cross-Origin-Resource-Policy",
            getattr(self, "resource_policy", "same-site"),
        )
        # Pages and state hold the token; output.mp4 is rewritten in place.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _port(self) -> int:
        return _server_port(self.server)

    def _authenticated(self, media_token: str | None = None) -> str | None:
        return request_authenticated(
            self.headers, token=self.token, port=self._port(), media_token=media_token
        )

    def _unauthorized(self, path: str) -> None:
        if path.startswith("/api/") or path == "/segments":
            body = json.dumps({"error": UNAUTHENTICATED_MESSAGE}).encode()
            kind = "application/json"
        else:
            body = (UNAUTHENTICATED_MESSAGE + "\n").encode()
            kind = "text/plain; charset=utf-8"
        self.send_response(401)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _launch(self, path: str, query: str) -> None:
        """Swap a launch URL's ``?token=`` for the cookie, then drop it.

        A wrong token sets nothing and gets 401.
        """
        pairs = parse_qsl(query, keep_blank_values=True)
        supplied = next((v for k, v in pairs if k == LAUNCH_TOKEN_PARAM), "")
        if not token_matches(supplied, self.token):
            self._unauthorized(path)
            return
        rest = [(k, v) for k, v in pairs if k != LAUNCH_TOKEN_PARAM]
        location = _same_path(path) + (f"?{urlencode(rest)}" if rest else "")
        self.send_response(303)
        self.send_header("Location", location)
        self.send_header(
            "Set-Cookie",
            f"{wizard_cookie_name(self._port())}={self.token}; "
            "HttpOnly; SameSite=Strict; Path=/",
        )
        self.send_header("Content-Length", "0")
        self.end_headers()


class _LocalHTTPServer(ThreadingHTTPServer):
    """A threading server that serves at most ``MAX_CONNECTIONS`` at once."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
    ) -> None:
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(address, handler)

    def process_request(self, request: object, client_address: object) -> None:
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)  # type: ignore[arg-type]
            return
        try:
            super().process_request(request, client_address)  # type: ignore[arg-type]
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request: object, client_address: object) -> None:
        try:
            super().process_request_thread(request, client_address)  # type: ignore[arg-type]
        finally:
            self._slots.release()


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
    token: str,
) -> type[BaseHTTPRequestHandler]:
    run_token = token

    class Handler(_GuardedHandler):
        token = run_token
        timeout = SOCKET_TIMEOUT_S

        def _redirect(self) -> None:
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send(self, kind: str, data: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _reject(self, message: str, status: int = 400) -> None:
            self._send("text/plain; charset=utf-8", message.encode(), status)

        def _not_found(self) -> None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _refused(self, *, state_changing: bool) -> bool:
            reason = request_rejection(
                self.headers,
                port=self._port(),
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
            if self._authenticated() is None:
                self._unauthorized(path)
                return
            if jobs_dir is None or job_id is None or path not in {"/accept", "/note"}:
                self._not_found()
                return
            try:
                length = content_length(self.headers)
            except BodyError as exc:
                self._reject(str(exc), status=exc.status)
                return
            fields = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
            # The cookie comes along by itself; the page's own token must too.
            supplied = fields.pop("token", [""])[0] or self.headers.get(TOKEN_HEADER)
            if not token_matches(supplied, run_token):
                self._reject("missing or wrong review token; reload the page", 403)
                return
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
            parsed = urlparse(self.path)
            path = parsed.path
            if LAUNCH_TOKEN_PARAM in parse_qs(parsed.query, keep_blank_values=True):
                self._launch(path, parsed.query)
                return
            if self._authenticated() is None:
                self._unauthorized(path)
                return
            if path == "/":
                try:
                    payload = segment_review_payload(job_dir)
                except ReviewError as exc:
                    self._reject(str(exc))
                    return
                page = with_wizard_token(split_review_page(payload), run_token)
                self._send("text/html; charset=utf-8", page.encode())
                return
            if path.endswith(".bmp"):
                still = _still_file(job_dir, path[:-4] + ".ppm")
                if still is None:
                    self._not_found()
                    return
                try:
                    data = gray_ppm_to_bmp(still.read_bytes())
                except ReviewError as exc:
                    self._reject(str(exc))
                    return
                self._send("image/bmp", data)
                return
            if path == "/segments":
                try:
                    payload = segment_review_payload(job_dir)
                except ReviewError as exc:
                    error = json.dumps({"error": str(exc)}).encode()
                    self._send("application/json", error, 404)
                    return
                self._send("application/json", json.dumps(payload).encode())
                return
            still = _still_file(job_dir, path)
            if still is None:
                self._not_found()
                return
            self._send("image/x-portable-pixmap", still.read_bytes())

    return Handler


# Shot stills are named by their start time in milliseconds.
_STILL_URL = re.compile(r"/shot_stills/(\d{9}\.jpg)")
# The segment loop's still grid (see :func:`.segment_loop.still_name`).
_GRID_URL = re.compile(r"/segment_stills/(t_\d{8}\.jpg)")

_API_POSTS = {
    "/api/intro": "/intro",
    "/api/llm": "/llm",
    "/api/local-llm": "/local-llm",
    "/api/setup": "/setup",
    "/api/pick": "/pick",
    "/api/new": "/new",
    "/api/browse": "/browse",
    "/api/goto": "/goto",
    "/api/segments": "/segments",
    "/api/preview": "/preview",
    "/api/scene": "/scene",
    "/api/ready": "/ready",
    "/api/result": "/result",
    # Stops the segmenting or render in progress (see WizardController.cancel).
    "/api/cancel": "/cancel",
    # Lets an interrupted step's work start again (WizardController.resume).
    "/api/resume": "/resume",
    # Shows another job, by its ``job`` field (WizardController.open_job).
    "/api/jobs/open": "/jobs/open",
}
# Files of any job, by its id: /jobs/<id>/output.mp4, thumb.jpg, a kept
# version under outputs/, and the stills and previews the current job has.
_JOB_MEDIA = re.compile(r"/jobs/([a-z0-9]{8,32})(/.+)")
_VERSION_URL = re.compile(r"/outputs/(v\d{3,6}\.mp4)")


def _job_media(
    job_dir: Path, path: str, *, scoped: bool = False
) -> tuple[Path, str] | None:
    """Map a media path inside one job to its file and type, or ``None``.

    Only these named files are served, never one a symlink points out of the
    job with. ``thumb.jpg`` and the kept versions are served by job id only.
    """
    if path == "/output.mp4":
        output = job_dir / "output.mp4"
        return (output, "video/mp4") if output.is_file() else None
    if scoped and path == "/thumb.jpg":
        thumb = job_thumb(job_dir)
        return (thumb, "image/jpeg") if thumb is not None else None
    version = _VERSION_URL.fullmatch(path) if scoped else None
    if version is not None:
        kept = job_dir / "outputs" / version.group(1)
        usable = kept.is_file() and not kept.is_symlink()
        return (kept, "video/mp4") if usable else None
    still = _STILL_URL.fullmatch(path)
    if still is not None:
        image = job_dir / STILLS_DIR / still.group(1)
        return (image, "image/jpeg") if image.is_file() else None
    grid = _GRID_URL.fullmatch(path)
    if grid is not None:
        image = job_dir / STILLS_DIR_NAME / grid.group(1)
        usable = image.is_file() and not image.is_symlink()
        return (image, "image/jpeg") if usable else None
    if path.startswith("/previews/"):
        name = path.removeprefix("/previews/")
        match = PREVIEW_NAME.fullmatch(name)
        if match is None:
            return None
        preview = job_dir / PREVIEWS_DIR / name
        kind = "video/mp4" if match.group(4) == "mp4" else "image/jpeg"
        return (preview, kind) if preview.is_file() else None
    return None


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


def _is_media_path(path: str) -> bool:
    """A URL :func:`_job_media` may serve: the only GETs ``?t=`` may sign."""
    return (
        path == "/output.mp4"
        or path.startswith("/previews/")
        or _JOB_MEDIA.fullmatch(path) is not None
        or _STILL_URL.fullmatch(path) is not None
        or _GRID_URL.fullmatch(path) is not None
    )


def _wizard_handler(
    controller: WizardController, token: str
) -> type[BaseHTTPRequestHandler]:
    run_token = token

    class Handler(_GuardedHandler):
        token = run_token
        timeout = SOCKET_TIMEOUT_S

        def _redirect(self, location: str = "/") -> None:
            self.send_response(303)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _html(self, page: str, status: int = 200) -> None:
            # Pages hold the token; the shared headers stop framing them. No
            # page may show the Gemini key (it is taken out before the
            # token goes in, so the token is never touched).
            body = with_wizard_token(redact_secrets(page), token).encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _unauthorized(self, path: str) -> None:
            if path.startswith("/api/"):
                # With CORS headers, so the shell can tell "no token" apart.
                self._json({"error": UNAUTHENTICATED_MESSAGE}, status=401)
                return
            super()._unauthorized(path)

        def _gate(self, path: str, query: dict[str, list[str]]) -> bool:
            """Refuse (401/403) a GET or HEAD without the token; True if refused.

            ``?t=`` counts for media only. A cross-site media load (an
            ``<img>`` or ``<video>`` on another site) needs it: the cookie
            never counts there, so a page elsewhere cannot embed the video.
            """
            media = _is_media_path(path)
            signed = query.get(MEDIA_TOKEN_PARAM, [None])[0] if media else None
            how = self._authenticated(signed)
            if how is None:
                self._unauthorized(path)
                return True
            cross_site = self.headers.get("Sec-Fetch-Site") == "cross-site"
            if media and how != "media" and cross_site:
                self._reject("cross-site media request refused", status=403)
                return True
            if how == "media":
                self.resource_policy = "cross-origin"
            return False

        def _reject(self, message: str, status: int = 400) -> None:
            self._html(
                f"<!DOCTYPE html><html><body><p>{escape(message)}</p>"
                f'<p><a href="/">Back</a></p></body></html>',
                status=status,
            )

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
            # No error or progress line may carry the Gemini key out.
            body = json.dumps(redact_payload(payload)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            origin = self.headers.get("Origin")
            if origin and origin.rstrip("/") in allowed_origins(self._port()):
                self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)

        def _not_found(self) -> None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_file(self, path: Path, content_type: str, *, head: bool) -> None:
            """Serve a file with single-range support, streamed in chunks.

            A file removed (or locked, on Windows) since it was found is a 404.
            """
            try:
                opened = path.open("rb")
            except (FileNotFoundError, PermissionError, IsADirectoryError):
                self._not_found()
                return
            with opened as handle:
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
                except OSError:
                    # Players drop a range request as soon as they seek, and
                    # a paused one that stops reading times out.
                    return

        def _media_file(self, path: str) -> tuple[Path, str] | None:
            """Map a media URL to a file inside a job, or ``None``.

            ``/jobs/<id>/...`` names the job; a bare path is the current job's.
            """
            scoped = _JOB_MEDIA.fullmatch(path)
            if scoped is not None:
                try:
                    job_dir = controller.store.job_dir(scoped.group(1))
                except JobError:
                    return None
                if job_dir.is_symlink() or not job_dir.is_dir():
                    return None
                return _job_media(job_dir, scoped.group(2), scoped=True)
            state = controller.current_state()
            if not state.job_id:
                return None
            return _job_media(controller.store.job_dir(state.job_id), path)

        def do_HEAD(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            if self._refused(path, state_changing=False):
                return
            if self._gate(path, parse_qs(parsed.query)):
                return
            try:
                media = self._media_file(path)
            except (WizardError, OSError):
                media = None
            if media is None:
                self._not_found()
                return
            self._send_file(media[0], media[1], head=True)

        def _view(self) -> dict[str, object]:
            # Only sent to a request that already proved it holds the token.
            return {**controller.api_view(), "token": token}

        def _token_refused(self, path: str, supplied: str | None) -> bool:
            if token_matches(self.headers.get(TOKEN_HEADER) or supplied, token):
                return False
            reason = "missing or wrong wizard token; reload the page"
            if path.startswith("/api/"):
                self._json({"error": reason}, status=403)
            else:
                self._reject(reason, status=403)
            return True

        def _read_fields(self) -> dict[str, list[str]]:
            length = content_length(self.headers)
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
            self.send_header(
                "Access-Control-Allow-Headers",
                f"Content-Type, Accept, {TOKEN_HEADER}, X-File-Name",
            )
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def _upload(self) -> None:
            """Stream a dropped video to disk, then pick it.

            The body is the file itself, so it skips the small-body limit
            and gets its own cap; the token must come as a header.
            """
            if not token_matches(self.headers.get(TOKEN_HEADER), token):
                self._json(
                    {"error": "missing or wrong wizard token; reload the page"},
                    status=403,
                )
                return
            raw = (self.headers.get("Content-Length") or "").strip()
            if not raw.isdigit() or int(raw) <= 0:
                self._json({"error": "the upload needs a Content-Length"}, status=411)
                return
            length = int(raw)
            if length > MAX_UPLOAD_BYTES:
                self._json({"error": "the file is too large to upload"}, status=413)
                return
            if controller.busy():
                self._json({"error": "the last step is still running"}, status=409)
                return
            name = unquote(self.headers.get("X-File-Name") or "")
            try:
                dest = controller.upload_target(name)
            except WizardError as exc:
                self._json({"error": str(exc)}, status=415)
                return
            left = length
            try:
                with open_private_new(dest) as out:
                    while left:
                        chunk = self.rfile.read(min(_UPLOAD_CHUNK, left))
                        if not chunk:
                            break
                        out.write(chunk)
                        left -= len(chunk)
            except OSError as exc:
                dest.unlink(missing_ok=True)
                self._json({"error": f"could not save the upload: {exc}"}, status=500)
                return
            if left:
                dest.unlink(missing_ok=True)
                self._json({"error": "the upload stopped before the end"}, status=400)
                return
            controller.pick_upload(dest, name)
            self._json(self._view())

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if self._refused(path, state_changing=True):
                return
            # Checked before the body is read, so no stranger can upload.
            if self._authenticated() is None:
                self._unauthorized(path)
                return
            if path in _UPLOAD_PATHS:
                self._upload()
                return
            try:
                fields = self._read_fields()
            except WizardError as exc:
                status = exc.status if isinstance(exc, BodyError) else 400
                if path.startswith("/api/"):
                    self._json({"error": str(exc)}, status=status)
                    return
                self._reject(str(exc), status=status)
                return
            if self._token_refused(path, fields.pop("token", [""])[0]):
                return
            action = _API_POSTS.get(path, path)
            try:
                controller.handle_post(action, fields)
            except WizardError as exc:
                # 409: the page shows a job the wizard has since left.
                status = 409 if isinstance(exc, WizardConflict) else 400
                if path.startswith("/api/"):
                    self._json({"error": str(exc)}, status=status)
                    return
                self._reject(str(exc), status=status)
                return
            if path.startswith("/api/"):
                self._json(self._view())
                return
            self._redirect("/")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)
            runs = query.get("run", [""])[0] == "1"
            if self._refused(path, state_changing=runs):
                return
            supplied = query.get(LAUNCH_TOKEN_PARAM, [""])[0]
            if runs:
                # A ?run=1 link carries the token itself, cookie or not.
                if self._token_refused(path, supplied):
                    return
            elif LAUNCH_TOKEN_PARAM in query:
                self._launch(path, parsed.query)
                return
            elif self._gate(path, query):
                return
            try:
                if path == "/api/state":
                    self._json(self._view())
                    return
                if path == "/api/jobs":
                    self._json({"jobs": controller.jobs_view()})
                    return
                # Long passes run on the controller's worker thread. These
                # GETs start one when none is running and return at once;
                # callers poll /api/state (or the page refreshes).
                if path == "/api/render" and runs:
                    controller.start_render()
                    self._json(self._view())
                    return
                if path == "/render" and runs:
                    controller.start_render()
                    self._redirect("/")
                    return
                if path == "/api/segment" and runs:
                    controller.start_segmenting()
                    self._json(self._view())
                    return
                if path == "/segment" and runs:
                    controller.start_segmenting()
                    self._redirect("/")
                    return
                if path == "/api/analyze" and runs:
                    controller.start_analyze()
                    self._json(self._view())
                    return
                if path == "/analyze" and runs:
                    controller.start_analyze()
                    self._redirect("/")
                    return
                # Preview the next scene in the background (``--prefetch``
                # only); the view's segment_preview.next shows how it goes.
                if path == "/api/prefetch" and runs:
                    controller.start_prefetch()
                    self._json(self._view())
                    return
                if path == "/":
                    self._html(controller.render())
                    return
                media = self._media_file(path)
                if media is not None:
                    self._send_file(media[0], media[1], head=False)
                    return
                if path.startswith("/api/"):
                    self._json({"error": f"unknown endpoint {path}"}, status=404)
                    return
                self._not_found()
            except (WizardError, ReviewError, OSError) as exc:
                if path.startswith("/api/"):
                    self._json({"error": str(exc)}, status=400)
                    return
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
        prefetch: bool = False,
    ) -> None:
        self.job_dir = job_dir
        # The wizard and the split page both need it on every request.
        self.token: str | None = new_wizard_token()
        self.controller: WizardController | None = None
        wizard_state = load_job_wizard_state(job_dir)
        if wizard_state is not None and wizard_state.step is not WizardStep.INTRO:
            controller = WizardController(
                jobs_dir or job_dir.parent, job_id=job_dir.name, prefetch=prefetch
            )
            self.controller = controller
            handler: type[BaseHTTPRequestHandler] = _wizard_handler(
                controller, self.token
            )
        else:
            handler = _legacy_handler(
                job_dir, jobs_dir=jobs_dir, job_id=job_id, token=self.token
            )
        self._httpd = _LocalHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        name = host.decode() if isinstance(host, bytes) else host
        return f"http://{name}:{port}"

    @property
    def launch_url(self) -> str:
        """The URL to open in a browser; it carries the token once."""
        return launch_url(self.url, self.token or "")

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
        if self.controller is not None:
            self.controller.shutdown()


class WizardServer:
    """Serve the Gemini wizard for a jobs directory."""

    def __init__(
        self,
        jobs_dir: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        controller: WizardController | None = None,
        token: str | None = None,
        prefetch: bool = False,
    ) -> None:
        self.controller = controller or WizardController(jobs_dir, prefetch=prefetch)
        if prefetch:
            self.controller.prefetch = True
        self.token = token or new_wizard_token()
        self._httpd = _LocalHTTPServer(
            (host, port),
            _wizard_handler(self.controller, self.token),
        )
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        name = host.decode() if isinstance(host, bytes) else host
        return f"http://{name}:{port}"

    @property
    def launch_url(self) -> str:
        """The URL to open in a browser; it carries the token once."""
        return launch_url(self.url, self.token)

    def start(self) -> str:
        self._thread.start()
        return self.url

    def wait(self) -> None:
        self._thread.join()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._thread.join(timeout=5)
        self._httpd.server_close()
        self.controller.shutdown()
