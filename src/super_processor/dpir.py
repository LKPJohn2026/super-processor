"""DPIR: a stronger, GPU denoise that a scene gets only when the editor asks.

DPIR (Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser
Prior") denoises with DRUNet, a U-Net that is told how much noise to remove:
its fourth input channel is a flat map of the noise level. KAIR publishes the
colour weights (``drunet_color.pth``, MIT licence) and spandrel (MIT) builds
the network from them.

It is never Gemini's choice. A scene's level (0 to :data:`MAX_LEVEL`) moves
only when the editor's own note says so, read by :func:`note_level`: "more
denoise" or "still noisy" raises it by one, "less denoise" lowers it, and
"no DPIR" turns it off. The level lives on the scene's recipe beside the
editor's other choices, so a new Gemini proposal keeps it.

Where it runs, it replaces the spatial half of the look's FFmpeg
``hqdn3d``: the decoded frames go through DRUNet on the GPU before SPAN and
before the lanczos base of the strength blend, so both halves of the blend
are cleaned the same way. The temporal half of ``hqdn3d`` stays, because
DRUNet cleans each frame on its own and does not smooth flicker between
frames. Without an NVIDIA GPU (a pass-through scene on a machine with
none), the scene gets the strongest ``hqdn3d`` instead, and the wizard says
so.

The weights (about 130 MB) are fetched the first time a scene needs them,
from one pinned URL, over HTTPS, from GitHub's hosts only, and kept only if
their size and SHA-256 match. ``SUPER_PROCESSOR_DPIR_WEIGHTS`` points at a
copy of the file instead, and ``SUPER_PROCESSOR_DPIR_DOWNLOAD=0`` turns the
download off. torch and spandrel are imported only when DPIR runs.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import IO, Any

from ._dist import GPU_INSTALL_HINT
from .look import ShotLook

MAX_LEVEL = 3
# DRUNet's noise level per DPIR level, on the 0-255 scale it was trained on.
# 15 is KAIR's (and spandrel's) default; one step either side of it.
_SIGMAS = {1: 10.0, 2: 15.0, 3: 25.0}
# What a scene without a GPU gets instead: hqdn3d at its strongest.
FALLBACK_DENOISE = 1.0

DPIR_DIR = "DPIR"
DPIR_FILE = "drunet_color.pth"
DPIR_SHA256 = "479abe3c5327dfd10ff54a80ec7d4098ca80752a5c9492cdff31cee430bec4b4"
DPIR_SIZE = 130_579_305
DPIR_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/" + DPIR_FILE
DPIR_WEIGHTS_ENV = "SUPER_PROCESSOR_DPIR_WEIGHTS"
DPIR_DOWNLOAD_ENV = "SUPER_PROCESSOR_DPIR_DOWNLOAD"
# The release asset redirects once, to GitHub's asset storage.
_DOWNLOAD_HOSTS = frozenset(
    {
        "github.com",
        "release-assets.githubusercontent.com",
        "objects.githubusercontent.com",
    }
)
_DOWNLOAD_TIMEOUT_S = 60.0
# The whole download, however it trickles in: about 70 KB a second.
_DOWNLOAD_DEADLINE_S = 30 * 60.0
_MAX_REDIRECTS = 5
_CHUNK = 1 << 20
# DRUNet halves the picture three times: sides a multiple of 8.
_MULTIPLE = 8
# Frames per DRUNet call: it is far heavier than SPAN per pixel (64-512
# channels at up to full size). On an RTX 5090 at 1440x1080 one frame took
# 1.3 GB at 17 frames a second, two 2.5 GB at 22, four no faster.
_BATCH_PIXELS = 3_000_000
_MAX_BATCH = 4


class DpirError(RuntimeError):
    """The DPIR weights or runtime are missing or wrong."""


def level_of(raw: Any) -> int:
    """A stored level, or 0 for anything that is not a whole number in range."""
    if isinstance(raw, bool) or not isinstance(raw, int):
        return 0
    return raw if 0 <= raw <= MAX_LEVEL else 0


def sigma(level: int) -> float:
    """DRUNet's noise level for ``level`` (1 to :data:`MAX_LEVEL`), 0-1 scale."""
    if level not in _SIGMAS:
        raise DpirError(f"DPIR level must be 1 to {MAX_LEVEL}, not {level!r}")
    return _SIGMAS[level] / 255.0


# A note is read clause by clause, as the size in a note is (see
# :func:`super_processor.shot_plan.note_scale`); the last clause that names
# DPIR or the amount of denoising decides, so a note moves one step at most.
_CLAUSE = re.compile(r"[,;!?\n]|[.:](?!\d)|\b(?:and|but)\b")
_NEGATION = re.compile(r"\b(?:not|never|without|don't|dont|doesn't|isn't)\b")
_NOISE = r"(?:noise|noisy|grain|grainy|speckles?)"
_DENOISE = r"(?:denois\w*|noise\s+reduction)"
# Checked in this order in each clause.
_OFF = re.compile(
    rf"\b(?:no|without|stop|skip)\s+(?:the\s+)?(?:dpir|{_DENOISE})\b"
    r"|\bdpir\s+off\b|\bturn\s+off\s+(?:the\s+)?dpir\b|\bdon'?t\s+denois"
)
_HOLD = re.compile(rf"\bno\s+more\s+{_DENOISE}")
_LESS = re.compile(
    rf"\bless\s+{_DENOISE}|\b{_DENOISE}\s+less\b|\btoo\s+much\s+{_DENOISE}"
    r"|\btoo\s+(?:smooth|plastic|waxy)\b|\bover-?denois"
)
_MORE = re.compile(
    rf"\bmore\s+{_DENOISE}|\b{_DENOISE}\s+more\b|\bstronger\s+{_DENOISE}"
    rf"|\b(?:still|too|very)\s+{_NOISE}"
    rf"|\btoo\s+much\s+{_NOISE}|\b(?:less|fewer)\s+(?:noise|speckles?)\b"
    rf"|\b(?:reduce|remove|clean\s+up|get\s+rid\s+of)\s+(?:the\s+|more\s+)?{_NOISE}"
    r"|\b(?:use|try|turn\s+on|with)\s+(?:the\s+)?dpir\b|\bdpir\s+on\b"
)


def note_level(note: str, current: int) -> int:
    """The scene's DPIR level after the editor's ``note``.

    Only the note's words decide. Asking for more denoise raises the level by
    one (up to :data:`MAX_LEVEL`), less lowers it by one, and "no DPIR" or
    "no denoise" turns it off; a note that says none of these keeps it.
    One note moves the level one step at most, however many ways it asks:
    "still noisy, more denoise" is one request. After an "off", the last
    clause that asks for more, less or no more decides the step.
    "Less grain" is not a request for DPIR: grain is also the look's own
    finishing grain, which Gemini sets.
    """
    level = level_of(current)
    step = 0
    text = note.lower().replace("’", "'")
    for clause in _CLAUSE.split(text):
        if _OFF.search(clause):
            level, step = 0, 0
        elif _HOLD.search(clause):
            step = 0
        elif _LESS.search(clause):
            step = -1
        elif _MORE.search(clause) and not _NEGATION.search(clause):
            step = 1
    return max(0, min(MAX_LEVEL, level + step))


def ffmpeg_look(look: ShotLook, level: int) -> ShotLook:
    """The look FFmpeg applies when DPIR cannot run: hqdn3d at its strongest."""
    return replace(look, denoise=FALLBACK_DENOISE) if level else look


def gpu_clean_filter(look: ShotLook, level: int) -> str:
    """The decoder's clean step when DPIR runs next on the GPU.

    DRUNet cleans each frame on its own, which hqdn3d's spatial half barely
    does, but it does not smooth flicker between frames. So the decoder keeps
    the look's deblock and hqdn3d's temporal half, and drops only its spatial
    half, which DPIR replaces.
    """
    return look.clean_filter(spatial=not level)


def weights_path(models_dir: Path) -> Path:
    """``SUPER_PROCESSOR_DPIR_WEIGHTS``, or ``<models>/DPIR/drunet_color.pth``."""
    raw = os.environ.get(DPIR_WEIGHTS_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    return models_dir / DPIR_DIR / DPIR_FILE


def download_allowed() -> bool:
    return os.environ.get(DPIR_DOWNLOAD_ENV, "").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_CHUNK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _matches(path: Path) -> bool:
    try:
        return path.stat().st_size == DPIR_SIZE and _sha256_file(path) == DPIR_SHA256
    except OSError:
        return False


_DOWNLOAD_LOCK = threading.Lock()
# How often a render waiting for another render's download checks its Stop.
_LOCK_POLL_S = 0.5


@contextlib.contextmanager
def _download_lock(should_stop: Callable[[], None] | None) -> Any:
    """Hold :data:`_DOWNLOAD_LOCK`, calling ``should_stop`` while waiting.

    Another render may be downloading for up to :data:`_DOWNLOAD_DEADLINE_S`;
    a Stop on this one must not wait for that to end.
    """
    while not _DOWNLOAD_LOCK.acquire(timeout=_LOCK_POLL_S):
        if should_stop is not None:
            should_stop()
    try:
        yield
    finally:
        _DOWNLOAD_LOCK.release()


def resolve_weights(
    models_dir: Path,
    *,
    opener: Callable[[str], IO[bytes]] | None = None,
    should_stop: Callable[[], None] | None = None,
) -> Path:
    """The weights file, checked by size and hash, fetched first if missing.

    A file from ``SUPER_PROCESSOR_DPIR_WEIGHTS`` is never replaced or
    downloaded over: if it is wrong, that is an error. Raises
    :class:`DpirError` with what to do when the file cannot be had.
    ``should_stop`` is called between downloaded chunks, and while waiting
    for another render's download; whatever it raises
    (the render's :class:`~super_processor.cancel.Cancelled`) ends the
    download, keeping nothing. The whole download has
    :data:`_DOWNLOAD_DEADLINE_S` to finish.
    """
    path = weights_path(models_dir)
    pinned = bool(os.environ.get(DPIR_WEIGHTS_ENV, "").strip())
    with _download_lock(should_stop):
        if path.is_file():
            if _matches(path):
                return path
            if pinned or not download_allowed():
                raise DpirError(
                    f"{path} is not the expected DPIR weights ({DPIR_FILE}, "
                    f"sha256 {DPIR_SHA256[:12]}...). Download {DPIR_URL} again."
                )
        elif pinned:
            raise DpirError(
                f"DPIR weights were not found at {path} ({DPIR_WEIGHTS_ENV})."
            )
        elif not download_allowed():
            raise DpirError(
                f"DPIR weights were not found at {path}, and {DPIR_DOWNLOAD_ENV} "
                f"turns the download off. Download {DPIR_URL} (MIT) there."
            )
        _download(path, opener or _open_pinned, should_stop)
    return path


class _GitHubOnly(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only over HTTPS to one of GitHub's hosts."""

    max_redirections = _MAX_REDIRECTS

    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Any:
        _require_github(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _require_github(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or (parts.hostname or "") not in _DOWNLOAD_HOSTS:
        raise DpirError(
            f"refusing to fetch the DPIR weights from {parts.scheme}://"
            f"{parts.hostname or ''}: only GitHub over HTTPS is allowed"
        )


def _open_pinned(url: str) -> IO[bytes]:  # pragma: no cover - needs network
    _require_github(url)
    opener = urllib.request.build_opener(_GitHubOnly())
    request = urllib.request.Request(url, headers={"User-Agent": "super-processor"})
    response: IO[bytes] = opener.open(request, timeout=_DOWNLOAD_TIMEOUT_S)
    final = getattr(response, "geturl", lambda: url)()
    _require_github(final)
    return response


def _download(
    path: Path,
    opener: Callable[[str], IO[bytes]],
    should_stop: Callable[[], None] | None = None,
) -> None:
    """Fetch :data:`DPIR_URL` into ``path``; only a file with the pinned size
    and hash is ever put there (an atomic rename of a private temp file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    hasher = hashlib.sha256()
    got = 0
    deadline = time.monotonic() + _DOWNLOAD_DEADLINE_S
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as out:
            try:
                response = opener(DPIR_URL)
            except (OSError, urllib.error.URLError) as exc:
                raise DpirError(_offline(path, exc)) from exc
            with contextlib.closing(response):
                while True:
                    if should_stop is not None:
                        should_stop()
                    if time.monotonic() > deadline:
                        raise DpirError(
                            _offline(path, TimeoutError("the download took too long"))
                        )
                    try:
                        block = response.read(_CHUNK)
                    except OSError as exc:
                        raise DpirError(_offline(path, exc)) from exc
                    if not block:
                        break
                    got += len(block)
                    if got > DPIR_SIZE:
                        raise DpirError(
                            f"the DPIR download from {DPIR_URL} is larger than "
                            f"{DPIR_SIZE} bytes; nothing was kept"
                        )
                    hasher.update(block)
                    out.write(block)
        if got != DPIR_SIZE or hasher.hexdigest() != DPIR_SHA256:
            raise DpirError(
                f"the DPIR download from {DPIR_URL} is not the expected file "
                f"({got} bytes, sha256 {hasher.hexdigest()[:12]}...); nothing "
                "was kept"
            )
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _offline(path: Path, exc: BaseException) -> str:
    return (
        f"could not download the DPIR weights ({exc}). Download {DPIR_URL} "
        f"(MIT) to {path}, or set {DPIR_WEIGHTS_ENV} to a copy of it."
    )


def read_verified(path: Path) -> bytes:
    """The weights' bytes, read once and checked: the bytes loaded are the
    bytes hashed, so a file swapped after :func:`resolve_weights` is never
    loaded."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise DpirError(f"cannot read the DPIR weights at {path}: {exc}") from exc
    if len(data) != DPIR_SIZE or hashlib.sha256(data).hexdigest() != DPIR_SHA256:
        raise DpirError(
            f"{path} is not the expected DPIR weights; delete it to fetch it again"
        )
    return data


def load_state(torch: Any, weights: Path) -> Any:
    """The DRUNet state dict from the verified bytes, tensors only."""
    data = read_verified(weights)
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def import_gpu_modules() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        spandrel = importlib.import_module("spandrel")
    except ImportError as exc:
        raise DpirError(
            "DPIR needs PyTorch and spandrel. Install the gpu extra "
            f"({GPU_INSTALL_HINT})."
        ) from exc
    return torch, spandrel


def load_dpir(weights: Path) -> Any:  # pragma: no cover - needs CUDA and torch
    """DRUNet on the GPU in half precision, ready for :func:`denoise`."""
    torch, spandrel = import_gpu_modules()
    described = spandrel.ModelLoader().load_from_state_dict(load_state(torch, weights))
    if type(described.model).__name__ != "DRUNet" or described.scale != 1:
        raise DpirError(f"{weights.name} is not a DRUNet denoiser")
    model = described.model.eval()
    return model.cuda().half().to(memory_format=torch.channels_last)


def batch_size(width: int, height: int) -> int:
    """Frames per DRUNet call for a ``width`` x ``height`` picture."""
    pixels = max(1, width * height)
    return max(1, min(_MAX_BATCH, round(_BATCH_PIXELS / pixels)))


def denoise(model: Any, rgb: Any, level: int) -> Any:  # pragma: no cover - torch
    """Float RGB frames (N x 3 x h x w, 0-1) through DRUNet at ``level``.

    The sides are padded (edges repeated) to a multiple of 8 and cropped
    back, so any frame size works. Runs in the model's precision; returns
    float32 in 0-1.
    """
    torch = importlib.import_module("torch")
    f = importlib.import_module("torch.nn.functional")
    count, _channels, height, width = rgb.shape
    pad_h, pad_w = -height % _MULTIPLE, -width % _MULTIPLE
    x = rgb
    if pad_h or pad_w:
        x = f.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    dtype = next(model.parameters()).dtype
    noise = torch.full(
        (count, 1, x.shape[-2], x.shape[-1]), sigma(level), device=x.device, dtype=dtype
    )
    x = torch.cat([x.to(dtype), noise], 1).contiguous(memory_format=torch.channels_last)
    del noise
    out = model(x)[..., :height, :width]
    return out.float().clamp_(0.0, 1.0)
