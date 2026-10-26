"""FlashVSR in unified GPU memory, for clips that do not fit in free VRAM.

The normal path retries a clip that runs out of GPU memory in shorter pieces
(see :class:`~super_processor.upscale.FlashVsrEngine`). When even the shortest
piece does not fit, the clip is restored once more in a child process whose
PyTorch allocator hands out CUDA managed memory: blocks stay on the GPU while
there is room and the driver pages the rest to system RAM. Measured at 2x of
854x480 with 16, 12, and 8 GiB free, every clip finished at 0.43-0.70 s a
frame against 0.27-0.30 s normally, with the same picture.

It runs in a child because PyTorch only lets the allocator change before CUDA
has started in a process, and the wizard has started it long before. The
allocator (``managed_alloc.cu``) is built with ``nvcc`` on first use. Linux
only: on Windows, managed memory cannot be larger than free VRAM.

System RAM is bounded three ways. The child works in the shortest pieces,
since FlashVSR's memory grows with the frames in one call. The allocator
refuses past a byte limit: the free VRAM plus a share of the RAM that is
available when the child starts (see :func:`memory_limit`). And the parent
stops the child if the RAM still available falls below half the reserve
left for everything else. Without these, one 12 s preview next to a game
paged 80 GiB into RAM and the kernel began killing programs.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .look import ShotLook
from .regions import regions_from_dicts

if TYPE_CHECKING:
    from .upscale import UpscaleRequest

# Set in the child, so a clip that runs out of memory there is not handed to
# yet another child.
MANAGED_MEMORY_ENV = "SUPER_PROCESSOR_MANAGED_MEMORY"
# Lines the child writes for the parent start with this; FlashVSR prints its
# own lines to the same stdout.
_MARK = "@@super-processor "
_SOURCE = Path(__file__).with_name("managed_alloc.cu")
# The allocator's byte limit, read by managed_alloc.cu.
LIMIT_ENV = "SUPER_PROCESSOR_MANAGED_LIMIT"
_GIB = 1 << 30
# RAM left for everything else: a quarter of it, and never under 8 GiB.
RAM_RESERVE_SHARE = 0.25
RAM_RESERVE_MIN = 8 * _GIB
# The child never takes more than half the machine's RAM.
RAM_SHARE_MAX = 0.5
# Below this much RAM to spare, the child is not started at all.
RAM_SHARE_MIN = 2 * _GIB
WATCH_INTERVAL_S = 0.5


@dataclass(frozen=True, slots=True)
class MemoryLimit:
    """What the child may hold, and when the parent stops it."""

    limit: int  # bytes the allocator may take from the driver
    floor: int  # stop the child when MemAvailable falls below this


def read_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    """``/proc/meminfo`` in bytes."""
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            scale = 1024 if parts[1:] == ["kB"] else 1
            values[name] = int(parts[0]) * scale
    return values


def memory_limit(free_vram: int, meminfo: dict[str, int]) -> MemoryLimit:
    """The child's byte limit: free VRAM plus RAM the rest can spare.

    A reserve of RAM is left for everything else (the desktop, a game, the
    wizard itself); of what is available past it, the child takes at most
    half the machine. Too little to spare raises, so no child is started.
    """
    from .upscale import GpuOutOfMemory

    total = meminfo.get("MemTotal", 0)
    available = meminfo.get("MemAvailable", 0)
    reserve = max(RAM_RESERVE_MIN, int(total * RAM_RESERVE_SHARE))
    share = min(available - reserve, int(total * RAM_SHARE_MAX))
    if share < RAM_SHARE_MIN:
        raise GpuOutOfMemory(
            "there is not enough free system RAM to fall back on "
            f"({available / _GIB:.1f} GiB available, {reserve / _GIB:.1f} GiB "
            "kept for other programs)"
        )
    return MemoryLimit(limit=max(free_vram, 0) + share, floor=reserve // 2)


def _free_vram() -> int:
    """Free GPU memory as this process sees it; 0 without CUDA."""
    torch = sys.modules.get("torch")
    try:
        return int(torch.cuda.mem_get_info()[0]) if torch is not None else 0
    except Exception:  # noqa: BLE001 - no GPU means no VRAM to count
        return 0


def in_managed_memory() -> bool:
    """True inside the child that runs FlashVSR in managed memory."""
    return os.environ.get(MANAGED_MEMORY_ENV) == "1"


def find_nvcc() -> Path | None:
    """The CUDA compiler: ``CUDA_HOME``, ``PATH``, then the usual installs."""
    candidates: list[Path] = []
    for name in ("CUDA_HOME", "CUDA_PATH"):
        raw = os.environ.get(name, "").strip()
        if raw:
            candidates.append(Path(raw) / "bin" / "nvcc")
    found = shutil.which("nvcc")
    if found:
        candidates.append(Path(found))
    candidates += [Path("/opt/cuda/bin/nvcc"), Path("/usr/local/cuda/bin/nvcc")]
    return next((path for path in candidates if path.is_file()), None)


def allocator_library(build_dir: Path | None = None) -> Path:
    """The built allocator, compiling ``managed_alloc.cu`` when needed.

    The file name carries a hash of the source, so an edited allocator is
    built again rather than an old one loaded.
    """
    from .upscale import UpscaleError, models_dir

    if not sys.platform.startswith("linux"):
        # Windows drivers give managed memory no demand paging, so it cannot
        # grow past free VRAM; macOS has no CUDA.
        raise UpscaleError("falling back to system RAM needs Linux")
    source = _SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:12]
    folder = build_dir or models_dir() / "managed-alloc"
    library = folder / f"managed_alloc-{digest}.so"
    if library.is_file():
        return library
    nvcc = find_nvcc()
    if nvcc is None:
        raise UpscaleError(
            "the CUDA compiler (nvcc) was not found, so FlashVSR cannot fall "
            "back to system RAM. Install the CUDA toolkit or set CUDA_HOME."
        )
    folder.mkdir(parents=True, exist_ok=True)
    partial = library.with_suffix(f".{os.getpid()}.tmp")
    completed = subprocess.run(
        [str(nvcc), "-O2", "-shared", "-Xcompiler", "-fPIC"]
        + ["-o", str(partial), str(_SOURCE)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise UpscaleError(
            "could not build the system-RAM allocator for FlashVSR"
            + (f": {detail[-1]}" if detail else "")
        )
    partial.replace(library)
    return library


def request_to_dict(request: UpscaleRequest) -> dict[str, Any]:
    """What the child needs to rebuild ``request``; the callback stays here."""
    return {
        "source": str(request.source),
        "output": str(request.output),
        "scale": request.scale,
        "strength": request.strength,
        "look": request.look.to_dict(),
        "regions": [region.to_dict() for region in request.regions],
    }


def request_from_dict(data: dict[str, Any]) -> UpscaleRequest:
    from .upscale import UpscaleRequest

    return UpscaleRequest(
        source=Path(data["source"]),
        output=Path(data["output"]),
        scale=data["scale"],
        strength=data["strength"],
        look=ShotLook.from_dict(data["look"]),
        regions=regions_from_dicts(data["regions"]),
    )


def upscale_in_managed_memory(
    request: UpscaleRequest,
    *,
    chunk_s: float,
    build_dir: Path | None = None,
    meminfo: Path = Path("/proc/meminfo"),
) -> Path:
    """Restore ``request`` in a child process that uses managed GPU memory.

    The child works in pieces of ``chunk_s`` under a byte limit, and is
    stopped if available RAM falls below the floor anyway. Progress lines
    from the child reach ``request.on_progress``; a failure in the child is
    raised here with its message.
    """
    from .upscale import GpuOutOfMemory, UpscaleError

    library = allocator_library(build_dir)
    bounds = memory_limit(_free_vram(), read_meminfo(meminfo))
    env = dict(os.environ, **{MANAGED_MEMORY_ENV: "1", LIMIT_ENV: str(bounds.limit)})
    failure: tuple[str, str] | None = None
    low_ram = threading.Event()
    done = threading.Event()
    with subprocess.Popen(
        [sys.executable, "-m", "super_processor.managed_memory", str(library)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    ) as child:
        watch = threading.Thread(
            target=_watch_ram,
            args=(child, bounds.floor, meminfo, low_ram, done),
            daemon=True,
        )
        watch.start()
        try:
            assert child.stdin is not None and child.stdout is not None
            child.stdin.write(
                json.dumps({**request_to_dict(request), "chunk_s": chunk_s})
            )
            child.stdin.close()
            for line in child.stdout:
                if not line.startswith(_MARK):
                    continue
                kind, _, rest = line[len(_MARK) :].rstrip("\n").partition(" ")
                if kind == "progress" and request.on_progress is not None:
                    request.on_progress(float(rest))
                elif kind == "error":
                    name, _, message = rest.partition(" ")
                    failure = (name, message)
        finally:
            done.set()
            watch.join()
    code = child.returncode
    if low_ram.is_set():
        raise GpuOutOfMemory(
            "FlashVSR was stopped because system RAM ran low "
            f"(under {bounds.floor / _GIB:.1f} GiB available)"
        )
    if failure is not None:
        name, message = failure
        raise (GpuOutOfMemory if name == "GpuOutOfMemory" else UpscaleError)(message)
    if code != 0:
        raise UpscaleError(f"the FlashVSR system-RAM run stopped (exit code {code})")
    if not request.output.is_file():
        raise UpscaleError("FlashVSR produced no output")
    return request.output


def _watch_ram(
    child: subprocess.Popen[str],
    floor: int,
    meminfo: Path,
    low_ram: threading.Event,
    done: threading.Event,
) -> None:
    """Kill ``child`` once available RAM falls below ``floor``.

    The allocator's limit covers the blocks it hands out, not what the driver
    and the rest of the system need around them; this is the last guard.
    Killing the child returns every managed page at once.
    """
    while not done.wait(WATCH_INTERVAL_S):
        try:
            available = read_meminfo(meminfo).get("MemAvailable")
        except OSError:
            continue
        if available is not None and available < floor:
            low_ram.set()
            child.kill()
            return


def _report(kind: str, text: str) -> None:
    print(f"{_MARK}{kind} {text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Child entry: switch the allocator before CUDA starts, then restore."""
    args = sys.argv[1:] if argv is None else argv
    # torch is the optional gpu extra, so it is not imported by name.
    torch = importlib.import_module("torch")

    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        args[0], "managed_malloc", "managed_free"
    )
    torch.cuda.memory.change_current_allocator(allocator)
    from dataclasses import replace

    from .upscale import FlashVsrEngine, UpscaleError

    data = json.loads(sys.stdin.read())
    request = replace(
        request_from_dict(data),
        on_progress=lambda done: _report("progress", f"{done:.4f}"),
    )
    try:
        FlashVsrEngine(chunk_s=float(data["chunk_s"])).upscale(request)
    except UpscaleError as exc:
        first = str(exc).splitlines()[0] if str(exc) else ""
        _report("error", f"{type(exc).__name__} {first}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
