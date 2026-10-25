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
allocator (``managed_alloc.cu``) is built with ``nvcc`` on first use.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
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
    request: UpscaleRequest, *, build_dir: Path | None = None
) -> Path:
    """Restore ``request`` in a child process that uses managed GPU memory.

    Progress lines from the child reach ``request.on_progress``; a failure in
    the child is raised here with its message.
    """
    from .upscale import GpuOutOfMemory, UpscaleError

    library = allocator_library(build_dir)
    env = dict(os.environ, **{MANAGED_MEMORY_ENV: "1"})
    failure: tuple[str, str] | None = None
    with subprocess.Popen(
        [sys.executable, "-m", "super_processor.managed_memory", str(library)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=env,
    ) as child:
        assert child.stdin is not None and child.stdout is not None
        child.stdin.write(json.dumps(request_to_dict(request)))
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
    code = child.returncode
    if failure is not None:
        name, message = failure
        raise (GpuOutOfMemory if name == "GpuOutOfMemory" else UpscaleError)(message)
    if code != 0:
        raise UpscaleError(f"the FlashVSR system-RAM run stopped (exit code {code})")
    if not request.output.is_file():
        raise UpscaleError("FlashVSR produced no output")
    return request.output


def _report(kind: str, text: str) -> None:
    print(f"{_MARK}{kind} {text}", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Child entry: switch the allocator before CUDA starts, then restore."""
    args = sys.argv[1:] if argv is None else argv
    import torch

    allocator = torch.cuda.memory.CUDAPluggableAllocator(
        args[0], "managed_malloc", "managed_free"
    )
    torch.cuda.memory.change_current_allocator(allocator)
    from dataclasses import replace

    from .upscale import FlashVsrEngine, UpscaleError

    request = replace(
        request_from_dict(json.loads(sys.stdin.read())),
        on_progress=lambda done: _report("progress", f"{done:.4f}"),
    )
    try:
        FlashVsrEngine().upscale(request)
    except UpscaleError as exc:
        first = str(exc).splitlines()[0] if str(exc) else ""
        _report("error", f"{type(exc).__name__} {first}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
