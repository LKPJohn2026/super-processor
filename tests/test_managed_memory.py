"""The system-RAM fallback for clips whose shortest piece does not fit."""

from __future__ import annotations

import shutil
import stat
import sys
from pathlib import Path

import pytest
from test_frame_alignment import _fake_flashvsr, _numbered, _numbers, needs_ffmpeg

from super_processor import managed_memory
from super_processor.look import ShotLook
from super_processor.managed_memory import (
    MANAGED_MEMORY_ENV,
    allocator_library,
    request_from_dict,
    request_to_dict,
    upscale_in_managed_memory,
)
from super_processor.regions import Region
from super_processor.upscale import (
    FlashVsrEngine,
    GpuOutOfMemory,
    UpscaleError,
    UpscaleRequest,
)

# The fallback is Linux only, and the fake executables are shell scripts.
linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the fallback is Linux only"
)


def _script(path: Path, body: str) -> Path:
    """An executable that runs ``body`` in this Python.

    A shell wrapper, since a shebang cannot hold the space in a path like
    the project's own.
    """
    code = path.with_suffix(".py")
    code.write_text(body)
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{code}" "$@"\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_the_child_gets_the_whole_request(tmp_path: Path) -> None:
    request = UpscaleRequest(
        tmp_path / "in.mp4",
        tmp_path / "out.mp4",
        scale=4,
        strength=0.3,
        look=ShotLook(denoise=0.2, contrast=1.1),
        regions=(Region("faces", 0.1, 0.2, 0.5, 0.6),),
        on_progress=print,
    )
    back = request_from_dict(request_to_dict(request))
    assert back == request
    assert back.on_progress is None


@linux_only
def test_the_allocator_is_built_once_per_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = tmp_path / "calls"
    nvcc = _script(
        tmp_path / "nvcc",
        "import sys\n"
        f"open({str(calls)!r}, 'a').write('built\\n')\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "open(out, 'wb').write(b'so')\n",
    )
    monkeypatch.setattr(managed_memory, "find_nvcc", lambda: nvcc)
    first = allocator_library(tmp_path / "build")
    again = allocator_library(tmp_path / "build")
    assert first == again and first.read_bytes() == b"so"
    assert first.name.startswith("managed_alloc-") and first.suffix == ".so"
    assert calls.read_text() == "built\n"
    assert list((tmp_path / "build").iterdir()) == [first]


@linux_only
def test_no_compiler_is_a_plain_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(managed_memory, "find_nvcc", lambda: None)
    with pytest.raises(UpscaleError, match="nvcc"):
        allocator_library(tmp_path / "build")


@linux_only
def test_a_failed_build_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nvcc = _script(
        tmp_path / "nvcc",
        "import sys\nprint('error: bad', file=sys.stderr)\nsys.exit(1)\n",
    )
    monkeypatch.setattr(managed_memory, "find_nvcc", lambda: nvcc)
    with pytest.raises(UpscaleError, match="error: bad"):
        allocator_library(tmp_path / "build")
    assert list((tmp_path / "build").iterdir()) == []


# nvcc calls the host C++ compiler from PATH.
@linux_only
@pytest.mark.skipif(
    managed_memory.find_nvcc() is None or shutil.which("c++") is None,
    reason="nvcc and a C++ compiler required",
)
def test_the_real_allocator_compiles(tmp_path: Path) -> None:
    library = allocator_library(tmp_path / "build")
    assert library.stat().st_size > 0


_CHILD = """
import json, os, shutil, sys
assert os.environ[{env!r}] == "1"
assert sys.argv[1:3] == ["-m", "super_processor.managed_memory"]
request = json.loads(sys.stdin.read())
print("FlashVSR's own chatter")
for done in (0.5, 1.0):
    print("@@super-processor progress", done, flush=True)
if {fail!r}:
    print("@@super-processor error GpuOutOfMemory out of RAM too", flush=True)
    sys.exit(1)
shutil.copy(request["source"], request["output"])
"""


def _child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail: bool) -> None:
    library = tmp_path / "managed_alloc.so"
    library.write_bytes(b"so")
    monkeypatch.setattr(
        managed_memory, "allocator_library", lambda build_dir=None: library
    )
    body = _CHILD.format(env=MANAGED_MEMORY_ENV, fail=fail)
    python = _script(tmp_path / "python", body)
    monkeypatch.setattr(sys, "executable", str(python))


@linux_only
def test_the_child_restores_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _child(tmp_path, monkeypatch, fail=False)
    source = tmp_path / "in.mp4"
    source.write_bytes(b"clip")
    seen: list[float] = []
    out = upscale_in_managed_memory(
        UpscaleRequest(source, tmp_path / "out.mp4", on_progress=seen.append)
    )
    assert out.read_bytes() == b"clip"
    assert seen == [0.5, 1.0]


@linux_only
def test_the_childs_error_comes_back_as_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _child(tmp_path, monkeypatch, fail=True)
    with pytest.raises(GpuOutOfMemory, match="out of RAM too"):
        upscale_in_managed_memory(
            UpscaleRequest(tmp_path / "in.mp4", tmp_path / "out.mp4")
        )


@needs_ffmpeg
def test_the_shortest_piece_out_of_memory_goes_to_system_ram(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_flashvsr(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MAX_FRAMES", "5")
    handed: list[UpscaleRequest] = []

    def in_ram(request: UpscaleRequest, **_kwargs: object) -> Path:
        handed.append(request)
        shutil.copy(request.source, request.output)
        return request.output

    monkeypatch.setattr(managed_memory, "upscale_in_managed_memory", in_ram)
    source = _numbered(tmp_path / "in.mp4", 60)
    output = FlashVsrEngine().upscale(
        UpscaleRequest(source, tmp_path / "out.mp4", scale=2)
    )
    assert len(handed) == 1 and handed[0].source == source.resolve()
    assert _numbers(output) == _numbers(source)


@needs_ffmpeg
def test_the_child_does_not_start_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_flashvsr(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MAX_FRAMES", "5")
    monkeypatch.setenv(MANAGED_MEMORY_ENV, "1")
    source = _numbered(tmp_path / "in.mp4", 60)
    with pytest.raises(GpuOutOfMemory, match="system RAM"):
        FlashVsrEngine().upscale(UpscaleRequest(source, tmp_path / "out.mp4", scale=2))


@needs_ffmpeg
def test_a_failed_fallback_still_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The conftest fallback refuses, as a missing nvcc would.
    _fake_flashvsr(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_MAX_FRAMES", "5")
    source = _numbered(tmp_path / "in.mp4", 60)
    with pytest.raises(GpuOutOfMemory, match="Close other programs that use the GPU"):
        FlashVsrEngine().upscale(UpscaleRequest(source, tmp_path / "out.mp4", scale=2))


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="checks other systems")
def test_other_systems_say_it_needs_linux(tmp_path: Path) -> None:
    with pytest.raises(UpscaleError, match="needs Linux"):
        allocator_library(tmp_path / "build")
