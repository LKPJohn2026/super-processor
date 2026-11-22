"""The SPAN weights are deserialized from the very bytes that were hashed,
torch is new enough that ``weights_only`` holds, and every install hint names
the distribution actually published on PyPI.

No torch is needed: a stand-in ``torch.load`` records what it is given.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import re
from pathlib import Path
from typing import Any

import pytest

from super_processor import _dist, span_sr, upscale
from super_processor.span_sr import SpanError

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = (ROOT / "pyproject.toml").read_text("utf-8")
GOOD = b"the real SPAN weights, for this test"
BAD = b"swapped in after the hash check"


class _FakeTorch:
    def __init__(self, on_load: Any = None) -> None:
        self.calls: list[tuple[bytes, dict[str, Any]]] = []
        self.on_load = on_load

    def load(self, source: Any, **kwargs: Any) -> Any:
        assert isinstance(source, io.BytesIO), "torch.load must get the bytes"
        if self.on_load is not None:
            self.on_load()
        self.calls.append((source.getvalue(), kwargs))
        return {"params_ema": {"w": 1}}


@pytest.fixture
def weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "models" / span_sr.SPAN_DIR / span_sr.SPAN_FILE
    path.parent.mkdir(parents=True)
    path.write_bytes(GOOD)
    monkeypatch.setattr(span_sr, "SPAN_SHA256", hashlib.sha256(GOOD).hexdigest())
    monkeypatch.delenv(span_sr.SPAN_WEIGHTS_ENV, raising=False)
    return path


def test_swap_after_the_early_check_is_refused_before_loading(weights: Path) -> None:
    # Finding 17: resolve_span_weights passes, then the file is replaced
    # (e.g. while the render waits for the GPU lock).
    checked = span_sr.resolve_span_weights(weights.parents[1])
    weights.write_bytes(BAD)
    torch = _FakeTorch()
    with pytest.raises(SpanError, match="not the expected SPAN weights"):
        span_sr.load_state(torch, checked)
    assert torch.calls == []


def test_load_uses_the_hashed_bytes_not_the_path(weights: Path) -> None:
    # A swap during deserialization changes nothing: torch never reopens it.
    torch = _FakeTorch(on_load=lambda: weights.write_bytes(BAD))
    state = span_sr.load_state(torch, weights)
    assert state == {"w": 1}
    [(data, kwargs)] = torch.calls
    assert hashlib.sha256(data).hexdigest() == span_sr.SPAN_SHA256
    assert kwargs == {"map_location": "cpu", "weights_only": True}


def test_unwrapped_state_and_unreadable_file(weights: Path) -> None:
    class Plain(_FakeTorch):
        def load(self, source: Any, **kwargs: Any) -> Any:
            super().load(source, **kwargs)
            return {"w": 2}

    assert span_sr.load_state(Plain(), weights) == {"w": 2}
    with pytest.raises(SpanError, match="cannot read"):
        span_sr.read_verified_weights(weights.parent / "missing.pth")


def _floor(requirement: str) -> tuple[int, ...]:
    match = re.search(r">=\s*([0-9.]+)", requirement)
    assert match, requirement
    return tuple(int(part) for part in match.group(1).split("."))


def test_gpu_extra_requires_a_torch_without_the_weights_only_bypass() -> None:
    # CVE-2025-32434: torch < 2.6 can run code from a "weights_only" load.
    # Read by hand: tomllib is not in Python 3.10, which CI still runs.
    block = re.search(r"^gpu = \[(.*?)^\]", PYPROJECT, re.M | re.S)
    assert block
    [torch_req] = re.findall(r'^\s*"(torch\b[^"]*)"', block.group(1), re.M)
    assert _floor(torch_req) >= (2, 6)


def test_dist_name_is_the_published_project_name() -> None:
    name = re.search(r'^\[project\]\s*^name = "([^"]+)"', PYPROJECT, re.M)
    assert name
    assert _dist.DIST_NAME == name.group(1) == "ancient-processor"
    assert _dist.GPU_INSTALL_HINT == 'python -m pip install "ancient-processor[gpu]"'


def test_missing_torch_hint_names_ancient_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Finding 19: the hint once said 'super-processor[gpu]', another
    # PyPI name entirely.
    real = importlib.import_module

    def fake(name: str, *args: Any) -> Any:
        if name in {"torch", "spandrel"}:
            raise ImportError(name)
        return real(name, *args)

    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(SpanError) as caught:
        span_sr.load_span(Path("x.pth"))
    message = str(caught.value)
    assert 'pip install "ancient-processor[gpu]"' in message
    assert "super-processor[" not in message


def test_no_gpu_hint_names_ancient_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upscale, "cuda_available", lambda: False)
    with pytest.raises(upscale.UpscaleError) as caught:
        upscale._span_ready(2)
    assert '"ancient-processor[gpu]"' in str(caught.value)


_PIP_NAME = re.compile(r"pip install\s+(?:-[-\w]+\s+)*['\"]?([A-Za-z0-9][\w.-]*)")


def test_every_pip_install_line_names_the_published_distribution() -> None:
    files = [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]
    for folder, pattern in (("src/super_processor", "*.py"), ("web/src", "*.ts*")):
        files.extend((ROOT / folder).rglob(pattern))
    offenders = []
    for path in files:
        text = path.read_text("utf-8")
        text = text.replace("{DIST_NAME}", _dist.DIST_NAME)
        for name in _PIP_NAME.findall(text):
            if name != _dist.DIST_NAME:
                offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert offenders == []
