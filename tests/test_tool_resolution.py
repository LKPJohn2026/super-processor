"""Outside programs are found on PATH, never in the current folder.

On Windows a program started by bare name is looked for in the folder the
terminal is open in before PATH, so an ``ffmpeg.exe`` left in a downloaded
folder would run instead of the real one. Every program the package starts
goes through tools.resolve_argv, whose search skips empty, ``.`` and other
relative PATH entries. These tests plant a fake ffmpeg in the current
folder and check it is never chosen, on every system.
"""

from __future__ import annotations

import ast
import os
import stat
from pathlib import Path

import pytest

import super_processor
from super_processor import doctor, tools
from super_processor.tools import (
    find_tool,
    resolve_argv,
    tool_path,
    windows_system_tool,
)
from super_processor.wizard import WizardController


def _fake_tool(folder: Path, name: str = "ffmpeg") -> Path:
    """An executable ``name`` (and ``name.exe``) in ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    for file_name in (name, f"{name}.exe"):
        fake = folder / file_name
        fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    return folder / name


@pytest.fixture
def planted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A fake ffmpeg in the current folder and in ./relative, one on PATH."""
    here = tmp_path / "download"
    _fake_tool(here)
    _fake_tool(here / "relative")
    real = _fake_tool(tmp_path / "bin").parent
    monkeypatch.chdir(here)
    return here, real


def _found_in(found: str | None, folder: Path) -> bool:
    return found is not None and Path(found).parent == folder


def test_the_current_folder_is_never_searched(planted: tuple[Path, Path]) -> None:
    here, real = planted
    search = os.pathsep.join(["", ".", "relative", str(real)])
    found = find_tool("ffmpeg", search_path=search)
    assert _found_in(found, real)
    # Only cwd-like entries: nothing is found, though ./ffmpeg is there.
    assert (
        find_tool("ffmpeg", search_path=os.pathsep.join(["", ".", "relative"])) is None
    )


def test_the_resolver_reads_the_real_path(
    planted: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _here, real = planted
    monkeypatch.setenv("PATH", os.pathsep.join(["", ".", str(real)]))
    assert _found_in(tool_path("ffmpeg"), real)
    assert _found_in(resolve_argv(["ffmpeg", "-version"])[0], real)
    assert resolve_argv(["ffmpeg", "-version"])[1:] == ["-version"]
    # doctor's check and the wizard's ffmpeg use the same search.
    found = doctor.which("ffmpeg")
    assert found is not None and Path(found).parent == real.resolve()
    controller = WizardController(Path("jobs-not-made"))
    assert _found_in(controller.ffmpeg_bin, real)


def test_a_program_missing_from_path_keeps_its_bare_name(
    planted: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", os.pathsep.join(["", "."]))
    # Never ./ffmpeg: the caller's own "not found" error is kept instead.
    assert tool_path("ffmpeg") == "ffmpeg"
    assert find_tool("ffmpeg") is None
    assert doctor.which("ffmpeg") is None


def test_a_path_with_a_folder_is_kept(tmp_path: Path) -> None:
    fake = _fake_tool(tmp_path / "tools")
    assert tool_path(str(fake)) == str(fake)
    assert find_tool(str(fake)) == str(fake)
    assert find_tool(str(tmp_path / "missing" / "ffmpeg")) is None
    assert resolve_argv([]) == []
    assert find_tool("") is None


def test_windows_rules(planted: tuple[Path, Path]) -> None:
    """PATHEXT endings, ';' lists and quoted entries; never the current folder."""
    _here, real = planted
    search = ";".join(["", ".", "relative", "C:relative", f'"{real}"'])
    found = find_tool("ffmpeg", search_path=search, windows=True, pathext=".EXE")
    assert found is not None and Path(found) == real / "ffmpeg.exe"
    assert (
        find_tool("ffmpeg", search_path=".;relative", windows=True, pathext=".EXE")
        is None
    )
    # A name with its own ending is looked for as it is.
    found = find_tool("ffmpeg.exe", search_path=str(real), windows=True, pathext=".EXE")
    assert found is not None and Path(found).name == "ffmpeg.exe"


def test_windows_guard_keeps_the_current_folder_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Set, then removed, so the teardown removes what the guard sets.
    monkeypatch.setenv(tools.NO_CWD_SEARCH_ENV, "unset")
    monkeypatch.delenv(tools.NO_CWD_SEARCH_ENV)
    monkeypatch.setattr(tools, "_is_windows", lambda: False)
    tools.guard_windows_search()
    assert tools.NO_CWD_SEARCH_ENV not in os.environ
    monkeypatch.setattr(tools, "_is_windows", lambda: True)
    tools.guard_windows_search()
    assert os.environ[tools.NO_CWD_SEARCH_ENV] == "1"


def test_windows_programs_come_from_system32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYSTEMROOT", r"D:\Win")
    assert windows_system_tool("taskkill.exe") == r"D:\Win\System32\taskkill.exe"
    monkeypatch.delenv("SYSTEMROOT")
    monkeypatch.delenv("WINDIR", raising=False)
    assert windows_system_tool("WindowsPowerShell", "v1.0", "powershell.exe") == (
        r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    )


def _is_subprocess_launch(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in {"run", "Popen", "call", "check_call", "check_output"}
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
    )


def test_every_program_the_package_starts_is_resolved() -> None:
    """Each subprocess launch in the package passes its argv through
    resolve_argv, so no program is ever started by a bare name."""
    package = Path(super_processor.__file__).parent
    bare: list[str] = []
    launches = 0
    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_subprocess_launch(node):
                continue
            launches += 1
            first = node.args[0] if node.args else None
            if not (
                isinstance(first, ast.Call)
                and isinstance(first.func, ast.Name)
                and first.func.id == "resolve_argv"
            ):
                bare.append(f"{module.name}:{node.lineno}")
    assert launches > 20
    assert bare == []
