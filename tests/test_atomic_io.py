"""Disk helpers: atomic writes and reads that ride out Windows sharing errors."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from super_processor import atomic_io
from super_processor.atomic_io import (
    read_text_retrying,
    rmtree_quietly,
    unlink_quietly,
    unlink_retrying,
    write_text_atomic,
)


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(atomic_io, "RETRY_S", 0)


def test_write_survives_a_replace_that_is_refused_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    target.write_text("old\n", encoding="utf-8")
    real_replace = Path.replace
    calls = {"replace": 0}

    def flaky_replace(self: Path, other: Path) -> Path:
        calls["replace"] += 1
        if calls["replace"] < 3:
            raise PermissionError(13, "file is open in another thread")
        return real_replace(self, other)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    assert write_text_atomic(target, "new\n") == target
    assert target.read_text(encoding="utf-8") == "new\n"
    assert calls["replace"] == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json"]


def test_a_write_that_never_gets_the_file_keeps_the_old_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    target.write_text("old\n", encoding="utf-8")

    def locked(self: Path, other: Path) -> Path:
        raise PermissionError(13, "file is open in another program")

    monkeypatch.setattr(Path, "replace", locked)
    with pytest.raises(PermissionError):
        write_text_atomic(target, "new\n")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert list(tmp_path.glob("*.tmp")) == []


def test_concurrent_writers_never_share_a_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "segments.json"
    temps: list[str] = []
    real_write = Path.write_text
    seen = threading.Lock()

    def recording_write(self: Path, data: str, *args: object, **kwargs: object) -> int:
        with seen:
            temps.append(self.name)
        return real_write(self, data, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", recording_write)
    errors: list[BaseException] = []

    def hammer(text: str) -> None:
        try:
            for _ in range(100):
                write_text_atomic(target, text)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(t,)) for t in ("a\n", "b\n")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert len(temps) == 200
    assert len(set(temps)) == 200
    assert all(name.startswith("segments.json.") for name in temps)
    assert target.read_text(encoding="utf-8") in {"a\n", "b\n"}
    assert list(tmp_path.glob("*.tmp")) == []


def test_read_retries_a_sharing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "plan.json"
    target.write_text("{}\n", encoding="utf-8")
    real_read = Path.read_text
    calls = {"read": 0}

    def flaky_read(self: Path, *args: object, **kwargs: object) -> str:
        calls["read"] += 1
        if calls["read"] < 3:
            raise PermissionError(13, "file is being replaced")
        return real_read(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "read_text", flaky_read)
    assert read_text_retrying(target) == "{}\n"
    assert calls["read"] == 3


def test_read_gives_up_with_the_last_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"read": 0}

    def locked(self: Path, *args: object, **kwargs: object) -> str:
        calls["read"] += 1
        raise PermissionError(13, "file is open in another program")

    monkeypatch.setattr(Path, "read_text", locked)
    with pytest.raises(PermissionError):
        read_text_retrying(tmp_path / "plan.json")
    assert calls["read"] == atomic_io.RETRIES


def test_read_of_a_missing_file_fails_at_once(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_text_retrying(tmp_path / "missing.json")


def test_quiet_removals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    unlink_quietly(tmp_path / "missing.mp4")
    rmtree_quietly(tmp_path / "missing")

    kept = tmp_path / "elsewhere"
    kept.mkdir()
    (kept / "keep.txt").write_text("keep\n", encoding="utf-8")
    link = tmp_path / "work"
    try:
        link.symlink_to(kept, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not allowed here")
    rmtree_quietly(link)
    assert not link.exists()
    assert (kept / "keep.txt").is_file()

    folder = tmp_path / "previews"
    (folder / "work").mkdir(parents=True)
    (folder / "work" / "shot_000_in.mp4").write_bytes(b"x")
    rmtree_quietly(folder)
    assert not folder.exists()

    held = tmp_path / "held.mp4"
    held.write_bytes(b"x")

    def refused(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "file is open in another program")

    monkeypatch.setattr(Path, "unlink", refused)
    unlink_quietly(held)
    assert held.is_file()


def test_unlink_retrying_rides_out_a_held_file_and_says_when_it_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unlink_retrying(tmp_path / "missing.json")
    held = tmp_path / "shots.json"
    held.write_text("{}", encoding="utf-8")
    real_unlink = Path.unlink
    calls = {"unlink": 0}

    def flaky(self: Path, missing_ok: bool = False) -> None:
        calls["unlink"] += 1
        if calls["unlink"] < 3:
            raise PermissionError(13, "file is open in another thread")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", flaky)
    unlink_retrying(held)
    assert not held.exists() and calls["unlink"] == 3

    held.write_text("{}", encoding="utf-8")

    def refused(self: Path, missing_ok: bool = False) -> None:
        raise PermissionError(13, "file is open in another program")

    monkeypatch.setattr(Path, "unlink", refused)
    with pytest.raises(PermissionError):
        unlink_retrying(held)
    assert held.is_file()
