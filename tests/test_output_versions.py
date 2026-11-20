"""Kept output versions: numbering, links and copies, the index, orphans."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from super_processor import output_versions
from super_processor.output_versions import (
    OutputVersion,
    archive_output,
    current_version,
    list_versions,
    version_name,
)
from super_processor.upscale import delivery_marker_path


def _output(job_dir: Path, data: bytes) -> Path:
    """Replace ``output.mp4`` as the renderer does: a new file, one rename."""
    job_dir.mkdir(parents=True, exist_ok=True)
    work = job_dir / "work.mp4"
    work.write_bytes(data)
    output = job_dir / "output.mp4"
    work.replace(output)
    return output


def _index(job_dir: Path) -> list[dict[str, object]]:
    data = json.loads((job_dir / "outputs" / "versions.json").read_text())
    entries = data["versions"]
    assert isinstance(entries, list)
    return entries


def test_versions_are_numbered_in_order_and_listed_oldest_first(
    tmp_path: Path,
) -> None:
    for data in (b"first", b"second", b"third"):
        _output(tmp_path, data)
        archive_output(tmp_path, note=data.decode())

    versions = list_versions(tmp_path)
    assert [v.name for v in versions] == ["v001.mp4", "v002.mp4", "v003.mp4"]
    assert [v.note for v in versions] == ["first", "second", "third"]
    assert [(tmp_path / "outputs" / v.name).read_bytes() for v in versions] == [
        b"first",
        b"second",
        b"third",
    ]
    assert [entry["file"] for entry in _index(tmp_path)] == [
        "v001.mp4",
        "v002.mp4",
        "v003.mp4",
    ]


def test_nothing_is_pruned_however_many_versions_there_are(tmp_path: Path) -> None:
    for number in range(12):
        _output(tmp_path, f"render {number}".encode())
        archive_output(tmp_path)

    assert [v.number for v in list_versions(tmp_path)] == list(range(1, 13))
    assert (tmp_path / "outputs" / "v001.mp4").read_bytes() == b"render 0"


def test_no_output_keeps_nothing(tmp_path: Path) -> None:
    assert archive_output(tmp_path) is None
    assert list_versions(tmp_path) == []
    assert not (tmp_path / "outputs").exists()


def test_an_output_already_kept_is_not_kept_twice(tmp_path: Path) -> None:
    _output(tmp_path, b"result")
    first = archive_output(tmp_path, kind="render")
    again = archive_output(tmp_path, kind="earlier")

    assert first is not None and again == first
    assert [v.number for v in list_versions(tmp_path)] == [1]
    assert current_version(tmp_path) == first


@pytest.mark.skipif(os.name == "nt", reason="hard links are POSIX-only here")
def test_on_posix_a_version_is_a_hard_link_that_keeps_the_old_picture(
    tmp_path: Path,
) -> None:
    output = _output(tmp_path, b"old picture")
    archive_output(tmp_path)
    kept = tmp_path / "outputs" / "v001.mp4"
    assert kept.stat().st_ino == output.stat().st_ino

    # The next render replaces output.mp4; the version keeps its own file.
    _output(tmp_path, b"new picture")
    assert kept.read_bytes() == b"old picture"
    assert kept.stat().st_ino != output.stat().st_ino
    assert current_version(tmp_path) is None


def test_without_links_a_version_is_a_copy_and_still_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(output_versions, "can_link", lambda: False)
    output = _output(tmp_path, b"copied")
    version = archive_output(tmp_path)
    kept = tmp_path / "outputs" / "v001.mp4"

    assert kept.read_bytes() == b"copied"
    assert kept.stat().st_ino != output.stat().st_ino
    assert version is not None
    assert (version.size, version.mtime_ns) == (
        output.stat().st_size,
        output.stat().st_mtime_ns,
    )
    assert current_version(tmp_path) == version


def test_a_disk_without_hard_links_still_claims_names_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_links(_src: object, _dst: object) -> None:
        raise OSError("hard links are not supported here")

    monkeypatch.setattr(output_versions, "can_link", lambda: False)
    monkeypatch.setattr(os, "link", no_links)
    _output(tmp_path, b"one")
    archive_output(tmp_path)
    _output(tmp_path, b"two")
    archive_output(tmp_path)

    folder = tmp_path / "outputs"
    assert (folder / "v001.mp4").read_bytes() == b"one"
    assert (folder / "v002.mp4").read_bytes() == b"two"
    assert not list(folder.glob("*.part"))


def test_the_delivery_marker_is_copied_beside_the_version(tmp_path: Path) -> None:
    output = _output(tmp_path, b"delivery")
    marker = delivery_marker_path(output)
    marker.write_text('{"version": 3}\n', encoding="utf-8")
    archive_output(tmp_path)

    kept_marker = delivery_marker_path(tmp_path / "outputs" / "v001.mp4")
    assert kept_marker.read_text() == '{"version": 3}\n'
    # A copy, not a link: rewriting the output's marker leaves it alone.
    marker.write_text('{"version": 4}\n', encoding="utf-8")
    assert kept_marker.read_text() == '{"version": 3}\n'


def test_move_renames_the_output_into_outputs(tmp_path: Path) -> None:
    output = _output(tmp_path, b"before restart")
    version = archive_output(tmp_path, kind="earlier", move=True)

    assert version is not None and version.kind == "earlier"
    assert not output.exists()
    assert (tmp_path / "outputs" / "v001.mp4").read_bytes() == b"before restart"


def test_a_failed_move_puts_the_output_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_name(_folder: Path, _part: Path) -> int:
        raise output_versions.VersionError("no free name")

    monkeypatch.setattr(output_versions, "_claim_next", no_name)
    output = _output(tmp_path, b"still here")
    with pytest.raises(output_versions.VersionError):
        archive_output(tmp_path, move=True)

    assert output.read_bytes() == b"still here"
    assert not list((tmp_path / "outputs").glob("*.part"))


def test_a_moved_output_counts_as_kept_when_its_index_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full_disk(*_args: object) -> None:
        raise OSError("No space left on device")

    monkeypatch.setattr(output_versions, "_write_index", full_disk)
    monkeypatch.setattr(output_versions, "_copy_marker", full_disk)
    output = _output(tmp_path, b"before restart")
    version = archive_output(tmp_path, kind="earlier", move=True)

    assert version is not None and version.number == 1
    assert not output.exists()
    # Listed as recovered until the next archive writes the index.
    assert [(v.name, v.kind) for v in list_versions(tmp_path)] == [
        ("v001.mp4", "recovered")
    ]
    assert (tmp_path / "outputs" / "v001.mp4").read_bytes() == b"before restart"


def test_orphan_files_are_listed_as_recovered_and_join_the_index(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "outputs"
    folder.mkdir()
    (folder / "v002.mp4").write_bytes(b"left by a stopped archive")
    (folder / "v002.mp4.delivery.json").write_text("{}")
    (folder / "notes.txt").write_text("not a version")

    listed = list_versions(tmp_path)
    assert [(v.number, v.kind) for v in listed] == [(2, "recovered")]
    # Read-only: listing wrote no index.
    assert not (folder / "versions.json").exists()

    _output(tmp_path, b"next")
    version = archive_output(tmp_path)
    assert version is not None and version.number == 3
    assert [(entry["file"], entry["kind"]) for entry in _index(tmp_path)] == [
        ("v002.mp4", "recovered"),
        ("v003.mp4", "render"),
    ]


def test_an_index_entry_whose_file_is_gone_is_left_out(tmp_path: Path) -> None:
    _output(tmp_path, b"one")
    archive_output(tmp_path)
    _output(tmp_path, b"two")
    archive_output(tmp_path)
    (tmp_path / "outputs" / "v001.mp4").unlink()

    assert [v.number for v in list_versions(tmp_path)] == [2]


def test_an_unreadable_index_still_lists_the_files(tmp_path: Path) -> None:
    _output(tmp_path, b"one")
    archive_output(tmp_path, note="kept")
    (tmp_path / "outputs" / "versions.json").write_text("{not json")

    assert [(v.number, v.kind) for v in list_versions(tmp_path)] == [(1, "recovered")]


def test_concurrent_archives_get_their_own_numbers(tmp_path: Path) -> None:
    sources = []
    for number in range(8):
        job = tmp_path / f"copy{number}"
        _output(job, f"render {number}".encode())
        sources.append(job / "output.mp4")
    folder = tmp_path / "outputs"
    folder.mkdir()
    barrier = threading.Barrier(len(sources))
    claimed: list[int] = []
    errors: list[BaseException] = []

    def claim(source: Path, index: int) -> None:
        part = folder / f".{index}.part"
        os.link(source, part)
        barrier.wait()
        try:
            # The name claim alone, as a second server on the folder does it.
            claimed.append(output_versions._claim_next(folder, part))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(target=claim, args=(source, index))
        for index, source in enumerate(sources)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert errors == []
    assert sorted(claimed) == list(range(1, 9))
    kept = sorted((folder / version_name(n)).read_bytes() for n in claimed)
    assert kept == sorted(source.read_bytes() for source in sources)
    assert not list(folder.glob("*.part"))


def test_concurrent_archives_of_one_output_keep_it_once(tmp_path: Path) -> None:
    _output(tmp_path, b"one result")
    barrier = threading.Barrier(4)
    kept: list[OutputVersion | None] = []

    def keep() -> None:
        barrier.wait()
        kept.append(archive_output(tmp_path))

    threads = [threading.Thread(target=keep) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert len(kept) == 4 and all(version == kept[0] for version in kept)
    assert [entry["file"] for entry in _index(tmp_path)] == ["v001.mp4"]


def test_a_version_round_trips_through_its_dict() -> None:
    version = OutputVersion(
        number=7,
        kind="render",
        created_at="2026-11-20T10:00:00+00:00",
        size=10,
        mtime_ns=123,
        note="less detail at 2:00",
        extra={"scale": 2},
    )
    assert OutputVersion.from_dict(version.to_dict()) == version
    with pytest.raises(ValueError):
        OutputVersion.from_dict({**version.to_dict(), "number": 0})
