"""Job folders and what the program writes in them are private to the user.

Another account on the same computer must not be able to list or copy the
editor's uploads, outputs, Gemini notes or wizard state. On Linux and macOS
that means folders 0700 and files 0600, whatever the umask; Windows has no
such bits, so these tests skip there.
"""

from __future__ import annotations

import os
import socket
import stat
from collections.abc import Iterator
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import pytest

from super_processor.atomic_io import open_private_new, write_text_atomic
from super_processor.gemini import ChatTurn, append_chat_turn
from super_processor.jobs import DEFAULT_JOBS_DIR_ENV, JobError, JobStore
from super_processor.output_versions import archive_output, outputs_dir
from super_processor.review import TOKEN_HEADER, WizardServer
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    job_state_path,
    save_job_wizard_state,
    session_state_path,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="POSIX mode bits; Windows profiles are per-user"
)

TOKEN = "test-token"


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def private(path: Path) -> bool:
    """Nobody but the owner can read, write or enter ``path``."""
    return mode(path) & 0o077 == 0


@pytest.fixture(autouse=True)
def _usual_umask() -> Iterator[None]:
    """The umask most systems ship (022), which makes 0755/0644 by default."""
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "videos" / "family.mp4"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"not really a video")
    return source


def test_a_new_job_store_and_job_are_private(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    manifest = store.create(_source(tmp_path))
    job_dir = store.job_dir(manifest.job_id)
    assert mode(store.root) == 0o700
    assert mode(job_dir) == 0o700
    assert mode(job_dir / "manifest.json") == 0o600


def test_state_chat_and_other_writes_are_private(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_dir = store.job_dir(store.create(_source(tmp_path)).job_id)
    save_job_wizard_state(job_dir, WizardState(step=WizardStep.SEGMENTS))
    assert mode(job_state_path(job_dir)) == 0o600
    append_chat_turn(job_dir, ChatTurn(role="user", text="the faces are soft"))
    (chat,) = job_dir.glob("gemini_chat*.json")
    assert mode(chat) == 0o600
    # Replacing a file that an older version left 0644 makes it private too.
    loose = job_dir / "notes.json"
    loose.write_text("{}\n", encoding="utf-8")
    assert mode(loose) == 0o644
    write_text_atomic(loose, "{}\n")
    assert mode(loose) == 0o600


def test_kept_outputs_folder_is_private(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_dir = store.job_dir(store.create(_source(tmp_path)).job_id)
    (job_dir / "output.mp4").write_bytes(b"video")
    assert archive_output(job_dir, kind="render") is not None
    assert mode(outputs_dir(job_dir)) == 0o700


def test_the_home_jobs_folder_and_old_job_folders_are_tightened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(DEFAULT_JOBS_DIR_ENV, raising=False)
    # What an older version left: everything 0755.
    root = home / ".super-processor" / "jobs"
    store = JobStore()
    assert store.root == root.resolve()
    old = store.job_dir(store.create(_source(tmp_path)).job_id)
    for folder in (home / ".super-processor", root, old):
        folder.chmod(0o755)
    store = JobStore()
    store.ensure_root()
    assert mode(home / ".super-processor") == 0o700
    assert mode(root) == 0o700
    store.load(old.name)
    assert mode(old) == 0o700


def test_a_fresh_home_folder_is_private_from_the_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(DEFAULT_JOBS_DIR_ENV, raising=False)
    JobStore().ensure_root()
    assert mode(home / ".super-processor") == 0o700
    assert mode(home / ".super-processor" / "jobs") == 0o700


def test_a_chosen_jobs_folder_keeps_its_mode_but_its_jobs_are_private(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    store = JobStore(shared)
    job_dir = store.job_dir(store.create(_source(tmp_path)).job_id)
    assert mode(shared) == 0o755  # the user's own folder is not changed
    assert mode(job_dir) == 0o700


def test_a_jobs_folder_every_user_may_change_is_refused(tmp_path: Path) -> None:
    open_root = tmp_path / "open"
    open_root.mkdir()
    open_root.chmod(0o777)
    with pytest.raises(JobError, match="every user"):
        JobStore(open_root).create(_source(tmp_path))
    with pytest.raises(JobError, match="--jobs-dir"):
        WizardController(open_root)
    # The sticky bit (as on /tmp) keeps users to their own entries: allowed.
    open_root.chmod(0o1777)
    JobStore(open_root).create(_source(tmp_path))


def test_open_private_new_is_owner_only_and_exclusive(tmp_path: Path) -> None:
    target = tmp_path / "upload.mp4"
    with open_private_new(target) as out:
        out.write(b"x")
    assert mode(target) == 0o600
    with pytest.raises(FileExistsError):
        open_private_new(target)


def _free_port() -> int:
    """A free port in 46000-46999, kept away from the user's own servers."""
    for port in range(46000, 47000):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    return 0


class _Video:
    width = 320
    height = 240


class _Facts:
    format_name = "mov,mp4,m4a,3gp,3g2,mj2"
    has_video = True
    duration_s = 8.0

    def primary_video(self) -> _Video:
        return _Video()


def test_a_dropped_video_is_saved_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("super_processor.wizard.probe_file", lambda _path: _Facts())
    monkeypatch.setattr(
        "super_processor.wizard.require_supported_timing", lambda *_a, **_k: None
    )
    jobs = tmp_path / "jobs"
    controller = WizardController(jobs)
    server = WizardServer(jobs, token=TOKEN, controller=controller, port=_free_port())
    base = urlparse(server.start())
    try:
        connection = HTTPConnection(base.hostname or "127.0.0.1", base.port)
        connection.request(
            "POST",
            "/api/upload",
            body=b"video bytes",
            headers={TOKEN_HEADER: TOKEN, "X-File-Name": "family.mp4"},
        )
        response = connection.getresponse()
        response.read()
        assert response.status == 200
    finally:
        server.stop()
    uploads = jobs / "uploads"
    (saved,) = uploads.iterdir()
    assert mode(jobs) == 0o700
    assert mode(uploads) == 0o700
    assert mode(saved) == 0o600
    state = controller.current_state()
    assert state.job_id
    job_dir = controller.store.job_dir(state.job_id)
    assert mode(job_dir) == 0o700
    assert private(job_state_path(job_dir))
    assert private(session_state_path(jobs))
