"""``super-processor review``: which server it starts, and the run worker GET."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from super_processor.cli import run_review_command
from super_processor.jobs import JobError, JobStore
from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_session_state,
)

TOKEN = "test-token"


class _Server:
    """Stands in for the wizard or review server; records how it was used."""

    made: list[_Server] = []
    token: str | None = "tok-123"
    interrupt = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.calls: list[str] = []
        _Server.made.append(self)

    def start(self) -> str:
        self.calls.append("start")
        return "http://127.0.0.1:9"

    def wait(self) -> None:
        self.calls.append("wait")
        if self.interrupt:
            raise KeyboardInterrupt

    def stop(self) -> None:
        self.calls.append("stop")


@pytest.fixture
def servers(monkeypatch: pytest.MonkeyPatch) -> type[_Server]:
    _Server.made = []
    monkeypatch.setattr("super_processor.cli.WizardServer", _Server)
    monkeypatch.setattr("super_processor.cli.ReviewServer", _Server)
    return _Server


def test_review_without_a_job_serves_the_wizard_and_prints_its_token(
    tmp_path: Path,
    servers: type[_Server],
    capsys: pytest.CaptureFixture[str],
) -> None:
    jobs = tmp_path / "jobs"
    code = run_review_command(argparse.Namespace(jobs_dir=jobs, job_id=None))
    assert code == 0
    (server,) = servers.made
    assert server.args == (jobs,)
    assert server.calls == ["start", "wait"]
    # The jobs root exists before the wizard writes its session there.
    assert jobs.is_dir()
    assert capsys.readouterr().out.splitlines() == [
        "http://127.0.0.1:9",
        "wizard token: tok-123",
    ]


def test_review_of_a_job_serves_that_job(
    tmp_path: Path,
    servers: type[_Server],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = JobStore(tmp_path)
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")
    job_id = store.create(source).job_id
    monkeypatch.setattr(_Server, "token", None)
    code = run_review_command(argparse.Namespace(jobs_dir=tmp_path, job_id=job_id))
    assert code == 0
    (server,) = servers.made
    assert server.args == (store.job_dir(job_id),)
    assert server.kwargs == {"jobs_dir": store.root, "job_id": job_id}
    # A split review page has no token to print.
    assert capsys.readouterr().out.splitlines() == ["http://127.0.0.1:9"]


def test_review_of_an_unknown_job_fails_before_serving(
    tmp_path: Path, servers: type[_Server]
) -> None:
    with pytest.raises(JobError):
        run_review_command(argparse.Namespace(jobs_dir=tmp_path, job_id="nope"))
    assert servers.made == []


def test_ctrl_c_stops_the_server_cleanly(
    tmp_path: Path, servers: type[_Server], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_Server, "interrupt", True)
    code = run_review_command(argparse.Namespace(jobs_dir=tmp_path, job_id=None))
    assert code == 0
    (server,) = servers.made
    assert server.calls == ["start", "wait", "stop"]


def test_render_run_link_starts_the_render_only_on_the_render_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    controller = WizardController(tmp_path)
    renders: list[WizardStep] = []

    def _fake_render() -> None:
        renders.append(controller.current_state().step)
        save_session_state(tmp_path, WizardState(step=WizardStep.DONE))

    monkeypatch.setattr(controller, "run_render", _fake_render)
    server = WizardServer(tmp_path, token=TOKEN, controller=controller)
    base = server.start()
    try:
        # The removed enhance step's run link is gone.
        with pytest.raises(HTTPError) as retired:
            urlopen(base + f"/enhance?run=1&token={TOKEN}")
        assert retired.value.code == 404
        retired.value.close()
        # On any other step the link only shows the page.
        save_session_state(tmp_path, WizardState(step=WizardStep.PICK_FILE))
        with urlopen(base + f"/render?run=1&token={TOKEN}") as response:
            response.read()
        assert controller.wait_idle(timeout=10) and renders == []
        save_session_state(tmp_path, WizardState(step=WizardStep.RENDERING))
        with urlopen(base + f"/render?run=1&token={TOKEN}") as response:
            response.read()
        assert controller.wait_idle(timeout=10)
        assert renders == [WizardStep.RENDERING]
        with urlopen(base + "/") as response:
            assert "<h1>Done</h1>" in response.read().decode()
    finally:
        server.stop()
