"""``super-processor review``: which server it starts, and the run worker GET."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest
from wizard_client import get

from super_processor.cli import build_parser, run_review_command
from super_processor.jobs import JobError, JobStore
from super_processor.review import WizardServer
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_job_wizard_state,
    save_session_state,
)

TOKEN = "test-token"


class _Server:
    """Stands in for the wizard server; records how it was used."""

    made: list[_Server] = []
    token = "tok-123"
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
    return _Server


def test_review_without_a_job_serves_the_wizard_and_prints_its_token(
    tmp_path: Path,
    servers: type[_Server],
    capsys: pytest.CaptureFixture[str],
    web_app: Path,
) -> None:
    jobs = tmp_path / "jobs"
    code = run_review_command(argparse.Namespace(jobs_dir=jobs, job_id=None))
    assert code == 0
    (server,) = servers.made
    assert server.args == (jobs,)
    assert server.calls == ["start", "wait"]
    # The jobs root exists before the wizard writes its session there.
    assert jobs.is_dir()
    # The app is built in, so the printed address is all there is to open.
    assert capsys.readouterr().out.splitlines() == [
        "http://127.0.0.1:9",
        "wizard token: tok-123",
        "open in your browser: http://127.0.0.1:9/?token=tok-123",
    ]


def test_without_a_built_app_review_also_prints_the_dev_server_steps(
    tmp_path: Path,
    servers: type[_Server],
    capsys: pytest.CaptureFixture[str],
    no_web_app: Path,
) -> None:
    code = run_review_command(argparse.Namespace(jobs_dir=tmp_path, job_id=None))
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "http://127.0.0.1:9",
        "wizard token: tok-123",
        "open in your browser: http://127.0.0.1:9/?token=tok-123",
    ]
    assert lines[3] == (
        "this checkout has no built web app: run `npm ci && npm run build:package` "
        "in web/ and restart, or run the Vite dev server: "
        "VITE_PROXY_TARGET=http://127.0.0.1:9 npm run dev, then open "
        "http://localhost:5173/#token=tok-123"
    )


def _wizard_job(store: JobStore, tmp_path: Path, *, state: bool = True) -> str:
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")
    job_id = store.create(source).job_id
    if state:
        save_job_wizard_state(
            store.job_dir(job_id), WizardState(step=WizardStep.RESULT, job_id=job_id)
        )
    return job_id


def test_review_of_a_job_opens_that_job_in_the_app(
    tmp_path: Path,
    servers: type[_Server],
    capsys: pytest.CaptureFixture[str],
    web_app: Path,
) -> None:
    store = JobStore(tmp_path)
    job_id = _wizard_job(store, tmp_path)
    code = run_review_command(argparse.Namespace(jobs_dir=tmp_path, job_id=job_id))
    assert code == 0
    (server,) = servers.made
    assert server.args == (store.root,)
    assert server.kwargs == {"job_id": job_id, "prefetch": False}
    # The same launch address: the app's first state is that job.
    assert capsys.readouterr().out.splitlines() == [
        "http://127.0.0.1:9",
        "wizard token: tok-123",
        "open in your browser: http://127.0.0.1:9/?token=tok-123",
    ]


def test_review_of_a_job_the_wizard_did_not_make_points_to_segment(
    tmp_path: Path,
    servers: type[_Server],
    capsys: pytest.CaptureFixture[str],
) -> None:
    store = JobStore(tmp_path)
    job_id = _wizard_job(store, tmp_path, state=False)
    code = run_review_command(argparse.Namespace(jobs_dir=tmp_path, job_id=job_id))
    assert code == 2
    assert servers.made == []
    out = capsys.readouterr()
    assert out.out == ""
    assert f"job {job_id} was not made by the wizard" in out.err
    assert f"super-processor segment {job_id} --accept" in out.err
    assert "--note TEXT" in out.err


def test_review_of_a_job_lands_the_app_on_it(tmp_path: Path, web_app: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_id = _wizard_job(store, tmp_path)
    # Another window's session shows a different step.
    save_session_state(store.root, WizardState(step=WizardStep.PICK_FILE))
    server = WizardServer(store.root, job_id=job_id, token=TOKEN)
    base = server.start()
    try:
        with get(base + "/api/state", TOKEN) as response:
            view = json.load(response)
        assert (view["job_id"], view["step"]) == (job_id, "result")
    finally:
        server.stop()


def test_prefetch_is_off_unless_review_is_asked_for_it(
    tmp_path: Path, servers: type[_Server]
) -> None:
    parser = build_parser()
    plain = parser.parse_args(["--jobs-dir", str(tmp_path), "review"])
    assert plain.prefetch is False
    assert run_review_command(plain) == 0
    asked = parser.parse_args(["--jobs-dir", str(tmp_path), "review", "--prefetch"])
    assert asked.prefetch is True
    assert run_review_command(asked) == 0
    job_id = _wizard_job(JobStore(tmp_path), tmp_path)
    one_job = parser.parse_args(
        ["--jobs-dir", str(tmp_path), "review", job_id, "--prefetch"]
    )
    assert run_review_command(one_job) == 0
    assert [server.kwargs["prefetch"] for server in servers.made] == [
        False,
        True,
        True,
    ]


def test_servers_hand_the_prefetch_choice_to_the_wizard(tmp_path: Path) -> None:
    plain = WizardServer(tmp_path)
    plain.start()
    plain.stop()
    assert plain.controller.prefetch is False
    server = WizardServer(tmp_path, prefetch=True)
    server.start()
    try:
        assert server.controller.prefetch is True
    finally:
        server.stop()
    store = JobStore(tmp_path)
    source = tmp_path / "a.mp4"
    source.write_bytes(b"x")
    job_id = store.create(source).job_id
    job_dir = store.job_dir(job_id)
    save_job_wizard_state(
        job_dir, WizardState(step=WizardStep.PREVIEW_SEGMENT, job_id=job_id)
    )
    review = WizardServer(store.root, job_id=job_id, prefetch=True)
    review.start()
    try:
        assert review.controller.prefetch is True
        assert review.controller.current_state().job_id == job_id
    finally:
        review.stop()


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
        # The removed enhance step's run link is gone, as is the page's own.
        for gone in ("/enhance?run=1", "/render?run=1", "/api/enhance?run=1"):
            with pytest.raises(HTTPError) as retired:
                get(base + gone, TOKEN)
            assert retired.value.code == 404, gone
            retired.value.close()
        # On any other step the link only answers with the state.
        save_session_state(tmp_path, WizardState(step=WizardStep.PICK_FILE))
        with get(base + "/api/render?run=1", TOKEN) as response:
            assert json.load(response)["step"] == "pick_file"
        assert controller.wait_idle(timeout=10) and renders == []
        save_session_state(tmp_path, WizardState(step=WizardStep.RENDERING))
        with get(base + "/api/render?run=1", TOKEN) as response:
            response.read()
        assert controller.wait_idle(timeout=10)
        assert renders == [WizardStep.RENDERING]
        with get(base + "/api/state", TOKEN) as response:
            assert json.load(response)["step"] == "done"
    finally:
        server.stop()
