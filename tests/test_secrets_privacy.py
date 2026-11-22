"""Secrets and privacy: what leaves the machine, and where the key may show.

* The copy and stills sent to Gemini carry no metadata (GPS, title, ...).
* ``doctor`` never prints a key.
* The Gemini key travels in a header, never a URL, and no error, state file,
  API answer or page can carry it.
* A replaced key is used at once.
* The upload is deleted from the File API once nothing needs it.
* NaN and Infinity from Gemini never reach a plan.
"""

from __future__ import annotations

import email.message
import http.client
import io
import json
import logging
import math
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from wizard_client import get

from super_processor import doctor, gemini, shots
from super_processor.gemini import (
    API_KEY_HEADER,
    GeminiClient,
    GeminiError,
    GeminiVideo,
    UrllibGeminiTransport,
    _parse_json_object,
    redact_payload,
    redact_secrets,
    store_gemini_api_key,
)
from super_processor.review import WizardServer
from super_processor.segment_loop import gemini_copy_command, make_gemini_copy
from super_processor.shot_plan import ShotRecipe, finish, new_plan, save_plan
from super_processor.shots import Shot, ShotList, extract_still, save_shots
from super_processor.upscale import UpscaleError, UpscaleSpan
from super_processor.wizard import (
    GEMINI_VIDEO_FILE,
    WizardController,
    WizardState,
    WizardStep,
    job_state_path,
    save_job_wizard_state,
    save_session_state,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg required",
)

# Fake keys only: shaped like a Gemini key, never a real one.
KEY = "AIzaFAKE0000SENTINEL0000000000000000000"
NEW_KEY = "AIzaFAKE1111ROTATED11111111111111111111"
VIDEO = GeminiVideo(
    name="files/abc", uri="https://example.test/files/abc", expires_at=4e9
)


def _pair_before(command: list[str], flag: str, value: str, output: str) -> bool:
    """True when ``flag value`` appears in ``command`` before ``output``."""
    end = command.index(output)
    return any(command[i] == flag and command[i + 1] == value for i in range(end - 1))


# Findings 9 and 12: no metadata in what Gemini gets ----------------------------


def test_the_gemini_copy_command_drops_metadata_chapters_and_other_streams() -> None:
    out = "/tmp/x/gemini_upload.part.mp4"
    command = gemini_copy_command(Path("in.mov"), Path(out), ffmpeg_bin="ffmpeg")
    assert _pair_before(command, "-map_metadata", "-1", out)
    assert _pair_before(command, "-map_chapters", "-1", out)
    assert _pair_before(command, "-map", "0:v:0", out)
    for flag in ("-an", "-sn", "-dn"):
        assert flag in command[: command.index(out)]
    # Output options: after the input, so they apply to the copy.
    assert command.index("-map_metadata") > command.index("-i")


def test_make_gemini_copy_runs_that_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    class _Proc:
        def __init__(self, argv: list[str], **_: Any) -> None:
            seen.append(argv)
            Path(argv[-1]).write_bytes(b"x")
            self.stdout = io.StringIO("progress=end\n")
            self.returncode = 0

        def __enter__(self) -> _Proc:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(subprocess, "Popen", _Proc)
    make_gemini_copy(tmp_path / "in.mov", tmp_path, 1.0, ffmpeg_bin="ffmpeg")
    assert seen, "ffmpeg was not run"
    assert _pair_before(seen[0], "-map_metadata", "-1", seen[0][-1])
    assert _pair_before(seen[0], "-map_chapters", "-1", seen[0][-1])


def test_stills_for_gemini_carry_no_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def run(command: list[str], **_: Any) -> str:
        seen.append(command)
        Path(command[-1]).write_bytes(b"jpeg")
        return ""

    monkeypatch.setattr(shots, "_run", run)
    dest = tmp_path / "s.jpg"
    extract_still(tmp_path / "in.mov", 1.0, dest)
    assert _pair_before(seen[0], "-map_metadata", "-1", str(dest))
    assert _pair_before(seen[0], "-map_chapters", "-1", str(dest))


def _tagged_clip(tmp_path: Path) -> Path:
    """A 1 s clip with a location, title, author, comment, time and chapter."""
    meta = tmp_path / "meta.txt"
    meta.write_text(
        ";FFMETADATA1\n"
        "[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=500\ntitle=SENTINEL-chapter\n",
        encoding="utf-8",
    )
    clip = tmp_path / "tagged.mov"
    tags = {
        "title": "SENTINEL-title",
        "artist": "SENTINEL-artist",
        "comment": "SENTINEL-comment",
        "location": "+48.8584+002.2945/",
        "creation_time": "2020-01-01T00:00:00Z",
    }
    command = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i"]
    command += ["testsrc2=size=160x90:rate=24:duration=1", "-i", str(meta)]
    command += ["-map", "0:v", "-map_chapters", "1"]
    for name, value in tags.items():
        command += ["-metadata", f"{name}={value}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)]
    subprocess.run(command, check=True)
    return clip


def _probe_tags(path: Path) -> str:
    return subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format_tags:stream_tags",
            "-show_chapters",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@needs_ffmpeg
def test_the_uploaded_copy_and_stills_keep_none_of_the_source_metadata(
    tmp_path: Path,
) -> None:
    clip = _tagged_clip(tmp_path)
    before = _probe_tags(clip)
    # The source really carries them, so the check below means something.
    assert "SENTINEL-title" in before and "48.8584" in before
    copy = make_gemini_copy(clip, tmp_path, 1.0)
    after = _probe_tags(copy)
    assert "SENTINEL" not in after and "48.8584" not in after
    assert "2020-01-01" not in after
    assert json.loads(after).get("chapters") in (None, [])
    still = extract_still(clip, 0.5, tmp_path / "still.jpg")
    data = still.read_bytes()
    assert b"SENTINEL" not in data and b"48.8584" not in data


# Finding 11: doctor never prints a key ---------------------------------------


def test_doctor_shows_only_that_a_key_is_set_and_its_last_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SENTINEL-openai-wxyz")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SENTINEL-abcd")
    monkeypatch.setenv(
        "SUPER_PROCESSOR_LLM_BASE_URL",
        "http://user:pw-SENTINEL@host:8080/v1?k=tok-SENTINEL#frag-SENTINEL",
    )
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    results = doctor.collect_doctor_report(tmp_path / "jobs")
    text = doctor.format_doctor_text(results)
    as_json = doctor.doctor_json(results)
    for out in (text, as_json):
        assert "SENTINEL" not in out
        assert "pw-" not in out and "k=tok" not in out
        assert "host:8080/v1" in out
        assert "...wxyz" in out and "...abcd" in out
        # The Gemini key: its source and last 4, never the rest.
        assert "...0000" in out and "GEMINI_API_KEY environment variable" in out


def test_doctor_masks_short_keys_entirely_and_odd_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert doctor.mask_secret("short") == "set"
    assert doctor.safe_url("not a url") == "unreadable URL"
    assert doctor.safe_url("http://[::1]:9/x?y=1") == "http://[::1]:9/x"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "not set" in doctor.check_gemini_key().detail
    monkeypatch.setenv("SOME_TOKEN", "tok-SENTINEL-1234")
    result = doctor.check_model_endpoint("x", "SOME_TOKEN")
    assert result.path is None and "SENTINEL" not in result.detail


# Finding 14: the key in a header, and never in any text ----------------------


class _Response:
    def __init__(self, body: dict[str, Any], headers: dict[str, str] | None = None):
        self._body = json.dumps(body).encode()
        self.headers = email.message.Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        return None


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[urllib.request.Request]]:
    """Every request the real transport makes; none leaves the machine."""
    requests: list[urllib.request.Request] = []
    file = {"name": "files/abc", "uri": "u", "state": "ACTIVE"}

    def urlopen(req: urllib.request.Request, timeout: float = 0) -> _Response:
        requests.append(req)
        if req.get_header("X-goog-upload-command") == "start":
            return _Response({}, {"X-Goog-Upload-URL": "https://upload.test/s?id=1"})
        if req.full_url.startswith("https://upload.test/"):
            return _Response({"file": file})
        if ":generateContent" in req.full_url:
            return _Response({"candidates": []})
        if req.full_url.endswith("/files/abc"):
            return _Response(file)
        return _Response({"models": [{"name": "models/m"}]})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    yield requests


def test_every_gemini_request_sends_the_key_in_a_header_only(
    tmp_path: Path, sent: list[urllib.request.Request]
) -> None:
    transport = UrllibGeminiTransport()
    transport.generate(model="gemini-x", api_key=KEY, body={"contents": []})
    upload = tmp_path / "up.mp4"
    upload.write_bytes(b"video")
    transport.upload_file(
        api_key=KEY, path=upload, mime_type="video/mp4", display_name="up.mp4"
    )
    transport.get_file(api_key=KEY, name="files/abc")
    transport.list_models(api_key=KEY)
    transport.delete_file(api_key=KEY, name="files/abc")
    methods = {req.get_method() for req in sent}
    assert methods == {"GET", "POST", "DELETE"}
    for req in sent:
        assert "key=" not in req.full_url and KEY not in req.full_url
        if not req.full_url.startswith("https://upload.test/"):
            # urllib stores header names capitalized.
            assert req.get_header(API_KEY_HEADER.capitalize()) == KEY


def test_a_file_name_that_is_not_one_never_reaches_a_url(
    sent: list[urllib.request.Request],
) -> None:
    with pytest.raises(GeminiError, match="not a Gemini file name"):
        UrllibGeminiTransport().get_file(api_key=KEY, name="files/../models")
    assert sent == []


def test_a_key_with_a_space_or_line_break_is_refused_without_quoting_it() -> None:
    for bad in ("AIzaFAKE SENTINEL-123456789", "AIzaFAKE\nSENTINEL-123456789"):
        with pytest.raises(GeminiError) as caught:
            store_gemini_api_key(bad)
        assert "SENTINEL" not in str(caught.value)
        with pytest.raises(GeminiError) as caught:
            UrllibGeminiTransport().generate(model="m", api_key=bad, body={})
        assert "SENTINEL" not in str(caught.value)
        with pytest.raises(GeminiError):
            GeminiClient(api_key=bad, transport=UrllibGeminiTransport())


@pytest.mark.parametrize(
    "error",
    [
        http.client.InvalidURL(f"URL can't contain control characters. '?key={KEY}'"),
        ValueError(f"Invalid header value b'{KEY}\\n'"),
    ],
)
def test_an_error_that_quotes_the_request_is_reported_without_it(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    transport = UrllibGeminiTransport()
    calls = (
        lambda: transport.generate(model="m", api_key=KEY, body={}),
        lambda: transport.get_file(api_key=KEY, name="files/abc"),
        lambda: transport.list_models(api_key=KEY),
        lambda: transport.delete_file(api_key=KEY, name="files/abc"),
    )
    for call in calls:
        with pytest.raises(GeminiError) as caught:
            call()
        assert KEY not in str(caught.value)
        assert caught.value.__cause__ is None
        assert "invalid request" in str(caught.value)


def test_a_network_error_that_names_the_key_is_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
        raise urllib.error.URLError(f"proxy said no to ?key={KEY}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(GeminiError) as caught:
        UrllibGeminiTransport().generate(model="m", api_key=KEY, body={})
    assert KEY not in str(caught.value) and caught.value.__cause__ is None


def test_redaction_takes_out_known_keys_key_shapes_and_key_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemini.remember_secret("custom-secret-value-1")
    monkeypatch.setenv("GEMINI_API_KEY", "env-secret-value-22")
    text = (
        f"a {KEY} b custom-secret-value-1 c env-secret-value-22 "
        "d https://x.test/p?key=abc&pageSize=1"
    )
    out = redact_secrets(text)
    for secret in (KEY, "custom-secret-value-1", "env-secret-value-22", "abc&"):
        assert secret not in out
    assert "pageSize=1" in out
    # A short test key is never treated as a secret.
    gemini.remember_secret("k")
    assert redact_secrets("keep") == "keep"
    assert redact_payload({"e": [KEY, 1, None], KEY: True}) == {
        "e": ["***", 1, None],
        "***": True,
    }


def _seeded_job(
    tmp_path: Path, step: WizardStep, gemini_client: Any = None
) -> tuple[WizardController, Path]:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"not-a-real-video")
    controller = WizardController(tmp_path / "jobs", gemini=gemini_client)
    job_id = controller.store.create(source).job_id
    job_dir = controller.store.job_dir(job_id)
    state = WizardState(step=step, job_id=job_id)
    save_job_wizard_state(job_dir, state)
    save_session_state(controller.jobs_dir, state)
    return controller, job_dir


def test_a_failure_quoting_the_key_reaches_no_state_file_api_or_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    controller, job_dir = _seeded_job(tmp_path, WizardStep.SEGMENTS)
    leak = http.client.InvalidURL(f"URL can't contain control characters. '?key={KEY}'")
    controller._land_failure(None, leak)
    state = controller.current_state()
    assert state.error and KEY not in state.error
    assert KEY not in job_state_path(job_dir).read_text(encoding="utf-8")
    # Even an error saved by some other path is taken out on the way out.
    state.error = f"something quoted {KEY}"
    state.progress = {"phase": "x", "pct": 1, "message": f"?key={KEY}"}
    save_job_wizard_state(job_dir, state)
    assert KEY not in job_state_path(job_dir).read_text(encoding="utf-8")
    controller._save_video_record(job_dir, None, f"upload failed: {KEY}")
    assert KEY not in (job_dir / GEMINI_VIDEO_FILE).read_text(encoding="utf-8")

    server = WizardServer(controller.jobs_dir, controller=controller, token="t" * 43)
    monkeypatch.setattr(
        controller, "api_view", lambda: {"error": f"x {KEY}", "nested": [KEY]}
    )
    base = server.start()
    try:
        with get(f"{base}/api/state", server.token) as response:
            body = response.read().decode()
        assert KEY not in body and "***" in body
        monkeypatch.setattr(controller, "render", lambda: f"<p>{KEY}</p>")
        with get(f"{base}/", server.token) as response:
            page = response.read().decode()
        assert KEY not in page
    finally:
        server.stop()


# Finding 16: a replaced key is used at once ------------------------------------


class _DeleteRecorder:
    def __init__(self) -> None:
        self.deleted: list[tuple[str, str]] = []

    def generate(self, **_: Any) -> Any:  # pragma: no cover - never asked
        raise AssertionError("no generate expected")

    def delete_file(self, *, api_key: str, name: str) -> None:
        self.deleted.append((api_key, name))


def _record_video(job_dir: Path) -> None:
    (job_dir / GEMINI_VIDEO_FILE).write_text(
        json.dumps({"video": VIDEO.to_dict(), "error": None}), encoding="utf-8"
    )


def _record(job_dir: Path) -> Any:
    return json.loads((job_dir / GEMINI_VIDEO_FILE).read_text(encoding="utf-8"))


def test_replace_key_takes_effect_at_once_and_deletes_with_the_old_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    store_gemini_api_key(KEY)
    controller, job_dir = _seeded_job(tmp_path, WizardStep.SEGMENTS)
    controller.run_cleanup = lambda work: work()  # type: ignore[assignment]
    old = controller.gemini()
    assert old.api_key == KEY
    recorder = _DeleteRecorder()
    old.transport = recorder
    _record_video(job_dir)
    controller.handle_post("/setup", {"api_key": [NEW_KEY]})
    assert controller.gemini().api_key == NEW_KEY
    assert controller.gemini() is not old
    # The old project's upload is deleted with the old key, and forgotten.
    assert recorder.deleted == [(KEY, "files/abc")]
    assert _record(job_dir) == {"video": None, "error": None}


def test_an_injected_client_is_kept_when_the_key_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    fake = GeminiClient(api_key="k", transport=_DeleteRecorder())
    controller, _job_dir = _seeded_job(tmp_path, WizardStep.SEGMENTS, fake)
    controller.handle_post("/setup", {"api_key": [NEW_KEY]})
    assert controller.gemini() is fake


# Finding 25: the upload is deleted once nothing needs it -----------------------


class _Watcher:
    """A fake client that can delete, as the wizard sees it."""

    def __init__(self, fails: bool = False) -> None:
        self.fails = fails
        self.deleted: list[GeminiVideo] = []

    def delete_video(self, video: GeminiVideo) -> bool:
        self.deleted.append(video)
        if self.fails:
            raise GeminiError("no network")
        return True


def test_starting_a_new_job_deletes_the_old_upload(tmp_path: Path) -> None:
    watcher = _Watcher()
    controller, job_dir = _seeded_job(tmp_path, WizardStep.SEGMENTS, watcher)
    controller.run_cleanup = lambda work: work()  # type: ignore[assignment]
    _record_video(job_dir)
    controller.handle_post("/new", {})
    assert watcher.deleted == [VIDEO]
    assert _record(job_dir) == {"video": None, "error": None}
    # Nothing left to delete the second time.
    controller.start_new_job()
    assert watcher.deleted == [VIDEO]


def test_moving_to_the_render_deletes_the_upload(tmp_path: Path) -> None:
    watcher = _Watcher()
    controller, job_dir = _seeded_job(tmp_path, WizardStep.READY_TO_RENDER, watcher)
    controller.run_cleanup = lambda work: work()  # type: ignore[assignment]
    shot_list = ShotList([Shot(0.0, 3.0), Shot(3.0, 6.0)])
    save_shots(job_dir, shot_list)
    plan = new_plan(shot_list, 2)
    save_plan(job_dir, finish(plan, list(range(len(plan.recipes)))))
    _record_video(job_dir)
    controller.handle_post("/ready", {"action": ["render"]})
    assert controller.current_state().step is WizardStep.RENDERING
    assert watcher.deleted == [VIDEO]
    assert _record(job_dir) == {"video": None, "error": None}


def test_a_failed_delete_never_stops_the_wizard(tmp_path: Path) -> None:
    watcher = _Watcher(fails=True)
    controller, job_dir = _seeded_job(tmp_path, WizardStep.SEGMENTS, watcher)
    ran: list[bool] = []

    def run(work: Any) -> None:
        ran.append(True)
        with pytest.raises(GeminiError):
            work()

    controller.run_cleanup = run
    _record_video(job_dir)
    controller.start_new_job()
    assert ran and controller.current_state().step in (
        WizardStep.PICK_FILE,
        WizardStep.SETUP,
    )


def test_cleanup_runs_in_the_background_by_default(tmp_path: Path) -> None:
    done = threading.Event()

    class _Signals(_Watcher):
        def delete_video(self, video: GeminiVideo) -> bool:
            done.set()
            return True

    watcher = _Signals()
    controller, job_dir = _seeded_job(tmp_path, WizardStep.SEGMENTS, watcher)
    _record_video(job_dir)
    controller.start_new_job()
    assert done.wait(5)


def test_the_client_deletes_best_effort_and_logs_without_the_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    recorder = _DeleteRecorder()
    client = GeminiClient(api_key=KEY, transport=recorder)
    assert client.delete_video(VIDEO)
    assert recorder.deleted == [(KEY, "files/abc")]

    class _Fails(_DeleteRecorder):
        def delete_file(self, *, api_key: str, name: str) -> None:
            raise GeminiError(f"refused {api_key}")

    with caplog.at_level(logging.WARNING, logger="super_processor.gemini"):
        assert not GeminiClient(api_key=KEY, transport=_Fails()).delete_video(VIDEO)
    assert "files/abc" in caplog.text and KEY not in caplog.text

    class _NoDelete:
        def generate(self, **_: Any) -> Any:  # pragma: no cover - never asked
            raise AssertionError

    assert not GeminiClient(api_key=KEY, transport=_NoDelete()).delete_video(VIDEO)


def test_deleting_a_file_already_gone_counts_as_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(req: urllib.request.Request, timeout: float = 0) -> Any:
        raise urllib.error.HTTPError(
            req.full_url, 404, "gone", email.message.Message(), io.BytesIO(b"{}")
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    UrllibGeminiTransport().delete_file(api_key=KEY, name="files/abc")

    def refuse(req: urllib.request.Request, timeout: float = 0) -> Any:
        raise urllib.error.HTTPError(
            req.full_url, 403, "no", email.message.Message(), io.BytesIO(b"{}")
        )

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(GeminiError, match="403"):
        UrllibGeminiTransport().delete_file(api_key=KEY, name="files/abc")


# Finding 24: NaN and Infinity from Gemini never reach a plan -------------------


@pytest.mark.parametrize(
    ("text", "field"),
    [
        ('{"start_s": 0, "end_s": 5, "scale": 2, "strength": NaN}', "strength"),
        ('{"start_s": 0, "end_s": Infinity, "scale": 2, "strength": 0.5}', "end_s"),
        ('{"start_s": -Infinity, "end_s": 5, "scale": 2, "strength": 0.5}', "start_s"),
        ('{"start_s": 0, "end_s": 1e999, "scale": 2, "strength": 0.5}', "end_s"),
        (
            '{"start_s": 0, "end_s": 5, "scale": 2, "strength": 1' + "0" * 30 + "}",
            "strength",
        ),
    ],
)
def test_a_non_finite_number_in_a_reply_is_read_as_missing(
    text: str, field: str
) -> None:
    parsed = _parse_json_object(text)
    assert parsed[field] is None
    assert all(
        value is None or math.isfinite(value)
        for value in parsed.values()
        if not isinstance(value, str)
    )

    class _Replies:
        def generate(self, **_: Any) -> Any:
            return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    client = GeminiClient(api_key="k", transport=_Replies())
    raw = client.revise_upscale_params(
        note="less",
        prior_scale=2,
        prior_strength=0.5,
        prior_start_s=0,
        prior_end_s=5,
        duration_s=5,
    )
    # The result-note path builds its span from this reply: refused, so no
    # NaN strength or endless span ever reaches upscale_plan.json.
    with pytest.raises(UpscaleError):
        UpscaleSpan.from_dict(raw)
    assert "NaN" not in json.dumps(raw) and "Infinity" not in json.dumps(raw)


@pytest.mark.parametrize(
    "fields",
    [
        {"strength": math.nan},
        {"strength": math.inf},
        {"strength": -math.inf},
        {"start_s": math.nan},
        {"end_s": math.inf},
        {"end_s": math.nan},
    ],
)
def test_a_span_refuses_non_finite_numbers(fields: dict[str, float]) -> None:
    data: dict[str, Any] = {"start_s": 0.0, "end_s": 5.0, "scale": 2, "strength": 0.5}
    data.update(fields)
    with pytest.raises(UpscaleError):
        UpscaleSpan.from_dict(data)


def test_a_recipe_drops_non_finite_numbers() -> None:
    recipe = ShotRecipe.from_dict(
        {"strength": math.nan, "shimmer": math.inf, "preview_start": math.nan}
    )
    assert math.isfinite(recipe.strength)
    assert recipe.shimmer is None and recipe.preview_start is None
