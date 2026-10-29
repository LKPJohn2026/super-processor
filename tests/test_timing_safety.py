"""CFR admission is based on decoded timestamps, not rate metadata or engines."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from super_processor import probe, upscale
from super_processor.probe import UnsupportedTimingError, require_cfr_timing
from super_processor.shot_plan import new_plan, render_previews
from super_processor.shots import Shot, ShotList
from super_processor.upscale import (
    FakeUpscaleEngine,
    FlashVsrEngine,
    UpscaleError,
    UpscalePlan,
    UpscaleRequest,
    UpscaleSpan,
    apply_range_revise,
    plain_upscale,
    render_plan,
    render_shots,
    restore_clip,
    upscale_all,
    video_stream_facts,
)
from super_processor.wizard import (
    WizardController,
    WizardState,
    WizardStep,
    save_session_state,
)

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe required",
)


def _payload(
    pts: list[int], *, rate: str = "10/1", duration: int = 10
) -> dict[str, Any]:
    return {
        "streams": [{"r_frame_rate": rate, "time_base": "1/100"}],
        "frames": [{"pts": value, "pkt_duration": duration} for value in pts],
    }


def _mock_probe(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "-show_frames" in command
        assert "-read_intervals" not in command  # The entire timeline is checked.
        assert kwargs["timeout"] > 0
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr("super_processor.probe.subprocess.run", run)
    monkeypatch.setattr(probe, "find_ffprobe", lambda: "ffprobe")


@pytest.mark.parametrize("pts", [[0, 10, 20, 30], [200, 210, 220, 230], [0]])
def test_uniform_decoded_timestamps_are_supported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pts: list[int]
) -> None:
    source = tmp_path / "clip.mp4"
    source.touch()
    _mock_probe(monkeypatch, _payload(pts))
    require_cfr_timing(source)


def test_fractional_rate_allows_clock_quantization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "clip.mkv"
    source.touch()
    payload = _payload([0, 42, 83, 125, 167, 208], rate="24/1", duration=41)
    payload["streams"][0]["time_base"] = "1/1000"
    _mock_probe(monkeypatch, payload)
    require_cfr_timing(source)


@pytest.mark.parametrize(
    "pts",
    [
        [0, 10, 30, 40],  # A dropped frame / presentation gap.
        [0, 10, 10, 20],
        [0, 10, 5, 20],
        [0, 11, 22, 33],  # Uniform, but incompatible with the renderer's rate.
        [0, 10, 20, 31],  # A late change is not hidden by sampling the prefix.
    ],
)
def test_nonuniform_or_incompatible_timing_is_refused_despite_equal_rates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pts: list[int]
) -> None:
    source = tmp_path / "clip.mp4"
    source.touch()
    payload = _payload(pts)
    payload["streams"][0]["avg_frame_rate"] = "10/1"
    assert probe.detect_vfr("10/1", "10/1") is False
    _mock_probe(monkeypatch, payload)
    with pytest.raises(UnsupportedTimingError, match="VFR/unsupported.*CFR") as exc:
        require_cfr_timing(source)
    assert "conversion may change frames/timing" in str(exc.value)


@pytest.mark.parametrize("defect", ["empty", "missing_pts", "rate", "duration", "tail"])
def test_unverifiable_timing_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    source = tmp_path / "clip.mp4"
    source.touch()
    payload = _payload([0, 10, 20])
    if defect == "empty":
        payload["frames"] = []
    elif defect == "missing_pts":
        del payload["frames"][1]["pts"]
    elif defect == "rate":
        payload["streams"][0]["r_frame_rate"] = "0/0"
    elif defect == "duration":
        del payload["frames"][-1]["pkt_duration"]
    else:
        payload["frames"][-1]["pkt_duration"] = 20
    _mock_probe(monkeypatch, payload)
    with pytest.raises(UnsupportedTimingError, match="VFR/unsupported"):
        require_cfr_timing(source)


@pytest.mark.parametrize("failure", ["exit", "decode_error", "json", "timeout"])
def test_failed_decode_never_admits_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = tmp_path / "clip.mp4"
    source.touch()

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 600)
        return subprocess.CompletedProcess(
            command,
            1 if failure == "exit" else 0,
            "broken" if failure == "json" else json.dumps(_payload([0, 10])),
            "decode error" if failure == "decode_error" else "",
        )

    monkeypatch.setattr("super_processor.probe.subprocess.run", run)
    with pytest.raises(UnsupportedTimingError, match="VFR/unsupported"):
        require_cfr_timing(source)


def _clip(path: Path, *, vfr: bool = False, rate: str = "20") -> Path:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=96x64:rate={rate}",
    ]
    if vfr:
        command.extend(["-vf", "setpts=if(lt(N\\,30)\\,2*N\\,N+30)"])
    command.extend(
        [
            "-frames:v",
            "60",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-bf",
            "2",
            str(path),
        ]
    )
    subprocess.run(command, check=True, capture_output=True)
    return path


@needs_ffmpeg
@pytest.mark.parametrize("rate", ["20", "24000/1001", "30000/1001"])
def test_synthetic_cfr_with_b_frames_renders_with_fake_engine(
    tmp_path: Path, rate: str
) -> None:
    source = _clip(tmp_path / "cfr.mp4", rate=rate)
    require_cfr_timing(source)
    duration = float(probe.probe_file(source).duration_s or 0)
    output = tmp_path / "output.mp4"
    render_shots(
        source,
        output,
        (
            UpscaleSpan(0, duration / 2),
            UpscaleSpan(duration / 2, duration, strength=0.2),
        ),
        engine=FakeUpscaleEngine(),
        duration_s=duration,
    )
    assert video_stream_facts(output)[0] == 60


@needs_ffmpeg
@pytest.mark.parametrize("vfr", [False, True])
def test_wizard_checks_timestamps_before_job_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, vfr: bool
) -> None:
    source = _clip(tmp_path / "input.mp4", vfr=vfr)
    facts = probe.probe_file(source)
    # Even misleading metadata must not bypass the separate decoded scan.
    facts.is_vfr = False
    video = facts.primary_video()
    assert video is not None
    video.avg_frame_rate = video.r_frame_rate
    monkeypatch.setattr("super_processor.wizard.probe_file", lambda _source: facts)
    controller = WizardController(tmp_path / "jobs", upscale_engine=FakeUpscaleEngine())
    save_session_state(controller.jobs_dir, WizardState(step=WizardStep.PICK_FILE))
    controller.handle_post("/pick", {"path": [str(source)]})
    state = controller.current_state()
    if vfr:
        assert state.step is WizardStep.PICK_FILE
        assert state.job_id is None
        assert "VFR/unsupported" in (state.error or "")
        assert "conversion may change frames/timing" in (state.error or "")
        assert controller.store.list_jobs() == []
    else:
        assert state.step is WizardStep.SAMPLING
        assert state.job_id is not None
        assert len(controller.store.list_jobs()) == 1


class _NoWorkEngine:
    def upscale(self, request: UpscaleRequest) -> Path:
        pytest.fail("unsupported media reached the engine")


@needs_ffmpeg
@pytest.mark.parametrize(
    "entry",
    [
        "plan",
        "shots",
        "revise",
        "restore",
        "plain",
        "fake",
        "flash",
        "batch",
        "flash_batch",
        "previews",
    ],
)
def test_vfr_is_rejected_before_work_or_overwriting_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    source = _clip(tmp_path / "vfr.mp4", vfr=True)
    output = tmp_path / "output.mp4"
    output.write_bytes(b"previous output must survive")
    marker = upscale.delivery_marker_path(output)
    marker.write_bytes(b"previous delivery marker")
    request = UpscaleRequest(source, output)
    span = UpscaleSpan(0, 4.5)
    engine = _NoWorkEngine()

    def no_gpu() -> tuple[Path, Path]:
        pytest.fail("unsupported media reached GPU readiness / loading")

    monkeypatch.setattr(upscale, "_flashvsr_ready", no_gpu)
    with pytest.raises(UpscaleError, match="VFR/unsupported"):
        if entry == "plan":
            render_plan(
                source, output, UpscalePlan((span,)), engine=engine, duration_s=4.5
            )
        elif entry == "shots":
            render_shots(source, output, (span,), engine=engine, duration_s=4.5)
        elif entry == "revise":
            apply_range_revise(
                source, output, UpscaleSpan(1, 2), output, engine=engine, duration_s=4.5
            )
        elif entry == "restore":
            restore_clip(engine, request, use_upscale=True)
        elif entry == "plain":
            plain_upscale(request)
        elif entry == "fake":
            FakeUpscaleEngine().upscale(request)
        elif entry == "flash":
            FlashVsrEngine().upscale(request)
        elif entry == "batch":
            upscale_all(engine, [request])
        elif entry == "flash_batch":
            FlashVsrEngine().upscale_many([request])
        else:
            shots = ShotList(shots=[Shot(0, 4.5)])
            render_previews(
                source, tmp_path, shots, new_plan(shots, 2), [0], engine=engine
            )
    assert output.read_bytes() == b"previous output must survive"
    assert marker.read_bytes() == b"previous delivery marker"
    assert not (tmp_path / "shot_work").exists()
    assert not (tmp_path / "range_work").exists()
    assert not (tmp_path / "previews").exists()


@needs_ffmpeg
@pytest.mark.parametrize("flash", [False, True])
def test_batch_preflights_all_inputs_before_starting_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flash: bool
) -> None:
    good = _clip(tmp_path / "good.mp4")
    bad = _clip(tmp_path / "bad.mp4", vfr=True)
    requests = [
        UpscaleRequest(source, tmp_path / f"out_{i}.mp4")
        for i, source in enumerate([good, bad])
    ]

    def no_gpu() -> tuple[Path, Path]:
        pytest.fail("batch loaded the GPU before checking every input")

    monkeypatch.setattr(upscale, "_flashvsr_ready", no_gpu)
    with pytest.raises(UpscaleError, match="VFR/unsupported"):
        if flash:
            FlashVsrEngine().upscale_many(requests)
        else:
            upscale_all(_NoWorkEngine(), requests)
    assert all(not request.output.exists() for request in requests)


@needs_ffmpeg
@pytest.mark.parametrize("batch_method", [False, True])
def test_batch_uses_engine_binary_without_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_method: bool
) -> None:
    executable = shutil.which("ffmpeg")
    assert executable is not None
    ffmpeg = str(Path(executable).resolve())
    ffprobe = upscale._ffprobe_for(ffmpeg)
    if not Path(ffprobe).is_absolute():
        pytest.skip("requires ffprobe beside the explicit ffmpeg binary")
    source = _clip(tmp_path / "good.mp4")
    bad = _clip(tmp_path / "bad.mp4", vfr=True)
    engine = FakeUpscaleEngine(ffmpeg_bin=ffmpeg)
    requests = [UpscaleRequest(source, tmp_path / "out.mp4")]
    batches: list[list[UpscaleRequest]] = []

    def many(batch: list[UpscaleRequest]) -> list[Path]:
        batches.append(batch)
        return [engine.upscale(request) for request in batch]

    if batch_method:
        monkeypatch.setattr(engine, "upscale_many", many, raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    assert upscale_all(engine, requests) == [requests[0].output]
    assert video_stream_facts(requests[0].output, ffprobe_bin=ffprobe)[0] == 60
    assert len(batches) == int(batch_method)
    refused = [
        UpscaleRequest(source, tmp_path / "untouched.mp4"),
        UpscaleRequest(bad, tmp_path / "bad_out.mp4"),
    ]
    with pytest.raises(UpscaleError, match="VFR/unsupported"):
        upscale_all(engine, refused)
    assert not any(request.output.exists() for request in refused)
    assert len(batches) == int(batch_method)
    with pytest.raises(UpscaleError, match="Could not decode timestamps"):
        upscale_all(engine, requests, ffmpeg_bin=str(tmp_path / "ffmpeg"))


@needs_ffmpeg
def test_batch_binary_override_without_engine_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = shutil.which("ffmpeg")
    assert executable is not None
    ffmpeg = str(Path(executable).resolve())
    if not Path(upscale._ffprobe_for(ffmpeg)).is_absolute():
        pytest.skip("requires ffprobe beside the explicit ffmpeg binary")
    source = _clip(tmp_path / "good.mp4")

    class ExternalEngine:
        def upscale(self, request: UpscaleRequest) -> Path:
            return request.output

    request = UpscaleRequest(source, tmp_path / "out.mp4")
    monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
    assert upscale_all(ExternalEngine(), [request], ffmpeg_bin=ffmpeg) == [
        request.output
    ]


@needs_ffmpeg
def test_nested_entry_checks_reuse_scan_but_replaced_sources_are_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _clip(tmp_path / "input.mp4")
    original_run = subprocess.run
    scans: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if "-show_frames" in command and command[-1] == str(source):
            scans.append(command)
        return original_run(command, **kwargs)

    monkeypatch.setattr("super_processor.probe.subprocess.run", run)
    output = tmp_path / "output.mp4"
    render_plan(
        source,
        output,
        UpscalePlan((UpscaleSpan(0, 3),)),
        engine=FakeUpscaleEngine(),
        duration_s=3,
    )
    require_cfr_timing(source, ffprobe="ffprobe")
    assert len(scans) == 1
    before = output.read_bytes()
    _clip(source, vfr=True)
    with pytest.raises(UpscaleError, match="VFR/unsupported"):
        render_plan(
            source,
            output,
            UpscalePlan((UpscaleSpan(0, 4.5),)),
            engine=_NoWorkEngine(),
            duration_s=4.5,
        )
    assert len(scans) == 2
    assert output.read_bytes() == before


def test_source_changed_during_decode_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.mp4"
    source.touch()

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        source.write_bytes(source.read_bytes() + b"changed")
        return subprocess.CompletedProcess(
            command, 0, json.dumps(_payload([0, 10])), ""
        )

    monkeypatch.setattr("super_processor.probe.subprocess.run", run)
    for _ in range(2):
        with pytest.raises(UnsupportedTimingError, match="source changed"):
            require_cfr_timing(source, ffprobe="ffprobe")


@needs_ffmpeg
def test_chunked_fake_inference_scans_original_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_frame_alignment import _fake_flashvsr

    _fake_flashvsr(tmp_path, monkeypatch)
    source = _clip(tmp_path / "input.mp4")
    original_run = subprocess.run
    scans: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if "-show_frames" in command:
            scans.append(command)
        return original_run(command, **kwargs)

    monkeypatch.setattr("super_processor.probe.subprocess.run", run)
    output = FlashVsrEngine(chunk_s=1).upscale(
        UpscaleRequest(source, tmp_path / "out" / "clip.mp4")
    )
    assert len(scans) == 1
    assert scans[0][-1] == str(source)
    assert video_stream_facts(output)[0] == 60


def test_stream_end_can_verify_missing_final_frame_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.mp4"
    source.touch()
    payload = _payload([100, 110, 120])
    del payload["frames"][-1]["pkt_duration"]
    payload["streams"][0].update(start_pts=100, duration_ts=30)
    _mock_probe(monkeypatch, payload)
    require_cfr_timing(source)


@needs_ffmpeg
def test_resumed_wizard_render_keeps_previous_output_on_timing_rejection(
    tmp_path: Path,
) -> None:
    source = _clip(tmp_path / "vfr.mp4", vfr=True)
    controller = WizardController(tmp_path / "jobs", upscale_engine=_NoWorkEngine())
    manifest = controller.store.create(source)
    job_dir = controller.store.job_dir(manifest.job_id)
    output = job_dir / "output.mp4"
    output.write_bytes(b"previous result")
    save_session_state(
        controller.jobs_dir,
        WizardState(step=WizardStep.RENDERING, job_id=manifest.job_id),
    )
    controller.run_render()
    state = controller.current_state()
    assert state.step is WizardStep.RESULT
    assert "VFR/unsupported" in (state.error or "")
    assert output.read_bytes() == b"previous result"
