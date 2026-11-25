"""The undervolt note: GPU heat readings and the slower-and-hotter rule."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from super_processor.gpu_heat import (
    HOT_C,
    UNDERVOLT_NOTE,
    WARMUP_DONE,
    WINDOW_S,
    GpuHeat,
    SlowdownWatch,
    read_gpu_heat,
    slowing_from_heat,
)
from super_processor.wizard import WizardController, _aged

COOL = GpuHeat(temp_c=65.0, power_w=480.0, thermal_slowdown=False)
HOT = GpuHeat(temp_c=HOT_C, power_w=500.0, thermal_slowdown=False)
THROTTLED = GpuHeat(temp_c=70.0, power_w=420.0, thermal_slowdown=True)


def _smi(
    monkeypatch: pytest.MonkeyPatch, replies: dict[str, tuple[int, str]]
) -> list[str]:
    """Answer nvidia-smi by the field-name prefix it is asked for."""
    asked: list[str] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        query = command[1]
        prefix = next(name for name in replies if name in query)
        asked.append(prefix)
        code, out = replies[prefix]
        return subprocess.CompletedProcess(command, code, out, "")

    monkeypatch.setattr(subprocess, "run", run)
    return asked


def test_reads_temperature_power_and_thermal_slowdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _smi(
        monkeypatch,
        {"clocks_event_reasons": (0, "83, 498.5, Not Active, Active, Not Active\n")},
    )
    assert read_gpu_heat() == GpuHeat(83.0, 498.5, True)


def test_falls_back_to_the_older_field_names(monkeypatch: pytest.MonkeyPatch) -> None:
    asked = _smi(
        monkeypatch,
        {
            "clocks_event_reasons": (2, "Field is not a valid field to query.\n"),
            "clocks_throttle_reasons": (
                0,
                "61, 120, Not Active, Not Active, Not Active",
            ),
        },
    )
    assert read_gpu_heat() == GpuHeat(61.0, 120.0, False)
    assert asked == ["clocks_event_reasons", "clocks_throttle_reasons"]


@pytest.mark.parametrize(
    "out",
    [
        "",
        "83, 498.5\n",
        "[N/A], 498.5, Active, Active, Active\n",
        "83, [N/A], Active, Active, Active\n",
    ],
)
def test_unreadable_answers_give_no_reading(
    monkeypatch: pytest.MonkeyPatch, out: str
) -> None:
    _smi(
        monkeypatch,
        {"clocks_event_reasons": (0, out), "clocks_throttle_reasons": (0, out)},
    )
    assert read_gpu_heat() is None


def test_a_tool_that_refuses_both_field_names_gives_no_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _smi(
        monkeypatch,
        {
            "clocks_event_reasons": (6, "No devices were found\n"),
            "clocks_throttle_reasons": (6, "No devices were found\n"),
        },
    )
    assert read_gpu_heat() is None
    assert asked == ["clocks_event_reasons", "clocks_throttle_reasons"]


def test_no_nvidia_tool_gives_no_reading(tmp_path: Path) -> None:
    assert read_gpu_heat(str(tmp_path / "no-nvidia-smi")) is None


def test_a_hung_tool_gives_no_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 10)

    monkeypatch.setattr(subprocess, "run", run)
    assert read_gpu_heat() is None


@pytest.mark.parametrize(
    ("recent", "heat", "noted"),
    [
        (0.8, HOT, True),  # exactly 20% slower at exactly 80 C
        (0.5, GpuHeat(HOT_C - 0.1, 500.0, False), False),  # just under 80 C
        (0.5, THROTTLED, True),  # the driver slows a cooler GPU for heat
        (0.81, HOT, False),  # 19% slower is within the scene-to-scene spread
        (0.5, COOL, False),  # slower but cool: something else slowed it
        (1.2, THROTTLED, False),  # hot but not slower
    ],
)
def test_the_rule_needs_both_slower_and_hot(
    recent: float, heat: GpuHeat, noted: bool
) -> None:
    assert slowing_from_heat(1.0, recent, heat) is noted


def test_a_render_with_no_baseline_speed_is_never_slower() -> None:
    assert not slowing_from_heat(0.0, 0.0, THROTTLED)
    assert not slowing_from_heat(-0.1, -0.5, THROTTLED)


def _render(
    watch: SlowdownWatch, start: float, stop: float, rate: float, done: float
) -> float:
    """Feed progress every 5 s from ``start`` to ``stop`` seconds at ``rate``
    of the render per second, from ``done``; the share done after."""
    now = start
    while now <= stop:
        watch.progress(now, done)
        done += rate * 5
        now += 5
    return done


def test_notes_once_when_the_render_slows_while_hot() -> None:
    watch = SlowdownWatch()
    watch.heat(HOT)
    done = _render(watch, 0, WINDOW_S * 2, 0.001, WARMUP_DONE)
    assert watch.note() is None  # steady speed
    assert watch.recent_rate() == pytest.approx(0.001)
    _render(watch, WINDOW_S * 2 + 5, WINDOW_S * 4, 0.0007, done)
    # The recent speed is the last minute's alone, not the whole render's.
    assert watch.recent_rate() == pytest.approx(0.0007)
    assert watch.note() == UNDERVOLT_NOTE
    assert watch.note() is None  # only once per render


def test_no_note_while_the_gpu_is_cool_or_unread() -> None:
    unread = SlowdownWatch()
    cool = SlowdownWatch()
    cool.heat(COOL)
    cool.heat(None)  # a failed reading keeps the last good one
    for watch in (unread, cool):
        done = _render(watch, 0, WINDOW_S * 2, 0.001, WARMUP_DONE)
        _render(watch, WINDOW_S * 2 + 5, WINDOW_S * 4, 0.0005, done)
        assert watch.note() is None


def test_warm_up_and_a_short_render_set_no_baseline() -> None:
    watch = SlowdownWatch()
    watch.heat(THROTTLED)
    # Fast warm-up below WARMUP_DONE is ignored, so it cannot be the baseline.
    watch.progress(0.0, 0.0)
    watch.progress(1.0, WARMUP_DONE / 2)
    assert watch.recent_rate() is None
    _render(watch, 2, WINDOW_S / 2, 0.001, WARMUP_DONE)
    assert watch.note() is None  # under a minute: no baseline, no recent speed
    assert watch.recent_rate() is None


def test_the_baseline_is_the_first_minute_after_warm_up() -> None:
    # A render that starts slowly while hot (a cold cache, say) and stays
    # at that speed has not slowed down; only a drop from its own first
    # minute counts, so a later fast stretch does not raise the bar.
    watch = SlowdownWatch()
    watch.heat(THROTTLED)
    done = _render(watch, 0, WINDOW_S, 0.0005, WARMUP_DONE)
    done = _render(watch, WINDOW_S + 5, WINDOW_S * 3, 0.001, done)
    _render(watch, WINDOW_S * 3 + 5, WINDOW_S * 5, 0.0005, done)
    assert watch.recent_rate() == pytest.approx(0.0005)
    assert watch.note() is None


def test_the_wizard_shows_the_note_under_the_render_progress(tmp_path: Path) -> None:
    controller = WizardController(tmp_path / "jobs")
    state = controller.current_state()
    shown: list[float] = []

    class Noting(SlowdownWatch):
        def note(self) -> str | None:
            return UNDERVOLT_NOTE

    tracked = controller._watched(Noting(), shown.append)
    tracked(0.5)
    assert shown == [0.5]
    controller._progress(state, "rendering", 50, "Upscaling on the GPU: 50%")
    assert state.progress is not None
    assert state.progress["note"] == UNDERVOLT_NOTE
    # What /api/state sends while the render runs.
    sent = _aged(state.progress, time.time())
    assert sent is not None and sent["note"] == UNDERVOLT_NOTE
    controller._heat_note = None
    controller._progress(state, "rendering", 60, "Upscaling on the GPU: 60%")
    assert "note" not in state.progress
