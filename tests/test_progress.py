"""Overall percentage and time-left estimates for wizard workers."""

from __future__ import annotations

import pytest

from super_processor.progress import (
    PREVIEWS_BAND,
    RENDER_BAND,
    StepClock,
    overall_pct,
)
from super_processor.wizard import _aged


def test_overall_moves_forward_through_the_job() -> None:
    steps = [
        overall_pct("sampling", 0),
        overall_pct("proposing", 100),
        overall_pct("analyzing", 0, segment_index=0, segment_count=3),
        overall_pct("analyzing", 100, segment_index=0, segment_count=3),
        overall_pct("analyzing", 50, segment_index=2, segment_count=3),
        overall_pct("analyzing", 100, segment_index=2, segment_count=3),
        overall_pct("rendering", 0),
        overall_pct("rendering", 100),
    ]
    assert steps == sorted(steps)
    assert steps[0] == 0 and steps[-1] == 100
    assert steps[2] == PREVIEWS_BAND[0]
    assert steps[3] == PREVIEWS_BAND[0] + 10
    assert steps[6] == RENDER_BAND[0]


def test_a_revise_render_counts_on_its_own() -> None:
    assert overall_pct("rendering", 40, revising=True) == 40
    # Out-of-range input stays inside its band.
    assert overall_pct("analyzing", 250, segment_index=9, segment_count=2) == 38


def test_steady_work_gives_the_true_time_left() -> None:
    clock = StepClock(started_at=0.0)
    # The first report comes after a 60 s model load; it must not count.
    assert clock.update(0.0, 60.0) is None
    left = None
    for second in range(61, 161):
        left = clock.update((second - 60) / 400, float(second))
    # 100 s for a quarter of the work: 300 s to go.
    assert left == pytest.approx(300.0, rel=0.01)
    assert clock.elapsed(160.0) == 160.0


def test_the_estimate_follows_a_lasting_slowdown() -> None:
    clock = StepClock(started_at=0.0)
    clock.update(0.0, 0.0)
    done, now = 0.0, 0.0
    for _ in range(100):  # 0.5% a second
        now += 1.0
        done += 0.005
        clock.update(done, now)
    for _ in range(120):  # then half as fast
        now += 1.0
        done += 0.0025
        left = clock.update(done, now)
    true_left = (1 - done) / 0.0025
    average_only = (1 - done) / (done / now)
    assert left is not None
    # Closer to the new speed than the step's average alone would be.
    assert abs(left - true_left) < abs(average_only - true_left)


def test_no_estimate_until_there_is_something_to_go_on() -> None:
    clock = StepClock(started_at=0.0)
    assert clock.time_left(5.0) is None
    clock.update(0.1, 1.0)
    assert clock.update(0.1, 50.0) is None  # no movement
    assert clock.update(0.101, 51.0) is None  # too little measured
    # Time passing without a report counts down the estimate.
    clock.update(0.2, 101.0)
    left = clock.time_left(101.0)
    assert left is not None
    assert clock.time_left(131.0) == pytest.approx(left - 30.0)


def test_the_view_counts_down_between_saves() -> None:
    saved = {"pct": 50, "eta_s": 100.0, "elapsed_s": 20.0, "updated_at": 1000.0}
    aged = _aged(saved, 1012.5)
    assert aged is not None
    assert aged["eta_s"] == 87.5 and aged["elapsed_s"] == 32.5
    assert _aged({**saved, "eta_s": None}, 2000.0)["eta_s"] is None  # type: ignore[index]
    assert _aged(saved, 9999.0)["eta_s"] == 0.0  # type: ignore[index]
    assert _aged(None, 1.0) is None
    assert _aged({"pct": 5}, 1.0) == {"pct": 5}
