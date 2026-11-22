"""Notice a render that slows down while the GPU runs hot, and say so once.

A SPAN render keeps an RTX 5090 near 500 W for as long as it runs. On a
20-minute 1080p film the case warmed until the CPU, though only a third
busy, reached 92 C (its limit is 95 C). The app does not slow itself down
for heat: when the render gets slower while the GPU is hot or throttling,
the progress screen suggests undervolting the GPU, which usually keeps
nearly all the speed for much less heat.

GPU readings come from ``nvidia-smi``; without it nothing is noticed.
"""

from __future__ import annotations

import subprocess
from collections import deque
from dataclasses import dataclass

from .tools import resolve_argv

UNDERVOLT_NOTE = (
    "Your GPU is running hot and the render has slowed down to keep it cool. "
    "Undervolting the GPU (a lower voltage curve or power limit in your GPU "
    "tool) usually keeps nearly all the speed with much less heat."
)
# Rates are measured over this much render time.
WINDOW_S = 60.0
# The first part of a render warms caches and the GPU clock up; its speed is
# not a fair baseline.
WARMUP_DONE = 0.02
# The note shows when the last minute ran at least this much slower than the
# first, while the driver slows the GPU for heat or it is at least HOT_C.
# Encode cost varies about 10% between scenes, so a smaller drop is noise;
# an RTX 5090 starts lowering its clocks in the low 80s C.
SLOWER_BY = 0.20
HOT_C = 80.0
# Reasons the driver lowers the GPU clock for heat. Drivers from 2024 on name
# them clocks_event_reasons; older ones clocks_throttle_reasons.
_THERMAL_REASONS = ("sw_thermal_slowdown", "hw_thermal_slowdown", "hw_slowdown")


@dataclass(frozen=True, slots=True)
class GpuHeat:
    """One reading of the GPU's heat."""

    temp_c: float
    power_w: float
    # The driver is lowering the clock for heat right now.
    thermal_slowdown: bool


def read_gpu_heat(nvidia_smi: str = "nvidia-smi") -> GpuHeat | None:
    """The first GPU's temperature, power, and thermal slowdown, or None."""
    for prefix in ("clocks_event_reasons", "clocks_throttle_reasons"):
        fields = ",".join(
            ["temperature.gpu", "power.draw"]
            + [f"{prefix}.{reason}" for reason in _THERMAL_REASONS]
        )
        try:
            completed = subprocess.run(
                resolve_argv(
                    [
                        nvidia_smi,
                        f"--query-gpu={fields}",
                        "--format=csv,noheader,nounits",
                    ]
                ),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            continue
        line = (completed.stdout.splitlines() or [""])[0]
        values = [value.strip() for value in line.split(",")]
        if len(values) != 2 + len(_THERMAL_REASONS):
            continue
        try:
            temp, power = float(values[0]), float(values[1])
        except ValueError:
            continue
        return GpuHeat(temp, power, any(v == "Active" for v in values[2:]))
    return None


def slowing_from_heat(baseline_rate: float, recent_rate: float, heat: GpuHeat) -> bool:
    """True when the render is slower than it started and the GPU is hot.

    ``baseline_rate`` and ``recent_rate`` are the share of the render done
    per second, over its first minute after warm-up and over its last
    minute. ``heat`` is the latest GPU reading.
    """
    if baseline_rate <= 0:
        return False
    slower = recent_rate <= baseline_rate * (1 - SLOWER_BY)
    hot = heat.thermal_slowdown or heat.temp_c >= HOT_C
    return slower and hot


class SlowdownWatch:
    """Follow a render's progress and GPU heat; say once when to suggest
    undervolting.

    Feed it ``progress(now, done)`` from the render's progress callback and
    ``heat(sample)`` from a sampler; ``note()`` then returns the note the
    first time :func:`slowing_from_heat` holds, and None otherwise.
    """

    def __init__(self) -> None:
        self._points: deque[tuple[float, float]] = deque()
        self._baseline: float | None = None
        self._start: tuple[float, float] | None = None
        self._heat: GpuHeat | None = None
        self._noted = False

    def progress(self, now: float, done: float) -> None:
        if done < WARMUP_DONE:
            return
        if self._start is None:
            self._start = (now, done)
        self._points.append((now, done))
        while len(self._points) > 1 and now - self._points[1][0] >= WINDOW_S:
            self._points.popleft()
        if self._baseline is None and now - self._start[0] >= WINDOW_S:
            self._baseline = _rate(self._start, (now, done))

    def heat(self, sample: GpuHeat | None) -> None:
        if sample is not None:
            self._heat = sample

    def recent_rate(self) -> float | None:
        """Share done per second over the last minute, once there is one."""
        if len(self._points) < 2:
            return None
        first, last = self._points[0], self._points[-1]
        if last[0] - first[0] < WINDOW_S * 0.9:
            return None
        return _rate(first, last)

    def note(self) -> str | None:
        if self._noted or self._baseline is None or self._heat is None:
            return None
        recent = self.recent_rate()
        if recent is None or not slowing_from_heat(self._baseline, recent, self._heat):
            return None
        self._noted = True
        return UNDERVOLT_NOTE


def _rate(first: tuple[float, float], last: tuple[float, float]) -> float:
    seconds = last[0] - first[0]
    return (last[1] - first[1]) / seconds if seconds > 0 else 0.0
