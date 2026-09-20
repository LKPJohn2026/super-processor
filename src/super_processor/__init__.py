"""Public package interface for Super Processor."""

from importlib.metadata import PackageNotFoundError, version

from ._core import (
    histogram_u8,
    mean_luma,
    percentile_u8,
    sad_u8,
    sum_squares,
    variance_u8,
)
from .jobs import (
    InvalidTransitionError,
    JobError,
    JobManifest,
    JobNotFoundError,
    JobState,
    JobStore,
    default_jobs_root,
)
from .probe import MediaFacts, ProbeError, StreamFacts, probe_file
from .recipe import OpName, Recipe, RecipeError, empty_recipe

try:
    __version__ = version("super-processor")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = [
    "InvalidTransitionError",
    "JobError",
    "JobManifest",
    "JobNotFoundError",
    "JobState",
    "JobStore",
    "MediaFacts",
    "OpName",
    "ProbeError",
    "Recipe",
    "RecipeError",
    "StreamFacts",
    "__version__",
    "default_jobs_root",
    "empty_recipe",
    "histogram_u8",
    "mean_luma",
    "percentile_u8",
    "probe_file",
    "sad_u8",
    "sum_squares",
    "variance_u8",
]
