"""Public package interface for Super Processor."""

from importlib.metadata import PackageNotFoundError, version

from ._core import mean_luma, sum_squares
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
    "mean_luma",
    "probe_file",
    "sum_squares",
]
