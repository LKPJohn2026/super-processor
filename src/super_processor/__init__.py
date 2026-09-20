"""Public package interface for Super Processor."""

from importlib.metadata import PackageNotFoundError, version

from ._core import mean_luma, sum_squares

try:
    __version__ = version("super-processor")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = ["__version__", "mean_luma", "sum_squares"]
