def sum_squares(count: int) -> int:
    """Return the sum of squared integers below ``count``."""

def mean_luma(samples: bytes | bytearray | memoryview) -> float:
    """Return the arithmetic mean of a contiguous 8-bit luma buffer."""

def histogram_u8(
    samples: bytes | bytearray | memoryview,
    out_256: bytes | bytearray | memoryview,
) -> None:
    """Fill a 256-bin histogram for an 8-bit buffer."""

def percentile_u8(samples: bytes | bytearray | memoryview, percentile: float) -> float:
    """Return an approximate percentile from an 8-bit buffer via histogram."""

def variance_u8(samples: bytes | bytearray | memoryview) -> float:
    """Return the population variance of an 8-bit buffer."""

def sad_u8(
    left: bytes | bytearray | memoryview,
    right: bytes | bytearray | memoryview,
) -> int:
    """Return the sum of absolute differences between two equal-length buffers."""
