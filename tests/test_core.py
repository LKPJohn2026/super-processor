import pytest

from super_processor import mean_luma, sum_squares


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (1, 0),
        (5, 30),
        (100, 328_350),
    ],
)
def test_sum_squares(count: int, expected: int) -> None:
    assert sum_squares(count) == expected


def test_sum_squares_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        sum_squares(-1)


@pytest.mark.parametrize(
    "samples",
    [
        bytes((0, 64, 128, 255)),
        bytearray((0, 64, 128, 255)),
        memoryview(bytes((0, 64, 128, 255))),
    ],
)
def test_mean_luma_accepts_contiguous_buffers(
    samples: bytes | bytearray | memoryview,
) -> None:
    assert mean_luma(samples) == pytest.approx(111.75)


def test_mean_luma_rejects_empty_buffer() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        mean_luma(b"")
