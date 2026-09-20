"""Command-line entry point for Super Processor."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__, mean_luma, sum_squares


def positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="super-processor",
        description="Local-first AI-assisted video processing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="print the installed version")

    self_test = subparsers.add_parser(
        "self-test",
        help="verify that the compiled Cython extension is working",
    )
    self_test.add_argument(
        "--sample-size",
        type=positive_int,
        default=10_000,
        help="number of integers used by the compiled test (default: 10000)",
    )
    return parser


def run_self_test(sample_size: int) -> int:
    """Exercise compiled functions and return a process exit code."""
    observed = sum_squares(sample_size)
    expected = sample_size * (sample_size - 1) * (2 * sample_size - 1) // 6
    luma = mean_luma(bytes((16, 64, 128, 235)))

    if observed != expected or luma != 110.75:
        print("Cython extension self-test failed")
        return 1

    print(
        "Cython extension OK "
        f"(sum_squares({sample_size})={observed}, mean_luma={luma:.2f})"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""
    arguments = build_parser().parse_args(argv)
    command = str(arguments.command)

    if command == "version":
        print(__version__)
        return 0
    if command == "self-test":
        return run_self_test(int(arguments.sample_size))

    raise AssertionError(f"unhandled command: {command}")
