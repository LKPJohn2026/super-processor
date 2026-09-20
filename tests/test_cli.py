import argparse

import pytest

from super_processor import __version__, cli


def test_version_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_self_test_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["self-test", "--sample-size", "5"]) == 0
    output = capsys.readouterr().out
    assert "Cython extension OK" in output
    assert "sum_squares(5)=30" in output


def test_self_test_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def incorrect_sum_squares(_: int) -> int:
        return -1

    monkeypatch.setattr(cli, "sum_squares", incorrect_sum_squares)

    assert cli.run_self_test(5) == 1
    assert "self-test failed" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_non_positive_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="greater than zero"):
        cli.positive_int(value)
