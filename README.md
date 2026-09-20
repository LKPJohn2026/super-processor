# Super Processor

[![CI](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml)

Local-first AI-assisted video processing without generative frame synthesis.

Super Processor is being built around a constrained pipeline: computer-vision
measurements and optional language/vision models produce a validated recipe,
then local FFmpeg operations process the user's existing footage. Models never
write arbitrary shell commands or generate replacement frames.

The repository currently provides a clean, buildable Cython foundation for the
performance-sensitive analysis code. Read
[the system design](docs/system-design.md) for the product decisions and market
positioning.

## Included foundation

- PEP 517 packaging with setuptools, Cython, and `setuptools-scm`
- a compiled extension in `src/super_processor/_core.pyx`
- a typed Python API and installable `super-processor` command
- Ruff, mypy, pytest, and coverage configuration
- Linux, macOS, and Windows CI
- tagged, multi-platform GitHub Releases
- optional trusted publishing to PyPI

## Requirements

- Python 3.10 or newer
- a C compiler supported by the selected Python version
- FFmpeg will be required by the video-processing application, but is not
  required for the current toolchain self-test

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run all local checks:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
python -m build
python -m twine check dist/*
super-processor self-test
```

## CLI

```bash
super-processor version
super-processor self-test
super-processor doctor
super-processor doctor --json
super-processor job create ./clip.mp4
super-processor job show JOB_ID
super-processor job list
python -m super_processor self-test --sample-size 100
```

Jobs are stored under `~/.super-processor/jobs` by default. Override with
`--jobs-dir` or `SUPER_PROCESSOR_JOBS_DIR`.

## Project layout

```text
.
├── .github/workflows/
│   ├── ci.yml
│   └── release.yml
├── docs/system-design.md
├── src/super_processor/
│   ├── __init__.py
│   ├── __main__.py
│   ├── _core.pyx
│   ├── _core.pyi
│   ├── cli.py
│   └── py.typed
├── tests/
├── MANIFEST.in
├── pyproject.toml
└── setup.py
```

## Releases

Push a semantic-version tag to build platform wheels, create a source
distribution, verify every artifact, and publish a GitHub Release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

PyPI publishing is disabled by default. To enable trusted publishing, configure
the repository's `pypi` environment and PyPI trusted publisher, then set the
GitHub Actions repository variable `PYPI_PUBLISH` to `true`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development and release checks.
