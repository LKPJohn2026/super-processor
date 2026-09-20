# Contributing

## Development environment

Super Processor supports CPython 3.10 and newer. A working C compiler is
required because editable installs compile the Cython extension.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Required checks

Run the same checks used by CI before opening a pull request:

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
python -m build
python -m twine check dist/*
```

To apply automatic formatting:

```bash
python -m ruff format .
python -m ruff check --fix .
```

Do not commit generated extension sources, compiled binaries, wheels, coverage
reports, caches, or `*.egg-info` directories.

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Update `README.md` or `docs/` when public behavior or decisions change.
- Explain any performance trade-off in Cython changes.
- Confirm that CI passes on Linux, macOS, and Windows.

## Releases

Releases use semantic version tags. `setuptools-scm` derives the package version
from Git, so no version file is edited manually.

```bash
git tag -s v0.1.0 -m "Super Processor 0.1.0"
git push origin v0.1.0
```

The release workflow builds and tests source and wheel distributions before
creating a GitHub Release. PyPI trusted publishing remains disabled unless the
repository variable `PYPI_PUBLISH` is set to `true` and the `pypi` environment
has been configured with a matching PyPI trusted publisher.
