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

Run these routine checks before opening a pull request (no live API calls
or GPU work):

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -m 'not gemini_live'
python -m build
python -m twine check dist/*
```

### Live Gemini tests (explicit opt-in only)

Keep `-m 'not gemini_live'` on routine runs even if a key is already present:
plain `pytest` can run live tests when `GEMINI_API_KEY` is set. To intentionally
exercise the real service, configure a dedicated test key outside tracked files
and run this separately:

```bash
python -m pytest -m gemini_live --no-cov
```

Without `GEMINI_API_KEY`, these tests skip. CI maps its `Gemini_API_Test` secret
to that variable only in the separate live job. Live tests send test prompts
and context to Google, consume quota, and may incur charges; local chat logs
may retain requests/responses. Do not supply production secrets, private footage,
or personal data. Manual wizard calls also send selected stills/previews and
notes. `--no-cov` avoids applying the whole-suite coverage threshold to this
small live-only selection. No GPU is needed for these live tests.

### Frontend checks

For frontend integration changes, use the installed Node/TypeScript/Vite tools
(no API key, GPU or browser needed):

```bash
cd web
npm ci
npm test
npm run build
```

The Node regressions simulate effect subscription lifecycles and failed/hung
requests and exercise real dev/preview proxy forwarding. Browser end-to-end
acceptance is separate; `npm run a11y` needs a built app and an already-installed
Playwright Chromium. See [connection and recovery](docs/ui-wizard.md#react-shell-connection-and-recovery)
for the printed backend port and manual browser checks.

To apply automatic formatting:

```bash
python -m ruff format .
python -m ruff check --fix .
```

Do not commit generated extension sources, compiled binaries, wheels, coverage
reports, caches, or `*.egg-info` directories.

### Packaging smoke check

`python -m build` builds the sdist, then a wheel from that sdist, using the
build dependencies declared in `pyproject.toml`; `twine check` validates their
metadata without uploading anything. The sdist must also retain the tracked
`web/` sources, especially `web/index.html`. Only generated Cython annotation
HTML under `src/super_processor/` is excluded, not all HTML in the repository.
`tests/test_packaging.py` checks these manifest rules with setuptools, an
existing development/build dependency.

For a clean-source frontend check, build into an ignored directory:

```bash
python -m build --outdir .dogfood/packaging/dist
python -m twine check .dogfood/packaging/dist/*
mkdir -p .dogfood/packaging/unpacked
# Replace VERSION with the version in the generated sdist filename.
tar -xzf .dogfood/packaging/dist/super_processor-VERSION.tar.gz -C .dogfood/packaging/unpacked
cd .dogfood/packaging/unpacked/super_processor-VERSION/web
npm ci
npm run build
```

Use a fresh output/unpack directory for each check to avoid validating stale
artifacts. `npm ci` may fetch locked development dependencies from the registry;
these checks do not contact Gemini or run restoration/GPU jobs. Keep build and
unpacked artifacts ignored and unstaged.

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
