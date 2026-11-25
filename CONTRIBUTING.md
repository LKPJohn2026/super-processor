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
npm run build:package   # the app super-processor review serves
npm run check:package
```

`npm run build` makes the GitHub Pages build in `web/dist`.
`npm run build:package` writes the app the wizard serves into
`src/super_processor/web_dist`; restart `super-processor review` to use it.
That folder is ignored by git: CI and the release build it and put it in the
sdist and every wheel. Without it, `review` serves a short page saying how to
build it or how to point the Vite dev server at the wizard
(`VITE_PROXY_TARGET=<printed address> npm run dev`).

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
exactly pinned build dependencies declared in `pyproject.toml` (CI and the
release use the hash-checked copies instead; see [Supply chain](#supply-chain)); `twine check` validates their
metadata without uploading anything. The sdist must also retain the tracked
`web/` sources, especially `web/index.html`, and the built app in
`src/super_processor/web_dist` when it is there (run `npm run build:package`
first, or the packages have no app). Only generated Cython annotation HTML
under `src/super_processor/` is excluded, not all HTML in the repository.
`tests/test_packaging.py` checks these manifest rules with setuptools, an
existing development/build dependency. Set
`SUPER_PROCESSOR_REQUIRE_WEB_DIST=1` to make the build fail when the app is
missing, as CI and the release do.

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

### Supply chain

The release workflow can publish to PyPI and write GitHub Releases, so
everything it runs is pinned:

- Every action in `.github/workflows/` is pinned to a full commit SHA with the
  version as a comment (`uses: actions/checkout@<sha> # v7.0.1`). Dependabot's
  `github-actions` entry bumps the SHA and the comment together. Never go back
  to a tag or branch such as `@v7` or `@release/v1`.
- Workflow-level `permissions` are read-only. Only the jobs that need more get
  it: `github-release` (`contents: write`, runs only GitHub's own actions and
  the `gh` CLI), `pypi` (`id-token: write`, in the `pypi` environment) and the
  Pages `deploy` job (`pages: write`, `id-token: write`, in `github-pages`, on
  pushes to `main` only). Pull-request runs never get a write token.
- The build toolchain is exact. `[build-system].requires` in `pyproject.toml`
  uses `==` pins that match `.github/requirements/build.in`. The release (and
  the CI `package` job) installs `.github/requirements/build.txt` and
  `tools.txt` with `pip install --require-hashes` and builds with
  `--no-isolation`, so nothing is resolved fresh from PyPI when a tag is
  pushed. cibuildwheel does the same through `CIBW_BEFORE_BUILD`,
  `CIBW_BEFORE_TEST` and `CIBW_BUILD_FRONTEND: "build; args: --no-isolation"`.
- `tests/test_workflow_hardening.py` fails if any of this regresses: an
  unpinned `uses:`, a workflow-level write permission, `${{ github.event.* }}`
  inside a `run:` script, an unhashed `pip install` in `release.yml`, or
  build-system requirements that are not exact pins matching `build.txt`.

To change a build tool version, edit `pyproject.toml` and
`.github/requirements/build.in` (or `tools.in`) together, then regenerate the
locks with the commands at the top of `build.txt` and `tools.txt` (run them
from the repository root; `uv pip compile` only writes the output file and
does not touch `.venv`). Bump the `--exclude-newer` date in both commands to
about two weeks before today, so a release that appeared only days ago is not
picked up.

These settings live in GitHub, not in the repository, and must be set by a
maintainer:

- **Branch protection for `main`**: block force pushes and deletion, and
  require pull requests with the CI checks (`Quality`, `Test (...)`,
  `Frontend regressions and build`, `Build distributions`) to pass.
- **Tag ruleset for `v*`**: only maintainers may create tags, and no one may
  move or delete them.
- **`pypi` environment**: add a required reviewer and limit deployments to
  tags matching `v*`. Bind the PyPI trusted publisher to the workflow
  `release.yml` and the environment `pypi`.
- **`github-pages` environment**: limit deployments to the `main` branch.
- **Actions settings**: turn on "Require actions to be pinned to a full-length
  commit SHA".
- **Security settings**: turn on Dependabot alerts and Dependabot security
  updates (version updates in `.github/dependabot.yml` do not replace them).
