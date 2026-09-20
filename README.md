# Super Processor

[![CI](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml)

Local-first AI-assisted video processing without generative frame synthesis.

Super Processor enhances existing footage through measured, constrained FFmpeg
pipelines. Computer-vision estimators propose a validated Recipe JSON; optional
local/BYOK planners may refine that recipe. Models never write shell commands
or generate replacement frames.

## Requirements

- Python 3.10 or newer and a C compiler (Cython extension)
- FFmpeg with `libx265` (and optionally `libvmaf` for offline regression)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

super-processor doctor
super-processor diagnose ./clip.mp4 --max-size-mb 80
super-processor plan JOB_ID --instruction "less denoise"
super-processor preview JOB_ID --open auto
super-processor apply JOB_ID --approve
super-processor show JOB_ID
```

Safe mode is on by default (`--safe-mode`): planning uses CV priors and
deterministic NL patches. Pass `--no-safe-mode` to allow advisory local LLM
calls that still cannot emit shell commands.

When a `--max-size-mb` cap implies a bitrate below the quality floor, diagnosis
and validation fail unless you pass `--acknowledge-size-risk`.

```bash
super-processor models
super-processor models --consent-remote-frames
```

See [CHANGELOG.md](CHANGELOG.md) for 1.x hardening notes.

## Development checks

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
python -m build
super-processor self-test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/system-design.md](docs/system-design.md).
