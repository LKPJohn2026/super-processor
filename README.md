# Super Processor

[![CI](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml)

Local-first AI-assisted video processing without generative frame synthesis.

Super Processor enhances existing footage. A localhost wizard runs a local
FlashVSR restoration upscale, then lets you revise a time range with a note;
Gemini structured output only turns that note into a range, scale, and
strength. FFmpeg trims, splices, and encodes from engineering-owned argument
lists. Models never write shell commands.

## Requirements

- Python 3.10 or newer and a C compiler (Cython extension)
- FFmpeg with `libx265` (and optionally `libvmaf` for offline regression)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/) for the wizard

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

export GEMINI_API_KEY=your_key   # or paste it in the wizard setup screen
super-processor doctor
super-processor review
```

The wizard walks through Gemini setup, file pick (up to 1080p and 30 minutes),
a FlashVSR upscale on an NVIDIA GPU, and a result screen where a note re-runs
one time range.

Legacy single-grade and segment CLI commands remain available:

```bash
super-processor diagnose ./clip.mp4 --max-size-mb 80
super-processor plan JOB_ID --instruction "less denoise"
super-processor preview JOB_ID --open auto
super-processor apply JOB_ID --approve
super-processor show JOB_ID
```

See [docs/ui-wizard.md](docs/ui-wizard.md) and
[docs/system-design.md](docs/system-design.md).

The `web/` app is a React shell for that wizard. GitHub Pages can host it as
a static preview. The GPU upscale still runs only from `super-processor review`.
Locally: `cd web && npm install && npm run dev` (proxy target
`VITE_PROXY_TARGET`, default `http://127.0.0.1:45143`).

## Development checks

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest
python -m build
super-processor self-test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).
