# Super Processor

[![CI](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml)

Local-first AI-assisted video processing without generative frame synthesis.

Super Processor enhances existing footage through measured, constrained FFmpeg
pipelines. A localhost wizard uses Gemini structured outputs to propose
timeline splits and per-segment ops; engineering-owned templates build FFmpeg
argument lists. Models never write shell commands or generate replacement
frames.

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

The wizard walks through Gemini setup, file pick, split choice, per-segment
enhancement with short previews, then enhance-and-concat render (same aspect).

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
