# Super Processor

[![CI](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml)

Local-first AI video restoration: FlashVSR upscaling, with a strength
control for how much new detail it adds.

Super Processor enhances existing footage. A localhost wizard has Gemini split
the video into segments from a grid of local stills, then plans each segment
in turn with bounded settings and a short before/after preview you approve or
send back with a note. It runs a local FlashVSR restoration upscale (or a
plain resize where FlashVSR would only harm), then lets you revise a time
range with a note. Gemini structured output only labels, picks numbers inside
fixed bounds, and checks previews. FFmpeg detects, measures, trims, splices, and encodes from
engineering-owned argument lists. Models never write shell commands.

## Requirements

- Python 3.10 or newer and a C compiler (Cython extension)
- FFmpeg 6 or newer and `ffprobe`, with both `libx264` (previews and
  restoration delivery) and `libx265` (HEVC export). `super-processor doctor`
  requires both encoders for readiness; hardware codecs are optional.
  The wizard uses Gemini to propose segments from local stills, then measures
  those ranges with `blockdetect`, `blurdetect`, `signalstats`, `hqdn3d`, and
  `psnr`. `scdet` is used by legacy shot detection, not wizard segmentation.
  `libvmaf` is optional for offline regression.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/) for the wizard
- For the upscale itself: an NVIDIA GPU with CUDA, PyTorch, and a FlashVSR
  v1.1 checkout with its weights (see [GPU setup](#gpu-setup))

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
segment review, a before/after preview of each segment in turn, a FlashVSR
upscale on an NVIDIA GPU, and a result screen where a note re-runs one time
range.

The wizard and restoration entry points currently require verified constant
frame rate (CFR). Before a job is created or restoration starts, a CPU decode
checks the complete presentation timeline, not just reported frame-rate
metadata. VFR, nonuniform timing, or timing that cannot be verified is refused
before GPU work or replacing an existing output. There is no automatic
normalization: external CFR conversion may change frames/timing. This is an
interim safety limit, not VFR support.

Without a GPU, the wizard works up to the Segments screen (the still grid and
Gemini's segments). The segment previews and the upscale need FlashVSR on
CUDA.

## GPU setup

Install the GPU extra:

```bash
python -m pip install -e ".[dev,gpu]"
```

Clone FlashVSR where the wizard looks for it (`.dogfood/models/FlashVSR`
under the folder you run from, or set `FLASHVSR_HOME` to another checkout;
`SUPER_PROCESSOR_MODELS` moves the whole models folder):

```bash
git clone https://github.com/OpenImagingLab/FlashVSR .dogfood/models/FlashVSR
```

Download the `JunhaoZhuang/FlashVSR-v1.1` weights into
`.dogfood/models/FlashVSR/examples/WanVSR/FlashVSR-v1.1/`. The wizard needs
these three files there:

- `diffusion_pytorch_model_streaming_dmd.safetensors`
- `LQ_proj_in.ckpt`
- `TCDecoder.ckpt`

Install FlashVSR's own dependencies as its README describes. Then check each
piece:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from super_processor.upscale import resolve_flashvsr_weights as r; print(r())"
super-processor doctor
ffmpeg -hide_banner -filters | grep -E ' (blockdetect|blurdetect|signalstats|hqdn3d|psnr) '
```

The first should print `CUDA: True`, the second the weights folder (it raises
with the missing file names otherwise). Doctor should report required tools
ready, and the final command should show all five wizard measurement filters.
Legacy shot detection additionally needs `scdet`.

## Testing on your own footage

```bash
export GEMINI_API_KEY=your_key
mkdir -p .dogfood/jobs
super-processor --jobs-dir .dogfood/jobs review
```

Open the printed URL and go pick → segments → previews → upscale → result. Short
clips (30–60 seconds) with one clear problem each are the most useful: blocky
web video, noisy low light, old SD footage, a face close-up, and a shot with
a sign or text. On each screen, note where the tool was wrong:

- **Segments**: cuts in the wrong place, or contents (faces, hands, text)
  missed (`segments.json`; `shots.json` holds the measurements).
- **Previews**: waxy skin, changed faces, garbled text, or crawling texture in a
  preview; face and text boxes in the wrong place; a shimmer number that does
  not match what you see (`shot_plan.json`).
- **Result**: seam warnings with no visible seam, or visible seams with no
  warning (`render_report.json`).

Each job folder under `.dogfood/jobs/` keeps those files, the previews, and
`gemini_chat.json`. The thresholds they report against are first guesses from
synthetic clips and are meant to be tuned from real footage:

| What | Where | Value |
|---|---|---|
| Problem hints | `shots.py` | blocky ≥ 18, soft ≥ 6, noisy ≥ 2, dark < 0.25, bright > 0.75, flat < 0.35 |
| Shimmer | `stability.py` `SHIMMER_LIMIT` | 1.6 |
| Seams | `stability.py` `SEAM_LIMIT` | 2.5 |
| Protected regions | `regions.py` `REGION_STRENGTH` | faces and hands 0.3, text 0.15 |

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
Start `super-processor review` first and copy its printed backend URL: the
server chooses an ephemeral port, not necessarily `45143`. In `web/`, install
with `npm ci`, then run `VITE_PROXY_TARGET=http://127.0.0.1:PRINTED_PORT npm run dev`
(replace `PRINTED_PORT` with the actual port). This proxies API, uploads, stills,
previews, and output. For a built app, use the same variable with
`npm run preview` after `npm run build`. The fallback proxy target
`http://127.0.0.1:45143` works only if that is the backend's actual address.

Alternatively, enter the printed backend URL in the local shell's connection
form or open `http://localhost:5173/?api=http://127.0.0.1:PRINTED_PORT` for a
direct connection (preview uses port `4173`). See
[frontend connection and recovery](docs/ui-wizard.md#react-shell-connection-and-recovery)
for CORS, HTTPS restrictions, and retry behavior. Offline frontend regressions:
`cd web && npm test && npm run build` (no API key, GPU, or browser required).

## Development checks

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -m 'not gemini_live'
python -m build
super-processor self-test
```

Routine tests exclude live Gemini calls even when `GEMINI_API_KEY` is set.
Live tests are a separate, explicit opt-in: with a test key configured, run
`python -m pytest -m gemini_live --no-cov`. These contact Google, send test
prompts/context, and consume API quota (potentially incurring charges).
Wizard use also sends selected stills/previews and notes; do not use sensitive
footage without permission. See [CONTRIBUTING.md](CONTRIBUTING.md) for testing
and packaging checks, and [CHANGELOG.md](CHANGELOG.md) for changes.
