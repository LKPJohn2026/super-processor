# Super Processor

[![CI](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/LKPJohn2026/super-processor/actions/workflows/ci.yml)

Local-first AI video restoration: SPAN 2x upscaling, with a strength
control for how much new detail it adds.

Super Processor enhances existing footage. A localhost wizard has Gemini watch
a small, silent copy of the video, split it into segments, and choose the
output size, then plans each segment in turn with bounded settings and a short
before/after preview you approve or send back with a note. It runs a local
SPAN restoration upscale (or a plain resize where the upscaler would only
harm), then lets you revise a time range with a note. Gemini structured output
only labels, picks numbers inside fixed bounds, and checks previews. FFmpeg
detects, measures, trims, splices, and encodes from engineering-owned argument
lists. Models never write shell commands.

## Requirements

- Python 3.10 or newer and a C compiler (Cython extension)
- FFmpeg 6 or newer and `ffprobe`, with `libx264`. Every output is H.264:
  restoration delivery uses `h264_nvenc` when a test encode works on this
  machine and `libx264` otherwise, and the CLI export is `libx264` (or an
  H.264 hardware encoder named with `--encoder`). `super-processor doctor`
  requires `libx264` and reports whether NVENC and CUDA decoding actually
  run here. The
  wizard makes a small, audio-free copy for Gemini to watch, then measures the
  segments Gemini proposes with `blockdetect`, `blurdetect`, `signalstats`,
  `hqdn3d`, and `psnr`. `scdet` is used by legacy shot detection, not wizard
  segmentation. `libvmaf` is optional for offline regression.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/) for the wizard
- For the upscale itself: an NVIDIA GPU with CUDA, PyTorch, spandrel, and the
  SPAN weights file (see [GPU setup](#gpu-setup))

## Quick start

Releases are on PyPI as `ancient-processor`; the command it installs is
still `super-processor`:

```bash
python -m pip install ancient-processor        # add [gpu] for the upscale
```

From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

export GEMINI_API_KEY=your_key   # or paste it in the wizard setup screen
super-processor doctor
super-processor review
```

The wizard walks through Gemini setup, file pick (up to 1080p and 30 minutes),
segment review, a before/after preview of each segment in turn, a SPAN
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
Gemini's segments). The segment previews and the upscale need SPAN on CUDA.

## GPU setup

Install the GPU extra (PyTorch and spandrel, which loads the SPAN network):

```bash
python -m pip install -e ".[dev,gpu]"
```

Download the weights, `2xLiveActionV1_SPAN_490000.pth` (about 9 MB), to where
the wizard looks for them, `.dogfood/models/SPAN/` under the folder you run
from. The link is pinned to the commit that relicensed them as Apache-2.0
(see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)):

```bash
mkdir -p .dogfood/models/SPAN
curl -L -o .dogfood/models/SPAN/2xLiveActionV1_SPAN_490000.pth \
  https://github.com/jcj83429/upscaling/raw/5d8cdd2e17750b64be39ccab3a8763d91128fe15/2xLiveActionV1_SPAN/2xLiveActionV1_SPAN_490000.pth
sha256sum .dogfood/models/SPAN/2xLiveActionV1_SPAN_490000.pth
# 8b166c75831ea7f694d9058ee9c8df8148af8cc1d2b57e69e6581b15cab572f7
```

`SUPER_PROCESSOR_SPAN_WEIGHTS=/path/to/file.pth` reads the weights from
another place; `SUPER_PROCESSOR_MODELS` moves the whole models folder (the
file is then expected in its `SPAN/` folder). The wizard checks the file's
sha256 before it loads it and names the expected one if it differs. Then
check each piece:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from super_processor.span_sr import resolve_span_weights as r; from super_processor.upscale import models_dir as m; print(r(m()))"
super-processor doctor
ffmpeg -hide_banner -filters | grep -E ' (blockdetect|blurdetect|signalstats|hqdn3d|psnr) '
```

The first should print `CUDA: True`, the second the weights file (it raises
with the expected path or hash otherwise). Doctor should report required tools
ready, and the final command should show all five wizard measurement filters.
Legacy shot detection additionally needs `scdet`.

SPAN sees one frame at a time and the frames go to the GPU in small batches,
so GPU memory follows the batch, not the clip: about 4-9 GB for 1440x1080 to
2880x2160. Each clip streams in one pass with nearly all of the work on the
GPU. FFmpeg decodes the source with NVDEC (`-hwaccel cuda`; codecs NVDEC
cannot read fall back to software decoding) and applies the look's clean
filter. On the GPU the frames go from YUV to RGB, through SPAN and the
strength blend with a lanczos upscale, and back to YUV, and `h264_nvenc`
encodes them from a pipe, so nothing is written to scratch disk. Only a
look's grade and grain, when it has them, still run on the CPU. On an RTX
5090, 1500 frames of 1440x1080 to 2880x2160 ran at 26.8 frames a second end
to end with 0.054 CPU-seconds per frame, and the CPU was at 76 C after the
minute. The previous stream blended and encoded with x264 on the CPU: 22.2
frames a second at 0.43 CPU-seconds per frame (about 9.5 of 32 threads, half
of it in x264), and a 20-minute film took the CPU to 95 C. Without NVENC
(Macs, AMD GPUs, CI) the delivery encode falls back to `libx264`. If the
GPU runs out of memory, the upscale retries with smaller batches, down to one
frame, and then stops with advice to close other GPU programs.

### Why SPAN and not FlashVSR

Up to 3.7.0 the upscaler was FlashVSR v1.1 (or FlashVSR+). A bake-off on an
RTX 5090 found that FlashVSR's output, blended at strength 0.5 as the wizard
uses it, was almost the same as plain SPAN 2x on SD footage (consistency 41.9
against 41.5 dB, detail 32.1 against 32.6, added shimmer +0.04 against +0.07),
but it took 97 minutes against 3 minutes for a 20-minute film and 26 GB
against 2 GB of GPU memory. On 1440x1080, FlashVSR did not fit a 32 GB GPU at
all, which is why it was limited to sources up to 1280 px. Recent jobs used
strength 0.5 or less for 85% of segments, never above 0.7, and never 4x. SPAN
replaced it outright, with 4x and the system-RAM fallback.

### Output size

While it proposes segments, Gemini also picks the output size for the whole
video and says why: 1x (keep the size, no upscaling) or 2x (the default). Both
are offered for every source the wizard accepts (up to 1080p). The Segments
screen shows Gemini's choice and reason, and the Ready screen shows the reason
while that choice stands; you can still change the size there. Segment
previews render at the plan's scale, and a 1x job never runs SPAN for its
previews.

### Progress and time left

Both interfaces show a bar for the whole job ("N% of the job") and a line for
the running step: its percentage, what it is doing, and "about X left", or
"X so far" while Gemini is thinking. Segmenting is the first 8% of the job,
the previews the next 30% (split evenly per segment), and the render the
rest; a render from a result note counts 0–100% on its own. The time left
ignores time before the step's first report, such as loading SPAN, and
follows both the step's average speed and its recent speed. In the render,
an upscaled frame counts more than a frame that is only resized.

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

See [docs/ui-wizard.md](docs/ui-wizard.md),
[docs/system-design.md](docs/system-design.md), the
[ordered release plan](docs/release-plan.md), and the
[acceptance matrix and pending evidence](docs/acceptance.md).

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
Wizard use also uploads an audio-free copy of the whole video (long edge at most
960 px) to Google's File API, where it is kept about 48 hours, and sends
selected stills, notes, and chat context; do not use sensitive footage without
permission. See [CONTRIBUTING.md](CONTRIBUTING.md) for testing
and packaging checks, and [CHANGELOG.md](CHANGELOG.md) for changes.
