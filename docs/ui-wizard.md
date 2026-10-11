# Localhost Wizard

This document is the screen inventory and contract for the localhost website
served by `super-processor review`. Implementation lives in `wizard.py`,
`wizard_pages.py`, `gemini.py`, and the review HTTP server.

## Entry

```bash
super-processor review
super-processor --jobs-dir .dogfood/jobs review
super-processor review JOB_ID
```

Without a job id the server starts at the introduction screen. With a job id it
resumes that job’s wizard state when present, otherwise shows the legacy split
review page when `segments.json` exists.

## Live dogfood (real Gemini)

```bash
python -m pip install -e ".[dev]"
export GEMINI_API_KEY=…   # or paste on the setup screen
mkdir -p .dogfood/jobs
# Optional: use a short local clip under .dogfood/clips/
super-processor doctor
super-processor --jobs-dir .dogfood/jobs review
```

Walk introduction → LLM type (Gemini) → setup → pick file → upscale → result.
Send one note with a time range (for example "less artificial detail from 2s
to 5s"), wait for the range revise, then choose happy. Confirm `output.mp4` in
the job directory. CI also runs `@pytest.mark.gemini_live` against
`Gemini_API_Test` on main.

## Local-only server

The wizard answers only requests whose `Host` is `127.0.0.1`, `localhost`, or
`[::1]`. A request carrying an `Origin` must come from the wizard itself, the
Vite dev or preview server (`:5173`, `:4173`), or the published React shell.
Add origins, comma separated, with `SUPER_PROCESSOR_ALLOWED_ORIGINS`. A
cross-site GET that would start work (for example an image tag pointing at
`/render?run=1`) is refused.

Upscale runs on a single worker thread. `GET /render?run=1` and
`GET /api/render?run=1` start a pass when none is running and return at once.
The page refreshes, and the React shell polls `/api/state` (which reports
`busy`) until the step changes. Posts are refused while a pass runs.

## Range revise and re-encoding

The upscaled picture is encoded once into delivery settings: libx264 CRF 16,
a forced IDR frame every second, no B-frames. A sidecar
`output.mp4.delivery.json` records those settings. A result note re-upscales
its range widened out to the nearest keyframes, and the head and tail are
stream-copied, so earlier parts of the output are never encoded again. A note
that changes scale re-renders the whole clip. Clips handed to FlashVSR are
encoded losslessly (`-qp 0`).

## Strength

`strength` (0.0–1.0) is how much of FlashVSR's fine detail reaches the
output. Shapes, colour, and layout always come from a lanczos upscale of the
(cleaned) source: the blend splits both pictures at a Gaussian blur of 1.5
source pixels and keeps the source's coarse layer, so the result is
`base + strength × (detail(FlashVSR) − detail(base))`. At 0.0 the output is a
plain upscale with nothing invented; at 1.0 it is the source's structure with
all of FlashVSR's fine detail. FlashVSR can sharpen what is there but cannot
move, reshape, or recolour it. FlashVSR always runs at its upstream settings
(sparse ratio 2.0, local range 11). A note such as "less artificial detail
from 2s to 5s" lowers strength for that range only.

### Protected regions

Before the first proposal, every shot whose contents include faces, hands,
or text gets three stills (10%, 50%, 90% through it), and Gemini boxes them
so one box covers the thing's whole path. Boxes are padded by 3% of the
frame. Inside a box, the FlashVSR share is held to 0.3 for faces and hands
and 0.15 for text (or the shot's strength, if lower), and it rises back to
the shot's strength over a 3% feather outside the box, so there is no seam.
The Looks screen draws the boxes over both previews. A shot that contains
faces, hands, or text but got no boxes (Gemini failed or found none) is held
to 0.6 as a whole instead. Boxes are per shot, not tracked frame by frame.

## Input limits

The pick step probes the file and refuses it before a job is created when
the video is larger than 1080p (long edge over 1920 or short edge over 1080,
so portrait 1080x1920 is allowed) or longer than 30 minutes. A file with no
video stream, or no readable resolution or duration, is refused too.

## Serving output

`/output.mp4` and files under `previews/` are streamed in 1 MiB chunks with
single-range support (`206 Partial Content`, `416` for an unsatisfiable
range, `HEAD`), so players can seek without the server loading the file into
memory. Responses carry `Cache-Control: no-store` because a revise rewrites
`output.mp4` in place. The final mux uses `-movflags +faststart`.

## Frame alignment

Range revises and chunk joins cut by time, so each FlashVSR clip must hold
exactly the frames it was given, at the source rate. After every FlashVSR
call the clip is checked with ffprobe. A difference of up to 8 frames (or 2%)
at the end of the clip, or a rounded frame rate, is fixed losslessly (trim,
or repeat the last frame, and restamp at the source rate). A larger
difference stops the job. A splice that would change the output's total
frame count is refused and the previous `output.mp4` is kept.

## Screen inventory

| Step | Route | Purpose |
|---|---|---|
| Introduction | `GET /` (state `intro`) | Brand hero |
| LLM type | `GET/POST /llm` (state `llm_choice`) | A: no local LLM → Gemini; B: have local LLM → stub |
| Local LLM stub | `GET/POST /local-llm` (state `local_llm_stub`) | Deferred placeholder; route back to Gemini setup |
| Gemini setup | `GET/POST /setup` | Numbered Google AI Studio steps + paste key |
| Pick file | `GET/POST /pick` | Local video path (up to 1080p and 30 minutes) |
| Finding shots | `GET /shots?run=1` (`/api/scan?run=1`) | FFmpeg finds the cuts and measures each shot; Gemini labels each shot from one still |
| Shots | `POST /shots` (`/api/shots`) | Review the shot list: merge with next, split at a time, approve |
| Planning | `GET /looks?run=1` (`/api/plan?run=1`) | Gemini proposes each shot's settings; short previews render; Gemini checks them |
| Looks | `POST /looks` (`/api/looks`) | Before/after preview per shot: approve all, or a note that redoes one shot |
| Upscaling | `GET /render` | FlashVSR restore + upscale; FFmpeg trims, splices, encodes |
| Result | `GET/POST /result` | Play output; happy, or a note that retunes a time range |
| Done | `GET /` (state `done`) | Confirmation |
| New video | `POST /new` (`/api/new`) | From Done or Result: back to pick (or setup without a key); the old job stays on disk |

## State machine

```text
intro → llm_choice ┬─ gemini → setup → pick_file → finding_shots → shots → planning ⇄ looks → rendering → result ─┬─ happy → done
                   └─ local  → local_llm_stub → setup ─┘                                             ▲            │
                                                                                                     └── note ────┘
```

`planning ⇄ looks`: a note on one shot in Looks sends only that shot back
through planning.

Persisted in the job directory as `wizard_state.json` (step, job id, last
error) and `gemini_chat.json`. If finding the shots fails (FFmpeg cannot read
the file), the job is marked failed and the wizard returns to `pick_file`. If
only Gemini labelling fails, the shots keep the hints from the measurements and
the shots screen says why. If the previews fail (usually no GPU), the wizard
goes back to `shots` with the error. If Gemini cannot plan or check, the
recipes come from the measurements, the check is skipped, and the looks screen
says so. If the first upscale fails, the job is marked
failed and the wizard returns to `pick_file` with the error; a failed revise
stays on `result` with the previous output. "Start a new video" on Done or
Result goes back to `pick_file`. A state saved on a step of the removed split /
enhance flow (`analyzing`, `overview`, `choose_split`, `enhance`) loads as
`pick_file` with a message asking for the file again.

## LLM choice (sketch paths)

- **A — I don’t have a local LLM** → Gemini setup (Google AI Studio key).
- **B — I have a local LLM** → deferred stub screen (steps placeholder). The
  only action is “Use Gemini instead”, which continues to Gemini setup.
  No local inference is wired in this release.

## Shots (loop 1)

FFmpeg finds the cuts with `scdet` on a 320-pixel-wide copy. Shots shorter
than 1 s join a neighbour, and at most 40 shots are kept (the shortest merge
first). Each shot is then measured in its own short pass at 2 frames a second,
so filters that average over everything they have seen (`blockdetect`) do not
mix shots. Measurements are medians:

| Metric | From | Hint when |
|---|---|---|
| `blockiness` | `blockdetect` | ≥ 18 → `blocky` |
| `blur` | `blurdetect` | ≥ 6 → `soft` |
| `noise` | RMS difference from a spatially denoised copy (`hqdn3d` + `psnr`) | ≥ 2 → `noisy` |
| `brightness` | `signalstats` YAVG / 255 | < 0.25 → `dark`, > 0.75 → `overexposed` |
| `contrast` | `signalstats` (YHIGH − YLOW) / 255 | < 0.35 → `flat` |

The thresholds are rough and only seed the list. Gemini sees one still per shot
next to the numbers and returns a short label, the problems from a fixed list
(`blocky`, `noisy`, `soft`, `dark`, `overexposed`, `flat`, `color_cast`,
`shaky`), and what the shot contains from another (`faces`, `hands`, `text`,
`fine_pattern`). Names outside the lists are dropped. Merging two shots keeps
the first label and joins both lists; splitting measures both halves again.
Approving writes one upscale span per shot. Neighbouring spans with the same
settings render as one FlashVSR pass.

## Looks (loop 2)

Each approved shot gets a recipe: a FlashVSR `strength` and a look (the
bounded clean-up and finishing settings in `look.py`). The whole video shares
one scale; Gemini picks it on the first pass, and 4× is only offered when the
source's long edge is 960 pixels or less. Faces, hands, and text are boxed
and held down inside the boxes (see Protected regions); a shot that contains
them without boxes is capped at 0.6 as a whole.

Then every shot gets a preview: up to 3 s from its middle, rendered with its
recipe (FlashVSR loads once for all of them), plus the same seconds of the
source. Gemini compares a still from each before/after pair for
`identity_change`, `bad_anatomy`, `garbled_text`, `waxy_skin`,
`oversharpened`, `halos`, `fake_texture`, `color_shift`, `too_soft`, and
`flicker`. A shot that fails comes back with corrected settings and its
preview is redone once (no second check). The Looks screen shows both
previews, the settings that differ from neutral, Gemini's reason, the check
result, and the measured shimmer.

### Shimmer

A still cannot show texture that crawls, so each preview is also measured
(`stability.py`). For the restored clip and for the source scaled to the same
size, take the fine-detail layer (a frame minus a Gaussian blur of 1.5 source
pixels) and measure how much detail there is and how much it changes from one
frame to the next. The shimmer index is the restored clip's change-to-detail
ratio over the source's, so camera and subject motion, which move both,
cancel out. About 1 means the texture is as stable as the footage; a static
added texture scores lower. Above 1.6 the preview fails as `flicker`
whatever Gemini said, its strength drops by 0.15, and the preview is redone.
Gemini also sees the number. A clip with almost no fine detail is not judged.

### Seams

FlashVSR starts again at every chunk (every 7.5 s inside a long part), at
every boundary between shots with different settings, and at the edges of a
revised range. After each render, a 1 s window around each of those times is
read from the output and the source at 320 pixels wide. The largest
frame-to-frame jump in the output's window, relative to its median, is
compared with the source's at the same frame; a jump at least 2.5× what the
source does there (and at least 2 grey levels) is a seam. The Result screen
lists them with a ready-made note ("smooth 0:15 to 0:17"); that revise
re-renders the range in one pass, and the next render checks its new edges.
The check never fails a render; if it cannot run, the Result screen says so.

A note on one shot ("skin looks waxy") marks only that shot stale: Gemini
re-plans it with the note and its current settings, it gets a new preview and
check, and the others are left alone. Preview files carry a revision number
(`shot_002_r3_after.mp4`), so a browser never shows an older one. Approving
writes one upscale span per shot with its scale, strength, and look.

Without Gemini, recipes come from the measurements: deblock for `blocky`,
denoise for `noisy`, a lift for `dark`, contrast for `flat`, and slightly
lower strength where there was damage to clean.

## Gemini schema

The shot labels:

```json
{"shots": [{"index": 0, "label": "night street, neon sign", "issues": ["noisy", "dark"], "contains": ["text"]}]}
```

A recipe per shot (and, on the first pass, the scale):

```json
{"scale": 2, "shots": [{"index": 0, "strength": 0.4, "look": {"deblock": 0.3, "denoise": 0.2, "contrast": 1.05, "brightness": 0, "saturation": 1, "gamma": 1, "grain": 0}, "reason": "Blocky street at night; clean before upscaling."}]}
```

A preview check, with corrected settings for a failed shot:

```json
{"shots": [{"index": 0, "ok": false, "problems": ["waxy_skin"], "note": "Cheeks look plastic.", "strength": 0.3, "look": {"...": 0}}]}
```

Out-of-range numbers are clamped and unknown names dropped.

A result note becomes a range and knobs.

```json
{"start_s": 2.0, "end_s": 5.0, "scale": 2, "strength": 0.3}
```

`scale` must be 2 or 4 and `strength` 0–1; the range is clamped to the file.
A scale change re-renders the whole clip (see Range revise above).

## Multi-turn chat

`gemini_chat.json` stores ordered turns:

```json
{
  "version": 1,
  "turns": [
    {"role": "user", "text": "…"},
    {"role": "model", "text": "…", "structured": { }}
  ]
}
```

## Job files

| File | Meaning |
|---|---|
| `manifest.json` | Job id, source, state |
| `wizard_state.json` | Wizard step, job id, last error |
| `gemini_chat.json` | Result-note history |
| `shots.json` | Shot ranges, measurements, labels, and still names |
| `shot_stills/` | One JPEG per shot, served at `/shot_stills/<name>.jpg` |
| `shot_plan.json` | Per-shot recipe, reason, check result, preview revision, pending editor note |
| `previews/` | `shot_NNN_rN_before/after.mp4` and check stills, served at `/previews/<name>` |
| `upscale_plan.json` | Spans (one per approved shot) with scale, strength, and look, plus the pending revise |
| `output.mp4` | Current result |
| `render_report.json` | Seam check of the latest render: times checked and seams found |
| `output.mp4.delivery.json` | Encode settings marker for keyframe splices |
| `range_work/`, `flash_chunks/` | Work clips for revises and chunked FlashVSR |

## UI presentation

Server-rendered HTML in `wizard_pages.py` remains the live localhost wizard.
Shared shell includes a step rail (Setup · File · Upscale · Result),
brand mark, and wait estimates. Intro is a single composition with Super
Processor as the hero. The LLM choice screen uses lettered options matching the
product sketch.

`web/` is a separate React shell. GitHub Pages serves it as a static preview
of the same steps. When a local wizard is reachable (`/api/state`, a Vite
proxy, or `?api=http://127.0.0.1:<port>`), that shell drives introduction,
Gemini setup, file pick, GPU upscale, and the result. Pages itself cannot run
the GPU.

## CI coverage

`tests/test_wizard_full_flow.py` posts through the live `WizardServer` from
introduction → LLM choice → setup → pick → local upscale → result note →
range splice → done. The GPU engine is faked. Gemini, when the note is sent,
returns only a time range plus scale and strength.

## Non-goals

- Real local LLM inference (stub only)
- Vertical social export as default
- Classical CV diagnosis inside the wizard
- Running the GPU upscale from GitHub Pages
