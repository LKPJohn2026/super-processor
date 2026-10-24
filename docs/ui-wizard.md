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
`[::1]`. A request carrying an `Origin` must come from the wizard itself or the
Vite dev or preview server (`:5173`, `:4173`). The published React shell on
GitHub Pages is not allowed by default; to drive a local wizard from it, start
the wizard with
`SUPER_PROCESSOR_ALLOWED_ORIGINS=https://lkpjohn2026.github.io`. Add other
origins the same way, comma separated; `*`, `null`, and anything that is not a
plain `http(s)` origin are ignored. A cross-site GET that would start work (for
example an image tag pointing at `/render?run=1`) is refused.

Each run also makes a random token and prints it after the URL. Every POST and
every `?run=1` request must carry it, as an `X-Wizard-Token` header or a
`token` field. The wizard puts it in its own forms and links, and `/api/state`
returns it as `token`, which only the wizard's pages and allowed origins can
read. So a page on another site cannot drive the wizard even in a browser that
sends no `Origin` header. Scripts send the printed token. Bodies over 256 KiB
get 413, and a bad `Content-Length` gets 400.

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

Before the first segment is planned, every segment whose contents include
faces, hands, or text gets three stills (10%, 50%, 90% through it), and Gemini boxes them
so one box covers the thing's whole path. Boxes are padded by 3% of the
frame. Inside a box, the FlashVSR share is held to 0.3 for faces and hands
and 0.15 for text (or the shot's strength, if lower), and it rises back to
the shot's strength over a 3% feather outside the box, so there is no seam.
The segment preview draws the boxes over both previews. A segment that
contains faces, hands, or text but got no boxes (Gemini failed or found none)
is held to 0.6 as a whole instead. Boxes are per segment, not tracked frame by
frame.

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
| Gemini setup | `GET/POST /setup` | Numbered Google AI Studio steps + paste key. The key is saved to `credentials.json` in the user's config folder (mode 0600) and used on every later run; the screen says where the key in use comes from (`GEMINI_API_KEY`, the file, or an older keyring entry) |
| Pick file | `POST /browse` (`/api/browse`), `POST /upload` (`/api/upload`), `POST /pick` | "Choose a file…" opens the desktop's own file dialog on the wizard's machine (zenity through the XDG portal, then kdialog; `osascript` on macOS, PowerShell on Windows) and picks the real path. A file dropped on the page is uploaded: the body is the file, `X-File-Name` its name, `X-Wizard-Token` the token; it is streamed to `<jobs>/uploads/` (video extensions only, at most 32 GiB) and removed again if the pick refuses it. Typing a path is under "Type a path instead". Up to 1080p and 30 minutes |
| Go back | `POST /goto` (`/api/goto`) with `to=setup`, `file`, or `segments` | Setup, File, and Segments in the step rail are links when they can be reached, and each of those screens has a back link. Nothing is deleted: going back to the segments keeps the approved segments and previews, and approving them again carries on at the same segment. `/api/state` returns `navigation` |
| Finding segments | `GET /segment?run=1` (`/api/segment?run=1`) | States `sampling` then `proposing`: FFmpeg saves the still grid; Gemini proposes segments |
| Segments | `POST /segments` (`/api/segments`) | `action=approve`, or `action=note` with `note` to propose again |
| Analyzing | `GET /analyze?run=1` (`/api/analyze?run=1`) | The current segment only: Gemini proposes its settings, a short preview renders, Gemini checks it |
| Segment preview | `POST /preview` (`/api/preview`) | `action=approve` moves to the next segment; `action=redo` with `note` (and `resample=1` for a new window) redoes this one; `action=upscale` with `on=0` or `on=1` overrides Gemini's FlashVSR choice for this segment, which later proposals keep |
| Ready to render | `POST /ready` (`/api/ready`) | Choose the upscale: `scale=2`, `4` (when the source's long edge is 960 or less), or `1` ("Don't upscale": same size, clean-up and finishing only, no FlashVSR). `action=render` starts the full render from every approved segment |
| Upscaling | `GET /render` | FlashVSR restore + upscale; FFmpeg trims, splices, encodes |
| Result | `GET/POST /result` | Play output; happy, a note that retunes a time range, or a note with `restart=1` that starts over from the segments |
| Done | `GET /` (state `done`) | Confirmation |
| New video | `POST /new` (`/api/new`) | From Done or Result: back to pick (or setup without a key); the old job stays on disk |

## State machine

```text
intro → llm_choice ┬─ gemini → setup → pick_file → sampling → proposing → segments ─┐
                   └─ local  → local_llm_stub → setup ─┘          ▲             │      │
                                                                  └── note ─────┘      │ approve
                                                                                       ▼
          ┌──────────────── note (redo this segment) ──────────────┐            analyzing
          ▼                                                         │                │
      analyzing ──→ preview_segment ── approve, more segments ──→ analyzing          │
                         │                                                           │
                         └── approve, last segment → ready_to_render → rendering → result ─┬─ happy → done
                                                                          ▲                 │
                                                                          └──── note ───────┘
```

`segment_index` in `wizard_state.json` is the segment being analyzed or
previewed. A result note sent with `restart=1` ("Start over from the
segments") goes back to `sampling`: the segments, plans, previews, output, and
their work folders are deleted, `gemini_chat.json` is kept, and the note is
added to it, so every later Gemini call sees what the editor did not like. Sampling and proposing run in one worker; a note on the segments
goes back to `proposing` with the stills already saved.

Persisted in the job directory as `wizard_state.json` (step, job id, last
error, segment index, and the running worker's progress) and
`gemini_chat.json`. If sampling or proposing fails (FFmpeg cannot read the
file), the job is marked failed and the wizard returns to `pick_file`. If
Gemini cannot propose segments, the whole video is one segment and the
segments screen says why. If a segment's preview fails (usually no GPU), the
wizard goes back to `segments` with the error; approving again resumes at the
same segment. If Gemini cannot plan or check a segment, its recipe comes from
the measurements, the check is skipped, and the preview says so. If the first
upscale fails, the job is marked failed and the wizard returns to
`pick_file` with the error; a failed revise stays on `result` with the
previous output. "Start a new video" on Done or Result goes back to
`pick_file`.

A state saved on a step of an older flow loads as `pick_file` with a message
asking for the file again: the shot flow (`finding_shots`, `shots`,
`planning`, `looks`) and the split / enhance flow (`overview`,
`choose_split`, `enhance`, and `analyzing` in a state file older than
schema 3).

### Progress

While a worker runs, `/api/state` returns `progress`:
`{"phase": "analyzing", "pct": 50, "message": "Rendering previews on the GPU"}`.
While sampling and analyzing, the percentage marks fixed milestones (stills
saved, Gemini asked, previews rendered, previews checked). During the full
render it follows the work: each part counts for its share of the source
frames, FlashVSR reports after every chunk of about 8 s, and the bar runs from
5% to 90% with "Upscaling on the GPU: N% of the frames"; the seam check is the
last 10%. Both shells show them as a progress bar; `progress` is `null` when
nothing runs.

## LLM choice (sketch paths)

- **A — I don’t have a local LLM** → Gemini setup (Google AI Studio key).
- **B — I have a local LLM** → deferred stub screen (steps placeholder). The
  only action is “Use Gemini instead”, which continues to Gemini setup.
  No local inference is wired in this release.

## Segments (loop 1)

No video leaves the machine. FFmpeg saves a 480-pixel-wide still about every
second into `segment_stills/t_<ms>.jpg`, always the first frame and one
0.1 s before the end, and never more than 120: a longer clip keeps the even
spacing and gets fewer stills a second. Gemini sees every still with its time
and proposes segments where the photographic context changes (place, light,
camera, subject distance, image quality), each with a label and what it
contains from `faces`, `hands`, `text`, `fine_pattern`.

The reply is repaired to cover the clip exactly once: spans are clipped to
the clip and sorted; a gap is split down the middle; on an overlap the later
segment starts where it said; a span inside another adds no cut; and a cut
less than 0.5 s from the previous one (or from the end) is dropped, its
contents joining the segment around it. A note ("cut where we walk outside")
sends the segments back to Gemini with the note and the earlier exchange.

Approving measures each segment in its own short pass at 2 frames a second,
as the shot flow did, so filters that average over everything they have seen
(`blockdetect`) do not mix segments. Measurements are medians:

| Metric | From | Hint when |
|---|---|---|
| `blockiness` | `blockdetect` | ≥ 18 → `blocky` |
| `blur` | `blurdetect` | ≥ 6 → `soft` |
| `noise` | RMS difference from a spatially denoised copy (`hqdn3d` + `psnr`) | ≥ 2 → `noisy` |
| `brightness` | `signalstats` YAVG / 255 | < 0.25 → `dark`, > 0.75 → `overexposed` |
| `contrast` | `signalstats` (YHIGH − YLOW) / 255 | < 0.35 → `flat` |

The hints go to Gemini with each segment's still when it is planned.

## Segment previews (loop 2)

Segments are planned one at a time, in order. Each gets a recipe: a FlashVSR
`strength`, a look (the bounded clean-up and finishing settings in
`look.py`), and `use_upscale`. Gemini sets `use_upscale: false` where
FlashVSR would only harm (footage already clean and sharp, or fine texture it
would invent over); that segment is then resized with lanczos and keeps its
look. The whole video shares one scale; Gemini picks it on the first segment,
and 4× is only offered when the source's long edge is 960 pixels or less.
Faces, hands, and text are boxed and held down inside the boxes (see Protected
regions); a segment that contains them without boxes is capped at 0.6 as a
whole.

The preview is up to 3 s of the segment, rendered with its recipe, plus the
same seconds of the source. The window starts at a random point inside the
segment, seeded by job, segment, and preview count, and is kept when the
segment is redone, so a note is judged on the same seconds; tick "Preview a
different part" (`resample=1`) to pick a new one. Gemini compares a still from
the before/after pair for `identity_change`, `bad_anatomy`, `garbled_text`,
`waxy_skin`, `oversharpened`, `halos`, `fake_texture`, `color_shift`,
`too_soft`, and `flicker`. A segment that fails comes back with corrected
settings and its preview is redone once (no second check). The preview screen
shows both clips, the settings that differ from neutral, Gemini's reason, the
check result, and the measured shimmer.

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
every boundary between segments with different settings, and at the edges of a
revised range. After each render, a 1 s window around each of those times is
read from the output and the source at 320 pixels wide. The largest
frame-to-frame jump in the output's window, relative to its median, is
compared with the source's at the same frame; a jump at least 2.5× what the
source does there (and at least 2 grey levels) is a seam. The Result screen
lists them with a ready-made note ("smooth 0:15 to 0:17"); that revise
re-renders the range in one pass, and the next render checks its new edges.
The check never fails a render; if it cannot run, the Result screen says so.

A note on a segment ("skin looks waxy") sends that segment back: Gemini
re-plans it with the note and its current settings, and it gets a new preview
and check. Preview files carry a revision number (`shot_002_r3_after.mp4`), so
a browser never shows an older one. When the last segment is approved, the
ready screen lists every segment's settings; rendering writes one upscale span
per segment with its scale, strength, look, and `use_upscale`.

Without Gemini, recipes come from the measurements: deblock for `blocky`,
denoise for `noisy`, a lift for `dark`, contrast for `flat`, and slightly
lower strength where there was damage to clean.

## Gemini schema

The segments:

```json
{"segments": [{"start_s": 0, "end_s": 14.5, "label": "night street, neon sign", "contains": ["text"]}]}
```

A recipe for the current segment (and, on the first segment, the scale):

```json
{"scale": 2, "shots": [{"index": 0, "strength": 0.4, "use_upscale": true, "look": {"deblock": 0.3, "denoise": 0.2, "contrast": 1.05, "brightness": 0, "saturation": 1, "gamma": 1, "grain": 0}, "reason": "Blocky street at night; clean before upscaling."}]}
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

Calls use `models.generateContent`, not the Interactions API. The server
does not keep the thread. Each request resends `systemInstruction`,
`generationConfig`, and every stored turn as `contents` (`user` then
`model`). A model switch repeats that same body against the next model
name in the URL.

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
| `gemini_chat.json` | Every Gemini exchange of the job, replayed on each call |
| `segment_stills/` | The still grid, `t_<ms>.jpg` |
| `segments.json` | Proposed segments (schema 2, `kind: "wizard"`), whether they are approved, and the editor's last note |
| `shots.json` | Approved segments as the planner's shots: ranges, measurements, contents, boxes, still names |
| `shot_stills/` | One JPEG per segment, served at `/shot_stills/<name>.jpg` |
| `shot_plan.json` | Per-segment recipe, `use_upscale`, preview window, reason, check result, preview revision, pending editor note |
| `previews/` | `shot_NNN_rN_before/after.mp4` and check stills, served at `/previews/<name>` |
| `upscale_plan.json` | Spans (one per approved segment) with scale, strength, look, and `use_upscale`, plus the pending revise |
| `output.mp4` | Current result |
| `render_report.json` | Seam check of the latest render: times checked and seams found |
| `output.mp4.delivery.json` | Encode settings marker for keyframe splices |
| `range_work/`, `flash_chunks/` | Work clips for revises and chunked FlashVSR |

## UI presentation

Server-rendered HTML in `wizard_pages.py` remains the live localhost wizard.
Shared shell includes a step rail (Setup · File · Segments · Previews ·
Result; upscaling is a choice on the ready screen, not a step),
brand mark, and wait estimates. Intro is a single composition with Super
Processor as the hero. The LLM choice screen uses lettered options matching the
product sketch.

`web/` is a separate React shell. GitHub Pages serves it as a static preview
of the same steps. When a local wizard is reachable (`/api/state`, a Vite
proxy, or `?api=http://127.0.0.1:<port>`), that shell drives every step,
from introduction through the segments and their previews to the result, and
shows worker progress. Pages itself cannot run the GPU.

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
