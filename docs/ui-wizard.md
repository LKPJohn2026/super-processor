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

`strength` (0.0–1.0) is how much of the FlashVSR picture reaches the output.
The rest is a lanczos upscale of the same source frames, blended in the
delivery encode. At 0.0 the output is a plain upscale with nothing invented;
at 1.0 it is FlashVSR alone. FlashVSR always runs at its upstream settings
(sparse ratio 2.0, local range 11). A note such as "less artificial detail
from 2s to 5s" lowers strength for that range only.

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
| Upscaling | `GET /render` | FlashVSR restore + upscale; FFmpeg trims, splices, encodes |
| Result | `GET/POST /result` | Play output; happy, or a note that retunes a time range |
| Done | `GET /` (state `done`) | Confirmation |
| New video | `POST /new` (`/api/new`) | From Done or Result: back to pick (or setup without a key); the old job stays on disk |

## State machine

```text
intro → llm_choice ┬─ gemini → setup → pick_file → rendering → result ─┬─ happy → done
                   └─ local  → local_llm_stub → setup ─┘          ▲            │
                                                                  └── note ────┘
```

Persisted in the job directory as `wizard_state.json` (step, job id, last
error) and `gemini_chat.json`. If the first upscale fails, the job is marked
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

## Gemini schema

The wizard makes one Gemini call: a result note becomes a range and knobs.

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
| `upscale_plan.json` | Spans with scale and strength, plus the pending revise |
| `output.mp4` | Current result |
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
