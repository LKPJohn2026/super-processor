# Localhost Gemini Wizard

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

Walk introduction → LLM type (Gemini) → setup → pick file → wait for analyze →
overview → split (try Something else once) → enhance (preview, optional
Something else, Accept) → render → result (happy). Confirm `output.mp4` in the
job directory. CI also runs `@pytest.mark.gemini_live` against
`Gemini_API_Test` on main.

## Screen inventory

| Step | Route | Purpose |
|---|---|---|
| Introduction | `GET /` (state `intro`) | Brand hero: enhance, do not generate |
| LLM type | `GET/POST /llm` (state `llm_choice`) | A: no local LLM → Gemini; B: have local LLM → stub |
| Local LLM stub | `GET/POST /local-llm` (state `local_llm_stub`) | Deferred placeholder; route back to Gemini setup |
| Gemini setup | `GET/POST /setup` | Numbered Google AI Studio steps + paste key |
| Pick file | `GET/POST /pick` | Choose a local video path |
| Analyzing | `GET /analyze` | Sample frames + Gemini split call (~1–3 min) |
| Overview | `GET/POST /overview` | Duration, resolution, audio, highlight bullets |
| Split choice | `GET/POST /split` | Layouts A… + Something else revise loop |
| Segment enhance | `GET/POST /segment` | Options + short preview + Something else + accept |
| Rendering | `GET /render` | Encode parts + concat (~5–10 min) |
| Result | `GET/POST /result` | Play output; happy or Something else → split |
| Done | `GET /` (state `done`) | Confirmation |

Estimated waits shown in the UI:

- Analyze / upload-equivalent: about 1–3 minutes
- Final render: about 5–10 minutes (scale copy with duration when known)

## State machine

```text
intro → llm_choice ┬─ gemini → setup → pick_file → analyzing → overview
                   └─ local  → local_llm_stub → setup ─┘
                                                              │
                         free-text revise ←───────────────────┤ choose_split
                                                              ▼
                                                    enhance_segment(i) → next i
                                                              │
                         Something else revise ←──────────────┤
                                                              ▼
                                                         rendering
                                                              │
                                                              ▼
                                         result → happy → done
                                            │
                                            └── Something else → choose_split
```

Persisted in the job directory as `wizard_state.json` and `gemini_chat.json`.

## Something-else loops

1. **Split** — Note revises layouts via `propose_splits(user_text=…)`. Stay on
   `choose_split` until a layout is picked.
2. **Enhance** — Note combines prior options + user text via
   `revise_enhance` / `combine_enhance_revise_message`. Stay on the segment
   until preview + accept (no production cap; tests cover five rounds).
3. **Result** — “Something else” returns to `choose_split` with chat history
   kept; segment choices are redone for the new pass.

## LLM choice (sketch paths)

- **A — I don’t have a local LLM** → Gemini setup (Google AI Studio key).
- **B — I have a local LLM** → deferred stub screen (steps placeholder). The
  only action is “Use Gemini instead”, which continues to Gemini setup.
  No local inference is wired in this release.

## Gemini schemas

### Split layouts

```json
{
  "layouts": [
    {
      "segment_count": 3,
      "summary": "Indoor lift, outdoor daylight, closing dark",
      "segments": [
        {
          "start_s": 0.0,
          "end_s": 120.0,
          "label": "indoor low light",
          "issues": ["low_light", "low_contrast"]
        }
      ]
    }
  ],
  "highlights": ["Dialogue-heavy opening", "Bright exterior mid"]
}
```

Rules enforced after parse:

- `1 ≤ layouts ≤ 5`
- each layout `1 ≤ segment_count ≤ 20`
- segments contiguous, non-overlapping, within media duration
- each segment at least 5 seconds

### Per-segment options

```json
{
  "issues": ["low_light", "soft_focus"],
  "options": [
    {
      "id": "A",
      "label": "Lift shadows and mild contrast",
      "ops": [
        {"op": "contrast", "params": {"contrast": 1.2, "brightness": 0.1, "gamma": 1.05}}
      ]
    }
  ]
}
```

Rules:

- `3 ≤ options ≤ 5`
- each `op` is allowlisted; params clamped by the validator bounds
- no `reframe_vertical` / `encode_hevc_size_cap` from the wizard path

## Sampling rules

| Duration | Target sample rate |
|---|---|
| ≤ 60 s | up to 10 FPS |
| 60–600 s | interpolate 10 → 1 FPS |
| ≥ 600 s | 1 FPS |

Apply a hard cap of 160 frames by increasing stride. Extract JPEG or PPM
stills under `wizard_frames/`. Consent is implied by completing Gemini setup
and starting analysis.

## Multi-turn chat

`gemini_chat.json` stores ordered turns:

```json
{
  "version": 1,
  "turns": [
    {"role": "user", "text": "…", "frame_refs": ["wizard_frames/f_000.jpg"]},
    {"role": "model", "text": "…", "structured": { }}
  ]
}
```

## Job files

| File | Meaning |
|---|---|
| `manifest.json` | Job id, source, state |
| `probe.json` / media facts | FFprobe |
| `wizard_state.json` | Wizard step, layout index, segment cursor |
| `gemini_chat.json` | Multi-turn history |
| `split_layouts.json` | Last Gemini layout set |
| `segments.json` | Accepted timeline |
| `segment_choices.json` | Chosen option id + ops per segment |
| `segment_stills/` | UI stills |
| `wizard_frames/` | Frames sent / eligible for Gemini |
| `segment_encodes/` | Per-segment mp4 parts |
| `output.mp4` | Concat result |

## Mapping structured ops to FFmpeg

Chosen ops become a `Treatment` (ordered steps) for the existing
`build_segment_argv` / `render_chosen_plan` path. Templates own the argv. The
model never emits a shell string.

## UI presentation

Server-rendered HTML in `wizard_pages.py` (no SPA). Shared shell includes a
step rail (Setup · File · Split · Enhance · Result), brand mark, and wait
estimates. Intro is a single composition with Super Processor as the hero.
Choice screens use lettered options (A–E) matching the product sketch.

## CI coverage

`tests/test_wizard_full_flow.py` posts through the live `WizardServer` from
introduction → LLM choice → setup → pick → analyze → overview → split →
enhance (preview + accept) → render → result → done. Assertions check
screen titles, step-rail phases, and key CTAs so the localhost website stays
aligned with this inventory. Matrix CI runs it with other unit tests
(`-m "not gemini_live"`); Gemini responses are faked.

## Non-goals

- Real local LLM inference (stub only)
- Vertical social export as default
- Classical CV diagnosis inside the wizard
- Desktop shell / separate SPA framework
