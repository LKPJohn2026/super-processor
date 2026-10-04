# Localhost Gemini Wizard

This document is the screen inventory and contract for the wizard served by
`super-processor review`. Implementation lives in `wizard.py`, `gemini.py`, and
the review HTTP server.

## Entry

```bash
super-processor review
super-processor --jobs-dir .dogfood/jobs review
super-processor review JOB_ID
```

Without a job id the server starts at the introduction screen. With a job id it
resumes that job’s wizard state when present, otherwise shows the legacy split
review page when `segments.json` exists.

## Screen inventory

| Step | Route | Purpose |
|---|---|---|
| Introduction | `GET /` (state `intro`) | Product pitch: enhance, do not generate |
| Gemini setup | `GET/POST /setup` | Paste Google AI Studio API key |
| Pick file | `GET/POST /pick` | Choose a local video path |
| Analyzing | `GET /analyze` | Sample frames + Gemini split call (wait copy) |
| Overview | `GET /overview` | Duration, resolution, audio, highlight bullets |
| Split choice | `GET/POST /split` | Pick layout 3/4/5 or free-text revise |
| Segment enhance | `GET/POST /segment/<i>` | Options + short preview + accept |
| Rendering | `GET /render` | Encode parts + concat; show progress |
| Result | `GET/POST /result` | Play output; happy or revise |

Estimated waits shown in the UI:

- Analyze / upload-equivalent: about 1–3 minutes
- Final render: about 5–10 minutes (scale copy with duration when known)

## State machine

```text
intro → setup → pick_file → analyzing → overview → choose_split
                                              │
                    free-text revise ←────────┤
                                              ▼
                                    enhance_segment(i) → next i
                                              │
                                              ▼
                                         rendering
                                              │
                                              ▼
                         result → happy → done
                            │
                            └── revise → choose_split
```

Persisted in the job directory as `wizard_state.json` and `gemini_chat.json`.

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

Free-text revise on split or enhance appends a user turn and re-calls Gemini
with the same schema. Happy→revise on the result screen returns to
`choose_split` without wiping history.

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

## Non-goals

- Local LLM branch in setup (show deferred only)
- Vertical social export as default
- Classical CV diagnosis inside the wizard
- Desktop shell
