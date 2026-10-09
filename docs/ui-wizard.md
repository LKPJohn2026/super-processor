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

Install for the quality path: an NVIDIA driver, an RTX GPU, the `nvvfx` package,
and SeedVR2-3B weights (`SEEDVR2_3B_WEIGHTS`). ComfyUI is not required. Missing
`nvvfx` or weights is a wizard error. CI skips the real libraries and fakes both
backends.

Walk introduction → LLM type (Gemini) → setup → pick file → wait for analyze →
overview → enhance (capped SeedVR2 restore, then RTX VSR) → result (happy, or
one “too much detail” note and a second render). Confirm `output.mp4` in the
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
| Overview | `GET/POST /overview` | Duration, resolution, audio, highlight bullets, then Enhance |
| Rendering | `GET /render` | Whole-clip SeedVR2 restore (if strength &gt; 0) and RTX VSR (~5–10 min) |
| Result | `GET/POST /result` | Play output; shows strength, scale, and VSR quality; happy or a note |
| Done | `GET /` (state `done`) | Confirmation |

Estimated waits shown in the UI:

- Analyze / upload-equivalent: about 1–3 minutes
- Final render: about 5–10 minutes (scale copy with duration when known)

## State machine

```text
intro → llm_choice ┬─ gemini → setup → pick_file → analyzing → overview
                   └─ local  → local_llm_stub → setup ─┘
                                                              │
                                                              ▼
                                         rendering (whole clip, once)
                                                              │
                                                              ▼
                                         result → happy → done
                                            │
                                            └── note → Gemini UpscaleParams
                                                       → direction check
                                                       → rendering
```

The step rail is Setup, File, Enhance, Result. Rendering and the result screen
show restore strength, scale, and VSR quality.

Persisted in the job directory as `wizard_state.json` and `gemini_chat.json`.

## Something-else loop

A result note asks Gemini for new `UpscaleParams` (`restore_strength`, `scale`,
`vsr_quality` only). The direction check runs before the params are saved:

- strength is 0.0–0.35 (default 0.15); 0 skips SeedVR2; a note cannot raise the cap
- scale is 2, 3, or 4 (default 2), applied only by RTX VSR
- quality is `LOW`, `MEDIUM`, or `HIGH` (default `MEDIUM`); `ULTRA` is rejected
- notes that say artificial, plastic, over-sharpened, or not natural can only
  hold or lower strength and VSR quality
- a note that asks for more sharpness may raise those knobs only inside the caps

A revise reruns the whole clip. A time range in the note is prompt context only.
Gemini cannot emit a prompt, a model name, or FFmpeg argv.

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

## Wizard quality path

The wizard calls the two-pass stack in `upscale.py`: FFmpeg decodes frame
chunks, SeedVR2-3B restores at source size when strength is above 0, RTX VSR
scales, and FFmpeg encodes with `hevc_nvenc`. Legacy CLI diagnose/plan/apply
still use FFmpeg templates. The model never emits a shell string.

## UI presentation

Server-rendered HTML in `wizard_pages.py` (no SPA). Shared shell includes a
step rail (Setup · File · Enhance · Result), brand mark, and wait estimates.
Intro is a single composition with Super Processor as the hero. The enhance
and result screens show restore strength, scale, and VSR quality.

## CI coverage

`tests/test_wizard_full_flow.py` posts through the live `WizardServer` from
introduction → LLM choice → setup → pick → analyze → overview → render →
result → done. `tests/test_upscale.py` fakes SeedVR2 and `nvvfx.VideoSuperRes`
and skips real weights. Assertions check the strength cap and note direction.
Matrix CI runs them with other unit tests (`-m "not gemini_live"`).

## Non-goals

- Real local LLM inference (stub only)
- Vertical social export as default
- Classical CV diagnosis inside the wizard
- Desktop shell / separate SPA framework
