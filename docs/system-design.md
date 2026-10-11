# Super Processor — System Design

Super Processor is a local-first video processor. A localhost wizard finds
and measures the shots in footage you already shot, has Gemini label them for
you to approve, runs a local FlashVSR restoration upscale, then lets you revise
a time range with a plain-language note. Gemini only labels shots from fixed
lists, picks bounded per-shot settings, checks previews, and turns a note
into a bounded range, scale, and strength; FFmpeg
detects, measures, trims, splices, encodes, and copies audio. No model writes
shell commands. The CLI keeps the older
FFmpeg-filter grade paths for automation.

## Market context

Most products marketed as “AI video” emphasize text-to-video, image-to-video,
or generative restyling. Their strength is producing short creative clips from
little or no source material. Their failure modes are also well documented:

- anatomy and identity drift;
- objects that merge, disappear, or change shape;
- incorrect gravity, collisions, fluids, and other physical behavior;
- temporal flicker, repeated frames, and motion discontinuities;
- distorted text and symbols;
- camera paths or background context that change unexpectedly;
- coherence that degrades as clip duration grows.

Benchmarks such as
[VBench++](https://arxiv.org/abs/2411.13503),
[VideoPhy](https://arxiv.org/abs/2406.03520), and
[T2VPhysBench](https://arxiv.org/abs/2505.00337) evaluate parts of this
problem. These failures arise primarily when a model synthesizes pixels over
time; prompting alone does not guarantee physical or temporal consistency.

Super Processor targets a different market:

| Dimension | Generative video tools | Super Processor |
|---|---|---|
| Source | Synthesized frames | Existing footage |
| Typical duration | Short clips | Files up to 30 minutes at 1080p |
| AI role | Pixel generator | Restoration upscaler plus a bounded revise planner |
| Processing | Usually hosted | Local FlashVSR and FFmpeg |
| Main risk | Invented content and temporal drift | Invented fine texture (bounded by strength) and flicker |
| Recovery | Regenerate | Re-run a time range at a lower strength |
| Trust model | Inspect final pixels | Same frames and timing as the source; each revise is stored as JSON |

The closest alternatives are manual FFmpeg scripts, conventional nonlinear
editors, and black-box enhancement applications. Super Processor aims to
combine FFmpeg's determinism with guided diagnosis, while keeping every
operation inspectable.

## Product decisions

### Restoration upscale instead of an FFmpeg grade

The wizard picture path is a local GPU restoration upscale (FlashVSR). FFmpeg
probes, trims a cited time range, splices that range back, and copies audio.
Before the first render, FFmpeg splits the clip into shots and measures each
one, Gemini labels them, and the editor approves the list. Gemini then
proposes each shot's settings, short previews render, Gemini checks them for
restoration mistakes, and the editor approves the settings or sends one shot
back with a note (see `docs/ui-wizard.md`, Shots and Looks). Gemini only turns a result note into `start_s`,
`end_s`, `scale` (2 or 4), and `strength` (0–1). Re-runs read the original source for that range so detail
does not stack.

Each span in the plan is a shot with its own look (`look.py`): bounded FFmpeg
settings that always run in the same order.

| Stage | Settings | Why |
|---|---|---|
| Clean | `deblock`, `denoise` (0–1) | FlashVSR treats blocks and noise as detail; removing them first leaves it less to invent from |
| Restore | `scale`, `strength`, protected regions | FlashVSR's fine detail on the cleaned plain upscale's shapes and colour; held down inside face, hand, and text boxes |
| Finish | `contrast`, `brightness`, `saturation`, `gamma`, `grain` | Grade and texture on the restored picture |

Filter text is built from the numbers in code; model output never reaches
FFmpeg as text. None of these filters changes the frame count, and a
multi-shot render checks the joined result against the source's frame count.

Text-to-video, inpainting, and face reenactment stay out. A lower strength
asks the restorer for less invented texture. The job stores
`upscale_plan.json` with ordered spans so a later chunked runner can cover
clips up to about 30 minutes (about 8s pieces with a short overlap). The
prototype runs one span, then one range revise.

### Localhost wizard before a desktop shell

The primary interface is a multi-step wizard served on localhost by the CLI
(`super-processor review`). There is no Electron/Tauri shell in this train.

The CLI remains available for automation and legacy job commands. The wizard
binds to the same on-disk job store, so every choice stays inspectable as JSON
and stills.

### Media remains local

FFprobe, frame sampling, previews, and final FFmpeg encodes run on the user's
machine. Expected inputs can be large (roughly 4–10 GB and 90–120 minutes);
cloud upload of the full source is avoided.

Gemini receives only sampled frames the user has consented to send through the
wizard. It never receives the full source file by default.

### Gemini-only planning for the wizard

The wizard uses a Google AI Studio (Gemini) API key. Local Ollama/LM Studio and
other BYOK providers are deferred. Credentials come from the environment or OS
keyring (`GEMINI_API_KEY`).

The durable product value remains the validated processing control plane, not
dependence on one vendor. Additional providers can return later behind the same
structured-ops contract.

### Structured output instead of model-generated shell

Gemini emits parseable JSON through
[structured outputs](https://ai.google.dev/gemini-api/docs/structured-output).
The wizard has one schema: a result note becomes `start_s`, `end_s`, `scale`
(2 or 4), and `strength` (0–1). Values are validated and clamped to the file
before anything runs. The model has no shell tool, and FFmpeg argument lists
are built by engineering-owned code.

### Result note and range revise

The first pass restores the whole clip. The result screen asks whether the
user is happy; otherwise a note such as "less artificial detail from 2s to
5s" re-restores that range from the original source and splices it back.
Untouched parts of the output are stream-copied and never re-encoded. A note
that changes scale re-renders the whole clip. Strength is the share of the
FlashVSR picture in the output; the rest is a lanczos upscale of the same
frames.

### Inputs up to 1080p and 30 minutes

The pick step refuses files above 1080p (either orientation) or longer than 30
minutes, before a job is created.

### Gemini split and per-segment enhance (removed)

An earlier wizard had Gemini propose timeline split layouts and per-segment
FFmpeg filter options with short previews, then concatenated the graded
segments. Once the picture path became a FlashVSR upscale, that flow was no
longer reachable from the wizard, and it has been removed. The CLI `segment`
and `plans` commands still provide a classical split and grade.

### Focused operations (CLI grade path)

The processing vocabulary stays allowlisted:

| Operation | Role |
|---|---|
| Contrast | Exposure / contrast / gamma |
| White balance | Temperature / cast |
| Denoise | Parameter-capped denoise |
| Sharpen | Mild unsharp |
| Stabilize | Optional motion reduction |
| Trim | Keep at least five seconds |
| Encode | Software `libx265` by default (hardware encoders optional) |

`reframe_vertical` and `encode_hevc_size_cap` are out of wizard v1 defaults.

### Software HEVC with optional hardware

Final HEVC output defaults to `libx265`. NVENC, QSV, AMF, and VideoToolbox may
be selected when present. The software path remains the quality reference.

### Reference-based offline evaluation

Offline evaluation still uses clean→degraded fixtures and VMAF where available.
Runtime success for the wizard is plan validity, successful previews, concat
integrity, and user acceptance.

## Architecture

```text
Pick file (localhost wizard) ──► FFprobe ──► 1080p / 30 min check
   │
   ▼
FlashVSR restore + upscale (8 s chunks, frame count conformed)
   │
   ▼
lanczos blend at strength ──► delivery encode (IDR every 1 s)
   │
   ▼
result player ──► happy, or a note
                      │
                      ▼
          Gemini: note → range, scale, strength
                      │
                      ▼
          re-restore range from source ──► keyframe splice ──► result
```

The model has no shell tool. Only engineering-owned code creates process
arguments, and input/output paths are passed as argument-list elements rather
than interpolated shell strings.

## Job model

Each job has an on-disk directory containing:

- source identity and FFprobe facts;
- `wizard_state.json` (step, job id, last error);
- `gemini_chat.json` history of result notes;
- `upscale_plan.json` spans with scale and strength;
- `output.mp4` and its `output.mp4.delivery.json` encode marker;
- work folders for chunks and range splices.

Wizard steps:

```text
intro → llm_choice → setup → pick_file → rendering → result ─┬─ happy → done
                                             ▲               │
                                             └──── note ─────┘
```

A state file saved on a step of the removed split/enhance flow sends the user
back to the pick step. Cancellation terminates the FFmpeg process group; final
output is written atomically.

## CLI direction

```text
super-processor doctor
super-processor review              # localhost Gemini wizard
super-processor review JOB          # open wizard / review for a job
super-processor models
super-processor show JOB
```

Legacy `diagnose` / `plan` / `preview` / `apply` / `segment` / `plans` commands
may remain for compatibility; the wizard is the primary product path.

## Security and privacy

- Gemini API keys come from environment variables or the OS keychain and are
  redacted from logs.
- The validator accepts only declared operations and bounded parameters.
- User-selected paths are normalized and passed without invoking a shell.
- Sampled frames leave the machine only through the Gemini path after setup.
- Telemetry is opt-in and excludes media, secrets, and full filesystem paths.

## Boundary

Included:

- localhost wizard on the review server;
- local FlashVSR restoration upscale with FFmpeg trim, splice, and encode;
- Gemini structured result notes (range, scale, strength);
- CLI FFmpeg-filter grade paths (diagnose, plan, segment, plans);
- software encoding (optional hardware);
- reference-based regression helpers where already present.

Deferred:

- local LLM in the wizard;
- model-emitted FFmpeg shell strings;
- vertical social export as the wizard default;
- desktop shell (Tauri/Electron);
- classical CV as product diagnosis;
- cloud render workers;
- generative video features;
- multi-track nonlinear editing.

Super Processor's market position follows directly from these choices: use AI
to restore footage the camera already captured, never new scenes or frames,
and give the user a direct control over how much fine detail the restorer
adds.
