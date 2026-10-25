# Super Processor — System Design

Super Processor is a local-first video processor. A localhost wizard has
Gemini cut footage you already shot into segments from a grid of local
stills, plans and previews each segment in turn for you to approve, runs a
local FlashVSR restoration upscale, then lets you revise a time range or
start over with a plain-language note. Gemini only proposes segments, picks
bounded per-segment settings, checks previews, and turns a note into a
bounded range, scale, and strength; FFmpeg samples, measures, trims, splices,
encodes, and copies audio. No model writes
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
Before the first render, the segment loop (below) cuts the clip into
segments and gets each one's settings approved on a short preview. Gemini
turns a result note into `start_s`, `end_s`, `scale` (2 or 4), and
`strength` (0–1). Re-runs read the original source for that range so detail
does not stack.

Each span in the plan is a segment with its own look (`look.py`): bounded
FFmpeg settings that always run in the same order.

| Stage | Settings | Why |
|---|---|---|
| Clean | `deblock`, `denoise` (0–1) | FlashVSR treats blocks and noise as detail; removing them first leaves it less to invent from |
| Restore | `scale`, `strength`, protected regions | FlashVSR's fine detail on the cleaned plain upscale's shapes and colour; held down inside face, hand, and text boxes |
| Finish | `contrast`, `brightness`, `saturation`, `gamma`, `grain` | Grade and texture on the restored picture |

Filter text is built from the numbers in code; model output never reaches
FFmpeg as text. None of these filters changes the frame count, and a
multi-segment render checks the joined result against the source's frame
count.

A span can skip the restore stage (`use_upscale: false`). It is resized with
lanczos to the same scale and goes through the same delivery encode at
strength 0, so it keeps its look and lines up with its neighbours frame for
frame. Gemini sets it where FlashVSR would only harm.

Text-to-video, inpainting, and face reenactment stay out. A lower strength
asks the restorer for less invented texture. The job stores
`upscale_plan.json` with ordered spans, one per segment; FlashVSR runs long
spans in about 8 s pieces with a short overlap, so clips up to 30 minutes
fit in GPU memory.

When another program holds GPU memory, a piece that runs out is retried at
half the length, down to 17 frames. If even that does not fit, the clip is
restored once more in a child process whose PyTorch allocator uses CUDA
managed memory (`managed_alloc.cu`, built with `nvcc` on first use into the
models folder): blocks stay on the GPU while they fit and the driver pages
the rest to system RAM. It is slower but gives the same picture: at 2x of
854x480 with 8 GiB free, 0.53-0.70 s a frame against 0.27-0.30 s with the
GPU to itself. A child is needed because PyTorch only lets the allocator
change before CUDA starts in a process. It is Linux only: on Windows,
managed memory cannot be larger than free VRAM.

### Segment loop

The wizard works one segment at a time instead of planning every shot at
once. The steps are `sampling → proposing → segments → (analyzing ⇄
preview_segment) per segment → ready_to_render → rendering → result`.

- **Dense local sampling.** FFmpeg saves a still about every second (at most
  120, evenly spaced, first and last frame included) inside the job folder.
  Gemini proposes segment boundaries from those stills and their times; no
  video file is uploaded. FFmpeg's `scdet` is not used for this flow: a cut
  is not the same as a change in what the restoration needs.
- **Few, long segments.** Each segment is planned, previewed, and approved
  on its own, so Gemini is asked for as few as the footage needs: at most 10,
  each at least 3 s. Code holds that whatever the reply: the segments are
  clipped, sorted, made to cover the clip exactly once, and the shortest is
  merged into its shorter neighbour while a rule is broken, keeping what
  both contain.
- **Feedback on cuts.** The editor approves the segments or sends a note;
  Gemini proposes again with the note and the earlier exchange.
- **One segment at a time.** Each segment is measured, its faces, hands, and
  text boxed, its settings proposed, and a random window of up to 3 s inside
  it previewed and checked. The window is seeded by job, segment, and preview
  count, and is kept when the segment is redone, so a note is judged on the
  same seconds. The editor approves it, moving to the next segment, or sends
  a note that redoes only that segment. The scale is chosen on the first
  segment and shared by all.
- **Full render** builds `upscale_plan.json` from the approved recipes and
  renders it as before, with pass-through spans where `use_upscale` is false.

### Full restart keeps the chat

On the result screen a note can retune one time range (below) or start over.
Starting over deletes the segments, plans, previews, output, and their work
folders, keeps `gemini_chat.json` with the note added, and goes back to
sampling. Every Gemini call resends the stored chat, so the new segments and
settings are proposed with what the editor disliked in view.

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

Gemini receives only sampled stills the user has consented to send through the
wizard: the segment grid (480 pixels wide, at most 120), one still per
segment, the stills it boxes faces in, and a still from each preview pair. It
never receives a video file.

### Gemini-only planning for the wizard

The wizard uses a Google AI Studio (Gemini) API key. Local Ollama/LM Studio and
other BYOK providers are deferred. The key pasted on the setup screen is saved
to `credentials.json` in the user's config folder (`$XDG_CONFIG_HOME` or
`~/.config/super-processor` on Linux, `~/Library/Application Support` on macOS,
`%APPDATA%` on Windows; `SUPER_PROCESSOR_CONFIG_DIR` overrides it), readable
only by the user. The `GEMINI_API_KEY` environment variable wins over the file;
a key an older version left in the OS keyring is still read, last.

The durable product value remains the validated processing control plane, not
dependence on one vendor. Additional providers can return later behind the same
structured-ops contract.

### Structured output instead of model-generated shell

Gemini emits parseable JSON through
[structured outputs](https://ai.google.dev/gemini-api/docs/structured-output).
Each wizard call has its own schema: segments (times, label, contents), a
recipe per segment (strength, `use_upscale`, look, and once the scale),
protection boxes, a preview check, and a result note's range (`start_s`,
`end_s`, `scale` 2 or 4, `strength` 0–1). Values are validated, clamped, and
unknown names dropped before anything runs. The model has no shell tool, and FFmpeg argument lists
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
and `plans` commands still provide a classical split and grade. Their
`segments.json` shares the file name with the wizard's segment list; each
refuses to read or overwrite the other's (the wizard's carries
`kind: "wizard"`), so a job used by both reports which flow owns it.

The shot flow that came between (FFmpeg `scdet` cuts, one Gemini labelling
pass, then every shot planned at once on a Looks screen) was replaced by the
segment loop in 3.2.0.

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
still grid (~1/s, ≤ 120) ──► Gemini: segments ──► editor: approve or note
   │
   ▼  for each segment
measure + box faces/text ──► Gemini: recipe ──► 3 s preview (FlashVSR or resize)
   │                                              │
   │                         Gemini check ◄───────┘
   ▼
editor: approve (next segment) or note (redo this one)
   │
   ▼
FlashVSR restore + upscale per span (8 s chunks, frame count conformed),
pass-through spans resized
   │
   ▼
lanczos blend at strength ──► delivery encode (IDR every 1 s) ──► join + audio
   │
   ▼
result player ──► happy, a range note, or start over
                      │                       │
                      ▼                       ▼
          Gemini: note → range,       clear plans, keep chat,
          scale, strength             back to the still grid
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
- `wizard_state.json` (step, job id, last error, segment index, worker
  progress);
- `gemini_chat.json`, every Gemini exchange of the job, kept across a full
  restart;
- `segment_stills/` and `segments.json` (the proposal, its approval, and the
  editor's note);
- `shots.json` and `shot_plan.json` (each segment's measurements, boxes,
  recipe, preview window, and check), with `previews/`;
- `upscale_plan.json` spans with scale, strength, look, and `use_upscale`;
- `output.mp4` and its `output.mp4.delivery.json` encode marker;
- work folders for chunks and range splices.

Wizard steps:

```text
intro → llm_choice → setup → pick_file → sampling → proposing → segments
  → analyzing ⇄ preview_segment (per segment) → ready_to_render → rendering
  → result ─┬─ happy → done
            ├─ range note → rendering
            └─ restart note → sampling
```

A state file saved on a step of the removed shot flow or split/enhance flow
sends the user back to the pick step. Cancellation terminates the FFmpeg process group; final
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
