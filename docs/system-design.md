# Super Processor — System Design

Super Processor is a local-first video processor. It enhances existing footage
through measured, constrained FFmpeg pipelines. A localhost wizard guides
setup, Gemini-assisted segmentation, and per-segment enhancement. AI diagnoses
problems and proposes structured operations; it does not generate replacement
frames or write arbitrary shell commands.

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
| Typical duration | Short clips | Long-form files up to roughly two hours |
| AI role | Pixel generator | Diagnoser and constrained planner |
| Processing | Usually hosted | Local FFmpeg execution |
| Main risk | Invented content and temporal drift | Over-processing or poor encode choices |
| Recovery | Regenerate | Adjust a visible recipe and re-preview |
| Trust model | Inspect final pixels | Validate plan, preview, approve, and reproduce |

The closest alternatives are manual FFmpeg scripts, conventional nonlinear
editors, and black-box enhancement applications. Super Processor aims to
combine FFmpeg's determinism with guided diagnosis, while keeping every
operation inspectable.

## Product decisions

### Enhancement instead of generation

The product processes source pixels only. It excludes text-to-video, diffusion
inpainting, generative upscaling, face reenactment, and other forms of frame
synthesis.

This avoids the market's anatomy, physics, identity, and glyph-generation
problems by construction. The tool may transform footage, but it will not
invent scene content.

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

### Structured ops instead of model-generated shell

Gemini emits parseable JSON through
[structured outputs](https://ai.google.dev/gemini-api/docs/structured-output).
Schemas cover split layouts and per-segment enhancement options. Each option
lists allowlisted operations and bounded parameters.

Engineering-owned templates translate validated ops into FFmpeg argument lists.
The model has no shell tool. This confines mistakes to a rejectable data
structure and prevents invented filters or unsafe command strings from reaching
a process boundary.

### Gemini owns split and diagnosis

Classical computer-vision estimators are retired from the product path for
segmentation and diagnosis. The wizard samples frames with FFmpeg at a dynamic
rate, sends them to Gemini, and presents structured proposals to the user.

Legacy CLI diagnose/plan paths that used CV estimators may remain in the tree
for compatibility until a later cleanup tag. The wizard does not call them.

### Split choice among Gemini layouts

Gemini proposes several timeline layouts (typically three, four, or five
segments). The user picks one layout or replies with free text (“something
else”), which continues a multi-turn chat that returns a new structured layout
set. Hard maximum: 20 segments.

### Per-segment enhancement with short previews

For each accepted segment, Gemini proposes three to five enhancement options
(structured ops). The user picks one or revises with free text. A short preview
clip is encoded before the choice is committed. After every segment is accepted,
segments are encoded and concatenated at the source aspect ratio.

### Wizard export: enhance and concat

Wizard v1 keeps the source aspect. It does not default to full-timeline
vertical social export or a size-cap floor. Those code paths may remain unused
by the wizard; they are not part of the default render.

### Mandatory preview and explicit completion

A segment choice requires a short preview. Final render starts only after every
segment has an accepted option. The result screen asks whether the user is
happy; a revise path returns to split choice with chat history preserved.

### Focused operations

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
Pick file (localhost wizard)
   │
   ▼
FFprobe ──► versioned media facts
   │
   ▼
Dynamic-FPS frame sample ──► Gemini structured split layouts
   │
   ▼
User picks layout (or multi-turn revise)
   │
   ▼
Per segment: Gemini options ──► short preview ──► accept
   │
   ▼
schema / policy validator ──► FFmpeg templates
   │
   ▼
segment encodes + concat (same aspect)
   │
   ▼
result player + happy / revise
```

The model has no shell tool. Only the template layer creates process arguments,
and input/output paths are passed as argument-list elements rather than
interpolated shell strings.

## Job model

Each job has an on-disk directory containing:

- source identity and FFprobe facts;
- sampled frames / stills for Gemini and the UI;
- `gemini_chat.json` multi-turn history;
- split layouts and the accepted segment list;
- per-segment chosen ops and preview clips;
- validation reports;
- final output and redacted logs.

Wizard-oriented states reuse the job store where possible:

```text
imported → probed → split_proposed → split_accepted
         → plans_ready → plan_selected → encoding → complete
```

Revise-from-result may return to `split_proposed` with preserved chat history.
Cancellation terminates the FFmpeg process group; final output is written
atomically.

## Frame sampling for Gemini

Frame count is bounded by duration-dependent FPS:

- duration ≤ 60 s → up to 10 FPS;
- duration ≥ 600 s → 1 FPS;
- durations in between interpolate;
- a hard cap (about 120–180 frames) applies a further stride when needed.

Only those frames (or a contact sheet derived from them) are eligible for
upload after consent. Keyframe stills for the UI stay on disk locally.

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
- local FFmpeg/FFprobe processing;
- Gemini structured split and per-segment options;
- validated ops, short previews, enhance-and-concat render;
- software encoding (optional hardware);
- reference-based regression helpers where already present.

Deferred:

- local LLM in the wizard;
- model-emitted FFmpeg shell strings;
- vertical social export as the wizard default;
- desktop shell (Tauri/Electron);
- classical CV as product diagnosis;
- neural enhancement models;
- cloud render workers;
- generative video features;
- multi-track nonlinear editing.

Super Processor's market position follows directly from these choices: use AI
to make deterministic video tools easier and safer, while refusing to invent
the content being processed.
