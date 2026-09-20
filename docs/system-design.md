# Super Processor — System Design

Super Processor is a local-first command-line video processor. It enhances
existing footage through measured, constrained FFmpeg pipelines. AI diagnoses
problems and proposes recipes; it does not generate replacement frames or
write arbitrary shell commands.

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
| Main risk | Invented content and temporal drift | Over-processing, crop loss, or poor encode choices |
| Recovery | Regenerate | Adjust a visible recipe and re-preview |
| Trust model | Inspect final pixels | Validate plan, preview, approve, and reproduce |

The closest alternatives are manual FFmpeg scripts, conventional nonlinear
editors, and black-box enhancement applications. Super Processor aims to
combine FFmpeg's determinism with guided diagnosis, while keeping every
operation inspectable.

## Product decisions

### Enhancement instead of generation

The first version processes source pixels only. It excludes text-to-video,
diffusion inpainting, generative upscaling, face reenactment, and other forms
of frame synthesis.

This avoids the market's anatomy, physics, identity, and glyph-generation
problems by construction. It also establishes a clear trust contract: the tool
may transform footage, but it will not invent scene content.

### Command line before a desktop shell

The first interface is a CLI. Preview files open in VLC or `ffplay`; the
project does not build an embedded player or Electron/Tauri shell initially.

A CLI keeps jobs, manifests, diagnoses, and recipes explicit. It also supports
automation and batch workflows while the processing model is still evolving.
Using mature external players avoids coupling media-pipeline correctness to a
new UI.

### Media remains local

FFprobe, feature extraction, previews, and final FFmpeg encodes run on the
user's machine. The expected inputs—roughly 4–10 GB and 90–120 minutes—make
cloud upload expensive in time, bandwidth, privacy, and infrastructure cost.

Model calls are separate from media execution. A remote vision model may
receive sampled frames only after explicit user consent; it never receives the
full source by default.

### Local models and bring-your-own-key providers

The planner and optional vision model use one provider-neutral contract. Users
can select a local Ollama/LM Studio-compatible endpoint or configure their own
cloud API credentials.

The product's durable value is the validated processing control plane, not
dependence on one model vendor. Local inference optimizes privacy and cost;
cloud inference can optimize model capability.

### Structured recipes instead of model-generated commands

Models emit versioned Recipe JSON. Engineering-owned templates translate a
validated recipe into FFmpeg arguments.

The validator checks:

- JSON schema and recipe version;
- operation allowlist;
- parameter ranges;
- operation ordering and incompatibilities;
- agreement with FFprobe facts;
- disk, time, and output-size feasibility.

This confines model mistakes to a rejectable data structure. It prevents
invented filters, unsafe shell syntax, and unsupported parameter combinations
from reaching a process boundary.

### Automatic diagnosis before natural-language refinement

Computer-vision estimators inspect sampled windows and propose fixes. The user
can accept or disable each recommendation. Natural language is secondary and
patches an existing recipe—for example, “less denoise” or “warmer white
balance.”

This interaction does not require users to name filters, but it also avoids
turning an unconstrained prompt directly into a media pipeline.

### Mandatory preview and explicit approval

A final encode requires a successful preview and explicit approval. Previewing
is essential because denoise strength, stabilization crop, white balance, and
vertical framing are partly subjective. It also prevents an invalid choice
from consuming hours of software encoding.

### Focused first-version operations

The first processing vocabulary is deliberately small:

| Operation | Measurement | Execution |
|---|---|---|
| Contrast | Luma percentiles, clipping, histogram shape | FFmpeg curves/equalization |
| White balance | Neutral-pixel and color-cast estimates | FFmpeg color/temperature adjustment |
| Denoise | Flat-region noise and detail estimates | Parameter-capped FFmpeg denoise |
| Stabilize | Global motion and crop-loss estimates | Vid.stab or equivalent FFmpeg transform |
| Social export | Classical saliency (edge energy) and aspect constraints | Vertical reframe, scale, and HEVC encode |

These operations address common real-footage problems without requiring a
model to redraw textures or objects.

### Classical computer vision before neural enhancement

The first version uses classical vision and signal-processing estimators to
select FFmpeg parameters. On-device neural denoise, reframe, or upscale models
can be introduced later behind new capability and validation boundaries.

Classical estimators are easier to inspect, test, and execute on long videos.
Deferring neural enhancement also avoids introducing plastic skin, fabricated
texture, and temporal inconsistency into the initial trust model.

### Software HEVC before hardware encoders

Final HEVC output uses `libx265`. NVENC, QSV, AMF, and VideoToolbox are future
accelerators.

One software path reduces the initial test matrix and provides a consistent
quality reference. The CLI must expose realistic throughput estimates because
two-hour encodes may run for several hours.

### Full-timeline vertical export with a quality floor

Social export processes the full timeline rather than selecting highlight
clips. Before encoding, the planner calculates the bitrate available under the
requested size cap.

If duration, resolution, and size imply quality below a configured floor, the
tool refuses the job or requires explicit acknowledgment. It must never meet a
size target by silently producing unusable footage.

### Reference-based offline evaluation

Offline evaluation begins with clean source clips, controlled degradations, and
restored outputs. VMAF is used alongside operation-specific measurements:

- clipping and histogram distance for contrast;
- neutral and skin color error for white balance;
- residual noise and retained edges for denoise;
- residual camera motion and crop percentage for stabilization;
- subject retention, text readability, aspect, and bytes for social export.

Production footage usually has no clean reference. Runtime checks therefore
focus on plan validity, detail loss, subject crop, introduced flicker, text
readability, output structure, and user approval.

## Architecture

```text
Input file
   │
   ▼
FFprobe ──► versioned media facts
   │
   ▼
Sampled feature extraction ──► CV estimates ──► optional VLM advice
   │
   ▼
Diagnosis ──► planner LLM ──► Recipe JSON
                              │
                              ▼
                    schema / policy validator
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
             reject with errors      FFmpeg templates
                                              │
                                              ▼
                                  preview file + runtime checks
                                              │
                                         user approval
                                              │
                                              ▼
                                  full-timeline local encode
                                              │
                                              ▼
                                  output probe + final report
```

The model has no shell tool. Only the template layer creates process arguments,
and input/output paths are passed as argument-list elements rather than
interpolated shell strings.

## Job model

Each job has an on-disk directory containing:

- source identity and FFprobe facts;
- sampled feature data and contact sheets;
- diagnosis and recipe JSON;
- validation reports;
- preview files and runtime metrics;
- cached stabilization analysis;
- final output report and redacted logs.

The state machine is:

```text
imported → probed → diagnosed → planned → validated
         → previewed → approved → encoding → complete
```

Every transition is resumable. Cancellation terminates the FFmpeg process
group, and final output is written atomically so interrupted jobs do not appear
successful.

## Long-file strategy

Diagnosis uses stratified windows rather than decoding the full timeline:
opening, middle, closing, scene-change peaks, and windows with unusual feature
scores. Full passes occur only when required, such as stabilization analysis or
the final encode.

The orchestrator checks free space before starting, caches reusable analysis,
streams media rather than buffering it in memory, and records progress through
FFmpeg's machine-readable progress output.

## CLI direction

The intended command vocabulary is:

```text
super-processor doctor
super-processor models
super-processor diagnose INPUT
super-processor plan JOB [--instruction TEXT]
super-processor preview JOB [--start SECONDS] [--open vlc|ffplay]
super-processor apply JOB --approve
super-processor show JOB
```

The initialized repository exposes `version` and `self-test` first so packaging,
compiled extensions, and release automation can be verified independently of
the media pipeline.

## Security and privacy

- API credentials come from environment variables or the OS keychain and are
  redacted from logs.
- The validator accepts only declared operations and bounded parameters.
- User-selected paths are normalized and passed without invoking a shell.
- Sampled frames leave the machine only through an explicitly enabled remote
  vision provider.
- Telemetry is opt-in and excludes media, secrets, and full filesystem paths.

## First-version boundary

Included decisions:

- local CLI operation;
- local FFmpeg/FFprobe processing;
- CV-first diagnosis;
- local-model or BYOK planning;
- validated recipes and mandatory previews;
- contrast, white balance, denoise, stabilization, and full-timeline vertical
  HEVC export;
- software encoding and reference-based regression evaluation.

Deferred decisions:

- desktop UI;
- hardware encoder support;
- neural enhancement models (including learned face/saliency trackers);
- cloud render workers;
- generative video features;
- multi-track nonlinear editing;
- automatic highlight extraction.

Super Processor's market position follows directly from these choices: use AI
to make deterministic video tools easier and safer, while refusing to invent
the content being processed.
