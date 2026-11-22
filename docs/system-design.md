# Super Processor — System Design

Super Processor is a local-first video processor. A localhost wizard has
Gemini watch a small, audio-free copy of footage you already shot, cut it
into segments, and choose the output size; it then plans and previews each
segment in turn for you to approve, runs a local SPAN restoration
upscale, and lets you revise a time range or start over with a
plain-language note. Gemini only proposes segments and a scale from the
offered ones, picks bounded per-segment settings, checks previews, and turns
a note into a bounded range, scale, and strength; FFmpeg samples, measures,
trims, splices, encodes, and copies audio. No model writes shell commands.
The CLI keeps the older FFmpeg-filter grade paths for automation.

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
| Processing | Usually hosted | Local SPAN and FFmpeg |
| Main risk | Invented content and temporal drift | Invented fine texture (bounded by strength) and flicker |
| Recovery | Regenerate | Re-run a time range at a lower strength |
| Trust model | Inspect final pixels | Same frames and timing as supported CFR sources; each revise is stored as JSON |

The closest alternatives are manual FFmpeg scripts, conventional nonlinear
editors, and black-box enhancement applications. Super Processor aims to
combine FFmpeg's determinism with guided diagnosis, while keeping every
operation inspectable.

## Product decisions

### Interim timing safety: verified CFR only

The restoration path still converts frame indexes to times with a constant
frame-rate clock. It does not yet preserve arbitrary variable-frame-rate
(VFR) timelines. To keep the same-frames/timing contract honest, the wizard
checks timing before creating a job, and shared render, revise, preview, and
engine entry points check before GPU work or replacing an output.

The guard CPU-decodes the complete first video stream with ffprobe. Decoded
presentation timestamps must be strictly increasing and compatible with a
uniform rational clock at the rate the renderer uses, allowing only consistent
source-time-base quantization. The final frame duration must also agree with
that clock. Missing/unverifiable timestamps, decode failures, incompatible rate
metadata, and nonuniform timing fail closed. The `is_vfr` metadata heuristic is
informational, not an admission decision.

Successful checks use a bounded process-local cache keyed by resolved source,
file identity, size, and modification metadata. Nested calls and retries reuse
checks; work clips do not re-scan the original. Changed sources and new
processes require verification again. The guard does not normalize inputs;
errors advise that external CFR conversion may change frames/timing. Supporting
VFR later requires a source-frame/timestamp mapping across selection, inference,
blending, concatenation, revises, and audio, not a different average-rate guess.

### Restoration upscale instead of an FFmpeg grade

The wizard picture path is a local GPU restoration upscale (SPAN 2x). FFmpeg
probes, trims a cited time range, splices that range back, and copies audio.
Before the first render, the segment loop (below) cuts the clip into
segments and gets each one's settings approved on a short preview. Gemini
turns a result note into `start_s`, `end_s`, `scale`, and
`strength` (0–1). Re-runs read the original source for that range so detail
does not stack.

There is one model: SPAN (`span_sr.py`), the 2xLiveActionV1_SPAN weights
loaded through spandrel, in half precision on the GPU. It sees one frame at a
time, so a clip needs no padding to blocks and no extra frames; the frames go
to the GPU in batches of about four megapixels of input (at most 8 frames),
so GPU memory follows the batch, not the clip. The weights are checked by
size and sha256 before loading (`SUPER_PROCESSOR_SPAN_WEIGHTS` points at
another file): `load_span` reads the file once, hashes those bytes and hands
the same bytes to `torch.load(..., weights_only=True)`, so a file swapped
after the check is never deserialized. The gpu extra needs torch 2.6 or
newer, the first release where `weights_only=True` cannot be bypassed to run
code (CVE-2025-32434). One lock keeps two wizard threads from loading it at once, and
a batch of preview clips shares one load.

It replaced FlashVSR v1.1 and FlashVSR+ outright. In a bake-off on an RTX
5090, FlashVSR blended at strength 0.5 was almost the same as plain SPAN 2x
on SD (consistency 41.9 against 41.5 dB, detail 32.1 against 32.6, added
shimmer +0.04 against +0.07) at 97 against 3 minutes per 20-minute film and
26 against 2 GB of VRAM; 2x of 1440x1080 did not fit 32 GB with FlashVSR,
while SPAN does it at about 30-40 fps in about 4-9 GB. Recent jobs used
strength 0.5 or less for 85% of spans, never above 0.7, and never 4x.

Each span in the plan is a segment with its own look (`look.py`): bounded
FFmpeg settings that always run in the same order.

| Stage | Settings | Why |
|---|---|---|
| Clean | `deblock`, `denoise` (0–1) | The upscaler sharpens blocks and noise as if they were detail; removing them first leaves it less to sharpen |
| Restore | `scale`, `strength`, protected regions | SPAN's fine detail on the cleaned plain upscale's shapes and colour; held down inside face, hand, and text boxes |
| Finish | `contrast`, `brightness`, `saturation`, `gamma`, `grain` | Grade and texture on the restored picture |

Filter text is built from the numbers in code; model output never reaches
FFmpeg as text. None of these filters changes the frame count, and a
multi-segment render checks the joined result against the source's frame
count.

A span can skip the restore stage (`use_upscale: false`). It is resized with
lanczos to the same scale and goes through the same delivery encode at
strength 0, so it keeps its look and lines up with its neighbours frame for
frame. Gemini sets it where the upscaler would only harm.

Text-to-video, inpainting, and face reenactment stay out. A lower strength
keeps less of SPAN's fine detail. The job stores `upscale_plan.json` with
ordered spans, one per segment. Each clip streams through SPAN in one pass,
whatever its length, with the per-frame work on the GPU. An FFmpeg decoder
(`-hwaccel cuda` when a cached probe shows it works, otherwise software;
codecs NVDEC cannot read fall back to software inside FFmpeg) applies the
look's clean filter and writes packed `yuv420p` to a pipe. A reader thread
fills blocks of about 128 MiB of input frames. The main thread hands each
block to `span_sr.py`, which, in torch on the GPU, converts YUV to RGB
(BT.601 limited range, bilinear chroma), runs SPAN, makes a Lanczos-3 2x
base, mixes `base + strength × (detail(SPAN) − detail(base))` with a Gaussian
of 1.5 source pixels, and converts back to `yuv420p` (2x2-averaged chroma).
Per-pixel strength comes from a map built once per clip on the GPU that
mirrors the protected-region formula. A writer thread pipes the 2x `yuv420p`
frames into the delivery encoder, which applies only the look's finish filter
(grade and grain, on the CPU, only for looks that have one) and encodes. At
most two blocks wait to be upscaled and one to be written, so host memory
stays bounded and nothing goes to scratch disk. The output holds exactly the
source's frame count: a decoder a few frames short (up to 8, or 2%) repeats
the last frame, extra frames are dropped, and a larger shortfall stops the
render rather than splice a misaligned picture. Preview batches use the same
path under one model load. A pass-through span is the blend's base at
strength 0, so it is one encode straight from the source: NVDEC decode, the
look's clean step, an FFmpeg lanczos resize, the finish step, and the
delivery encoder. On 200 frames of 1440x1080 that took 2.6 s and 0.027
CPU-seconds a frame with NVENC, where a 2x `libx264` work file and a 4K
FFmpeg blend (whose blurs and lookup tables strength 0 threw away) took
5.0 s and 0.26, for identical frames. File inputs keep the FFmpeg blend.

On an RTX 5090, 1500 frames of 1440x1080 rendered to 2880x2160 at 26.8 fps
end to end with 0.054 CPU-seconds per frame (the timing preflight and probes
included); the CPU went from 70 C to 76 C over the minute. On 200 frames at
strength 0.5 its average Y, U and V were within 0.3 levels of the previous
stream's output, at 44.96 dB PSNR (SSIM 0.988) against it. That previous stream ran only SPAN on
the GPU: 22.2 fps at 0.43 CPU-seconds per frame (about 9.5 of 32 threads),
of which the x264 encode was about 50%, the FFmpeg blend at 4K (lanczos base,
two Gaussian blurs, mixes) 26%, host copies and the pipe 15%, and decoding
4%. With the GPU near 500 W heating the case, a 20-minute film took the CPU
to 95 C. The chunked pipeline before it used 1.21 CPU-seconds per frame, ran
a 16-core CPU at its 95 C limit, and wrote about 63 GB of lossless 4K chunk
files for a full-length 1080p film. Long and memory-pressure acceptance
remain pending.

When another program holds GPU memory and a batch runs out, it is retried
with a smaller batch, down to one frame. If even one frame does not fit, the
render stops with `GpuOutOfMemory` and advice to close other GPU programs.
The FlashVSR chunk-halving ladder and its system-RAM fallback are gone.

### Segment loop

The wizard works one segment at a time instead of planning every shot at
once. The steps are `sampling → proposing → segments → (analyzing ⇄
preview_segment) per segment → ready_to_render → rendering → result`.

- **Gemini watches the video.** FFmpeg writes `gemini_upload.mp4` in the job
  folder: no audio, long edge shrunk to at most 960 px (even dimensions),
  `libx264` veryfast CRF 28, original frame rate, and none of the source's
  metadata or chapters (`-map_metadata -1 -map_chapters -1`, no subtitle,
  data or attachment streams), so GPS location, title, author and
  recording time stay local. It goes to the Gemini File API with a
  resumable upload, and `gemini_video.json` records it so later requests
  reuse it while it has at least an hour left. The wizard deletes it from
  the File API (best effort, off the request thread, errors only logged)
  when the render starts, on "new video", and with the old key when the key
  is replaced, and clears the record so a later request uploads again;
  otherwise Google deletes it after about 48 hours.
  Gemini samples it at 1 fps: the segment request gets the whole video, and
  each segment's settings request gets that segment's stretch
  (`startOffset`/`endOffset`) with one full-size still. Video requests wait
  up to 300 s; a busy model (HTTP 500 or 503) is skipped at once rather than
  retried, and
  a timeout ends the model failover, since every model would read the same
  video.
- **Still grid as fallback.** FFmpeg also saves a still about every second
  (at most 120, evenly spaced, first and last frame included). The Segments
  screen shows them. If the copy or upload fails, or the client has no File
  API, Gemini proposes segments from those stills and their times, and the
  Segments screen says why. A segment request that fails with the video is
  asked again with the stills; the upload is kept for later requests.
  FFmpeg's `scdet` is not used for this flow: a cut is not the same as a
  change in what the restoration needs.
- **Gemini picks the scale.** With the segments, Gemini returns `scale` (1 =
  don't upscale, or 2) and `scale_reason` for the whole video; 2 is the
  default when it gives none. Both are offered for every source the wizard
  accepts (up to 1080p). A plan saved with scale 4, from before 4x was
  removed, loads and renders at 2x. The editor can still change it on Ready.
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
  a note that redoes only that segment. All segments share the scale chosen
  with the segments. Previews render at the plan's scale, and a 1x plan
  never runs SPAN for its previews.
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
machine. The wizard accepts verified CFR inputs up to 1080p and 30 minutes;
90–120 minute inputs are not supported. The full-quality source is not
uploaded to Gemini, and no audio is ever sent.

Gemini receives what the user has consented to send through the wizard: an
audio-free copy of the whole video (long edge at most 960 px, no metadata),
kept on Google's File API until the render starts (at most about 48 hours);
one still per segment, the stills it boxes
faces in, and a still from each preview pair; and, when the video cannot be
used, the segment grid (480 pixels wide, at most 120).

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
Each wizard call has its own schema: segments (times, label, contents) with
the output `scale` and its reason, a recipe per segment (strength,
`use_upscale`, look), protection boxes, a preview check, and a result note's
range (`start_s`, `end_s`, a `scale` the source allows, `strength` 0–1).
Values are validated, clamped, and unknown names dropped before anything runs.
The model has no shell tool, and FFmpeg argument lists are built by
engineering-owned code.

### Result note and range revise

The first pass restores the whole clip. The result screen asks whether the
user is happy; otherwise a note such as "less artificial detail from 2s to
5s" re-restores that range from the original source and splices it back.
For a single-recipe revision, untouched parts are stream-copied. A note that
changes scale or crosses multiple stored recipe spans re-renders the whole clip
from the source, preserving each span's look, regions, and pass-through choice.
Strength is the share of SPAN's
fine detail in the output; the rest is a lanczos upscale of the same
frames.

### Inputs up to 1080p and 30 minutes

The pick step refuses files above 1080p (either orientation) or longer than 30
minutes, before a job is created.

### Gemini split and per-segment enhance (removed)

An earlier wizard had Gemini propose timeline split layouts and per-segment
FFmpeg filter options with short previews, then concatenated the graded
segments. Once the picture path became a restoration upscale, that flow was no
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
| Encode | H.264: software `libx264` by default (H.264 hardware encoders optional) |

`reframe_vertical` and `encode_size_cap` are out of wizard v1 defaults.
Recipes saved with the old `encode_hevc_size_cap`, `hevc` export codec, or
HEVC encoder names load as their H.264 equivalents. The CLI export's default
CRF is 23: on 120 frames of 1440x1080, `libx264` `medium` CRF 23 made
40.63 dB in 4.0 MB where the `libx265` CRF 24 it replaced made 40.60 dB in
3.4 MB.

### H.264 only: NVENC delivery, libx264 elsewhere

Restoration delivery and wizard previews are H.264 only, from one of two
encoders. `h264_nvenc` (preset `p5`, tune `hq`, VBR at CQ 17,
`-b:v 0`, High profile) is used when a one-frame test encode succeeds;
otherwise `libx264` preset `faster`, CRF 16 (CI,
Macs, AMD). Both write `yuv420p` with a forced IDR every second and no
B-frames for range splicing, and every delivery encode of a job, plain and
pass-through parts included, uses the same encoder so parts join by stream
copy. A probe that works is kept per FFmpeg binary for the process; one that
fails is asked again after 60 s, so a passing failure (every NVENC session
held by another program when the wizard starts) does not keep a long-running
wizard on `libx264`. Each render, revise, and preview batch decides its
encoder once at the start, so a probe that changes its mind mid-render never
mixes encoders in one file. Probes pipe nothing: on Windows a timed-out
probe behind Chocolatey's shim would otherwise wait forever on pipes the
real `ffmpeg` still holds. Against a lossless 2880x2160 reference (200 frames, same keyframe
flags), `h264_nvenc` `p5` CQ 17 gave 81.8 MB, 44.29 dB PSNR and SSIM 0.98588
with almost no CPU; x264 `faster` CRF 16 gave 88.7 MB, 44.44 dB and 0.98655
for about 48 CPU-seconds. (CQ 16 was 92.9 MB at 44.97 dB, CQ 18 71.2 MB at
43.62 dB, and `p7` CQ 18 94.0 MB at 44.75 dB.)

NVENC holds VBR to the bitrate of the H.264 level, and left alone it picks
the lowest level the size and rate allow: 320x180 at 10 fps got level 1.1
(192 kbit/s), and moving grain came out at 26.9 dB where x264 made 41.8. So
NVENC is given at least level 4.1 (62.5 Mbit/s for High), or the lowest
level above it that the size and rate need (5.1 for 2880x2160 at 25 fps).
NVENC's H.264 took frames from 145x49 to 4096x4096 on the RTX 5090; outside
that the output is `libx264`.

The marker records `codec`, the encoder actually used, with its quality
setting and `version: 3`. A revise splices into an output only when its
marker matches this version and this machine's encoder; anything else
(outputs from earlier versions, or from the other encoder) is re-encoded
once into the current settings first, so one file never mixes encoders. No
path offers HEVC or AV1. The legacy CLI export (`plans --choose`, recipe
runs) is H.264 too: `libx264` by default, or `h264_nvenc`, `h264_qsv`,
`h264_amf`, or `h264_videotoolbox` by name. Doctor requires `libx264` only,
and reports whether `h264_nvenc` and CUDA decoding actually run (FFmpeg lists
them without a GPU).

### Reference-based offline evaluation

Offline evaluation still uses clean→degraded fixtures and VMAF where available.
Runtime success for the wizard is plan validity, successful previews, concat
integrity, and user acceptance.

## Architecture

```text
Pick file (localhost wizard) ──► FFprobe ──► 1080p / 30 min + decoded CFR check
   │
   ▼
silent copy (≤ 960 px) ──► Gemini File API, 1 fps (still grid if it fails)
   │
   ▼
Gemini: segments + scale ──► editor: approve or note
   │
   ▼  for each segment
measure + box faces/hands/text ──► Gemini: recipe ──► 3 s preview (SPAN or resize)
   │                                              │
   │                         Gemini check ◄───────┘
   ▼
editor: approve (next segment) or note (redo this one)
   │
   ▼
NVDEC decode + clean ──► GPU: YUV→RGB, SPAN 2x, lanczos blend at strength,
   │                     RGB→YUV (one pass per span; pass-through spans are one
   │                     NVDEC → lanczos → encode pass; no scratch files)
   ▼
h264_nvenc (or libx264) delivery encode (IDR every 1 s) ──► join + audio
   │
   ▼
result player ──► happy, a range note, or start over
                      │                       │
                      ▼                       ▼
          Gemini: note → range,       clear plans, keep chat,
          scale, strength             back to sampling
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
- `gemini_upload.mp4`, the audio-free copy Gemini watches, and
  `gemini_video.json`, its File API record (name, URI, expiry) or why it
  could not be used;
- `segment_stills/` and `segments.json` (the proposal, Gemini's scale and
  reason, its approval, and the editor's note);
- `shots.json` and `shot_plan.json` (each segment's measurements, boxes,
  recipe, preview window, and check), with `previews/`;
- `upscale_plan.json` spans with scale, strength, look, and `use_upscale`;
- `output.mp4` and its `output.mp4.delivery.json` encode marker;
- `range_work/`, the work folder for range splices.

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

- Gemini key lookup is `GEMINI_API_KEY`, then the per-user `credentials.json`,
  then a read-only legacy OS keyring fallback. Setup writes the credential file
  (0600 in a 0700 config directory), not a new keyring entry. Keys are redacted
  from logs; the file is a local secret, not an encrypted credential vault.
- The validator accepts only declared operations and bounded parameters.
- User-selected paths are normalized and passed without invoking a shell.
- Inputs must be real video containers. `probe.container_refusal` accepts a
  file only when every name in ffprobe's `format_name` is one of mov, mp4,
  m4a, 3gp, 3g2, mj2, matroska, webm, avi, mpegts or mpeg; concat, hls,
  dash, image2, lavfi, tee, sdp and every other reader are refused by the
  wizard pick (`input_limit_error`), `job probe` and `diagnose`, because a
  concat script named `.mp4` or a playlist makes FFmpeg decode other local
  files as the video. Every ffprobe of a file, and every FFmpeg `-i` of a
  file, is preceded by `-protocol_whitelist file` (`tools.FILE_INPUT`), so no
  input can reach a URL.
- Every program is started by full path. `tools.resolve_argv` wraps every
  subprocess argv in the package (a test scans for any that does not); it
  looks a bare name up on PATH with `tools.find_tool`, which skips empty,
  `.` and relative PATH entries, so on Windows an `ffmpeg.exe` in the current
  folder is never picked (Windows and `shutil.which` would otherwise look
  there first). PowerShell and `taskkill` come from `%SystemRoot%\System32`.
  The CLI also sets `NoDefaultCurrentDirectoryInExePath=1` on Windows, as a
  backstop for a tool that is not on PATH at all.
- Job data is private to the user on POSIX: `~/.super-processor` and its
  `jobs` folder are 0700 (tightened when an older version left them 0755),
  each job folder, `uploads/` and `outputs/` are created 0700, a job folder
  is tightened when opened, and files the program writes itself
  (`write_text_atomic`: wizard state, chat log, segment and plan JSON, pid
  files; the manifest; uploads) are 0600. FFmpeg's own outputs keep the
  umask's mode but sit in those 0700 folders. A `--jobs-dir` /
  `SUPER_PROCESSOR_JOBS_DIR` folder that already exists keeps its mode, and
  one that is world-writable without the sticky bit is refused. Windows has
  no POSIX mode bits; these steps are skipped there and the user's profile
  folder keeps job data private.
- An audio-free copy of the whole video (long edge at most 960 px), selected
  stills, preview stills, notes, and conversation context leave the machine
  through Gemini after setup, without the source's metadata (GPS, title,
  author, comments, creation time) or chapters. The copy is deleted from
  Google's File API when the render starts, on a new video or a replaced
  key, and Google removes it after about 48 hours in any case. Obtain
  permission for sensitive footage.
- The Gemini key is sent in the `x-goog-api-key` header, never in a URL.
  Keys with spaces or line breaks are refused. Every error, progress line,
  saved wizard state, API answer and page has known keys (and anything
  shaped like a Gemini key or a `key=` parameter) replaced with `***`.
  `doctor` shows only that a key is set and its last 4 characters.
- Numbers in Gemini's answers must be finite: `NaN`, `Infinity` and
  numbers too big for a float are refused when the answer is read.
  Local job JSON and chat logs may retain that context.
- No telemetry implementation is present. An opt-in telemetry policy is not an
  implemented feature.

## Boundary

Included:

- localhost wizard on the review server;
- local SPAN restoration upscale with FFmpeg trim, splice, and encode;
- Gemini structured result notes (range, scale, strength);
- CLI FFmpeg-filter grade paths (diagnose, plan, segment, plans);
- H.264 encoding everywhere (`h264_nvenc`, or `libx264` without it; the
  legacy CLI also takes the other H.264 hardware encoders);
- reference-based regression helpers where already present.

Deferred:

- local LLM in the wizard;
- vertical social export as the wizard default;
- desktop shell (Tauri/Electron);
- classical CV as product diagnosis;
- cloud render workers;
- generative video features;
- multi-track nonlinear editing.

Model-generated shell commands and FFmpeg shell strings are **prohibited**,
not deferred features. Models supply bounded structured data only.

## Acceptance and release evidence

The [acceptance matrix](acceptance.md) separates synthetic regression mechanics
from real GPU, Gemini, and browser acceptance. Real-footage calibration and
full seam coverage remain open: seam candidates are the renderer's part
edges and revised-range edges. Each part streams in one pass and OOM retries
shrink only the GPU batch, so neither adds joins. Do not infer quality, memory fit, or full seam coverage from fake-engine
tests or earlier timing examples.

The [ordered release plan](release-plan.md) records the pushed implementation
ledger and validation checkpoints. `v3.7.0` is proposed, not released; live
acceptance, remote CI verification, and release authorization are pending.

Super Processor's market position follows directly from these choices: use AI
to restore footage the camera already captured, never new scenes or frames,
and give the user a direct control over how much fine detail the restorer
adds.
