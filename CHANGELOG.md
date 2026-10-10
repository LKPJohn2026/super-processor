# Changelog

All notable changes to Super Processor are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Local FlashVSR restoration upscale on the wizard picture path. FFmpeg trims,
  splices, and copies audio. Gemini turns a result note into a time range,
  scale, and strength, then the cited range is restored from the original source.

### Fixed

- The wizard serves only localhost: a foreign `Host`, a foreign `Origin`, or
  a cross-site request that would start work is refused with 403. CORS answers
  only the wizard itself, the Vite dev/preview ports, and the published React
  shell; add origins with `SUPER_PROCESSOR_ALLOWED_ORIGINS`.
- Upscale and analyze passes run on one worker thread. `/render?run=1` and
  `/api/render?run=1` start a pass and return at once; a reload or a second
  tab no longer starts a second GPU pass on the same job. Posts are refused
  while a pass runs. FlashVSR loads are serialized process-wide.
- A result note that changes scale on part of the video re-renders the whole
  clip at that scale instead of splicing two resolutions into one stream.
  A same-scale note trims the spans it overlaps instead of dropping them.
- Range revises no longer re-encode the parts of the video the note did not
  name. The upscaled picture is encoded once with a keyframe every second and
  no B-frames; a revise snaps out to those keyframes and stream-copies the
  head and tail. Work clips handed to FlashVSR are lossless.
- A Gemini call that times out moves on to the next candidate model instead
  of failing the request. The per-model read timeout is 60 seconds (was 120).
- Upscale strength now does what it says. It is the share of the FlashVSR
  picture in the output; the rest is a lanczos upscale of the same frames,
  mixed in the delivery encode. 0.0 invents nothing, 1.0 is FlashVSR alone.
  Before, strength only switched FlashVSR's attention settings, so every value
  from 0.0 to 0.54 produced the same output. FlashVSR now always runs at its
  upstream settings.
- Every FlashVSR clip is checked against the frame count and frame rate of
  the clip it was given. Up to 8 frames (or 2%) of difference at the end, or
  a rounded fps, is conformed losslessly; more raises before anything is
  spliced. A range revise that would change the output's total frame count
  is refused and the previous output is kept.

### Planned

- Wire a real local LLM path behind the deferred stub screen (OpenAI-compatible
  or native multimodal endpoint).
- Chunked FlashVSR execution for clips up to about 30 minutes.

## [2.6.0] - 2026-10-03

### Added

- Localhost website UI for the wizard: polished server-rendered screens with
  step rail, LLM type choice (Gemini live + deferred local stub), lettered
  split/enhance options, and the full sketch flow through result/happy.
- CI HTTP test drives the wizard website from introduction through done
  (FakeTransport Gemini + FFmpeg), asserting each screen’s copy and step rail.

### Fixed

- Wizard tests that encode tiny clips skip when `ffmpeg` is missing from
  `PATH` (cibuildwheel / headless release environments).

## [2.5.0] - 2026-10-03

### Added

- Enhance **Something else** loop: `combine_enhance_revise_message` /
  `revise_enhance` feed the user note plus prior structured options back into
  Gemini; the wizard stays on the segment until an option is previewed and
  accepted (tests cover five revise rounds).

## [2.4.0] - 2026-10-03

### Added

- Localhost Gemini wizard via `super-processor review` (no job id): setup,
  pick file, structured split choice, per-segment options with short previews,
  enhance-and-concat render, and a happy/revise result step.
- Gemini structured-output client (`gemini.py`) with chat persistence,
  dynamic-FPS sampling, allowlisted ops validation, and failover across
  multimodal models with usable RPM.
- Design docs: revised [docs/system-design.md](docs/system-design.md) and new
  [docs/ui-wizard.md](docs/ui-wizard.md).
- CI job **Gemini live API** calls the real Gemini structured-output endpoints
  using the repository secret `Gemini_API_Test` (mapped to `GEMINI_API_KEY`).

### Changed

- Wizard product path uses Gemini for split/diagnosis; classical CV estimators
  are not used by the wizard. Export keeps source aspect (no vertical social
  default).
- Matrix unit tests run with `-m "not gemini_live"` and never receive the
  Gemini secret; live split/enhance/multi-turn checks run only in the dedicated
  job.

### Fixed

- Gemini API key storage works without an OS keyring backend (headless CI).
- Live Gemini calls close HTTP error bodies, retry transient 503s, and skip
  image-generation models when discovering candidates.
- Windows CI retries Chocolatey FFmpeg installs and skips ffmpeg-dependent
  wizard tests when the binary is missing from `PATH`.

## [2.3.1] - 2026-09-26

### Fixed

- Hardware-decode sample tests pass an explicit ffmpeg path, so wheel
  builds no longer fail when the image has no FFmpeg binary.

## [2.3.0] - 2026-09-26

### Added

- `doctor` lists `hevc_cuvid`, `hevc_qsv`, `hevc_amf`, and
  `hevc_videotoolbox` next to the hardware encoders.
- `segment JOB --decode NAME` reads sample frames with that decoder. One-hertz
  rows keep the same shape. A decoder this ffmpeg does not have fails before
  sampling.
- A frame the device cannot decode is read again in software. The sample row
  and the segment label match a software read.

### Fixed

- The hardware-encoder doctor check and `plans --choose` test no longer
  require `ffmpeg` on `PATH`. Wheel builds were failing when the image had
  no FFmpeg binary.

## [2.2.0] - 2026-09-26

### Added

- `doctor` reports `hevc_nvenc`, `hevc_qsv`, `hevc_amf`, and
  `hevc_videotoolbox` as present or missing. `libx265` stays required.
- A recipe may name one of those encoders. Any other codec name fails
  validation. The default remains `libx265`.
- `plans JOB --choose N --encoder NAME` renders with that encoder's own
  rate-control flags. Trim and the filter graph stay the same. A missing
  device fails with the doctor line before any segment is encoded.

## [2.1.0] - 2026-09-26

### Added

- `review JOB` serves a localhost split page. It lists each segment's range,
  context, problem, and still. Accept and a note call the existing `segment`
  command.

## [2.0.0] - 2026-09-26

### Added

- A named fault filters one segment's treatments. A sentence maps onto that
  fault list only after a fault rejection. Neighbors keep the treatment they
  already shared.
- `plans JOB --choose N` is the approval to render. Each kept segment is
  encoded with libx265, audio is cut on the same timestamps, and the parts
  are concatenated.
- The size-cap bitrate uses the sum of the kept durations.

### Fixed

- A split of an oversized range stays at least 5 seconds from both ends, so a
  loud change at the edge cannot leave a piece shorter than 5 seconds.

## [1.3.0] - 2026-09-26

### Added

- A fixed treatment list for each segment problem. Silhouette lifts shadows
  harder than low light. Strong denoise cannot share a treatment with sharpen,
  and steps stay in allowlist order.
- Five timeline plans. The first is the highest score. Later plans change as
  many look groups as they can, and neighbors that share a look share a
  treatment. `plans JOB` writes `plans.json` and prints each assignment.
- A three-second preview window around each key frame, clamped inside that
  treatment's trim, and one cached encode of that window.

## [1.2.0] - 2026-09-26

### Added

- Split loop for a local file: one-hertz sample rows, segments of 5–120
  seconds (at most 15, files over 30 minutes refused), a context and problem
  label, one still per segment, and shared look groups.
- `segment` prints a proposed split. `--note` re-splits inside those bounds
  (too many, too few, a boundary move, or a relabel). `--accept` records the
  timeline; a second accept leaves the job unchanged.
- Sharpen and trim on the recipe allowlist. Sharpen stays inside a mild
  unsharp range. A trim must keep at least 5 seconds.

## [1.1.0] - 2026-09-20

### Added

- Classical saliency-aware 9:16 reframe (edge-energy subject tracking) with
  planned crop windows wired into FFmpeg templates.
- Structured LLM recipe patches (`{"patches":[...]}`) outside `--safe-mode`,
  still clamped to allowlisted ops/params.
- Optional OS keyring lookup for API keys via `keyring` when installed.
- Doctor checks for `libx265`, vid.stab / `deshake`, and optional `libvmaf`.
- `--acknowledge-size-risk` on diagnose / plan / apply / job reframe so a
  below-floor size cap becomes an explicit warning instead of a hard error.
- Live encode progress on stderr during preview and apply.

### Changed

- Package classifier moved from Pre-Alpha to Alpha.
- Local `.dogfood/` artifacts are gitignored.

## [1.0.0] - 2026-09-20

### Added

- End-to-end product loop: diagnose → plan → preview → apply → show.
- CV look/motion estimators, recipe validation, FFmpeg templates, QA, VMAF
  helper, and safe-mode LLM/VLM advisory planning.
- Soft stabilize fallback to `deshake` when vid.stab is unavailable.
