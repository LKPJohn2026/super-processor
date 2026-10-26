# Changelog

All notable changes to Super Processor are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- The system-RAM fallback can no longer run the computer out of memory. A
  preview next to a game paged about 80 GiB into RAM and the kernel began
  killing programs. Three limits now hold it:
  - The child works in the shortest pieces (17 frames) instead of starting
    again at 8 s, since FlashVSR's memory grows with the frames in a call.
  - Its allocator stops at a byte limit: the free VRAM plus what RAM can be
    spared, which is what is available past a reserve of a quarter of the
    RAM (at least 8 GiB), and never more than half the machine. Past the
    limit it returns its cache to the driver and then refuses, which ends
    in the usual out-of-memory message. It throws to refuse: PyTorch does
    not check a null block, which came back as a tensor at address 0 and
    broke the CUDA context.
  - The wizard stops the child if available RAM falls below half that
    reserve, whatever the cause.
  With too little RAM to spare, no child is started. The allocator's cache
  also reuses any block up to twice the size asked for and returns itself
  to the driver once more than 2 GiB sits idle: reusing only exact sizes
  had left 13.6 of 27.6 GiB idle for one 72-frame preview. That preview,
  with 8 GiB of VRAM free, now peaks at 22.3 GiB held and leaves 53.6 of
  78.9 GiB of RAM available, in 45 s.

## [3.6.0] - 2026-10-25

### Added

- FlashVSR uses system RAM as well when the GPU is too full. If even the
  shortest piece (17 frames) runs out of GPU memory, the clip is restored
  once more in a child process that uses CUDA unified memory: what fits
  stays on the GPU and the driver pages the rest to system RAM. It is slower,
  not different: at 2x of 854x480 with 16, 12, or 8 GiB free, every clip
  finished at 0.43-0.70 s a frame (0.27-0.30 s with the GPU to itself) and
  matched the normal output's SSIM. Its allocator keeps freed blocks for
  reuse like PyTorch's own; asking the driver for each block made it three
  to four times slower. It is built with `nvcc` on first use; without the
  CUDA toolkit, or off Linux (Windows drivers do not let managed memory
  grow past free VRAM), the run fails as before, with the same advice.

### Changed

- At most 10 segments, each at least 3 s, and fewer is better. Gemini is
  asked for that, and the repair holds it whatever the reply: while there are
  too many segments or one is too short, the shortest is merged into its
  shorter neighbour, keeping the longer one's label and what both contain.
  Gemini's 22 segments for the Sintel trailer (several only 1 s long) become
  10 of 3.9-6.9 s with every face, hand, and text flag kept. The minimum was
  0.5 s, and there was no maximum.

- FlashVSR fits next to other programs on the GPU. Its working memory grows
  with the frames in one call (at 2x of 854x480: 15.5 GiB for 16 frames,
  16.9 GiB for 24, against 3.4 GiB of weights), so a clip that runs out of
  memory is retried in pieces of half the length, down to 17 frames. Each
  finished piece goes to disk before the next starts, so memory follows the
  piece, not the clip. With a game holding 9 GiB, a 130-frame clip now
  finishes in 10 pieces instead of failing. Pieces are planned in whole
  frames, and the last one is full length so no remainder falls below what
  FlashVSR accepts. Keeping the weights in RAM was measured too: it saves
  only about 0.5 GiB, so it is not used.

### Fixed

- A failed FlashVSR call no longer keeps GPU memory. Its traceback held the
  frames' tensors and the pipeline, and the error raised from it kept that
  alive, so a preview that ran out of memory left 13 GiB taken until the
  server restarted, and the retry failed too. The traceback is now cleared
  before the memory is released, and the pipeline's own caches are cleared
  after every clip.

## [3.5.2] - 2026-10-25

### Fixed

- Every segment lines up with the source frame for frame. Parts were cut
  half a frame before their first frame; with the time written to six
  decimals that sits on FFmpeg's rounding edge, and in a file that keeps one
  tick per frame (the Sintel trailer) 141.5/24 rounded down, so 4 of its 14
  segments started a frame early and the seam check flagged the fade at
  0:09.8. Parts are now cut a quarter frame early. FlashVSR also returns up
  to three frames short of a clip, and the last frame used to be repeated in
  their place; clips now get that many copies of the last frame before
  FlashVSR, which are trimmed off after, so the tail is real frames.
- A look no longer pushes black below video black or white above video
  white. FFmpeg's `eq` scales around the middle of 0-255, so the trailer's
  credits look (contrast 1.10, brightness +0.02) took black from 16 to 7.5,
  and black visibly changed at the cut into it. The finish step now ends by
  holding luma to 16-235 and chroma to 16-240, after the grain; black now
  stays at the source's 16.5 across that cut.
- The final audio mux no longer drops frames. It used `-shortest`, so a
  source whose audio ends before its picture (the Sintel trailer: 51.95 s of
  audio, 52.21 s of picture) lost its last frames after the frame-count check
  had passed. The mux now runs exactly as long as the picture, cutting longer
  audio there, and keeps the picture without audio if the frame count would
  still change.
- FlashVSR no longer crops the picture. Its loader centre-crops the upscaled
  frame to whole 128-pixel blocks, so an 854x480 clip at 2x came back as the
  middle 1664x896 of 1708x960, and the delivery blend laid that over the full
  frame squeezed to the same size, out of line toward the edges. Each clip is
  now padded with its own edge pixels to whole blocks before FlashVSR and
  cropped back after, so the output is exactly the source times the scale and
  lines up with it (SSIM against the source 0.97, was 0.87, on a trailer
  clip).

## [3.5.1] - 2026-10-24

### Changed

- The Gemini key pasted on the setup screen is saved to `credentials.json` in
  the user's config folder (`~/.config/super-processor` on Linux,
  `~/Library/Application Support/super-processor` on macOS,
  `%APPDATA%\super-processor` on Windows; `SUPER_PROCESSOR_CONFIG_DIR`
  overrides it), readable only by the user, and used on every later run. It
  used to live in the server process and, best-effort, the OS keyring, so a
  restart could lose it. `GEMINI_API_KEY` still wins; an old keyring entry is
  read last and never written. The setup screen says where the key in use
  comes from, and a key that cannot be saved is an error instead of silently
  lasting only until the next restart.

### Fixed

- Tests no longer touch the real keyring or config folder. A test that saved
  a key used to overwrite the user's keyring entry with `abc123`.

## [3.5.0] - 2026-10-23

### Added

- Go back and forth between Setup, File, and Segments. Those steps in the rail
  are links when they can be reached, and each screen has a back link
  (`POST /goto`). Going back keeps the job: approving the same segments again
  carries on at the segment you were on.
- The file step opens the desktop's own file dialog ("Choose a file…",
  through the XDG portal on Linux) and uses the real path, or takes a video
  dropped on the page, which is streamed into `<jobs>/uploads/` and removed
  again if it cannot be used. Typing a path moved under "Type a path
  instead".
- Upscaling is a choice on the ready screen: 2×, 4× when the source is small
  enough, or "Don't upscale" (scale 1: same size, clean-up and finishing only,
  no FlashVSR). Each segment preview can also turn FlashVSR on or off for
  that segment; later Gemini proposals keep the editor's choice.

### Changed

- The step rail drops Upscale: Setup · File · Segments · Previews · Result.
- A file that cannot be read says so in one line, naming the file, instead of
  showing FFprobe's output.

## [3.4.0] - 2026-10-23

### Changed

- Gemini errors read as one line. The API's error body (several kilobytes of
  JSON with help links and quota metrics) becomes what went wrong and when to
  try again: "daily quota used up; try again in about 16 h", "the API key was
  rejected; replace it on the setup screen", "this model is no longer
  available". When several models fail the same way, the failover names them
  together once.
- The Segments screen shows a timeline bar with one block per segment, as
  wide as its share of the video, and each segment with the sampled still
  nearest its middle, its time range and length, and what it contains as
  tags. The approve button says how many segments it approves.
- When Gemini could not split the video, the Segments screen says so instead
  of "Gemini split the video", and the reason sits in a notice box; the
  segment preview shows its warnings the same way.
- The pick screen describes the segment loop instead of a whole-file pass.
- The React shell matches all of the above.

### Added

- `GET /segment_stills/t_<ms>.jpg` serves the still grid to the Segments
  screen; other names and symbolic links are refused.

## [3.3.2] - 2026-10-23

### Changed

- The render's progress follows the work instead of jumping between fixed
  steps. Each part counts for its share of the source frames, FlashVSR
  reports after every chunk of about 8 s, and the bar reads "Upscaling on
  the GPU: N% of the frames". `render_plan`, `render_shots`, and
  `apply_range_revise` take an `on_progress` callback, and `UpscaleRequest`
  carries one to the engine.

## [3.3.1] - 2026-10-23

### Changed

- `docs/system-design.md` describes the segment loop: local still sampling,
  repaired Gemini segments, one preview per segment with a seeded window,
  optional pass-through spans, the full restart that keeps the chat, the
  shared `segments.json` name, and the job files and steps that go with
  them.

## [3.3.0] - 2026-10-23

### Added

- Full restart from the result screen. A note sent with "Start over from the
  segments" (`restart=1` on `POST /api/result`) clears the job's segments,
  plans, previews, output, and work folders, keeps `gemini_chat.json` with
  the note added, and goes back to sampling. Gemini then proposes segments
  and settings again with the whole earlier exchange in view. A note without
  `restart=1` still retunes one time range.

## [3.2.0] - 2026-10-23

### Changed

- The wizard now runs the segment loop: pick → segments → one preview per
  segment → ready → upscale → result. Gemini proposes segments from a local
  still grid; the editor approves them or sends a note. Each segment is then
  planned, previewed, and checked on its own, and approved or redone with a
  note before the next. A ready screen lists every segment's settings before
  the full render. `SUPER_PROCESSOR_SEGMENT_LOOP` is gone.
- Both the HTML pages and the React shell show the new steps, with a progress
  bar on the busy ones. The step rail reads Setup · File · Segments ·
  Previews · Upscale · Result.
- Gemini's segments say what each contains (faces, hands, text,
  fine_pattern), so those segments are boxed and held down as shots were.

### Removed

- The shot flow's wizard steps (`finding_shots`, `shots`, `planning`,
  `looks`), their routes (`/api/scan`, `/api/shots`, `/api/plan`,
  `/api/looks`), and shot merge and split in the wizard. A job saved on one of
  those steps goes back to the pick step with a message.

## [3.1.0] - 2026-10-23

### Added

- Spans and previews marked `use_upscale: false` skip FlashVSR. The clip is
  resized with lanczos and goes through the same delivery encode at strength
  0, so it keeps its look, matches its neighbours' size, and the joined
  output still holds every source frame once. Full renders, range revises,
  and previews all follow it, and `upscale_plan.json` spans carry the flag.

## [3.0.0] - 2026-10-23

### Added

- The segment loop runs in the wizard when `SUPER_PROCESSOR_SEGMENT_LOOP=1`
  is set. After the pick, one worker samples the stills and asks Gemini for
  segments (`GET /api/segment?run=1`). `POST /api/segments` approves them or
  sends them back with a `note`. Each segment is then planned, previewed,
  and checked on its own (`GET /api/analyze?run=1`); `POST /api/preview`
  approves it and moves to the next, or redoes it with a `note`
  (`resample=1` picks a new preview window). When every segment is
  approved, `POST /api/ready` with `action=render` starts the full render.
  `/api/state` returns `segments`, `segment_preview`, and `segment_index`.
  The page shell does not show these steps yet.
- Each segment's preview is a random window of up to 3s inside it, seeded
  by job, segment, and preview count, and kept across redos.

## [2.9.0] - 2026-10-23

### Added

- Wizard steps for the segment loop (`sampling`, `proposing`, `segments`,
  `analyzing`, `preview_segment`, `ready_to_render`) and a `segment_index`
  in the wizard state. Nothing leads to them yet; the page shell shows a
  placeholder on them.
- Worker progress. The scan, planning, and render workers record
  `{phase, pct, message}` in the wizard state as they go, and `/api/state`
  returns it as `progress` while a worker runs.

### Changed

- The wizard state file is schema 3. Schema 2 files load unchanged; a file
  older than schema 3 on the removed flow's `analyzing` step still goes back
  to the pick step.

## [2.8.0] - 2026-10-23

### Added

- Gemini proposes the segment loop's segments from the still grid and its
  times (`GeminiClient.propose_segments`). A re-proposal carries the
  editor's note, and the saved chat is replayed so Gemini sees what it
  proposed before. `repair_coverage` turns any reply into segments that
  cover the clip once: gaps are split down the middle, an overlap is cut
  where the later segment starts, and slivers under 0.5s are folded in.
- Gemini can set `use_upscale: false` on a shot's look so it skips FlashVSR.
  The setting is kept in `shot_plan.json`; rendering follows it from a
  later release.

## [2.7.1] - 2026-10-23

### Fixed

- The FlashVSR runner test skips when `ffmpeg` is missing from `PATH`, so
  the release wheel tests pass. The 2.7.0 release built no wheels and was
  not published; 2.7.1 ships its changes.

## [2.7.0] - 2026-10-23

### Security

- The wizard makes a random token each run and prints it after the URL.
  Every POST and `?run=1` request must carry it (`X-Wizard-Token` header or
  `token` field); the wizard's own pages and `/api/state` hand it out, and
  pages refuse to be framed. Another site can no longer drive the wizard in a
  browser that omits `Origin`. This replaces the `replace=1` key guard, which
  any page able to post could also send.
- A junk or negative `Content-Length` now gets 400 instead of dropping the
  connection.

### Changed

- Picking a video through a symbolic link works again; the link is followed.
- Docs: the GitHub Pages shell needs
  `SUPER_PROCESSOR_ALLOWED_ORIGINS=https://lkpjohn2026.github.io` to drive a
  local wizard.

### Added

- Segment loop groundwork. `segment_loop.py` samples a still grid from the
  source (about one a second, at most 120, always the first and last frame)
  into `segment_stills/t_{ms}.jpg` inside the job folder, and keeps the
  wizard's segment list in `segments.json` (schema 2, `kind: "wizard"`). The
  older segment review refuses to read or overwrite that file, and the
  wizard refuses the older review's. The wizard does not use the loop yet.
- Shimmer check. Each preview's fine-detail layer is measured for how much
  it changes frame to frame relative to how much there is, against the
  source. A preview whose texture is over 1.6x less stable than the source's
  fails as `flicker` even when Gemini passed it, loses 0.15 strength, and is
  redone; Gemini also sees the number. The Looks screen shows it.
- Seam check. After every render, each place FlashVSR restarted (chunk
  boundaries, boundaries between unlike shots, revise edges) is compared
  with the source; a frame jump 2.5x beyond the source's is reported on the
  Result screen with a ready-made "smooth" note. Results are in
  `render_report.json`.

- Protected regions. Before planning, Gemini boxes the faces, hands, and text
  in each shot that has them, from stills at its start, middle, and end. In
  the delivery blend the FlashVSR share is held to 0.3 inside face and hand
  boxes and 0.15 inside text boxes, feathered so there is no seam. The Looks
  screen draws the boxes over the previews. A shot whose boxes could not be
  placed falls back to the whole-shot 0.6 cap.

- Per-shot looks (loop 2). After the shot list is approved, Gemini proposes a
  strength and look for each shot (and one scale for the video), each shot
  gets a short before/after preview, and Gemini checks the previews for
  identity changes, bad anatomy, garbled text, waxy skin, oversharpening,
  halos, fake texture, colour shifts, and softness. A failed shot gets
  corrected settings and a new preview. The new Looks screen shows each pair;
  the editor approves all or writes a note that redoes one shot. Strength is
  capped at 0.6 on shots with faces, hands, or text, and 4x is only offered
  for sources up to 960 pixels on the long edge. Without Gemini the recipes
  come from the measurements. New steps `planning` and `looks`, routes
  `/looks`, `/api/plan`, `/api/looks`, and `/previews/`.
- `FlashVsrEngine.upscale_many` restores several short clips with one model
  load; the previews use it.

- Shot review (loop 1). After a file is picked, FFmpeg finds the cuts
  (`scdet`), measures each shot in its own pass (blockiness, blur, noise,
  brightness, contrast), and saves a still per shot. Gemini labels each shot
  from its still and numbers, using fixed lists of problems and contents. The
  new Shots screen lets the editor merge or split shots and approve the list;
  approval writes one upscale span per shot. If Gemini cannot label the
  shots, the list keeps the measurement hints and says why. New steps
  `finding_shots` and `shots`, routes `/shots`, `/api/scan`, `/api/shots`,
  and `/shot_stills/`.

- Per-shot looks. Each span in `upscale_plan.json` now carries a `look` of
  bounded FFmpeg settings applied in a fixed order around FlashVSR: clean
  (`deblock`, `denoise`) before it, finish (`contrast`, `brightness`,
  `saturation`, `gamma`, `grain`) after it. Cleaning first gives FlashVSR
  fewer compression blocks and less noise to sharpen into made-up texture.
  The plain-upscale base of the strength blend is cleaned the same way.
  Plans without a `look` load as neutral, so existing jobs render as before.
- A plan with several shots renders each shot on its own and joins them by
  stream copy. Shot edges snap to source frames, and the join is checked to
  hold exactly the source's frame count. Neighbouring shots with the same
  settings render as one part, so FlashVSR loads once per run of alike shots.

### Changed

- Gemini stays on the legacy `generateContent` API. Each call resends the
  stored chat as `contents`, and a model switch repeats that full request
  against the next model. The Interactions API is not used.

- The strength blend now takes shapes, colour, and layout from the plain
  upscale of the source and only fine detail (below a 1.5 source-pixel
  Gaussian) from FlashVSR. FlashVSR can sharpen what is there but can no
  longer move, reshape, or recolour it, at any strength. Strength 1.0 is no
  longer "FlashVSR alone"; it is the source's structure with all of
  FlashVSR's fine detail.
- The 0.6 strength cap on shots with faces, hands, or text now applies only
  when no protection boxes were placed.
- A scale change on part of the clip keeps every shot's strength and look
  and only changes the scale, instead of collapsing the plan into one span.
  A result note keeps the look of the shot it names.

### Fixed

- The wizard no longer traps the user after Done. Done and Result have a
  "Start a new video" button (`POST /new`, `/api/new`) that keeps the old job
  on disk and returns to the pick screen, or to setup when no Gemini key is
  available. Before, `super-processor review` reopened the same Done page.
- A first upscale that fails (no CUDA, missing FlashVSR weights, a bad file)
  marks the job failed and returns to the pick screen with the error, instead
  of showing a Result page with no video. A failed range revise still keeps
  the previous `output.mp4` on the Result screen.
- Intro copy, the static shell tour, and the system design no longer claim
  "no invented frames, faces, or scenes" or that FFmpeg applies the repairs.
  They now say FlashVSR adds fine detail, strength sets how much, and every
  frame and its timing are kept.

### Removed

- The wizard's Gemini split / FFmpeg enhance flow: the analyze, overview,
  split choice, and per-segment enhance steps, their pages and API routes
  (`/analyze`, `/overview`, `/split`, `/segment`, `/previews/`, stills), the
  segment-choices render, and Gemini's split and enhance calls and schemas.
  Since the switch to FlashVSR, picking a file goes straight to the upscale,
  so none of it was reachable. A saved wizard state on one of those steps now
  loads as the pick step. The CLI `diagnose`, `plan`, `segment`, and `plans`
  commands are unchanged.

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
- `output.mp4` and preview clips are streamed with HTTP Range support (206,
  416, HEAD) instead of being read whole into memory, and sent with
  `Cache-Control: no-store` so a revise is not hidden by the browser cache.
  The final mux writes the `moov` atom first (`+faststart`).
- The wizard refuses inputs above 1080p (1920x1080 in either orientation) or
  longer than 30 minutes at the pick step, before a job is created.

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
