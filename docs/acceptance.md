# Reliability acceptance

This is the repeatable acceptance contract for the ordered hardening train,
not a claim that it has shipped. See the [release plan](release-plan.md) for the
commit ledger, historical validation checkpoints, and release authorization.
Proposed version: **v3.7.0**, pending gates. No tag or release has been created
for this train.

## Evidence rules

- Record the exact tested commit, commands, environment, artifacts, and outcome.
  A subset passing does not establish a full-suite pass for a later commit.
- Synthetic CPU/FFmpeg tests and `FakeUpscaleEngine` establish mechanics only.
  They do not establish restoration quality, GPU memory fit, live Gemini visual
  judgment, or browser compatibility.
- Use owned/licensed footage with permission for sampled images and notes to
  leave the machine. Gemini consumes quota and may incur charges. Never publish
  credentials, private media, full private paths, or unredacted chat logs.
- Keep footage, outputs, reports, and local configuration out of tracked source.
  Link a consented/redacted evidence location; retain failures as well as passes.
- No new quality threshold is justified by synthetic tests alone. Record false
  positives/negatives and the reviewer decision before tuning thresholds.

## Current contract and known gaps

- **Verified CFR only.** The wizard accepts at most 1080p (either orientation)
  and 30 minutes. It CPU-decodes the complete first video stream to verify the
  renderer's clock before job creation; shared restoration entry points check
  before GPU work or output replacement. VFR, nonuniform or unverifiable timing
  is explicitly unsupported and must be refused early. No automatic CFR
  normalization or full source-timestamp-preserving VFR support is claimed.
- Restoration delivery and previews are H.264 only:
  `h264_nvenc` (`p5`, CQ 17, level 4.1 or higher, 145x49 to 4096x4096) when
  a cached test encode works, else `libx264` `faster` CRF 16; the marker records the encoder and `version: 3`, and an
  output from an earlier version or the other encoder is re-encoded once
  before a revise splices into it. The legacy CLI export is H.264 too
  (`libx264` by default); nothing writes HEVC or AV1. Doctor requires
  `libx264` only and reports whether NVENC and CUDA decoding run.
- Synthetic tests force `libx264` and fake the GPU block with an FFmpeg
  nearest-neighbour 2x, so CI establishes streaming mechanics only; set
  `SUPER_PROCESSOR_TEST_ENCODER=h264_nvenc` to run the suite against NVENC
  locally. The GPU colour/blend/strength-map math is checked with torch on the
  CPU where CUDA is absent, and against the FFmpeg blend path only on CUDA
  with the real weights.
- Revisions must retain each stored span's look, protected regions, and
  `use_upscale` choice, including after reopening. Mixed-recipe or scale-changing
  revisions may re-render the full source rather than stream-copy untouched spans.
- Protection is per content kind: a face box does not remove the fallback cap
  for unboxed hands or text. Boxes are not frame-by-frame tracking, and protection
  is not a guarantee against identity/anatomy/text damage.
- **Seam coverage.** The checker covers coalesced parts and actual expanded
  revision splice edges. Each part streams through SPAN and the delivery
  encoder in one pass, and out-of-memory retries shrink only the GPU batch
  (down to one frame), so neither adds joins; the chunked upscale, its
  chunk-halving ladder, and the managed-memory fallback are gone. Real
  memory-pressure tests remain pending. Do not claim all seams are covered.
- Shimmer `1.6`, seam `2.5`, and region strength caps are existing heuristics,
  not newly calibrated real-footage guarantees. Seam warnings are advisory.
- No telemetry implementation is present. Model-generated shell commands are
  prohibited, not a deferred mode.

## Representative footage matrix

Start with short 30–60 second CFR clips that isolate each issue, then run mixed
and long cases. Compare moving before/after footage, not just stills. All rows
below are **PENDING** real acceptance; regression coverage is not a substitute.

| Case | Exercise | Acceptance / evidence to record | Status |
|---|---|---|---|
| Faces | Close-up, head turn, occlusion, multiple skin tones; lower-strength redo | No newly visible identity drift, waxy skin, halos, or crawling detail; inspect box coverage and fallback when face boxes are absent | PENDING |
| Hands | Fingers moving across objects and frame edges | No newly invented anatomy/texture; valid hand coverage or fallback cap retained | PENDING |
| Text | Signs, subtitles, small moving lettering; face and text together with text box missing | Legibility not degraded or invented; per-kind missing-box fallback remains even when a face box exists | PENDING |
| Fine patterns | Fabric, fences, brick, foliage, already-sharp detail | No objectionable moiré or crawling; compare SPAN on/off and shimmer reports against human judgment | PENDING |
| Motion | Pan, fast subject, hard cut, low-motion hold; spans longer than one streaming block (about 128 MiB of input frames) | No new discontinuity, repeated/dropped frames, or visible flash at block edges; inspect part joins and advisory seam report | PENDING |
| SPAN on SD | 480p/576p sources, soft and blocky, at 2x; strengths 0.3, 0.5 and 0.7 | Detail added without waxy skin, halos or crawl; shimmer near the bake-off's (+0.07 over the source at 0.5); record upscale fps and peak VRAM | PENDING |
| SPAN on 1080p | 1920x1080 and 1440x1080 sources at 2x (3840x2160 and 2880x2160) | Renders without OOM; record end-to-end fps (1500 frames of 1440x1080 through SpanEngine: 26.8 fps on an RTX 5090, 0.054 CPU-seconds per frame, CPU 70 C to 76 C over the minute; the GPU-path prototype: 26.2 fps, 0.044; the previous CPU-blend stream: 22.2 fps, 0.43), peak VRAM (about 4-9 GB), CPU use and temperature, the encoder in the marker, and that no scratch files are written | PENDING |
| Encoders and decode | The same clip with `h264_nvenc` and with `libx264` forced; a source codec NVDEC cannot decode; a look with clean and finish filters; a revise into an output made with the other encoder | Both encoders: H.264 High, IDR every 1 s, no B-frames, source frame count; software decode fallback works and the clean filter still applies; the revise re-encodes the output once, then splices | PENDING |
| Noise / damage | Low light, compression blocks, soft SD material, clean control | Useful cleanup without waxy detail/halos or invented texture; record recipe, strength, and false shimmer/seam warnings | PENDING |
| Mixed pass-through | Alternate restored and `use_upscale: false` spans, including equal-looking neighboring recipes | Pass-through avoids SPAN, retains look and common dimensions; toggle boundaries and coalesced parts match report candidates | PENDING |
| Repeated revisions | Within one span, across mixed spans, scale change, then reopen and revise again | Preserve look/regions/toggles; use original source rather than stacking detail; expanded splice edges checked; failed revise retains previous output | PENDING |
| Audio | Speech/lip sync and sharp audio/visual cue near start, joins, revised range, and end | Record source/output frame counts, durations and cue offsets; no introduced drift, loss, click, or truncation after render and repeated revisions | PENDING |
| Long / 1080p | Landscape and portrait at the supported limit, including a near-30-minute source | Complete timing preflight, preview, render, and revise (including a revise of an output from 3.7.0 or earlier, which is re-encoded into the new delivery settings before the splice); record wall time (about 23 minutes for 20 minutes of 1080p with the previous CPU-blend stream; the GPU path should be faster), peak RAM/VRAM, disk use, responsiveness, counts, and audio alignment | PENDING |
| Memory pressure | 2x of a 1080p source with another program holding most of the free VRAM | Smaller-batch retries, then a clear `GpuOutOfMemory` error naming other GPU programs; correct output dimensions/counts when it fits; record cancellation, output preservation, and that no extra joins appear | PENDING |
| Saved 4x plans | Reopen a job whose `upscale_plan.json` was saved at scale 4, with and without its 4x `output.mp4`, and with a 4x Gemini choice in `segments.json` | Loads and renders at 2x; Ready offers only 1 and 2; Segments shows no 4x choice; Result says 4× while the 4x file is on disk, and a result note re-renders the whole file at 2x | PENDING |
| Unsupported timing / limits | True VFR, misleading rate metadata, missing/nonmonotonic timestamps, >1080p, >30 minutes | Clear early refusal before job creation/GPU work/output replacement; no silent normalization; supported rational-rate CFR control accepted | PENDING manual confirmation; synthetic regressions only |

For audio/frame comparisons, record measured offsets and the acceptance tolerance
chosen before the run; do not silently relax it after seeing a failure. If a case
cannot run on available hardware, record the limitation rather than marking it
passed. Prior short GPU experiments are not completion of this matrix.

## Live Gemini walkthrough

Follow the [current wizard dogfood flow](ui-wizard.md#live-dogfood-real-gemini):
setup → file → segment proposal/review/note → per-segment analysis, protection,
preview/check/redo/approval → ready → render → result revise → done. Also test
restart-with-chat and explicit `review JOB_ID` resumption with another review
window open. Record the selected Gemini model, sanitized responses, failures,
fallback behavior, quota/rate-limit recovery, and whether preview judgments
agree with moving footage. The separate `gemini_live` API tests do not prove
these visual outcomes.

**Evidence: PENDING** — model, run date, exact commit, job/artifact references,
reviewer, and outcome have not been recorded for final acceptance.

## Browser acceptance (both interfaces)

Run the same core flow in the server-rendered wizard and the React shell. Record
browser/OS/version rather than assuming one browser represents all platforms.
For React, exercise local Vite dev and production preview proxies and an allowed
direct backend connection using the actual printed ephemeral port.

- Upload and native file pick; refusal cleanup and readable errors.
- Segment stills and before/after clips load; output playback and seeking work
  with range responses, including after a revise without stale cached media.
- Progress subscription detaches/re-attaches without restarting work; polling
  recovers from transient errors, exhausted retries, and reconnecting to a new
  backend port/token. Retry connection must not duplicate work.
- Allowed-origin upload preflight succeeds; disallowed origins/invalid tokens
  fail. HTTPS Pages is not assumed to bypass mixed-content/local-network rules.
- Explicit job resumption remains on the selected job; a new job leaves the old
  one intact. Cancellation terminates work within the bounded worker behavior
  and a failed revise leaves the last valid output usable.
- Check keyboard navigation, focus/error presentation, and accessible progress.
  Node regressions/builds are not browser or accessibility acceptance.

**Evidence: PENDING** — browser/OS matrix, exact commit, run date, screenshots or
redacted logs, accessibility results, and reviewer outcomes remain unrecorded.
No real-browser pass is claimed.

## Offline, installation, and final-tree gates

Use [CONTRIBUTING.md](../CONTRIBUTING.md) for exact routine commands: Ruff format
and lint, strict mypy, full `pytest -m 'not gemini_live'` with configured coverage,
frontend `npm test` and `npm run build`, source/wheel build, and `twine check`.
Build the frontend from a fresh unpacked sdist to verify `web/index.html` and
tracked sources survive packaging. Install a fresh artifact and run self-test
and doctor, checking the required `libx264` encoder. Keep live API checks
separate and explicit.

Local validation of the source/test tree now committed as `c004fdc` (subsequent
documentation edits do not change behavior), on Linux / CPython 3.14.8:

```bash
.venv/bin/python -m pytest -m 'not gemini_live'
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
# From web/:
npm test
npm run build
```

Results: **587 passed, 1 skipped, 2 live tests deselected**, **89.42% coverage**,
**160.67 seconds**; Ruff/format and mypy passed (109 checked files). All **17**
frontend regressions and the production build passed with Node 26.12.1 locally.
CI uses Node 22; its remote results must be checked separately. The CI packaging
job now depends on frontend tests/build.

CUDA discovery reports one available device; no real GPU inference or visual
acceptance was run. Playwright Chromium is absent and no browser was downloaded.
Live Gemini calls and private media uploads were not performed locally. These
capability checks do not complete any real-footage or browser acceptance row.

### Fresh artifact verification

Built from `c004fdc2a95c16a152a527964b093583eef6e7ef` plus the then-staged
README/design/acceptance documentation. Later evidence-only documentation edits
are not included in these artifacts. Packaging used CPython 3.13 in an isolated
build environment; no release version was assigned.

- `python -m build --no-isolation --outdir .dogfood/packaging/final-dist`:
  source and wheel built; `python -m twine check` passed for both.
- Archive inspection confirmed `web/index.html` and tracked frontend inputs in
  the sdist; generated C/annotation pages and private/untracked content excluded.
- Fresh isolated wheel installation passed `super-processor self-test
  --sample-size 1000` and `super-processor doctor --json` (exit 0, `ok: true`,
  both required encoders present). No checkout import was used.
- `npm ci --offline` and `npm run build` passed inside the unpacked sdist.
  npm warned about a blocked esbuild postinstall, but the actual build succeeded.
- Manifest regression passed with build tools installed and skipped without
  setuptools, as intended for wheel-only environments. Running that skipped
  module alone gives pytest exit 5 (no runnable tests); it is not a test pass.

Local development artifacts (ignored under `.dogfood/packaging/final-dist/`):

| Artifact | SHA256 |
|---|---|
| `super_processor-3.6.4.dev9+gc004fdc2a.d20261029.tar.gz` | `d7b30d5e4157131540628de8b76736b9b87b64aca70444b563d0610c8202f4fd` |
| `super_processor-3.6.4.dev9+gc004fdc2a.d20261029-cp313-cp313-linux_x86_64.whl` | `4c32ca517d279ecfb9934d63f37b423d5cc4a859e1cb4742ba05e8d59cd55e2f` |

Build/install/doctor and frontend logs plus `SHA256SUMS` are kept locally under
`.dogfood/packaging/final-validation/`. Release artifacts must be rebuilt and
verified from the approved tagged release; these local artifacts are not a
multi-platform release gate pass.

## Final release evidence record

| Field | Evidence |
|---|---|
| Tested source/test commit | `c004fdc`; later changes are documentation only; local `.claude/` excluded |
| Full Python suite, coverage, duration; Ruff and mypy | 587 passed, 1 skipped, 2 deselected; 89.42%; 160.67 s; Ruff/format/mypy passed |
| Frontend tests/build and frontend CI run URL | 17 passed and production build passed locally; remote CI URL/outcome PENDING |
| Supported-platform CI run URLs / outcomes | PENDING |
| Fresh local sdist/wheel, metadata, install/self-test/doctor, unpacked-web build | PASSED for `c004fdc` plus staged documentation; CPython 3.13 wheel, isolated install and offline unpacked-web build; these are development validation artifacts, not release artifacts |
| GPU model, VRAM, driver/CUDA/PyTorch/spandrel, SPAN weights sha256 (expected `8b166c75831ea7f694d9058ee9c8df8148af8cc1d2b57e69e6581b15cab572f7`) | PENDING |
| Footage matrix, timings, peak RAM/VRAM, audio/frame comparisons | PENDING |
| Live Gemini model, consented evidence, visual review | PENDING |
| Browser/OS/version matrix and accessibility review | PENDING |
| Seam coverage disposition / threshold calibration decision | PENDING |
| Known-limitations review and explicit release approval | PENDING |
| Release commit, version/tag, workflow/artifact URLs | PENDING — no tag or release created |

Do not create a tag or public release automatically when documentation or
implementation commits are pushed. Only an explicitly approved release follows
the gated procedure in the release plan.
