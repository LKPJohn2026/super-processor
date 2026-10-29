# Reliability hardening and release plan

Baseline: `v3.6.3`, plus explicit job-resumption fix `e7ac3ef`.
This is a reliability train, not a new wizard or an expansion into deferred features.

## Commit and push discipline

Each numbered change is reviewed and tested, committed on the current branch,
and pushed to `origin/main` before implementation of the next change starts.
Stage only files belonging to that change; leave local configuration and footage
untracked. If validation or pushing fails, stop that sequence and resolve or
record the blocker. Do not amend published commits or force-push.

## Ordered work

1. **Frame integrity.** Remove unsafe assumptions about variable-frame-rate (VFR)
   footage. Full support must preserve decoded frame identity, presentation
   timestamps, duration, and audio alignment through model inference, chunks,
   previews, and revisions. CFR normalization is not equivalent. An explicit,
   early refusal of unsupported timing is an acceptable interim safety fix, but
   must be documented as a limitation, not claimed as VFR support.
2. **User choices and content protection.** Preserve pass-through choices during
   result revisions; retain fallback protection for unboxed content kinds.
3. **Quality checks.** Derive seam candidates from the same coalesced parts and
   chunk joins as the renderer. Synthetic tests establish mechanics; real footage
   must calibrate shimmer, seam, and region thresholds.
4. **Frontend integration.** Complete media proxies and upload CORS, document the
   backend port, and make polling/subscriptions recoverable. Add automated
   frontend regression coverage without requiring a paid API or GPU.
5. **Operations and installation.** Bound cancellation, correct progress units,
   check required wizard encoders, make live tests explicitly opt-in in routine
   instructions, and verify the web entry point survives source packaging.
6. **Acceptance and documentation.** Reconcile the design contract and record
   repeatable GPU, Gemini, browser, and packaging checks. Unavailable checks stay
   explicitly pending; never infer quality from a fake engine.

Each item may use more than one focused commit, with a push between commits.

## Validation gates

- Focused regression tests for every behavior change.
- Ruff lint and format, strict mypy, and the full non-live Python suite with its
  configured coverage threshold. Record timeouts/failures, not just passing subsets.
- Frontend type-check/build and automated API/progress/media regressions.
- Build sdist and wheel, run `twine check`, and build the React app from an
  unpacked sdist if it ships there. CI must pass on supported platforms.
- Real footage: faces, hands, text, patterns, motion, low light, compression,
  mixed pass-through/restored segments, repeated revisions, audio synchronization,
  long/1080p inputs, 4x and memory-pressure recovery. Record timing, memory, visible
  failures and any threshold decisions. Do not transmit private footage without
  explicit consent.
- Live Gemini visual segmentation, boxing, recipe and preview checks are separate
  opt-in tests; they may consume quota and transmit sampled images.
- A live browser walkthrough of both interfaces includes upload, playback/seeking,
  progress, transient failure recovery, permitted origins, and job resumption.

## Tag and release procedure

Proposed target: **v3.7.0** (subject to the final scope and compatibility review).
No tag or release is created merely because implementation commits are pushed.

1. Resolve or explicitly scope every open gate below; review the final diff and CI.
2. Prepare a separate release commit: move Unreleased notes to the chosen version
   and date, include known limitations and actual validation evidence. Push it.
3. Verify `origin/main` points to the exact tested release commit and the chosen
   tag does not exist. Do not edit `_version.py`; setuptools-scm derives it.
4. With release approval, create a signed annotated tag where signing is available
   (otherwise use an explicitly approved unsigned annotated tag) on that exact
   commit, then push that single tag. Never move an existing published tag.
5. Tag pushes matching `v*` trigger `.github/workflows/release.yml`: source build,
   platform wheel builds/tests, artifact validation, then a public GitHub Release.
   PyPI publication additionally requires `PYPI_PUBLISH=true` and the configured
   trusted-publisher environment. Check that setting before pushing a release tag.
6. Verify workflow success, attached artifacts, install/self-test from a released
   artifact, and release notes. A failed release is repaired with a new commit and
   version, not a retag.

## Open gates / known limitations

- Full source-timestamp-preserving VFR support is not established. If this train
  ships an early timing refusal instead, release notes must state that limitation.
- Real-footage quality thresholds and real GPU/Gemini/browser acceptance need
  recorded evidence; earlier trailer experiments are useful but not a full matrix.
- The wizard uses H.264 delivery; HEVC/hardware selection belongs to legacy CLI
  paths unless a separate product decision expands scope.
- Local LLMs, desktop shells, cloud workers, generative features and multi-track
  editing remain out of scope. Model-generated shell commands remain prohibited.
