# Reliability hardening and release plan

Baseline: `v3.6.3`, followed by explicit job-resumption fix `e7ac3ef`.
This is a reliability train, not a new wizard or an expansion into deferred
features. Proposed target: **v3.7.0**, pending acceptance and release approval.
No tag or release has been created for this train.

## Commit and push discipline

Review and test each focused change, commit it on the current branch, and push
to `origin/main` before implementing the next change. Stage only that change's
files; leave local configuration and footage untracked. If validation or pushing
fails, resolve or record the blocker before continuing. Do not amend published
commits, force-push, or move published tags.

## Delivered commits

These commits were pushed sequentially. Delivery does not satisfy every release
gate; the [acceptance record](acceptance.md) distinguishes the evidence.

| Order | Commit | Delivered scope |
|---|---|---|
| 0 | `e7ac3ef` | Explicit wizard job resumption stays on the selected job |
| 1 | `8d9ef27` | Ordered hardening plan and release gates |
| 2 | `4535e53` | Refuse unsupported timing before restoration |
| 3 | `5aeb5d4` | Preserve per-shot choices and per-kind missing-region protection |
| 4 | `07dbfbe` | Check seams at retained render and revision joins |
| 5 | `58fd92b` | Frontend media, subscriptions, token isolation, and recovery |
| 6 | `0c84ee0` | Worker cancellation/progress, installation checks, and packaging |
| 7 | `8b0d878` | Persist complete revision context and portable timing preflight |
| 8 | `c004fdc` | Gate packaging on frontend tests/build; optional dev-only manifest test |

Design reconciliation and the final evidence record are documentation-only
follow-up work after these commits. Their commit is identifiable in Git history
without embedding a self-referential commit hash here.

## Ordered milestones and disposition

1. **Frame integrity — safety guard delivered.** Admission verifies the complete
   decoded timeline against the renderer's rational CFR clock. Unsupported VFR,
   nonuniform, and unverifiable timing is refused before job creation/GPU work
   or output replacement. This is **not VFR support** or automatic normalization.
   Full VFR support needs source-frame/timestamp mapping through inference,
   chunks, previews, revisions, and audio; it remains future work.
2. **User choices/protection — mechanics delivered.** Revisions preserve look,
   regions, pass-through choices, and the complete previous requested range.
   Missing boxes retain fallback protection per content kind. Protection quality
   still needs real-footage acceptance.
3. **Quality checks — nominal joins corrected.** Candidates share coalesced,
   frame-snapped parts, retained chunk joins, and expanded revision splice edges.
   Actual shorter joins from runtime OOM retries/custom chunks/managed-memory
   fallback are not fully recorded and may be missed. Calibration remains open.
4. **Frontend — offline regressions delivered and wired into CI.** Media/upload
   routing, CORS, backend-port guidance, token isolation, subscriptions, timeouts,
   retries, and explicit recovery are covered. Real-browser E2E is still pending.
5. **Operations/installation — fixes delivered.** Bounded cancellation, correct
   progress units, required encoders, live-test separation, and the packaged web
   entry point have regressions. Fresh-artifact checks and platform CI remain
   distinct from local unit-test evidence.
6. **Acceptance/documentation — contract recorded, live gates open.** Current
   segment/preview flow, verified-CFR limits, wizard H.264 versus CLI HEVC,
   credential precedence, absent telemetry, and prohibited model shell commands
   are documented. Use the [footage matrix](acceptance.md#representative-footage-matrix)
   to collect actual GPU/Gemini/browser evidence, not inferred quality claims.

## Latest local validation

Source/test behavior corresponds to `c004fdc`; subsequent plan/acceptance edits
are documentation only. Commands and environment are recorded in
[acceptance.md](acceptance.md#offline-installation-and-final-tree-gates).

- Full non-live Python suite: **587 passed, 1 skipped, 2 deselected**;
  **89.42% coverage**, **160.67 seconds**, Linux / CPython 3.14.8.
- Ruff lint/format and strict mypy: passed (109 checked source/test files).
- Frontend: **17 regressions passed**, TypeScript and production build passed.
- Fresh local sdist and CPython 3.13 wheel built from `c004fdc` plus staged
  documentation; metadata checks, isolated wheel installation/self-test/doctor,
  and offline frontend build from the unpacked sdist passed. Artifact hashes
  and qualifications are in the acceptance record. They are development builds,
  not the final tagged multi-platform release artifacts.
- CUDA discovery: available, one device. No real restoration/quality acceptance
  was run. Playwright Chromium is absent; no browser E2E pass is claimed.
- Remote supported-platform/frontend CI outcomes and all live acceptance remain
  to be verified before release. No repository-secret or live-API result is
  inferred from a local non-live suite.

## Tag and release procedure

Proposed target: **v3.7.0**, subject to final scope/compatibility review. Pushing
implementation commits is not authorization to create a public release.

1. Complete every gate in the [final evidence record](acceptance.md#final-release-evidence-record)
   or obtain explicit approval for a scoped limitation. Review CI on the exact
   candidate, real-footage results, and the runtime OOM seam gap.
2. With release approval, prepare a separate release commit: move Unreleased
   notes to the chosen version/date, with actual validation and known limitations.
   Push that commit before proceeding to the tag. Do not edit `_version.py`;
   setuptools-scm derives the version.
3. Verify `origin/main` is the exact tested release commit and the chosen tag
   does not exist. Create a signed annotated tag where signing is available;
   otherwise obtain approval for an unsigned annotated tag. Push only that tag.
4. A `v*` tag triggers `.github/workflows/release.yml`: source build, platform
   wheel builds/tests, metadata checks, then a **public GitHub Release**. PyPI
   additionally requires `PYPI_PUBLISH=true` and the trusted-publisher environment.
   Inspect publication settings before tagging; disabling PyPI does not disable
   the GitHub Release.
5. Verify workflow success, attached artifacts, installation/self-test from a
   released artifact, and release notes. Repair a failed release with a new
   commit/version, never by moving an existing tag.

## Remaining gates / explicit limits

- Full timestamp-preserving VFR support is absent; this train is verified-CFR only.
- Real-footage restoration quality, threshold calibration, GPU memory-pressure
  recovery, live Gemini visual decisions, browser/accessibility checks, and
  supported-platform CI need recorded outcomes or an approved scope decision.
- Runtime OOM/fallback joins are not completely covered by seam candidates.
- Wizard delivery is H.264; HEVC/hardware selection stays in legacy CLI scope.
- Telemetry is absent. Local LLMs, desktop shells, cloud workers, generative
  features, and multi-track editing remain out of scope. Model-generated shell
  commands are prohibited, not deferred.
- `v3.7.0` is proposed only; no tag or public release has been created.
