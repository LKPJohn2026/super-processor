# Changelog

All notable changes to Super Processor are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
