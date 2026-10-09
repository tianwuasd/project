# ImageCHD Primary Workflow Implementation Plan

> For agentic workers: use superpowers:executing-plans inline; retain one independent final code review.

Goal: make the data-supported multi-stage CHD reproduction the default and isolate the old implementation in back.
Architecture: native/reusable prepared data -> six-stage training -> image-only fused prediction -> separate evaluation; shared model and server utilities.
Tech Stack: existing Python 3.11+, PyTorch, NumPy, SciPy, nibabel, PyYAML.
Spec: docs/superpowers/specs/2026-10-09-imagechd-main.md

## Global constraints
- No patient data, caches, weights or credentials in Git.
- Keep preprocessing/training/prediction independent; no target access during prediction.
- Preserve seven native foreground IDs; unknown=255; no fabricated initial-vessel labels or diagnoses.
- Latest zmic44 profile, existing Conda reuse, private CHD runtime, NAS data/results, two threads, allocated idle GPU only.
- Reuse old v1 caches while marking their reduced-resolution origin.

## Review focus
- Unknown label neighborhoods and ROI generation must not turn ignored voxels into supervised background.
- Collection integrity must reject missing/mixed/stale stage checkpoints before inference.
- Interrupted runs must not appear completed or usable as formal training.
- Primary entry points must not import or accidentally invoke archived modules.
- Prediction and evaluation must distinguish original-grid geometry from unverified millimetre calibration.

## Task 1: archive and data/model contract
- [x] Snapshot all tracked source at fc33c8c into back/paper-v1; record provenance.
- [x] Add meaningful failing tests for native/v1 caches, masked labels, blood targets and six-stage config.
- [x] Implement cache compatibility, shared geometry, six-stage config/data/loss modules; retain native shape by default.
- [x] Run targeted tests and save result.

## Task 2: multi-stage training, fused prediction and evaluation
- [x] Add failing end-to-end test, collection integrity and label-free prediction checks.
- [x] Adapt existing training, frozen encoder, four-way fusion and blood refinement to seven labels.
- [x] Add separate ignored-aware Dice evaluation, model collection metadata, incomplete and smoke safeguards.
- [x] Run targeted synthetic and real-cache end-to-end tests.

## Task 3: primary launchers, server and documentation
- [x] Route root/CLI/wizard/server to primary tasks and isolate old modules/tests/config/docs in back.
- [x] Add default-route and independent-dispatch regression tests.
- [x] Rewrite README/workflow/server/data/paper mapping and verification docs for one primary path.
- [x] Run full active suite, archived suite in isolated subprocess, lint/shell checks and review.
- Delivery: commit, update project GitHub subdirectory with removals limited to tracked migrated files, and verify remote commit before reporting completion.