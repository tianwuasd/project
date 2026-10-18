# Cross-fold GPU queue implementation plan

> Execute inline with superpowers:executing-plans; request an independent code review before publication.

**Goal:** Fill idle authorized GPUs with ready models from later folds while preserving each fold's scientific protocol and recovery artifacts.

**Spec:** User request on 2026-10-18: finish `.codex/` first (published as `733f6b4`), then fix idle held GPUs at a fold's tail. Global concurrency remains the selected 1–5 limit; each `fold_i/blood_lstm` depends only on `fold_i/blood2d`. CPU training stays sequential. Existing frozen contracts and completed-fold hashes remain compatible.

**Architecture:** Reuse one dependency-aware process queue for all unfinished fold/stage pairs. Extract the existing single-fold preparation/publication lifecycle into a shared context manager, keeping workers and scientific settings unchanged. Publish a global queue and per-fold projections for existing log/status/resume readers.

**Tech stack:** Existing Python/PyTorch training and standard-library subprocess queue; no new dependencies.

## Tasks

- [x] Back up affected source, docs and skill files before edits.
- [x] Add regression tests for later-fold overlap, same-fold dependencies, total concurrency, failure cleanup and stop/restart.
- [x] Extend `imagechd/stage_queue.py` with explicit dependencies and log/result/state callbacks; preserve the single-fold defaults.
- [x] Extract `imagechd/training_session.py` from `train.py` for shared contract validation, stage commits and failure records; reuse GPU authorization callbacks.
- [x] Add `imagechd/cross_fold_queue.py`; prepare and lock unfinished fold outputs, validate completed folds, dispatch one global queue, publish each completed fold through the existing root manifest contract.
- [x] Integrate the GPU branch in `cross_validation.py`, retaining CPU/check behavior and legacy resume compatibility.
- [x] Update read-only server inspection/status to show all active folds and the global GPU count; keep old serial run records readable.
- [x] Run relevant queue, cross-validation, recovery and server tests; run Linux subprocess checks and independent review; fix supported findings.
- [x] Update usage docs, `.codex/maintenance.md`, version and reusable server skill; validate and publish, then verify mirrors.

## Review focus

- A late-fold failure or interrupt preserves already published complete folds and cleans all live workers.
- Invalid frozen data/config rejects resume before overwriting fold artifacts; completed hashes survive interrupted validation.
- Per-fold `blood2d` is never substituted by another fold's encoder, including after resume.
- Global queue size never becomes per-fold multiplied GPU usage; no worker can escape the candidate pool.
- Queries remain standard-library-only, reject stale pre-resume queues, and stop/resume still discovers root and nested workers.

No server restart or real GPU benchmark is implied by local software tests. Deliver update/resume instructions after publication.
