# HU and partial-label preprocessing implementation plan

> For agentic workers: implement sequentially in this session. The user has approved the workflow and requested implementation and a reliable local short test; no additional design approval is pending.

**Goal:** Reproducible source-only nnU-Net preprocessing with explicit partial supervision, recoverable resource scheduling and measured local acceptance.

**Architecture:** Preserve the existing analysis and legacy preparation paths. A new workflow coordinator validates private decisions and frozen roles; isolated numerical jobs stage, plan and preprocess. A training adapter consumes per-case supervision metadata through nnU-Net batch keys.

**Tech stack:** Python 3.11, nnunetv2 2.8.1, PyTorch, nibabel, psutil. Audit and numerical environments remain separate.

**Spec:** User-approved HU/marginal-loss workflow, supplied through private policy and frozen-cohort JSON at runtime. No real policy, identifier, scan or report belongs in this repository.

## Constraints

- Reader-scaled input is assumed HU, scale 1, offset 0; official source-training CTNormalization only.
- Excel dimensions must match exactly; shared local array geometry accepts only identical grids or the documented unit-grid +1 placeholder origin. Never translate voxels.
- Preserve frozen patient roles; explicit alias and exclusions; one deterministic index scan per patient for the main workflow.
- CPU and RAM at most 80%, at least 4 GiB reserve, live admission and one-second memory checks; maximum three recoveries with reduced budgets.
- Every reusable output is bound to request, code and artifact hashes. Only owned processes are terminated.
- Missing classes merge into ambiguous background for marginal CE and merged-space Dice. No unannotated-class Dice score.

## Review focus

Stale completion markers, source file mutation, partial-label batch metadata loss, memory-pressure retries and train/development leakage must fail explicitly or recover without changing results.

## Tasks

- [x] Add `preprocessing/workflow.py`: policy validation, alias/exclusion, frozen roles, index scans, CLI request and owner lock. Test role and hash failures.
- [x] Add `preprocessing/queue.py` and `jobs.py`: isolated jobs, live CPU/RAM/IO admission, stage events, artifact commits, resume, official planning and preprocessing. Test corruption and failure recovery.
- [x] Add `training/marginal.py` and `trainer.py`: differentiable marginal CE/Dice, key-bound metadata, masked validation, deep supervision. Test exact CE reduction, unknown probability redistribution and optimizer update.
- [x] Add reusable `short_test.py`: synthetic official end-to-end, real-case opt-in, equivalence, resume and tamper rejection. Report every assertion and elapsed time.
- [x] Run full relevant test suite and local real-volume serial/adaptive comparison; inspect logs and output classes; record actual limitations and benchmark.
- [x] Update usage, version, dependency-bound resume contract and release manifest; prepare a validated code-only publication. Store the verified publication receipt in private work_root.
