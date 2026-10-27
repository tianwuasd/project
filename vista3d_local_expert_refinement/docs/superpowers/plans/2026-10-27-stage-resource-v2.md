# VISTA stage resource v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved live allocation design in project code, verify it locally, publish the project, then preserve Skill v1 and deliver a complete VISTA v2 template.
**Architecture:** One stage-specific live policy supplies truthful CPU/RAM budgets and decisions. Deterministic persistent loading, bounded transfer, capacity-checked cooperative training and persistent inference consume it; assessment observes the same route without producing reusable resource selections.
**Tech Stack:** Python stdlib control plane; pinned nnU-Net 2.8.1/PyTorch 2.8 CUDA on target; unittest; local CPU API runtime and conditional CUDA tests.
**Spec:** docs/superpowers/specs/2026-10-27-stage-resource-policy-design.md (sections 5–9 and 13; user approved implementation and subsequent Skill update).

## Global Constraints
- Project code and local tests BEFORE Skill modifications. Work only in this dedicated publication checkout; leave the separate research checkout untouched.
- CPU budget floor(0.80 × currently permitted available CPU shares), RAM budget 80% of currently available memory; fresh decisions at prepare, train, predict and CPU-tail entries.
- Retain data/label/fit-cal contracts, patch/spacing/resampler, Dice+CE/deep supervision, SGD/PolyLR, seed0, 250 epochs × 250 training / 50 validation updates. Automatic batch and its sampling changes are approved.
- One direct resource configuration plus bounded capacity/error recovery; no candidate speed ranking or copying assessment UUID/workers/batch into production. Extra resources remain allocated absent real pressure/fault/degradation.
- Respect inherited GPU visibility and external PIDs; all acquired compatible idle GPUs cooperate. Join at full-epoch checkpoint boundaries; release owned processes/keepers before CPU-only tail.
- Preserve proven GitHub code-only fetch/object inventory/offline archive/isolated release, authentication and runtime continuity. Never publish data, secrets, environments, weights as code.
- Baseline local success is not target Linux CUDA/NCCL or real-data performance proof. All decisions, exclusions, missing evidence and fallbacks are logged.
- Follow TDD for behavioral changes, focused tests during iteration and relevant full suite before committing. Update stale tests to the new contract without discarding error/recovery coverage.
- Implementers cannot spawn subagents, publish, update Skill, or alter root project bookkeeping. Root coordinates reviews and publication. No parallel implementation agents.

## Review Focus
- Nested cgroup quota/affinity, zero idle/RAM and one-core hosts must not yield fabricated capacity or invalid thread counts (Task 1).
- Prefetched-but-unconsumed batches across epoch, checkpoint and worker failure must replay exactly and close cleanly (Tasks 2–3).
- More ranks than global budget, compile failures and membership/OOM recovery must preserve scientific/checkpoint contracts (Tasks 3–4).
- Partial prediction failure, model persistence, dynamic GPU ownership and CPU-tail sequencing must not lose cases or retain cards (Task 5).
- Diagnostic archive names/content and publication routes must exclude raw data/weights/secrets while including complete errors and decisions (Tasks 5–7).

### Task 1: Live CPU/RAM facts and direct stage policy
**Files:** Create `05_src/live_resources.py`; modify `05_src/task_resources.py`, `05_src/training_policy.py`; add `tests/test_live_resources.py`, adjust focused resource-policy tests.
**Interfaces:** Produce stdlib `live_resources.snapshot()` dict with permitted/available/budget CPU, memory budgets, affinity/physical/NUMA, cgroup evidence, missing evidence; `case_policy(snapshot, case_count, bytes_per_case, supported_threads=4)` dict; `training_policy(snapshot, rank_count, batch_bytes, history=None)` dict with rank settings main_threads/worker_threads/train_workers/val_workers/prefetch, limits/reasons; `feedback_policy(settings, windows, capacity)` dict. Freeze/document exact field names in report for downstream tasks. Existing public task_resources wrappers remain compatible.
- [ ] Write failing tests for E239→C191, 8-rank aggregate <=191, single ample rank16workers×2threads/main4, validation4×2, memory aggregation incl both resident pools, no SHM gate for pipe transport, nested cgroups/zero resources/physical mapping, 12-case capacity with no8cap, high I/O without zeroing compute, wait two windows expansion and no automatic shrink on no gain.
- [ ] Run focused tests RED; implement pure policy plus injectable fact collection. CPU load must respect permitted CPUs (no whole-host-idle multiplied into tiny affinity where measurable). Record actual uncertainties rather than claiming effective utilization.
- [ ] Replace legacy fixed CPU caps and resource selectors in wrappers using direct policy; keep old receipt/schema readers only for historical viewing. Unknown resources yield explicit bounded fallback, known insufficient resources block rather than overcommit.
- [ ] Run focused GREEN, relevant existing tests; commit and report exact APIs, commands and remaining stage integration assigned later.

### Task 2: Persistent deterministic supply and actual worker threads
**Files:** Modify `05_src/batch_pool.py`, `05_src/nnunet_batches.py`; create `05_src/nnunet_loader_adapter.py` if needed; add/adjust batch-loader tests and local effect probe.
**Interfaces:** Consume Task1 rank allocation. Preserve existing EpochBatchIterator callers/state contract while adding configurable worker_threads, bounded phase steps, persistent reset and consumed-vs-prefetched state. Expose explicit pause/close and applied thread/transport/queue observations for coordinator.
- [ ] Read upstream pinned loader implementation. Test RED that configured threads reach the inner actual loader, persistent worker PIDs survive epoch reset, sequence and resume match sync loading, queues stop at phase remaining, inactive pool stops prefetch, process failure/timeout closes and reports.
- [ ] Implement project adapter without editing site-packages; source/version provenance and lossless scientific operations. Remove hard inner threadpool limit1 only through explicit configured adapter.
- [ ] Implement persistent ordered workers with bounded CPU/IPC and optional reusable shared buffers, owned cleanup and byte-pipe fallback. Count actual memory path, not imagined SHM. Avoid stale epoch output and repeated process startup/case cache loss.
- [ ] Verify real pinned nnU-Net loader API and deterministic checkpoint tests. Produce a short synthetic CPU supply before/after effect record with actual timings/PIDs, no speed claim for real CT. Commit and report interface details.

### Task 3: GPU execution, asynchronous transfer and atomic checkpoints
**Files:** Modify `05_src/project_trainers.py`, `05_src/cooperative_training.py`, `05_src/prepare_nnunet.py`; create focused `05_src/batch_transfer.py` and execution helper if needed; add transfer/execution/checkpoint tests.
**Interfaces:** Consume Task2 iterator/state/pause interface. Produce real rank report with fixed/allocated/reserved peaks, CPU supply/copy/compute/validation/checkpoint timing, applied execution modes; pure capacity helper may be in training_policy. Existing launch/result contracts remain compatible.
- [ ] RED tests: configure_paths does not overwrite approved compile; high-performance vs strict reproducibility applied inside each rank; compile fallback logs reason; deep-supervision weights identical to frozen baseline; capacity uses one reservation margin and initialized optimizer state; unknown/nonpositive slope retains proven batch.
- [ ] Apply AMP plus controlled compile(backend=inductor,mode=default) and cuDNN fixed-shape benchmark, separate compile cost/cache identity. Preserve loss/optimizer/scaler/checkpoint keys; if actual compatibility fails, logged eager fallback rather than pretending compile passed.
- [ ] Add bounded two-slot CUDA stream transfer with pinned background prep, readiness/reuse events and CPU fallback; preserve finite loss/gradient checks. Prefetched CPU/GPU batches cannot advance checkpoint logical consumed position; bound phase length and pause inactive pool.
- [ ] Assemble full model/optimizer/scaler/all-rank RNG/source/consumed state then one atomic save, avoiding parent write+reload+rewrite. Test actual pinned CPU checkpoint/resume and optional unoccupied local CUDA functional path; record conditional gaps honestly. Commit.

### Task 4: Single-config cooperative capacity and independent assessment
**Files:** Modify `05_src/cooperative_control.py`, `05_src/training_policy.py`, `05_src/assessment_run.py`, rank-affinity integration and related tests.
**Interfaces:** Consume Task1 budgets, Task2 pools, Task3 report/capacity. Keep tune entry name if callers require it but change semantics to one direct capacity check. Produce versioned stage decisions/events and performance sufficiency without static recommendations.
- [ ] RED tests: assessment followed by changed CPU/RAM/GPU creates fresh formal decision; old seven-day receipt cannot bypass; no trial ranking path; all ready compatible GPUs retained; 8-rank aggregate CPU and NUMA allowed affinity remain bounded.
- [ ] Replace two-candidate optimization with direct rank/worker/batch capacity calculation, single inferred batch validation and bounded capacity backoff (max3 reductions). Unknown demand uses approved historical A5.04/D0.77 only as estimate. Equal per-rank batch and explicit samples/epoch.
- [ ] Keep one-card immediate start, idle-card discovery and epoch-boundary membership/checkpoint resume. Real OOM stops whole group and resumes last complete checkpoint, logs discarded uncommitted updates; noncapacity errors surface, no broad silent retry.
- [ ] Incorporate slowest-rank rolling feedback >=20steps AND>=30seconds, two windows wait>20% and spare resources expands workers safely; never shrink merely no speed gain. Assessment remains short12train/2val plus recovery/predict, function vs effect statuses separate, no reusable resource recommendation.
- [ ] Run focused coordinator/fleet/assessment/restore suites and relevant full control suite; commit and report.

### Task 5: Prepare, persistent inference, truthful UI and diagnostic export
**Files:** Modify `05_src/server_prepare.py`, `05_src/assessment_prepare.py`, `05_src/evaluation_parallel.py`, `05_src/evaluation_phases.py`, `05_src/task_progress.py`, `05_src/task_queries.py`, `05_src/task_panel.py`, `05_src/panel_publication/upload.py`; create focused diagnostic archive helper/tests.
**Interfaces:** Consume Task1 fresh policies, Task4 decisions. Preserve menu1/2/3/a/r/d and u/c/w/v/l/o etc. Add one simple local diagnostic-export menu entry; publication preview remains distinct.
- [ ] RED tests for 12case budget/no8cap, fit-only fingerprint parallelism, leaf threads (incl Blosc) within budget, persistent one-model-per-GPU across cases, CPU-tail only after GPU workers/keepers exit, failure leaves completed cases reusable.
- [ ] Integrate fresh per-stage/per-natural-dispatch CPU/RAM observations and direct queue dispatch; adjust supported threads without pretending single-thread resampler parallelizes. Record per-phase durations, library thread observations and post-case peak evidence.
- [ ] Persistent per-GPU inference worker loads one compatible fold/model at a time and loops cases, preserving tile .5/Gaussian/mirroring/bestcheckpoint/raw-grid. CPU prep/restore/metrics queues bounded by actual dimensions/memory and CPU, release before CPU-only tail.
- [ ] Add zip diagnostics taskselect→export→path; include decision/snapshot/event/report/rankstderr/error/progress logs; reject symlink/path escape and omit images/weights/env/secrets/caches. v artifact collection includes new diagnostics/optimization/tuning logs while weights remain explicit separate approved collection.
- [ ] Replace misleading startup budget line with current permitted/assigned/actual/missing values and short reasons; assessment reports functional pass separately from effect evidence. Run focused tests including private updater and task creation continuity, commit.

### Task 6: Integration, local verification, release documentation and project publication
**Files:** Tests runner; version; `docs/STAGE_RESOURCE_V2.md`, machine-readable validation; `.codex` records (root owns); integration fixes scoped to Tasks1–5.
**Interfaces:** Final project v2 source commit and validated file inventory for later Skill copy. Panel version0.5.0.
- [ ] Run full stdlib suite once; actual pinned loader/checkpoint/Gloo membership tests, synthetic actual preprocessing, no-network full workflow/control integration, source syntax. Correct concrete failures then rerun affected checks, never label conditional GPU checks passed.
- [ ] Inspect local CUDA availability/processes read-only; only short safe functional probe on unoccupied allowed GPU. Target server Linux/CUDA/NCCL/real data remain acceptance boundary; document exact commands/log paths and expected diagnostic contents.
- [ ] Record tested effect evidence (persistent PIDs/no speculative batches/thread propagation and synthetic throughput), all limits and missing evidence; no promised acceleration factor.
- [ ] Final broad project review and fix important findings. Commit version/docs/state. Check remote main, publish only authorized project changes via proven temporary Windowsproxy7897 route, no force, verify ref. Skill remains unchanged until all project tests/review complete.

### Task 7: Preserve Skill v1 and build/install/publish full VISTA v2 template
**Files:** Separate repo `D:/projects/codex-skill/server-engineering` and installed skill (only after Task6 complete); v1/v2 reference/assets/scripts/tests plus skill index.
**Interfaces:** Consume committed, reviewed VISTA project source and validation manifest from Task6. Keep CT legacy scripts/assets compatible; default new overall server work uses explicit v2 route, historical projects not automatically changed.
- [ ] Read skill-creator/writing-skills instructions; preserve existing8dirty drafts bytes privately before edits. Freeze committed/installed v1 baseline and provenance, separately preserve differences; do not silently bless dirty drafts as v1 or discard them.
- [ ] Produce complete readable VISTA v2 code snapshot, source manifest/archive and standalone scaffold/verify command independent of original local absolute path. Include every runtime/support dependency needed to reproduce panel; exclude data/research PDFs/env/credentials.
- [ ] Document each phase and algorithm, CPU80/RAM80, actual workers/threads, active vs resident pool budgets, NUMA/visibility, automatic batch/sampling, DDP checkpoint joins, transfer and recoveries, test vs formal independent decisions, logs and diagnosis. Link authoritative pinned docs.
- [ ] Document u GitHub→server, w committed code→main, v explicit artifactsSSH+LFS separatebranch, Windows7897/server17891, filtered metadata/explicitblob/offlinearchive/atomicrelease, runtime continuity and old bootstrap, network-vs-auth-vs-promisor distinctions; lessons and regression gates from prior mistakes.
- [ ] TDD template completeness/source hashes/executable scaffold/import/help/control tests, review/pressure scenarios for risky use (only permitted scope). Install verified source copy to installed Skill with authorized filesystem escalation if necessary. Commit/publish only scoped reviewed skill changes, verify refs and byte hashes. Root completes goal only after all requested outcomes and reports actual pending server acceptance.
