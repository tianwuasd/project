# Full assessment and automatic cooperative execution implementation plan

> For agentic workers: execute task by task with the subagent-driven-development workflow, task-scoped review and final branch review. Preserve existing drafts and unrelated work.

Goal: implement the user's independent real-data assessment, automatic resource selection, cooperative training/rejoining and simplified formal workflow.

Spec: ../specs/2026-10-27-independent-stages.md is the latest binding menu/task requirement. ../specs/2026-10-26-cooperative-training-design.md remains engineering background, superseded by later user requirements below.

User goal amendment2026-10-27: main-menu prepare/train/predict are independent boards with explicit upstream selection; combine resume/rerun into restart; add scoped deletion; fresh catalog only; preserve update/network/upload/status/GPU/process/log/result utilities. Do not scan old jobs.

## Global constraints

- User approved implementation with 250 CT epochs, 250 train and 50 validation iterations per epoch unchanged. User subsequently explicitly permits automatic global batch growth provided the sampling change appears in preview/logs. Preserve source split, labels and target exclusion; record reference and effective batch/sample counts.
- Latest user requires a complete isolated real-data assessment, default Wholeheart, choosing only model and dataset; formal task pages expose only prepare/train/predict with automated necessary prerequisites.
- Automatic GPU selection uses the intersection of actual server devices, inherited allocation and idle devices; dynamic server-wide upper bound, not fixed3/5; one-second shared polling; one qualified card may start. New cards can join training only through a verified complete-checkpoint regroup; prediction uses a dynamic case queue. Keep acquired GPUs through selected train/predict GPU phases, release on completion/failure/interruption. Never kill external processes.
- Every resource decision must save actual observations, candidates, timing distributions, missing evidence and selected reasons. Assessment parameters have identity/TTL checks before reuse; formal startup performs lightweight integrity/capacity checks, not redundant tuning when compatible.
- No data, credentials or private server logs in code publication. Source D:/projects/vista3d_local_expert_refinement has another active research change: work only in this publication checkout.
- User's latest instruction defers all skill changes/publication; retain prior skill drafts and exclude skill from this goal's release requirements until separately requested.
- Server/real GPU acceptance cannot be claimed from Windows, stubs or CPU tests. Current server information is historical/user-transcript evidence and must be labeled.

## Tasks

### Task 1: deterministic bounded batch pool

Files: create05_src/batch_pool.py, tests/test_batch_pool.py. No edits to shared trainer/queue/panel yet.

Interface: EpochBatchIterator(loader, *, workers=0, seed=0, epoch=0, rank=0, stage='train', prefetch=2, timeout_seconds=180, pin_memory=False). Implement __iter__/__next__, set_epoch(epoch, start_index=0), state_dict(), load_state_dict(state), close(), _finish alias. Loader is a picklable iterator returning an existing batch object/dict. A module-level deterministic seed function derives from seed/epoch/rank/stage/sequence with a stable hash; independently executing a sequence must give the same random batch in serial and spawn workers. Preserve parent RNG around serial data generation. Seed Python and available NumPy/Torch CPU RNGs; never initialize CUDA in a worker. Reseed loader-private RNG through an explicit optional reseed/reset hook where needed; document remaining loader assumptions.

Use spawn, bounded in-flight batches, deterministic ordered consumption regardless of completion order, bounded waits, useful child traceback/PID/stage/sequence on failure and cleanup of all owned children. On POSIX inherit the existing workflow lock FD using multiprocessing.reduction.DupFd separately for each child; register actual worker PID identity through the existing task_worker helper when managed. Do not explicitly unlock shared locks. Do not create new process sessions for data children; keep parent workflow process-group ownership. Pin batches only when explicitly requested in the main process.

Tests: real spawn toy loaders, deliberately out-of-order completion, same seeded output with0/2workers, exact next-index state roundtrip/epoch reset, failed worker and timeout cleanup, finite prefetch, serial parent RNG unchanged. Use CPU toy data, no real cases/CUDA/network. TDD red/green evidence. Commit only these two files and report concerns explicitly.

### Task 2: resource contracts and tuning

Add typed/validated stdlib policy and observations. Remove all fixed5-card bounds across UI/request/queue/evaluation. Record inherited allocation, free cards, CPU/RAM/SHM/I-O/NUMA evidence and missing capabilities. Build bounded CPU/batch/world candidates under the protected formal budget; select by same-budget end-to-end timing, not samples/s alone. Persist receipts and TTL identity; update existing preflight to measure >=10 steady steps with phase timing and candidate failure/timeout handling.

### Task 3: cooperative runner and recovery

Add same-experiment distributed launcher using fixed-version nnU-Net APIs and per-rank logs. Validate global batch partitioning, loss/gradient reductions and full rank state. Persist complete checkpoint commits atomically. Support controlled complete-epoch regroup for additional eligible GPUs without changing formal budget; record topology history and deterministic data position. Add process/FD inheritance, cancellation and GPU hold lifecycle. Verify real CPU2-rank gradient/checkpoint logic plus target-CUDA acceptance hook.

Task3a implements the reusable bounded launcher first, separately from the application integration. Create05_src/ddp_runtime.py and tests/test_ddp_runtime.py (a small test fixture module may be added if required for spawn imports). Interface: launch_ranks(target, arguments, gpu_uuids, log_dir, *, backend='nccl', timeout_seconds=600, init_timeout_seconds=60, lock_fd=None) -> list[dict]. Target is an importable 'module:function'; worker invokes function(rank,world_size,arguments) after init. Use fresh spawn processes; GPU backend sets full group CUDA_VISIBLE_DEVICES and set_device(rank), CPU gloo tests never initialize CUDA. Keep each rank's stdout/stderr, traceback and success result separate under a unique invocation ID; parent returns only after all ranks exit0 and matching result records exist. Parent detects failure/timeouts/cancel, cleans its rank processes, preserves lock FD until actual exit and restores signal handlers. Use existing managed child PID registry. Parent rank launcher must not initialize CUDA. Preserve selected UUID-to-rank map. No torchrun CLI wrappers that fail to pass locks to children.

Additional interfaces: capture_rank_state(device_rank=None) and restore_rank_state(state,device_rank=None) capture Python/NumPy/TorchCPU RNG and ONLY the current rank's CUDA RNG if requested (not all GPUs). gather_rank_states(local_state) returns all rank states using the initialized group. Exact model/checkpoint lifecycle remains root's application integration, not this isolated helper.

Task3a tests use actual CPU Gloo2rank SGD: compare aggregated gradients/weights to same-global-batch single-process reference, verify every rank's RNG state roundtrip and toy optimizer checkpoint/resumed step, assert CUDA uninitialized. Cover worker failure, missing/partial result, timeout, cancellation, reusable fresh launches and process cleanup. Linux DupFd/lock testing may be conditionally skipped on Windows but must be explicitly reported. Do not run real data/GPU/network-outbound or change trainer/queue/panel. Commit only task-owned helper/tests/fixture.

### Task 4: full real-data assessment

Add independent assessment job kind, unique assessment/results/workspace, few real source cases, explicit limited training budget and full-size patch; run environment, metadata/data checks, preprocessing, tuning, actual train/validation, checkpoint load/resumed update, prediction, metrics/report and failure diagnostics. Isolate all weights/reports from formal tasks; report coverage and untested GPU topologies honestly. Never use target cases to tune. Produce compatible recommendations with conservative evidence scope, not a general performance certificate.

### Task 5: simple panel and automatic resource handoff

Expose full assessment + formal prepare/train/predict, model/dataset selection with Wholeheart default, automatic paths/names/GPU pool/CPU config and concise preview. Keep history/log/status/stop/update/navigation. Formal tasks automatically validate/reuse compatible assessment and cache inputs. All checks remain reachable in assessment; required safety/capacity checks remain in actual execution. Maintain old-task read/restore compatibility without rewriting frozen code.

### Task 6: updater cleanup and validation/release

Fix owned stdout/stderr pipe closure and repeated-update resource warnings in actual updater; retain CT-origin evidence, legacy bootstrap, isolated releases and code-only downloads. Run focused/task reviews, full controls, real subprocess and CPU distributed tests, official API CPU checks where available, Bash syntax and no-site-packages panel queries. Produce requirement-to-evidence audit, clearly separate server pending items, publish authorized code only with brief update/assessment commands. Do not update/publish skill.

## Review focus

- New cards cannot silently alter samples/updates or corrupt world-size-specific RNG/optimizer state.
- Idle sampling is not exclusive system allocation; validate ownership and exclude unknown external PIDs.
- Assessment on few cases does not prove largest formal data/shape capacity; preserve maximum-shape/runtime gates and evidence scope.
- Multiprocessing data children and rank processes must retain workflow lock ownership until they actually exit.
- No failed/partial assessment or recommendation can pass as compatible; keep failure evidence and block affected execution appropriately.
