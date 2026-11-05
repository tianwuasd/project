# STU-Net-S multi-GPU implementation plan

**Goal:** Apply V2 multi-GPU execution while retaining the fixed scientific workload.

**Architecture:** A CPU-only supervisor owns the run lock, immutable code snapshot and rank processes. Ranks use the existing model, sampler, checkpoint primitives and bounded CPU supply, with distributed sufficient-statistic loss and a shared resource plan. Reconfiguration happens only after a complete epoch checkpoint.

**Tech Stack:** Python, PyTorch DDP/NCCL; CPU Gloo for deterministic synthetic numerical checks.

**Spec:** ../specs/2026-11-05-stunet-multigpu.md

## Global constraints

Keep global B4, patch128³, 1000 epochs, 250/50 attempts, lr .001 and existing partial-label roles by default. No hidden batch/patch changes, old-signature bypass, external PID termination or server job launch. Keep preprocessing main unchanged.

## Review focus

Unequal valid voxel counts; all-ignore and all-partial ranks; unconsumed prefetch; non-finite gradient on one rank; changed membership during a partial epoch; whole-group versus per-rank RAM; failed child cleanup.

## Tasks

- [x] Implement differentiable global loss reductions and CPU Gloo gradient/update comparisons for 1/2/3/4 ranks, ignore8, missing MYO and deep supervision.
- [x] Add global-slot deterministic patch sampling, whole-group supply admission, and layout/resource tests.
- [x] Add rank runtime: DDP, overflow consensus, per-rank RNG checkpoint/restore, full-size admission, synchronized safe points and complete-epoch performance records.
- [x] Add supervisor/CLI: visible idle GPU selection, bounded waits, epoch-only expansion, regression rollback, bounded pressure recovery and owned-child cleanup.
- [x] Verify frozen code and signatures, update help/docs, run existing regression tests plus targeted distributed/replay/real local CUDA checks. Report Linux/NCCL and real throughput limits accurately.
- [x] Review the final diff, regenerate the code manifest/package and record local acceptance. Synchronization is restricted to the tested development branch; its exact commit/remote receipt is retained with the private acceptance record.

## Local acceptance

88 unit/regression checks passed. Standalone real CPU/Gloo DDP passed for global4 layouts [4]/[2,2]/[2,1,1]/[1,1,1,1], with complete/mixed/partial/all-ignore supervision and both Dice reductions, values/gradients/real SGD updates checked at 1e-10. One-rank overflow propagation, scaler growth reset and per-rank RNG/update replay passed. Actual nnU-Net augmentation reconstructs identical global patches across layouts.

Local CUDA acceptance used three real prepared cases, 64³/B1, 2 epochs × 8 attempts plus patch validation. Continuous workers1 versus paused-at6 workers1/resumed workers0 matched all 130 model tensors, 128 optimizer states, scaler, RNG, cursors and scientific signature exactly; 16 attempts, 12 updates, 4 normal AMP overflows. These are workflow checks, not segmentation accuracy results. Linux/NCCL, real multi-CUDA group churn/pressure recovery and formal128³/B4 capacity/throughput remain server acceptance work; policy unit tests do not replace it.
