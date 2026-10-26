# Training timing and preflight sampling plan

Goal: diagnose the reported 3090 slowdown without changing the training budget or scientific inputs.

Design: retain the existing nnU-Net training loop, losses, augmentation, sampling and checkpoints. Measure batch acquisition separately from the training/validation step. The latter includes host-to-device transfer and GPU work; it is not a CUDA-kernel-only measurement. Observe existing iterations exactly once. Formal timing adds no per-step CUDA synchronization; the pinned parent already returns CPU loss values. Smoke timing retains its existing synchronization boundaries.

Files: add `05_src/training_timing.py` and focused tests; adapt `project_trainers.py`, `smoke_train.py` and `preflight_gate.py`.

Constraints: keep 250 epochs, 250 train and 50 validation iterations, patch/batch, seed, augmentation and optimizer unchanged. Old jobs retain frozen code. No deployment, stopping or automatic resumption of server tasks. Timing failures must not discard a checkpoint or fail an otherwise successful training run.

- [ ] Test timing with fake clocks, failed steps, unchanged iterator consumption and parent cleanup.
- [ ] Add per-epoch batch-wait/step wall timing, periodic log summaries and a local JSONL record.
- [ ] Use 12 existing preflight train steps (2 warmup + 10 measured), record sample counts and separated phases, and retain checkpoint recovery verification.
- [ ] Run focused tests, the existing control suite and real CPU checkpoint API regression; document that server performance remains unmeasured.

User evidence: task `20261026_194953_c3a989d12d`, RTX3090, patch160x192x192, batch2, reported one-sample train8.0785s, two-sample validation0.3033s, allocated peak12.7807GiB, threads2, augmentation_workers0. This does not establish a causal performance diagnosis.
