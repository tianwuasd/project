# Capped cooperative batch and usable CPU supply

User instruction: maximum global batch 8. Additional GPUs split the same capped
batch instead of increasing it to 10/12. Fix workers that cannot receive work
because their outstanding-batch queue is smaller than the process pool.

Scope: new tasks only; keep 250 epochs, 250 training attempts, 50 validation
iterations, native case/patch sampling, current full-source/held-out-center
protocol, CPU/RAM80% and VRAM reserve. Preserve old frozen tasks/checkpoints.
No promise of linear speedup or bit-identical random streams across membership.

- [x] Add pure capped layouts: 4 cards -> 2/2/2/2, 6 -> 2/2/1/1/1/1,
  8 -> 1 each; capacity may lower batch, never silently exceed 8.
- [x] Integrate layouts into capacity, actual rank arguments, per-rank host
  estimates, checkpoint-boundary membership changes, receipts and panel output.
- [x] Preserve sample-weighted CE and global batch Dice for unequal ranks;
  reduce per-rank sufficient statistics before gathering. Verify actual CPU
  Gloo gradients against an unsplit batch of 8, including deep supervision.
- [x] Jointly reserve worker-private memory and at least one outstanding slot
  per worker in both resident pools. Expansion must preserve that invariant.
- [x] Add the supplied six-rank RAM/payload regression; prove actual workers
  produce batches, bounded aggregate CPU/RAM, and unchanged replay behavior.
- [x] Run focused, full control, pinned API and distributed gradient checks.
  Record limits; target Linux CUDA/NCCL/real CT speed remains unverified.

Alternative rejected: use only 4 of 6 cards to keep batches equal; that would
not satisfy the requested cooperative use of additional compatible cards.
Unequal rank batches require loss weighting, not just changing a batch integer.
