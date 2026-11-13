# Bounded segmentation overlap implementation plan

> **For agentic workers:** Native implementation with an independent final correctness review.

**Goal:** Overlap bounded CPU preparation and CPU export with native-CT GPU inference.

**Architecture:** Extend the frozen case queue with a prepare phase and bounded ready cases; retain the existing backend and original-grid label exporter.

**Tech Stack:** Python, gzip, nibabel, UUID-bound PyTorch processes, TotalSegmentator 2.18.0 and pynrrd/slicerio.

**Spec:** tools/coarse_segmentation/docs/overlap-design.md

## Global constraints

- Original HU/mm data; lossless temporary decompression only; no saved resampled CT.
- Explicit GPU pool, one owned inference per GPU, shared-GPU memory admission preserved.
- Bounded preparing/ready cases, total workers, RAM, I/O and conservative shared disk reservations.
- Exactly two approved real CT volumes, zero training updates, test root limited to 30 GiB and deleted afterward.
- Source, tests and generic documentation only in GitHub.

## Review focus

- Invalid prepared receipts and changed source bytes must be rejected before inference.
- Waiting ready cases remain charged against disk capacity.
- CPU stages never initialize CUDA or consume a GPU slot.
- Stop and resume preserve durable source identities, owned processes and case artifacts.
- Higher workers actually permit independent CPU/GPU stages while respecting all caps.

## Tasks

- [ ] Add real lossless preparation, source/cache integrity receipts and interruption cleanup; verify using two small geometry fixtures.
- [ ] Add phase caps, ready bounds, CPU memory estimates, disk accounting and private stage timing events to the actual scheduler.
- [ ] Add CLI controls/defaults and update generic usage documentation without changing model algorithms.
- [ ] Run two approved CT volumes through the real direct runner, verify overlap and final labels, and check resume without new inference.
- [ ] Review, synchronize maintained/public code, publish a checked whitelist and remove the approved test root.
