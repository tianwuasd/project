# Independent terminal validation

Baseline: GitHub main `1bc139a19137af0950d0d45022b9b04bf0e8e8d7`, fetched and confirmed before changes on 2026-10-23. Product version 0.2.0.

Tests use synthetic 8×8×8 CT volumes, mock TotalSegmentator masks and synthetic process owners. Covered: preserved image bytes and native geometry, full mask/source verification, frozen requests/code, input changes while queued, independent rerun directories and lineage, merge-only reuse, cgroup ancestors, one shared thread/memory budget, delayed and partial GPU availability, two simultaneously active mock GPU inference workers, serialized cold cache, CPU-only merge, stop while waiting, persisted error logs, menu cancellation, read-only dependency-free queries, managed release reopening and detached real CPU planning/resume.

Windows: 64 tests collected, 62 passed, 2 POSIX-only cases skipped (2026-10-23; 26.4 seconds). This includes the existing web/CLI regression and real detached CPU planning/resume. Dependency-free help/history, Bash syntax and argument forwarding, Ruff F/I and Git whitespace checks passed. Linux CI results will be recorded after publication. POSIX-specific cases verify inherited lock lifetime after the owner exits and stopping a registered orphan model without affecting an unrelated process.

Independent review prompted fixes for inherited-lock release, orphan model stopping, explicit CPU/CUDA installation choices, input mutation before dispatch, managed release reopening and failed-job log visibility. Targeted review probes passed. The updated server-engineering reference passed a missing-reference baseline and follow-up retrieval scenario; this is documentation validation, not model inference validation.

No real model weights, patient CT, remote SSH deployment or GPU hardware runs were performed. Queue correctness does not prove a speedup, a reliable real-model peak-memory estimate or clinical segmentation quality. Linux synthetic CI is separate from target-server verification.
