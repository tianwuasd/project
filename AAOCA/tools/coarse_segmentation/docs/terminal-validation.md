# Independent terminal validation

## Paired preprocessing 1.3 / terminal 0.3 (2026-10-23)

The terminal now consumes completed paired preprocessing artifacts. Windows: preprocessing and actual producer-to-consumer integration **61 passed**; segmentation **79 collected, 77 passed, 2 POSIX-only skipped** (29.924 seconds). New synthetic tests exercise unclipped HU preservation, exact normalized pairing, both output hashes, frozen provenance, active-writer exclusion, invalid metadata, read-only source locks, path containment without source writes, historical input reuse and mocked inference with labels aligned to normalized CT.

The preprocessing 1.3 wheel built offline, installed into a separate directory and produced verified paired synthetic images. Dependency-free terminal help/history, Bash syntax and Ruff F/I checks on changed modules passed. The Linux workflow also runs preprocessing and integration tests; a published run is recorded separately after completion.

Independent review found and verified fixes for mutation before invalid runtime-path rejection, malformed progress summaries crashing selection and unnecessary write access to the source lock. The server-engineering method reference passed baseline gap identification and a follow-up retrieval/application scenario for both paired and older normalized-only caches.

No target-server deployment, real model inference or patient-data processing occurred. Grid/contract correctness does not validate real segmentation quality or performance. Historical validation below describes the earlier release.

## Previous terminal 0.2

Baseline: GitHub main `1bc139a19137af0950d0d45022b9b04bf0e8e8d7`, fetched and confirmed before changes on 2026-10-23. Product version 0.2.0.

Tests use synthetic 8×8×8 CT volumes, mock TotalSegmentator masks and synthetic process owners. Covered: preserved image bytes and native geometry, full mask/source verification, frozen requests/code, input changes while queued, independent rerun directories and lineage, merge-only reuse, cgroup ancestors, one shared thread/memory budget, delayed and partial GPU availability, two simultaneously active mock GPU inference workers, serialized cold cache, CPU-only merge, stop while waiting, persisted error logs, menu cancellation, read-only dependency-free queries, managed release reopening and detached real CPU planning/resume.

Windows: 64 tests collected, 62 passed, 2 POSIX-only cases skipped (2026-10-23; 26.4 seconds). This includes the existing web/CLI regression and real detached CPU planning/resume. Dependency-free help/history, Bash syntax and argument forwarding, Ruff F/I and Git whitespace checks passed.

Linux: **64 passed**, no skips, Ubuntu 24.04 / Python 3.11, 11.864 seconds. [CI run 39906387653](https://github.com/tianwuasd/project/actions/runs/39906387653) passed for code commit `aa8f321c15a08721e41c20aad5eb52c0914b1da0`; job `120259221272`. This includes inherited lock lifetime after the owner exits and stopping a registered orphan model without affecting an unrelated process.

Independent review prompted fixes for inherited-lock release, orphan model stopping, explicit CPU/CUDA installation choices, input mutation before dispatch, managed release reopening and failed-job log visibility. Targeted review probes passed. The updated server-engineering reference passed a missing-reference baseline and follow-up retrieval scenario; this is documentation validation, not model inference validation.

The maintained local source and GitHub source were synchronized and verified by SHA256 (48 files). The server-engineering source and installed reference also matched; skill commit `cedaef772454d94539d046a9943d5524928fd4a9` was published separately. The original local README was missing the previous coarse-segmentation link; synchronization added both the existing link and the new terminal entry without discarding unrelated content.

No real model weights, patient CT, remote SSH deployment or GPU hardware runs were performed. Queue correctness does not prove a speedup, a reliable real-model peak-memory estimate or clinical segmentation quality. Linux synthetic CI is separate from target-server verification.
