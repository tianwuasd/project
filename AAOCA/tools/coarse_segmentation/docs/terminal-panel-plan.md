# AAOCA independent segmentation panel implementation plan

**Goal:** Deliver a standalone terminal panel with resource-bounded parallel inference and recoverable named jobs.
**Architecture:** Standard-library controller and frozen job launcher; inference environment runs a shared queue and isolated case subprocesses. Reuse the existing model/geometry/merge functions and private setup operations.
**Tech Stack:** Python 3.11+, NumPy/NiBabel/psutil; pinned TotalSegmentator 2.18.0; Linux process/OS locks.
**Spec:** [terminal-panel-design.md](terminal-panel-design.md)

## Constraints and review focus

Keep HU/mm and existing labels/model flags unchanged. Preserve other repository changes. No patient inference in development. Test absent GPUs, changing UUID visibility, nested cgroups, corrupt history/checkpoints, owner death with live descendants, and simultaneous submissions.

- [x] Task 1: resources.py and executor.py: resource allocation, stable GPU identities, standalone infer/merge checkpoints, immutable input hashes, native decompression. Test limits, geometry equivalence and invalid checkpoints.
- [x] Task 2: scheduler.py and batch_worker.py: shared queue, dynamic availability, cold-cache serialization, CPU tail, detached ownership and inherited locks. Test concurrency, stop, partial availability and Linux orphan lock.
- [x] Task 3: workbench.py and terminal.py plus AAOCA root launchers: new/history/resume/rerun, settings and raw input lineage, environment/license, queries and updates. Test new directories, freeze, canceled forms and old output readability.
- [x] Task 4: documentation, full synthetic regression, Linux CI, independent review, source mirror synchronization and GitHub publication. Preserve exact verification limits and update reusable template where applicable.
