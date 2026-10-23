# Independent AAOCA segmentation workbench

2026-10-23. User selected a Chinese terminal panel inside AAOCA, after synchronizing the existing repository. This adapts server-engineering without changing the eleven-class method.

Use root launchers `start_segmentation.py/.sh`. Keep the existing preprocessing and web panels independent. Reuse private environment installation, license validation and model cache operations. A separate runtime contains settings, named jobs, immutable requests and code snapshots. Each new/rerun job gets a unique result directory. Resume uses the original snapshot and manifest; successful artifacts are reused only after full hashes match. History and queries need only the standard library.

Steps: header plan (CPU), model inference (three existing TotalSegmentator tasks), merge/QC (CPU). New tasks may reuse a historical raw-HU manifest; merge-only tasks may reference verified inference checkpoints. Normalized preprocessing outputs are incompatible. Rerun creates new outputs; input references and parent lineage are explicit.

One queue owns the entire run. Explicit allocated GPU candidates intersect actual inherited visibility and are frozen by UUID; never enlarge CUDA visibility. One case per GPU, bounded total parallelism, free-memory/process checks before each dispatch, partial availability and cancellable waiting. CPU preparation occurs before GPU waiting. Release each GPU when inference exits, before CPU-only merge. First successful inference warms shared model downloads before concurrent inference begins.

CPU affinity and ancestor cgroup limits, short load sample and available RAM bound one shared CPU/thread/memory budget. Header-based estimates cover native and 0.5-mm intermediate grids, with explicit limits and recorded uncertainty. Active subprocesses share the budget, native libraries get bounded threads, and I/O concurrency caps active case processes. This is not an upstream hard RSS guarantee or measured speedup. Each compressed CT is unpacked once per attempt, without interpolation, and removed after inference. Hash validation remains complete.

Detached owner, PID creation identity, runtime and output OS locks inherited by model descendants on Linux, cooperative targeted stop, preserved completed checkpoints. All attempts and failures stay visible. Freeze source at submit; update uses bounded Git fetch/fallback, backup and fast-forward, and never changes running snapshots. No actual patient processing or target-server deployment during development.

Acceptance: synthetic geometry equivalence, checkpoint tamper/source change rejection, CPU/GPU budget and visibility, delayed/partial GPU availability, cold cache serialization, independent rerun, canceled forms, history, Linux detached stop/orphan locks, legacy tests. Record software tests separately from real hardware/model quality validation.
