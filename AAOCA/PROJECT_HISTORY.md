# Development history and handoff

This is the public, non-patient-specific history. Detailed patient mappings and local paths belong to the private workspace memory.

## 2026-10-23: paired preprocessing 1.3 and segmentation 0.3

Preprocessing now publishes normalized float32 CT and unclipped resampled HU together on one grid, with separate hashes and round-trip validation. The independent segmentation terminal defaults to selecting a completed preprocessing task and uses its HU companion for TotalSegmentator. Frozen source receipts, writer exclusion, paired-image hashes and geometry checks protect selection, execution and reuse. Labels align with the normalized CT; results remain separate from preprocessing. Legacy raw-HU manifests and frozen segmentation jobs remain available. Normalized-only older preprocessing tasks require a new task from original HU for this workflow; clipped intensities cannot be inverted. Synthetic tests cover actual producer-to-consumer integration, mocked inference and corrupt/incomplete input rejection; no target-server or real-model run is implied.

## 2026-10-23: independent server workbench 1.2

Added a Chinese terminal workbench following the CHD task/configuration layout, with separate startup from raw-data conversion. Only preprocessing is connected. Spacing, CPU and RAM default to automatic selection with manual overrides. Resource allocation uses visible affinity/cgroup limits, sampled idle CPU, 90% available memory, case working-set estimates and bounded I/O; concrete spacing is frozen while automatic resources are refreshed at each attempt. Named history, read-only queries, explicit background start/stop/resume, logs and registry dispatch locking reuse the existing preprocessing engine. Synthetic tests cover menu cancellation, allocation boundaries and verified background reuse; no patient processing or target-server deployment is implied. See SERVER_WORKBENCH.md.

## September 2026: conversion and inspection

Added local DICOM, roster and PDF inventory, identity checks, CT NIfTI conversion, ultrasound frame/static-image output, reports, progress, pause and resumable processing. Investigated identifier conflicts, duplicate SOP identifiers, missing pixel payloads, incomplete geometry and invalid frame timing. Keep original files and audit evidence.

## October 2026: evidence-based recovery

Separated malformed ultrasound headers from unrelated image processing, accepted verified complete duplicate instances, and split explicitly tagged cardiac phases only after geometry checks. Confirmed identity rules are external local policies. No fabricated missing pixels or guessed ultrasound time vectors were introduced.

## October 13–15: incremental projects and repair UI

The maintained pipeline accepts any combination of CT, ultrasound, PDF and roster inputs. SQLite preserves source registration, successful units and attempt history. Appending data processes new or changed units; one unit's failure does not block unrelated units. Reports use eight mutually exclusive completeness categories and three issue families. PDF text insufficiency is informational.

Repair now has a separate window for unique-name matching, verified complete-copy recovery and explicit phase splitting. It shares a run lock with the main window and supports safe pause/stop and before/after project statistics. “Only update report” also maintains classification folders; it does not rescan sources or certify pixel quality.

## October 16–17: source-data research and report clarity

Investigated missing-pixel evidence and published related reconstruction research separately in the monorepo's medical-image-preprocessing component. Estimated reconstruction must remain a separate experimental output. Reports describe precise ultrasound failure modes and distinguish calibration review from conversion failure.

## October 17: relocation and reproducible research CT tools

The local data owner reorganized the workspace. Launchers now locate environments relative to code, research tools discover a configured workspace, and historical path mappings are read from a local file rather than baked into public code. Existing output references are relative to the project root. The report title is persisted per result project so TAPVC reports retain their own title.

Published the complete maintained incremental pipeline rather than private-path compatibility redirects. Added synthetic regression coverage for moved projects, escaped report titles and path-prefix mapping. An optional venv-local Tcl/Tk resource locator supports portable Windows runtimes.

The research tools select only registered successful CT outputs, preserve provenance, check copies, compare NIfTI metadata and provide a separately saved HU-preserving resampling workflow. The current local work reached a small pilot; this is not a claim of full-dataset preprocessing. Nominal spacing, actual sampling geometry, interpolation and round-trip validation are recorded. No model-specific intensity normalization, inter-patient registration or training is automatically performed.

## October 18: paired CHD research data preparation

Added a standalone resumable batch tool for AES ZIP segmentation labels and paired NIfTI images. It validates archive paths, reads full voxel payloads and gzip trailers, checks integer label values and image/label grids, and verifies lossless research copies with SHA256. Readable pairs with mismatched geometry are retained separately for review; geometry, intensity units and anatomical label meanings are not guessed. No resampling, normalization, registration or training is performed. Credentials and dataset manifests remain local. Synthetic tests cover damaged inputs, grid differences, unsafe archives, source changes, output conflicts, restart behavior and process locking.

## October 23: portable AAOCA preprocessing

Added an independently installable CPU package with fixed HU clipping [-1000,1000] and normalization to [-1,1]. Native resolution is retained by default; explicit target spacing uses the research tool's voxel-cell FOV and centre-preserving grid with slab-based linear interpolation, RAS axis ordering and preserved obliquity. The package records actual geometry and applies NIfTI intensity scaling once.

Added header-only planning, original-manifest or directory input, isolated case failures, source/configuration/output hash checks, full voxel round-trip validation, atomic case publication, OS locking, resource preflight, CLI/status and a Linux launcher. Source scans and prior successful results are preserved. Synthetic tests cover numerical, geometry, integrity, restart and CLI behavior. This release does not claim a production server deployment or a full-dataset processing run.

## October 23: preprocessing 1.1 workflow and CPU controls

Added explicit combined prepare/start commands, inspectable nnU-Net-inspired spacing suggestions, and a bounded representative-case benchmark. Automatic geometry uses full-FOV RAS headers and does not run nnU-Net network planning or its default CT normalization. Fixed clip[-1000,1000]/1000 remains unchanged; there is no implicit 0.5 mm target.

Compressed sources are expanded once with CRC validation. Bounded slab threads, independently limited I/O, ancestor cgroup CPU/RAM checks and memory-aware case admission expose resource controls without changing voxel results. Records include inverse coordinate mappings, downsampling axes and stage/performance measurements. Detached jobs freeze inputs/configuration, validate process creation identity, support cooperative stop and verified resume, and retain separate worker logs. Code and dependency changes require a new output directory.

Synthetic preprocessing tests and existing desktop regression passed; package installation and background lifecycle were checked locally on Windows. Target Linux/server performance and real-data preprocessing were not executed. Local synthetic measurements are not server speedup claims.

## October 23: independent segmentation terminal 0.2

Added a separate root segmentation launcher and Chinese terminal workflow while retaining the original web panel. Named immutable jobs record raw-HU input lineage, code snapshots, full source/artifact hashes, resumable inference checkpoints and separate rerun outputs. CPU merge/QC can reuse verified masks after GPU workers exit. A shared resource queue intersects explicitly allocated UUIDs with inherited visibility and bounds case concurrency, CPU threads, estimated memory and I/O. Cold model-cache initialization is serialized; compressed inputs are expanded losslessly once per attempt. Scientific model flags, eleven labels and native geometry are unchanged.

POSIX descendant locks, registered orphan cleanup, independent release updates, private license setup and synthetic lifecycle/parallel tests accompany the change. Validation scope is documented separately; publication does not imply deployment or real CT/model/GPU throughput validation.

## Maintenance contract

Maintain the code and its synthetic tests together. Local deployment and this code mirror must be compared by file hashes before synchronization. Keep all local clinical records out of Git. Do not repeat completed historical recovery scripts or launch full data processing merely to update handoff documentation.
