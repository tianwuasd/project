# Development history and handoff

This is the public, non-patient-specific history. Detailed patient mappings and local paths belong to the private workspace memory.

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

## Maintenance contract

Maintain the code and its synthetic tests together. Local deployment and this code mirror must be compared by file hashes before synchronization. Keep all local clinical records out of Git. Do not repeat completed historical recovery scripts or launch full data processing merely to update handoff documentation.
