# Label-only segmentation implementation plan

**Goal:** Direct raw-HU inference with three existing tasks and one editable segmentation file per volume.

**Architecture:** Reuse the current inference checkpoint and CPU export stages. A standard NRRD writer retains independent, potentially overlapping segments on the original affine. Final-result verification precedes intermediate-mask verification on resume.

**Spec:** label-only-design.md

## Global constraints

- TotalSegmentator 2.18.0; original HU/mm input; no external resampling or normalized image copies.
- Source grid, eleven structure identities and overlaps preserved.
- Remove only validated, locally owned inference masks after successful atomic export.
- Old frozen jobs retain their contract. Public files contain no case-specific information.

## Tasks

- [ ] Implement editable export and full read-back verification using pynrrd; validate affine, binary masks, names, colors and overlap preservation with two small geometry samples and independent slicerio reading.
- [ ] Integrate final-file reuse, input integrity and owned-mask cleanup into executor and scheduler; make raw-HU label-only output the new terminal and CLI default.
- [ ] Update user instructions and dependency constraints; validate one actual raw CT through the real three-model backend and then validate reuse, without training.
- [ ] Review changed source and export lifecycle, synchronize maintained/public copies, publish only a reviewed file whitelist to GitHub and verify the remote files.
- [ ] Record bounded test measurements, verify process exit and remove the approved test directory.

## Review focus

- Oblique or negative-direction affine must remain geometrically correct in Slicer RAS space.
- Overlapping structures must survive round trip without overwrite.
- Corrupted final labels must never be treated as reusable after mask cleanup.
- Cleanup must never follow links or remove historical source-task masks.
- Publication must exclude all patient-specific paths, records, secrets and outputs.
