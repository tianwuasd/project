# ImageCHD diagnosis implementation plan

Approved design: [specification](../specs/2026-10-10-diagnosis-design.md). Implement inline in the existing task checkout; use the isolated publication checkout for the existing GitHub project.

1. Implement audited XLSX/CSV labels (blank defaults to negative), predicted-mask-only descriptors, native disease codes and source hashes.
2. Implement three-valued candidate rules, per-disease shallow trees, real branch explanations, train/val selection and held-out metrics with abstention coverage.
3. Add independent CLI, desktop menu and CPU server steps with remembered output paths; preserve segmentation behavior.
4. Verify the actual workbook import, synthetic end-to-end commands, missing evidence, unsupported classes, patient/test isolation, package layout, server dispatch and shell syntax.
5. Request a fresh read-only review, address findings, document verification and publish code/docs only.

Implementation notes: user explicitly requested blank-as-negative and separate diagnosis engineering steps. Candidate rules require reviewed anatomical evidence; trees are an additional research baseline. Existing model/preprocessing structures remain in imagechd/, new diagnosis code lives in diagnosis/. Verification results are recorded in docs/verification.md; local data and results remain ignored.
