# Direct segmentation implementation plan

> **For agentic workers:** Use native implementation in this session, followed by an independent review. Steps use checkbox syntax for tracking.

**Goal:** Run AAOCA segmentation directly with explicit GPU allocation and shared-GPU memory admission.

**Architecture:** The command-line entry submits immutable requests to the existing case queue. The queue binds one case per selected GPU and rechecks available memory before each model.

**Tech Stack:** Python, argparse, UUID, CSV, PyTorch, TotalSegmentator 2.18.0, pynrrd and slicerio.

**Spec:** tools/coarse_segmentation/docs/direct-run-design.md

## Global constraints

- Native HU/mm input and original-grid editable labels; no external preprocessed CT outputs.
- Explicit visible GPU candidates; default minimum 8 GiB plus 2 GiB reserve; existing compute processes permitted.
- No private input, identifiers, credentials, weights or outputs in publication.
- One actual CT, zero training updates; remove approved test artifacts after verification.

## Review focus

- UUID spellings and remapped visible indices must identify the same physical GPU.
- Occupied GPUs with enough memory may run; the reserve is counted once.
- Each GPU receives at most one owned case concurrently.
- Stop and resume retain owned-process and frozen-input boundaries.
- Direct entry, queries and environment reuse must not invoke any interactive panel.

## Tasks

- [ ] Normalize probe UUIDs and validate both spellings, invalid UUIDs and visibility rejection with real dependency calls.
- [ ] Add direct command parsing, environment reuse, frozen submission, foreground/background monitoring and explicit queries; test argument rejection and read-only status.
- [ ] Add shared-memory admission and per-model checks; test occupied-GPU selection and reserve thresholds plus actual device isolation.
- [ ] Run one actual native CT through direct submission and verify final format, cleanup and reuse without training.
- [ ] Review the resulting code, synchronize maintained/public files, publish a code-only whitelist, verify remote contents and delete test artifacts.
