# Author U-Net Alignment Implementation Plan

> Use superpowers:executing-plans for inline implementation and one independent final review.

**Goal:** Replace the custom U-Net core with a traceable port of author-published 2D/3D networks, retaining gate as a separate optional extension.
**Architecture:** source-aligned valid core; mirror/crop image-grid adapter; optional output gate. Versioned checkpoint contract.
**Tech Stack:** existing PyTorch/Numpy/PyYAML; no new runtime dependency or Caffe installation.
**Spec:** ../specs/2026-10-09-unet-author-alignment.md

## Global constraints
- Source files and licenses are bundled unchanged with SHA256 provenance.
- Default CHD practical widths are explicit adaptations; author-width profile and smoke profile remain distinguishable.
- Existing prepared data and task/server entry points remain usable; old weights require retraining.
- No full GPU accuracy claim and no claim that this is an official PyTorch package.

## Review focus
- Caffe source's atypical final 2D upconv, retained 3D upconv widths, concat order and dropout.
- Center alignment for odd/tiny output grids and repeated symmetric mirror extension.
- Frozen 2D features remain compatible with ConvLSTM; gate is not a hidden feature-encoder modification.
- Changed architecture cannot silently load legacy model collections.
- Wheel includes original references/licenses and runnable smoke configuration.

## Task 1: Source-aligned core and adapters
- [x] Add source signature/shape/equivalence and grid/gate tests; observe failure.
- [x] Implement models/unet.py core, models/grid.py adapter, models/gate.py extension.
- [x] Verify source signatures, raw output and independent graph output, gradients, geometry.

## Task 2: Integration and delivery
- [x] Update configs/presets, model factory, versioned weights, architecture records, package data.
- [x] Verify full suite, six-stage real-cache smoke, unlabeled native prediction, installed demo.
- [x] Update usage/source/mapping docs; independent review found no actionable confirmed bugs.
- [ ] Commit local; synchronize project GitHub subtree; verify remote HEAD.
