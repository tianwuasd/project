# CHD paper core alignment implementation plan

> Execute inline in this session. Keep the existing working tree and its verified backup; no commit or deployment is required.

**Goal:** Align the data-supported segmentation core with Xu et al., Medical Image Analysis 90 (2023), 102953, Figure 4, sections 4.2/4.3/5.1 and Table 3.

**Architecture:** Add the CHD paper U-Net separately from the preserved Freiburg reference port. Switch the six-stage factory, packaged presets and checkpoint identity together. Keep Excel resampling, seven-structure labels, server workflows and existing recovery contracts.

**Tech stack:** PyTorch, NumPy, SciPy, YAML, pytest.

**Spec:** Local PDF in `D:/code_project/heart_disease/A clinically applicable AI system for diagnosis of congenital heart diseases based on computed tomography images.pdf`; extracted text in `D:/code_project/.runs/chd-paper-align-20261013/paper.txt`.

## Constraints and review focus

- Backup `back/before-paper-alignment-20261013` contains 125 verified working-tree files, including Excel resampling. Reverified unchanged on 2026-10-15.
- Preserve eight output classes (background + seven ImageCHD structures); do not invent initial-vessel/PV/SVC/IVC annotations.
- Full paper settings are large: validate their tensors on the meta device; execute numerical training/prediction tests with explicitly reduced settings.
- Old checkpoints must fail clearly rather than load into a different network. Old preprocessing caches remain compatible.
- Odd grids and small smoke inputs must preserve coordinates and gradients. Seven-frame augmentation must be shared across the sequence.
- Paper omissions (padding, activation slope, exact recurrent merge, Dice/CE combination) must be documented as implementation choices; do not claim parameter-count or clinical-result equivalence.

## Task 1: Paper network and configuration

Files: new `src/chd_ct/models/chd_unet.py`, `tests/test_chd_paper.py`; factory/configs/presets; training/checkpoint identity.

- [x] Add tests for Table 3 settings, five/eight-scale shapes, IN/LeakyReLU blocks, position branch pool4/conv64/conv64/class-conv/up4, temporal gradients and old-checkpoint rejection.
- [x] Observe the new architecture tests fail before implementation.
- [x] Implement `CHDUNet(dim, classes, base, levels, spatial_gate)` with `features(x)` returning the penultimate feature map. Use same-padding double convolutions, transpose-convolution upsampling and concatenation. Preserve reference `UNet`/`GridUNet`.
- [x] Use defaults 3D base64/48, five scales, batch4, 15 epochs; 2D 512/base32/eight scales/batch4/12 epochs; LSTM 512/base32/two layers/seven frames/batch1/three epochs. Keep smoke size16/base2/three scales.
- [x] Switch architecture/model-format IDs and reject the old architecture explicitly.

## Task 2: Training semantics

Files: `imagechd/augment.py`, `imagechd/losses.py`, tests.

- [x] Use weighted Dice plus ordinary cross entropy; class weights apply to Dice as described in section 4.3. Keep ignore255 masking for both losses.
- [x] Use scale/rotation/deformation for 3D; rotation/flipping without scaling for 2D and shared sequence augmentation. Keep amplitudes explicit as assumptions.
- [x] Verify nonzero finite gradients and sequence consistency; retain the existing 50/25/25 learning-rate schedule and frozen encoder.

## Task 3: Integration and documentation

Files: README, paper mapping, source/workflow/code guides, backup index.

- [x] Run the complete test suite with a fresh temporary directory under `D:/code_project/.runs`, then lint.
- [x] Inspect the final diff, verify source/preset agreement and backup checksums.
- [x] Document exact aligned settings, parameter-count differences, remaining data/fold/postprocessing limitations and model migration. Report what was actually tested, without claiming full-size training.

## Decisions

- Existing user authorization is to update the current project and back it up first; keep those changes together in place.
- The recurrent cell equations and direction concatenation remain explicit engineering choices because the paper does not supply an executable recurrent graph. Match its stated 32 channels per direction, two layers and seven frames; report the resulting parameter count.
- Retain the old `chd-author-unet.yaml` as a historical configuration with an explicit incompatible architecture. The current entry point directs users to `chd.yaml`.

- Review ruling: default batch4/base48/128³ retained 30.105 GiB of forward tensors before backward, already exceeding a 24 GiB GPU. Added optional non-reentrant activation checkpointing for encoder/decoder and position blocks, enabled in paper presets and disabled in smoke. This preserves real batch4 and Dice semantics. Meta saved storage is now 9.015 GiB; peak GPU memory remains unverified. Output/loss/parameter-gradient equivalence tests pass; independent reviewer rechecked the fix.

## Completion evidence

2026-10-15: full suite 159 passed / 10 POSIX-only skipped (43.00s); Ruff check and changed-code format checks passed; diff whitespace check passed. CPU smoke on three existing Excel-resampled real cases completed six stages, prediction and evaluation; test prediction ct_1002 restored original 512×512×137 grid and affine. Backup 125 checksums and all three source/preset pairs verified. Independent reviewer rechecked activation checkpoint fix with no remaining blocking findings. No full-scale GPU training or clinical validation; no commit/push.
