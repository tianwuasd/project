# Unlabelled research CT preparation

These tools are optional and separate from the desktop conversion environment. They operate on a local research workspace with `AAOCA/`, `TAPVC/`, `results/AAOCA_results/`, `results/TAPVC_result/` and `CT总清单.json`. They do not create segmentation labels or train a model.

Set `CHD_PROJECT_ROOT` to the current workspace. Optional `CHD_WORK_DIR` selects the checkpoint/log directory and `CHD_REFERENCE_ROOT` selects a CHD reference dataset. When run from inside an existing workspace, its root can be detected automatically.

```powershell
$env:CHD_PROJECT_ROOT='D:/research/congenital_heart'
$env:CHD_REFERENCE_ROOT='D:/reference/CHD'
python tools/research_ct/organize_ct.py audit
python tools/research_ct/organize_ct.py copy
python tools/research_ct/research_preprocess.py verify
```

`audit` writes a plan from the registered successful CT units; `copy` copies and verifies files using that plan; `samples` reads selected full pixel arrays. These are real operations, not status commands. Do not rerun them only to inspect existing state.

`research_preprocess.py verify` uses synthetic data. `pilot` processes up to three cases (reusing a recorded pilot selection when present), while `all` processes the complete CT manifest. Request these data operations deliberately. Existing research outputs are reused only after configuration/input/output hash checks.

The current numerical configuration uses a nominal 0.5 mm target with RAS axis ordering and preserved obliquity, voxel-cell field of view and centre. Rounding the shape means actual output spacing may differ slightly; the affine records that actual grid. The source image is retained. Output remains float32 HU, with no clipping, grey normalization, image registration or cropping. This target is a research option, not a claimed universal clinical or nnU-Net default.

Each result records source/output hashes, shapes, transforms, actual spacing and independent interpolation/whole-payload round-trip checks. Large outputs use separable interpolation with disk-backed intermediate arrays. Review available disk, RAM and the target resolution before a full run. Display windows in previews do not modify output HU.

Historical plans can be read after relocation using a private `project.local.json`:

```json
{"path_mappings":[{"old":"D:/old/research","new":"D:/research/congenital_heart"}]}
```

Mappings apply only on exact path prefixes and use the most specific prefix first. They do not rewrite old evidence files. All data, local mappings, logs and checkpoints remain outside this public code directory.

Install optional dependencies in a separate environment. Choose a PyTorch build compatible with the intended CPU/GPU. The original pilot used nnU-Net 2.8.1; the recorded kernel hash and configuration prevent silent reuse under a different implementation.
