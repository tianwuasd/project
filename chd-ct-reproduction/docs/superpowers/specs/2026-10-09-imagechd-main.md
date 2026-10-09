# ImageCHD as the primary multi-stage workflow

User intent: adapt and reuse the existing paper workflow on ImageCHD, make it the only active default, preserve the previous path in back, and update launchers/configuration/documentation/tests/GitHub. Independent preprocessing, training and prediction remain required.

Architecture: retain shared U-Net/BiConvLSTM implementations and environment/GPU safeguards. The active package implements six data-supported stages: crop64/crop128, all64/all128, blood2d, blood_lstm. Reuse the previous ROI, frozen feature encoder, scale fusion and blood-boundary refinement approach with native ImageCHD IDs. The two initial-vessel stages and disease diagnosis are explicitly unavailable because the dataset does not provide the required supervision or verified physical calibration.

Preprocessing: retain native canonical shape by default, normalize each image by its own 1st/99th percentile, store image/optional target plus geometry and patient split metadata. Support the existing imagechd7-v1 resized caches for reuse, with the resolution limitation recorded. Unknown target IDs remain ignored (255), absent foreground annotations exclude a case. Source files remain immutable.

Training: six stages run from prepared caches only. Train/val are patient-disjoint; test never selects models. Frozen blood2d encoder is bundled in the recurrent checkpoint. Model collection records stages, cache manifest hash, label/normalization protocol and short-test status. Smoke and preflight are explicitly distinguishable from full training.

Prediction: only cached images plus a complete compatible model collection; no targets. Predicted coarse ROI, four anatomy votes (1:1:2:2), blood-boundary refinement, and restore original grid. Evaluation is a separate command with ignored voxels excluded and foreground Dice denominators explicit. No fabricated disease output.

Interfaces: root preprocess.py/train.py/predict.py/evaluate.py and start.py; server scripts call those same tasks after environment/resource checks. Model and server settings remain separate. back/paper-v1 is a self-contained source snapshot and never imported by main code.

Validation: targeted tests for native/v1 cache handling, ignored voxels and patient isolation, six-stage synthetic end-to-end, prediction without target/raw access, incomplete/incompatible models, short-test protection, independent dispatch, and source snapshot isolation. Run a real ImageCHD short test on existing cached data; verify output grid. No claim of paper accuracy or real-server CUDA validation.