# Three-fold training and evaluation plan

**Goal:** Complete the paper's three-fold training protocol for the six data-supported ImageCHD stages, preserve patient separation and expose fold-level training results.

**Source:** Xu et al. MedIA 2023 sections 4.3/5.1; local extracted paper text under `D:/code_project/.runs/chd-paper-align-20261013/paper.txt`.

**Backup:** `back/before-threefold-training-20261015`, 134 working-tree files with verified SHA-256, including the paper core and Excel resampling.

## Design decisions

- Three patient-grouped outer test folds cover the development cohort exactly once. Each fold reserves approximately 10% of all development patients for validation from the other two thirds; the rest train. Exact counts and patient/case IDs are saved. Seeded assignment and validation rounding are explicit implementation choices.
- Default `development_scope: auto`: automatic filename split caches use all retained patients as the development cohort; explicit patient CSV caches preserve their test patients as external holdout and cross-validate train+val only. `all` or `train_val` can override the scope explicitly in YAML. No cache rewriting/copying.
- `chd.yaml` defaults to three folds. Existing `chd-smoke.yaml` remains a single-split regression/demo config; add `chd-cv-smoke.yaml` for reduced three-fold tests. Old single-split model collections remain readable.
- Store `fold_0/`, `fold_1/`, `fold_2/`, each with a complete six-stage model collection, its own encoder/LSTM pair and recovery files. Root `models.json` references all three with hashes and the split plan. Root training contract binds config, split plan and cached file hashes.
- Folds execute sequentially; the existing within-fold GPU stage queue is retained. Workers use the frozen fold assignment; locks and server process discovery include nested fold workers. Completed folds can be verified and skipped on resume.
- Inference selects the held-out fold for known development patients (OOF). Independent/new patients use the mean of the three per-stage probability outputs before class selection; each recurrent model uses its own frozen encoder. This prevents three-fold ensemble scores on patients seen by ensemble members.
- `test --split auto` selects `cv` for development data with a CV model, otherwise legacy test split; `--split cv` explicitly scores all supplied development cases. Separate evaluation reports per-fold and pooled case-weighted Dice. Test labels never select checkpoints.
- Server model queries show all folds and epoch/loss; log lookup and process checks discover `fold_*/queue.json`. No server connection or formal GPU training is required for this code change.

## Implementation and checks

- [x] Add deterministic patient fold construction and selection tests; prove disjoint train/val/test, test coverage, group integrity, scope policy and small cohort behavior.
- [x] Add three-fold orchestration, frozen contracts, worker assignments, completed-fold resume, summary and checkpoint integrity checks.
- [x] Add stage-probability ensemble, OOF routing, CV selection and per-fold evaluation, retaining old single-split behavior.
- [x] Update server inspection/recovery, launchers, configs and documentation.
- [x] Run reduced 18-stage integration plus inference/evaluation, tampering and resume regressions; run full suite and lint. Independently review leakage, model pairing and recovery boundaries.

Implementation review found source-config drift across folds; fixed by loading the frozen root effective configuration in every fold and checking child config/source/mode before publication. Dedicated regression edits the original YAML during training and verifies all 18 stages use the initial settings. Real three-case Excel-resampled smoke and all three native-grid restoration checks passed. Final test counts and limitations are recorded in `docs/verification.md`.

## Boundaries

The 97 usable ImageCHD cases are not the paper's 1282 development/2468 clinical cohorts. Missing init/PV/SVC/IVC supervision, unresolved network parameter differences, precise augmentation and clinical metrics remain as documented. A smoke run proves software behavior, not paper accuracy.
