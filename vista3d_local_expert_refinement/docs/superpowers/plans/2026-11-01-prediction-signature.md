# Prediction Signature Compatibility Implementation Plan

**Goal:** 0.7.0 completed training can feed patched prediction without rewriting its signature, source task or weights.

**Architecture / approved design:** Preserve the optional positive-integer selected_global_batch_size from the training record in prediction's expected signature. Retain strict equality of every other signature field and split. Bind the reviewed 0.7.0→0.7.1 prediction-only change to exact fingerprints of the complete source-code maps in a release-side audit JSON. Keep audited transitions cumulatively; never accept arbitrary code differences based on a version label. Old frozen jobs keep their own code; new prediction jobs use copied inputs.

**Tech Stack:** Python stdlib control tests; local nnU-Net API tests; Git code-only release; existing Skill v2 bundler.

**Scope:** User approved the diagnosed fix, reuse of the immediate previous 0.7.0 training, and subsequent Skill v2 synchronization. Broader earlier-version migration is excluded. CPU/GPU policies, training, preprocessing, prediction algorithms and evaluation protocol unchanged.

## Tasks

- [ ] Add failing freeze_models fixtures for selected batch4, nondefault batch, old source-map transition, source bytes retained and strict mismatch refusal.
- [ ] Patch only prediction signature/code comparison; version0.7.1; create exact code-map audit after final source changes.
- [ ] Verify source/weight bytes unchanged, transition audit exact, unknown code and invalid metadata rejected; run focused prediction/input tests and quick check.
- [ ] Review complete diff; publish and verify project main. Recovery is update panel then create a new prediction selecting original completed training, not restart old frozen prediction.
- [ ] Rebuild Skill v2 from the published fixed commit, update compatibility rule/reference, verify, publish and install; keep v1/CT unchanged.

## Review focus

- Do not substitute prediction defaults for the training batch.
- No dropping unknown signature fields or accepting arbitrary training-code changes.
- Missing/corrupt compatibility audit must fail closed for differing code maps.
- Keep completed source training/weights untouched and preserve frozen model identities.
- Future small fixes retain audited edges; semantic/structure changes need explicit compatibility assessment.
