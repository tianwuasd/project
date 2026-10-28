# Full source supervised data implementation plan

Latest explicit user correction: retain a whole held-out center for independent
testing. Expand only the source train/validation pool, removing selector and
unlabeled reservations. Keep native nnU-Net random case/foreground patch sampling,
crop/pad and augmentation; do not add per-case epoch coverage.

Default ct_holdG:30 train/8 validation/20 G test. ct_holdA:30/8/20;
ct_holdB:32/8/18 because two B cases remain quarantined. Each source center
contributes 4 seeded validation cases and its remaining eligible cases to train.
No target label enters preprocessing, capacity tuning, training or best selection.

1. RED regression: all source cases used, target IDs unchanged, no source
   reservations, new protocol identity and rejection of old preparation.
2. Update source split/counts and UI/docs; preserve fold names, target evaluation
   and sampler bytes. Retain old frozen tasks and require new preprocessing.
3. Run focused/full control and available CPU synthetic nnU-Net API regressions;
   no patient computation or target-server jobs.
4. Publish corrected project first; rebuild/validate/publish/install Skill v2
   from that exact commit, preserving v1 and concurrent CT changes.

Keep250 epochs×250 training attempts/50 validation steps, CPU/RAM80, cooperative
GPU/batch policy and4 GiB reserve. Do not rewrite signatures of old caches.
