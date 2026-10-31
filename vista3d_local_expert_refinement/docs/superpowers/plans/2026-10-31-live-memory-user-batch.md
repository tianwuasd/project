# Live memory protection and user-selected global batch implementation plan

User authorization: 2026-10-31 requests real-time memory adjustment, user-selected batch default4, at most one nonempty rank per selected sample, project GitHub and Skill update. The preceding memory diagnosis and current instruction define the requested change; no extra training budget changes are authorized.

Goal: freeze the selected global batch (default4) through single/multi-GPU membership changes; sample actual host/cgroup memory while ranks run, gate supply expansion using fresh headroom, stop/reap the owned group before reserve exhaustion and recover from the latest complete checkpoint with a smaller host budget. Sudden external allocation may still outrun sampling; no absolute OOM guarantee. CPU80%, VRAM4GiB reserve,250x250/50 and native sampling remain.

Architecture: `memory_guard.py` owns cheap live memory/cgroup/event collection, reserve decisions and pressure errors; `ddp_runtime.py` supervises 1s checks and owned descendant cleanup; `cooperative_control.py` preserves requested batch and bounded host-memory recovery; `training_feedback.py` gates growth with live headroom. UI/request validation and worker environment persist the selected batch. Old frozen jobs remain unchanged and compatible completed preprocessing is reused.

- [x] Add regression tests for live headroom, protected recovery, unconsumed work/owned cleanup, selected batch default/validation/CLI,1/2/3/4-rank partitions of global4 and extra-card admission.
- [x] Implement fixed requested batch and capacity admission. Single-card OOM does not silently reduce the batch: wait for additional admitted cards or stop with an actionable capacity message; all chosen ranks receive nonempty partitions.
- [x] Implement live host/cgroup checks throughout each capacity/formal invocation, admission with reserve, measured process-tree memory evidence, fresh growth gating, bounded smaller-host-budget checkpoint recovery; preserve external-process ownership boundaries and log missing evidence.
- [x] Validate control/CPU API/distributed paths appropriate to changes. Explicitly distinguish synthetic/Gloo results from target-server CUDA/NCCL and real CT acceptance. Verify unchanged preprocessing identity producers.
- [ ] Update panel/docs/handoff, preserve concurrent repository updates, publish project main and verify remote SHA.
- [ ] Rebuild Skill v2 source/ZIP/manifest from that exact published commit; update concise operating references, validate generated template and installed Skill; preserve v1/CT assets; publish and verify Skill main.

Behavioral acceptance: global4 partitions [4],[2,2],[2,1,1],[1,1,1,1]; no fifth training rank. Selected batch persists in preview, frozen plan, environment, capacity/formal args, logs and checkpoint identity. Current low memory prevents worker growth; reserve breach produces structured evidence, owned rank/worker exit, checkpoint-aware recovery or safe stop, never an unclassified infinite retry. SIGKILL without OOM evidence is not relabelled as proven OOM.
