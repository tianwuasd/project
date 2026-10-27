# Final resource execution fixes

Goal: close the four reviewed execution gaps without changing scientific outputs.
Spec: ../specs/2026-10-27-stage-resource-policy-design.md sections 5.1, 5.3, 6, 13.2, 13.6; exact scoped brief in the private project-final-fix-brief.md.

The approved repair is one wave, implemented by one agent, with no publication or GPU execution. All 250/250/50, source fit/cal scope, checkpoint and allocated GPU contracts remain frozen.

- [x] Add failing execution regressions for read slots/release, observed memory compatibility, supervised preparation lifecycle, and export concurrency/parity.
- [x] Add read_admission.py: process-shared OS-lock slots; wrap only actual reader calls, preserving all compute allocations. Integrate CPU queue, fingerprint factory, prediction reader and parallel export.
- [x] Add case_memory.py: same-run completed observations keyed by stage, shape/spacing and producer identity; conservative full RSS high-water floor, no subtraction of shared pages. Keep active reservations at least their original or corrected requirement.
- [x] Replace prediction's nested pool with a queue-owned CPU preparation process. Transfer workflow FD with Popen, record identity before requests, supervise both processes, clean both independently after dead GPU leader.
- [x] Make training feedback report unavailable causal classification, retain approved two-window expansion on generic wait, and apply supported prefetch reduction when selected; record requested/applied settings.
- [x] Use the CPU command queue for source export with per-case atomic writes and deterministic aggregate commit; extend strict producer identity for new dependencies.
- [x] Run focused CPU controls, real installed nnU-Net synthetic preparation/prediction APIs and affected loader checks, then one final full control aggregate. Record exact evidence, skips, inventory and risks in docs and private report; commit only scoped project files.

Review focus: killed leader with live writer; unrelated/recycled identity; readers released before computation ends; stale/incompatible RAM measurements; partial export failure cannot commit aggregate completion.
