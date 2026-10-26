# Menu update download repair

**Goal:** Fix the observed 45-second fetch timeout while retaining authenticated, isolated project updates and visible progress.

**Evidence:** The server read official main `f345e372df7d6b5a6e252fff26a3d01cd0638c36`, then the bare-cache fetch exceeded 45 seconds. Runtime SSH/key/proxy settings match the established route. This is a download timeout, not evidence of rejected credentials.

**Approach:** Keep the existing updater and full history/fast-forward checks. A bounded 900-second fetch with Git's actual progress and five-second waiting updates addresses the failure. Keeping 45 seconds cannot handle the observed transfer; an unlimited wait gives no useful bound. Query budget is 30 seconds and local archive budget 180 seconds. These are implementation choices within the already requested reliable panel and complete progress display.

**Scope:** `task_update.py` owns download timing, stderr log and errors; `task_panel.py` gets version 0.3.3. Existing source snapshots, training protocol and authentication settings retain their current contracts. No server training or credential change.

- [x] Add regressions for slow-but-successful communication, retained partial progress on timeout, child cleanup and unchanged release pointer after failure; first run demonstrates the gaps.
- [x] Poll `communicate` without losing cumulative stdout/stderr, write actual Git stderr to the update backup, add `--progress`, and apply separate query/download/archive budgets.
- [x] Show the concrete failure beside the backup/log location, retain previous pointer on timeout/cancel, and verify the existing offline two-update/cache-recovery flow.
- [x] Run the full local control suite and syntax checks; update recovery/update instructions and project handoff records, sync maintained files to publication checkout, publish and verify remote main.

**Validation boundary:** Synthetic subprocess/local Git tests establish software behavior. The actual server proxy transfer and menu completion remain user-run acceptance checks. Linux process-group behavior is separate from Windows checks.
