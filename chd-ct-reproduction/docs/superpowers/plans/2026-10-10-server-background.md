# CHD background workflows implementation plan

User-authorized scope: adopt wholeheart detached execution, multi-select the server menu, prompt/save only relevant task settings, update project and server skill and publish both existing repositories.

- [x] Add CPU-aware setting resolution and a deterministic multi-step plan; bind current-job outputs and reject unavailable prerequisites before submission.
- [x] Add detached worker, inherited whole-job runtime lock, persistent logs and per-step state; preserve foreground debugging, fail fast and report stale workers without killing unrelated processes.
- [x] Verify task ordering, dependency hand-off, signal handling, duplicate submission and real CPU workflows; independently review code.
- [x] Update server/menu docs and skill examples, install and validate the skill. Publish both verified commits to the existing project and skill repositories as the delivery step.

Keep existing single-step scripts and medical/model behavior. No implicit extra preprocessing or training. No GPU prompts/settings for CPU-only steps. Server source metadata remains historical until measured on zmic44.

Verification: 93 Windows tests passed; 4 POSIX tests passed separately in WSL. Independent review fixes cover relative paths, reset propagation, GPU hand-off and startup failure status. Skill baseline found outdated foreground/sequential guidance; updated references add multi-select detached execution and task-specific settings.
Independent skill forward test passed for cached smoke/train/test with two allocated GPUs and a separate CPU-only diagnosis workflow. Skill validator and 9 server-info regressions passed; installed resources match source hashes.
