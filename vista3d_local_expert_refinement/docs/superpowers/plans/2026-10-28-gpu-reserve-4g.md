# GPU safety reserve 4 GiB implementation plan

User-approved change: increase the cooperative GPU capacity safety reserve floor
from 2 GiB to 4 GiB. Keep the 20 GiB planner ceiling and the existing 10% floor
for larger GPUs; subtract the reserve from freshly observed free memory once.

1. Add CPU-only regression cases for 24 GiB/23 GiB free -> 19 GiB usable,
   planner ceiling without double subtraction, larger-card 10% preservation,
   and per-rank batch inference using the resulting capacity. Observe RED.
2. Centralize reserve constants in gpu_budget.py and use them in admission,
   capacity explanations and decision receipts. Preserve scientific budgets,
   preparation identity, PID/visibility guards and bounded OOM recovery.
3. Update panel patch version, engineering explanation and validation inventory;
   run focused GPU/control regressions and the existing panel control suite.
4. Commit/publish the project first. Rebuild Skill v2 from that exact commit,
   update the guide/source map/provenance, verify generated copies and retain v1.
5. Publish and install only the changed Skill paths with preimage verification;
   check installed generation and hashes. No target-server jobs are started.
