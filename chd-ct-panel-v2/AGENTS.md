# CT V2 maintenance

This directory is the independent CT V2 control plane. Preserve the sibling
`chd-ct-reproduction` byte-for-byte. The user explicitly requested main integration
on 2026-11-01 after the initial independent-branch delivery. Main is now the formal
delivery; retain compatibility with codex/ct-panel-v2. Maintenance follows the
checked repository branch, supports only these two branches, and never switches it.

The scientific implementation is CT 0.24: six networks, threefold evaluation,
predicted ROI, raw votes, validation-gated limited completion and GPU guard.
Do not substitute nnU-Net or its training hyperparameters from the engineering
template. The template supplies lifecycle/resource principles. Read README.md,
V2_DESIGN.md and docs/V2_TEMPLATE_MAP.md before changing interfaces.

2026-11-01 explicit user request adds STU-Net-S as a separate scientific method
inside this V2 project. Its embedded foundation_engine uses the 0.7.0 engine,
with its own roots, identities, environment and registry. Keep the original CT
method unchanged; the above non-substitution rule still applies to that method.
The integrated entry is start_server_v2.sh; V1/start_server.sh remains untouched.

Maintain separate package/artifact identities, roots, registry and release
environments. No V1 caches/models/tasks are accepted by the V2 panel. All
submissions/resumes use verified frozen source. Queries never install or submit.
Test actual data-loading settings, source graphs, interruption and branch safety.
Do not restart real training without user authorization; synthetic CPU tests are
allowed. Report Linux/CUDA skips and limits honestly.

Publish source/configuration/tests/docs only. Patient data, private server notes,
credentials, runtimes, logs and weights are excluded from code commits.
