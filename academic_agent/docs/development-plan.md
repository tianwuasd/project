# Academic Agent V1 Implementation Plan

> For agentic workers: execute tasks with `superpowers:executing-plans` or `superpowers:subagent-driven-development`, keep the development ledger current, and review each completed component. User authorized implementation and uploading the complete folder on 2026-09-08.

**Goal:** Deliver a runnable local V1 covering multi-source discovery, provenance-aware library management, reading evidence, comparison and a cited review, with Codex and an explicit offline demonstration.

**Architecture:** Python contracts and SQLite own domain state. LangGraph sequences durable phases. Source, parser and model backend interfaces are replaceable. Codex noninteractive subprocesses operate on bounded supplied text; the deterministic demonstration backend is labelled synthetic throughout.

**Tech Stack:** Python >=3.11; Pydantic 2, httpx, pypdf, LangGraph with SQLite checkpointing, pytest, Ruff. Actual environment versions are locked and recorded after installation.

**Spec:** [Design](design.md), based on the reviewed 2026-09-07 design. This document refines the approved M0–M2 scope and records implementation decisions.

## Global Constraints

- Everything belonging to this application lives in `academic_agent/`; repository-wide CI is the one infrastructure exception.
- Git Flow: bootstrap main; develop; feature/academic-agent-v1; reviewed merge to develop; release/academic-agent-0.1.0; merge release to main and back to develop; tag academic-agent-v0.1.0. No force pushes.
- Own original implementation. No source copied from noncommercial ARS packages. Record inspirations and dependency licenses.
- Foreign network calls use explicitly configured proxy routing (this machine: 127.0.0.1:7897), domestic/local calls direct. No TLS bypass. HTTP clients have timeouts, capped retries, response size caps and safe redirect handling.
- Defaults: <=100 candidates, <=20 reading documents, <=3 model workers; <=2 network retries, <=1 schema repair. Model call count and timeout limits are enforced before dispatch. Unknown spend is never zero.
- Credentials are never logged, exported or committed. V1 uses existing Codex login; no standalone paid API credential is reused. Future providers implement the same Backend protocol.
- An abstract is never labelled as full-text reading. Scanned PDFs produce an explicit needs-OCR result, not invented text. Plain Markdown/text and PDF text layers are supported in V1; heavy OCR remains a parser extension.
- Stable source/DOI IDs and document content hashes; citation keys immutable. Same-title different-author records are not automatically merged. Preprint version links retain separate documents.
- Every claim references existing evidence. Exact quote matching and semantic support are separate statuses. Model reviewers receive independently selected full available context, not only the writer's excerpts.
- Initial import and review export check publication status; known retracted evidence cannot support ordinary conclusions. Unknown/unreachable checks remain explicit.
- Immutable files precede transactional registration. Reject cancelled jobs, expired attempts and stale project revisions. Uncertain model attempts retain budget reservations. Resume reuses successful content/version/backend keyed jobs.
- Offline fixtures demonstrate mechanics only. Never market test pass counts as scholarly accuracy, novelty proof or human evaluation.

## Scope decisions

1. V1 ends at M2. Trends, novelty candidates, automatic experiments and full manuscript authoring remain documented next milestones.
2. The local CLI is available (0.153.2), authenticated using ChatGPT in a normal user process. The restricted tool process lacks the Windows user-profile token. Use `codex exec` first; expose doctor diagnostics and do not bypass product sandbox restrictions.
3. Existing Python is 3.14. Validate dependencies there and CI on 3.11/3.12/3.14; do not install a second global runtime merely to match the proposal's tentative 3.12 choice.
4. LangGraph checkpointing and SQLite job persistence have separate jobs: graph phase resume vs idempotent domain work. Test both rather than asserting either covers everything.
5. Model-native prose uses verified claim IDs. Deterministic rendering assembles citation anchors and comparison tables to prevent fabricated bibliography links; natural-language review synthesis must also pass claim-level review.
6. V1 supports explicit manual review decisions with provenance and an independent Codex reviewer. No automatic label implies human approval.

## Interfaces and ownership

`models.py` is shared and owned by the coordinator. `Work`, `SearchPage`, `Document`, `Evidence`, `Claim`, `ReadingCard`, `ReviewVerdict`, `ModelJob`, `ModelResult` are Pydantic models. Components consume/produce these models; serialized JSON has schema_version=1. Source adapters own network field mappings, Store owns identity matching and durable state, workflow owns sequencing, exporters own formatting only.

| Shared pair | Producer → consumer | Preflight decision |
|---|---|---|
| Sources / Store | SearchPage.records → upsert_work | All source IDs and provenance retained; Store assigns canonical ID |
| Parser / Evidence | Document.pages → deterministic spans | Page number is physical PDF page, 1-based; abstract uses page 0 |
| Evidence / Backend | spans + bounded document → ReadingCard | Backend may reference supplied evidence IDs only |
| Backend / Verifier | claims → verdicts | Full supplied context available to independent reviewer; missing verdict blocks support |
| Store / Workflow | immutable artifact ID → checkpoint | Commit conditional on attempt/revision; completed jobs reused |
| Workflow / Export | approved claims + source state → report | Restriction/unknown status displayed; no dangling references |

## Task 0: Project contracts and environment

Files: `pyproject.toml`, `src/academic_agent/models.py`, `config.py`, `tests/test_models.py`, `docs/development-ledger.md`.

- [x] Create validated contracts, configuration and source/parser/backend protocols; reject out-of-range limits and unsafe project IDs.
- [x] Isolate dependencies under ignored `.venv`, create lockfile and validate CLI runtime diagnostics.
- [x] Run contract tests: negative limits, invalid read status, author normalization, malformed model results.
- [x] Record baseline commit, environment and decisions before component dispatch.

## Task 1: Multi-source discovery

Files: `network.py`, `sources.py`, `tests/test_sources.py`.

Interfaces: `Source.search(query: str, limit: int) -> SearchPage`; adapters create source-qualified temporary Work IDs and preserve raw response provenance. `HttpTransport.get_json(url, params)` and `get_bytes(url, params)` return bounded payloads and never expose credential query strings in errors.

- [x] Implement OpenAlex, Semantic Scholar, Crossref and arXiv with bounded pagination, source-specific parsing, status checks, retry/partial distinctions and raw response capture.
- [x] Provide concurrent source collection with independent failures and deterministic result ordering.
- [x] Test missing DOI/abstract, nested authors, arXiv versions, retraction flags, pagination cap, 429 retry, HTML error response and second-page failure.
- [x] Run small live probes separately from offline contract tests; save redacted findings.

## Task 2: Storage, import, documents and evidence

Files: `store.py`, `documents.py`, `bibliography.py`, `tests/test_store.py`, `tests/test_documents.py`, `tests/test_bibliography.py`.

Interfaces: `Store.upsert_work(work) -> Work`; `Store.add_document(document)`, `Store.evidence(work_id) -> list[Evidence]`; `parse_document(path, work_id) -> Document`; BibTeX/RIS file imports return Work objects with import provenance.

- [x] Implement SQLite migrations, conservative identity resolution, immutable document storage, metadata/abstract/full-text levels, stable citekeys and screening decisions.
- [x] Generate exact spans deterministically with page/offset/hash and verify references on model output.
- [x] Add transactional job leases, project revision fencing and reservation state; use separate DB connections per operation.
- [x] Test same-title collisions, DOI variants, preprint versions, lost/expired leases, cancellation, stale commits, file hash tampering, import round-trip and scanned PDFs.

## Task 3: Codex backend and evidence review

Files: `backends.py`, `analysis.py`, `tests/test_backends.py`, `tests/test_analysis.py`.

Interfaces: `Backend.run(job: ModelJob) -> ModelResult`; `read(work, document, evidence, backend) -> ReadingCard`; `review(card, document, evidence, backend) -> list[ReviewVerdict]`. Model outputs are schema-validated and evidence references checked independently.

- [x] Implement Codex subprocess with explicit read-only sandbox, disabled tools where supported, structured schema, process-tree cancellation, bounded output, timeouts, no inherited project/MCP instructions and redacted diagnostics.
- [x] Implement marked synthetic fixture backend and manual input/output exchange as an honest fallback for restricted hosts; neither masquerades as live AI.
- [x] Read problem/method/data/result/limitation/contribution; independent review covers all claims and full bounded document context.
- [x] Test fake CLI success/nonzero/timeout/malformed JSON, nonexistent evidence, negated/changed numeric claims, partial support and missing review verdict.
- [x] Run one real Codex end-to-end reading/review on an original synthetic research note; record model/runtime status and distinguish it from human accuracy evaluation.

## Task 4: Workflow, CLI and exports

Files: `workflow.py`, `cli.py`, `exports.py`, `__main__.py`, `tests/test_workflow.py`, `tests/test_cli.py`, `examples/`.

- [x] CLI commands: init, doctor, search, import, attach, screen, read, review, ask, export, status, run/resume, demo. Provide --help, understandable Chinese README and machine-readable errors.
- [x] LangGraph phases discover → prepare → read → review → export, backed by checkpoints and fenced domain jobs; resume after process interruption.
- [x] Export literature JSON/CSV/BibTeX, reading cards, comparison matrix, outline, cited review, evidence ledger, query log, gaps and quality report. Escape unsafe Markdown/CSV/BibTeX content.
- [x] Never overwrite human-edited drafts; create immutable version directories plus a latest pointer. Out-of-date input and publication state invalidate dependent artifacts.
- [x] Test zero hits, unavailable sources, abstract-only run, partial corpus, resume cache, exhausted call budget, cancelled late results, stale export and CLI packaging.

## Task 5: Validation, review and Git Flow release

Files: `docs/validation.md`, `docs/git-flow.md`, `docs/architecture.md`, `docs/extending.md`, `docs/research-sources.md`, `CHANGELOG.md`, repository `.github/workflows/academic-agent.yml`.

- [x] Run pytest, Ruff, wheel build and installed CLI smoke test. CI runs offline tests on Windows and Linux with declared supported Pythons.
- [x] Independent component/whole-branch review, fix actionable defects, recheck changed behavior.
- [x] Run public source probes and a real Codex reading/review; record exact outcomes and unresolved environment limitations.
- [x] Record what is not validated: domain recall, human 120-claim dataset, scholarly coverage targets, multi-agent quality advantage, OCR. Provide evaluation procedure; do not fabricate scores.
- [x] Commit focused changes, push feature/develop/release/main with merge commits and annotated version tag, verify remote paths and commit IDs. User explicitly requested upload; no further upload confirmation needed.

## Future academic acceptance

The prior proposal's human-labelled recall, precision, semantic support and coverage thresholds remain acceptance goals for real-topic validation, not automated release claims. Use a frozen development/held-out split, per-topic metrics, error/refusal analysis and a single-agent baseline when real topics are supplied.

## Release completion record

All implementation and review gates are satisfied. Frozen candidate 1fd7fc0 passed all six CI jobs; local full suite passed 192 tests. Git Flow release details and the final authoritative refs are described in [release-0.1.0.md](release-0.1.0.md).
