# 0.3.0 Innovation Candidate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从当前已复核证据生成真实模型驱动的研究候选，经有界反例检索和独立审查，输出可追溯的多维评估。

**Architecture:** 沿用 Store、MeteredBackend 和 Codex/Exchange 协议。新增独立的证据快照、分析产物、创新任务和反例检索模块；CLI 只负责参数与状态展示。保留候选历史，把补充资料后的评估存为新版本。

**Tech Stack:** Python >=3.11, Pydantic 2, SQLite, 现有模型后端；不新增运行依赖，不升级 uv.lock 中的第三方版本。

**Spec:** `academic_agent/docs/spec-v0.3.md`；已核对现有接口见 `academic_agent/docs/next-stage-contract-notes.md`。

## Global Constraints

- 0.2.0 已完成代码验收，但正式发布被自动审批审查拦截并等待用户确认。允许从其已验收的本地 develop 建立独立 feature 分支进行本阶段本地研发；正式推送仍保持暂停，不通过其他分支、PR 或间接方式绕过发布审批。本阶段发行仍须先完成 0.2.0 发布及其远端核验。
- Windows/Linux、中文说明、UTF-8、既有项目数据库和 CLI 兼容；所有操作测试限原创合成材料与独立 `.runs` 目录。
- 至多 3 个候选；每个候选最多 2 轮反例检索，每轮最多 2 条查询，每来源每查询最多 10 条结果。用户重复相同轮次表示恢复，不暗中新增轮次。
- 不生成全球首次状态、无依据的综合分数或虚构实验结果。作者贡献、系统假设和材料范围内的评估分开保存。
- 每次模型调用经 `MeteredBackend.validated()`，领域验证成功后缓存；生成、评估、审查共用预算。预算暂停不自动提高限额，Exchange 等待可恢复。
- 完整上下文超过上限时拒绝并建议显式缩小文献范围；不截断后假装看完。模型只引用输入的工作、陈述、证据 ID。
- 固定 `tests/fixtures/trends-0.2.0/` 保留原始字节并继续校验；不修改旧趋势 schema、算法或渲染来配合新搜索字段。
- 一次一个实现任务，每任务独立审查，最终整版审查；审计文件保留。本地研发继续按用户授权执行；正式分支和标签的远端发布仍遵循本节第一项的待确认门槛。

## File Boundaries

以下路径均相对 `academic_agent/`，仓库 CI 文件另行注明。

| 文件 | 责任 |
|---|---|
| `src/academic_agent/research_context.py` | 冻结当前资料、核对陈述与证据依赖、计算不被自身模型计量改变的输入指纹 |
| `src/academic_agent/research_artifacts.py` | 有范围限制的不可变分析产物、哈希校验与最新版本指针 |
| `src/academic_agent/innovation_models.py` | 候选、反例轮次、评估和独立审查的严格数据契约 |
| `src/academic_agent/innovation_analysis.py` | 有界结构化模型生成、评估、独立审查；不直接搜索或写数据库 |
| `src/academic_agent/innovations.py` | 上述任务的预算、恢复、版本和状态协调 |
| `src/academic_agent/challenges.py` | 候选反例查询的轮次、来源状态、已保留 ID 和待办 |
| `src/academic_agent/demo_innovation.py` | 原创多文档演示数据协议；真实资料必须拒绝 |

既有 `analysis.py` 继续负责原文阅读/复核；创新假设不写进其 `Claim(category="result")`。

### Task 1: Current evidence snapshot and immutable analysis records

**Files:** Create `src/academic_agent/research_context.py`, `src/academic_agent/research_artifacts.py`, `tests/test_research_context.py`, `tests/test_research_artifacts.py`, `docs/research-artifacts.md`.

**Interfaces:**
- Consumes existing `Store`, `current_cards(store)`, `decisions(store, cards)`, `verify_span`, `load_config`, `atomic_json`, `digest`.
- Produces strict `ResearchSnapshot` with selected work IDs, current works/documents/evidence/cards/decisions, project question, search coverage and gaps; `freeze_research(store: Store, *, work_ids: list[str] | None = None, require_supported: bool = True) -> ResearchSnapshot`; `research_fingerprint(snapshot: ResearchSnapshot) -> str`; `assert_research_current(store: Store, snapshot: ResearchSnapshot) -> None`.
- Produces `read_research(root: Path, *, work_ids: list[str] | None = None, require_supported: bool = True) -> ResearchSnapshot` for readonly status/guide consumers. It shares the same projection and validation as `freeze_research`, opens an existing database with SQLite `mode=ro`, and never instantiates a writable `Store`. Missing/config-only projects raise ValueError with an explicit missing-state explanation without creating directories, databases or blobs.
- Encode the existing absolute database path with `Path.resolve().as_uri()` before query parameters. Use normal read transactions and committed WAL state; never use `immutable=1` on the live project. SQLite's own WAL/SHM coordination files are allowed, while authoritative research data and unrelated paths must remain unchanged. Cover an idle open connection with a newly committed Work and literal `#`/`%` paths.
- Produces `publish_research(store: Store, kind: str, snapshot: ResearchSnapshot, payloads: dict[str, bytes]) -> Path`; `load_research(directory: Path) -> dict`. Allowed kinds are `innovation`, `challenge`, `assessment`; loader returns verified `manifest` and named `files`, rejecting malformed or changed files.

- [x] Write a tests-first fixture using `seed_demo(tmp_path); runner = Runner(tmp_path, offline=True); runner.run()`. Freeze it and assert exactly two selected papers and twelve supported claims. Append an unrelated analysis log and rerun a cached model task; fingerprint must be identical. Change a source document, manual verdict, screening, question or search record in separate cases; fingerprint must change or freeze must reject stale input.

```python
seed_demo(tmp_path)
runner = Runner(tmp_path, offline=True)
runner.run()
before = freeze_research(runner.store)
runner.store.log("innovation_progress", {"phase": "review"})
assert research_fingerprint(freeze_research(runner.store)) == research_fingerprint(before)
assert len(before.work_ids) == 2
```

- [x] Run `python -m pytest tests/test_research_context.py tests/test_research_artifacts.py -q --basetemp .runs/test-v03-context`; record the initial missing-module failure before implementation.
- [x] Implement stable ordered snapshots. Default selection is included, non-retracted works; explicit IDs must exist and satisfy those conditions. With `require_supported=True`, every selected work must have a current readable document and at least one supported claim; reject empty selection. With `False`, retain unread/unreviewed entries as gaps and only expose validated evidence. Always retain project coverage, including pending/excluded/retracted counts, so selection is visible.
- [x] Verify `read_research` and `freeze_research` give the same meaningful snapshot for an existing original project; test missing/config-only paths remain unchanged and a metadata source-abstract replacement invalidates both views. Share validity logic between the two readers so future guide behavior cannot diverge from model-input validity.
- [x] Fingerprint meaningful selected metadata, document bytes/IDs, exact evidence, reading cards, effective review decisions, question/scope and search events. Exclude capture timestamps, model attempts, success counts, innovation/assessment logs, budget limits and operational proxy values. Do not use `exports.snapshot_token` for this fingerprint. Prompt payloads exclude volatile capture time as well, preserving resume identity.
- [x] Implement host-controlled version directories under `analyses/<kind>/`, exact file allowlists, canonical manifest, symlink/path traversal rejection, payload hashes and atomic latest pointer. Require a before/after read fence, then `BEGIN IMMEDIATE` and a current-input check before publication. Hashes detect modification, not author authenticity. Preserve earlier versions and user edits; report corruption rather than silently substituting new content.
- [x] Cover unknown IDs, stale source abstracts, foreign evidence, malformed decisions, source mutation during freeze/publish, altered payload/manifest, hostile latest path and unchanged historical records. Run the two focused files, Ruff and fixed 0.2 compatibility test; document exact results and commit `feat: freeze current reviewed evidence for research tasks`.

### Task 2: Actual candidate generation and independent grounding review

**Files:** Create `src/academic_agent/innovation_models.py`, `src/academic_agent/innovation_analysis.py`, `src/academic_agent/innovations.py`, `src/academic_agent/demo_innovation.py`, `tests/test_innovation_generation.py`; modify `src/academic_agent/backends.py`; create `docs/innovation-candidates.md`.

**Interfaces:**
- Consumes Task 1 snapshot/fingerprint/publication functions, existing `Backend`, `ModelJob(kind="synthesize")`, `ModelResult`, `MeteredBackend.validated()`.
- Produces strict `InnovationCandidate`, `CandidateBatch`, `GroundingReview`; `generate_candidates(snapshot: ResearchSnapshot, backend: Backend, *, timeout_seconds: int, max_input_chars: int) -> CandidateBatch`; `review_candidates(snapshot: ResearchSnapshot, batch: CandidateBatch, backend: Backend, *, timeout_seconds: int, max_input_chars: int) -> GroundingReview`.
- Produces `create_innovations(runner: Runner, *, work_ids: list[str] | None = None) -> dict`, returning `status`, `input_hash`, `output`, `candidate_ids`, `gaps`; status is `complete`, `needs_input`, `budget_paused`, `stale`, or `failed`. Resume means calling it again with unchanged selection.
- Publishes kind `innovation` with fixed payload names: `candidates.json` (`CandidateBatch`), `grounding-review.json` (`GroundingReview`), `report.md` (Chinese report with scope limits), `references.bib` (host-generated bibliography from real selected records), and `evidence.json` (host-built referenced claim/evidence/decision appendix). Later tasks consume these verified names. No model-provided citation keys or URLs are trusted.

Model-context boundary: use the stable snapshot as the dependency basis, but construct an explicit model projection. Preserve the complete selected documents, exact claim/evidence/decision chains and citable selected metadata. Represent retrieval through query/time/source/status/count/limit/truncation/error/retained-ID coverage plus a hash of the full saved event; do not send raw provider responses or unselected document/metadata bodies to the model. Keep the full event in the immutable snapshot for provenance. This is a declared projection, not silent truncation of selected paper context; the normal input-cap rejection still applies after projection.

- [x] Write first failing tests with a recording backend that returns structurally valid original proposals. Assert two separate jobs with full available document contexts, task/version-specific schemas, no invented reference IDs, deterministic host IDs, and no `result` Claim created. A separate bad backend emits an unknown work/claim/evidence ID and must yield no published candidate batch and no successful cached output.
- [x] Define actual candidate fields: ID, `statement_type="system_hypothesis"`, question, closest-work references, claimed difference, grounding claim/evidence references, possible counterexamples, falsifiable hypothesis, minimum validation, data/compute requirements, risks, one or two targeted queries, material cutoff. Require nonblank content and at most three candidates; references form valid work→claim→evidence chains. Author contribution excerpts keep their source type and are not themselves novelty judgments.
- [x] Implement generation and independent review prompts/schema through existing bounded `ModelJob`. Review checks claims about the supplied papers against complete context; its supported verdict means grounding within supplied evidence, never proof of novelty or experimental success. Missing/duplicate review items fail validation. Preserve partial/contradicted/insufficient verdicts visibly.
- [x] Test a large original raw search response containing unselected metadata: no raw sentinel or unselected body enters either model job, full selected document text and retrieval limitations remain, and the saved raw-event hash participates in stable identity. Raw response duplication must not make an otherwise bounded selected-paper task unusable.
- [x] Implement `create_innovations` with separately validated generation and review calls. A completed generation stays cacheable when review pauses; no latest publication until the pair and current source snapshot pass host validation. Stable candidate IDs derive from input fingerprint and normalized candidate fields, excluding run times.
- [x] Extend DemoBackend by dispatching explicit versioned synthesis tasks to `demo_innovation.py`. Require original marker lines in every input document, exact validated spans and a known fixture protocol. Use bundled original scheduling examples to demonstrate workload-dependent candidate hypotheses; reject real or mixed inputs. Codex and Exchange remain generic adapters.
- [x] Run `python -m pytest tests/test_innovation_generation.py tests/test_backends.py -q --basetemp .runs/test-v03-generation` and Ruff. Cover budget pause after generation, Exchange response resume, input oversize before dispatch, stale input before commit, invalid output not cached, existing read/review demo behavior. Write report and commit `feat: generate and independently review research candidates`.

### Task 3: Bounded counterexample search with recoverable coverage

**Files:** Create `src/academic_agent/challenges.py`, `tests/test_challenges.py`; modify `src/academic_agent/workflow.py`, `src/academic_agent/innovation_models.py`, `src/academic_agent/innovation_analysis.py`; update `docs/innovation-candidates.md`.

**Interfaces:**
- Extends `Runner.search(query: str, *, cached: bool = False, limit: int | None = None, request_id: str | None = None) -> dict`. `None` preserves existing query identity/limit. Explicit limit is a strict integer 1..100, with separate cache identity; effective limit is also bounded by project cap. An optional request ID is saved outside the old summary on an actual dispatch; it does not change cache identity or retroactively relabel a previously cached event.
- Search returns `retained_work_ids: list[str]` in the returned object and stores that field at the search event top level. The existing event `summary` stays byte/schema compatible; cached old events without IDs return an empty list plus explicit `retained_ids_available=False`, never fabricated IDs.
- Produces strict `ChallengeRound`; `challenge_candidate(runner: Runner, candidate_id: str, *, round_number: int = 1) -> dict`, returning status, verified candidate identity, round, per-query source summaries, retained IDs, unresolved tasks and output path. Round number is 1 or 2; requesting the same round resumes failed parts.
- Extracts shared pure validators in `innovation_analysis.py`: `validate_candidate_batch(snapshot: ResearchSnapshot, batch: CandidateBatch) -> CandidateBatch` and `validate_grounding_review(snapshot: ResearchSnapshot, batch: CandidateBatch, review: GroundingReview) -> GroundingReview`. Revalidate mutable inputs, research input hashes, host-derived candidate IDs and current reference chains without model calls. Generation/review reuse the same checks; do not copy their validation logic into the artifact reader.
- Produces `load_candidate(root: Path, candidate_id: str) -> dict` in `challenges.py`, returning `candidate`, `batch`, `review`, `snapshot`, `version_path`, `manifest_hash` with typed validated objects and a host path. Find historical candidates through verified innovation versions, bind batch/review input hashes to the manifest snapshot, and call the shared validators. No writable Store or model is needed to read history. Task 4 consumes this loader; keep `challenges.py` independent of `innovations.py` to avoid a circular service import.
- Produces readonly `read_challenges(root: Path, candidate_id: str) -> dict` with `rounds: list[ChallengeRound]`, `outputs: list[str]`, `hits: list[ChallengeHit]`, `unresolved_tasks: list[dict]`, and `status`. `ChallengeHit` exposes work ID, current status list, screening reason, access level and publication status. Read current pending/excluded states from the existing database using normal encoded readonly access; they are not all present in `ResearchSnapshot.works`, and historical receipts do not establish their current state.
- Produces readonly `candidate_currentness(root: Path, candidate_id: str) -> dict` with exactly `candidate_id`, `current: bool`, and `gaps: list[dict]`. It wraps the same premise-validity policy used by `challenge_candidate`, returning safe categorical gaps. Changes to retrieval source configuration alone do not invalidate the candidate's underlying evidence; changed question, project queries, title or material/evidence/review dependencies remain checked. Task 4 consumes this helper without duplicating premise policy.
- `ChallengeRound.receipts` records the current applicable receipt per query; `search_attempts` retains verified historical attempts. The retained-hit union includes all attempts, so records saved during a partial failure do not disappear when a successful retry returns different results. Determine current retry failures from current receipts, retaining earlier failures as history; bind attempts to exact saved event content.

- [x] Write failing mocked-source tests that independently count invocations. Two queries across configured sources must pass a limit of 10 to each adapter; a third round is rejected before network work. Repeat a completed round makes zero new source calls. Partial failure retries the failed query only, retaining the successful query's artifact and truthful source state.
- [x] Extend search without changing old default cache keys or `summary`. Return actual deduplicated retained IDs after the project cap; record dropped count. Validate integer limits before dispatch, and distinguish explicit-small-limit cache from old searches. Publication refresh remains bounded by retained IDs.
- [x] Save actual effective limit outside the old strict `summary` and verify the Task 2 model projection consumes new search-event retained IDs and limits. For old records where those values were never saved, keep their unavailability explicit; do not invent historic IDs or limits.
- [x] Persist challenge identity from candidate content, round, queries, configured sources and effective limit. Reuse completed query receipts even if another query failed. Do not infer which papers were retained from unordered titles or metadata similarity. Persist a unique per-dispatch request ID before each search; a crash after search logging but before receipt publication recovers by that ID plus exact query/scope, with receipts bound to the full saved event hash.
- [x] Include normalized candidate identity/content and batch input hash in round identity, but exclude artifact version/manifest hash/capture time. Pin the chosen candidate manifest as provenance and validate that original artifact on resume. Re-saving an identical cached candidate must not reset its round count or repeat successful queries; test that exact sequence. Document deterministic resolution of duplicate valid candidate IDs across equivalent saved versions, while rejecting inconsistent content/hash bindings.
- [x] A newly requested round 2 is a fresh bounded search even when its candidate queries match round 1. Scope recovery to that round's durable start and persisted dispatch request ID plus exact query/scope; an older healthy general-search cache must not silently satisfy a new round. Round 1 need not have succeeded before round 2 starts with revised configured sources. Existing round queries/sources/limits remain frozen: a configuration mismatch cannot silently create a second same-number round. Test failed round 1, fresh round 2, then restored round 1 configuration and resume; independently count adapter calls and prove no event cross-attribution.
- [x] Compute deterministic coverage statuses for not-searched, empty, partial/unavailable/rate-limited, truncated, dropped-at-cap, pending-screen, excluded-with-reason, unread, unreviewed and ready. An empty source result never asserts absence outside the search scope. New records retain normal pending screening; excluded records and unknown publication checks remain disclosed.
- [x] Test partial failure retaining C followed by successful retry retaining D: both actual retained IDs remain covered, current screening/read gaps are refreshed, and the earlier source failure is historical after successful recovery rather than an unresolved current retry failure.

```python
# With two mocked query groups, first succeeds and second is rate limited.
first = challenge_candidate(runner, candidate_id, round_number=1)
assert first["status"] == "source_failed"
second = challenge_candidate(runner, candidate_id, round_number=1)
assert second["round"] == 1
assert successful_query_call_count == 1
```

- [x] Test new-paper deduplication/cap, offline rejection, stale or tampered candidate, source failures, empty and unread hits, configured source changes and cache recovery. Run `tests/test_challenges.py`, `tests/test_workflow.py`, `tests/test_innovation_generation.py`, `tests/test_trend_compatibility.py` with a fresh basetemp and Ruff. Record evidence and commit `feat: track bounded counterexample searches and unresolved evidence`.

### Task 4: Evidence-limited assessment and historical freshness

**Files:** Modify `src/academic_agent/innovation_models.py`, `src/academic_agent/innovation_analysis.py`, `src/academic_agent/innovations.py`, `src/academic_agent/demo_innovation.py`; create `tests/test_innovation_assessment.py`; update `docs/innovation-candidates.md`.

**Interfaces:**
- Produces strict `InnovationAssessment` with four named dimensions `novelty_clues`, `research_value`, `feasibility`, `evidence_strength`, each containing textual judgment, reasons and valid references; gaps and minimum validation are explicit. No overall numeric score.
- Assessment uses schema 1 and task version `innovation-assessment-v1`; independent `AssessmentReview` uses schema 1 and `innovation-assessment-review-v1`. Reasons are text with claim/evidence references. Host-derived review unit IDs cover every generated judgment, reason, gap description and validation proposal/falsification condition. Scope/cutoff are host metadata, or must also enter review if they contain generated free text. Review items cover the exact unit set. Host coverage limits and evidence availability cannot be cleared by model output or presented as scholarly truth certification.
- Publishes kind `assessment` with exactly `candidate.json`, `assessment.json`, `assessment-review.json`, `context.json`, `report.md`, `references.bib`, and `evidence.json`. Context pins historical candidate provenance and current coverage; analysis output paths and analysis-version UUID-only changes are excluded from semantic input identity. Original document/source paths remain dependencies under the shared research fingerprint and premise policy: changing those provenance fields can stale an assessment even when text bytes match. This task does not implement transparent source-file relocation.
- Consumes Task 3 `load_candidate(root, candidate_id)` and shared pure candidate/review validators for historical identity, `candidate_currentness(root, candidate_id)` for premise validity, and `read_challenges(root, candidate_id)` for actual query/current-hit coverage; use the documented final contracts rather than reimplementing candidate lookup, reference checks or premise policy.
- Produces `assess_candidate(runner: Runner, candidate_id: str) -> dict` and `innovation_status(root: Path) -> dict`. Assessment returns the same operational statuses as generation and retains coverage/missing-input information; status uses `read_research` and verified artifact reads, distinguishing historic candidate availability from latest assessment freshness without creating writable Store state.
- Keep the assessment signature without a new selection argument. Oversized complete included context returns safe `model_input_too_large` before dispatch, with guidance to explicitly screen unrelated material according to the actual research scope while preserving candidate premises, or manually choose a backend-compatible input configuration. Do not alter screening automatically, recommend dropping counterevidence to obtain a conclusion, invent an unavailable CLI option, or truncate content.

- [x] Write failing tests for both a read/reviewed counterexample and metadata-only hits. In the latter case, assessment must retain insufficient evidence and actionable gaps; the host must not permit a model output to clear recorded failed/unread/truncated states. A missing challenge is recorded as not-searched, even if the model confidently asserts novelty.
- [x] Build assessment input from the immutable historic candidate, all applicable challenge receipts and a current `freeze_research(..., require_supported=False)` snapshot. Preserve the candidate as a historical idea when new work arrives. Expose source references only when current and validated; expired candidate evidence is a gap, not a currently supported premise.
- [x] Implement actual assessment generation and a separate independent review job with complete bounded context. Use `statement_type="scope_limited_assessment"`, material scope/cutoff, four dimensions and a falsifiable minimum validation proposal; reject unknown references. Review all factual grounding units, retaining partial/insufficient/contradicted findings. Enforce recorded coverage limitations in host-generated report sections as well as prompts.
- [x] Publish Chinese report, structured candidate/assessment/review, evidence appendix and BibTeX citations through Task 1 artifact functions. Report hypotheses as proposed work, keep original author contributions attributed, and do not create observed results. Escape Markdown/CSV values and only emit safe source URLs using existing helpers.
- [x] Test raising only budget permits cached resume; document/manual-review/screening/question/search changes invalidate current assessment; own completion logs do not. Test removed/retracted dependencies, no search, failed search, unread and excluded hits, schema failure, malicious source instructions, editable historic report preserved and unchanged candidate identity.
- [x] Run `python -m pytest tests/test_innovation_assessment.py tests/test_innovation_generation.py -q --basetemp .runs/test-v03-assessment` and Ruff. Record results and commit `feat: assess research candidates within verified evidence limits`.

### Task 5: CLI guidance and installed usage journey

**Files:** Modify `src/academic_agent/cli.py`, `src/academic_agent/guide.py`, `src/academic_agent/usage.py`, `tests/test_usage.py`, `README.md`; create `tests/test_innovation_cli.py`, `docs/innovation-walkthrough.md`.

**Interfaces:** Consumes Tasks 2–4 service functions. Produces the commands below, UTF-8 JSON statuses, nonzero exits for blocked/failed operations, and readonly guide actions using complete argv arrays.

```text
academic-agent --project projects/study innovate --work-id work_A --work-id work_B
academic-agent --project projects/study challenge candidate_ID --round 1
academic-agent --project projects/study assess-innovation candidate_ID
academic-agent --project projects/study innovation-status
academic-agent self-test --output .runs/usage-v03
```

- [x] Add failing subprocess tests for candidate generation, IDs and output paths, budget pause/resume, unknown candidate, invalid third round and a Chinese/space project path. Assert JSON can be decoded from stdout and guide does not create a database, spend model budget or run searches.
- [x] Wire argparse commands to services with global options before subcommands; generation `--work-id` may repeat. Operational errors get actionable Chinese explanations, commands and existing project budget details; never echo provider credentials or full prompts into errors.
- [x] Extend guide to show generation readiness, specific failed query, pending screening/reading/review, Exchange request, budget pause, historical idea and stale assessment. Resolve latest paths through verified artifact APIs; missing/corrupt artifacts yield repair guidance.
- [x] Extend bundled self-test with one bounded original candidate→assessment journey and cache-resume/staleness checks. Real search failures are covered by mocked contract tests; the offline self-test must state it does not contact sources. The report verifies actual CLI output, expected candidate count, referenced files, synthetic labels, review status, no invented global novelty and no repeated completed model attempts.
- [x] Extend the existing exact scenario-set assertion in `tests/test_usage.py` for the new named scenario, preserving all eleven 0.2 scenarios and their mandatory checks; do not weaken it to a loose count/subset assertion.
- [x] Document both original offline walkthrough and real-use sequence, including how to supply readable counterexample material. Run focused CLI tests, one standalone self-test and Ruff; commit `feat: expose candidate workflows and practical usage checks`.

### Task 6: Release evidence and Git Flow

**Files:** Update `pyproject.toml`, `uv.lock`, `src/academic_agent/__init__.py`, CHANGELOG, README, `docs/validation-v0.3.md`, `docs/project-roadmap.md` and this plan. CLI derives version from the package constant introduced in 0.2.

**Interfaces:** Consumes all reviewed components and installed package; produces 0.3.0 wheel/sdist, usage evidence, release branch and annotated tag.

- [x] After per-task reviews, run full pytest with fresh `.runs` basetemp, Ruff and old static trend verification. Build wheel/sdist without upgrades or network; install the wheel with lock-pinned runtime dependencies only and verify import origin from `site-packages` outside the source tree. The isolated build backend was cache-resolved from `setuptools>=77` because this uv version does not lock build-system requirements; the wheel records `setuptools (84.0.0)`.
- [x] Run installed-package self-test in a fresh directory. Inspect report assertions and generated candidate/assessment citations, not just exit code. Record wall time as one observation, no performance or academic-accuracy estimate.
- [x] Complete U09 through installed CLI with an original added counterexample: retain an existing candidate, import/attach/read/review the added material, then assess the same candidate again and verify current references plus unchanged historical bytes. Use explicitly controlled Exchange responses when the fixed Demo protocol rejects the added material; validate actual request/response files and CLI outputs, and label this as protocol/state acceptance, not real model understanding. Runtime installation stays independent of test packages.
- [x] Run one bounded authorized Codex generation/review smoke on original synthetic text, reusing existing authorization without exposing keys. Run a separate small public counterexample query through an existing adapter; record actual availability, limits and unread gaps. Do not confuse demo success with real model success or human academic evaluation.
- [x] Dispatch final whole-branch independent review with the complete commit-range diff and prior findings ledger. Resolve Important/Critical defects with covered tests and scoped re-review; update validation with actual evidence and remaining limitations. Technical approval at `a9ed210`; both guide findings closed, 68 covering tests and Ruff pass. Integration `cbb9c71` brings published 0.2 ancestry and documentation only; final regression/build gates remain separate.
- [x] Merge feature into develop, create `release/academic-agent-0.3.0`, merge release to main and back into develop; add `academic-agent-v0.3.0`, push authorized refs through configured proxy, verify remote heads and six Windows/Linux × Python CI jobs. Preserve audit worktrees; no force push or research-data commit.
- [x] Mark 0.3 complete only after all release gates. Keep the persistent overall goal active for 0.4 writing and domain-extension acceptance.

## Plan self-review

Spec coverage: Task 1 owns stable dependencies and preservation; Task 2 owns real generation, explicit claim types, bounded contexts and review; Task 3 owns source boundaries and resumable countersearch; Task 4 owns multidimensional assessment and stale history; Task 5 owns user operations and offline use tests; Task 6 owns real interfaces, installed artifacts and release. No academic quality percentage is inferred from engineering tests.

Interface check: services consistently consume `ResearchSnapshot`, use `research_fingerprint` instead of export counters, and return operational status dictionaries. Search IDs live outside the old strict summary. Cache-resume identity excludes budget/run-time changes, while source/scope changes remain meaningful. Historical candidate identity and current assessment validity are separate. Implementation and review progress is recorded in the task checkboxes and validation report; later tasks remain unverified until their own gates.

Readonly preflight: existing `Store.__init__` writes schema state, so status/guide consumes a root-path readonly reader rather than constructing Store. Both snapshot entrypoints share source-abstract, card and decision validity; this explicitly carries the 0.2 guide-review finding into the next architecture.

Release closure: user-confirmed normal atomic push and remote refs verified; main 617d557, develop 46d3630, release abdf3da, new tag target 617d557. Run34353904196 and all six jobs succeeded. Final445tests and installed usage/fix gates passed. Overall goal remains active for0.4.
