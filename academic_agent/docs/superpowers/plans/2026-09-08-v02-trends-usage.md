# 0.2.0 Trends and Usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付可重算的语料描述统计，以及研究者可运行的操作指引和使用自检。

**Architecture:** 新增独立 trends 模块冻结现有 Store 输入并确定性计算，采用现有不可变导出与提交前快照检查模式。guide 和 usage 模块经 CLI 接入，不新增后台服务或模型权限。

**Tech Stack:** Python, Pydantic, SQLite, argparse, pytest, Ruff；沿用 uv.lock。

**Spec:** `academic_agent/docs/spec-v0.2.md`。后续总目标在 `academic_agent/docs/project-roadmap.md`。

## Global Constraints

- Python >=3.11；Windows/Linux；沿用锁定依赖，不新增运行时依赖，不改变 0.1 数据库结构和既有 CLI 参数语义。
- 数据输出默认中文说明、UTF-8 JSON/Markdown/CSV。
- 所有测试目录使用项目 `.runs` 或 pytest 显式 basetemp，绝不操作用户实际项目。
- 审查报告保存在本计划独立 `.superpowers` 路径，不删除已被要求保留的历史审查文件。
- 本地测试 Python 可复用主工作区 `.venv/Scripts/python.exe`，设置本工作树 `src` 为导入路径；安装包测试另建环境。

### Task 1: Frozen corpus and deterministic trend artifacts

**Files:** Create `src/academic_agent/trends.py`, `tests/test_trends.py`, `docs/trends.md`（均相对 academic_agent）。

**Interfaces:**
- Consumes `Store`, `Work`, `load_config`, `atomic_json`, `snapshot_token`, `escape_md`, `_csv` existing functions.
- Produces `CorpusAnnotations`, `CorpusSnapshot`, `ComputationRecord` strict models; `freeze_corpus(store: Store, *, annotations: CorpusAnnotations | None = None, date_basis: str = "record_year", as_of_year: int | None = None) -> CorpusSnapshot`; `compute_trends(snapshot: CorpusSnapshot) -> ComputationRecord`; `build_trends(store: Store, *, annotations: CorpusAnnotations | None = None, date_basis: str = "record_year") -> Path`; `verify_trends(directory: Path) -> dict` raising ValueError on failure.

- [x] Write independent arithmetic tests with included records in 2022 (2), 2023 (1), missing year (1), future year (1); pending/excluded/retracted ignored. Assert counts against those constants, labelled duplicates count once, denominators and gaps explicit.

```python
snapshot = freeze_corpus(store, as_of_year=2024)
result = compute_trends(snapshot)
assert compute_trends(CorpusSnapshot.model_validate_json(snapshot.model_dump_json())) == result
# Fixture expectations: 2022=2, 2023=1, missing=1, future=1; no domain-growth statement.
```

- [x] Run the new tests and record expected initial import/behavior failures. Implement strict annotations and snapshots, preserving work/source/document revisions, reporting errors and cutoff dates, rejecting unknown IDs or snapshot races.
- [x] Implement deterministic counting with fixed algorithm version and canonical input hash; tests cover invalid/bool dates, alternate date basis never falling back, current year, empty corpus, duplicate labels, unknown annotations, missing year/labels, related works, failed/truncated queries and local-only coverage.
- [x] Implement versioned JSON/CSV/Markdown publish with BEGIN IMMEDIATE and snapshot-token recheck. Record calculation formulas and diagnostic thresholds; content-hash all artifacts and verify recomputation without Store. Tests corrupt each authoritative input/output, missing files, concurrent mutation, unchanged repeated computation, old human edits preserved.
- [x] Run `python -m pytest tests/test_trends.py -q` and Ruff; write report including commands/results and commit `feat: add reproducible corpus trend analysis`.

### Task 2: CLI guidance and installed-package usage self-test

**Files:** Create `src/academic_agent/guide.py`, `src/academic_agent/usage.py`, `tests/test_usage.py`; modify `src/academic_agent/cli.py`, README, `.github/workflows/academic-agent.yml`.

**Interfaces:**
- Consumes Task 1 public models/functions, `Runner.status`, current_cards/decisions, Store, and existing CLI commands.
- Produces `project_guide(root: Path) -> dict` containing Chinese summary, blockers, and `actions: list[dict]` with `argv: list[str]`; `run_self_test(output: Path) -> dict` containing boolean `passed`, scenario list and report paths. CLI propagates failure with nonzero exit.

- [x] Write guide tests proving no network/model and correct actions for missing, empty, pending, absent text, stale reading/review/export. Commands must remain argv arrays for paths/queries containing quotes or spaces.
- [x] Extend CLI parser/dispatch. New commands below must return valid UTF-8 JSON; verify-trends requires only explicit directory, self-test does not initialize default project.

```text
academic-agent --project projects/demo guide
academic-agent --project projects/demo trends --date-basis record_year --annotations annotations.json
academic-agent verify-trends projects/demo/analyses/trends/VERSION
academic-agent self-test --output .runs/usage-test
```

- [x] Implement self-test via `[sys.executable, '-m', 'academic_agent.cli', ...]` subprocess argv, offline/demo inputs and strict expected outcomes. Use spec's eight core scenarios plus readable/blank PDF cases; record observations, durations, stdout/error and each assertion. Generate original PDFs with existing pypdf writer and a standard font/text content stream: readable text uses the existing original FIXTURES, blank page must produce needs_ocr without a model call. Refuse preexisting output. Always produce partial failure report on expected runtime errors. Do not evaluate API/model quality through synthetic results.
- [x] Add subprocess acceptance tests: Unicode path, executable entrypoint, nonzero negative case, actual report file and reference checks, package self-test repeat refusal. Ensure child processes are bounded (timeouts), env UTF-8 and no provider credentials inspected. CI uses unique fresh `.runs/usage-ci`, runs self-test after demo, uploads usage output on failure via artifact action.
- [x] Run focused tests, an actual standalone self-test and Ruff; write report and commit `feat: add usage guidance and offline acceptance journeys`.

### Task 3: Release validation and goal ledger

**Files:** Update `pyproject.toml`, `uv.lock`, CLI version and `src/academic_agent/__init__.py` version, CHANGELOG, README, `docs/validation-v0.2.md`, project roadmap and this plan. Add `tests/fixtures/trends-0.2.0/`, `tests/test_trend_compatibility.py`, and `.gitattributes` for a byte-preserved original compatibility baseline; include those test inputs and release documents in the source distribution via `MANIFEST.in`.

**Interfaces:** Consumes released CLI and full suite; produces version 0.2.0 wheel/sdist, usage evidence, Git Flow branches and tag.

- [x] Run all tests from worktree and Ruff. Independently review Tasks 1–2 then full branch, fixing Important/Critical findings with covered tests and scoped re-review. Final merged release worktree: 289 passed in 72.54s; final feature CI: six environments passed. Residual WAL correction is adjudicated and independently exercised through installed CLI; rulings are recorded in release-0.2.0.md.
- [x] Save original fixed trend artifacts with hand-checkable counts; verify them without regeneration, preserve file bytes on Windows/Linux checkout, and carry this baseline into later versions. Initial compatibility test passed; full cross-platform CI remains a release gate.
- [x] Update version with `uv lock --offline`; if cache cannot resolve metadata, use existing foreign proxy policy for bounded locked dependency resolution without upgrades. Build wheel/sdist, install wheel into new environment, run installed CLI self-test outside source tree and verify module origin is site-packages.
- [x] Use a small public-source query from an existing authorized adapter, record source status, cap, real metadata sample, resulting arithmetic and reproducibility. No model calls necessary for this milestone; respect rate limits, report unavailable service honestly.
- [ ] Document exact evidence, known limits and next milestone. Commit, merge feature to develop, create release/academic-agent-0.2.0, final gates, merge main/backmerge develop, tag academic-agent-v0.2.0. Push authorized branches/tag through configured proxy, verify remote and six-environment CI.
- [ ] Mark only 0.2.0 complete in roadmap. Persistent overall goal remains active for 0.3.0 and 0.4.0.

## Plan self-review

Spec coverage: Task 1 owns statistical integrity/immutable publication; Task 2 owns actionable CLI/actual usage journeys; Task 3 owns installed artifact/real source smoke/release evidence. Shared interfaces and names are fixed above; no task edits another task's module before its review. Statistical diagnosis uses metadata rather than model-generated counts; method labels and harmonized dates require explicit annotations. No new academic accuracy claim is an acceptance criterion.
