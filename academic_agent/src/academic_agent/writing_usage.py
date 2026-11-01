"""Installed writing/domain journeys and the bounded opt-in real-backend observation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .config import atomic_json, load_config
from .demo import FIXTURES
from .filesystem import io_path
from .models import ModelJob, ModelResult, PublicationCheck, SearchPage, Work, digest
from .research_context import freeze_research
from .store import Store
from .workflow import MeteredBackend, Runner

WRITING_SCENARIO_NAMES = (
    "writing_full_journey",
    "writing_revision_recovery",
    "writing_external_pack",
    "writing_stale_rebind_cache",
    "writing_readonly_tamper",
)

NEGATIVE_DETECTION_LAYERS = {
    "W01": "controlled_exchange_semantic_review",
    "W02": "controlled_exchange_semantic_review",
    "W03": "controlled_exchange_semantic_review",
    "W04": "controlled_exchange_semantic_review",
    "W05": "controlled_exchange_semantic_review",
    "W06": "controlled_exchange_semantic_review",
    "W07": "host_capture_then_controlled_exchange_review",
    "formal_artifact_tamper": "host_manifest_validation",
    "unknown_demo_pack": "demo_protocol_refusal",
}

_QUESTION = "How do the two scheduling reports differ, and what can they support?"
_SEMANTIC_CONFLICT_MARKERS = (
    "All schedulers always improve", "400 simulated participants", "总会改进",
    "A and B both report waiting time from 10 to 8",
    "The priority rule applies to all workloads",
    "We therefore validated improvement on 2000 new jobs",
    "Both A and B reduce waiting time from 10 to 8",
    "A and B recruited 40 and 60 real participants",
    "B improves waiting time under bursty arrivals",
)
_WRITING_FILES = {
    "context.json", "outline.json", "draft.json", "review.json", "revision.json",
    "raw-edit.bin", "report.md", "references.bib", "evidence.json", "manifest.json",
}
_DOMAIN_FILES = {
    "context.json", "domain.json", "domain-review.json", "comparison.md",
    "references.bib", "evidence.json", "manifest.json",
}


class ActualCallCapExceeded(RuntimeError):
    """Raised before an uncached reservation would exceed the opt-in cap."""


@dataclass
class ActualCallLedger:
    limit: int
    requested_model: str | None
    reservations: int = 0
    attempts: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)


class _CappedStore:
    def __init__(self, store: Store, ledger: ActualCallLedger):
        self._store = store
        self._ledger = ledger

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def start_job(self, kind: str, work_id: str, input_hash: str, backend: str,
                  max_calls: int, timeout: int = 300):
        job_id = "job_" + digest(f"{kind}:{work_id}:{input_hash}:{backend}")[:32]
        existing = next((row for row in self._store.jobs() if row["id"] == job_id), None)
        if existing is not None and existing["status"] == "succeeded":
            return self._store.start_job(kind, work_id, input_hash, backend, max_calls, timeout)
        if self._ledger.reservations >= self._ledger.limit:
            raise ActualCallCapExceeded(
                "The four-call acceptance cap blocked a new reservation before dispatch"
            )
        ticket = self._store.start_job(kind, work_id, input_hash, backend, max_calls, timeout)
        if not ticket.cached:
            self._ledger.reservations += 1
            self._ledger.events.append({
                "event": "reservation", "job_id": ticket.id, "attempt_id": ticket.attempt_id,
                "kind": kind, "work_id": work_id, "input_hash": input_hash,
                "backend": backend, "ordinal": self._ledger.reservations,
            })
        return ticket


class _AttemptBackend:
    def __init__(self, backend: Any, ledger: ActualCallLedger):
        self._backend = backend
        self._ledger = ledger
        self.name = backend.name

    def run(self, job: ModelJob) -> ModelResult:
        self._ledger.attempts += 1
        event = {
            "event": "backend_attempt", "ordinal": self._ledger.attempts,
            "task": job.payload.get("task"), "kind": job.kind,
            "requested_model": self._ledger.requested_model,
        }
        self._ledger.events.append(event)
        result = self._backend.run(job)
        event.update({
            "result_status": result.status,
            "observed_backend": result.backend,
            "observed_model": None,
            "usage": result.usage,
            "error": result.error,
        })
        return result


def install_actual_call_cap(runner: Runner, *, max_new_calls: int) -> ActualCallLedger:
    """Install a cap outside MeteredBackend so it guards Store.start_job before reservation."""
    if max_new_calls != 4:
        raise ValueError("Live writing acceptance requires exactly 4 new calls")
    ledger = ActualCallLedger(limit=max_new_calls, requested_model=runner.config.model)
    backend = _AttemptBackend(runner.backend.backend, ledger)
    store = _CappedStore(runner.store, ledger)
    runner.backend = MeteredBackend(backend, store, runner.config)
    return ledger


def write_external_pack(directory: Path) -> Path:
    """Write the controlled data-only pack outside the installed package."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "external-domain.json"
    atomic_json(path, {
        "schema_version": 1,
        "contract": "writing-domain-1",
        "name": "external_simulation_boundaries",
        "version": "1",
        "terminology": [{
            "term": "observation unit",
            "meaning": "The unit explicitly observed in the source material.",
        }],
        "search_hints": ["simulation scope and arrival conditions"],
        "screening_hints": ["Retain negative results and transfer limits."],
        "fields": [
            {"id": "observation_unit", "label": "Observation unit",
             "description": "Reported unit of observation.", "value_type": "text", "unit": None},
            {"id": "arrival_pattern", "label": "Arrival pattern",
             "description": "Concrete arrival distribution or pattern reported by the source.",
             "value_type": "text", "unit": None},
            {"id": "transfer_limit", "label": "Transfer limit",
             "description": "Reported limit on transfer beyond the studied simulation.",
             "value_type": "text", "unit": None},
        ],
        "comparison_columns": ["transfer_limit", "arrival_pattern", "observation_unit"],
        "evaluation_criteria": [{
            "id": "scope_transfer", "text": "Keep simulation scope and transfer limits explicit."
        }],
        "writing_structure": [{
            "id": "boundary_argument",
            "title": "Boundary argument",
            "purpose": "Compare reported scope and transfer limits before proposing transfer.",
        }],
    })
    return path


def _data(command: dict[str, Any]) -> dict[str, Any]:
    value = command.get("parsed_json")
    return value if isinstance(value, dict) else {}


def _output_directory(project: Path, command: dict[str, Any]) -> Path:
    output = _data(command).get("output")
    if not isinstance(output, str) or not output:
        raise ValueError("Command did not return a concrete output directory")
    supplied = Path(output)
    path = supplied.resolve() if supplied.is_absolute() else (project / supplied).resolve()
    if not path.is_relative_to(project.resolve()) or not path.is_dir():
        raise ValueError("Command output directory is missing or outside its project")
    return path


def _json(path: Path) -> Any:
    return json.loads(io_path(path).read_text(encoding="utf-8"))


def _file_hashes(root: Path, names: set[str] | None = None) -> dict[str, str]:
    paths = [path for path in root.rglob("*") if path.is_file()]
    if names is not None:
        paths = [path for path in paths if path.name in names]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(io_path(path).read_bytes()).hexdigest()
        for path in paths
    }


def _mixed_provenance_matches(
        child: dict[str, dict[str, Any]], parent: dict[str, dict[str, Any]],
        revision: dict[str, Any], *, parent_path: str, parent_manifest_hash: str,
        human_author: str) -> bool:
    """Require genuine child-level human/model provenance with verified reuse pins."""
    changed = set(revision.get("changed_ids", []))
    reused = set(revision.get("reused_from", {}))
    if not changed or not reused or changed | reused != set(child) or changed & reused:
        return False
    if not all(
        child.get(block_id, {}).get("origin") == "human"
        and child[block_id].get("author") == human_author
        and bool(child[block_id].get("created_at"))
        for block_id in changed
    ):
        return False
    inherited_fields = ("origin", "author", "created_at", "backend", "model_identifier")
    for block_id in reused:
        child_item = child.get(block_id, {})
        parent_item = parent.get(block_id, {})
        pin = revision["reused_from"].get(block_id, {})
        if child_item.get("origin") != "model":
            return False
        if any(child_item.get(field) != parent_item.get(field) for field in inherited_fields):
            return False
        expected_path = parent_item.get("source_version_path") or parent_path
        expected_hash = parent_item.get("source_manifest_hash") or parent_manifest_hash
        if (child_item.get("source_version_path") != expected_path
                or child_item.get("source_manifest_hash") != expected_hash
                or pin.get("version_path") != parent_path
                or pin.get("manifest_hash") != parent_manifest_hash):
            return False
    return True


def _pure_reorder_matches(
        parent_blocks: list[dict[str, Any]], reviewed_blocks: list[dict[str, Any]],
        pending_revision: dict[str, Any], reviewed_revision: dict[str, Any],
        pending_review: dict[str, Any], reviewed_review: dict[str, Any]) -> bool:
    """Recognize an isolated reorder with stable blocks/units and ordered full review."""
    parent_by_id = {block["id"]: block for block in parent_blocks}
    reviewed_by_id = {block["id"]: block for block in reviewed_blocks}
    parent_order = [block["id"] for block in parent_blocks]
    reviewed_order = [block["id"] for block in reviewed_blocks]
    stable_blocks = (
        set(parent_by_id) == set(reviewed_by_id)
        and parent_order != reviewed_order
        and all(parent_by_id[block_id] == reviewed_by_id[block_id]
                for block_id in parent_by_id)
    )
    expected_units = [
        unit["id"] for block in reviewed_blocks for unit in block.get("units", [])
    ]
    actual_units = [item.get("unit_id") for item in reviewed_review.get("items", [])]
    empty_change_fields = all(
        pending_revision.get(field) == [] and reviewed_revision.get(field) == []
        for field in ("changed_ids", "added_ids", "deleted_ids", "pending_ids")
    )
    return (
        stable_blocks and empty_change_fields
        and set(pending_revision.get("reused_from", {})) == set(parent_by_id)
        and set(reviewed_revision.get("reused_from", {})) == set(parent_by_id)
        and pending_revision.get("reordered") is True
        and reviewed_revision.get("reordered") is True
        and pending_revision.get("stage") == "pending"
        and reviewed_revision.get("stage") == "reviewed"
        and pending_review == {"items": []}
        and actual_units == expected_units
    )


def _project_tree_snapshot(root: Path) -> dict[str, Any]:
    """Snapshot every project file byte plus directory topology, including empty dirs."""
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    directories = [
        path.relative_to(root).as_posix() for path in entries if path.is_dir()
    ]
    files: dict[str, dict[str, Any]] = {}
    for path in entries:
        if path.is_file():
            raw = io_path(path).read_bytes()
            files[path.relative_to(root).as_posix()] = {
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            }
    return {"directories": directories, "files": files}


def _readonly_snapshot_result(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare full project snapshots with only documented SQLite lock bookkeeping allowed."""
    before_files = before["files"]
    after_files = after["files"]
    before_names = set(before_files)
    after_names = set(after_files)
    added = sorted(after_names - before_names)
    removed = sorted(before_names - after_names)
    changed = sorted(
        name for name in before_names & after_names
        if before_files[name] != after_files[name]
    )
    empty_sha = hashlib.sha256(b"").hexdigest()
    allowed_added = [
        name for name in added
        if name == "research.db-shm"
        or (name == "research.db-wal"
            and after_files[name] == {"bytes": 0, "sha256": empty_sha})
    ]
    allowed_changed = [name for name in changed if name == "research.db-shm"]
    unexpected = {
        "directories_changed": before["directories"] != after["directories"],
        "added": sorted(set(added) - set(allowed_added)),
        "removed": removed,
        "changed": sorted(set(changed) - set(allowed_changed)),
    }
    return {
        "passed": not unexpected["directories_changed"]
        and not unexpected["added"] and not unexpected["removed"]
        and not unexpected["changed"],
        "sqlite_housekeeping": {
            "added": allowed_added, "changed": allowed_changed,
            "scope": "research.db-shm lock index and absent-to-empty research.db-wal only",
        },
        "unexpected": unexpected,
    }


def _package_payload_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    """Hash all package payload files, excluding interpreter-generated bytecode caches."""
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if (not path.is_file() or "__pycache__" in relative.parts
                or path.suffix.lower() in {".pyc", ".pyo"}):
            continue
        raw = io_path(path).read_bytes()
        files[relative.as_posix()] = {
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }
    return files


def _check_files(recorder: Any, command: dict[str, Any], directory: Path,
                 expected: set[str], label: str) -> None:
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    recorder.check(command, label, actual == expected, sorted(expected), sorted(actual))


def _prepare_original_project(recorder: Any, scenario: dict[str, Any], projects: Path) -> tuple[Path, list[str]]:
    project = projects / "writing-original"
    inputs = recorder.output / "writing-inputs"
    inputs.mkdir(exist_ok=True)
    ris = inputs / "original-a-b.ris"
    lines: list[str] = []
    for title, _ in FIXTURES:
        lines.extend([
            "TY  - JOUR", f"TI  - {title}", "AU  - Academic Agent original fixture",
            "PY  - 2026", "ER  -", "",
        ])
    ris.write_text("\n".join(lines), encoding="utf-8")
    recorder.project_command(
        project, "init", "--title", "0.4 installed writing journey", "--backend", "demo",
        "--max-model-calls", "60",
    )
    initial_config = load_config(project)
    initial_config.max_input_chars = 500_000
    atomic_json(project / "project.json", initial_config.model_dump())
    imported = recorder.project_command(project, "import", str(ris))
    rows = _data(imported).get("imported", [])
    work_ids = [item.get("id") for item in rows if isinstance(item, dict) and item.get("id")]
    recorder.check(imported, "原创 RIS 实际导入 A/B 两篇材料", len(work_ids) == 2, 2, len(work_ids))
    if len(work_ids) != 2:
        return project, work_ids

    work_ids = [item.id for item in sorted(
        (Store(project).work(work_id) for work_id in work_ids), key=lambda item: item.title
    )]
    for index, ((_, text), work_id) in enumerate(zip(FIXTURES, work_ids), 1):
        source = inputs / f"study-{index}.txt"
        source.write_text(text, encoding="utf-8")
        recorder.project_command(project, "attach", work_id, str(source))
    recorder.project_command(
        project, "screen", *work_ids, "--decision", "include", "--reason",
        "Explicitly included by the synthetic demo command",
    )
    read = recorder.project_command(project, "read")
    review = recorder.project_command(project, "review")
    recorder.check(read, "两篇完整原文均生成六条逐篇证据陈述",
                   [item.get("claims") for item in _data(read).get("results", [])] == [6, 6],
                   [6, 6], [item.get("claims") for item in _data(read).get("results", [])])
    recorder.check(review, "两篇阅读卡均完成独立逐条复核",
                   all(item.get("status") == "ok" for item in _data(review).get("results", []))
                   and len(_data(review).get("results", [])) == 2,
                   "2 results with status=ok", _data(review).get("results"))
    exported = recorder.project_command(project, "export")
    recorder.check(exported, "来源阅读和复核后生成实际引用/证据导出",
                   Path(_data(exported).get("export", "")).is_dir(),
                   "concrete export directory", _data(exported).get("export"))
    recorder.check(scenario, "写作项目使用固定显式初始预算和完整上下文上限且未提高",
                   (load_config(project).max_model_calls, load_config(project).max_input_chars)
                   == (60, 500_000),
                   {"max_model_calls": 60, "max_input_chars": 500_000},
                   {"max_model_calls": load_config(project).max_model_calls,
                    "max_input_chars": load_config(project).max_input_chars})
    return project, work_ids


def _selected_args(work_ids: list[str]) -> tuple[str, ...]:
    return tuple(part for work_id in work_ids for part in ("--work-id", work_id))


def _scenario_full(recorder: Any, scenario: dict[str, Any], projects: Path,
                   shared: dict[str, Any]) -> None:
    project, work_ids = _prepare_original_project(recorder, scenario, projects)
    if len(work_ids) != 2:
        return
    trends = recorder.project_command(project, "trends", "--date-basis", "record_year")
    trends_path = Path(_data(trends).get("trends", ""))
    config = load_config(project)
    config.backend = "exchange"
    atomic_json(project / "project.json", config.model_dump())
    innovation, innovation_receipts = _exchange_command(
        recorder, project, ("innovate", *_selected_args(work_ids)), 2, _external_response,
    )
    candidate_ids = _data(innovation).get("candidate_ids", [])
    recorder.check(innovation, "候选生成返回一个实际可核验 ID", len(candidate_ids) == 1, 1, candidate_ids)
    if len(candidate_ids) != 1:
        return
    candidate_id = candidate_ids[0]
    candidate_path = _output_directory(project, innovation)
    assessment, assessment_receipts = _exchange_command(
        recorder, project, ("assess-innovation", candidate_id), 2, _external_response,
    )
    assessment_path = _output_directory(project, assessment)
    domain, domain_receipts = _exchange_command(
        recorder, project,
        ("domain-extract", "--question", _QUESTION, *_selected_args(work_ids),
         "--domain", "builtin:scheduling"),
        4, _external_response,
    )
    domain_path = _output_directory(project, domain)
    _check_files(recorder, domain, domain_path, _DOMAIN_FILES, "领域版本包含闭包文件集")
    outline, outline_receipts = _exchange_command(
        recorder, project,
        ("writing-outline", "--question", _QUESTION, *_selected_args(work_ids),
         "--domain-version", str(domain_path), "--trends", str(trends_path),
         "--candidate-version", str(candidate_path), "--candidate-id", candidate_id,
         "--assessment-version", str(assessment_path)),
        2, _external_response,
    )
    outline_path = _output_directory(project, outline)
    draft, draft_receipts = _exchange_command(
        recorder, project, ("writing-draft", str(outline_path)), 2, _external_response,
    )
    draft_path = _output_directory(project, draft)
    _check_files(recorder, draft, draft_path, _WRITING_FILES, "写作版本包含闭包文件集")
    draft_data = _json(draft_path / "draft.json")
    paragraph = "".join(
        unit["text"] for block in draft_data["blocks"] if block["role"] == "paragraph"
        for unit in block["units"]
    )
    recorder.check(draft, "正文实际综合 A 10→8 与 B 10→11、突发到达及模拟边界",
                   all(phrase in paragraph for phrase in (
                       "10 to 8", "10 to 11", "bursty arrivals", "simulated jobs",
                       "do not establish effectiveness across real systems",
                   )),
                   "opposed numbers, conditions and simulation limit", paragraph)
    references = (draft_path / "references.bib").read_text(encoding="utf-8")
    evidence = _json(draft_path / "evidence.json")
    context = _json(draft_path / "context.json")["context"]
    source_context = evidence["source_context"]
    cited_sources = [source for unit in evidence["units"] for reference in unit["references"]
                     for source in reference.get("sources", [])]
    chains_valid = all(
        source["work_id"] in source_context["work_ids"]
        and source_context["documents"][source["document_id"]]["work_id"] == source["work_id"]
        and source_context["claims"][source["claim_id"]]["work_id"] == source["work_id"]
        and source_context["decisions"][source["claim_id"]]["document_id"]
        == source["document_id"]
        and source_context["decisions"][source["claim_id"]]["verdict"] == "supported"
        and set(source["evidence_ids"])
        <= set(source_context["decisions"][source["claim_id"]]["evidence_ids"])
        for source in cited_sources
    )
    citekeys = [item["citekey"] for item in context["research"]["works"].values()]
    report = (draft_path / "report.md").read_text(encoding="utf-8")
    source_links = [
        f"[@{context['research']['works'][work_id]['citekey']}]"
        f"(../../../{quote(document['source_path'], safe='/')})"
        for work_id in work_ids
        for document in context["research"]["documents"].values()
        if document["work_id"] == work_id
    ]
    computation = context["trend"]["computation"]
    recorder.check(draft, "正文引用键与逐块来源/复核审计均可见",
                   references.count("@") == 2 and bool(evidence.get("block_provenance"))
                   and bool(evidence.get("units")) and cited_sources and chains_valid
                   and all(f"{{{key}," in references for key in citekeys)
                   and all(link in report for link in source_links)
                   and all(
                       f"`{evidence_id}` (page {source_context['evidence'][evidence_id]['page']}, "
                       f"chars {source_context['evidence'][evidence_id]['start']}:"
                       f"{source_context['evidence'][evidence_id]['end']})" in report
                       for source in cited_sources for evidence_id in source["evidence_ids"]
                   ),
                   "2 BibTeX entries plus visible per-unit source/evidence Markdown chains", {
                       "bib_entries": references.count("@"), "evidence_keys": sorted(evidence),
                       "cited_sources": cited_sources, "chains_valid": chains_valid,
                       "source_links": source_links,
                   })
    recorder.check(draft, "趋势统计保留实际分母、算法版本、公式和输入作品 ID",
                   computation["algorithm_version"] == "corpus-trends-1"
                   and computation["annual"] == [{
                       "schema_version": 1, "year": 2026, "count": 2,
                       "denominator": 2, "share": 1.0, "current_year": True,
                   }]
                   and "annual_count[y] /" in computation["formulas"]["annual_share"]
                   and set(computation["work_ids"]["eligible"]) == set(work_ids),
                   "denominator=2, algorithm/formula and both work IDs", computation)
    status = recorder.project_command(project, "writing-status")
    recorder.check(status, "只读状态分别显示当前性、manuscript 范围与 reviewed 阶段",
                   any(version.get("current") and version.get("extent") == "manuscript"
                       and version.get("stage") == "reviewed" and version.get("review_complete")
                       for document in _data(status).get("documents", [])
                       for version in document.get("versions", [])),
                   "a current reviewed manuscript", _data(status).get("documents"))
    base_outline, base_outline_receipts = _exchange_command(
        recorder, project,
        ("writing-outline", "--question", _QUESTION, *_selected_args(work_ids),
         "--domain-version", str(domain_path)),
        2, _external_response,
    )
    base_outline_path = _output_directory(project, base_outline)
    base_draft, base_draft_receipts = _exchange_command(
        recorder, project, ("writing-draft", str(base_outline_path)), 2, _external_response,
    )
    base_draft_path = _output_directory(project, base_draft)
    statistic_project = recorder.output / "statistic-render-project"
    shutil.copytree(io_path(project), io_path(statistic_project))
    statistic_domain = statistic_project / domain_path.relative_to(project)
    statistic_trends = statistic_project / trends_path.relative_to(project)
    statistic_outline, statistic_receipts = _exchange_command(
        recorder, statistic_project,
        ("writing-outline", "--question", _QUESTION, *_selected_args(work_ids),
         "--domain-version", str(statistic_domain), "--trends", str(statistic_trends)),
        2, _statistic_outline_response,
    )
    statistic_path = _output_directory(statistic_project, statistic_outline)
    statistic_report = (statistic_path / "report.md").read_text(encoding="utf-8")
    recorder.check(
        statistic_outline,
        "冻结统计脚注显示计算路径、pin 及全部可用原始来源链接",
        "Frozen statistic `counts.eligible`" in statistic_report
        and "trend pin `" in statistic_report
        and all(link in statistic_report for link in source_links),
        "frozen statistic path/pin plus both original source links",
        {"source_links": source_links, "report": statistic_report},
    )
    controlled = (innovation_receipts + assessment_receipts + domain_receipts + outline_receipts
                  + draft_receipts + base_outline_receipts + base_draft_receipts
                  + statistic_receipts)
    recorder.check(scenario, "RIS provenance 保持原样且所有新增语义阶段使用 fingerprinted Exchange preset",
                   len(controlled) == 18 and all(item["fingerprint"] for item in controlled)
                   and all(Store(project).work(work_id).provenance for work_id in work_ids),
                   "18 controlled receipts and authentic import provenance", controlled)
    shared.update({
        "project": project, "work_ids": work_ids, "trends": trends_path,
        "candidate_id": candidate_id, "candidate": candidate_path,
        "assessment": assessment_path, "domain": domain_path,
        "outline": outline_path, "draft": draft_path,
        "base_outline": base_outline_path, "base_draft": base_draft_path,
    })


def _scenario_revision(recorder: Any, scenario: dict[str, Any], shared: dict[str, Any]) -> None:
    project: Path = shared["project"]
    parent: Path = shared["draft"]
    edit = recorder.output / "submitted-human-edit.json"
    recorder.project_command(project, "writing-export-edit", str(parent), "--output", str(edit))
    payload = _json(edit)
    payload["outline"][0]["text"] = "All schedulers always improve"
    paragraph = next(block for block in payload["paragraphs"] if block["role"] == "paragraph")
    paragraph["text"] = "Study A used 400 simulated participants.\r\n因此两种策略总会改进。🙂 e\u0301"
    raw = ("\ufeff" + json.dumps(payload, ensure_ascii=False, indent=2)).replace("\n", "\r\n").encode("utf-8")
    edit.write_bytes(raw)
    imported = recorder.project_command(
        project, "writing-import", str(parent), str(edit), "--author", "Acceptance Researcher",
        "--reason", "Exercise exact human numeric correction handling",
    )
    pending = _output_directory(project, imported)
    pending_revision = _json(pending / "revision.json")
    pending_review = _json(pending / "review.json")
    recorder.check(imported, "人工文件在模型前按原 bytes/hash 和错误数字保存为 pending",
                   (pending / "raw-edit.bin").read_bytes() == raw
                   and pending_revision.get("input_sha256") == hashlib.sha256(raw).hexdigest()
                   and pending_revision.get("stage") == "pending"
                   and pending_review == {"items": []}
                   and "400 simulated participants" in (pending / "raw-edit.bin").read_text(
                       encoding="utf-8-sig"),
                   "exact raw/hash/pending/empty review/wrong number", {
                       "raw_equal": (pending / "raw-edit.bin").read_bytes() == raw,
                       "revision": pending_revision, "review": pending_review,
                   })
    parent_hashes = _file_hashes(parent)
    budget_before = Store(project).budget()["consumed_or_reserved_calls"]
    reviewed, review_receipts = _exchange_command(
        recorder, project, ("writing-review", str(pending)), 2, _external_response,
    )
    reviewed_path = _output_directory(project, reviewed)
    decisions = _json(reviewed_path / "review.json")["items"]
    recorder.check(reviewed, "标题及含错误数字和 connector 的正文均进入整稿复核并被判为冲突",
                   sum(item["decision"] == "contradicted" for item in decisions) == 2,
                   "the 2 edited conflicting units are contradicted", decisions)
    recorder.check(reviewed, "复核子版本保留人类原 bytes 且不改变父版本",
                   (reviewed_path / "raw-edit.bin").read_bytes() == raw
                   and _file_hashes(parent) == parent_hashes,
                   "raw preserved and immutable parent", {
                       "raw_equal": (reviewed_path / "raw-edit.bin").read_bytes() == raw,
                       "parent_unchanged": _file_hashes(parent) == parent_hashes,
                   })
    reviewed_evidence = _json(reviewed_path / "evidence.json")
    provenance = {item["block_id"]: item for item in reviewed_evidence["block_provenance"]}
    parent_provenance = {
        item["block_id"]: item for item in _json(parent / "evidence.json")["block_provenance"]
    }
    reviewed_revision = _json(reviewed_path / "revision.json")
    all_child_ids = set(provenance)
    recorder.check(reviewed, "标题内容变化使整稿失效，同时仅变更块标为 human 并保留其余内容作者",
                   set(pending_revision["pending_ids"]) == all_child_ids
                   and _mixed_provenance_matches(
                       provenance, parent_provenance, reviewed_revision,
                       parent_path=str(parent.resolve()),
                       parent_manifest_hash=_json(parent / "manifest.json")["manifest_hash"],
                       human_author="Acceptance Researcher",
                   )
                   and reviewed_evidence["review_provenance"]["source"]
                   == "independent_model_call",
                   "whole-document invalidation with separate content provenance and review",
                   {"pending_revision": pending_revision,
                    "reviewed_revision": reviewed_revision,
                    "child_provenance": list(provenance.values()),
                    "review_provenance": reviewed_evidence["review_provenance"]})
    repeated = recorder.project_command(project, "writing-review", str(pending))
    budget_after = Store(project).budget()["consumed_or_reserved_calls"]
    repeated_review = _json(_output_directory(project, repeated) / "review.json")
    recorder.check(repeated, "重复 resume 复用成功划分与整稿复核且不新增成功调用",
                   repeated_review == _json(reviewed_path / "review.json")
                   and budget_after == budget_before + 2,
                   {"review": "identical", "calls": budget_before + 2},
                   {"review_identical": repeated_review == _json(reviewed_path / "review.json"),
                    "calls": budget_after})

    mixed_edit = recorder.output / "mixed-provenance-body-only.json"
    recorder.project_command(
        project, "writing-export-edit", str(parent), "--output", str(mixed_edit),
    )
    mixed_value = _json(mixed_edit)
    changed_block = mixed_value["paragraphs"][0]
    changed_block["text"] = "We therefore validated improvement on 2000 new jobs"
    mixed_raw = json.dumps(mixed_value, ensure_ascii=False, indent=2).encode("utf-8")
    mixed_edit.write_bytes(mixed_raw)
    mixed_pending_command = recorder.project_command(
        project, "writing-import", str(parent), str(mixed_edit),
        "--author", "Mixed Provenance Researcher",
        "--reason", "Change one body block while retaining verified model blocks",
    )
    mixed_pending = _output_directory(project, mixed_pending_command)
    mixed_pending_revision = _json(mixed_pending / "revision.json")
    mixed_reviewed_command, mixed_receipts = _exchange_command(
        recorder, project, ("writing-review", str(mixed_pending)), 2, _external_response,
    )
    mixed_reviewed = _output_directory(project, mixed_reviewed_command)
    mixed_revision = _json(mixed_reviewed / "revision.json")
    mixed_evidence = _json(mixed_reviewed / "evidence.json")
    mixed_provenance = {
        item["block_id"]: item for item in mixed_evidence["block_provenance"]
    }
    parent_manifest_hash = _json(parent / "manifest.json")["manifest_hash"]
    from .writing import load_writing_version

    load_writing_version(mixed_reviewed)
    recorder.check(
        mixed_reviewed_command,
        "单一正文修改的子版本同时保留精确 human 与经父 manifest 验证的 model provenance",
        mixed_pending_revision["changed_ids"] == [changed_block["id"]]
        and mixed_pending_revision["pending_ids"] == [changed_block["id"]]
        and _mixed_provenance_matches(
            mixed_provenance, parent_provenance, mixed_revision,
            parent_path=str(parent.resolve()), parent_manifest_hash=parent_manifest_hash,
            human_author="Mixed Provenance Researcher",
        )
        and mixed_evidence["review_provenance"]["source"] == "independent_model_call",
        "one human changed block plus exact verified retained model blocks in the child",
        {
            "pending_revision": mixed_pending_revision,
            "reviewed_revision": mixed_revision,
            "child_provenance": list(mixed_provenance.values()),
            "verified_parent_provenance": list(parent_provenance.values()),
        },
    )
    recorder.check(
        mixed_pending_command,
        "混合 provenance 修订仅划分变更块并以两个 fingerprinted Exchange 调用复核整稿",
        [item["task"] for item in mixed_receipts]
        == ["writing-segment-v1", "writing-review-v1"],
        ["writing-segment-v1", "writing-review-v1"], mixed_receipts,
    )
    recorder.check(scenario, "负例检测层逐项记录，未把预设协议称为模型准确率",
                   set(NEGATIVE_DETECTION_LAYERS) >= {"W01", "W02", "W03", "W04", "W05", "W06", "W07"},
                   "W01-W07 detection layers", NEGATIVE_DETECTION_LAYERS)
    recorder.check(scenario, "错误数字的划分与复核响应绑定两个 Exchange fingerprints",
                   len(review_receipts) == 2, 2, review_receipts)

    semantic_edit = recorder.output / "controlled-w01-w06.json"
    semantic_export = recorder.project_command(
        project, "writing-export-edit", str(parent), "--output", str(semantic_edit),
    )
    semantic = _json(semantic_edit)
    wrong_text = {
        "W05": "A and B both report waiting time from 10 to 8",
        "W03": "The priority rule applies to all workloads",
        "W04": "We therefore validated improvement on 2000 new jobs",
        "W01": "Both A and B reduce waiting time from 10 to 8",
        "W02": "A and B recruited 40 and 60 real participants",
        "W06": "B improves waiting time under bursty arrivals",
    }
    semantic["outline"][0]["text"] = wrong_text["W05"]
    semantic["outline"][1]["text"] = wrong_text["W03"]
    semantic["paragraphs"][0]["text"] = wrong_text["W04"]
    semantic["paragraphs"][1]["text"] = wrong_text["W01"]
    semantic["paragraphs"].extend([
        {"id": "new:w02", "role": "paragraph", "level": None, "text": wrong_text["W02"]},
        {"id": "new:w06", "role": "paragraph", "level": None, "text": wrong_text["W06"]},
    ])
    semantic_raw = json.dumps(semantic, ensure_ascii=False, indent=2).encode("utf-8")
    semantic_edit.write_bytes(semantic_raw)
    semantic_pending_command = recorder.project_command(
        project, "writing-import", str(parent), str(semantic_edit),
        "--author", "Controlled W01-W06 fixture",
        "--reason", "Exercise six independent semantic-review counterexamples",
    )
    semantic_pending = _output_directory(project, semantic_pending_command)
    recorder.check(semantic_pending_command, "W01-W06 原始错误文字在受控复核前完整保存",
                   (semantic_pending / "raw-edit.bin").read_bytes() == semantic_raw
                   and all(value in semantic_raw.decode("utf-8") for value in wrong_text.values()),
                   wrong_text, {"raw_sha256": hashlib.sha256(semantic_raw).hexdigest()})
    semantic_second_edit = recorder.output / "controlled-w01-w06-second.json"
    recorder.project_command(
        project, "writing-export-edit", str(semantic_pending), "--output", str(semantic_second_edit),
    )
    semantic_second = _json(semantic_second_edit)
    semantic_second["paragraphs"][0]["text"] += " A second unfinished correction remains required."
    semantic_second_edit.write_text(
        json.dumps(semantic_second, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    semantic_second_command = recorder.project_command(
        project, "writing-import", str(semantic_pending), str(semantic_second_edit),
        "--author", "Controlled W01-W06 second author",
        "--reason", "Revise one block while all prior segmentation targets remain unfinished",
    )
    semantic_second_pending = _output_directory(project, semantic_second_command)
    first_pending_ids = set(_json(semantic_pending / "revision.json")["pending_ids"])
    second_pending_revision = _json(semantic_second_pending / "revision.json")
    recorder.check(
        semantic_second_command,
        "二次导入保留全部存活待划分目标并加入当前修改，未提前恢复 reviewed",
        set(second_pending_revision["pending_ids"]) == first_pending_ids
        and second_pending_revision["stage"] == "pending"
        and _json(semantic_second_pending / "review.json") == {"items": []},
        "all surviving first-import targets remain pending with empty review",
        {"first_pending_ids": sorted(first_pending_ids), "second": second_pending_revision},
    )
    semantic_reviewed_command, semantic_receipts = _exchange_command(
        recorder, project, ("writing-review", str(semantic_second_pending)), 2, _external_response,
    )
    semantic_reviewed = _output_directory(project, semantic_reviewed_command)
    semantic_artifact = _json(semantic_reviewed / "review.json")
    recorder.check(semantic_reviewed_command,
                   "W01-W06 的标题、connector 与正文均进入完整有序复核并保留冲突决定",
                   len(semantic_artifact["items"]) == 6
                   and all(item["decision"] == "contradicted"
                           for item in semantic_artifact["items"]),
                   "6 controlled contradicted decisions", semantic_artifact)
    recorder.check(semantic_export, "W01-W06 使用明确标记的 fingerprinted Exchange 预设而非模型准确率",
                   len(semantic_receipts) == 2
                   and all(item["protocol"] == "controlled_exchange_preset"
                           for item in semantic_receipts),
                   "segmentation and independent review receipts", semantic_receipts)

    identity_edit = recorder.output / "add-delete-reorder.json"
    recorder.project_command(
        project, "writing-export-edit", str(semantic_reviewed), "--output", str(identity_edit),
    )
    identity_document = _json(identity_edit)
    deleted_id = identity_document["outline"].pop(1)["id"]
    identity_document["paragraphs"].insert(1, {
        "id": "new:ordered_boundary", "role": "paragraph", "level": None,
        "text": "A bounded added paragraph retains the two opposed simulated results.",
    })
    identity_document["paragraphs"] = [
        identity_document["paragraphs"][0],
        *reversed(identity_document["paragraphs"][1:]),
    ]
    identity_edit.write_text(json.dumps(identity_document, ensure_ascii=False, indent=4), encoding="utf-8")
    identity_pending_command = recorder.project_command(
        project, "writing-import", str(semantic_reviewed), str(identity_edit),
        "--author", "Identity acceptance author", "--reason", "Add delete and reorder blocks",
        "--delete-id", deleted_id,
    )
    identity_pending = _output_directory(project, identity_pending_command)
    identity_reviewed_command, identity_receipts = _exchange_command(
        recorder, project, ("writing-review", str(identity_pending)), 2, _external_response,
    )
    identity_reviewed = _output_directory(project, identity_reviewed_command)
    identity_revision = _json(identity_reviewed / "revision.json")
    recorder.check(identity_reviewed_command, "add/delete/reorder 保留稳定 ID、显式删除并按新顺序全文复核",
                   identity_revision["deleted_ids"] == [deleted_id]
                   and identity_revision["added_ids"]
                   and identity_revision["reordered"] is True
                   and identity_revision["stage"] == "reviewed"
                   and len(_json(identity_reviewed / "review.json")["items"])
                   == sum(len(block["units"]) for section in ("outline.json", "draft.json")
                          for block in _json(identity_reviewed / section)["blocks"]),
                   "explicit deletion, deterministic addition, reorder, complete review",
                   identity_revision)
    recorder.check(identity_pending_command, "身份修订的两次受控调用分别划分新增块并独立全文复核",
                   [item["task"] for item in identity_receipts]
                   == ["writing-segment-v1", "writing-review-v1"],
                   ["writing-segment-v1", "writing-review-v1"], identity_receipts)

    reorder_edit = recorder.output / "pure-reorder-only.json"
    recorder.project_command(
        project, "writing-export-edit", str(parent), "--output", str(reorder_edit),
    )
    reorder_value = _json(reorder_edit)
    reorder_value["paragraphs"] = list(reversed(reorder_value["paragraphs"]))
    reorder_edit.write_text(
        json.dumps(reorder_value, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    reorder_parent_hashes = _file_hashes(parent)
    parent_blocks = [
        *_json(parent / "outline.json")["blocks"], *_json(parent / "draft.json")["blocks"],
    ]
    reorder_pending_command = recorder.project_command(
        project, "writing-import", str(parent), str(reorder_edit),
        "--author", "Pure reorder acceptance author",
        "--reason", "Reorder existing body blocks without changing their content",
    )
    reorder_pending = _output_directory(project, reorder_pending_command)
    reorder_pending_revision = _json(reorder_pending / "revision.json")
    reorder_pending_review = _json(reorder_pending / "review.json")
    reorder_second_edit = recorder.output / "pure-reorder-second-import.json"
    recorder.project_command(
        project, "writing-export-edit", str(reorder_pending), "--output", str(reorder_second_edit),
    )
    reorder_second_command = recorder.project_command(
        project, "writing-import", str(reorder_pending), str(reorder_second_edit),
        "--author", "Pure reorder second author",
        "--reason", "Retain the unfinished reorder through a formatting-only second import",
    )
    reorder_second_pending = _output_directory(project, reorder_second_command)
    reorder_second_revision = _json(reorder_second_pending / "revision.json")
    recorder.check(
        reorder_second_command,
        "纯 reorder 的二次无语义导入继续保持 pending、零划分目标及空复核",
        reorder_second_revision["stage"] == "pending"
        and reorder_second_revision["pending_ids"] == []
        and _json(reorder_second_pending / "review.json") == {"items": []},
        "pending with no segmentation targets and empty review",
        reorder_second_revision,
    )
    reorder_reviewed_command, reorder_receipts = _exchange_command(
        recorder, project, ("writing-review", str(reorder_second_pending)), 1, _external_response,
    )
    reorder_reviewed = _output_directory(project, reorder_reviewed_command)
    reorder_reviewed_revision = _json(reorder_reviewed / "revision.json")
    reorder_reviewed_review = _json(reorder_reviewed / "review.json")
    reordered_blocks = [
        *_json(reorder_reviewed / "outline.json")["blocks"],
        *_json(reorder_reviewed / "draft.json")["blocks"],
    ]
    recorder.check(
        reorder_reviewed_command,
        "纯 reorder 经二次导入后保持 block/unit IDs 与内容并按新顺序完成整稿复核",
        [block["id"] for block in parent_blocks] != [block["id"] for block in reordered_blocks]
        and {block["id"]: block for block in parent_blocks}
        == {block["id"]: block for block in reordered_blocks}
        and reorder_pending_revision["reordered"] is True
        and reorder_pending_revision["stage"] == "pending"
        and reorder_pending_review == {"items": []}
        and all(reorder_pending_revision[field] == []
                for field in ("changed_ids", "added_ids", "deleted_ids", "pending_ids"))
        and reorder_second_revision["stage"] == "pending"
        and reorder_reviewed_revision["stage"] == "reviewed"
        and [item["unit_id"] for item in reorder_reviewed_review["items"]]
        == [unit["id"] for block in reordered_blocks for unit in block["units"]]
        and _file_hashes(parent) == reorder_parent_hashes,
        "stable content/IDs, carried pending reorder, ordered full review, immutable parent",
        {
            "first_pending_revision": reorder_pending_revision,
            "second_pending_revision": reorder_second_revision,
            "reviewed_revision": reorder_reviewed_revision,
        },
    )
    recorder.check(
        reorder_pending_command,
        "纯 reorder 两次导入均跳过划分调用且最终仅用一次 fingerprinted Exchange 完成复核",
        [item["task"] for item in reorder_receipts] == ["writing-review-v1"],
        ["writing-review-v1"], reorder_receipts,
    )

    outline_parent: Path = shared["base_outline"]
    format_edit = recorder.output / "outline-format-only.json"
    recorder.project_command(
        project, "writing-export-edit", str(outline_parent), "--output", str(format_edit),
    )
    format_value = _json(format_edit)
    format_raw = ("\ufeff" + json.dumps(format_value, ensure_ascii=False, indent=4)).replace(
        "\n", "\r\n"
    ).encode("utf-8")
    format_edit.write_bytes(format_raw)
    format_command = recorder.project_command(
        project, "writing-import", str(outline_parent), str(format_edit),
        "--author", "Outline acceptance author", "--reason", "JSON formatting only",
    )
    formatted_outline = _output_directory(project, format_command)
    formatted_revision = _json(formatted_outline / "revision.json")
    recorder.check(format_command, "空正文大纲的 BOM/CRLF 格式修订保持 reviewed outline 且零待划分项",
                   formatted_revision["extent"] == "outline"
                   and formatted_revision["stage"] == "reviewed"
                   and formatted_revision["pending_ids"] == []
                   and _json(formatted_outline / "draft.json")["blocks"] == []
                   and (formatted_outline / "raw-edit.bin").read_bytes() == format_raw,
                   "reviewed outline, empty body and exact formatting bytes", formatted_revision)

    outline_edit = recorder.output / "outline-human-title.json"
    recorder.project_command(
        project, "writing-export-edit", str(outline_parent), "--output", str(outline_edit),
    )
    outline_value = _json(outline_edit)
    outline_value["outline"][0]["text"] = "Revised bounded scheduling argument"
    outline_edit.write_text(json.dumps(outline_value, ensure_ascii=False, indent=2), encoding="utf-8")
    outline_pending_command = recorder.project_command(
        project, "writing-import", str(outline_parent), str(outline_edit),
        "--author", "Outline acceptance author", "--reason", "Clarify the outline title",
    )
    outline_pending = _output_directory(project, outline_pending_command)
    outline_reviewed_command, outline_review_receipts = _exchange_command(
        recorder, project, ("writing-review", str(outline_pending)), 2, _external_response,
    )
    outline_reviewed = _output_directory(project, outline_reviewed_command)
    outline_draft_command, outline_draft_receipts = _exchange_command(
        recorder, project, ("writing-draft", str(outline_reviewed)), 2, _external_response,
    )
    outline_draft = _output_directory(project, outline_draft_command)
    outline_pending_revision = _json(outline_pending / "revision.json")
    outline_pending_provenance = {
        item["block_id"]: item
        for item in _json(outline_pending / "evidence.json")["block_provenance"]
    }
    outline_reviewed_provenance = {
        item["block_id"]: item
        for item in _json(outline_reviewed / "evidence.json")["block_provenance"]
    }
    outline_parent_provenance = {
        item["block_id"]: item
        for item in _json(outline_parent / "evidence.json")["block_provenance"]
    }
    recorder.check(outline_draft_command, "人工编辑空正文大纲保持 outline 生命周期并生成实质 manuscript",
                   outline_pending_revision["extent"] == "outline"
                   and outline_pending_revision["stage"] == "pending"
                   and _json(outline_reviewed / "revision.json")["extent"] == "outline"
                   and _json(outline_reviewed / "revision.json")["stage"] == "reviewed"
                   and _mixed_provenance_matches(
                       outline_pending_provenance, outline_parent_provenance,
                       outline_pending_revision,
                       parent_path=str(outline_parent.resolve()),
                       parent_manifest_hash=_json(outline_parent / "manifest.json")["manifest_hash"],
                       human_author="Outline acceptance author",
                   )
                   and outline_reviewed_provenance == outline_pending_provenance
                   and _json(outline_draft / "revision.json")["extent"] == "manuscript"
                   and any(block["role"] == "paragraph"
                           for block in _json(outline_draft / "draft.json")["blocks"]),
                   "pending outline -> reviewed outline -> substantive manuscript",
                   {"pending": str(outline_pending), "reviewed": str(outline_reviewed),
                    "draft": str(outline_draft)})
    recorder.check(outline_reviewed_command, "大纲复核和正文生成各使用独立 fingerprinted 受控调用",
                   len(outline_review_receipts + outline_draft_receipts) == 4,
                   "4 controlled receipts", outline_review_receipts + outline_draft_receipts)
    _scenario_context_recapture(recorder, scenario, shared)
    shared.update({"pending": pending, "reviewed_edit": reviewed_path, "submitted_raw": raw})


def _scenario_context_recapture(
        recorder: Any, scenario: dict[str, Any], shared: dict[str, Any]) -> None:
    """Exercise an installed challenge-coverage recapture without provider traffic."""
    original: Path = shared["project"]
    project = recorder.output / "context-recapture-project"
    shutil.copytree(io_path(original), io_path(project))
    candidate = project / shared["candidate"].relative_to(original)
    candidate_id: str = shared["candidate_id"]
    work_ids: list[str] = shared["work_ids"]

    from . import sources
    from .challenges import challenge_candidate
    from .domain import load_domain_version
    from .writing import load_writing_version

    configured = list(load_config(project).sources)
    saved_sources = {name: sources.SOURCES[name] for name in configured}

    def controlled_source(name: str):
        class ControlledChallengeSource:
            def __init__(self, **kwargs):
                pass

            def search(self, query: str, limit: int) -> SearchPage:
                return SearchPage(source=name, queries=[query], records=[Work(
                    title="Installed retained unread challenge",
                    doi="10.1234/installed-unread-challenge",
                )])

        return ControlledChallengeSource

    try:
        for name in configured:
            sources.SOURCES[name] = controlled_source(name)
        challenge = challenge_candidate(Runner(project), candidate_id)
    finally:
        for name, implementation in saved_sources.items():
            sources.SOURCES[name] = implementation
    work_id, = challenge["retained_work_ids"]
    runner = Runner(project, offline=True)
    work = runner.store.work(work_id)
    work.publication_checks = [
        PublicationCheck(source="controlled-installed-fixture", status="no_notice_found"),
    ]
    runner.store.update_work(work)
    source = recorder.output / "installed-unread-challenge.txt"
    source.write_text(FIXTURES[0][1], encoding="utf-8")
    runner.attach(work_id, source)
    runner.store.screen(work_id, "include", "Retain for installed challenge reading.")

    domain_args = (
        "domain-extract", "--question", _QUESTION, *_selected_args(work_ids),
        "--domain", "builtin:scheduling",
    )
    domain_command, domain_receipts = _exchange_command(
        recorder, project, domain_args, 4, _external_response,
    )
    domain = _output_directory(project, domain_command)
    outline_command, outline_receipts = _exchange_command(
        recorder, project,
        ("writing-outline", "--question", _QUESTION, *_selected_args(work_ids),
         "--domain-version", str(domain), "--candidate-version", str(candidate),
         "--candidate-id", candidate_id),
        2, _external_response,
    )
    parent = _output_directory(project, outline_command)
    parent_artifact = load_writing_version(parent)
    parent_hit, = parent_artifact.context.challenge_context.hits
    assert parent_hit.work_id == work_id and "unread" in parent_hit.statuses

    _, read_receipts = _exchange_command(
        recorder, project, ("read", "--work-id", work_id), 1, _external_response,
    )
    _, source_review_receipts = _exchange_command(
        recorder, project, ("review", "--work-id", work_id), 1, _external_response,
    )
    fresh_domain = recorder.project_command(project, *domain_args)
    fresh_domain_path = _output_directory(project, fresh_domain)
    old_domain_hash = load_domain_version(domain).pin.semantic_hash
    fresh_domain_hash = load_domain_version(fresh_domain_path).pin.semantic_hash
    edit = recorder.output / "context-recapture-edit.json"
    recorder.project_command(project, "writing-export-edit", str(parent), "--output", str(edit))
    raw = edit.read_bytes()
    imported = recorder.project_command(
        project, "writing-import", str(parent), str(edit),
        "--author", "Context recapture acceptance author",
        "--reason", "Recapture completed challenge reading against the same base domain",
        "--domain-version", str(fresh_domain_path),
    )
    pending = _output_directory(project, imported)
    pending_artifact = load_writing_version(pending)
    pending_revision = pending_artifact.revision.model_dump(mode="json")
    pending_provenance = {
        item["block_id"]: item
        for item in _json(pending / "evidence.json")["block_provenance"]
    }
    parent_manifest = _json(parent / "manifest.json")["manifest_hash"]
    recorder.check(
        imported,
        "challenge unread→ready 改变完整写作语境而 base-domain hash 相同，全部内容重新划分复核",
        old_domain_hash == fresh_domain_hash
        and parent_artifact.context.challenge_context != pending_artifact.context.challenge_context
        and pending_artifact.context.challenge_context.hits[0].statuses == ["ready"]
        and pending_revision["stage"] == "pending"
        and set(pending_revision["pending_ids"]) == set(pending_provenance)
        and all(item["origin"] == "model" for item in pending_provenance.values())
        and all(item["source_version_path"] == str(parent.resolve())
                and item["source_manifest_hash"] == parent_manifest
                for item in pending_provenance.values())
        and (pending / "raw-edit.bin").read_bytes() == raw,
        "same domain semantics, changed full context, all pending, original content provenance",
        {
            "old_domain_hash": old_domain_hash,
            "fresh_domain_hash": fresh_domain_hash,
            "parent_challenge": parent_artifact.context.challenge_context.model_dump(mode="json"),
            "pending_challenge": pending_artifact.context.challenge_context.model_dump(mode="json"),
            "pending_revision": pending_revision,
            "block_provenance": list(pending_provenance.values()),
        },
    )
    reviewed_command, reviewed_receipts = _exchange_command(
        recorder, project, ("writing-review", str(pending)), 2, _external_response,
    )
    reviewed = load_writing_version(_output_directory(project, reviewed_command))
    recorder.check(
        reviewed_command,
        "语境重捕获后以重新划分及独立整稿复核完成有序闭包",
        reviewed.revision.stage == "reviewed"
        and [item.unit_id for item in reviewed.review.items] == [
            unit.id for block in [*reviewed.outline.blocks, *reviewed.draft.blocks]
            for unit in block.units
        ],
        "reviewed with exact ordered full-unit coverage",
        reviewed.revision.model_dump(mode="json"),
    )
    receipts = (
        domain_receipts + outline_receipts + read_receipts
        + source_review_receipts + reviewed_receipts
    )
    recorder.check(
        scenario,
        "语境重捕获使用十个 fingerprinted Exchange 响应及本地受控检索，无 provider/model 调用",
        len(receipts) == 10
        and all(item["protocol"] == "controlled_exchange_preset" for item in receipts),
        "10 fingerprinted Exchange responses plus controlled in-process search adapters",
        {"exchange": receipts, "challenge": challenge, "configured_sources": configured},
    )


def _scenario_stale_rebind(recorder: Any, scenario: dict[str, Any], shared: dict[str, Any]) -> None:
    project: Path = shared["project"]
    parent: Path = shared["base_draft"]
    work_ids: list[str] = shared["work_ids"]
    from .writing import load_writing_version

    document_id = load_writing_version(parent).revision.document_id
    live_ready = recorder.output / "live-ready-project"
    screening_clone = recorder.output / "screening-stale-project"
    manual_clone = recorder.output / "manual-review-stale-project"
    shutil.copytree(io_path(project), io_path(live_ready))
    shutil.copytree(io_path(project), io_path(screening_clone))
    shutil.copytree(io_path(project), io_path(manual_clone))
    live_domain = live_ready / shared["external_domain"].relative_to(project)
    live_outline = live_ready / shared["external_outline"].relative_to(project)
    atomic_json(recorder.output / "live-inputs.json", {
        "project": str(live_ready),
        "external_pack": str(shared["external_pack"]),
        "domain_version": str(live_domain),
        "outline_version": str(live_outline),
        "max_new_calls": 4,
    })
    recorder.project_command(
        screening_clone, "screen", work_ids[0], "--decision", "exclude",
        "--reason", "Independent screening staleness probe",
    )
    screening_stale = recorder.project_command(
        screening_clone, "writing-status", "--document-id", document_id, expected=2,
    )
    recorder.check(screening_stale, "筛选决定独立变化使完整历史可读、current=false 并返回恢复 argv",
                   any(version.get("available") and not version.get("current")
                       for document in _data(screening_stale).get("documents", [])
                       for version in document.get("versions", []))
                   and isinstance(_data(screening_stale).get("suggested_argv"), list),
                   "available stale version with recovery actions", _data(screening_stale))

    manual_snapshot = freeze_research(Store(manual_clone))
    claim_id = manual_snapshot.supported_claim_ids[0]
    Runner(manual_clone, offline=True).manual_review(
        claim_id, "supported", "Independent refreshed human rationale.", "Acceptance reviewer",
    )
    manual_stale = recorder.project_command(
        manual_clone, "writing-status", "--document-id", document_id, expected=2,
    )
    recorder.check(manual_stale, "人工逐条复核独立变化使历史可读但过期并保留具体后续动作",
                   any(version.get("available") and not version.get("current")
                       for document in _data(manual_stale).get("documents", [])
                       for version in document.get("versions", []))
                   and isinstance(_data(manual_stale).get("suggested_argv"), list),
                   "available stale version after manual review with actions",
                   {"claim_id": claim_id, "status": _data(manual_stale)})
    changed = recorder.output / "source-refresh-study-1.txt"
    changed.write_text(FIXTURES[0][1] + "\n", encoding="utf-8")
    recorder.project_command(project, "attach", work_ids[0], str(changed))
    stale = recorder.project_command(project, "writing-status", expected=2)
    serialized = json.dumps(_data(stale), ensure_ascii=False)
    recorder.check(stale, "来源路径身份变化使完整历史仍可读但 current 明确过期",
                   '"available": true' in serialized.lower() and '"current": false' in serialized.lower(),
                   "available=true/current=false", _data(stale))
    config = load_config(project)
    config.backend = "exchange"
    atomic_json(project / "project.json", config.model_dump())
    _, read_receipts = _exchange_command(
        recorder, project, ("read", "--work-id", work_ids[0]), 1, _external_response,
    )
    _, source_review_receipts = _exchange_command(
        recorder, project, ("review", "--work-id", work_ids[0]), 1, _external_response,
    )
    fresh_domain, refresh_receipts = _exchange_command(
        recorder, project,
        ("domain-extract", "--question", _QUESTION, *_selected_args(work_ids),
         "--domain", "builtin:scheduling"),
        4, _external_response,
    )
    fresh_domain_path = _output_directory(project, fresh_domain)
    edit = recorder.output / "source-refresh-preserved-edit.json"
    recorder.project_command(project, "writing-export-edit", str(parent), "--output", str(edit))
    raw = edit.read_bytes()
    rejected = recorder.project_command(
        project, "writing-import", str(parent), str(edit), "--author", "Acceptance Researcher",
        "--reason", "Preserve text across source refresh", expected=2,
    )
    gaps = _data(rejected).get("gaps", [])
    next_step = _data(rejected).get("next_step", {})
    recorder.check(rejected, "未显式重绑领域时返回实际 domain-extract 与 writing-import 恢复参数",
                   _data(rejected).get("reason") == "domain_rebind_required"
                   and len(gaps) == 1 and isinstance(gaps[0].get("extract_argv"), list)
                   and isinstance(_data(rejected).get("resume_argv"), list)
                   and next_step.get("parent_version") == str(parent)
                   and next_step.get("edit_file") == str(edit),
                   "domain_rebind_required, exact extract argv, and import resume inputs",
                   _data(rejected))
    rebound = recorder.project_command(
        project, "writing-import", str(parent), str(edit), "--author", "Acceptance Researcher",
        "--reason", "Preserve text across source refresh", "--domain-version", str(fresh_domain_path),
    )
    rebound_path = _output_directory(project, rebound)
    revision = _json(rebound_path / "revision.json")
    recorder.check(rebound, "显式 fresh domain 重绑保留提交 bytes、作者、原因与 old/new pins",
                   (rebound_path / "raw-edit.bin").read_bytes() == raw
                   and revision.get("author") == "Acceptance Researcher"
                   and revision.get("reason") == "Preserve text across source refresh"
                   and set((revision.get("domain_rebind") or {})) == {"old", "new"},
                   "preserved raw/author/reason/domain_rebind", revision)
    reviewed, review_receipts = _exchange_command(
        recorder, project, ("writing-review", str(rebound_path)), 2, _external_response,
    )
    reviewed_path = _output_directory(project, reviewed)
    recorder.check(reviewed, "重绑后完成重新划分与当前全文复核",
                   _json(reviewed_path / "revision.json").get("stage") == "reviewed"
                   and bool(_json(reviewed_path / "review.json").get("items")),
                   "reviewed with complete items", {
                       "revision": _json(reviewed_path / "revision.json"),
                       "items": len(_json(reviewed_path / "review.json").get("items", [])),
                   })
    refresh_all = read_receipts + source_review_receipts + refresh_receipts + review_receipts
    recorder.check(scenario, "来源刷新和重绑复核使用八个 fingerprinted controlled responses",
                   len(refresh_all) == 8,
                   8, refresh_all)
    shared.update({"fresh_domain": fresh_domain_path, "rebound": reviewed_path})


def _claim_evidence(job: ModelJob, category: str) -> str:
    claim = next(item for item in job.payload["research"]["claims"] if item["category"] == category)
    return claim["evidence_ids"][0]


def _external_response(job: ModelJob) -> dict[str, Any]:
    task = job.payload.get("task")
    if task is None and job.kind in {"read", "review"}:
        from .backends import DemoBackend

        result = DemoBackend().run(job)
        if result.status != "succeeded" or result.data is None:
            raise ValueError(f"Controlled source response failed: {result.error}")
        return result.data
    if task == "innovation-generation-v1":
        research = job.payload["research"]
        chains = [chain for work in research["works"] for chain in work["reviewed_claims"]
                  if chain["decision"]["verdict"] == "supported"]
        contributions = [chain for chain in chains if chain["claim"]["category"] == "contribution"]
        chosen = contributions or chains[:1]
        return {"candidates": [{
            "statement_type": "system_hypothesis",
            "question": "Can workload-aware switching reduce waiting time across held-out simulations?",
            "closest_work_ids": [work["work"]["id"] for work in research["works"]],
            "closest_claim_ids": [chain["claim"]["id"] for chain in chosen],
            "claimed_difference": "Select a scheduling strategy by workload instead of assuming universal improvement.",
            "grounding_claim_ids": [chain["claim"]["id"] for chain in chains[:2]],
            "grounding_evidence_ids": sorted({item["id"] for chain in chains[:2]
                                                for item in chain["evidence"]}),
            "possible_counterexamples": ["Bursty arrivals can reverse the apparent waiting-time improvement."],
            "falsifiable_hypothesis": "A preregistered switch lowers waiting time on held-out synthetic workloads.",
            "minimum_validation": "Compare against both fixed strategies on the two known and held-out workloads.",
            "data_requirements": "Original synthetic traces with arrival and service times.",
            "compute_requirements": "A reproducible discrete-event simulation.",
            "risks": ["The switch can overfit the two known workloads."],
            "targeted_queries": ["workload adaptive scheduling waiting time counterexample"],
            "material_cutoff": research["material_cutoff"],
        }]}
    if task == "innovation-grounding-review-v1":
        candidate = job.payload["batch"]["candidates"][0]
        return {"items": [{
            "candidate_id": candidate["id"], "verdict": "partial",
            "rationale": "The controlled materials support workload differences, not novelty or an unrun experiment.",
            "claim_ids": candidate["grounding_claim_ids"],
            "evidence_ids": candidate["grounding_evidence_ids"],
        }]}
    if task == "innovation-assessment-v1":
        chains = [chain for work in job.payload["research"]["works"]
                  for chain in work["reviewed_claims"] if chain["decision"]["verdict"] == "supported"]
        refs = {
            "claim_ids": [chain["claim"]["id"] for chain in chains],
            "evidence_ids": sorted({evidence_id for chain in chains
                                     for evidence_id in chain["decision"]["evidence_ids"]}),
        }
        reason = {"text": "The original simulations support only a bounded comparison.", **refs}
        result = {name: {"judgment": "Requires human evaluation", **refs, "reasons": [reason]}
                  for name in ("novelty_clues", "research_value", "feasibility", "evidence_strength")}
        result.update(gaps=[], minimum_validation={
            "proposal": "Preregister a comparison on the known and held-out synthetic workloads.",
            "falsification_condition": "Reject the switch if it does not beat both fixed strategies.",
        })
        return result
    if task == "innovation-assessment-review-v1":
        refs = job.payload["assessment"]["research_value"]
        return {"items": [{
            "unit_id": unit["unit_id"], "verdict": "partial",
            "rationale": "This unit is supported only within the supplied original synthetic materials.",
            "claim_ids": refs["claim_ids"], "evidence_ids": refs["evidence_ids"],
        } for unit in job.payload["units"]]}
    if task == "domain-extract-v1":
        title = job.payload["research"]["work"]["title"]
        is_a = title.endswith("A")
        data_evidence = _claim_evidence(job, "data")
        method_evidence = _claim_evidence(job, "method")
        limit_evidence = _claim_evidence(job, "limitation")
        field_ids = [item["id"] for item in job.payload["domain"]["fields"]]
        if field_ids == ["simulated_jobs", "arrival_pattern", "waiting_time", "baseline"]:
            values = {
                "simulated_jobs": (40 if is_a else 60, [data_evidence]),
                "arrival_pattern": ((None, []) if is_a else ("bursty", [method_evidence])),
                "waiting_time": (("10 to 8 simulation units", [_claim_evidence(job, "result")]) if is_a
                                 else ("10 to 11 simulation units", [_claim_evidence(job, "result")])),
                "baseline": (("round-robin", [method_evidence]) if is_a else (None, [])),
            }
        else:
            values = {
                "observation_unit": ("simulated jobs", [data_evidence]),
                "arrival_pattern": ((None, []) if is_a else ("bursty", [method_evidence])),
                "transfer_limit": (
                    ("does not establish real-world effectiveness", [limit_evidence]) if is_a
                    else ("cannot establish a general trend across real systems", [limit_evidence])
                ),
            }
        return {"fields": [{
            "field_id": item["id"],
            "status": "reported" if values[item["id"]][0] is not None else "not_reported",
            "value": values[item["id"]][0], "evidence_ids": values[item["id"]][1],
        } for item in job.payload["domain"]["fields"]]}
    if task == "domain-review-v1":
        return {"items": [{
            "field_id": item["field_id"], "decision": "supported",
            "reason": "Controlled protocol checked this field against the complete original source.",
        } for item in job.payload["fields"]]}
    if task == "writing-outline-v1":
        return {"blocks": [{
            "role": "heading", "level": 1, "units": [{
                "text": "Boundary argument", "kind": "connector", "references": [],
                "purpose": "Use the external pack boundary_argument structure.",
            }]
        }, {
            "role": "paragraph", "level": None, "units": [{
                "text": "Compare the opposed simulated results before considering transfer.",
                "kind": "synthesis", "references": [_synthesis_reference(job, "result")],
                "purpose": "Plan a bounded cross-paper synthesis.",
            }]
        }]}
    if task == "writing-draft-v1":
        return {"blocks": [{
            "role": "heading", "level": 1, "units": [{
                "text": "Boundary argument across simulated workloads", "kind": "connector",
                "references": [], "purpose": "Keep the pack structure visible.",
            }]
        }, {
            "role": "paragraph", "level": None, "units": [{
                "text": ("Study A reports 10 to 8 for 40 simulated jobs, while Study B reports "
                         "10 to 11 for 60 simulated jobs under bursty arrivals; these simulations "
                         "do not establish effectiveness across real systems."),
                "kind": "synthesis", "references": [_synthesis_reference(job, "result")],
                "purpose": "State the numerical counterexample and its boundary.",
            }]
        }]}
    if task == "writing-review-v1":
        ids = job.payload["target_ids"]
        text_by_id = {unit["id"]: unit["text"]
                      for section in ("outline", "draft")
                      for block in job.payload[section]["blocks"] for unit in block["units"]}
        return {"items": [{
            "unit_id": unit_id,
            "decision": ("contradicted" if any(marker in text_by_id[unit_id]
                                                  for marker in _SEMANTIC_CONFLICT_MARKERS)
                         else "supported"),
            "reason": ("Controlled review retained and rejected the wrong number or unconditional claim."
                       if any(marker in text_by_id[unit_id]
                              for marker in _SEMANTIC_CONFLICT_MARKERS) else
                       "Controlled protocol checked the complete ordered unit against both original sources."),
        } for unit_id in ids]}
    if task == "writing-segment-v1":
        blocks = []
        for block in job.payload["raw_blocks"]:
            if block["role"] == "heading":
                spans = [{"start": 0, "end": len(block["text"]), "kind": "connector",
                          "references": [], "purpose": "Preserve the submitted heading."}]
            else:
                spans = [{"start": 0, "end": len(block["text"]), "kind": "synthesis",
                          "references": [_synthesis_reference(job, "result")],
                          "purpose": "Preserve the exact submitted cross-paper text."}]
            blocks.append({"block_id": block["id"], "spans": spans})
        return {"blocks": blocks}
    raise ValueError(f"No controlled response for {task}")


def _statistic_outline_response(job: ModelJob) -> dict[str, Any]:
    """Return the normal controlled response with one host-catalogued statistic unit."""
    response = _external_response(job)
    if job.payload.get("task") == "writing-outline-v1":
        statistic = next(
            item
            for item in job.payload["context"]["reference_catalog"]["statistics"]
            if item["path"] == "counts.eligible"
        )
        response["blocks"][1]["units"][0] = {
            "text": "The frozen eligible-work count is reported from the selected trend snapshot.",
            "kind": "frozen_statistic",
            "references": [{
                "type": "frozen_statistic",
                "pin_hash": statistic["pin_hash"],
                "computation_path": statistic["path"],
            }],
            "purpose": "Report a frozen corpus statistic with its host identity.",
        }
    return response


def _synthesis_reference(job: ModelJob, category: str) -> dict[str, Any]:
    sources = job.payload["context"]["reference_catalog"]["sources"]
    research = job.payload["context"]["research"]
    selected = []
    for work_id in job.payload["context"]["selection"]["work_ids"]:
        card = research["cards"][work_id]
        claim_id = next(item["id"] for item in card["claims"] if item["category"] == category)
        selected.append(next(item for item in sources if item["work_id"] == work_id
                             and item["claim_id"] == claim_id))
    return {"type": "synthesis", "sources": selected}


def _answer_exchange(project: Path, responder: Callable[[ModelJob], dict[str, Any]]) -> dict[str, Any]:
    logical_exchange = project / "exchange"
    pending = [path for path in io_path(logical_exchange).glob("*.request.json")
               if not path.with_name(path.name.replace(".request.", ".response.")).exists()]
    if len(pending) != 1:
        raise ValueError(f"Expected exactly one pending Exchange request, found {len(pending)}")
    request_path = pending[0]
    request = _json(request_path)
    job = ModelJob.model_validate(request["job"])
    response_path = request_path.with_name(request_path.name.replace(".request.", ".response."))
    atomic_json(response_path, {"fingerprint": request["fingerprint"], "data": responder(job)})
    logical_request = logical_exchange / request_path.name
    logical_response = logical_exchange / response_path.name
    return {
        "request": str(logical_request), "response": str(logical_response),
        "fingerprint": request["fingerprint"], "task": job.payload.get("task"),
        "protocol": "controlled_exchange_preset",
    }


def _exchange_command(recorder: Any, project: Path, args: tuple[str, ...], calls: int,
                      responder: Callable[[ModelJob], dict[str, Any]], *,
                      pending_expected: int = 2) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipts = []
    for _ in range(calls):
        pending = recorder.project_command(project, *args, expected=pending_expected)
        data = _data(pending)
        direct = data.get("status") == "needs_input"
        batch = any(
            isinstance(item, dict) and item.get("status") == "failed"
            and "needs_input" in str(item.get("error", ""))
            for item in data.get("results", [])
        )
        if not (direct or batch):
            raise ValueError(f"Exchange command did not request input: {_data(pending)}")
        receipts.append(_answer_exchange(project, responder))
    return recorder.project_command(project, *args), receipts


def _scenario_external(recorder: Any, scenario: dict[str, Any], shared: dict[str, Any]) -> None:
    project: Path = shared["project"]
    work_ids: list[str] = shared["work_ids"]
    external = write_external_pack(recorder.output / "external-pack-input")
    package = Path(__file__).resolve().parent
    before = _package_payload_snapshot(package)
    config = load_config(project)
    if config.backend != "exchange":
        config.backend = "exchange"
        atomic_json(project / "project.json", config.model_dump())
    args = (
        "domain-extract", "--question", _QUESTION, *_selected_args(work_ids),
        "--domain", str(external),
    )
    domain, domain_receipts = _exchange_command(recorder, project, args, 4, _external_response)
    domain_path = _output_directory(project, domain)
    fields = _json(domain_path / "domain.json")["fields"]
    comparison = (domain_path / "comparison.md").read_text(encoding="utf-8")
    recorder.check(domain, "外部包实际改变字段、列顺序、缺失项和值并保持逐篇证据",
                   [item["field_id"] for item in next(iter(fields.values()))] == [
                       "observation_unit", "arrival_pattern", "transfer_limit"]
                   and "not reported [supported]" in comparison
                   and "bursty [supported]" in comparison
                   and "| transfer_limit | arrival_pattern | observation_unit |" in comparison
                   and all(item["evidence_ids"] for rows in fields.values() for item in rows
                           if item["status"] == "reported"),
                   "external fields/order/not_reported/bursty/traceability", {
                       "fields": fields, "comparison": comparison,
                   })
    outline_args = (
        "writing-outline", "--question", _QUESTION, *_selected_args(work_ids),
        "--domain-version", str(domain_path),
    )
    outline, outline_receipts = _exchange_command(recorder, project, outline_args, 2, _external_response)
    outline_path = _output_directory(project, outline)
    draft, draft_receipts = _exchange_command(
        recorder, project, ("writing-draft", str(outline_path)), 2, _external_response,
    )
    draft_path = _output_directory(project, draft)
    serialized = json.dumps({
        "outline": _json(outline_path / "outline.json"),
        "draft": _json(draft_path / "draft.json"),
    }, ensure_ascii=False)
    recorder.check(draft, "boundary_argument 结构进入大纲正文并保留原始数值边界",
                   "Boundary argument" in serialized and "10 to 8" in serialized
                   and "10 to 11" in serialized,
                   "boundary argument and numeric synthesis", {
                       "contains_boundary": "Boundary argument" in serialized,
                       "contains_10_to_8": "10 to 8" in serialized,
                       "contains_10_to_11": "10 to 11" in serialized,
                   })
    optional_outline_args = (
        "writing-outline", "--question", _QUESTION, *_selected_args(work_ids),
        "--domain-version", str(domain_path), "--trends", str(shared["trends"]),
        "--candidate-version", str(shared["candidate"]),
        "--candidate-id", shared["candidate_id"],
        "--assessment-version", str(shared["assessment"]),
    )
    optional_outline_command, optional_outline_receipts = _exchange_command(
        recorder, project, optional_outline_args, 2, _external_response,
    )
    optional_outline = _output_directory(project, optional_outline_command)
    optional_draft_command, optional_draft_receipts = _exchange_command(
        recorder, project, ("writing-draft", str(optional_outline)), 2, _external_response,
    )
    optional_parent = _output_directory(project, optional_draft_command)
    config.backend = "demo"
    atomic_json(project / "project.json", config.model_dump())
    refused = recorder.project_command(project, *args, expected=2)
    recorder.check(refused, "严格 Demo 对未知外部包明确拒绝而不伪造领域值",
                   _data(refused).get("status") == "failed",
                   "failed", _data(refused))
    pack_value = _json(external)
    pack_value["version"] = "2"
    atomic_json(external, pack_value)
    pack_stale = recorder.project_command(
        project, "writing-status", "--document-id",
        _json(optional_parent / "revision.json")["document_id"],
        expected=2,
    )
    recorder.check(pack_stale, "外部 pack bytes 独立变化使历史完整可读但明确过期并给出恢复动作",
                   any(version.get("available") and not version.get("current")
                       for document in _data(pack_stale).get("documents", [])
                       for version in document.get("versions", [])),
                   "available stale version with top-level recovery argv",
                   _data(pack_stale))
    config.backend = "exchange"
    atomic_json(project / "project.json", config.model_dump())
    fresh_domain_command, fresh_domain_receipts = _exchange_command(
        recorder, project, args, 4, _external_response,
    )
    fresh_domain = _output_directory(project, fresh_domain_command)
    optional_edit = recorder.output / "optional-pins-domain-rebind.json"
    recorder.project_command(
        project, "writing-export-edit", str(optional_parent), "--output", str(optional_edit),
    )
    optional_raw = optional_edit.read_bytes()
    rebound_command = recorder.project_command(
        project, "writing-import", str(optional_parent), str(optional_edit),
        "--author", "Pinned acceptance author", "--reason", "Rebind a fresh base-only domain",
        "--domain-version", str(fresh_domain),
    )
    rebound_pending = _output_directory(project, rebound_command)
    rebound_command, rebound_receipts = _exchange_command(
        recorder, project, ("writing-review", str(rebound_pending)), 2, _external_response,
    )
    rebound_path = _output_directory(project, rebound_command)
    from .domain import load_domain_version
    from .writing import load_writing_version

    rebound_artifact = load_writing_version(rebound_path)
    rebound_revision = rebound_artifact.revision
    recorder.check(rebound_command,
                   "fresh base-only domain 重绑保留正文并重捕获非空 trend/candidate/assessment pins",
                   (rebound_path / "raw-edit.bin").read_bytes() == optional_raw
                   and rebound_revision.author == "Pinned acceptance author"
                   and rebound_revision.reason == "Rebind a fresh base-only domain"
                   and rebound_revision.domain_rebind is not None
                   and rebound_artifact.domain_pin == load_domain_version(fresh_domain).pin
                   and rebound_artifact.context.trend is not None
                   and rebound_artifact.context.candidate is not None
                   and rebound_artifact.context.assessment is not None
                   and rebound_revision.stage == "reviewed",
                   "preserved raw plus recaptured non-null optional pins and full review",
                   {"revision": rebound_revision.model_dump(mode="json"),
                    "selection": rebound_artifact.context.selection.model_dump(mode="json")})
    all_receipts = (domain_receipts + outline_receipts + draft_receipts
                    + optional_outline_receipts + optional_draft_receipts
                    + fresh_domain_receipts + rebound_receipts)
    after = _package_payload_snapshot(package)
    recorder.check(
        scenario, "外部 pack 全旅程前后安装包的全部代码与数据 payload bytes/file-set 不变",
        before == after and bool(before),
        "identical complete package payload map; interpreter bytecode caches excluded",
        {
            "before": before, "after": after,
            "excluded_interpreter_caches": ["__pycache__ directories", "*.pyc", "*.pyo"],
            "exclusion_reason": "Python may generate bytecode caches while importing modules; "
                                "they are not wheel payload files.",
        },
    )
    recorder.check(scenario, "所有外部协议响应均绑定原 request fingerprint 并标为 preset",
                   len(all_receipts) == 18
                   and all(item["fingerprint"] and item["protocol"] == "controlled_exchange_preset"
                           for item in all_receipts),
                   "18 fingerprinted controlled preset responses", all_receipts)
    shared.update({"external_pack": external, "external_domain": fresh_domain,
                   "external_outline": outline_path, "external_draft": draft_path,
                   "optional_rebound": rebound_path})


def _scenario_readonly_tamper(recorder: Any, scenario: dict[str, Any], shared: dict[str, Any]) -> None:
    project: Path = shared["project"]
    version: Path = shared["rebound"]
    status_before = _project_tree_snapshot(project)
    status = recorder.project_command(project, "writing-status")
    status_after = _project_tree_snapshot(project)
    status_snapshot = _readonly_snapshot_result(status_before, status_after)
    recorder.check(
        status, "writing-status 不改应用状态并单独记录只读 SQLite sidecar housekeeping",
        status_snapshot["passed"],
        "all application/main-DB/existing-WAL paths unchanged; only bounded SQLite housekeeping",
        {"before": status_before, "after": status_after, "comparison": status_snapshot},
    )
    guide_before = _project_tree_snapshot(project)
    guide = recorder.project_command(project, "guide")
    guide_after = _project_tree_snapshot(project)
    guide_snapshot = _readonly_snapshot_result(guide_before, guide_after)
    guide_actions = _data(guide).get("actions")
    recorder.check(guide, "guide 不改应用状态且返回结构化操作 argv",
                   guide_snapshot["passed"] and isinstance(guide_actions, list)
                   and bool(guide_actions)
                   and all(isinstance(action, dict)
                           and isinstance(action.get("argv"), list) and action["argv"]
                           for action in guide_actions),
                   "unchanged application state and structured guide actions with argv", {
                       "before": guide_before, "after": guide_after,
                       "comparison": guide_snapshot,
                       "guide_actions": guide_actions,
                       "guide_stdout_truncated": guide.get("stdout_truncated"),
                   })
    exported = recorder.output / "readonly-export.json"
    export_before = _project_tree_snapshot(project)
    recorder.project_command(project, "writing-export-edit", str(version), "--output", str(exported))
    export_after = _project_tree_snapshot(project)
    export_snapshot = _readonly_snapshot_result(export_before, export_after)
    recorder.check(
        scenario, "项目外导出不改应用状态且产生具体人工副本",
        export_snapshot["passed"] and exported.is_file(),
        "unchanged application state and concrete external edit file",
        {"before": export_before, "after": export_after,
         "comparison": export_snapshot, "export": str(exported)},
    )
    tampered = recorder.output / "v-tampered-writing-version"
    shutil.copytree(version, tampered)
    report = tampered / "report.md"
    report.write_bytes(report.read_bytes() + b"\nformal tamper")
    refused = recorder.project_command(
        project, "writing-export-edit", str(tampered), "--output",
        str(recorder.output / "must-not-exist.json"), expected=1,
    )
    recorder.check(refused, "正式写作产物被改动后由 manifest 层拒绝",
                   _data(refused).get("type") in {"ValueError", "WritingError"}
                   and not (recorder.output / "must-not-exist.json").exists(),
                   "manifest validation failure and no export", _data(refused))

    long_seed = recorder.output / "long-exchange-seed"
    long_ris = recorder.output / "long-exchange.ris"
    long_ris.write_text(
        "TY  - JOUR\nTI  - Long path Exchange boundary\n"
        "AU  - Academic Agent acceptance fixture\nPY  - 2026\nER  -\n",
        encoding="utf-8",
    )
    long_init = recorder.project_command(
        long_seed, "init", "--title", "Long path Exchange boundary",
        "--backend", "exchange", "--max-model-calls", "60",
    )
    long_config = load_config(long_seed)
    long_config.max_input_chars = 500_000
    atomic_json(long_seed / "project.json", long_config.model_dump())
    recorder.check(
        long_init, "独立长路径 fixture 从创建时保持固定60预算与500000上下文上限",
        (load_config(long_seed).max_model_calls, load_config(long_seed).max_input_chars)
        == (60, 500_000),
        {"max_model_calls": 60, "max_input_chars": 500_000},
        {"max_model_calls": load_config(long_seed).max_model_calls,
         "max_input_chars": load_config(long_seed).max_input_chars},
    )
    long_import = recorder.project_command(long_seed, "import", str(long_ris))
    long_rows = _data(long_import).get("imported", [])
    long_work_id = long_rows[0]["id"]
    recorder.project_command(
        long_seed, "screen", long_work_id, "--decision", "include",
        "--reason", "Explicitly include the controlled long-path source",
    )
    long_source = recorder.output / "long-exchange-source.txt"
    long_source.write_text(FIXTURES[0][1] + "\n\n", encoding="utf-8")
    recorder.project_command(long_seed, "attach", long_work_id, str(long_source))

    long_project = recorder.output / "交换 长路径 #100%"
    while len(str(long_project / "exchange" / ("f" * 64 + ".response.json"))) <= 300:
        long_project /= "中文 空格 #100% xxxxxxxxxxxxxxxx"
    shutil.copytree(io_path(long_seed), io_path(long_project))
    long_read, long_receipts = _exchange_command(
        recorder, long_project, ("read", "--work-id", long_work_id), 1, _external_response,
    )
    recorder.check(long_read, "安装命令以普通逻辑路径完成超过 260 字符的 fingerprinted Exchange 往返",
                   _data(long_read).get("results", [{}])[0].get("status") == "ok"
                   and len(long_receipts) == 1
                   and len(long_receipts[0]["response"]) > 260
                   and not long_receipts[0]["response"].startswith("\\\\?\\"),
                   "successful controlled Exchange using a >260 logical path", long_receipts)


def run_writing_scenarios(recorder: Any, projects: Path) -> None:
    """Append the explicit installed writing/domain acceptance scenarios to a recorder."""
    shared: dict[str, Any] = {}
    scenarios = [
        (WRITING_SCENARIO_NAMES[0], "完整来源、领域与跨文献正文旅程",
         lambda item: _scenario_full(recorder, item, projects, shared)),
        (WRITING_SCENARIO_NAMES[1], "人工原文捕获、划分、复核与缓存恢复",
         lambda item: _scenario_revision(recorder, item, shared)),
        (WRITING_SCENARIO_NAMES[2], "源码外领域包与受控 Exchange 协议",
         lambda item: _scenario_external(recorder, item, shared)),
        (WRITING_SCENARIO_NAMES[3], "来源过期、显式领域重绑与全文复核",
         lambda item: _scenario_stale_rebind(recorder, item, shared)),
        (WRITING_SCENARIO_NAMES[4], "只读状态、导出及正式产物篡改拒绝",
         lambda item: _scenario_readonly_tamper(recorder, item, shared)),
    ]
    for name, title, function in scenarios:
        recorder.scenario(name, title, function)


def _document_id(question: str, work_ids: list[str], domain_path: str) -> str:
    value = json.dumps(["writing-live-document-1", question, work_ids, domain_path],
                       ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "doc_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def run_live_acceptance(*, project: Path, output: Path, external_pack: Path,
                        domain_version: Path, outline_version: Path,
                        max_new_calls: int) -> dict[str, Any]:
    """Run exactly two domain and two writing calls with no retry or search."""
    if max_new_calls != 4:
        raise ValueError("Live writing acceptance requires exactly 4 new calls")
    project = Path(project).resolve()
    output = Path(output).resolve()
    external_pack = Path(external_pack).resolve()
    domain_version = Path(domain_version).resolve()
    outline_version = Path(outline_version).resolve()
    if output.exists():
        raise ValueError("Live acceptance output directory must not already exist")
    if not project.is_dir() or not external_pack.is_file() or not domain_version.is_dir() \
            or not outline_version.is_dir():
        raise ValueError("Project, external pack, domain version and outline version must already exist")

    from .domain import extract_domain, load_domain_version
    from .writing import create_draft, load_writing_version
    from .writing_models import WritingSelection

    prepared_domain = load_domain_version(domain_version)
    prepared_outline = load_writing_version(outline_version)
    selected = prepared_domain.context.selection
    if set(prepared_outline.context.selection.work_ids) != set(selected.work_ids):
        raise ValueError("Prepared domain and outline must use the same original work set")
    runner = Runner(project, offline=False)
    if runner.config.backend != "codex":
        raise ValueError("Live writing acceptance requires a project configured with the codex backend")
    ledger = install_actual_call_cap(runner, max_new_calls=max_new_calls)
    output.mkdir(parents=True)
    before_budget = runner.store.budget()
    one_work = [selected.work_ids[0]]
    live_selection = WritingSelection(
        document_id=_document_id(selected.question, one_work, str(external_pack)),
        question=selected.question,
        work_ids=one_work,
        domain_path=str(external_pack),
        trend_directory=None,
        candidate_directory=None,
        candidate_id=None,
        assessment_directory=None,
    )
    domain_result: dict[str, Any] = {
        "status": "not_run", "output": None,
        "gaps": [{"reason": "domain_observation_not_started"}],
    }
    draft_result: dict[str, Any] = {
        "status": "not_run", "output": None,
        "gaps": [{"reason": "domain_observation_incomplete"}],
    }
    try:
        domain_result = extract_domain(runner, live_selection)
    except Exception as exc:
        domain_result = {
            "status": "failed", "output": None,
            "gaps": [{"reason": "bounded_live_domain_exception", "detail": str(exc)}],
        }
    if domain_result.get("status") == "complete":
        try:
            draft_result = create_draft(runner, outline_version)
        except Exception as exc:
            draft_result = {
                "status": "failed", "output": None,
                "gaps": [{"reason": "bounded_live_draft_exception", "detail": str(exc)}],
            }
    after_budget = runner.store.budget()
    domain_output = (project / domain_result["output"]).resolve() \
        if domain_result.get("output") else None
    draft_output = (project / draft_result["output"]).resolve() \
        if draft_result.get("output") else None
    observations: dict[str, Any] = {"domain": None, "draft": None}
    inspection_failed = False
    if domain_output is not None:
        try:
            observations["domain"] = _inspect_live_domain(domain_output, one_work[0])
        except Exception as exc:
            inspection_failed = True
            observations["domain"] = {
                "status": "inspection_failed", "path": str(domain_output),
                "error": f"{type(exc).__name__}: {exc}",
            }
    if draft_output is not None:
        try:
            observations["draft"] = _inspect_live_draft(draft_output)
        except Exception as exc:
            inspection_failed = True
            observations["draft"] = {
                "status": "inspection_failed", "path": str(draft_output),
                "error": f"{type(exc).__name__}: {exc}",
            }
    complete = (
        domain_result.get("status") == "complete"
        and draft_result.get("status") == "complete"
        and ledger.reservations == ledger.attempts == 4
        and not inspection_failed
    )
    report = {
        "schema_version": 1,
        "status": "complete" if complete else "incomplete",
        "claim": "bounded_observation_only; no model-accuracy claim",
        "limits": {"max_new_calls": 4, "automatic_retries": 0, "searches": 0,
                   "budget_increases": 0},
        "inputs": {
            "project": str(project), "external_pack": str(external_pack),
            "external_pack_sha256": hashlib.sha256(external_pack.read_bytes()).hexdigest(),
            "prepared_domain_version": str(domain_version),
            "prepared_outline_version": str(outline_version),
            "selected_work_ids": one_work,
            "domain_original_pages": sum(len(document.pages) for document in
                                         prepared_domain.context.research.documents.values()
                                         if document.work_id in one_work),
            "draft_original_pages": sum(len(document.pages) for document in
                                        prepared_outline.context.research.documents.values()),
        },
        "budget": {"before": before_budget, "after": after_budget},
        "calls": {"reservations": ledger.reservations, "attempts": ledger.attempts,
                  "requested_model": ledger.requested_model, "events": ledger.events},
        "results": {"domain": domain_result, "draft": draft_result},
        "observations": observations,
        "provenance": {
            "actual": "CodexBackend using the existing local Codex login",
            "preset": "none in this opt-in run; offline usage report labels controlled Exchange separately",
            "observed_model_identity": "unavailable unless the backend reports it; never inferred",
        },
    }
    atomic_json(output / "report.json", report)
    return report


def _inspect_live_domain(path: Path, work_id: str) -> dict[str, Any]:
    payload = _json(path / "domain.json")
    review = _json(path / "domain-review.json")
    return {
        "path": str(path), "fields": payload.get("fields", {}).get(work_id),
        "review": review.get("review", {}).get(work_id),
        "condition_check": "manual inspection required; status is not a model-accuracy judgment",
    }


def _inspect_live_draft(path: Path) -> dict[str, Any]:
    draft = _json(path / "draft.json")
    text = "".join(unit["text"] for block in draft.get("blocks", []) for unit in block.get("units", []))
    return {
        "path": str(path), "text": text,
        "mentions_10_to_8": "10" in text and "8" in text,
        "mentions_10_to_11": "10" in text and "11" in text,
        "mentions_simulation_condition": "simulat" in text.lower(),
        "review": _json(path / "review.json"),
        "condition_check": "recorded for independent human inspection; no accuracy claim",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the bounded 0.4 real-backend writing observation")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-pack", type=Path, required=True)
    parser.add_argument("--domain-version", type=Path, required=True)
    parser.add_argument("--outline-version", type=Path, required=True)
    parser.add_argument("--max-new-calls", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_new_calls != 4:
        _parser().error("--max-new-calls must be exactly 4")
    try:
        report = run_live_acceptance(
            project=args.project, output=args.output, external_pack=args.external_pack,
            domain_version=args.domain_version, outline_version=args.outline_version,
            max_new_calls=args.max_new_calls,
        )
    except Exception as exc:
        print(json.dumps({"status": "incomplete", "error": str(exc), "type": type(exc).__name__},
                         ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
