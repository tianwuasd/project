"""Read-only, actionable project guidance for researchers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config
from .documents import abstract_document, verify_span
from .filesystem import io_path, sqlite_uri
from .models import Document, Evidence, ReadingCard, ReviewReport, Work, digest
from .trends import CorpusSnapshot, verify_trends


def _action(root: Path, command: list[str], reason: str) -> dict[str, Any]:
    return {
        "summary": reason,
        "argv": ["academic-agent", "--project", str(root), *command],
    }


def _preserve_offline(actions: list[dict[str, Any]], offline: bool) -> list[dict[str, Any]]:
    if offline:
        for action in actions:
            argv = action.get("argv")
            if isinstance(argv, list) and argv[:2] == ["academic-agent", "--project"]:
                argv.insert(3, "--offline")
    return actions


def _read_blob(root: Path, relative: str, expected_hash: str) -> bytes:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not io_path(path).is_file():
        raise ValueError("项目产物路径缺失或越界")
    content = io_path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_hash:
        raise ValueError("项目产物哈希不匹配")
    return content


def _direct_child(root: Path, parent: str, version: Any) -> Path | None:
    if not isinstance(version, str) or not version or Path(version).name != version:
        return None
    base = (root / parent).resolve()
    candidate = (base / version).resolve()
    return candidate if candidate.parent == base else None


def _json_pointer(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(io_path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _artifact_directory(root: Path, output: Any) -> Path:
    if not isinstance(output, str) or not output:
        raise ValueError("创新产物路径缺失")
    supplied = Path(output)
    path = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if not path.is_relative_to(root) or not io_path(path).is_dir():
        raise ValueError("创新产物路径缺失或越界")
    return path


def _readonly_state(root: Path, config) -> dict[str, Any]:
    db = root / "research.db"
    empty = {
        "works": [], "documents": {}, "cards": {}, "reviewed_work_ids": set(),
        "jobs": [], "logs": [], "budget_used": 0, "snapshot": None, "integrity_errors": [],
    }
    if not io_path(db).is_file():
        return empty
    uri = sqlite_uri(db)
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN")
            works = [Work.model_validate_json(row[0]) for row in connection.execute("SELECT data FROM works")]
            rows = list(connection.execute(
                "SELECT work_id,path,sha FROM documents WHERE active=1 ORDER BY work_id"
            ))
            documents: dict[str, Document] = {}
            integrity_errors = []
            for row in rows:
                try:
                    documents[row["work_id"]] = Document.model_validate_json(
                        _read_blob(root, row["path"], row["sha"])
                    )
                except (OSError, UnicodeError, ValueError) as exc:
                    integrity_errors.append(str(exc))
            evidence: dict[str, list[Evidence]] = {}
            for row in connection.execute("SELECT document_id,data FROM evidence ORDER BY id"):
                try:
                    evidence.setdefault(row["document_id"], []).append(Evidence.model_validate_json(row["data"]))
                except ValueError as exc:
                    integrity_errors.append(f"证据记录无法验证：{exc}")
            logs = [{"kind": row["kind"], "data": json.loads(row["data"]), "created": row["created"]}
                    for row in connection.execute("SELECT kind,data,created FROM logs ORDER BY created")]
            jobs = [dict(row) for row in connection.execute(
                "SELECT id,kind,status,error FROM jobs ORDER BY created"
            )]
            budget_used = connection.execute(
                "SELECT count(*) FROM attempts WHERE reservation IN ('reserved','spent','uncertain')"
            ).fetchone()[0]
            revision = connection.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
            log_count = connection.execute("SELECT count(*) FROM logs").fetchone()[0]
            succeeded = connection.execute("SELECT count(*) FROM jobs WHERE status='succeeded'").fetchone()[0]
        finally:
            connection.close()
    except (OSError, sqlite3.Error, ValueError, json.JSONDecodeError) as exc:
        return {**empty, "integrity_errors": [f"无法只读检查研究数据库：{exc}"]}

    cards: dict[str, ReadingCard] = {}
    works_by_id = {work.id: work for work in works}
    for event in logs:
        if event["kind"] != "reading_card":
            continue
        try:
            card = ReadingCard.model_validate(event["data"]["card"])
        except (KeyError, ValueError):
            continue
        document = documents.get(card.work_id)
        work = works_by_id.get(card.work_id)
        source_abstract_current = not (
            document and work and document.parser == "source-abstract"
            and document.id != abstract_document(work).id
        )
        spans = {span.id: span for span in evidence.get(card.document_id, [])}
        valid_claims = bool(card.claims) and all(
            claim.work_id == card.work_id and claim.evidence_ids
            and all(identifier in spans and verify_span(spans[identifier], document)
                    for identifier in claim.evidence_ids)
            for claim in card.claims
        ) if document else False
        if document and document.id == card.document_id and valid_claims and source_abstract_current:
            cards[card.work_id] = card
    reviewed_claims: dict[str, set[str]] = {}
    for event in logs:
        if event["kind"] != "review":
            continue
        try:
            report = ReviewReport.model_validate(event["data"]["report"])
            card = cards.get(report.work_id)
            if (card and report.document_id == card.document_id
                    and event["data"].get("card_hash") == digest(card.model_dump_json())):
                claims = {claim.id: claim for claim in card.claims}
                for verdict in report.verdicts:
                    claim = claims.get(verdict.claim_id)
                    if claim and verdict.evidence_ids and set(verdict.evidence_ids).issubset(claim.evidence_ids):
                        reviewed_claims.setdefault(report.work_id, set()).add(verdict.claim_id)
        except (KeyError, ValueError):
            continue
    for event in logs:
        if event["kind"] != "manual_review":
            continue
        data = event["data"]
        card = cards.get(data.get("work_id")) if isinstance(data, dict) else None
        if (card and data.get("card_hash") == digest(card.model_dump_json())
                and any(claim.id == data.get("claim_id") for claim in card.claims)):
            reviewed_claims.setdefault(card.work_id, set()).add(data["claim_id"])
    reviewed = {work_id for work_id, card in cards.items()
                if card.claims and {claim.id for claim in card.claims}.issubset(reviewed_claims.get(work_id, set()))}
    token = (revision, log_count, succeeded, digest(config.model_dump_json()))
    return {"works": works, "documents": documents, "cards": cards, "reviewed_work_ids": reviewed,
            "jobs": jobs, "logs": logs, "budget_used": budget_used, "snapshot": token,
            "integrity_errors": integrity_errors}


def project_guide(root: Path, *, offline: bool = False) -> dict:
    """Describe safe next CLI actions without changing project state or invoking providers."""
    root = Path(root).resolve()
    project_file = root / "project.json"
    if not io_path(project_file).is_file():
        return {
            "summary": "项目尚未创建；先初始化项目，再逐步导入或检索文献。",
            "blockers": ["项目尚未创建"],
            "actions": _preserve_offline(
                [_action(root, ["init", "--title", "研究标题"], "初始化独立研究项目")],
                offline,
            ),
            "state": {"project_exists": False},
        }
    config = load_config(root)
    state = _readonly_state(root, config)
    works: list[Work] = state["works"]
    blockers = list(dict.fromkeys(state["integrity_errors"]))
    actions: list[dict[str, Any]] = []
    public_state: dict[str, Any] = {
        "project_exists": True,
        "works": len(works),
        "pending": sum(work.screening == "pending" for work in works),
        "included": sum(work.screening == "include" and not work.retracted for work in works),
        "budget_used": state["budget_used"],
        "budget_limit": config.max_model_calls,
        "latest_export_current": False,
        "latest_trends_current": False,
        "innovation": {"generation_ready": False, "candidates": []},
        "writing": {"status": "needs_input", "documents": [], "gaps": [], "recovery": []},
    }
    if not works:
        blockers.append("项目尚无文献")
        for query in config.queries:
            if query.strip():
                actions.append(_action(root, ["search", query], f"按配置检索式查找文献：{query}"))
        actions.append(_action(root, ["import", "bibliography.ris"], "导入已有 RIS 或 BibTeX 书目"))
        summary = "项目尚无文献；可按配置检索式检索，或导入本地书目。"
    else:
        summary = f"已记录 {len(works)} 篇文献；按当前状态完成逐篇筛选、全文关联、阅读和复核。"

    searches: dict[str, dict[str, Any]] = {}
    for event in state["logs"]:
        if event["kind"] == "search" and isinstance(event["data"], dict):
            query = str(event["data"].get("summary", {}).get("query", "")).strip()
            searches[query] = event["data"]
    source_names = {"crossref": "Crossref", "openalex": "OpenAlex",
                    "semantic_scholar": "Semantic Scholar", "arxiv": "arXiv"}
    for query, event in searches.items():
        sources = event.get("summary", {}).get("sources", [])
        failed = [source for source in sources if source.get("status") not in {"ok", "empty"}]
        if not failed:
            continue
        detail = "、".join(
            f"{source_names.get(str(source.get('source', '')).lower(), source.get('source') or '未知来源')}"
            f"={source.get('status') or 'unknown'}" for source in failed
        )
        blockers.append(f"最近一次检索有来源失败或不完整：{detail}")
        retry = _action(root, ["search", query], f"人工确认后显式重试检索：{query}") if query else None
        if retry and all(action["argv"] != retry["argv"] for action in actions):
            actions.append(retry)

    pending = [work for work in works if work.screening == "pending"]
    if pending:
        blockers.append(f"有 {len(pending)} 篇文献待逐篇筛选")
        actions.extend(_action(root, ["screen", work.id, "--decision", "include", "--reason", "请填写逐篇筛选理由"],
                               f"人工检查并记录《{work.title}》的筛选决定") for work in pending)

    included = [work for work in works if work.screening == "include" and not work.retracted]
    documents: dict[str, Document] = state["documents"]
    effective_documents: dict[str, Document] = {}
    for work in included:
        document = documents.get(work.id)
        candidate = abstract_document(work)
        if document is None or document.parser == "source-abstract" and document.id != candidate.id:
            document = candidate
        effective_documents[work.id] = document
    no_text = [work for work in included
               if effective_documents[work.id].level in {"metadata_only", "needs_ocr"}]
    if no_text:
        blockers.append(f"有 {len(no_text)} 篇纳入文献没有可读文本")
        actions.extend(_action(root, ["attach", work.id, "文献全文.pdf"],
                               f"为《{work.title}》关联可读 PDF、Markdown 或 TXT") for work in no_text)

    readable = [work for work in included if work not in no_text]
    unread = [work for work in readable if work.id not in state["cards"]]
    if unread:
        blockers.append(f"有 {len(unread)} 篇可读文献尚未生成当前阅读卡")
        actions.extend(_action(root, ["read", "--work-id", work.id], f"阅读《{work.title}》") for work in unread)
    unreviewed = [work for work in readable if work.id in state["cards"]
                  and work.id not in state["reviewed_work_ids"]]
    if unreviewed:
        blockers.append(f"有 {len(unreviewed)} 篇当前阅读卡尚未复核")
        actions.extend(_action(root, ["review", "--work-id", work.id], f"复核《{work.title}》的当前阅读卡")
                       for work in unreviewed)

    failed_jobs = [job for job in state["jobs"] if job["status"] in {"failed", "running"}]
    if failed_jobs:
        blockers.append(f"有 {len(failed_jobs)} 个模型任务失败、等待输入或仍在占用")
    if config.backend == "exchange" and any("Manual response required" in (job.get("error") or "")
                                             for job in failed_jobs):
        blockers.append("exchange 后端正在等待人工响应文件")
    if state["budget_used"] >= config.max_model_calls:
        blockers.append(f"模型调用预算已用尽（{state['budget_used']}/{config.max_model_calls}）")

    if works:
        from .challenges import read_challenges
        from .innovations import innovation_status
        from .research_context import read_research

        try:
            read_research(root, require_supported=True)
            public_state["innovation"]["generation_ready"] = True
        except ValueError:
            pass
        innovation = innovation_status(root)
        candidate_rows = innovation.get("candidates", []) if isinstance(innovation, dict) else []
        public_state["innovation"]["candidates"] = candidate_rows
        innovation_history_exists = io_path(root / "analyses" / "innovation").exists()
        if innovation.get("status") == "failed" and innovation_history_exists:
            reasons = [gap.get("reason") for gap in innovation.get("gaps", []) if isinstance(gap, dict)]
            blockers.append(f"创新历史或当前研究输入无法验证：{'、'.join(filter(None, reasons)) or 'unknown'}")
        if public_state["innovation"]["generation_ready"] and not candidate_rows:
            actions.append(_action(root, ["innovate"], "从当前已复核证据生成并独立复核创新候选"))
        for row in candidate_rows:
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                continue
            if row.get("historical_available"):
                blockers.append(f"候选 {candidate_id} 有可核验的历史版本；历史存在不代表当前前提仍成立")
            if not row.get("premises_current"):
                blockers.append(f"候选 {candidate_id} 的研究前提已过期")
                if public_state["innovation"]["generation_ready"]:
                    actions.append(_action(root, ["innovate"], "研究输入已变化，重新生成当前候选"))
            try:
                challenge = read_challenges(root, candidate_id)
                challenge_status = challenge.get("status")
                tasks = challenge.get("unresolved_tasks", [])
                if challenge_status == "not_searched":
                    unfinished = next((task for task in tasks if task.get("reason") == "not_searched"), {})
                    query = unfinished.get("query", "未知检索式")
                    round_number = str(unfinished.get("round") or 1)
                    blockers.append(
                        f"候选 {candidate_id} 第 {round_number} 轮反例检索式尚未取得来源收据：{query}"
                    )
                    actions.append(_action(
                        root, ["challenge", candidate_id, "--round", round_number],
                        f"恢复第 {round_number} 轮尚未搜索的反例检索式：{query}",
                    ))
                elif challenge_status == "source_failed":
                    failure_reasons = {"partial", "unavailable", "rate_limited", "retained_ids_unavailable"}
                    failed = next((task for task in tasks if task.get("reason") in failure_reasons), {})
                    query = failed.get("query", "未知检索式")
                    round_number = str(failed.get("round") or 1)
                    reason = failed.get("reason", "未知来源错误")
                    blockers.append(
                        f"候选 {candidate_id} 第 {round_number} 轮反例检索式来源失败（{reason}）：{query}"
                    )
                    actions.append(_action(
                        root, ["challenge", candidate_id, "--round", round_number],
                        f"恢复第 {round_number} 轮来源失败的反例检索式（{reason}）：{query}",
                    ))
                elif challenge_status == "needs_reading":
                    blockers.append(f"候选 {candidate_id} 的反例材料尚待筛选或阅读")
                elif challenge_status == "needs_assessment":
                    blockers.append(f"候选 {candidate_id} 的反例材料已有当前证据，尚待四维评估")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                blockers.append(f"候选 {candidate_id} 的反例检索历史无法验证：{exc}")
            assessment = row.get("assessment", {})
            assessment_status = assessment.get("status") if isinstance(assessment, dict) else "invalid"
            if assessment_status == "missing":
                actions.append(_action(root, ["assess-innovation", candidate_id],
                                       f"对候选 {candidate_id} 进行四维评估并独立复核"))
            elif assessment_status in {"stale", "invalid"}:
                blockers.append(f"候选 {candidate_id} 的评估已过期或无法验证")
                actions.append(_action(root, ["assess-innovation", candidate_id],
                                       f"重新评估候选 {candidate_id} 的当前证据"))
            if isinstance(assessment, dict) and assessment.get("output"):
                try:
                    _artifact_directory(root, assessment["output"])
                except ValueError as exc:
                    blockers.append(f"候选 {candidate_id} 的评估产物需要修复：{exc}")

    from .writing import load_writing_version, writing_status

    writing = writing_status(root)
    public_state["writing"] = {**writing, "recovery": []}
    for document in writing.get("documents", []):
        for version in document.get("versions", []):
            output = version.get("output")
            if not isinstance(output, str) or not output:
                continue
            version_path = (root / output).resolve()
            if not version.get("available"):
                blockers.append(f"写作版本 {version.get('version', 'unknown')} 无法验证；请从可信备份恢复")
                continue
            try:
                artifact = load_writing_version(version_path)
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if version.get("current"):
                if version.get("stage") == "pending":
                    actions.append(_action(
                        root, ["writing-review", str(version_path)],
                        f"划分并独立复核 pending {version.get('extent')} 版本 {version_path.name}",
                    ))
                elif version.get("extent") == "outline":
                    actions.append(_action(
                        root, ["writing-draft", str(version_path)],
                        f"从已复核大纲 {version_path.name} 生成并独立复核正文",
                    ))
                continue
            reasons = [gap.get("reason") for gap in version.get("gaps", []) if isinstance(gap, dict)]
            blockers.append(
                f"写作版本 {version_path.name} 的来源、复核、领域包或固定依赖已变化："
                f"{'、'.join(filter(None, reasons)) or 'unknown'}"
            )
            selection = artifact.context.selection
            extract_command = ["domain-extract", "--question", selection.question]
            for work_id in selection.work_ids:
                extract_command.extend(["--work-id", work_id])
            extract_command.extend(["--domain", selection.domain_path])
            extract_action = _action(
                root, extract_command,
                f"按版本 {version_path.name} 保存的问题、作品范围和领域包重新提取字段",
            )
            actions.append(extract_action)
            if artifact.revision.origin == "human" and artifact.revision.parent_version:
                parent_path = (root / "analyses" / "writing" / artifact.revision.parent_version).resolve()
                raw_edit = (version_path / "raw-edit.bin").resolve()
                public_state["writing"]["recovery"].append({
                    "version": version_path.name,
                    "domain_extract_argv": extract_action["argv"],
                    "after_extract": {
                        "summary": (
                            "领域提取完成后，将它返回的实际 output 目录作为 --domain-version；"
                            "保持下面的 parent、原始编辑 bytes、作者、理由和删除清单不变。"
                        ),
                        "parent_version": str(parent_path),
                        "edit_file": str(raw_edit),
                        "author": artifact.revision.author,
                        "reason": artifact.revision.reason,
                        "delete_ids": artifact.revision.deleted_ids,
                        "then": "对 writing-import 返回的实际 pending output 运行 writing-review。",
                    },
                })
                blockers.append(
                    f"人工作者版本 {version_path.name} 的原文仍保存在 {raw_edit}；"
                    "取得新的实际领域 output 后再以 --domain-version 导入并独立复核"
                )

    latest_export = _json_pointer(root / "exports" / "latest.json")
    if latest_export and isinstance(latest_export.get("snapshot"), list):
        public_state["latest_export_current"] = tuple(latest_export["snapshot"]) == state["snapshot"]
    if latest_export and not public_state["latest_export_current"]:
        blockers.append("最新导出已过期；研究内容、任务结果或配置已变化")

    latest_trends = _json_pointer(root / "analyses" / "trends" / "latest.json")
    if latest_trends:
        directory = _direct_child(root, "analyses/trends", latest_trends.get("version"))
        try:
            if directory is None:
                raise ValueError("趋势 latest.json 含不安全版本路径")
            verified = verify_trends(directory)
            snapshot = CorpusSnapshot.model_validate_json(io_path(directory / "corpus.json").read_bytes())
            public_state["latest_trends_current"] = (
                tuple(verified["snapshot"]) == state["snapshot"]
                and snapshot.as_of_year == datetime.now(timezone.utc).year
            )
        except (OSError, UnicodeError, ValueError) as exc:
            blockers.append(f"最新趋势产物无法验证：{exc}")
        if not public_state["latest_trends_current"] and not any("无法验证" in item for item in blockers):
            blockers.append("最新趋势分析已过期，或统计截止年不是当前年份")

    if state["reviewed_work_ids"] or latest_export:
        if not public_state["latest_export_current"]:
            actions.append(_action(root, ["export"], "按当前可用阅读与复核结果生成新版本导出"))
    if works:
        actions.append(_action(root, ["trends"], "生成当前冻结语料的描述统计"))
    _preserve_offline(actions, offline)
    unique_actions = []
    seen_argv = set()
    for action in actions:
        key = tuple(action["argv"])
        if key not in seen_argv:
            unique_actions.append(action)
            seen_argv.add(key)
    return {"summary": summary, "blockers": list(dict.fromkeys(blockers)), "actions": unique_actions,
            "state": public_state}
