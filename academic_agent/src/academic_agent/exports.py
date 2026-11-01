"""Snapshot exports built from current evidence and explicit review decisions."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .bibliography import export_bibtex
from .config import atomic_json, load_config
from .documents import abstract_document, verify_span
from .filesystem import io_path
from .models import ReadingCard, ReviewReport, digest, utcnow
from .store import Store

CATEGORIES = {"problem": "研究问题", "method": "方法", "data": "数据", "result": "结果",
              "limitation": "局限", "contribution": "作者贡献陈述"}


def escape_md(value: str) -> str:
    value = html.escape(value, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~])", r"\\\1", value).replace("\r", " ").replace("\n", " ")


def safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        if parts.scheme in {"https", "http"} and parts.hostname and not parts.username and not parts.password:
            return value.replace("(", "%28").replace(")", "%29").replace(" ", "%20").replace("\n", "")
    except ValueError:
        pass
    return ""


def _csv(rows: list[list]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    for row in rows:
        values = []
        for value in row:
            text = "" if value is None else str(value)
            values.append("'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text)
        writer.writerow(values)
    return stream.getvalue()


def snapshot_token(store: Store, connection=None) -> tuple[int, int, int, str]:
    if connection is None:
        with store.connect() as conn:
            conn.execute("BEGIN")
            return snapshot_token(store, conn)
    return (
        connection.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0],
        connection.execute("SELECT count(*) FROM logs").fetchone()[0],
        connection.execute("SELECT count(*) FROM jobs WHERE status='succeeded'").fetchone()[0],
        digest(load_config(store.root).model_dump_json()),
    )


def current_cards(store: Store, *, kind: str = "reading_card", question: str | None = None) -> dict[str, ReadingCard]:
    cards = {}
    for event in store.logs(kind):
        if question is not None and event["data"].get("question") != question:
            continue
        card = ReadingCard.model_validate(event["data"]["card"])
        document = store.active_document(card.work_id)
        if document and document.parser == "source-abstract" and document.id != abstract_document(store.work(card.work_id)).id:
            continue
        if document and document.id == card.document_id:
            evidence = {span.id: span for span in store.evidence(card.work_id)}
            if all(claim.work_id == card.work_id and claim.evidence_ids
                   and all(e in evidence and verify_span(evidence[e], document) for e in claim.evidence_ids)
                   for claim in card.claims):
                cards[card.work_id] = card
    return cards


def decisions(store: Store, cards: dict[str, ReadingCard]) -> dict[str, dict]:
    result = {}
    for event in store.logs("review"):
        report = ReviewReport.model_validate(event["data"]["report"])
        card = cards.get(report.work_id)
        if not card or report.document_id != card.document_id or event["data"]["card_hash"] != digest(card.model_dump_json()):
            continue
        claims = {c.id: c for c in card.claims}
        for verdict in report.verdicts:
            claim = claims.get(verdict.claim_id)
            if claim:
                valid = bool(verdict.evidence_ids) and set(verdict.evidence_ids).issubset(claim.evidence_ids)
                result[verdict.claim_id] = {**verdict.model_dump(), "reviewer": report.backend,
                                           "verdict": verdict.verdict if valid else "insufficient",
                                           "type": "model"}
    for event in store.logs("manual_review"):
        data = event["data"]
        card = cards.get(data["work_id"])
        if card and data["card_hash"] == digest(card.model_dump_json()) and any(c.id == data["claim_id"] for c in card.claims):
            result[data["claim_id"]] = {**data, "type": "manual"}
    return result


def build_export(store: Store, title: str, question: str, *, config_hash: str | None = None) -> Path:
    """Publish a new immutable version only while the snapshot is still current."""
    for work in store.works():
        document = store.active_document(work.id)
        if document and document.parser == "source-abstract":
            candidate = abstract_document(work)
            if candidate.id != document.id:
                store.add_document(candidate)
    token = snapshot_token(store)
    if config_hash is not None and token[3] != config_hash:
        raise ValueError("Project configuration changed; recreate Runner before exporting")
    works = store.works()
    cards = current_cards(store)
    verdicts = decisions(store, cards)
    documents = {w.id: store.active_document(w.id) for w in works}
    spans = [e for w in works for e in store.evidence(w.id)]
    supported, withheld = [], []
    for work in works:
        card = cards.get(work.id)
        if not card:
            continue
        for claim in card.claims:
            decision = verdicts.get(claim.id, {"verdict": "insufficient", "rationale": "尚未复核"})
            item = {"work": work, "claim": claim, "decision": decision}
            if work.screening == "include" and not work.retracted and decision["verdict"] == "supported":
                supported.append(item)
            else:
                withheld.append(item)
    demo = any("demo" in card.backend.lower() for card in cards.values())
    notice = ("**合成演示：包含原创建模样例，不是实际论文研究结论。**\n\n" if demo else "")
    notice += ("本稿由当前文献库中通过证据复核的条目组成，仍需作者审阅与综合。"
               "模型复核不等同于人工核验；未发现出版通知不等同于文献可靠。"
               "摘要、需 OCR 和出版状态未知的资料在下表中明确标注。\n\n")
    review = [f"# {escape_md(title)} — 文献综述工作稿\n\n", notice]
    if question:
        review.append(f"研究问题：{escape_md(question)}\n\n")
    review += ["## 资料范围\n\n", "| 文献 | 阅读层级 | 筛选 | 最近出版检查 |\n|---|---|---|---|\n"]
    for work in works:
        checks = "; ".join(f"{c.source}: {c.status}" for c in work.publication_checks[-2:]) or "unknown"
        doc = documents[work.id]
        review.append(f"| {escape_md(work.title)} | {doc.level if doc else 'metadata_only'} | {work.screening} | {escape_md(checks)} |\n")
    for category, label in CATEGORIES.items():
        review.append(f"\n## {label}\n\n")
        items = [item for item in supported if item["claim"].category == category]
        if not items:
            review.append("当前没有已复核且满足纳入条件的陈述。\n")
        for item in items:
            claim, work = item["claim"], item["work"]
            review.append(f"- {escape_md(claim.text)} [{work.citekey}](#{work.citekey})\n")
    review.append("\n## 参考文献\n\n")
    for work in works:
        url = safe_url(work.urls[0]) if work.urls else ""
        title_md = escape_md(work.title)
        if url:
            title_md = f"[{title_md}]({url})"
        review.append(f'<a id="{work.citekey}"></a>\n\n{work.citekey} · {title_md} · '
                      f'{escape_md("; ".join(work.authors))} · {work.year or "年份未知"}\n\n')

    reading = ["# 阅读卡片\n\n", notice]
    for work in works:
        card = cards.get(work.id)
        reading.append(f"\n## {escape_md(work.title)}\n\n")
        if not card:
            reading.append("尚无当前文档版本的阅读卡片。\n")
            continue
        reading.append(f"文档：{card.document_id}；后端：{escape_md(card.backend)}\n\n")
        for claim in card.claims:
            state = verdicts.get(claim.id, {}).get("verdict", "insufficient")
            reading.append(f"- **{CATEGORIES[claim.category]} / {state}**：{escape_md(claim.text)}\n"
                           f"  陈述 `{claim.id}`；证据 {', '.join(claim.evidence_ids)}\n")
        if card.missing:
            reading.append("\n未找到：" + escape_md("；".join(card.missing)) + "\n")

    comparison = [["work_id", "title", "level", *CATEGORIES.values()]]
    for work in works:
        doc = documents[work.id]
        comparison.append([work.id, work.title, doc.level if doc else "metadata_only", *[
            "；".join(item["claim"].text for item in supported if item["work"].id == work.id and item["claim"].category == category)
            for category in CATEGORIES]])
    table = ["# 文献比较矩阵\n\n只填入当前已支持的陈述，空格表示没有可用的已复核陈述。\n\n",
             "| " + " | ".join(comparison[0][1:]) + " |\n",
             "| " + " | ".join("---" for _ in comparison[0][1:]) + " |\n"]
    table += ["| " + " | ".join(escape_md(str(cell)) for cell in row[1:]) + " |\n" for row in comparison[1:]]
    gap_lines = ["# 证据缺口与待办\n\n这些是当前材料的缺口，不构成领域创新性结论。\n\n"]
    for work in works:
        doc = documents[work.id]
        gaps = []
        if not doc or doc.level != "full_text":
            gaps.append("补充可读全文" if not doc or doc.level != "needs_ocr" else "补充 OCR 或检查空白页")
        if work.screening != "include":
            gaps.append("完成筛选并记录理由")
        if work.retracted:
            gaps.append("已标记撤稿；不得用作普通结论依据")
        if not work.publication_checks or work.publication_checks[-1].status == "unknown":
            gaps.append("出版状态仍未知")
        if work.id not in cards:
            gaps.append("生成当前文档阅读卡")
        if gaps:
            gap_lines.append(f"- {escape_md(work.title)}：{escape_md('；'.join(gaps))}\n")
    gap_lines.append(f"\n未纳入正文的陈述：{len(withheld)}。原因见 quality.json 和 evidence-ledger.json。\n")
    coverage = {level: sum(bool(doc and doc.level == level) for doc in documents.values())
                for level in ("full_text", "abstract_only", "needs_ocr", "metadata_only")}
    coverage["metadata_only"] += sum(doc is None for doc in documents.values())
    quality = {
        "schema_version": 1, "created_at": utcnow(), "snapshot": token, "synthetic_demo": demo,
        "works": len(works), "included": sum(w.screening == "include" for w in works), "document_levels": coverage,
        "supported_claims": len(supported), "withheld_claims": len(withheld), "reading_cards": len(cards),
        "known_retracted": sum(w.retracted for w in works), "budget": store.budget(),
        "human_accuracy_evaluation": "not_performed", "source_recall_evaluation": "not_performed",
        "review_scope": "Full bounded available document per model review; see review event scope",
        "failed_or_pending_jobs": [{"id": j["id"], "kind": j["kind"], "status": j["status"], "error": j["error"]}
                                   for j in store.jobs() if j["status"] != "succeeded"],
    }
    ledger = {"evidence": [e.model_dump() for e in spans], "claims": [
        {"claim": item["claim"].model_dump(), "review": item["decision"], "work_id": item["work"].id,
         "included_in_review": item in supported} for item in supported + withheld]}
    payloads = {
        "review.md": "".join(review), "reading-cards.md": "".join(reading), "comparison.md": "".join(table),
        "comparison.csv": _csv(comparison), "gaps.md": "".join(gap_lines),
        "outline.md": "# 写作提纲\n\n" + "\n".join(f"{i}. {label}" for i, label in enumerate(
            ["范围与检索方法", "研究问题与分组依据", "方法及数据比较", "证据支持的结果", "局限与证据缺口", "待人工综合的讨论"], 1)) + "\n",
        "references.bib": export_bibtex(works),
        "library.csv": _csv([["id", "citekey", "title", "authors", "year", "doi", "screening"]] +
                            [[w.id, w.citekey, w.title, "; ".join(w.authors), w.year, w.doi, w.screening] for w in works]),
        "library.json": json.dumps([w.model_dump() for w in works], ensure_ascii=False, indent=2),
        "quality.json": json.dumps(quality, ensure_ascii=False, indent=2),
        "evidence-ledger.json": json.dumps(ledger, ensure_ascii=False, indent=2),
        "query-log.json": json.dumps(store.logs("search"), ensure_ascii=False, indent=2),
    }
    exports_dir = store.root / "exports"
    io_path(exports_dir).mkdir(exist_ok=True)
    version = "v-" + utcnow().replace(":", "").replace("+", "-").replace(".", "-") + "-" + uuid.uuid4().hex[:8]
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=io_path(exports_dir)))
    hashes = {}
    for name, content in payloads.items():
        data = content.encode("utf-8")
        with (staging / name).open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        hashes[name] = hashlib.sha256(data).hexdigest()
    atomic_json(staging / "manifest.json", {"schema_version": 1, "snapshot": token, "files": hashes})
    destination = exports_dir / version
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if snapshot_token(store, connection) != token:
            raise ValueError("Project changed during export; retry to build a current snapshot")
        staging.rename(io_path(destination))
        atomic_json(exports_dir / "latest.json", {"version": version, "snapshot": token})
    return destination
