"""Frozen, bounded corpus descriptions; never an estimate of a research field."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import atomic_json, load_config
from .exports import _csv, escape_md, snapshot_token
from .filesystem import io_path
from .models import PublicationCheck, Work, utcnow
from .store import Store

ALGORITHM_VERSION = "corpus-trends-1"
DateBasis = Literal["record_year", "first_publication_year", "formal_publication_year"]
Year = Annotated[int, Field(ge=1000, le=9999)]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FILES = frozenset({"corpus.json", "computation.json", "annual.csv", "labels.csv", "report.md"})
LEVELS = ("metadata_only", "abstract_only", "full_text", "needs_ocr")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Must be nonempty after trimming whitespace")
    return value


class StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, revalidate_instances="always")
    schema_version: int = Field(default=1, ge=1, le=1)


class WorkAnnotation(StrictRecord):
    year: Year | None = None
    year_source: str = ""
    labels: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("labels")
    @classmethod
    def normalize_labels(cls, value: list[str]) -> list[str]:
        return sorted({_nonempty(label) for label in value})

    @model_validator(mode="after")
    def require_sources(self) -> WorkAnnotation:
        if self.year is not None and not self.year_source.strip():
            raise ValueError("An annotated year requires a nonempty year_source")
        if self.labels and not self.rationale.strip():
            raise ValueError("Method labels require a nonempty rationale")
        return self


class CorpusAnnotations(StrictRecord):
    version: str
    annotator: str
    works: dict[str, WorkAnnotation] = Field(default_factory=dict)
    label_definitions: dict[str, str] = Field(default_factory=dict)

    @field_validator("version", "annotator")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        return _nonempty(value)

    @field_validator("label_definitions")
    @classmethod
    def normalize_definitions(cls, value: dict[str, str]) -> dict[str, str]:
        result = {_nonempty(label): _nonempty(definition) for label, definition in value.items()}
        if len(result) != len(value):
            raise ValueError("Duplicate normalized label definitions")
        return dict(sorted(result.items()))

    @model_validator(mode="after")
    def check_definitions(self) -> CorpusAnnotations:
        if self.label_definitions:
            unknown = {label for annotation in self.works.values() for label in annotation.labels} \
                - self.label_definitions.keys()
            if unknown:
                raise ValueError(f"Labels missing from fixed definitions: {sorted(unknown)}")
        return self


class FrozenCheck(PublicationCheck):
    model_config = StrictRecord.model_config
    schema_version: int = Field(default=1, ge=1, le=1)


class FrozenWork(Work):
    """Retain raw metadata years so invalid values are counted, never coerced."""

    model_config = StrictRecord.model_config
    schema_version: int = Field(default=1, ge=1, le=1)
    year: Any = None
    publication_checks: list[FrozenCheck] = Field(default_factory=list)

    @field_validator("year")
    @classmethod
    def scalar_year(cls, value: Any) -> Any:
        if value is not None and type(value) not in {int, str, float, bool}:
            raise ValueError("Metadata year must be a JSON scalar")
        return value


class FrozenDocument(StrictRecord):
    id: str
    work_id: str
    content_hash: Hash
    artifact_hash: Hash
    level: Literal["metadata_only", "abstract_only", "full_text", "needs_ocr"]
    parser: str
    created_at: str


class SearchResponse(StrictRecord):
    source: str
    status: str = "unknown"
    queries: list[str] = Field(default_factory=list)
    total: int | None = Field(default=None, ge=0)
    records: int = Field(default=0, ge=0)
    truncated: bool | None = None
    errors: list[str] = Field(default_factory=list)
    retrieved_at: str | None = None


class SourceSummary(StrictRecord):
    source: str
    status: str = "unknown"
    records: int = Field(default=0, ge=0)
    truncated: bool | None = None
    errors: list[str] = Field(default_factory=list)


class SearchSummary(StrictRecord):
    query: str = ""
    retained: int = Field(default=0, ge=0)
    dropped_at_project_cap: int = Field(default=0, ge=0)
    sources: list[SourceSummary] = Field(default_factory=list)


class SearchEvent(StrictRecord):
    id: str
    created: str
    query_key: str = ""
    summary: SearchSummary
    responses: list[SearchResponse]


class ProjectScope(StrictRecord):
    title: str
    question: str
    sources: list[str]
    queries: list[str]
    max_candidates: int = Field(ge=1, le=1000)
    max_documents: int = Field(ge=1, le=100)
    max_model_calls: int = Field(ge=1, le=1000)


class CorpusSnapshot(StrictRecord):
    source_snapshot: tuple[int, int, int, Hash]
    as_of_year: Year
    date_basis: DateBasis
    project: ProjectScope
    works: list[FrozenWork]
    documents: list[FrozenDocument]
    search_log: list[SearchEvent]
    annotations: CorpusAnnotations | None = None

    @model_validator(mode="after")
    def consistent_references(self) -> CorpusSnapshot:
        ids = [work.id for work in self.works]
        if any(not identifier.strip() for identifier in ids) or len(set(ids)) != len(ids):
            raise ValueError("Empty or duplicate work IDs in corpus")
        if self.annotations and (unknown := self.annotations.works.keys() - set(ids)):
            raise ValueError(f"Unknown annotation work IDs: {sorted(unknown)}")
        document_works = [document.work_id for document in self.documents]
        if len(document_works) != len(set(document_works)) or set(document_works) - set(ids):
            raise ValueError("Invalid active document work IDs")
        if len({document.id for document in self.documents}) != len(self.documents):
            raise ValueError("Duplicate document IDs")
        if len({event.id for event in self.search_log}) != len(self.search_log):
            raise ValueError("Duplicate search log IDs")
        _canonical(self.model_dump(mode="json"))
        return self


class Parameters(StrictRecord):
    date_basis: DateBasis
    as_of_year: Year
    small_corpus_threshold: int = 50
    small_year_threshold: int = 10
    annual_denominator: str = "eligible records with valid nonfuture selected year"
    label_denominator: str = "all eligible records, including missing/invalid/future years"


class AnnualRow(StrictRecord):
    year: Year
    count: int = Field(ge=0)
    denominator: int = Field(ge=0)
    share: float | None
    current_year: bool


class LabelRow(StrictRecord):
    label: str
    definition: str
    count: int = Field(ge=0)
    denominator: int = Field(ge=0)
    share: float | None
    annual: list[AnnualRow]


class Diagnostic(StrictRecord):
    code: str
    message: str


class ComputationRecord(StrictRecord):
    algorithm_version: Literal["corpus-trends-1"] = ALGORITHM_VERSION
    input_hash: Hash
    parameters: Parameters
    work_ids: dict[str, list[str]]
    counts: dict[str, int]
    annual: list[AnnualRow]
    labels: list[LabelRow]
    document_levels: dict[str, int]
    arxiv_share: float | None
    diagnostics: list[Diagnostic]
    formulas: dict[str, str]


def freeze_corpus(store: Store, *, annotations: CorpusAnnotations | None = None,
                  date_basis: str = "record_year", as_of_year: int | None = None) -> CorpusSnapshot:
    """Read a coherent revision without modifying the library or contacting sources."""
    token = snapshot_token(store)
    config = load_config(store.root)
    # Work.year in V1 is coercive. Read its serialized value before Pydantic can turn true into 1.
    with store.connect() as connection:
        works = [FrozenWork.model_validate_json(row[0]) for row in connection.execute("SELECT data FROM works")]
        document_rows = {row["id"]: dict(row) for row in connection.execute(
            "SELECT id,work_id,sha FROM documents WHERE active=1")}
    documents = []
    for work in sorted(works, key=lambda work: work.id):
        document = store.active_document(work.id)
        if document:
            row = document_rows.get(document.id)
            if row is None or document.work_id != work.id or row["work_id"] != work.id:
                raise ValueError("Project changed or document identity mismatch during corpus freeze")
            documents.append(FrozenDocument(id=document.id, work_id=work.id, content_hash=document.content_hash,
                                            artifact_hash=row["sha"], level=document.level, parser=document.parser,
                                            created_at=document.created_at))
    events = []
    for event in store.logs("search"):
        data = event["data"]
        responses = []
        for response in data.get("responses", []):
            responses.append(SearchResponse.model_validate({
                **{key: response[key] for key in ("source", "status", "queries", "total", "truncated",
                                                 "errors", "retrieved_at") if key in response},
                "records": len(response.get("records", [])),
            }))
        events.append(SearchEvent(id=event["id"], created=event["created"], query_key=data.get("query_key", ""),
                                  summary=data.get("summary", {}), responses=responses))
    snapshot = CorpusSnapshot(
        source_snapshot=token, as_of_year=datetime.now(timezone.utc).year if as_of_year is None else as_of_year,
        date_basis=date_basis, project=ProjectScope.model_validate(config.model_dump(include=set(ProjectScope.model_fields))),
        works=sorted(works, key=lambda work: work.id), documents=documents,
        search_log=sorted(events, key=lambda event: (event.created, event.id)),
        annotations=CorpusAnnotations.model_validate_json(annotations.model_dump_json()) if annotations else None,
    )
    if snapshot_token(store) != token:
        raise ValueError("Project changed during corpus freeze; retry")
    return snapshot


def _share(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def compute_trends(snapshot: CorpusSnapshot) -> ComputationRecord:
    """Pure calculation over a validated snapshot. No clock, database, network or model."""
    snapshot = CorpusSnapshot.model_validate_json(snapshot.model_dump_json())
    annotations = snapshot.annotations.works if snapshot.annotations else {}
    groups = {name: [] for name in ("candidates", "included", "excluded", "pending", "retracted", "eligible",
                                   "missing_year", "invalid_year", "future_year", "valid_year", "labelled",
                                   "unlabelled", "related", "arxiv_identified")}
    annual_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    label_year_counts: Counter[tuple[str, int]] = Counter()
    levels = dict.fromkeys(LEVELS, 0)
    documents = {document.work_id: document for document in snapshot.documents}
    for work in sorted(snapshot.works, key=lambda work: work.id):
        groups["candidates"].append(work.id)
        groups[{"include": "included", "exclude": "excluded", "pending": "pending"}[work.screening]].append(work.id)
        if work.retracted:
            groups["retracted"].append(work.id)
        if work.screening != "include" or work.retracted:
            continue
        groups["eligible"].append(work.id)
        document = documents.get(work.id)
        levels[document.level if document else "metadata_only"] += 1
        if work.relations:
            groups["related"].append(work.id)
        if work.arxiv_id and work.arxiv_id.strip():
            groups["arxiv_identified"].append(work.id)
        annotation = annotations.get(work.id)
        year = work.year if snapshot.date_basis == "record_year" else annotation.year if annotation else None
        if year is None:
            groups["missing_year"].append(work.id)
            valid_year = False
        elif type(year) is not int or not 1000 <= year <= 9999:
            groups["invalid_year"].append(work.id)
            valid_year = False
        elif year > snapshot.as_of_year:
            groups["future_year"].append(work.id)
            valid_year = False
        else:
            groups["valid_year"].append(work.id)
            annual_counts[year] += 1
            valid_year = True
        labels = annotation.labels if annotation else []
        groups["labelled" if labels else "unlabelled"].append(work.id)
        for label in sorted(set(labels)):
            label_counts[label] += 1
            if valid_year:
                label_year_counts[label, year] += 1
    counts = {name: len(ids) for name, ids in groups.items()}
    years = list(range(min(annual_counts), max(annual_counts) + 1)) if annual_counts else []
    annual = [AnnualRow(year=year, count=annual_counts[year], denominator=counts["valid_year"],
                        share=_share(annual_counts[year], counts["valid_year"]), current_year=year == snapshot.as_of_year)
              for year in years]
    definitions = snapshot.annotations.label_definitions if snapshot.annotations else {}
    catalog = set(definitions) | {label for annotation in annotations.values() for label in annotation.labels}
    labels = [LabelRow(label=label, definition=definitions.get(label, "未另附定义；解释限于逐文献人工理由"),
                       count=label_counts[label], denominator=counts["eligible"],
                       share=_share(label_counts[label], counts["eligible"]), annual=[
                           AnnualRow(year=year, count=label_year_counts[label, year], denominator=annual_counts[year],
                                     share=_share(label_year_counts[label, year], annual_counts[year]),
                                     current_year=year == snapshot.as_of_year) for year in years])
              for label in sorted(catalog)]
    diagnostics = [Diagnostic(code="descriptive_only", message=
        "仅描述冻结语料内部组成；检索与筛选具有选择偏差，不推断领域总体增长、显著性或未来趋势。"),
        Diagnostic(code="record_unit", message=
        "计数单位为库内去重书目记录，不等同独立研究；预印本与正式版关系不自动合并，也不构成独立重复验证。"),
        Diagnostic(code="arxiv_identifier_only", message=
        "arXiv 标识覆盖仅表示记录带有该标识，不代表 arXiv 来源召回、检索覆盖或论文质量。"),
        Diagnostic(code="no_human_evaluation", message="未执行人工评测；出版检查未发现通知不等同可靠，状态可能过期。")]
    def note(code: str, message: str) -> None:
        diagnostics.append(Diagnostic(code=code, message=message))
    if snapshot.date_basis == "record_year":
        note("mixed_record_year", "record_year 来自元数据，出版/上线/版本年份语义未统一；人工年份批注不覆盖此口径。")
    else:
        note("annotated_year_only", "当前年份口径仅使用带来源说明的人工年份批注；无批注记录计为缺失，不回退元数据。")
    if counts["eligible"] < 50 or any(row.count < 10 for row in annual if not row.current_year):
        note("small_sample", "小样本提醒：合格纳入总量 <50 或任一实际比较年份 <10；仅作描述。")
    if any(row.current_year for row in annual):
        note("current_year_incomplete", f"截止年 {snapshot.as_of_year} 为当前统计年，尚未结束，单列且不作完整年度比较。")
    if not counts["eligible"]:
        note("empty_corpus", "没有合格纳入记录；零分母占比显示为空，不解释为 0%。")
    if any(counts[name] for name in ("missing_year", "invalid_year", "future_year")):
        note("year_gaps", "缺失、非法及未来年份单列，不参与年度分母；观察年份之间无记录的年份明确补零。")
    if counts["unlabelled"]:
        note("label_gaps", "部分记录没有人工方法标签；未标注不等于不使用该方法。")
    if catalog and not definitions:
        note("label_definitions_unspecified", "未提供独立标签定义；固定词表取本次人工批注，含义只由人工理由说明。")
    if counts["related"]:
        note("related_records", "存在疑似关系或来源别名记录；关系完整性与独立性未人工核验。")
    if not snapshot.search_log:
        note("local_coverage_unknown", "没有检索日志，本地导入的检索覆盖未知；配置中的检索式不表示已实际检索。")
    elif any(provenance.get("source") in {"local_import", "ris_import", "bibtex_import"}
             for work in snapshot.works for provenance in work.provenance):
        note("local_coverage_unknown", "语料混有本地导入记录；其检索覆盖未知，已有在线检索日志不能补足本地导入的覆盖证明。")
    for event in snapshot.search_log:
        sources = event.summary.sources
        if not event.responses or any(response.status not in {"ok", "empty"} or response.truncated is not False
                                      or response.errors for response in event.responses) or any(
                source.status not in {"ok", "empty"} or source.truncated is not False
                or source.errors for source in sources):
            note("search_incomplete", f"检索日志 {event.id} 存在失败、截断或缺失信息，详见检索范围表。")
        if event.summary.dropped_at_project_cap:
            note("project_cap", f"检索日志 {event.id} 因项目候选上限丢弃 {event.summary.dropped_at_project_cap} 条。")
    if any("synthetic" in work.source_ids or any(check.source == "synthetic" for check in work.publication_checks)
           for work in snapshot.works):
        note("synthetic_demo", "语料含明确标记的原创合成样例，仅用于演示流程，不是实际论文研究结论。")
    note("bounded_retrieval", f"配置候选上限 {snapshot.project.max_candidates}、阅读文档上限 {snapshot.project.max_documents}；"
         "所有来源的覆盖、更新延迟和检索排序均可能影响样本，成功检索亦不代表全面覆盖。")
    return ComputationRecord(input_hash=_hash(_canonical(snapshot.model_dump(mode="json"))),
        parameters=Parameters(date_basis=snapshot.date_basis, as_of_year=snapshot.as_of_year),
        work_ids=groups, counts=counts, annual=annual, labels=labels, document_levels=levels,
        arxiv_share=_share(counts["arxiv_identified"], counts["eligible"]), diagnostics=diagnostics,
        formulas={
            "eligible": "screening == include AND no publication check marked retracted; one stored work ID once",
            "annual_count": "count eligible IDs with selected year == y, type int (not bool), 1000 <= y <= as_of_year",
            "annual_share": "annual_count[y] / count(valid nonfuture selected years in eligible IDs)",
            "label_count": "count eligible IDs whose deduplicated manually annotated labels contain label",
            "label_share": "label_count / count(all eligible IDs); multiple labels may total above 100%",
            "label_annual_share": "count eligible IDs with label and year y / annual_count[y]",
            "zero_denominator": "undefined (JSON null; empty CSV cell), never zero percent",
            "document_levels": "active document level counts over eligible IDs; missing document = metadata_only",
            "arxiv_share": "eligible IDs with nonempty arxiv_id / all eligible IDs; identifier coverage only",
        })


def _retrieval_markdown(events: list[SearchEvent]) -> str:
    """Present both log summaries and source responses without merging their evidence."""
    statuses = {"ok": "成功", "empty": "无结果", "partial": "部分返回", "unavailable": "不可用",
                "rate_limited": "限流", "unknown": "未知", "": "未知"}
    sources = {"crossref": "Crossref", "openalex": "OpenAlex", "semantic_scholar": "Semantic Scholar",
               "arxiv": "arXiv"}
    def status(value: str) -> str:
        return statuses.get(value.strip(), f"未知（原值：{value}）")
    def source(value: str) -> str:
        return sources.get(value.strip(), value.strip() or "未知（未记录）")
    def truncated(value: bool | None) -> str:
        return "未知" if value is None else "是" if value else "否"
    def row(values: list) -> str:
        return "| " + " | ".join(escape_md(str(value)) for value in values) + " |\n"
    lines = ["### 检索范围表\n\n"]
    if not events:
        lines.append("未记录检索日志；检索覆盖未知。\n")
    for index, event in enumerate(events, 1):
        lines += [f"#### 检索记录 {index}\n\n",
                  f"日志 ID：{escape_md(event.id)}；日志记录时间：{escape_md(event.created.strip() or '未知（未记录）')}。\n\n",
                  f"执行检索式：{escape_md(event.summary.query.strip() or '未知（未记录）')}。\n\n",
                  f"项目保留记录数：{event.summary.retained}；因项目候选上限丢弃：{event.summary.dropped_at_project_cap}。\n\n"]
        if event.responses:
            lines.append("| 来源 | 实际检索式 | 状态 | 检索时间 | 来源报告总量 | 返回记录数 | 截断 | 已记录错误 |\n"
                         "|---|---|---|---|---:|---:|---|---|\n")
            for response in event.responses:
                queries = "；".join(query.strip() for query in response.queries if query.strip())
                lines.append(row([source(response.source), queries or "未知（未记录）", status(response.status),
                                  (response.retrieved_at or "").strip() or "未知（未记录）",
                                  response.total if response.total is not None else "未知", response.records,
                                  truncated(response.truncated), "；".join(response.errors) or "无已记录错误"]))
        else:
            lines.append("未记录来源响应；来源报告总量、检索时间及实际检索式未知。\n")
        if event.summary.sources:
            lines.append("\n来源日志摘要（与来源响应分别保留，不以一方覆盖另一方）：\n\n"
                         "| 来源 | 摘要状态 | 摘要返回记录数 | 摘要截断 | 摘要已记录错误 |\n|---|---|---:|---|---|\n")
            lines.extend(row([source(item.source), status(item.status), item.records, truncated(item.truncated),
                              "；".join(item.errors) or "无已记录错误"]) for item in event.summary.sources)
        lines.append("\n")
    return "".join(lines)


def _payloads(snapshot: CorpusSnapshot, result: ComputationRecord) -> dict[str, bytes]:
    annual = [["year", "count", "denominator", "share", "current_year"]] + [
        [row.year, row.count, row.denominator, row.share, row.current_year] for row in result.annual]
    labels = [["label", "definition", "year", "count", "denominator", "share", "current_year"]]
    for row in result.labels:
        labels.append([row.label, row.definition, "all", row.count, row.denominator, row.share, ""])
        labels.extend([row.label, row.definition, year.year, year.count, year.denominator, year.share, year.current_year]
                      for year in row.annual)
    def proportion(value: float | None) -> str:
        return "未定义（分母为零）" if value is None else f"{100 * value:.2f}%"
    lines = [f"# {escape_md(snapshot.project.title)} — 冻结语料描述统计\n\n",
             f"日期口径：{snapshot.date_basis}；计算截止年：{snapshot.as_of_year}；研究修订：{snapshot.source_snapshot[0]}。\n\n",
             f"算法：{result.algorithm_version}；规范输入 SHA-256：`{result.input_hash}`。\n\n",
             "## 边界与诊断\n\n"]
    lines.extend(f"- {escape_md(d.message)}\n" for d in result.diagnostics)
    lines += ["\n## 计数与分母\n\n", "筛选分类互斥；撤稿数与筛选分类重叠。年份、标签、文档层级、关系和标识覆盖均限于合格纳入记录。\n\n",
              "| 指标 | 数量 |\n|---|---:|\n"]
    names = {"candidates": "候选总量", "included": "筛选纳入（含撤稿）", "excluded": "筛选排除", "pending": "待筛选",
             "retracted": "已标记撤稿（全部候选）", "eligible": "合格纳入", "missing_year": "年份缺失", "invalid_year": "非法年份",
             "future_year": "未来年份", "valid_year": "年度分母", "labelled": "有人工标签", "unlabelled": "无人工标签",
             "related": "疑似关系记录", "arxiv_identified": "有 arXiv 标识"}
    lines.extend(f"| {names[name]} | {count} |\n" for name, count in result.counts.items())
    lines.append(f"\narXiv 标识覆盖：{result.counts['arxiv_identified']}/{result.counts['eligible']}，"
                 f"{proportion(result.arxiv_share)}。\n\n")
    lines.append("活动文档覆盖（合格纳入记录）：\n\n| 文档层级 | 数量 |\n|---|---:|\n")
    level_names = {"metadata_only": "仅元数据", "abstract_only": "仅摘要", "full_text": "可读全文",
                   "needs_ocr": "需 OCR（文字识别）"}
    lines.extend(f"| {level_names[level]} | {count} |\n" for level, count in result.document_levels.items())
    lines.append("\n")
    lines += ["## 年度组成\n\n", "零值只说明本语料没有记录；当前年未结束。\n\n",
              "| 年份 | 数量 | 年度分母 | 语料内占比 | 未结束年 |\n|---|---:|---:|---:|---|\n"]
    lines.extend(f"| {row.year} | {row.count} | {row.denominator} | {proportion(row.share)} | {'是' if row.current_year else '否'} |\n"
                 for row in result.annual)
    lines += ["\n## 人工方法标签\n\n", "每篇同标签只计一次，允许多标签，总占比可超过 100%。逐年标签分母为当年合格记录数，见 labels.csv。\n\n"]
    if snapshot.annotations:
        lines.append(f"批注版本：{escape_md(snapshot.annotations.version)}；批注者：{escape_md(snapshot.annotations.annotator)}。\n\n")
    else:
        lines.append("未提供人工批注。\n\n")
    lines.append("| 标签 | 固定定义 | 数量 | 全部合格记录分母 | 占比 |\n|---|---|---:|---:|---:|\n")
    lines.extend(f"| {escape_md(row.label)} | {escape_md(row.definition)} | {row.count} | {row.denominator} | {proportion(row.share)} |\n"
                 for row in result.labels)
    configured_sources = "；".join(value.strip() for value in snapshot.project.sources if value.strip())
    configured_queries = "；".join(value.strip() for value in snapshot.project.queries if value.strip())
    lines += ["\n## 检索范围与日志\n\n", f"研究问题：{escape_md(snapshot.project.question.strip() or '未填写')}。\n\n",
              "配置来源：" + escape_md(configured_sources or "未配置") + "。\n\n",
              "配置检索式（不保证已执行）：" + escape_md(configured_queries or "未配置") + "。\n\n",
              f"候选上限 {snapshot.project.max_candidates}；阅读文档上限 {snapshot.project.max_documents}；模型调用上限 {snapshot.project.max_model_calls}。\n\n"]
    lines.append(_retrieval_markdown(snapshot.search_log))
    lines += ["\n## 计算公式\n\n"]
    formulas = [
        "合格纳入：已筛选纳入且没有任何撤稿标记的书目记录，每个库内文献 ID 只计一次。",
        "年度数量：选定年份为整数（排除布尔值）、介于 1000 与截止年之间，且等于该年的合格记录数。",
        "年度占比：该年合格记录数 ÷ 所有合法且不晚于截止年的合格记录数。",
        "标签数量：人工标签包含该标签的合格记录数，同篇重复标签只计一次。",
        "标签占比：该标签的合格记录数 ÷ 全部合格记录数；允许多标签，总占比可超过 100%。",
        "逐年标签占比：该年含该标签的合格记录数 ÷ 该年全部合格记录数。",
        "零分母：占比未定义，JSON 中为空值，CSV 中为空单元格，不解释为 0%。",
        "文档层级：按合格记录的活动文档层级计数，没有活动文档时计为仅元数据。",
        "arXiv 标识覆盖：带有非空 arXiv 标识的合格记录数 ÷ 全部合格记录数，仅代表标识存在。",
    ]
    lines.extend(f"- {formula}\n" for formula in formulas)
    lines += ["\n## 验证与溯源\n\n", "corpus.json 保留全部 Work 字段、活动文档 ID/内容及产物哈希、检索摘要、人工理由及年份来源。"
              "verify-trends 可离线核验文件哈希并完全重算。校验仅证明内部一致性，未签名文件不能证明外部真实性；"
              "改稿应保留为独立副本。原始文献内容及出版状态仍需人工复核。\n"]
    return {"corpus.json": _canonical(snapshot.model_dump(mode="json")),
            "computation.json": _canonical(result.model_dump(mode="json")),
            "annual.csv": _csv(annual).encode("utf-8"), "labels.csv": _csv(labels).encode("utf-8"),
            "report.md": "".join(lines).encode("utf-8")}


class Manifest(StrictRecord):
    algorithm_version: Literal["corpus-trends-1"]
    input_hash: Hash
    source_snapshot: tuple[int, int, int, Hash]
    files: dict[str, Hash]
    manifest_hash: Hash

    @field_validator("files")
    @classmethod
    def exact_files(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != FILES:
            raise ValueError("Manifest files must exactly match the five fixed artifact names")
        return value


@dataclass(frozen=True)
class VerifiedTrends:
    manifest: Manifest
    files: dict[str, bytes]
    snapshot: CorpusSnapshot
    computation: ComputationRecord
    summary: dict


def build_trends(store: Store, *, annotations: CorpusAnnotations | None = None,
                 date_basis: str = "record_year") -> Path:
    """Publish a fresh directory and atomic latest pointer behind the Store revision fence."""
    snapshot = freeze_corpus(store, annotations=annotations, date_basis=date_basis)
    result = compute_trends(snapshot)
    payloads = _payloads(snapshot, result)
    directory = store.root / "analyses" / "trends"
    directory.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=directory))
    for name, data in payloads.items():
        with (staging / name).open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    manifest = {"schema_version": 1, "algorithm_version": ALGORITHM_VERSION, "input_hash": result.input_hash,
                "source_snapshot": list(snapshot.source_snapshot),
                "files": {name: _hash(data) for name, data in payloads.items()}}
    manifest["manifest_hash"] = _hash(_canonical(manifest))
    atomic_json(staging / "manifest.json", manifest)
    version = "v-" + utcnow().replace(":", "").replace("+", "-").replace(".", "-") + "-" + uuid.uuid4().hex
    destination = directory / version
    with store.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        if snapshot_token(store, connection) != snapshot.source_snapshot:
            raise ValueError("Project changed during trend build; retry")
        staging.rename(destination)
        atomic_json(directory / "latest.json", {"schema_version": 1, "version": version,
                                                "snapshot": list(snapshot.source_snapshot),
                                                "input_hash": result.input_hash,
                                                "manifest_hash": manifest["manifest_hash"]})
    return destination


def load_verified_trends(directory: Path) -> VerifiedTrends:
    """Capture and validate the complete fixed artifact set from one byte batch."""
    directory = Path(directory).resolve()
    filesystem_directory = io_path(directory)

    def read(name: str) -> bytes:
        path = filesystem_directory / name
        if path.is_symlink() or path.resolve().parent != filesystem_directory or not path.is_file():
            raise ValueError(f"Missing or unsafe trend artifact: {name}")
        return path.read_bytes()

    try:
        manifest_bytes = read("manifest.json")
        manifest = Manifest.model_validate_json(manifest_bytes)
        raw_manifest = json.loads(manifest_bytes)
        raw_manifest.pop("manifest_hash")
        if _hash(_canonical(raw_manifest)) != manifest.manifest_hash:
            raise ValueError("Manifest hash mismatch")
        actual = {name: read(name) for name in sorted(FILES)}
        for name, content in actual.items():
            if _hash(content) != manifest.files[name]:
                raise ValueError(f"Artifact hash mismatch: {name}")
        snapshot = CorpusSnapshot.model_validate_json(actual["corpus.json"])
        result = ComputationRecord.model_validate_json(actual["computation.json"])
        recomputed = compute_trends(snapshot)
        if manifest.input_hash != recomputed.input_hash or manifest.source_snapshot != snapshot.source_snapshot:
            raise ValueError("Manifest input hash or source snapshot mismatch")
        if result != recomputed:
            raise ValueError("Computation does not match frozen corpus recomputation")
        expected = _payloads(snapshot, recomputed)
        for name, content in expected.items():
            if content != actual[name]:
                raise ValueError(f"Artifact does not match deterministic recomputation: {name}")
        summary = {"valid": True, "algorithm_version": ALGORITHM_VERSION, "input_hash": result.input_hash,
                   "snapshot": list(snapshot.source_snapshot), "files_verified": len(FILES) + 1}
        return VerifiedTrends(manifest=manifest, files={"manifest.json": manifest_bytes, **actual},
                              snapshot=snapshot, computation=result, summary=summary)
    except (OSError, TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"Trend verification failed: {exc}") from exc


def verify_trends(directory: Path) -> dict:
    """Keep the legacy exact summary while sharing the byte-identity validator."""
    return load_verified_trends(directory).summary
