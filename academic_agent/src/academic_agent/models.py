"""Versioned contracts shared across replaceable research components."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1


class PublicationCheck(Record):
    source: str
    status: Literal["unknown", "no_notice_found", "retracted", "corrected"] = "unknown"
    checked_at: str = Field(default_factory=utcnow)
    detail: str = ""


class Work(Record):
    id: str = ""
    record_revision: int = Field(default=0, ge=0)
    title: str = Field(min_length=1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source_ids: dict[str, str] = Field(default_factory=dict)
    abstract: str = ""
    urls: list[str] = Field(default_factory=list)
    open_access_url: str | None = None
    venue: str = ""
    provenance: list[dict[str, Any]] = Field(default_factory=list)
    publication_checks: list[PublicationCheck] = Field(default_factory=list)
    relations: list[dict[str, str]] = Field(default_factory=list)
    citekey: str = ""
    screening: Literal["include", "exclude", "pending"] = "pending"
    screening_reason: str = ""

    @property
    def retracted(self) -> bool:
        return any(c.status == "retracted" for c in self.publication_checks)


class SearchPage(Record):
    source: str
    records: list[Work] = Field(default_factory=list)
    status: Literal["ok", "empty", "partial", "unavailable", "rate_limited"] = "ok"
    total: int | None = None
    truncated: bool = False
    queries: list[str] = Field(default_factory=list)
    raw_responses: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    retrieved_at: str = Field(default_factory=utcnow)


class Page(Record):
    number: int = Field(ge=0)
    text: str


class Document(Record):
    id: str
    work_id: str
    content_hash: str
    source_path: str = ""
    level: Literal["metadata_only", "abstract_only", "full_text", "needs_ocr"]
    pages: list[Page] = Field(default_factory=list)
    parser: str = ""
    diagnostics: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utcnow)


class Evidence(Record):
    id: str
    work_id: str
    document_id: str
    page: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str
    text_hash: str


Category = Literal["problem", "method", "data", "result", "limitation", "contribution"]
Verdict = Literal["supported", "partial", "contradicted", "insufficient"]


class Claim(Record):
    id: str = ""
    work_id: str = ""
    text: str = Field(min_length=1)
    category: Category
    evidence_ids: list[str] = Field(min_length=1)


class ReadingCard(Record):
    work_id: str
    claims: list[Claim] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    backend: str = ""
    document_id: str = ""


class ReviewVerdict(Record):
    claim_id: str
    verdict: Verdict
    rationale: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReviewReport(Record):
    work_id: str
    document_id: str
    verdicts: list[ReviewVerdict]
    backend: str = ""
    scope: str = ""


class ModelJob(Record):
    id: str
    kind: Literal["read", "review", "ask", "synthesize"]
    instruction: str
    payload: dict[str, Any]
    output_schema: dict[str, Any]
    timeout_seconds: int = Field(default=300, ge=1, le=1800)
    max_input_chars: int = Field(default=100_000, ge=100, le=500_000)


class ModelResult(Record):
    status: Literal["succeeded", "failed", "timed_out", "needs_input"]
    backend: str
    data: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] | None = None
    error: str = ""


class ProjectConfig(Record):
    title: str = Field(min_length=1)
    question: str = ""
    sources: list[str] = Field(default_factory=lambda: ["openalex", "semantic_scholar", "crossref"])
    queries: list[str] = Field(default_factory=list)
    max_candidates: int = Field(default=100, ge=1, le=1000)
    max_documents: int = Field(default=20, ge=1, le=100)
    max_model_calls: int = Field(default=60, ge=1, le=1000)
    model_workers: int = Field(default=1, ge=1, le=3)
    backend: Literal["codex", "demo", "exchange"] = "codex"
    model: str | None = None
    proxy_url: str | None = "http://127.0.0.1:7897"
    timeout_seconds: int = Field(default=300, ge=1, le=1800)
    max_input_chars: int = Field(default=100_000, ge=100, le=500_000)
