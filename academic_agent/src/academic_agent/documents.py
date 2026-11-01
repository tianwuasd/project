"""Local text-layer parsing and exact, versioned evidence spans."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

from pypdf import PdfReader

from .filesystem import io_path
from .models import Document, Evidence, Page, Work, digest


class Parser(Protocol):
    def parse(self, path: Path, work_id: str) -> Document: ...


def parse_document(path: Path, work_id: str) -> Document:
    path = Path(path).resolve()
    physical = io_path(path)
    if not physical.is_file():
        raise FileNotFoundError(path)
    if physical.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("Document exceeds 50 MiB")
    content_hash = hashlib.sha256(physical.read_bytes()).hexdigest()
    diagnostics: list[str] = []
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(physical)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF: provide a readable copy")
        if len(reader.pages) > 500:
            raise ValueError("PDF exceeds the 500-page parsing limit")
        pages = [Page(number=i + 1, text=page.extract_text() or "") for i, page in enumerate(reader.pages)]
        empty = [page.number for page in pages if not page.text.strip()]
        if empty:
            diagnostics.append(f"No text layer on pages {empty}; OCR or blank-page review required")
        level = "needs_ocr" if empty or not pages else "full_text"
        parser = "pypdf-text-layer"
    elif path.suffix.lower() in {".md", ".txt"}:
        text = physical.read_text(encoding="utf-8-sig")
        pages = [Page(number=1, text=text)]
        level = "full_text" if text.strip() else "metadata_only"
        parser = "utf8-text"
    else:
        raise ValueError("Supported documents: .pdf, .md, .txt")
    if sum(len(page.text) for page in pages) > 2_000_000:
        raise ValueError("Parsed document exceeds two million characters")
    return Document(
        id="doc_" + digest(work_id + content_hash)[:24],
        work_id=work_id,
        content_hash=content_hash,
        source_path=str(path),
        level=level,
        pages=pages,
        parser=parser,
        diagnostics=diagnostics,
    )


def abstract_document(work: Work) -> Document:
    content_hash = digest(work.abstract)
    return Document(
        id="abs_" + digest(work.id + content_hash)[:24],
        work_id=work.id,
        content_hash=content_hash,
        level="abstract_only" if work.abstract.strip() else "metadata_only",
        pages=[Page(number=0, text=work.abstract)] if work.abstract.strip() else [],
        parser="source-abstract",
    )


def make_evidence(document: Document, chunk_size: int = 1200) -> list[Evidence]:
    if chunk_size < 50:
        raise ValueError("Evidence chunks must contain at least 50 characters")
    evidence = []
    for page in document.pages:
        start = 0
        while start < len(page.text):
            end = min(start + chunk_size, len(page.text))
            if end < len(page.text):
                boundary = page.text.rfind("\n", start + chunk_size // 2, end)
                if boundary > start:
                    end = boundary + 1
            text = page.text[start:end]
            if text.strip():
                evidence.append(Evidence(
                    id="ev_" + digest(f"{document.id}:{page.number}:{start}:{end}")[:24],
                    work_id=document.work_id,
                    document_id=document.id,
                    page=page.number,
                    start=start,
                    end=end,
                    text=text,
                    text_hash=digest(text),
                ))
            start = end
    return evidence


def verify_span(evidence: Evidence, document: Document) -> bool:
    pages = {page.number: page.text for page in document.pages}
    return (
        evidence.document_id == document.id
        and evidence.work_id == document.work_id
        and evidence.page in pages
        and 0 <= evidence.start < evidence.end <= len(pages[evidence.page])
        and pages[evidence.page][evidence.start:evidence.end] == evidence.text
        and digest(evidence.text) == evidence.text_hash
    )
