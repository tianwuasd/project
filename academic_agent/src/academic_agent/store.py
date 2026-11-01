"""SQLite authority, immutable artifacts, conservative identities and fenced jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .bibliography import normalize_doi
from .documents import make_evidence, verify_span
from .filesystem import io_path, sqlite_path
from .models import Document, Evidence, Work, digest, utcnow


class StaleJobError(RuntimeError):
    pass


class StaleWorkError(RuntimeError):
    pass


class CandidateLimitError(ValueError):
    pass


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class JobTicket:
    id: str
    attempt_id: str
    lease_token: str
    revision: int
    cached: bool = False


def _normalize(value: str) -> str:
    return re.sub(r"\W+", "", value.casefold())


def _union(left: list, right: list) -> list:
    result, keys = [], set()
    for value in left + right:
        key = json.dumps(value.model_dump() if hasattr(value, "model_dump") else value, sort_keys=True)
        if key not in keys:
            result.append(value)
            keys.add(key)
    return result


def _source_aliases(work: Work) -> set[tuple[str, str]]:
    return set(work.source_ids.items()) | {
        (relation["source"], relation["id"]) for relation in work.relations
        if relation.get("type") == "source_id_alias" and "source" in relation and "id" in relation
    }


def _arxiv_version(identifier: str | None) -> int:
    match = re.search(r"v(\d+)$", identifier or "")
    return int(match[1]) if match else 0


def _abstract_provenance(work: Work) -> list[dict]:
    return [{"source": "arxiv_version", "arxiv_id": work.arxiv_id, "abstract": work.abstract}] \
        if work.arxiv_id and work.abstract else []


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        io_path(self.root).mkdir(parents=True, exist_ok=True)
        self.db = self.root / "research.db"
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
                INSERT OR IGNORE INTO meta VALUES ('revision', 0);
                INSERT OR IGNORE INTO meta VALUES ('schema_version', 1);
                CREATE TABLE IF NOT EXISTS works (id TEXT PRIMARY KEY, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, work_id TEXT NOT NULL REFERENCES works(id),
                    path TEXT NOT NULL, sha TEXT NOT NULL, active INTEGER NOT NULL, created TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS evidence (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id), data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS logs (id TEXT PRIMARY KEY, kind TEXT, data TEXT, created TEXT);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT, work_id TEXT, input_hash TEXT, backend TEXT,
                    status TEXT, attempt_id TEXT, lease_token TEXT, lease_until REAL, revision INTEGER,
                    path TEXT, sha TEXT, error TEXT, created TEXT);
                CREATE TABLE IF NOT EXISTS attempts (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), reservation TEXT,
                    usage TEXT, created TEXT);
            """)
            schema = connection.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
            if schema != 1:
                raise ValueError(f"Unsupported database schema version: {schema}")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(sqlite_path(self.db), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _bump(connection: sqlite3.Connection) -> None:
        connection.execute("UPDATE meta SET value=value+1 WHERE key='revision'")

    @property
    def revision(self) -> int:
        with self.connect() as connection:
            return connection.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]

    def works(self, included_only: bool = False) -> list[Work]:
        with self.connect() as connection:
            works = [Work.model_validate_json(row[0]) for row in connection.execute("SELECT data FROM works")]
        return [work for work in works if not included_only or work.screening == "include"]

    def work(self, work_id: str) -> Work:
        with self.connect() as connection:
            row = connection.execute("SELECT data FROM works WHERE id=?", (work_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown work: {work_id}")
            return Work.model_validate_json(row[0])

    def upsert_work(self, incoming: Work, *, max_works: int | None = None) -> Work:
        if max_works is not None and max_works < 0:
            raise ValueError("Candidate limit must be nonnegative")
        incoming = incoming.model_copy(deep=True)
        incoming.doi = normalize_doi(incoming.doi)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = [Work.model_validate_json(row[0]) for row in connection.execute("SELECT data FROM works")]
            match = None
            for old in existing:
                conflict = old.doi and incoming.doi and old.doi != incoming.doi
                same_source = bool(_source_aliases(old) & _source_aliases(incoming))
                same_arxiv = old.arxiv_id and incoming.arxiv_id and re.sub(
                    r"v\d+$", "", old.arxiv_id
                ) == re.sub(r"v\d+$", "", incoming.arxiv_id)
                same_bibliography = (
                    old.authors and incoming.authors
                    and _normalize(old.title) == _normalize(incoming.title)
                    and {_normalize(a) for a in old.authors} == {_normalize(a) for a in incoming.authors}
                    and old.year == incoming.year
                )
                if not conflict and (incoming.doi and old.doi == incoming.doi or same_source
                                     or same_arxiv or same_bibliography):
                    match = old
                    break
            if match:
                combined = match.model_copy(deep=True)
                for field in ("doi", "arxiv_id", "venue", "open_access_url", "year"):
                    if not getattr(combined, field) and getattr(incoming, field):
                        setattr(combined, field, getattr(incoming, field))
                if not combined.abstract and (not same_arxiv or
                                              _arxiv_version(incoming.arxiv_id) >= _arxiv_version(match.arxiv_id)):
                    combined.abstract = incoming.abstract
                if not combined.authors:
                    combined.authors = incoming.authors
                if incoming.arxiv_id and match.arxiv_id and incoming.arxiv_id != match.arxiv_id:
                    combined.relations = _union(combined.relations, [
                        {"type": "arxiv_version", "id": match.arxiv_id},
                        {"type": "arxiv_version", "id": incoming.arxiv_id},
                    ])
                    if same_arxiv and _arxiv_version(incoming.arxiv_id) > _arxiv_version(match.arxiv_id):
                        combined.arxiv_id = incoming.arxiv_id
                        combined.abstract = incoming.abstract
                combined.relations = _union(combined.relations, [
                    {"type": "source_id_alias", "source": source, "id": identifier}
                    for source, identifier in sorted(_source_aliases(match) | _source_aliases(incoming))
                    if source in match.source_ids and source in incoming.source_ids
                    and match.source_ids[source] != incoming.source_ids[source]
                ])
                combined.source_ids.update(incoming.source_ids)
                combined.urls = _union(combined.urls, incoming.urls)
                combined.provenance = _union(combined.provenance, incoming.provenance)
                combined.provenance = _union(combined.provenance,
                                             _abstract_provenance(match) + _abstract_provenance(incoming))
                combined.publication_checks = _union(combined.publication_checks, incoming.publication_checks)
                combined.relations = _union(combined.relations, incoming.relations)
            else:
                if max_works is not None and len(existing) >= max_works:
                    raise CandidateLimitError("Candidate limit reached")
                combined = incoming
                combined.id = "work_" + uuid.uuid4().hex[:16]
                combined.citekey = "ref_" + combined.id[5:]
                combined.record_revision = 0
                combined.provenance = _union(combined.provenance, _abstract_provenance(combined))
            if not match or combined != match:
                combined.record_revision = match.record_revision + 1 if match else 1
                connection.execute("INSERT INTO works VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                                   (combined.id, combined.model_dump_json()))
                self._bump(connection)
            return combined

    def update_work(self, work: Work) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT data FROM works WHERE id=?", (work.id,)).fetchone()
            if row is None:
                raise ValueError("Cannot update an unknown work")
            previous = Work.model_validate_json(row[0])
            if work.record_revision != previous.record_revision:
                raise StaleWorkError("Work changed since it was read; reload before updating")
            if work.citekey != previous.citekey:
                raise ValueError("Citation keys are immutable")
            updated = work.model_copy(deep=True)
            updated.publication_checks = _union(updated.publication_checks,
                                                [c for c in previous.publication_checks if c.status == "retracted"])
            if previous != updated:
                updated.record_revision += 1
                connection.execute("UPDATE works SET data=? WHERE id=?", (updated.model_dump_json(), work.id))
                self._bump(connection)
        work.record_revision = updated.record_revision
        work.publication_checks = updated.publication_checks

    def screen(self, work_id: str, decision: str, reason: str) -> None:
        if decision not in {"include", "exclude", "pending"} or not reason.strip():
            raise ValueError("Screening requires a valid decision and reason")
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT data FROM works WHERE id=?", (work_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown work: {work_id}")
            work = Work.model_validate_json(row[0])
            if (work.screening, work.screening_reason) != (decision, reason):
                work.screening, work.screening_reason = decision, reason
                work.record_revision += 1
                connection.execute("UPDATE works SET data=? WHERE id=?", (work.model_dump_json(), work_id))
                self._bump(connection)
        self.log("screening", {"work_id": work_id, "decision": decision, "reason": reason})

    def put_blob(self, payload: bytes, suffix: str = "json") -> tuple[str, str]:
        if suffix not in {"json", "pdf", "md", "txt"}:
            raise ValueError("Unsupported immutable file extension")
        sha = hashlib.sha256(payload).hexdigest()
        relative = f"artifacts/{sha}.{suffix}"
        path = self.root / relative
        physical = io_path(path)
        physical.parent.mkdir(parents=True, exist_ok=True)
        if physical.exists():
            if hashlib.sha256(physical.read_bytes()).hexdigest() != sha:
                raise ValueError("Immutable artifact was modified")
            return relative, sha
        fd, temporary = tempfile.mkstemp(dir=physical.parent, prefix=".pending-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, physical)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return relative, sha

    def read_blob(self, relative: str, sha: str) -> bytes:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Artifact path escapes project")
        content = io_path(path).read_bytes()
        if hashlib.sha256(content).hexdigest() != sha:
            raise ValueError("Artifact hash mismatch")
        return content

    def add_document(self, document: Document) -> Document:
        self.work(document.work_id)
        document = document.model_copy(deep=True)
        if document.source_path:
            original = Path(document.source_path)
            content = io_path(original).read_bytes()
            if hashlib.sha256(content).hexdigest() != document.content_hash:
                raise ValueError("Source changed after parsing")
            document.source_path, _ = self.put_blob(content, original.suffix.lower().lstrip("."))
        path, sha = self.put_blob(document.model_dump_json().encode("utf-8"))
        spans = make_evidence(document)
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT id FROM documents WHERE work_id=? AND active=1",
                                         (document.work_id,)).fetchone()
            if current and current[0] == document.id:
                return self.document(document.id)
            connection.execute("UPDATE documents SET active=0 WHERE work_id=?", (document.work_id,))
            connection.execute("INSERT INTO documents VALUES (?, ?, ?, ?, 1, ?) ON CONFLICT(id) DO UPDATE SET active=1",
                               (document.id, document.work_id, path, sha, utcnow()))
            for span in spans:
                connection.execute("INSERT OR IGNORE INTO evidence VALUES (?, ?, ?)",
                                   (span.id, document.id, span.model_dump_json()))
            self._bump(connection)
        return document

    def document(self, document_id: str) -> Document:
        with self.connect() as connection:
            row = connection.execute("SELECT path,sha FROM documents WHERE id=?", (document_id,)).fetchone()
        if row is None:
            raise ValueError("Document not found")
        return Document.model_validate_json(self.read_blob(row[0], row[1]))

    def active_document(self, work_id: str) -> Document | None:
        with self.connect() as connection:
            row = connection.execute("SELECT id FROM documents WHERE work_id=? AND active=1", (work_id,)).fetchone()
        return self.document(row[0]) if row else None

    def evidence(self, work_id: str) -> list[Evidence]:
        document = self.active_document(work_id)
        if document is None:
            return []
        with self.connect() as connection:
            spans = [Evidence.model_validate_json(row[0]) for row in connection.execute(
                "SELECT data FROM evidence WHERE document_id=? ORDER BY id", (document.id,))]
        if not all(verify_span(span, document) for span in spans):
            raise ValueError("Evidence no longer matches its document")
        return sorted(spans, key=lambda span: (span.page, span.start))

    def log(self, kind: str, data: Any) -> str:
        log_id = uuid.uuid4().hex
        with self.connect() as connection:
            connection.execute("INSERT INTO logs VALUES (?, ?, ?, ?)",
                               (log_id, kind, json.dumps(data, ensure_ascii=False), utcnow()))
        return log_id

    def logs(self, kind: str | None = None) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM logs WHERE kind=? ORDER BY created", (kind,)) if kind else connection.execute(
                "SELECT * FROM logs ORDER BY created")
            return [{"id": row["id"], "kind": row["kind"], "data": json.loads(row["data"]),
                     "created": row["created"]} for row in rows]

    def start_job(self, kind: str, work_id: str, input_hash: str, backend: str,
                  max_calls: int, timeout: int = 300) -> JobTicket:
        job_id = "job_" + digest(f"{kind}:{work_id}:{input_hash}:{backend}")[:32]
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            revision = connection.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row and row["status"] == "succeeded":
                self.read_blob(row["path"], row["sha"])
                return JobTicket(job_id, row["attempt_id"], row["lease_token"], row["revision"], True)
            if row and row["status"] == "running" and row["lease_until"] > time.time():
                raise RuntimeError("Job is already running")
            if row and row["status"] == "running":
                connection.execute("UPDATE attempts SET reservation='uncertain' WHERE id=?",
                                   (row["attempt_id"],))
            count = connection.execute("SELECT count(*) FROM attempts WHERE reservation IN ('reserved','spent','uncertain')").fetchone()[0]
            if count >= max_calls:
                raise BudgetExceeded("Model-call budget exhausted (including uncertain attempts)")
            attempt, lease = uuid.uuid4().hex, uuid.uuid4().hex
            connection.execute("""INSERT INTO jobs VALUES (?,?,?,?,?,'running',?,?,?,?,NULL,NULL,'',?)
                ON CONFLICT(id) DO UPDATE SET status='running',attempt_id=excluded.attempt_id,
                lease_token=excluded.lease_token,lease_until=excluded.lease_until,revision=excluded.revision,error=''""",
                               (job_id, kind, work_id, input_hash, backend, attempt, lease,
                                time.time() + timeout + 30, revision, utcnow()))
            connection.execute("INSERT INTO attempts VALUES (?, ?, 'reserved', NULL, ?)",
                               (attempt, job_id, utcnow()))
            return JobTicket(job_id, attempt, lease, revision)

    def finish_job(self, ticket: JobTicket, payload: dict, usage: dict | None = None) -> None:
        path, sha = self.put_blob(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (ticket.id,)).fetchone()
            revision = connection.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
            if (not row or row["status"] != "running" or row["attempt_id"] != ticket.attempt_id
                    or row["lease_token"] != ticket.lease_token or row["lease_until"] < time.time()
                    or row["revision"] != revision or ticket.revision != revision):
                raise StaleJobError("Late, cancelled, expired or stale-input result rejected")
            connection.execute("UPDATE jobs SET status='succeeded',path=?,sha=? WHERE id=?", (path, sha, ticket.id))
            connection.execute("UPDATE attempts SET reservation='spent',usage=? WHERE id=?",
                               (json.dumps(usage) if usage else None, ticket.attempt_id))

    def fail_job(self, ticket: JobTicket, error: str, uncertain: bool = True) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("UPDATE jobs SET status='failed',error=? WHERE id=? AND attempt_id=? AND status='running'",
                               (error, ticket.id, ticket.attempt_id))
            connection.execute("UPDATE attempts SET reservation=? WHERE id=? AND reservation='reserved'",
                               ("uncertain" if uncertain else "released", ticket.attempt_id))

    def cancel_job(self, job_id: str) -> None:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT attempt_id FROM jobs WHERE id=? AND status='running'", (job_id,)).fetchone()
            if row:
                connection.execute("UPDATE jobs SET status='cancelled' WHERE id=?", (job_id,))
                connection.execute("UPDATE attempts SET reservation='uncertain' WHERE id=?", (row[0],))

    def job_result(self, job_id: str) -> dict:
        with self.connect() as connection:
            row = connection.execute("SELECT path,sha FROM jobs WHERE id=? AND status='succeeded'", (job_id,)).fetchone()
        if row is None:
            raise ValueError("No successful job result")
        return json.loads(self.read_blob(row[0], row[1]))

    def jobs(self, kind: str | None = None) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute("SELECT * FROM jobs WHERE kind=? ORDER BY created", (kind,)) if kind else connection.execute(
                "SELECT * FROM jobs ORDER BY created")
            return [dict(row) for row in rows]

    def budget(self) -> dict:
        with self.connect() as connection:
            counts = {row[0]: row[1] for row in connection.execute(
                "SELECT reservation,count(*) FROM attempts GROUP BY reservation")}
        return {"attempts": counts, "consumed_or_reserved_calls": sum(counts.get(k, 0) for k in ("reserved", "spent", "uncertain")),
                "monetary_cost": None, "cost_status": "unknown"}
