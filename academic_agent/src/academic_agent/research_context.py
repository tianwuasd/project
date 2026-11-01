"""Current reviewed research, shared by writable workflows and readonly guides."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .config import load_config
from .documents import abstract_document, verify_span
from .exports import current_cards, decisions
from .filesystem import io_path, sqlite_uri
from .models import Document, Evidence, ReadingCard, ReviewReport, Verdict, Work, digest, utcnow
from .store import Store


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ResearchScope(_Strict):
    title: str
    sources: list[str]
    queries: list[str]


class ResearchCoverage(_Strict):
    total: int = Field(ge=0)
    included: int = Field(ge=0)
    pending: int = Field(ge=0)
    excluded: int = Field(ge=0)
    retracted: int = Field(ge=0)
    selected: int = Field(ge=0)


class ResearchGap(_Strict):
    work_id: str | None = None
    reason: str


class ResearchDecision(_Strict):
    claim_id: str
    work_id: str
    document_id: str
    card_hash: str
    verdict: Verdict
    rationale: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    evidence_ids: list[str]
    type: Literal["model", "manual"]


class ResearchSnapshot(_Strict):
    schema_version: Literal[1] = 1
    captured_at: str = Field(default_factory=utcnow)
    selection: Literal["default", "explicit"]
    require_supported: bool
    work_ids: list[str]
    works: dict[str, Work]
    documents: dict[str, Document]
    evidence: dict[str, Evidence]
    cards: dict[str, ReadingCard]
    decisions: dict[str, ResearchDecision]
    question: str
    scope: ResearchScope
    coverage: ResearchCoverage
    search_events: list[dict[str, JsonValue]]
    gaps: list[ResearchGap]

    @model_validator(mode="after")
    def valid_chains(self):
        if self.work_ids != sorted(set(self.work_ids)) or set(self.work_ids) != set(self.works):
            raise ValueError("Research selection must have unique, ordered known work IDs")
        if self.require_supported and not self.work_ids:
            raise ValueError("Research selection is empty")
        if (self.coverage.selected != len(self.work_ids)
                or self.coverage.total != self.coverage.included + self.coverage.pending + self.coverage.excluded
                or self.coverage.selected > self.coverage.included or self.coverage.retracted > self.coverage.total):
            raise ValueError("Research coverage is inconsistent")
        if any(key != work.id or work.screening != "include" or work.retracted
               for key, work in self.works.items()):
            raise ValueError("Research contains an ineligible work")
        by_work = {}
        for key, doc in self.documents.items():
            if key != doc.id or doc.work_id not in self.works or doc.work_id in by_work:
                raise ValueError("Research document has an invalid identity")
            if (doc.level not in {"full_text", "abstract_only"} or not doc.pages
                    or not any(p.text.strip() for p in doc.pages)
                    or len({p.number for p in doc.pages}) != len(doc.pages)):
                raise ValueError("Research document is not readable")
            if doc.parser == "source-abstract":
                expected = abstract_document(self.works[doc.work_id])
                if (doc.id, doc.content_hash, doc.pages) != (expected.id, expected.content_hash, expected.pages):
                    raise ValueError("Research source abstract is stale")
            by_work[doc.work_id] = doc
        for key, span in self.evidence.items():
            doc = self.documents.get(span.document_id)
            if key != span.id or not doc or not verify_span(span, doc):
                raise ValueError("Research contains foreign or invalid evidence")
        claims = {}
        for wid, card in self.cards.items():
            doc = by_work.get(wid)
            if wid != card.work_id or not doc or doc.id != card.document_id:
                raise ValueError("Research card has an invalid document")
            for claim in card.claims:
                if (not claim.id or claim.id in claims or claim.work_id != wid or not claim.evidence_ids
                        or any(e not in self.evidence or self.evidence[e].document_id != doc.id
                               for e in claim.evidence_ids)):
                    raise ValueError("Research claim has invalid evidence or identity")
                claims[claim.id] = claim
        for cid, decision in self.decisions.items():
            claim = claims.get(cid)
            card = self.cards.get(decision.work_id)
            if (not claim or cid != decision.claim_id or claim.work_id != decision.work_id or not card
                    or decision.document_id != card.document_id
                    or decision.card_hash != digest(card.model_dump_json())
                    or not set(decision.evidence_ids) <= set(claim.evidence_ids)
                    or not decision.rationale.strip() or not decision.reviewer.strip()
                    or decision.verdict == "supported" and not decision.evidence_ids):
                raise ValueError("Research decision has an invalid evidence dependency")
        if self.require_supported:
            supported_works = {self.decisions[c].work_id for c in self.supported_claim_ids}
            if supported_works != set(self.work_ids):
                raise ValueError("Every selected work needs a current readable document and supported claim")
        return self

    @property
    def supported_claim_ids(self) -> list[str]:
        return sorted(cid for cid, decision in self.decisions.items() if decision.verdict == "supported")

    def stable_payload(self) -> dict:
        """Full model context and fingerprint input; omit capture/operational counters."""
        value = self.model_dump(mode="json", exclude={"captured_at"})
        for work in value["works"].values():
            work.pop("record_revision", None)
        for document in value["documents"].values():
            document.pop("created_at", None)
        return value


def research_fingerprint(snapshot: ResearchSnapshot) -> str:
    # Revalidate mutable nested records instead of trusting Pydantic construction.
    checked = ResearchSnapshot.model_validate(snapshot.model_dump(), strict=True)
    return digest(json.dumps(checked.stable_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _blob(
    root: Path,
    relative: str,
    sha: str,
    *,
    source_hash_mismatches: dict[str, str] | None = None,
) -> bytes:
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError("Research document path escapes project")
    if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root)):
        raise ValueError("Research document path is a symlink")
    try:
        data = io_path(path).read_bytes()
    except OSError as exc:
        raise ValueError("Research document blob is missing or unreadable") from exc
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != sha and source_hash_mismatches is None:
        raise ValueError("Research document blob hash mismatch")
    if actual_hash != sha:
        source_hash_mismatches[relative] = actual_hash
    return data


class _Projection:
    """In-memory facade for existing card/decision rules; never initializes Store."""

    def __init__(self, works, documents, evidence, events):
        self._works, self._documents, self._evidence, self._events = works, documents, evidence, events

    def work(self, work_id):
        return self._works[work_id]

    def active_document(self, work_id):
        return self._documents.get(work_id)

    def evidence(self, work_id):
        return self._evidence.get(work_id, [])

    def logs(self, kind):
        return [e for e in self._events if e["kind"] == kind]


def _project(
    root: Path,
    connection: sqlite3.Connection,
    *,
    work_ids,
    require_supported,
    source_hash_mismatches: dict[str, str] | None = None,
) -> ResearchSnapshot:
    config = load_config(root)
    works = {}
    for row in connection.execute("SELECT id,data FROM works ORDER BY id"):
        work = Work.model_validate_json(row[1], strict=True)
        if row[0] != work.id:
            raise ValueError("Work row identity mismatch")
        works[work.id] = work
    if work_ids is not None and (not work_ids or len(work_ids) != len(set(work_ids))):
        raise ValueError("Explicit research selection is empty or duplicated")
    selected = sorted(work_ids if work_ids is not None else
                      [w.id for w in works.values() if w.screening == "include" and not w.retracted])
    for wid in selected:
        if wid not in works or works[wid].screening != "include" or works[wid].retracted:
            raise ValueError("Unknown, excluded, pending or retracted research work")
    documents, evidence, events, gaps = {}, {}, [], []
    for wid in selected:
        rows = list(connection.execute("SELECT id,path,sha FROM documents WHERE work_id=? AND active=1", (wid,)))
        if len(rows) > 1:
            raise ValueError("Multiple active research documents")
        if not rows:
            gaps.append(ResearchGap(work_id=wid, reason="missing_document"))
            continue
        row = rows[0]
        doc = Document.model_validate_json(_blob(root, row[1], row[2]), strict=True)
        if doc.id != row[0] or doc.work_id != wid:
            raise ValueError("Document row identity mismatch")
        if doc.source_path:
            _blob(
                root,
                doc.source_path,
                doc.content_hash,
                source_hash_mismatches=source_hash_mismatches,
            )
        if doc.parser == "source-abstract" and doc.id != abstract_document(works[wid]).id:
            gaps.append(ResearchGap(work_id=wid, reason="stale_source_abstract"))
            continue
        if doc.level not in {"full_text", "abstract_only"} or not any(p.text.strip() for p in doc.pages):
            gaps.append(ResearchGap(work_id=wid, reason="unreadable_document"))
            continue
        spans = []
        for span_row in connection.execute("SELECT id,data FROM evidence WHERE document_id=? ORDER BY id", (doc.id,)):
            span = Evidence.model_validate_json(span_row[1], strict=True)
            if span.id != span_row[0] or not verify_span(span, doc):
                raise ValueError("Foreign or invalid research evidence")
            spans.append(span)
        documents[wid], evidence[wid] = doc, spans
        if doc.level == "abstract_only":
            gaps.append(ResearchGap(work_id=wid, reason="abstract_only"))
    for row in connection.execute(
        "SELECT kind,data,created,id FROM logs WHERE kind IN ('reading_card','review','manual_review','search') "
        "ORDER BY created,rowid"
    ):
        data = json.loads(row[1])
        if not isinstance(data, dict):
            raise ValueError("Malformed research event")
        if row[0] == "reading_card":
            card = ReadingCard.model_validate(data.get("card"), strict=True)
            if card.work_id not in selected:
                continue
            doc = documents.get(card.work_id)
            if doc and card.document_id == doc.id:
                spans = {span.id for span in evidence[card.work_id]}
                if (len({c.id for c in card.claims}) != len(card.claims)
                        or any(not c.id or c.work_id != card.work_id or not set(c.evidence_ids) <= spans
                               for c in card.claims)):
                    raise ValueError("Current reading card has invalid claim or evidence identities")
        elif row[0] == "review":
            report = ReviewReport.model_validate(data.get("report"), strict=True)
            if report.work_id not in selected:
                continue
            if not isinstance(data.get("card_hash"), str):
                raise ValueError("Malformed review card hash")
            if len({v.claim_id for v in report.verdicts}) != len(report.verdicts):
                raise ValueError("Duplicate review decisions")
        elif row[0] == "manual_review":
            # Validate manual input too: exports.decisions intentionally accepts dicts.
            ResearchDecision.model_validate({**data, "type": "manual"})
        events.append({"kind": row[0], "data": data, "created": row[2], "id": row[3]})
    projection = _Projection(works, documents, evidence, events)
    cards = current_cards(projection)
    cards_by_hash = {digest(card.model_dump_json()): card for card in cards.values()}
    for event in projection.logs("manual_review"):
        manual = ResearchDecision.model_validate({**event["data"], "type": "manual"})
        card = cards_by_hash.get(manual.card_hash)
        if not card:
            continue  # An obsolete card hash must not make historical references current.
        claim = next((claim for claim in card.claims if claim.id == manual.claim_id), None)
        if (not claim or manual.work_id != card.work_id or manual.document_id != card.document_id
                or claim.work_id != manual.work_id
                or not set(manual.evidence_ids) <= set(claim.evidence_ids)
                or manual.verdict == "supported" and not manual.evidence_ids
                or not manual.rationale.strip() or not manual.reviewer.strip()):
            raise ValueError("Current manual review has invalid claim, work, document or evidence references")
    for event in projection.logs("review"):
        report = ReviewReport.model_validate(event["data"]["report"], strict=True)
        card = cards.get(report.work_id)
        if (not card or card.document_id != report.document_id
                or event["data"]["card_hash"] != digest(card.model_dump_json())):
            continue
        claims = {claim.id: claim for claim in card.claims}
        if set(claims) != {v.claim_id for v in report.verdicts}:
            raise ValueError("Current review has missing or foreign claim decisions")
        if any(not verdict.rationale.strip()
               or not set(verdict.evidence_ids) <= set(claims[verdict.claim_id].evidence_ids)
               or verdict.verdict == "supported" and not verdict.evidence_ids
               for verdict in report.verdicts):
            raise ValueError("Current review has malformed evidence decisions")
    verdicts = {}
    for cid, entry in decisions(projection, cards).items():
        if entry["type"] == "model":
            card = next(card for card in cards.values() if any(c.id == cid for c in card.claims))
            entry = {**entry, "work_id": card.work_id, "document_id": card.document_id,
                     "card_hash": digest(card.model_dump_json())}
            entry.pop("schema_version", None)
        verdicts[cid] = ResearchDecision.model_validate(entry)
    for wid in selected:
        if wid not in cards:
            gaps.append(ResearchGap(work_id=wid, reason="missing_current_card"))
        if not any(d.work_id == wid and d.verdict == "supported" for d in verdicts.values()):
            gaps.append(ResearchGap(work_id=wid, reason="missing_supported_claim"))
        if wid in cards and any(c.id not in verdicts for c in cards[wid].claims):
            gaps.append(ResearchGap(work_id=wid, reason="pending_review"))
    searches = [e for e in events if e["kind"] == "search"]
    if not searches:
        gaps.append(ResearchGap(reason="not_searched"))
    for event in searches:
        summary = event["data"].get("summary", {})
        if not isinstance(summary, dict) or not isinstance(summary.get("sources", []), list):
            raise ValueError("Malformed search coverage")
        for source in summary.get("sources", []):
            if not isinstance(source, dict):
                raise ValueError("Malformed search source")
            if source.get("status") not in {"ok", "empty"}:
                gaps.append(ResearchGap(reason="search_incomplete"))
            if source.get("truncated"):
                gaps.append(ResearchGap(reason="search_truncated"))
        if summary.get("dropped_at_project_cap"):
            gaps.append(ResearchGap(reason="project_cap_reached"))
    return ResearchSnapshot(
        selection="default" if work_ids is None else "explicit", require_supported=require_supported,
        work_ids=selected, works={wid: works[wid] for wid in selected},
        documents={d.id: d for d in sorted(documents.values(), key=lambda d: d.id)},
        evidence={e.id: e for e in sorted((e for spans in evidence.values() for e in spans), key=lambda e: e.id)},
        cards=dict(sorted(cards.items())), decisions=dict(sorted(verdicts.items())), question=config.question,
        scope=ResearchScope(title=config.title, sources=config.sources, queries=config.queries),
        coverage=ResearchCoverage(total=len(works), included=sum(w.screening == "include" for w in works.values()),
                                  pending=sum(w.screening == "pending" for w in works.values()),
                                  excluded=sum(w.screening == "exclude" for w in works.values()),
                                  retracted=sum(w.retracted for w in works.values()), selected=len(selected)),
        search_events=searches, gaps=gaps,
    )


def _read_once(root, work_ids, require_supported, *, source_hash_mismatches=None):
    # URI-encode literal # and %; normal transactions see committed WAL pages.
    with closing(sqlite3.connect(sqlite_uri(root / "research.db"), uri=True)) as connection:
        connection.execute("BEGIN")
        return _project(
            root,
            connection,
            work_ids=work_ids,
            require_supported=require_supported,
            source_hash_mismatches=source_hash_mismatches,
        )


def _diagnostic_projection(
    root: Path,
    *,
    work_ids: list[str] | None,
    require_supported: bool,
) -> tuple[ResearchSnapshot, dict[str, str]]:
    """Read one validated projection while recording stable source-only drift."""
    root = Path(root).resolve()
    if not io_path(root / "research.db").is_file() or not io_path(root / "project.json").is_file():
        raise ValueError("Missing research state")
    source_hash_mismatches: dict[str, str] = {}
    snapshot = _read_once(
        root,
        work_ids,
        require_supported,
        source_hash_mismatches=source_hash_mismatches,
    )
    return snapshot, source_hash_mismatches


def read_research(root: Path, *, work_ids: list[str] | None = None,
                  require_supported: bool = True) -> ResearchSnapshot:
    root = root.resolve()
    if not io_path(root / "research.db").is_file() or not io_path(root / "project.json").is_file():
        raise ValueError("Missing research state: existing research.db and project.json are required")
    try:
        before = _read_once(root, work_ids, require_supported)
        after = _read_once(root, work_ids, require_supported)
        if research_fingerprint(before) != research_fingerprint(after):
            raise ValueError("Research inputs changed during read; retry")
        return before
    except (OSError, sqlite3.Error, KeyError, TypeError) as exc:
        raise ValueError("Research state is missing, malformed or unreadable") from exc


def diagnose_research_state(
    root: Path,
    snapshot: ResearchSnapshot,
    *,
    work_ids: list[str] | None = None,
) -> tuple[Literal["current", "changed", "source_changed", "invalid"], ResearchSnapshot | None]:
    """Return a fully validated projection and distinguish valid drift from invalid state."""
    root = Path(root).resolve()
    try:
        snapshot = ResearchSnapshot.model_validate(snapshot.model_dump(mode="json"), strict=True)
        expected = research_fingerprint(snapshot)
        selection = work_ids
        if selection is None and snapshot.selection == "explicit":
            selection = snapshot.work_ids
        before, before_sources = _diagnostic_projection(
            root, work_ids=selection, require_supported=snapshot.require_supported
        )
        after, after_sources = _diagnostic_projection(
            root, work_ids=selection, require_supported=snapshot.require_supported
        )
        before_fingerprint = research_fingerprint(before)
        if before_fingerprint != research_fingerprint(after) or before_sources != after_sources:
            raise ValueError("Research inputs changed during diagnostic read")
    except (AttributeError, OSError, sqlite3.Error, ValueError, KeyError, TypeError):
        return "invalid", None
    if before_sources:
        return "source_changed", before
    if before_fingerprint != expected:
        return "changed", before
    return "current", before


def research_state_currentness(root: Path, snapshot: ResearchSnapshot) -> dict:
    """Verify current research state, classifying only isolated source hash drift as stale."""
    state, _ = diagnose_research_state(root, snapshot)
    if state == "invalid":
        return {"current": False, "gaps": [{"reason": "research_inputs_invalid"}]}
    if state in {"changed", "source_changed"}:
        return {"current": False, "gaps": [{"reason": "research_inputs_changed"}]}
    return {"current": True, "gaps": []}


def freeze_research(store: Store, *, work_ids: list[str] | None = None,
                    require_supported: bool = True) -> ResearchSnapshot:
    return read_research(store.root, work_ids=work_ids, require_supported=require_supported)


def assert_research_current(store: Store, snapshot: ResearchSnapshot) -> None:
    current = freeze_research(store, work_ids=snapshot.work_ids if snapshot.selection == "explicit" else None,
                              require_supported=snapshot.require_supported)
    if research_fingerprint(current) != research_fingerprint(snapshot):
        raise ValueError("Research inputs changed; freeze current evidence again")


def _assert_locked(store: Store, connection: sqlite3.Connection, snapshot: ResearchSnapshot) -> None:
    """Caller holds BEGIN IMMEDIATE; repeat external config/blob reads under that fence."""
    expected = research_fingerprint(snapshot)
    for _ in range(2):
        current = _project(store.root, connection,
                           work_ids=snapshot.work_ids if snapshot.selection == "explicit" else None,
                           require_supported=snapshot.require_supported)
        if research_fingerprint(current) != expected:
            raise ValueError("Research inputs changed before publication")
