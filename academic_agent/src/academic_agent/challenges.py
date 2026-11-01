"""Bounded counterexample retrieval, durable round recovery and readonly coverage."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from .config import load_config
from .filesystem import io_path, sqlite_uri
from .innovation_analysis import validate_candidate_batch, validate_grounding_review
from .innovation_models import (
    CandidateBatch,
    ChallengeHit,
    ChallengeQueryReceipt,
    ChallengeRound,
    ChallengeTask,
    GroundingReview,
)
from .models import Document, Work, digest
from .research_artifacts import _check_history, load_research, publish_research
from .research_context import (
    ResearchSnapshot,
    _blob,
    _diagnostic_projection,
    diagnose_research_state,
    freeze_research,
    read_research,
    research_fingerprint,
)
from .workflow import search_query_key

if TYPE_CHECKING:
    from .workflow import Runner


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _versions(root: Path, kind: str):
    base = root / "analyses" / kind
    _check_history(base)
    return [(path, load_research(path)) for path in sorted(base.glob("v-*"))]


def _candidate_from_artifact(directory: Path, artifact: dict, candidate_id: str) -> dict:
    if set(artifact["files"]) != {
        "candidates.json", "grounding-review.json", "report.md", "references.bib", "evidence.json",
    }:
        raise ValueError("Innovation version has unexpected payloads")
    snapshot = ResearchSnapshot.model_validate(artifact["manifest"]["snapshot"], strict=True)
    batch = validate_candidate_batch(snapshot, CandidateBatch.model_validate_json(
        artifact["files"]["candidates.json"], strict=True))
    review = validate_grounding_review(snapshot, batch, GroundingReview.model_validate_json(
        artifact["files"]["grounding-review.json"], strict=True))
    candidate = next((item for item in batch.candidates if item.id == candidate_id), None)
    if candidate is None:
        raise ValueError("Unknown candidate in verified innovation version")
    return {"candidate": candidate, "batch": batch, "review": review, "snapshot": snapshot,
            "version_path": Path(directory).resolve(), "manifest_hash": artifact["manifest"]["manifest_hash"]}


def load_candidate_version(directory: Path, candidate_id: str) -> dict:
    """Load one explicit immutable innovation version without consulting latest."""
    directory = Path(directory).resolve()
    return _candidate_from_artifact(directory, load_research(directory), candidate_id)


def load_candidate(root: Path, candidate_id: str) -> dict:
    """Find a verified historical proposal without creating a Store or model job.

    Identical republications resolve to the lexicographically first host version;
    round receipts pin their original manifest independently of this selection.
    """
    found = []
    for path, artifact in _versions(Path(root), "innovation"):
        try:
            found.append(_candidate_from_artifact(path, artifact, candidate_id))
        except ValueError as exc:
            if str(exc) != "Unknown candidate in verified innovation version":
                raise
    if not found:
        raise ValueError("Unknown candidate in verified innovation history")
    if any((item["candidate"], item["batch"].input_hash) !=
           (found[0]["candidate"], found[0]["batch"].input_hash) for item in found[1:]):
        raise ValueError("Candidate identity has inconsistent historical content")
    return found[0]


class _PremiseError(ValueError):
    def __init__(self, reason: str):
        super().__init__(f"Candidate is stale: {reason}")
        self.reason = reason


def _assert_premises(root: Path, loaded: dict) -> None:
    """Coverage growth is allowed; original selected evidence and scope stay fixed."""
    historical = loaded["snapshot"]
    state, current = diagnose_research_state(root, historical, work_ids=historical.work_ids)
    if state == "invalid" or current is None:
        raise _PremiseError("candidate_evidence_invalid") from None
    if (current.question != historical.question or current.scope.title != historical.scope.title
            or current.scope.queries != historical.scope.queries):
        raise _PremiseError("candidate_scope_changed")
    for field in ("documents", "evidence", "cards", "decisions"):
        old = historical.model_dump(mode="json")[field]
        new = current.model_dump(mode="json")[field]
        if field == "documents":
            for values in (old, new):
                for document in values.values():
                    document.pop("created_at", None)
        if old != new:
            raise _PremiseError("candidate_evidence_changed")
    # Notice refresh and conservative identity enrichment do not rewrite premises.
    fields = {"title", "authors", "year", "doi", "arxiv_id", "abstract"}
    for wid in historical.work_ids:
        if any(getattr(historical.works[wid], field) != getattr(current.works[wid], field) for field in fields):
            raise _PremiseError("candidate_source_changed")
    if state == "source_changed":
        raise _PremiseError("candidate_evidence_changed")


def candidate_version_currentness(root: Path, loaded: dict) -> dict:
    """Check the premises of one already selected candidate version."""
    candidate_id = loaded.get("candidate").id if loaded.get("candidate") is not None else ""
    try:
        _assert_premises(Path(root), loaded)
    except _PremiseError as exc:
        return {"candidate_id": candidate_id, "current": False, "gaps": [{"reason": exc.reason}]}
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return {"candidate_id": candidate_id, "current": False,
                "gaps": [{"reason": "candidate_artifact_invalid"}]}
    return {"candidate_id": candidate_id, "current": True, "gaps": []}


def candidate_currentness(root: Path, candidate_id: str) -> dict:
    """Readonly currentness using exactly the challenge premise policy, with safe gaps."""
    try:
        loaded = load_candidate(root, candidate_id)
    except (ValueError, OSError):
        return {"candidate_id": candidate_id, "current": False,
                "gaps": [{"reason": "candidate_artifact_invalid"}]}
    return candidate_version_currentness(root, loaded)


def _read_rows(root: Path) -> tuple[list[dict], dict[str, Work]]:
    database = root / "research.db"
    if not io_path(database).is_file():
        raise ValueError("Missing research state: existing research.db is required")
    with closing(sqlite3.connect(sqlite_uri(database), uri=True)) as connection:
        connection.execute("BEGIN")
        events = [{"id": row[0], "kind": row[1], "data": json.loads(row[2]), "created": row[3]}
                  for row in connection.execute("SELECT id,kind,data,created FROM logs ORDER BY rowid")]
        works = {}
        for wid, data in connection.execute("SELECT id,data FROM works ORDER BY id"):
            work = Work.model_validate_json(data, strict=True)
            if work.id != wid:
                raise ValueError("Work row identity mismatch")
            works[wid] = work
    return events, works


def _identity(loaded: dict, number: int, sources: list[str], limit: int) -> str:
    return digest(_canonical({"candidate": loaded["candidate"].model_dump(mode="json"),
                              "input_hash": loaded["batch"].input_hash, "round": number,
                              "queries": loaded["candidate"].targeted_queries,
                              "sources": sources, "effective_limit": limit}))


def _receipt(event: dict, queries: list[str], sources: list[str], limit: int) -> ChallengeQueryReceipt:
    data = event["data"]
    summary = data["summary"]
    if (summary["query"] not in queries or [s["source"] for s in summary["sources"]] != sources
            or data.get("effective_limit") != limit):
        raise ValueError("Search event does not match challenge query scope")
    available = "retained_work_ids" in data
    complete = bool(sources) and available and all(s["status"] in {"ok", "empty"} for s in summary["sources"])
    return ChallengeQueryReceipt(
        query=summary["query"], query_key=data["query_key"], search_event_id=event["id"],
        search_event_hash=digest(_canonical(event)),
        request_id=data["request_id"],
        created=event["created"], effective_limit=limit, sources=summary["sources"],
        project_candidate_cap=data["project_candidate_cap"],
        retained_work_ids=data.get("retained_work_ids", []), retained_ids_available=available,
        dropped_at_project_cap=summary["dropped_at_project_cap"], complete=complete,
    )


def _active_documents(
    root: Path,
    work_ids: list[str],
    *,
    source_hash_mismatches: dict[str, str] | None = None,
) -> dict[str, Document]:
    """Verify current attachments independently of evidence-selection eligibility."""
    documents = {}
    with closing(sqlite3.connect(sqlite_uri(root / "research.db"), uri=True)) as connection:
        connection.execute("BEGIN")
        for wid in work_ids:
            rows = list(connection.execute("SELECT id,path,sha FROM documents WHERE work_id=? AND active=1", (wid,)))
            if len(rows) > 1:
                raise ValueError("Multiple active challenge documents")
            if not rows:
                continue
            document_id, path, sha = rows[0]
            document = Document.model_validate_json(_blob(root, path, sha), strict=True)
            if document.id != document_id or document.work_id != wid:
                raise ValueError("Challenge document row identity mismatch")
            if document.source_path:
                _blob(
                    root,
                    document.source_path,
                    document.content_hash,
                    source_hash_mismatches=source_hash_mismatches,
                )
            documents[wid] = document
    return documents


def _hits(
    root: Path,
    retained: list[str],
    *,
    verified_research: ResearchSnapshot | None = None,
) -> list[ChallengeHit]:
    if verified_research is None:
        before = read_research(root, require_supported=False)
        before_research_sources = after_research_sources = None
    else:
        state, current_premise = diagnose_research_state(root, verified_research)
        if state == "invalid" or current_premise is None:
            raise ValueError("Research is invalid while reading challenge hit coverage")
        try:
            before, before_research_sources = _diagnostic_projection(
                root, work_ids=None, require_supported=False
            )
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Research is invalid while reading challenge hit coverage") from exc
    _, works = _read_rows(root)
    before_source_hashes: dict[str, str] | None = {} if verified_research is not None else None
    after_source_hashes: dict[str, str] | None = {} if verified_research is not None else None
    active_documents = _active_documents(
        root, retained, source_hash_mismatches=before_source_hashes
    )
    _, after_works = _read_rows(root)
    after_active_documents = _active_documents(
        root, retained, source_hash_mismatches=after_source_hashes
    )
    if verified_research is None:
        after = read_research(root, require_supported=False)
    else:
        try:
            after, after_research_sources = _diagnostic_projection(
                root, work_ids=None, require_supported=False
            )
        except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
            raise ValueError("Research is invalid while reading challenge hit coverage") from exc
    if (research_fingerprint(before) != research_fingerprint(after)
            or before_research_sources != after_research_sources
            or active_documents != after_active_documents
            or before_source_hashes != after_source_hashes
            or any(works.get(wid) != after_works.get(wid) for wid in retained)):
        raise ValueError("Research changed while reading challenge hit coverage")
    hits = []
    for wid in retained:
        work = works.get(wid)
        states = []
        active_document = active_documents.get(wid)
        level = active_document.level if active_document else "metadata_only"
        publication = "unknown"
        if work is None:
            states.append("missing_work")
        else:
            checks = work.publication_checks
            if work.retracted:
                publication = "retracted"
                states.append("retracted")
            elif any(c.status == "corrected" for c in checks):
                publication = "corrected"
            elif checks and all(c.status == "no_notice_found" for c in checks):
                publication = "no_notice_found"
            if publication == "unknown":
                states.append("publication_unknown")
            if work.screening == "pending":
                states.append("pending_screen")
            elif work.screening == "exclude":
                states.append("excluded_with_reason" if work.screening_reason.strip() else "excluded_without_reason")
            elif not work.retracted:
                document = next((d for d in before.documents.values() if d.work_id == wid), None)
                if document and level == "abstract_only":
                    states.append("abstract_only")
                card = before.cards.get(wid)
                if not document or not card:
                    states.append("unread")
                elif not card.claims or any(c.id not in before.decisions for c in card.claims):
                    states.append("unreviewed")
                else:
                    states.append("ready")
        hits.append(ChallengeHit(work_id=wid, statuses=states,
                                 screening_reason=work.screening_reason if work else "",
                                 access_level=level, publication_status=publication))
    return hits


def _coverage(queries: list[str], receipts: list[ChallengeQueryReceipt], hits: list[ChallengeHit], *,
              round_number: int | None = None):
    tasks = []
    searched = {r.query for r in receipts if r.sources}
    for query in queries:
        if query not in searched:
            tasks.append(ChallengeTask(reason="not_searched", query=query))
    for receipt in receipts:
        for source in receipt.sources:
            if source.status != "ok":
                tasks.append(ChallengeTask(reason=source.status, query=receipt.query, source=source.source,
                                           detail="; ".join(source.errors)))
            if source.truncated:
                tasks.append(ChallengeTask(reason="truncated", query=receipt.query, source=source.source))
        if receipt.dropped_at_project_cap:
            tasks.append(ChallengeTask(reason="dropped_at_cap", query=receipt.query,
                                       detail=str(receipt.dropped_at_project_cap)))
        if not receipt.retained_ids_available:
            tasks.append(ChallengeTask(reason="retained_ids_unavailable", query=receipt.query))
    tasks = [ChallengeTask.model_validate({**task.model_dump(), "round": round_number}) for task in tasks]
    for hit in hits:
        tasks.extend(ChallengeTask(reason=reason, work_id=hit.work_id, detail=hit.screening_reason)
                     for reason in hit.statuses if reason != "ready")
    return _coverage_status(tasks), tasks


def _coverage_status(tasks: list[ChallengeTask]) -> str:
    reasons = {task.reason for task in tasks}
    return ("source_failed" if reasons & {"partial", "unavailable", "rate_limited", "retained_ids_unavailable"}
            else "not_searched" if "not_searched" in reasons else
            "needs_reading" if reasons & {"pending_screen", "unread", "unreviewed", "missing_work",
                                           "excluded_without_reason"} else "needs_assessment")


def _rounds(root: Path, loaded: dict) -> list[tuple[ChallengeRound, Path | None]]:
    events, _ = _read_rows(root)
    positions = {event["id"]: i for i, event in enumerate(events)}
    starts = [event for event in events if event["kind"] == "challenge_start"
              and event["data"].get("candidate_id") == loaded["candidate"].id]
    artifacts = _versions(root, "challenge")
    results = []
    seen = set()
    for start in starts:
        data = start["data"]
        number = data["round"]
        if number in seen:
            raise ValueError("Duplicate durable challenge round boundary")
        seen.add(number)
        if data["identity"] != _identity(loaded, number, data["sources"], data["effective_limit"]):
            raise ValueError("Challenge identity disagrees with candidate content")
        if (data["queries"] != loaded["candidate"].targeted_queries
                or data["effective_limit"] != min(10, data["project_candidate_cap"])
                or data["query_keys"] != [search_query_key(q, data["sources"], data["project_candidate_cap"], limit=10)
                                           for q in data["queries"]]):
            raise ValueError("Challenge boundary has inconsistent query identities")
        pinned = next((a for _, a in _versions(root, "innovation")
                       if a["manifest"]["manifest_hash"] == data["candidate_manifest_hash"]), None)
        if pinned is None:
            raise ValueError("Challenge candidate provenance is missing")
        pinned_batch = CandidateBatch.model_validate_json(pinned["files"]["candidates.json"], strict=True)
        if loaded["candidate"] not in pinned_batch.candidates or pinned_batch.input_hash != loaded["batch"].input_hash:
            raise ValueError("Challenge pinned provenance disagrees with candidate")
        dispatches = [e for e in events if e["kind"] == "challenge_dispatch"
                      and e["data"].get("start_event_id") == start["id"]]
        requests = {}
        for dispatch in dispatches:
            request = dispatch["data"]
            query = request["query"]
            if (positions[dispatch["id"]] <= positions[start["id"]]
                    or query not in data["queries"] or request["request_id"] in requests
                    or request["query_key"] != data["query_keys"][data["queries"].index(query)]):
                raise ValueError("Challenge dispatch disagrees with durable round boundary")
            requests[request["request_id"]] = dispatch
        scoped = []
        for event in events:
            if event["kind"] != "search" or event["data"].get("request_id") not in requests:
                continue
            dispatch = requests[event["data"]["request_id"]]
            if (positions[event["id"]] <= positions[dispatch["id"]]
                    or event["data"].get("query_key") != dispatch["data"]["query_key"]
                    or event["data"].get("project_candidate_cap") != data["project_candidate_cap"]):
                raise ValueError("Challenge search does not match its durable dispatch")
            scoped.append(event)
        receipts, attempts = {}, []
        for event in scoped:
            if event["kind"] == "search" and event["data"].get("query_key") in data["query_keys"]:
                receipt = _receipt(event, data["queries"], data["sources"], data["effective_limit"])
                attempts.append(receipt)
                if receipt.query not in receipts or not receipts[receipt.query].complete:
                    receipts[receipt.query] = receipt
        output = None
        for path, artifact in artifacts:
            if set(artifact["files"]) != {"round.json"}:
                raise ValueError("Challenge version has unexpected payloads")
            saved = ChallengeRound.model_validate_json(artifact["files"]["round.json"], strict=True)
            if saved.start_event_id == start["id"]:
                for receipt in saved.search_attempts:
                    event = next((e for e in scoped if e["id"] == receipt.search_event_id), None)
                    if event is None or receipt != _receipt(event, data["queries"], data["sources"], data["effective_limit"]):
                        raise ValueError("Challenge receipt disagrees with durable search event")
                if (saved.identity != data["identity"] or saved.candidate_manifest_hash != data["candidate_manifest_hash"]
                        or saved.candidate_id != data["candidate_id"] or saved.round != number
                        or saved.queries != data["queries"] or saved.sources != data["sources"]
                        or saved.effective_limit != data["effective_limit"]):
                    raise ValueError("Challenge artifact disagrees with durable round boundary")
                if output is None or artifact["manifest"]["snapshot"]["captured_at"] > output[0]:
                    output = (artifact["manifest"]["snapshot"]["captured_at"], path, saved)
        ordered = [receipts[q] for q in data["queries"] if q in receipts]
        retained = sorted({wid for r in attempts for wid in r.retained_work_ids})
        status, tasks = _coverage(data["queries"], ordered, _hits(root, retained), round_number=number)
        results.append((ChallengeRound(
            identity=data["identity"], candidate_id=data["candidate_id"],
            candidate_manifest_hash=data["candidate_manifest_hash"], round=number,
            queries=data["queries"], sources=data["sources"], effective_limit=data["effective_limit"],
            start_event_id=start["id"], receipts=ordered, search_attempts=attempts, status=status,
            retained_work_ids=retained, unresolved_tasks=tasks),
            output[1] if output and output[2].search_attempts == attempts and output[2].receipts == ordered else None))
    return sorted(results, key=lambda pair: pair[0].round)


def read_challenges(
    root: Path,
    candidate_id: str,
    *,
    verified_research: ResearchSnapshot | None = None,
) -> dict:
    """Read verified rounds and recompute hit states from current committed research."""
    root = Path(root).resolve()
    loaded = load_candidate(root, candidate_id)
    rounds = _rounds(root, loaded)
    retained = sorted({wid for round_, _ in rounds for wid in round_.retained_work_ids})
    hits = _hits(root, retained, verified_research=verified_research)
    _, tasks = _coverage([] if rounds else loaded["candidate"].targeted_queries, [], hits)
    for round_, _ in rounds:
        _, query_tasks = _coverage(round_.queries, round_.receipts, [], round_number=round_.round)
        tasks.extend(query_tasks)
    status = _coverage_status(tasks)
    return {"rounds": [r for r, _ in rounds],
            "outputs": [str(path.relative_to(root)) for _, path in rounds if path],
            "hits": hits, "unresolved_tasks": [t.model_dump(mode="json") for t in tasks], "status": status}


def challenge_candidate(runner: Runner, candidate_id: str, *, round_number: int = 1) -> dict:
    if type(round_number) is not int or round_number not in {1, 2}:
        raise ValueError("Challenge round must be 1 or 2")
    loaded = load_candidate(runner.root, candidate_id)
    _assert_premises(runner.root, loaded)
    config = load_config(runner.root)
    if config != runner.config:
        raise ValueError("Runner configuration changed; recreate the runner")
    rounds = _rounds(runner.root, loaded)
    current = next(((r, p) for r, p in rounds if r.round == round_number), None)
    limit = min(10, config.max_candidates)
    identity = _identity(loaded, round_number, config.sources, limit)
    if current and current[0].identity != identity:
        raise ValueError("Challenge round scope changed; existing round cannot be reset")
    if current and all(r.complete for r in current[0].receipts) and len(current[0].receipts) == len(current[0].queries):
        if current[1] is not None:
            saved = ChallengeRound.model_validate_json(load_research(current[1])["files"]["round.json"], strict=True)
            path = current[1] if saved == current[0] else _publish(runner, current[0])
            return _result(runner.root, current[0], path)
    if runner.offline:
        raise ValueError("Challenge network search is unavailable in offline mode")
    if current:
        events, _ = _read_rows(runner.root)
        start = next(e for e in events if e["id"] == current[0].start_event_id)
        if start["data"]["project_candidate_cap"] != config.max_candidates:
            raise ValueError("Challenge project cap changed; restore it to resume failed queries")
    if current is None:
        if round_number == 2 and not any(r.round == 1 for r, _ in rounds):
            raise ValueError("Start challenge round 1 before round 2")
        queries = loaded["candidate"].targeted_queries
        runner.store.log("challenge_start", {
            "identity": identity, "candidate_id": candidate_id, "candidate_manifest_hash": loaded["manifest_hash"],
            "round": round_number, "queries": queries, "sources": config.sources, "effective_limit": limit,
            "project_candidate_cap": config.max_candidates,
            "query_keys": [search_query_key(q, config.sources, config.max_candidates, limit=10) for q in queries],
        })
    # Publishing after every query preserves successful work independently of latest.
    current = next(pair for pair in _rounds(runner.root, loaded) if pair[0].round == round_number)
    for query in current[0].queries:
        if not config.sources or any(r.query == query and r.complete for r in current[0].receipts):
            continue
        events, _ = _read_rows(runner.root)
        completed_requests = {e["data"].get("request_id") for e in events if e["kind"] == "search"}
        pending = [e for e in events if e["kind"] == "challenge_dispatch"
                   and e["data"].get("start_event_id") == current[0].start_event_id
                   and e["data"]["query"] == query and e["data"]["request_id"] not in completed_requests]
        request_id = pending[-1]["data"]["request_id"] if pending else uuid.uuid4().hex
        if not pending:
            runner.store.log("challenge_dispatch", {
                "start_event_id": current[0].start_event_id, "query": query, "request_id": request_id,
                "query_key": search_query_key(query, config.sources, config.max_candidates, limit=10),
            })
        runner.search(query, limit=10, request_id=request_id)
        _assert_premises(runner.root, loaded)
        current = next(pair for pair in _rounds(runner.root, loaded) if pair[0].round == round_number)
        path = _publish(runner, current[0])
        current = current[0], path
    if current[1] is None:
        current = current[0], _publish(runner, current[0])
    return _result(runner.root, *current)


def _publish(runner: Runner, round_: ChallengeRound) -> Path:
    snapshot = freeze_research(runner.store, require_supported=False)
    return publish_research(runner.store, "challenge", snapshot,
                            {"round.json": round_.model_dump_json(indent=2).encode("utf-8")})


def _result(root: Path, round_: ChallengeRound, path: Path) -> dict:
    return {"status": round_.status, "candidate_id": round_.candidate_id, "identity": round_.identity,
            "round": round_.round, "queries": [r.model_dump(mode="json") for r in round_.receipts],
            "retained_work_ids": round_.retained_work_ids,
            "unresolved_tasks": [t.model_dump(mode="json") for t in round_.unresolved_tasks],
            "output": str(path.relative_to(root))}
