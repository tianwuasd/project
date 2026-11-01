"""Durable research operations and an optional checkpointed LangGraph sequence."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TypedDict

from langchain_core.tracers.context import tracing_v2_callback_var
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langsmith import tracing_context

from . import analysis
from .backends import Backend, CodexBackend, DemoBackend, ExchangeBackend
from .bibliography import import_file
from .config import load_config
from .documents import abstract_document, parse_document
from .exports import build_export, current_cards, snapshot_token
from .filesystem import io_path, sqlite_path
from .models import ModelJob, ModelResult, PublicationCheck, ReadingCard, Work, digest
from .network import HttpTransport
from .sources import check_publication, collect_sources
from .store import CandidateLimitError, Store


class WorkflowPaused(RuntimeError):
    pass


def search_query_key(query: str, sources: list[str], project_cap: int, *, limit: int | None = None) -> str:
    """Preserve legacy default identities; explicit bounds have a separate namespace."""
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 100):
        raise ValueError("Search limit must be an integer from 1 to 100")
    identity = [query.strip(), sources, project_cap]
    if limit is not None:
        identity.extend(["explicit-limit-v1", limit, min(limit, project_cap)])
    return digest(json.dumps(identity))


@contextmanager
def _local_graph_context():
    """Suppress both LangChain callbacks and LangSmith parent/environment tracing."""
    token = tracing_v2_callback_var.set(None)
    try:
        with tracing_context(enabled=False, parent=False):
            yield
    finally:
        tracing_v2_callback_var.reset(token)


class MeteredBackend:
    """Every dispatch reserves budget; only successful results are cached."""
    def __init__(self, backend: Backend, store: Store, config):
        self.backend, self.store, self.config = backend, store, config
        self.name = backend.name
        self._pending = None

    @contextmanager
    def validated(self):
        """Commit only after the consumer validates the complete domain artifact."""
        scoped = MeteredBackend(self.backend, self.store, self.config)
        scoped._pending = []
        try:
            yield scoped
            for ticket, result in scoped._pending:
                self.store.finish_job(ticket, result.model_dump(), result.usage)
        except BaseException:
            for ticket, result in scoped._pending:
                self.store.fail_job(ticket, "Model artifact validation or commit failed; attempt remains reserved")
            raise

    def run(self, job: ModelJob) -> ModelResult:
        if self._pending is None:
            raise RuntimeError("Backend calls require a domain validation context")
        content = job.model_dump()
        content.pop("id", None)
        fingerprint = digest(json.dumps({"task": content, "model": self.config.model}, sort_keys=True))
        work = job.payload.get("work", {})
        work_id = work.get("id", "corpus") if isinstance(work, dict) else "corpus"
        ticket = self.store.start_job(job.kind, work_id, fingerprint, self.name,
                                      self.config.max_model_calls, job.timeout_seconds)
        if ticket.cached:
            return ModelResult.model_validate(self.store.job_result(ticket.id))
        try:
            result = self.backend.run(job)
            if result.status == "succeeded":
                self._pending.append((ticket, result))
            else:
                # Exchange explicitly does not call a provider when requesting input.
                self.store.fail_job(ticket, result.error or result.status,
                                    uncertain=result.status != "needs_input")
            return result
        except BaseException as exc:
            self.store.fail_job(ticket, f"Dispatch interrupted ({type(exc).__name__}); provider outcome uncertain")
            raise


class Runner:
    def __init__(self, root: Path, *, offline: bool = False, backend: Backend | None = None):
        self.root = root.resolve()
        self.config = load_config(self.root)
        self.store = Store(self.root)
        self.offline = offline
        if backend is None:
            if self.config.backend == "codex":
                backend = CodexBackend(model=self.config.model, scratch_dir=self.root / ".runs" / "codex",
                                       proxy_url=self.config.proxy_url)
            elif self.config.backend == "demo":
                backend = DemoBackend()
            else:
                backend = ExchangeBackend(self.root / "exchange")
        self.backend = MeteredBackend(backend, self.store, self.config)

    def refresh_publication(self, works: list[Work], *, stage: str) -> None:
        def check(work):
            if self.offline:
                return [PublicationCheck(source="offline", detail=f"{stage}: network check skipped explicitly")]
            return check_publication(work, proxy_url=self.config.proxy_url)
        with ThreadPoolExecutor(max_workers=min(3, len(works) or 1)) as pool:
            checks = list(pool.map(check, works))
        for work, notices in zip(works, checks):
            # Fetch current value after network I/O; do not overwrite earlier flags.
            current = self.store.work(work.id)
            current.publication_checks.extend(notices)
            self.store.update_work(current)
        self.store.log("publication_refresh", {"stage": stage, "work_ids": [w.id for w in works],
                                               "offline": self.offline})

    def search(self, query: str, *, cached: bool = False, limit: int | None = None,
               request_id: str | None = None) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("Search query cannot be empty")
        if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()
                                       or request_id != request_id.strip() or len(request_id) > 128):
            raise ValueError("Search request_id must be a nonblank string of at most 128 characters")
        key = search_query_key(query, self.config.sources, self.config.max_candidates, limit=limit)
        effective_limit = min(100 if limit is None else limit, self.config.max_candidates)
        if cached:
            previous = [e for e in self.store.logs("search") if e["data"]["query_key"] == key
                        and (request_id is None or e["data"].get("request_id") == request_id)]
            if previous and all(source["status"] in {"ok", "empty"}
                                for source in previous[-1]["data"]["summary"]["sources"]):
                data = previous[-1]["data"]
                return {"cached": True, **data["summary"],
                        "retained_work_ids": data.get("retained_work_ids", []),
                        "retained_ids_available": "retained_work_ids" in data,
                        "effective_limit": data.get("effective_limit"),
                        "search_event_id": previous[-1]["id"]}
        if self.offline:
            raise ValueError("Network search is unavailable in offline mode; import local records")
        pages = collect_sources(query, self.config.sources, effective_limit, self.config.proxy_url)
        retained = set()
        dropped = 0
        # Round-robin limits keep a high-volume first source from monopolising the corpus.
        for index in range(max((len(page.records) for page in pages), default=0)):
            for page in pages:
                if index >= len(page.records):
                    continue
                candidate = page.records[index]
                try:
                    work = self.store.upsert_work(candidate, max_works=self.config.max_candidates)
                except CandidateLimitError:
                    dropped += 1
                    continue
                retained.add(work.id)
        summary = {"query": query, "retained": len(retained), "dropped_at_project_cap": dropped,
                   "sources": [{"source": p.source, "status": p.status, "records": len(p.records),
                                "truncated": p.truncated, "errors": p.errors} for p in pages]}
        event_id = self.store.log("search", {"query_key": key, "summary": summary,
                                           "retained_work_ids": sorted(retained), "effective_limit": effective_limit,
                                           "project_candidate_cap": self.config.max_candidates,
                                           "request_id": request_id,
                                           "responses": [p.model_dump() for p in pages]})
        self.refresh_publication([self.store.work(work_id) for work_id in sorted(retained)], stage="initial_search")
        return {**summary, "retained_work_ids": sorted(retained), "retained_ids_available": True,
                "effective_limit": effective_limit, "search_event_id": event_id}

    def import_records(self, path: Path) -> list[Work]:
        records = import_file(path)
        if len(records) + len(self.store.works()) > self.config.max_candidates:
            raise ValueError("Import exceeds project candidate cap; split the bibliography or raise the configured limit")
        works = [self.store.upsert_work(work) for work in records]
        self.refresh_publication(works, stage="initial_import")
        self.store.log("import", {"filename": path.name, "work_ids": [w.id for w in works]})
        return [self.store.work(w.id) for w in works]

    def attach(self, work_id: str, path: Path) -> dict:
        document = self.store.add_document(parse_document(path, work_id))
        return {"document_id": document.id, "level": document.level, "diagnostics": document.diagnostics}

    def fetch(self, work_id: str) -> dict:
        if self.offline:
            raise ValueError("Document download is disabled in offline mode")
        work = self.store.work(work_id)
        if not work.open_access_url:
            raise ValueError("No open-access document URL; attach an accessible local copy")
        data = HttpTransport(self.config.proxy_url, max_bytes=50 * 1024 * 1024).get_bytes(work.open_access_url)
        if not data.lstrip().startswith(b"%PDF-"):
            raise ValueError("Open-access URL did not return a PDF; attach a local document")
        relative, _ = self.store.put_blob(data, "pdf")
        return self.attach(work_id, self.store.root / relative)

    def prepare(self) -> dict:
        works = self.store.works(included_only=True)
        if len(works) > self.config.max_documents:
            raise ValueError("Included papers exceed reading cap; revise screening or configuration")
        levels = {}
        for work in works:
            document = self.store.active_document(work.id)
            candidate = abstract_document(work)
            if document is None or document.parser == "source-abstract" and document.id != candidate.id:
                document = self.store.add_document(candidate)
            levels[work.id] = document.level
        return {"documents": levels}

    def read_selected(self, *, work_id: str | None = None, question: str = "") -> dict:
        self.prepare()
        works = [self.store.work(work_id)] if work_id else self.store.works(included_only=True)
        if len(works) > self.config.max_documents:
            raise ValueError("Reading cap exceeded")
        def run(work):
            try:
                if work.screening != "include":
                    raise ValueError("Paper must be explicitly included before reading")
                if work.retracted:
                    raise ValueError("Known retracted paper cannot support ordinary reading conclusions")
                document = self.store.active_document(work.id)
                if document is None:
                    raise ValueError("No prepared document")
                # Publication timestamps and unrelated provenance must not defeat text-version caching.
                model_work = work.model_copy(deep=True)
                model_work.publication_checks = []
                model_work.provenance = []
                if hasattr(model_work, "record_revision"):
                    model_work.record_revision = 0
                with self.backend.validated() as backend:
                    card = analysis.read(model_work, document, self.store.evidence(work.id), backend,
                                         timeout_seconds=self.config.timeout_seconds,
                                         max_input_chars=self.config.max_input_chars, question=question)
                self.store.log("answer" if question else "reading_card", {"card": card.model_dump(), "question": question})
                return {"work_id": work.id, "status": "ok", "claims": len(card.claims)}
            except Exception as exc:
                return {"work_id": work.id, "status": "failed", "error": str(exc)}
        with ThreadPoolExecutor(max_workers=self.config.model_workers) as pool:
            results = list(pool.map(run, works))
        return {"results": results, "budget": self.store.budget()}

    def review_selected(self, *, work_id: str | None = None, answers: bool = False, question: str | None = None) -> dict:
        cards = current_cards(self.store, kind="answer" if answers else "reading_card", question=question)
        if work_id:
            cards = {key: value for key, value in cards.items() if key == work_id}
            if not cards:
                raise ValueError("No reading card for the current document")
        eligible = {w.id for w in self.store.works(included_only=True) if not w.retracted}
        selected = [card for key, card in cards.items() if key in eligible]
        if len(selected) > self.config.max_documents:
            raise ValueError("Review cap exceeded")
        def run(card: ReadingCard):
            try:
                document = self.store.active_document(card.work_id)
                with self.backend.validated() as backend:
                    report = analysis.review(card, document, self.store.evidence(card.work_id), backend,
                                             timeout_seconds=self.config.timeout_seconds,
                                             max_input_chars=self.config.max_input_chars)
                self.store.log("review", {"report": report.model_dump(), "card_hash": digest(card.model_dump_json())})
                return {"work_id": card.work_id, "status": "ok", "verdicts": [v.model_dump() for v in report.verdicts]}
            except Exception as exc:
                return {"work_id": card.work_id, "status": "failed", "error": str(exc)}
        with ThreadPoolExecutor(max_workers=self.config.model_workers) as pool:
            results = list(pool.map(run, selected))
        return {"results": results, "budget": self.store.budget()}

    def manual_review(self, claim_id: str, verdict: str, reason: str, reviewer: str) -> dict:
        if verdict not in {"supported", "partial", "contradicted", "insufficient"} or not reason.strip() or not reviewer.strip():
            raise ValueError("Manual review requires a valid verdict, reason and reviewer name")
        for card in current_cards(self.store).values():
            for claim in card.claims:
                if claim.id == claim_id:
                    entry = {"work_id": card.work_id, "document_id": card.document_id, "claim_id": claim_id,
                             "card_hash": digest(card.model_dump_json()), "verdict": verdict, "rationale": reason,
                             "reviewer": reviewer, "evidence_ids": claim.evidence_ids}
                    self.store.log("manual_review", entry)
                    return entry
        raise ValueError("Claim is absent or belongs to an inactive document version")

    def export(self) -> Path:
        self.refresh_publication(self.store.works(), stage="pre_export")
        return build_export(self.store, self.config.title, self.config.question,
                            config_hash=digest(self.config.model_dump_json()))

    def status(self) -> dict:
        latest_path = self.root / "exports" / "latest.json"
        latest = json.loads(io_path(latest_path).read_text(encoding="utf-8")) if io_path(latest_path).exists() else None
        return {"project": str(self.root), "works": [w.model_dump() for w in self.store.works()],
                "jobs": [{k: j[k] for k in ("id", "kind", "status", "error")} for j in self.store.jobs()],
                "budget": self.store.budget(), "latest_export": latest,
                "latest_export_current": bool(latest and tuple(latest["snapshot"]) == snapshot_token(self.store))}

    def run(self, *, resume: bool = False, run_id: str = "main", stop_after: str | None = None) -> dict:
        if not run_id or len(run_id) > 100:
            raise ValueError("Run ID must contain 1–100 characters")
        class State(TypedDict, total=False):
            phase: str
            output: str
            revision: int
            config_hash: str
        def discover(state):
            failures = []
            for query in self.config.queries:
                result = self.search(query, cached=True)
                failures.extend({"source": source["source"], "status": source["status"]}
                                for source in result["sources"] if source["status"] not in {"ok", "empty"})
            if failures:
                raise WorkflowPaused("Discovery incomplete; resume to retry sources: " + json.dumps(failures))
            return {"phase": "discover"}
        def prepare(state):
            self.prepare()
            if self.store.works() and not self.store.works(included_only=True):
                raise WorkflowPaused("Screen candidate papers explicitly, then resume")
            return {"phase": "prepare"}
        def reading(state):
            result = self.read_selected()
            failures = [item for item in result["results"] if item["status"] != "ok"]
            if failures:
                raise WorkflowPaused(json.dumps({"results": failures}, ensure_ascii=False))
            return {"phase": "read"}
        def reviewing(state):
            result = self.review_selected()
            failures = [item for item in result["results"] if item["status"] != "ok"]
            if failures:
                raise WorkflowPaused(json.dumps({"results": failures}, ensure_ascii=False))
            return {"phase": "review"}
        def exporting(state):
            return {"phase": "export", "output": str(self.export())}
        graph = StateGraph(State)
        config_hash = digest(self.config.model_dump_json())
        def tracked(function):
            def node(state):
                return {**function(state), "revision": self.store.revision, "config_hash": config_hash}
            return node
        names = ["discover", "prepare", "read", "review", "export"]
        for name, function in zip(names, [discover, prepare, reading, reviewing, exporting]):
            graph.add_node(name, tracked(function))
        graph.add_edge(START, names[0])
        for left, right in zip(names, names[1:] + [END]):
            graph.add_edge(left, right)
        if stop_after is not None and stop_after not in names:
            raise ValueError("Unknown checkpoint phase")
        # Inherited environment/parent tracing is not authorization to transmit research.
        # The context covers checkpoint reads, graph execution and its error callbacks.
        with _local_graph_context(), closing(
            sqlite3.connect(sqlite_path(self.root / "checkpoints.db"), check_same_thread=False)
        ) as connection:
            app = graph.compile(checkpointer=SqliteSaver(connection), interrupt_after=[stop_after] if stop_after else None)
            config = {"configurable": {"thread_id": run_id}, "callbacks": []}
            state = app.get_state(config)
            if resume and not state.values:
                raise ValueError("No checkpoint exists for this run ID")
            changed = bool(state.values) and (state.values.get("revision") != self.store.revision
                                              or state.values.get("config_hash") != config_hash)
            # Older completed checkpoints may predate the recoverable discovery pause.
            for query in self.config.queries:
                key = digest(json.dumps([query.strip(), self.config.sources, self.config.max_candidates]))
                previous = [e for e in self.store.logs("search") if e["data"]["query_key"] == key]
                if previous and any(source["status"] not in {"ok", "empty"}
                                    for source in previous[-1]["data"]["summary"]["sources"]):
                    changed = True
            if resume and not state.next and not changed:
                return {**state.values, "complete": True, "already_complete": True,
                        "export_current": self.status()["latest_export_current"]}
            output = app.invoke(None if resume and not changed else {"phase": "start"}, config)
            return {**output, "complete": not bool(app.get_state(config).next)}
