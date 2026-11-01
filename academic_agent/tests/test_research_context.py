import sqlite3
from pathlib import Path

import pytest

from academic_agent.config import atomic_json, load_config
from academic_agent.demo import seed_demo
from academic_agent.documents import abstract_document, parse_document
from academic_agent.filesystem import io_path
from academic_agent.models import PublicationCheck, Work
from academic_agent.research_context import (
    assert_research_current,
    freeze_research,
    read_research,
    research_fingerprint,
)
from academic_agent.workflow import Runner


@pytest.fixture
def runner(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    runner.run()
    return runner


def test_snapshot_and_cached_model_identity(runner):
    before = freeze_research(runner.store)
    assert len(before.work_ids) == 2
    assert len(before.supported_claim_ids) == 12
    assert all(d.pages for d in before.documents.values())
    runner.store.log("innovation_progress", {"phase": "review"})
    calls = runner.store.budget()
    runner.read_selected()
    runner.review_selected()
    assert runner.store.budget() == calls
    after = freeze_research(runner.store)
    assert research_fingerprint(after) == research_fingerprint(before)
    assert after.stable_payload() == before.stable_payload()
    assert read_research(runner.root).stable_payload() == before.stable_payload()


@pytest.mark.parametrize("change", ["document", "verdict", "screening", "question", "search"])
def test_meaningful_changes(runner, change):
    before = freeze_research(runner.store)
    wid = before.work_ids[0]
    if change == "document":
        path = runner.root / "new.txt"
        path.write_text("Different original synthetic document.", encoding="utf-8")
        runner.store.add_document(parse_document(path, wid))
    elif change == "verdict":
        runner.manual_review(before.supported_claim_ids[0], "partial", "Scope uncertain", "Human")
    elif change == "screening":
        runner.store.screen(wid, "exclude", "Outside scope")
    elif change == "question":
        config = load_config(runner.root)
        config.question = "Another question"
        atomic_json(runner.root / "project.json", config.model_dump())
    else:
        runner.store.log("search", {"summary": {"query": "counterexample", "sources": []}, "responses": []})
    with pytest.raises(ValueError):
        assert_research_current(runner.store, before)


def test_runtime_config_does_not_change_identity(runner):
    before = freeze_research(runner.store)
    config = load_config(runner.root)
    config.max_model_calls += 1
    config.proxy_url = None
    config.timeout_seconds += 1
    atomic_json(runner.root / "project.json", config.model_dump())
    assert_research_current(runner.store, before)


def test_selection_and_unread_gaps(runner):
    wid = runner.store.upsert_work(Work(title="Original unread", screening="include")).id
    with pytest.raises(ValueError):
        freeze_research(runner.store)
    snapshot = freeze_research(runner.store, require_supported=False)
    assert wid in snapshot.work_ids
    assert any(g.work_id == wid for g in snapshot.gaps)
    assert snapshot.coverage.total == 3
    assert len(freeze_research(runner.store, work_ids=[snapshot.work_ids[0]], require_supported=False).work_ids) == 1
    for ids in [["unknown"], [], [wid, wid]]:
        with pytest.raises(ValueError):
            freeze_research(runner.store, work_ids=ids)
    runner.store.screen(wid, "exclude", "Not relevant")
    with pytest.raises(ValueError):
        freeze_research(runner.store, work_ids=[wid], require_supported=False)
    assert freeze_research(runner.store).coverage.excluded == 1


def test_retracted_and_pending_coverage(runner):
    pending = runner.store.upsert_work(Work(title="Pending original"))
    retracted = runner.store.upsert_work(Work(
        title="Retracted original", screening="include",
        publication_checks=[PublicationCheck(source="synthetic", status="retracted")]))
    snapshot = freeze_research(runner.store)
    assert snapshot.coverage.pending == snapshot.coverage.retracted == 1
    assert snapshot.coverage.total == 4
    assert pending.id not in snapshot.work_ids
    assert retracted.id not in snapshot.work_ids
    with pytest.raises(ValueError):
        freeze_research(runner.store, work_ids=[retracted.id], require_supported=False)


def test_explicit_selection_order_and_default_expansion(runner):
    snapshot = freeze_research(runner.store)
    explicit = freeze_research(runner.store, work_ids=list(reversed(snapshot.work_ids)))
    assert explicit.work_ids == snapshot.work_ids
    assert_research_current(runner.store, explicit)
    runner.store.upsert_work(Work(title="Another included original", screening="include"))
    with pytest.raises(ValueError):
        assert_research_current(runner.store, snapshot)


def test_search_gaps_preserve_limits_failures_and_unread_results(runner):
    event = {"summary": {"query": "original countersearch", "retained": 0, "dropped_at_project_cap": 3,
                         "sources": [{"source": "synthetic", "status": "unavailable", "truncated": True}]},
             "responses": [], "retained_work_ids": []}
    runner.store.log("search", event)
    snapshot = freeze_research(runner.store)
    assert snapshot.search_events[0]["data"] == event
    assert {g.reason for g in snapshot.gaps} >= {"search_incomplete", "search_truncated", "project_cap_reached"}


def test_strict_snapshot_rejects_nested_changes(runner):
    from academic_agent.research_context import ResearchSnapshot

    snapshot = freeze_research(runner.store)
    value = snapshot.model_dump()
    value["unexpected"] = 1
    with pytest.raises(ValueError):
        ResearchSnapshot.model_validate(value)
    snapshot.decisions[snapshot.supported_claim_ids[0]].evidence_ids = ["unknown"]
    with pytest.raises(ValueError):
        research_fingerprint(snapshot)


def test_stale_source_abstract_shared_readers(runner):
    work = runner.store.works()[0]
    work.abstract = "Original abstract"
    runner.store.update_work(work)
    runner.store.add_document(abstract_document(work))
    before = read_research(runner.root, require_supported=False)
    work.abstract = "Replacement abstract"
    runner.store.update_work(work)
    for read in [lambda: freeze_research(runner.store, require_supported=False),
                 lambda: read_research(runner.root, require_supported=False)]:
        snapshot = read()
        assert research_fingerprint(snapshot) != research_fingerprint(before)
        assert all(d.work_id != work.id for d in snapshot.documents.values())
        assert any(g.reason == "stale_source_abstract" for g in snapshot.gaps)


@pytest.mark.parametrize("configured", [False, True])
def test_missing_read_does_not_create(tmp_path, configured):
    root = tmp_path / "missing"
    if configured:
        root.mkdir()
        (root / "project.json").write_text('{"title":"Only config"}', encoding="utf-8")
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    with pytest.raises(ValueError, match="[Mm]issing"):
        read_research(root, require_supported=False)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_readonly_committed_wal_and_literal_path(tmp_path, monkeypatch):
    root = tmp_path / "literal#percent%20"
    seed_demo(root)
    runner = Runner(root, offline=True)
    runner.run()
    with sqlite3.connect(runner.store.db) as keeper:
        keeper.execute("PRAGMA wal_autocheckpoint=0")
        work = Work(id="wal_work", title="New committed work", screening="pending")
        keeper.execute("INSERT INTO works VALUES (?,?)", (work.id, work.model_dump_json()))
        keeper.commit()
        main_bytes = runner.store.db.read_bytes()
        blobs = {p: p.read_bytes() for p in (root / "artifacts").iterdir()}
        monkeypatch.setattr("academic_agent.store.Store.__init__", lambda *a: pytest.fail("Writable Store"))
        snapshot = read_research(root)
        assert snapshot.coverage.total == 3
        assert snapshot.coverage.pending == 1
        assert runner.store.db.read_bytes() == main_bytes
        assert all(p.read_bytes() == data for p, data in blobs.items())


@pytest.mark.parametrize("defect", ["foreign", "manual", "review", "blob", "unknown_claim", "card_evidence"])
def test_corruption_never_supports(runner, defect):
    snapshot = freeze_research(runner.store)
    card = snapshot.cards[snapshot.work_ids[0]]
    if defect == "foreign":
        span = next(e for e in snapshot.evidence.values() if e.work_id == card.work_id)
        span.work_id = "foreign"
        with runner.store.connect() as connection:
            connection.execute("UPDATE evidence SET data=? WHERE id=?", (span.model_dump_json(), span.id))
    elif defect == "manual":
        entry = runner.manual_review(card.claims[0].id, "supported", "Checked", "Human")
        entry["document_id"] = "foreign"
        runner.store.log("manual_review", entry)
    elif defect == "review":
        entry = runner.store.logs("review")[0]["data"]
        entry["report"]["verdicts"][0]["verdict"] = "invented"
        runner.store.log("review", entry)
    elif defect == "unknown_claim":
        entry = runner.store.logs("review")[0]["data"]
        entry["report"]["verdicts"][0]["claim_id"] = "foreign"
        runner.store.log("review", entry)
    elif defect == "card_evidence":
        card.claims[0].evidence_ids = ["foreign"]
        runner.store.log("reading_card", {"card": card.model_dump()})
    else:
        doc = next(iter(snapshot.documents.values()))
        (runner.root / doc.source_path).write_bytes(b"altered source")
    with pytest.raises(ValueError):
        freeze_research(runner.store)


def _replace_document_record(runner, document):
    blob, sha = runner.store.put_blob(document.model_dump_json().encode("utf-8"))
    with runner.store.connect() as connection:
        connection.execute(
            "UPDATE documents SET path=?,sha=? WHERE id=?", (blob, sha, document.id)
        )


def test_research_state_diagnostic_reports_only_an_intact_source_hash_change_as_stale(runner):
    import academic_agent.research_context as context

    snapshot = read_research(runner.root)
    document = next(iter(snapshot.documents.values()))
    source = io_path(runner.root / document.source_path)
    source.write_bytes(source.read_bytes() + b" changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_research(runner.root)
    assert context.research_state_currentness(runner.root, snapshot) == {
        "current": False,
        "gaps": [{"reason": "research_inputs_changed"}],
    }


@pytest.mark.parametrize("other_defect", ["malformed", "missing"])
def test_research_state_diagnostic_keeps_changed_source_plus_other_defect_invalid(runner, other_defect):
    import academic_agent.research_context as context

    snapshot = read_research(runner.root)
    documents = list(snapshot.documents.values())
    first = io_path(runner.root / documents[0].source_path)
    first.write_bytes(first.read_bytes() + b" changed")
    if other_defect == "malformed":
        with runner.store.connect() as connection:
            connection.execute("UPDATE logs SET data='{' WHERE kind='review'")
    else:
        io_path(runner.root / documents[1].source_path).unlink()
    assert context.research_state_currentness(runner.root, snapshot) == {
        "current": False,
        "gaps": [{"reason": "research_inputs_invalid"}],
    }


def test_research_state_diagnostic_rejects_unsafe_source_without_reading_it(runner, monkeypatch):
    import academic_agent.research_context as context

    snapshot = read_research(runner.root)
    document = next(iter(snapshot.documents.values()))
    unsafe = document.model_copy(update={"source_path": "../outside-research.txt"})
    _replace_document_record(runner, unsafe)
    actual_io_path = context.io_path
    attempted = []

    def checked_io_path(path):
        resolved = Path(path).resolve()
        attempted.append(resolved)
        assert resolved.is_relative_to(runner.root.resolve())
        return actual_io_path(path)

    monkeypatch.setattr(context, "io_path", checked_io_path)
    assert context.research_state_currentness(runner.root, snapshot) == {
        "current": False,
        "gaps": [{"reason": "research_inputs_invalid"}],
    }
    assert (runner.root.parent / "outside-research.txt").resolve() not in attempted


def test_research_state_diagnostic_reads_deliberately_long_source_through_io_path(runner, monkeypatch):
    import academic_agent.research_context as context

    original = read_research(runner.root)
    document = next(iter(original.documents.values()))
    data = io_path(runner.root / document.source_path).read_bytes()
    relative = Path("long-source")
    for index in range(4):
        relative /= f"segment-{index}-" + "x" * 55
    relative /= "paper.txt"
    source = runner.root / relative
    assert len(str(source.resolve())) > 260
    io_path(source).parent.mkdir(parents=True)
    io_path(source).write_bytes(data)
    migrated = document.model_copy(update={"source_path": relative.as_posix()})
    _replace_document_record(runner, migrated)
    snapshot = read_research(runner.root)

    actual_io_path = context.io_path
    opened = []

    def recording_io_path(path):
        opened.append(Path(path).resolve())
        return actual_io_path(path)

    monkeypatch.setattr(context, "io_path", recording_io_path)
    io_path(source).write_bytes(data + b" changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_research(runner.root)
    assert context.research_state_currentness(runner.root, snapshot) == {
        "current": False,
        "gaps": [{"reason": "research_inputs_changed"}],
    }
    assert source.resolve() in opened


def test_mutation_during_freeze_is_rejected(runner, monkeypatch):
    import academic_agent.research_context as context

    original = context.current_cards
    changed = False

    def mutate(projection):
        nonlocal changed
        result = original(projection)
        if not changed:
            changed = True
            work = runner.store.works()[0]
            work.title += " changed"
            runner.store.update_work(work)
        return result

    monkeypatch.setattr(context, "current_cards", mutate)
    with pytest.raises(ValueError, match="changed"):
        freeze_research(runner.store)


def test_source_mutation_during_freeze_is_rejected(runner, monkeypatch):
    import academic_agent.research_context as context

    document = next(iter(freeze_research(runner.store).documents.values()))
    original = context.current_cards

    def mutate(projection):
        result = original(projection)
        (runner.root / document.source_path).write_bytes(b"source changed during capture")
        return result

    monkeypatch.setattr(context, "current_cards", mutate)
    with pytest.raises(ValueError, match="hash mismatch"):
        freeze_research(runner.store)


@pytest.mark.parametrize("reader", ["freeze", "read"])
@pytest.mark.parametrize("field", ["claim_id", "work_id", "document_id", "evidence_ids"])
def test_current_manual_references_rejected(runner, reader, field):
    snapshot = freeze_research(runner.store)
    entry = snapshot.decisions[snapshot.supported_claim_ids[0]].model_dump(exclude={"type"})
    entry[field] = ["unknown_evidence"] if field == "evidence_ids" else "unknown_identity"
    runner.store.log("manual_review", entry)
    with pytest.raises(ValueError, match="manual"):
        if reader == "freeze":
            freeze_research(runner.store)
        else:
            read_research(runner.root)


@pytest.mark.parametrize("field", ["claim_id", "work_id", "document_id", "evidence_ids"])
def test_stale_manual_card_references_remain_ignored(runner, field):
    before = freeze_research(runner.store)
    entry = before.decisions[before.supported_claim_ids[0]].model_dump(exclude={"type"})
    entry["card_hash"] = "0" * 64
    entry[field] = ["old_evidence"] if field == "evidence_ids" else "old_identity"
    runner.store.log("manual_review", entry)
    assert research_fingerprint(freeze_research(runner.store)) == research_fingerprint(before)
    assert read_research(runner.root).stable_payload() == before.stable_payload()
