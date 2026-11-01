import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from academic_agent.documents import abstract_document
from academic_agent.models import PublicationCheck, Work
from academic_agent.store import BudgetExceeded, CandidateLimitError, StaleJobError, StaleWorkError, Store


def test_identity_is_conservative_and_citekey_stable(tmp_path):
    store = Store(tmp_path)
    first = store.upsert_work(Work(title="A study", authors=["Li, A"], year=2020, doi="HTTPS://DOI.ORG/10.1234/ABC"))
    store.add_document(abstract_document(first))
    merged = store.upsert_work(Work(title="A study updated", doi="10.1234/abc", abstract="New abstract"))
    assert merged.id == first.id and merged.citekey == first.citekey
    assert store.active_document(first.id) is not None
    assert store.upsert_work(Work(title="A study", authors=["Wang, B"], year=2020)).id != first.id
    assert store.upsert_work(Work(title="A study", authors=["Li, A"], year=2020, doi="10.1234/other")).id != first.id
    assert len(store.works()) == 3


def test_arxiv_versions_and_document_replacement(tmp_path):
    store = Store(tmp_path)
    first = store.upsert_work(Work(title="Study", arxiv_id="2401.00001v1", abstract="version one"))
    old_doc = store.add_document(abstract_document(first))
    second = store.upsert_work(Work(title="Study", arxiv_id="2401.00001v2", abstract="version two"))
    assert second.id == first.id
    assert {r["id"] for r in second.relations if r["type"] == "arxiv_version"} == {
        "2401.00001v1", "2401.00001v2"}
    assert second.abstract == "version two" and second.arxiv_id == "2401.00001v2"
    new_doc = store.add_document(abstract_document(second))
    assert old_doc.id != new_doc.id
    assert store.document(old_doc.id).pages[0].text == "version one"
    assert all(e.document_id == new_doc.id for e in store.evidence(first.id))
    old_again = store.upsert_work(Work(title="Study", arxiv_id="2401.00001v1", abstract="version one"))
    assert old_again.abstract == "version two" and old_again.arxiv_id == "2401.00001v2"
    assert {(p["arxiv_id"], p["abstract"]) for p in old_again.provenance
            if p["source"] == "arxiv_version"} == {
                ("2401.00001v1", "version one"), ("2401.00001v2", "version two")}
    assert store.active_document(first.id).id == new_doc.id


def test_stale_snapshot_cannot_erase_retraction_and_screen_preserves_it(tmp_path):
    store = Store(tmp_path)
    first = store.upsert_work(Work(title="Study"))
    stale = store.work(first.id)
    current = store.work(first.id)
    current.publication_checks.append(PublicationCheck(source="crossref", status="retracted"))
    store.update_work(current)
    assert current.record_revision == first.record_revision + 1
    stale.screening = "include"
    with pytest.raises(StaleWorkError):
        store.update_work(stale)
    store.screen(first.id, "include", "User decision after reading")
    latest = store.work(first.id)
    assert latest.retracted and latest.screening == "include"
    latest.publication_checks = []
    store.update_work(latest)
    assert store.work(first.id).retracted and latest.retracted


def test_metadata_upsert_invalidates_old_snapshot(tmp_path):
    store = Store(tmp_path)
    stale = store.upsert_work(Work(title="Study", doi="10.1234/study"))
    merged = store.upsert_work(Work(title="Study", doi=stale.doi, abstract="New evidence"))
    assert merged.record_revision == stale.record_revision + 1
    stale.screening_reason = "Outdated snapshot"
    with pytest.raises(StaleWorkError):
        store.update_work(stale)
    assert store.work(stale.id).abstract == "New evidence"


def test_legacy_work_revision_defaults_to_zero():
    assert Work.model_validate({"title": "Legacy"}).record_revision == 0
    with pytest.raises(ValueError):
        Work(title="Invalid", record_revision=-1)


def test_lower_arxiv_cannot_fill_missing_latest_abstract(tmp_path):
    store = Store(tmp_path)
    latest = store.upsert_work(Work(title="Study", arxiv_id="2401.00001v2"))
    merged = store.upsert_work(Work(title="Study", arxiv_id="2401.00001v1", abstract="Old result"))
    assert merged.id == latest.id and merged.abstract == ""
    assert merged.provenance[-1]["abstract"] == "Old result"


def test_source_alias_survives_three_imports(tmp_path):
    store = Store(tmp_path)
    first = store.upsert_work(Work(title="Study", authors=["Doe"], year=2024, source_ids={"openalex": "W1"}))
    second = store.upsert_work(Work(title="Study", authors=["Doe"], year=2024, source_ids={"openalex": "W2"}))
    third = store.upsert_work(Work(title="Corrected study title", source_ids={"openalex": "W1"}))
    assert first.id == second.id == third.id
    assert len(store.works()) == 1
    assert {r["id"] for r in third.relations if r["type"] == "source_id_alias"} == {"W1", "W2"}
    assert third.citekey == first.citekey


def test_candidate_limit_allows_existing_metadata_updates(tmp_path):
    store = Store(tmp_path)
    first = store.upsert_work(Work(title="One", doi="10.1234/one"), max_works=1)
    with pytest.raises(CandidateLimitError):
        store.upsert_work(Work(title="Two"), max_works=1)
    merged = store.upsert_work(Work(title="One", doi=first.doi, abstract="Added at cap"), max_works=1)
    assert merged.id == first.id and merged.abstract == "Added at cap"
    assert len(store.works()) == 1


def test_concurrent_candidate_insertions_respect_limit(tmp_path):
    store = Store(tmp_path)
    def insert(number):
        try:
            store.upsert_work(Work(title=f"Paper {number}"), max_works=3)
            return True
        except CandidateLimitError:
            return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(insert, range(8))) == 3
    assert len(store.works()) == 3


def test_file_tamper_and_escape_rejected(tmp_path):
    store = Store(tmp_path)
    path, sha = store.put_blob(b"original")
    (tmp_path / path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        store.read_blob(path, sha)
    with pytest.raises(ValueError, match="modified"):
        store.put_blob(b"original")
    with pytest.raises(ValueError, match="escapes"):
        store.read_blob("../outside.json", sha)


def test_job_success_cache_and_unknown_cost(tmp_path):
    store = Store(tmp_path)
    ticket = store.start_job("read", "work", "hash", "demo", 1)
    with pytest.raises(RuntimeError, match="running"):
        store.start_job("read", "work", "hash", "demo", 1)
    store.finish_job(ticket, {"answer": 3})
    cached = store.start_job("read", "work", "hash", "demo", 1)
    assert cached.cached
    assert store.job_result(cached.id) == {"answer": 3}
    assert store.budget()["monetary_cost"] is None
    with pytest.raises(BudgetExceeded):
        store.start_job("read", "work", "different", "demo", 1)


@pytest.mark.parametrize("invalidate", ["cancel", "revision", "lease", "attempt"])
def test_late_results_never_commit(tmp_path, invalidate):
    store = Store(tmp_path)
    ticket = store.start_job("read", "work", "hash", "codex", 2)
    if invalidate == "cancel":
        store.cancel_job(ticket.id)
    elif invalidate == "revision":
        store.upsert_work(Work(title="New source"))
    elif invalidate == "lease":
        with store.connect() as connection:
            connection.execute("UPDATE jobs SET lease_until=0")
    else:
        with store.connect() as connection:
            connection.execute("UPDATE jobs SET lease_until=0")
        store.start_job("read", "work", "hash", "codex", 2)
    with pytest.raises(StaleJobError):
        store.finish_job(ticket, {"must": "not commit"})
    assert store.jobs()[0]["status"] != "succeeded"


def test_uncertain_attempts_remain_reserved(tmp_path):
    store = Store(tmp_path)
    ticket = store.start_job("read", "work", "hash", "codex", 1)
    store.fail_job(ticket, "Unknown provider outcome")
    assert store.budget()["attempts"] == {"uncertain": 1}
    with pytest.raises(BudgetExceeded):
        store.start_job("read", "work", "hash", "codex", 1)


def test_parallel_reservations_respect_global_budget(tmp_path):
    Store(tmp_path)
    def reserve(number):
        try:
            Store(tmp_path).start_job("read", str(number), "hash", "codex", 3)
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        assert sum(pool.map(reserve, range(8))) == 3
    assert Store(tmp_path).budget()["consumed_or_reserved_calls"] == 3


def test_bad_schema_version_fails(tmp_path):
    store = Store(tmp_path)
    with sqlite3.connect(store.db) as conn:
        conn.execute("UPDATE meta SET value=9 WHERE key='schema_version'")
    with pytest.raises(ValueError, match="schema version"):
        Store(tmp_path)
