import hashlib
import json

import pytest

from academic_agent.config import atomic_json, create_project
from academic_agent.demo import FIXTURES, seed_demo
from academic_agent.exports import build_export, current_cards, snapshot_token
from academic_agent.models import ProjectConfig, PublicationCheck, SearchPage, Work
from academic_agent.workflow import Runner, WorkflowPaused


def test_offline_demo_exports_and_resume_use_cached_calls(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    first = runner.run(stop_after="read")
    assert first["phase"] == "read" and not first["complete"]
    assert runner.store.budget()["consumed_or_reserved_calls"] == 2
    # Recreate the entire Runner, exercising on-disk graph/domain recovery.
    resumed = Runner(tmp_path, offline=True).run(resume=True)
    assert resumed["complete"]
    assert runner.store.budget()["consumed_or_reserved_calls"] == 4
    path = tmp_path / "exports" / json.loads((tmp_path / "exports/latest.json").read_text())["version"]
    quality = json.loads((path / "quality.json").read_text(encoding="utf-8"))
    assert quality["supported_claims"] == 12 and quality["synthetic_demo"]
    assert quality["human_accuracy_evaluation"] == "not_performed"
    runner.run()
    assert runner.store.budget()["consumed_or_reserved_calls"] == 4
    assert runner.status()["latest_export_current"]


def test_manifest_hashes_and_human_edits_are_not_overwritten(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    first = runner.run()
    path = tmp_path / "exports" / json.loads((tmp_path / "exports/latest.json").read_text())["version"]
    manifest = json.loads((path / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == expected
    (path / "review.md").write_text("Human edited manuscript", encoding="utf-8")
    new_path = runner.export()
    assert str(new_path) != first["output"]
    assert (path / "review.md").read_text() == "Human edited manuscript"


def test_retracted_claims_excluded_even_if_manually_supported(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    runner.run()
    card = next(iter(current_cards(runner.store).values()))
    runner.manual_review(card.claims[0].id, "supported", "Reviewed evidence", "Test reviewer")
    work = runner.store.work(card.work_id)
    work.publication_checks.append(PublicationCheck(source="test", status="retracted"))
    runner.store.update_work(work)
    assert not runner.status()["latest_export_current"]
    path = runner.export()
    ledger = json.loads((path / "evidence-ledger.json").read_text(encoding="utf-8"))
    assert all(not item["included_in_review"] for item in ledger["claims"] if item["work_id"] == work.id)
    quality = json.loads((path / "quality.json").read_text(encoding="utf-8"))
    assert quality["known_retracted"] == 1 and quality["supported_claims"] == 6


def test_changed_document_invalidates_cards_and_decisions(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    runner.run()
    old = next(iter(current_cards(runner.store).values()))
    path = tmp_path / "replacement.txt"
    path.write_text(FIXTURES[0][1].replace("40 simulated", "45 simulated"), encoding="utf-8")
    runner.attach(old.work_id, path)
    assert old.work_id not in current_cards(runner.store)
    exported = runner.export()
    quality = json.loads((exported / "quality.json").read_text(encoding="utf-8"))
    assert quality["reading_cards"] == 1 and quality["supported_claims"] == 6


def test_export_rejects_concurrent_state_change(tmp_path, monkeypatch):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    from academic_agent import exports
    original = exports.atomic_json
    def racing_write(path, value):
        original(path, value)
        if path.name == "manifest.json":
            runner.store.log("screening", {"changed": True})
    monkeypatch.setattr(exports, "atomic_json", racing_write)
    with pytest.raises(ValueError, match="changed during export"):
        build_export(runner.store, "Study", "Question")
    assert not (tmp_path / "exports/latest.json").exists()


def test_budget_exhaustion_stops_graph_and_records_partial_progress(tmp_path):
    seed_demo(tmp_path)
    config = ProjectConfig.model_validate_json((tmp_path / "project.json").read_text(encoding="utf-8"))
    config.max_model_calls = 1
    atomic_json(tmp_path / "project.json", config.model_dump())
    runner = Runner(tmp_path, offline=True)
    with pytest.raises(WorkflowPaused, match="budget"):
        runner.run()
    assert runner.store.budget()["consumed_or_reserved_calls"] == 1
    assert len(current_cards(runner.store)) == 1
    # A partial corpus can still be exported explicitly and honestly.
    path = runner.export()
    assert json.loads((path / "quality.json").read_text())["supported_claims"] == 0


def test_screening_pause_then_resume_and_abstract_level(tmp_path):
    create_project(tmp_path, ProjectConfig(title="Abstract only", backend="demo", sources=[]))
    runner = Runner(tmp_path, offline=True)
    work = runner.store.upsert_work(Work(title="Synthetic abstract", abstract=FIXTURES[0][1]))
    with pytest.raises(WorkflowPaused, match="Screen"):
        runner.run()
    runner.store.screen(work.id, "include", "Matches scope")
    result = runner.run(resume=True)
    assert result["complete"]
    assert runner.store.active_document(work.id).level == "abstract_only"
    quality = json.loads((tmp_path / "exports" / json.loads((tmp_path / "exports/latest.json").read_text())["version"] / "quality.json").read_text())
    assert quality["document_levels"]["abstract_only"] == 1


def test_zero_hits_and_unavailable_sources_are_explicit(tmp_path, monkeypatch):
    create_project(tmp_path, ProjectConfig(title="No results", backend="demo", queries=["unknown"]))
    from academic_agent import workflow
    monkeypatch.setattr(workflow, "collect_sources", lambda *args: [SearchPage(source="one", status="empty"),
                                                                  SearchPage(source="two", status="unavailable", errors=["Service unavailable"])])
    runner = Runner(tmp_path)
    with pytest.raises(WorkflowPaused, match='"source": "two", "status": "unavailable"'):
        runner.run()
    assert not (tmp_path / "exports/latest.json").exists()
    assert runner.store.logs("search")[0]["data"]["summary"]["sources"][1]["status"] == "unavailable"
    assert runner.store.budget()["consumed_or_reserved_calls"] == 0


def test_sources_share_one_project_candidate_cap(tmp_path, monkeypatch):
    create_project(tmp_path, ProjectConfig(title="Bounded", backend="demo", max_candidates=3))
    from academic_agent import workflow
    pages = [SearchPage(source=name, records=[Work(title=f"{name} {i}", source_ids={name: str(i)}) for i in range(3)])
             for name in ("one", "two", "three")]
    monkeypatch.setattr(workflow, "collect_sources", lambda *args: pages)
    monkeypatch.setattr(workflow, "check_publication", lambda *args, **kwargs: [PublicationCheck(source="test")])
    runner = Runner(tmp_path)
    result = runner.search("keyword")
    assert len(runner.store.works()) == 3
    assert {next(iter(w.source_ids)) for w in runner.store.works()} == {"one", "two", "three"}
    assert result["dropped_at_project_cap"] == 6


def test_no_abstract_fails_without_invented_claims(tmp_path):
    create_project(tmp_path, ProjectConfig(title="Metadata", backend="demo", sources=[]))
    runner = Runner(tmp_path, offline=True)
    runner.store.upsert_work(Work(title="Only a title", screening="include", screening_reason="Test"))
    result = runner.read_selected()
    assert result["results"][0]["status"] == "failed"
    assert not current_cards(runner.store)
    assert runner.store.budget()["consumed_or_reserved_calls"] == 0


def test_no_resume_checkpoint_is_clear(tmp_path):
    seed_demo(tmp_path)
    with pytest.raises(ValueError, match="No checkpoint"):
        Runner(tmp_path, offline=True).run(resume=True)


def test_schema_invalid_model_result_does_not_poison_cache(tmp_path):
    from academic_agent.backends import DemoBackend
    from academic_agent.models import ModelResult
    class InitiallyMalformed(DemoBackend):
        calls = 0
        def run(self, job):
            self.calls += 1
            if self.calls == 1:
                return ModelResult(status="succeeded", backend=self.name, data={"wrong": "schema"})
            return super().run(job)
    seed_demo(tmp_path)
    backend = InitiallyMalformed()
    runner = Runner(tmp_path, offline=True, backend=backend)
    work_id = runner.store.works()[0].id
    assert runner.read_selected(work_id=work_id)["results"][0]["status"] == "failed"
    assert runner.store.jobs()[0]["status"] == "failed"
    assert runner.read_selected(work_id=work_id)["results"][0]["status"] == "ok"
    assert backend.calls == 2
    assert runner.store.budget()["consumed_or_reserved_calls"] == 2


def test_resume_restarts_required_phases_after_document_change(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    runner.run(stop_after="read")
    work = runner.store.works()[0]
    path = tmp_path / "changed.txt"
    path.write_text(FIXTURES[0][1].replace("40 simulated", "41 simulated"), encoding="utf-8")
    runner.attach(work.id, path)
    assert runner.run(resume=True)["complete"]
    assert len(current_cards(runner.store)) == 2
    assert any("41 simulated" in c.text for c in current_cards(runner.store)[work.id].claims)
    assert runner.store.budget()["consumed_or_reserved_calls"] == 5


def test_candidate_cap_still_allows_metadata_enrichment(tmp_path, monkeypatch):
    create_project(tmp_path, ProjectConfig(title="Cap", backend="demo", max_candidates=1))
    runner = Runner(tmp_path)
    work = runner.store.upsert_work(Work(title="Study", doi="10.1234/a"))
    from academic_agent import workflow
    monkeypatch.setattr(workflow, "collect_sources", lambda *args: [SearchPage(source="crossref", records=[
        Work(title="Study", doi="10.1234/a", abstract="New abstract")])])
    monkeypatch.setattr(workflow, "check_publication", lambda *args, **kwargs: [PublicationCheck(source="test")])
    runner.search("study")
    assert runner.store.work(work.id).abstract == "New abstract"


def test_new_source_abstract_invalidates_cards_before_direct_export(tmp_path):
    create_project(tmp_path, ProjectConfig(title="Abstract versions", backend="demo", sources=[]))
    runner = Runner(tmp_path, offline=True)
    work = runner.store.upsert_work(Work(title="Study", arxiv_id="2401.00001v1", abstract=FIXTURES[0][1],
                                         screening="include", screening_reason="Test"))
    runner.run()
    old_document = runner.store.active_document(work.id)
    assert any("40 simulated" in claim.text for claim in current_cards(runner.store)[work.id].claims)
    runner.store.upsert_work(Work(title="Study", arxiv_id="2401.00001v2",
                                 abstract=FIXTURES[0][1].replace("40 simulated", "45 simulated")))
    assert work.id not in current_cards(runner.store)
    assert not runner.status()["latest_export_current"]
    exported = runner.export()  # No prepare/read/run between source update and export.
    assert "40 simulated" not in (exported / "review.md").read_text(encoding="utf-8")
    assert json.loads((exported / "quality.json").read_text())["supported_claims"] == 0
    assert "45 simulated" in runner.store.active_document(work.id).pages[0].text
    assert runner.store.document(old_document.id) == old_document
    assert runner.status()["latest_export_current"]


@pytest.mark.parametrize("text", [FIXTURES[0][1], ""])
def test_source_refresh_preserves_manually_attached_document(tmp_path, text):
    create_project(tmp_path, ProjectConfig(title="Manual document", backend="demo", sources=[]))
    runner = Runner(tmp_path, offline=True)
    work = runner.store.upsert_work(Work(title="Study", arxiv_id="2401.00001v1", abstract=FIXTURES[0][1],
                                         screening="include", screening_reason="Test"))
    source = tmp_path / "attached.txt"
    source.write_text(text, encoding="utf-8")
    attached = runner.attach(work.id, source)["document_id"]
    runner.store.upsert_work(Work(title="Study", arxiv_id="2401.00001v2", abstract="Updated source abstract"))
    runner.prepare()
    runner.export()
    assert runner.store.active_document(work.id).id == attached


def test_export_configuration_freshness_and_stale_runner(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    runner.run()
    token = snapshot_token(runner.store)
    config = runner.config.model_copy(update={"title": "Changed title", "question": "Changed question"})
    atomic_json(tmp_path / "project.json", config.model_dump())
    assert not runner.status()["latest_export_current"]
    assert snapshot_token(runner.store)[:3] == token[:3]
    assert snapshot_token(runner.store)[3] != token[3]
    current = Runner(tmp_path, offline=True)
    assert not current.status()["latest_export_current"]
    with pytest.raises(ValueError, match="configuration changed"):
        runner.export()
    exported = current.export()
    review = (exported / "review.md").read_text(encoding="utf-8")
    assert "Changed title" in review and "Changed question" in review
    assert current.status()["latest_export_current"]
    manifest = json.loads((exported / "manifest.json").read_text())
    latest = json.loads((tmp_path / "exports/latest.json").read_text())
    assert manifest["snapshot"] == latest["snapshot"] == list(snapshot_token(current.store))
    assert current.store.budget()["consumed_or_reserved_calls"] == 4
    # Legacy pointers lacking configuration identity are conservatively stale.
    latest["snapshot"] = latest["snapshot"][:3]
    atomic_json(tmp_path / "exports/latest.json", latest)
    assert not current.status()["latest_export_current"]


def test_export_rejects_configuration_change_during_build(tmp_path, monkeypatch):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    runner.run()
    old_latest = (tmp_path / "exports/latest.json").read_bytes()
    from academic_agent import exports
    original = exports.atomic_json
    def racing_write(path, value):
        original(path, value)
        if path.name == "manifest.json":
            config = runner.config.model_copy(update={"question": "Changed during export"})
            atomic_json(tmp_path / "project.json", config.model_dump())
    monkeypatch.setattr(exports, "atomic_json", racing_write)
    with pytest.raises(ValueError, match="changed during export"):
        runner.export()
    assert (tmp_path / "exports/latest.json").read_bytes() == old_latest
    assert not runner.status()["latest_export_current"]


@pytest.mark.parametrize("failure", ["unavailable", "rate_limited", "partial"])
@pytest.mark.parametrize("resume", [True, False])
def test_discovery_recovers_transient_failure_without_losing_successes(tmp_path, monkeypatch, failure, resume):
    create_project(tmp_path, ProjectConfig(title="Recovery", backend="demo", queries=["study"], sources=["good", "retry"]))
    from academic_agent import workflow
    paper = Work(title="Study", source_ids={"good": "1"}, abstract=FIXTURES[0][1],
                 screening="include", screening_reason="Test")
    requests = []
    def collect(*args):
        requests.append(args)
        return [SearchPage(source="good", records=[paper]),
                SearchPage(source="retry", status=failure if len(requests) == 1 else "empty")]
    monkeypatch.setattr(workflow, "collect_sources", collect)
    monkeypatch.setattr(workflow, "check_publication", lambda *args, **kwargs: [PublicationCheck(source="test")])
    runner = Runner(tmp_path)
    with pytest.raises(WorkflowPaused, match="Discovery incomplete"):
        runner.run()
    assert len(requests) == 1 and len(runner.store.works()) == 1
    assert runner.read_selected()["results"][0]["status"] == "ok"
    assert runner.review_selected()["results"][0]["status"] == "ok"
    runner.export()  # Available corpus remains usable through explicit commands.
    assert runner.store.budget()["consumed_or_reserved_calls"] == 2
    assert Runner(tmp_path).run(resume=resume)["complete"]
    assert len(requests) == 2 and len(runner.store.works()) == 1
    assert runner.store.budget()["consumed_or_reserved_calls"] == 2
    assert runner.run(resume=True)["already_complete"]
    runner.run()
    assert len(requests) == 2  # Healthy zero hits are now reusable.


def test_all_unavailable_discovery_recovers_on_resume(tmp_path, monkeypatch):
    create_project(tmp_path, ProjectConfig(title="Recovery", backend="demo", queries=["study"], sources=["retry"]))
    from academic_agent import workflow
    monkeypatch.setattr(workflow, "collect_sources", lambda *args: [SearchPage(source="retry", status="unavailable")])
    runner = Runner(tmp_path)
    with pytest.raises(WorkflowPaused, match="unavailable"):
        runner.run()
    monkeypatch.setattr(workflow, "collect_sources", lambda *args: [SearchPage(source="retry", records=[
        Work(title="Recovered candidate", source_ids={"retry": "1"})])])
    monkeypatch.setattr(workflow, "check_publication", lambda *args, **kwargs: [PublicationCheck(source="test")])
    with pytest.raises(WorkflowPaused, match="Screen"):
        Runner(tmp_path).run(resume=True)
    assert len(runner.store.works()) == 1


def test_healthy_empty_discovery_is_cacheable_but_legacy_failed_completion_retries(tmp_path, monkeypatch):
    create_project(tmp_path, ProjectConfig(title="Empty", backend="demo", queries=["study"], sources=["one"]))
    from academic_agent import workflow
    requests = []
    def collect(*args):
        requests.append(args)
        return [SearchPage(source="one", status="empty")]
    monkeypatch.setattr(workflow, "collect_sources", collect)
    runner = Runner(tmp_path)
    assert runner.run()["complete"]
    assert runner.run(resume=True)["already_complete"]
    assert runner.run()["complete"]
    assert len(requests) == 1
    # Represent a pre-fix completed checkpoint with a latest failed discovery log.
    entry = runner.store.logs("search")[-1]["data"]
    entry["summary"]["sources"][0]["status"] = "unavailable"
    runner.store.log("search", entry)
    assert Runner(tmp_path).run(resume=True)["complete"]
    assert len(requests) == 2


def test_discovery_exception_is_retryable(tmp_path, monkeypatch):
    create_project(tmp_path, ProjectConfig(title="Recovery", backend="demo", queries=["study"]))
    from academic_agent import workflow
    def failure(*args):
        raise RuntimeError("Synthetic collector interruption")
    monkeypatch.setattr(workflow, "collect_sources", failure)
    runner = Runner(tmp_path)
    with pytest.raises(RuntimeError, match="collector interruption"):
        runner.run()
    monkeypatch.setattr(workflow, "collect_sources", lambda *args: [SearchPage(source="one", status="empty")])
    assert Runner(tmp_path).run(resume=True)["complete"]
