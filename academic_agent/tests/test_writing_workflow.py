import json
from urllib.parse import quote

import pytest
from test_domain_analysis import DomainBackend
from test_writing_analysis import WritingBackend

from academic_agent.config import atomic_json, load_config
from academic_agent.demo import FIXTURES, seed_demo
from academic_agent.documents import abstract_document
from academic_agent.domain import extract_domain, load_domain_version
from academic_agent.models import ModelResult
from academic_agent.research_artifacts import load_research, publish_research
from academic_agent.trends import build_trends
from academic_agent.workflow import Runner
from academic_agent.writing import (
    assert_writing_dependencies_current,
    create_draft,
    create_outline,
    load_writing_version,
    render_writing,
    writing_dependencies_current,
    writing_status,
)
from academic_agent.writing_models import (
    Draft,
    Revision,
    WritingArtifact,
    WritingError,
    WritingReview,
    WritingSelection,
)


def _selection(root):
    work_ids = [work.id for work in sorted(Runner(root, offline=True).store.works(), key=lambda work: work.title)]
    return WritingSelection(
        document_id="doc_" + "8" * 32,
        question="How do the scheduling reports differ?",
        work_ids=work_ids,
        domain_path="builtin:scheduling",
        trend_directory=None,
        candidate_directory=None,
        candidate_id=None,
        assessment_directory=None,
    )


@pytest.fixture
def project(tmp_path):
    seed_demo(tmp_path)
    Runner(tmp_path, offline=True).run()
    selection = _selection(tmp_path)
    domain = extract_domain(Runner(tmp_path, offline=True, backend=DomainBackend()), selection)
    assert domain["status"] == "complete"
    return tmp_path, selection, tmp_path / domain["output"]


def test_outline_then_manuscript_are_structured_reviewed_and_closed(project):
    root, selection, domain_path = project
    outline_result = create_outline(
        Runner(root, offline=True, backend=WritingBackend()), selection, domain_path
    )
    assert outline_result["status"] == "complete"
    progress = Runner(root, offline=True).store.logs("writing_progress")
    assert progress[0]["data"] == {
        "document_id": selection.document_id,
        "input_hash": outline_result["input_hash"],
        "stage": "identity",
        "status": "complete",
        "output": None,
        "error": None,
    }
    outline_path = root / outline_result["output"]
    outline = load_writing_version(outline_path)
    assert outline.revision.extent == "outline" and outline.revision.stage == "reviewed"
    assert outline.draft.blocks == []
    assert {item.unit_id for item in outline.review.items} == {
        unit.id for block in outline.outline.blocks for unit in block.units
    }

    draft_backend = WritingBackend()
    draft_result = create_draft(Runner(root, offline=True, backend=draft_backend), outline_path)
    assert draft_result["status"] == "complete"
    manuscript_path = root / draft_result["output"]
    manuscript = load_writing_version(manuscript_path)
    assert manuscript.revision.extent == "manuscript" and manuscript.revision.stage == "reviewed"
    all_units = [unit for block in [*manuscript.outline.blocks, *manuscript.draft.blocks] for unit in block.units]
    assert [item.unit_id for item in manuscript.review.items] == [unit.id for unit in all_units]
    assert [job.payload["task"] for job in draft_backend.jobs] == ["writing-draft-v1", "writing-review-v1"]
    review_job = draft_backend.jobs[-1]
    assert review_job.payload["outline"] == manuscript.outline.model_dump(mode="json")
    assert review_job.payload["draft"] == manuscript.draft.model_dump(mode="json")

    files = load_research(manuscript_path)["files"]
    assert files == render_writing(manuscript)
    report = files["report.md"].decode()
    assert "10 to 8" in report and "10 to 11" in report and "bursty" in report
    assert all(unit.text in report for unit in all_units)
    evidence = json.loads(files["evidence.json"])
    assert {unit["id"] for unit in evidence["units"]} == {unit.id for unit in all_units}
    assert "not human validation" in report
    for unit in all_units:
        sources = []
        for reference in unit.references:
            sources.extend(reference.sources if reference.type == "synthesis" else [reference])
        for source in sources:
            work = manuscript.context.research.works[source.work_id]
            document = manuscript.context.research.documents[source.document_id]
            locator = "../../../" + quote(document.source_path, safe="/")
            assert f"[@{work.citekey}]({locator})" in report
            for evidence_id in source.evidence_ids:
                item = manuscript.context.research.evidence[evidence_id]
                assert (
                    f"`{evidence_id}` (page {item.page}, chars {item.start}:{item.end})"
                    in report
                )


def test_outline_recaptures_an_explicit_optional_pin_from_a_base_only_domain(project):
    root, base_selection, domain_path = project
    trend_path = build_trends(Runner(root, offline=True).store).resolve()
    selected = base_selection.model_copy(update={"trend_directory": str(trend_path)})
    backend = WritingBackend()

    result = create_outline(Runner(root, offline=True, backend=backend), selected, domain_path)

    assert result["status"] == "complete"
    artifact = load_writing_version(root / result["output"])
    assert artifact.context.selection == selected
    assert artifact.context.trend is not None
    assert artifact.context.trend.artifact.version_path == str(trend_path)
    assert artifact.domain_pin == load_domain_version(domain_path).pin
    assert [job.payload["task"] for job in backend.jobs] == [
        "writing-outline-v1",
        "writing-review-v1",
    ]


def test_report_links_frozen_statistic_to_its_trend_sources(project):
    root, base_selection, domain_path = project
    trend_path = build_trends(Runner(root, offline=True).store).resolve()
    selected = base_selection.model_copy(update={"trend_directory": str(trend_path)})

    class StatisticBackend(WritingBackend):
        def run(self, job):
            response = super().run(job)
            if job.payload["task"] == "writing-outline-v1":
                statistic = job.payload["context"]["reference_catalog"]["statistics"][0]
                response.data["blocks"][1]["units"][0] = {
                    "text": "The frozen eligible count is retained from the selected trend snapshot.",
                    "kind": "frozen_statistic",
                    "references": [{
                        "type": "frozen_statistic",
                        "pin_hash": statistic["pin_hash"],
                        "computation_path": statistic["path"],
                    }],
                    "purpose": "Report a frozen corpus statistic.",
                }
            return response

    result = create_outline(
        Runner(root, offline=True, backend=StatisticBackend()), selected, domain_path
    )
    assert result["status"] == "complete"
    artifact = load_writing_version(root / result["output"])
    report = load_research(root / result["output"])["files"]["report.md"].decode()
    statistic = artifact.outline.blocks[1].units[0].references[0]
    assert f"Frozen statistic `{statistic.computation_path}`" in report
    assert f"trend pin `{statistic.pin_hash}`" in report
    source_ids = {
        work_id
        for work_ids in artifact.context.trend.computation.work_ids.values()
        for work_id in work_ids
    }
    assert source_ids
    for work_id in source_ids:
        work = artifact.context.research.works[work_id]
        document = next(
            value for value in artifact.context.research.documents.values()
            if value.work_id == work_id
        )
        locator = "../../../" + quote(document.source_path, safe="/")
        assert f"[@{work.citekey}]({locator})" in report


class AbstractResearchBackend:
    name = "controlled-abstract-research"

    def run(self, job):
        if job.kind == "read":
            evidence_id = job.payload["evidence"][0]["id"]
            data = {
                "claims": [
                    {
                        "text": f"Abstract-grounded {category} statement.",
                        "category": category,
                        "evidence_ids": [evidence_id],
                    }
                    for category in ("problem", "method", "data", "result", "limitation", "contribution")
                ],
                "missing": [],
            }
        else:
            data = {
                "verdicts": [
                    {
                        "claim_id": claim["id"],
                        "verdict": "supported",
                        "rationale": "Checked against the complete captured abstract.",
                        "evidence_ids": claim["evidence_ids"],
                    }
                    for claim in job.payload["reading_card"]["claims"]
                ]
            }
        return ModelResult(
            status="succeeded",
            backend=self.name,
            data=data,
            usage={"mode": "controlled_test", "model_calls": 1},
        )


def _abstract_project(root):
    seed_demo(root)
    runner = Runner(root, offline=True, backend=AbstractResearchBackend())
    fixture_text = {title: text for title, text in FIXTURES}
    for work in runner.store.works():
        work.abstract = fixture_text[work.title]
        runner.store.update_work(work)
        document = runner.store.add_document(abstract_document(work))
        assert document.level == "abstract_only" and document.source_path == ""
    assert all(item["status"] == "ok" for item in runner.read_selected()["results"])
    assert all(item["status"] == "ok" for item in runner.review_selected()["results"])
    selection = _selection(root)
    domain_result = extract_domain(Runner(root, offline=True, backend=DomainBackend()), selection)
    assert domain_result["status"] == "complete"
    return selection, root / domain_result["output"]


def _assert_abstract_target(version_path, artifact, report, work_id):
    work = artifact.context.research.works[work_id]
    document = next(
        item for item in artifact.context.research.documents.values()
        if item.work_id == work_id
    )
    label = f"@{work.citekey}; abstract-only document {document.id}"
    assert f"[{label}](context.json)" in report
    target = version_path / "context.json"
    assert target.is_file() and target.resolve() != version_path.parents[2].resolve()
    captured = json.loads(target.read_text(encoding="utf-8"))["context"]["research"]
    assert captured["documents"][document.id] == document.model_dump(mode="json")
    assert captured["documents"][document.id]["pages"] == [
        page.model_dump(mode="json") for page in document.pages
    ]
    return document


def test_report_abstract_source_and_synthesis_links_target_captured_context(tmp_path):
    selection, domain_path = _abstract_project(tmp_path)

    class DirectAndSynthesisBackend(WritingBackend):
        def run(self, job):
            response = super().run(job)
            if job.payload["task"] == "writing-outline-v1":
                synthesis = response.data["blocks"][1]["units"][0]
                first_source = synthesis["references"][0]["sources"][0]
                response.data["blocks"][1]["units"] = [
                    {
                        "text": "The first captured abstract supplies this bounded statement. ",
                        "kind": "source",
                        "references": [first_source],
                        "purpose": "Exercise a direct abstract-only source.",
                    },
                    synthesis,
                ]
            return response

    result = create_outline(
        Runner(tmp_path, offline=True, backend=DirectAndSynthesisBackend()),
        selection,
        domain_path,
    )
    assert result["status"] == "complete"
    version_path = tmp_path / result["output"]
    artifact = load_writing_version(version_path)
    report = load_research(version_path)["files"]["report.md"].decode()
    direct_unit, synthesis_unit = artifact.outline.blocks[1].units
    direct = direct_unit.references[0]
    document = _assert_abstract_target(version_path, artifact, report, direct.work_id)
    assert report.count("(context.json)") >= 3
    assert document.source_path == "" and document.level == "abstract_only"
    for unit in (direct_unit, synthesis_unit):
        assert unit.text in report
    for evidence_id in direct.evidence_ids:
        evidence = artifact.context.research.evidence[evidence_id]
        assert f"`{evidence_id}` (page {evidence.page}, chars {evidence.start}:{evidence.end})" in report
        assert document.pages[0].text[evidence.start:evidence.end] == evidence.text


def test_report_abstract_statistic_sources_target_captured_context(tmp_path):
    selection, domain_path = _abstract_project(tmp_path)
    trend_path = build_trends(Runner(tmp_path, offline=True).store).resolve()
    selected = selection.model_copy(update={"trend_directory": str(trend_path)})

    class AbstractStatisticBackend(WritingBackend):
        def run(self, job):
            response = super().run(job)
            if job.payload["task"] == "writing-outline-v1":
                statistic = job.payload["context"]["reference_catalog"]["statistics"][0]
                response.data["blocks"][1]["units"][0] = {
                    "text": "The frozen eligible count describes the captured abstract-only corpus.",
                    "kind": "frozen_statistic",
                    "references": [{
                        "type": "frozen_statistic",
                        "pin_hash": statistic["pin_hash"],
                        "computation_path": statistic["path"],
                    }],
                    "purpose": "Exercise statistic links for abstract-only sources.",
                }
            return response

    result = create_outline(
        Runner(tmp_path, offline=True, backend=AbstractStatisticBackend()),
        selected,
        domain_path,
    )
    assert result["status"] == "complete"
    version_path = tmp_path / result["output"]
    artifact = load_writing_version(version_path)
    report = load_research(version_path)["files"]["report.md"].decode()
    for work_id in artifact.context.trend.computation.work_ids["eligible"]:
        document = _assert_abstract_target(version_path, artifact, report, work_id)
        assert document.source_path == "" and document.level == "abstract_only"
    unit = artifact.outline.blocks[1].units[0]
    assert unit.text in report
    assert "../../../." not in report


def test_dependency_fence_distinguishes_stale_sources_from_invalid_domain_bytes(project):
    root, selection, domain_path = project
    domain = load_domain_version(domain_path)
    assert writing_dependencies_current(root, domain.context, domain.pin) == {"current": True, "gaps": []}
    source = root / next(iter(domain.context.research.documents.values())).source_path
    source.write_bytes(source.read_bytes() + b"changed")
    state = writing_dependencies_current(root, domain.context, domain.pin)
    assert state["current"] is False
    assert all(not gap["reason"].endswith("invalid") for gap in state["gaps"])
    with pytest.raises(WritingError) as caught:
        assert_writing_dependencies_current(root, domain.context, domain.pin)
    assert caught.value.status == "stale"

    source.write_bytes(source.read_bytes()[:-7])
    (domain_path / "domain.json").write_bytes((domain_path / "domain.json").read_bytes() + b" ")
    state = writing_dependencies_current(root, domain.context, domain.pin)
    assert state == {"current": False, "gaps": [{"reason": "domain_artifact_invalid"}]}
    with pytest.raises(WritingError) as caught:
        assert_writing_dependencies_current(root, domain.context, domain.pin)
    assert caught.value.status == "failed"


def test_domain_swap_after_model_validation_prevents_cache_commit_and_publication(project):
    root, selection, domain_path = project

    class SwapsDomain(WritingBackend):
        def run(self, job):
            result = super().run(job)
            if len(self.jobs) == 1:
                path = domain_path / "domain-review.json"
                path.write_bytes(path.read_bytes() + b" ")
            return result

    runner = Runner(root, offline=True, backend=SwapsDomain())
    result = create_outline(runner, selection, domain_path)
    assert result["status"] == "failed" and result["output"] is None
    assert any(job["status"] == "failed" for job in runner.store.jobs("synthesize"))
    assert not (root / "analyses" / "writing" / "latest.json").exists()


def test_source_change_during_model_returns_stale(project):
    root, selection, domain_path = project
    domain = load_domain_version(domain_path)
    source = root / next(iter(domain.context.research.documents.values())).source_path

    class ChangesSource(WritingBackend):
        def run(self, job):
            result = super().run(job)
            if len(self.jobs) == 1:
                source.write_bytes(source.read_bytes() + b"changed")
            return result

    result = create_outline(Runner(root, offline=True, backend=ChangesSource()), selection, domain_path)
    assert result["status"] == "stale" and result["output"] is None


def test_reviewed_generation_is_cached_when_publication_fails_then_retry_uses_zero_model_calls(
        project, monkeypatch):
    root, selection, domain_path = project
    import academic_agent.writing as writing

    actual_publish = writing.publish_research
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("controlled publication failure after review")
        return actual_publish(*args, **kwargs)

    monkeypatch.setattr(writing, "publish_research", fail_once)
    first_backend = WritingBackend()
    first = create_outline(Runner(root, offline=True, backend=first_backend), selection, domain_path)
    assert first["status"] == "failed" and len(first_backend.jobs) == 2
    first_ids = [job.id for job in first_backend.jobs]

    retry_backend = WritingBackend()
    retry = create_outline(Runner(root, offline=True, backend=retry_backend), selection, domain_path)
    assert retry["status"] == "complete" and retry_backend.jobs == []
    artifact = load_writing_version(root / retry["output"])
    assert first_ids[0] != first_ids[1]
    assert [unit.id for block in artifact.outline.blocks for unit in block.units] == \
        first_backend.jobs[1].payload["target_ids"]


def test_manuscript_full_review_is_cached_across_publication_failure(project, monkeypatch):
    root, selection, domain_path = project
    outline = create_outline(Runner(root, offline=True, backend=WritingBackend()), selection, domain_path)
    outline_path = root / outline["output"]
    import academic_agent.writing as writing

    actual_publish = writing.publish_research
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("controlled failure after complete manuscript review")
        return actual_publish(*args, **kwargs)

    monkeypatch.setattr(writing, "publish_research", fail_once)
    first_backend = WritingBackend()
    first = create_draft(Runner(root, offline=True, backend=first_backend), outline_path)
    assert first["status"] == "failed"
    assert [job.payload["task"] for job in first_backend.jobs] == ["writing-draft-v1", "writing-review-v1"]
    reviewed_targets = first_backend.jobs[-1].payload["target_ids"]

    retry_backend = WritingBackend()
    retry = create_draft(Runner(root, offline=True, backend=retry_backend), outline_path)
    assert retry["status"] == "complete" and retry_backend.jobs == []
    artifact = load_writing_version(root / retry["output"])
    assert [item.unit_id for item in artifact.review.items] == reviewed_targets


def test_source_change_at_publication_boundary_prevents_publish(project, monkeypatch):
    root, selection, domain_path = project
    domain = load_domain_version(domain_path)
    source = root / next(iter(domain.context.research.documents.values())).source_path
    import academic_agent.writing as writing

    actual_publish = writing.publish_research

    def change_then_publish(*args, **kwargs):
        source.write_bytes(source.read_bytes() + b"changed at publication")
        return actual_publish(*args, **kwargs)

    monkeypatch.setattr(writing, "publish_research", change_then_publish)
    result = create_outline(Runner(root, offline=True, backend=WritingBackend()), selection, domain_path)
    assert result["status"] == "stale" and result["output"] is None
    assert not (root / "analyses" / "writing" / "latest.json").exists()


def test_pending_outline_returns_review_action_without_dispatch(project):
    root, selection, domain_path = project
    domain = load_domain_version(domain_path)
    complete = create_outline(Runner(root, offline=True, backend=WritingBackend()), selection, domain_path)
    reviewed = load_writing_version(root / complete["output"])
    pending = WritingArtifact(
        context=reviewed.context,
        domain_pin=reviewed.domain_pin,
        outline=reviewed.outline,
        draft=Draft(blocks=[]),
        review=WritingReview(items=[]),
        revision=Revision(
            schema_version=1, document_id=selection.document_id, extent="outline", stage="pending",
            parent_version=None, parent_manifest_hash=None, origin="model", author="controlled-writing/unavailable",
            reason="pending independent review", created_at="2026-09-10T00:00:00+00:00", input_sha256=None,
            changed_ids=[], added_ids=[], deleted_ids=[], reordered=False, id_map={},
            pending_ids=[], reused_from={}, domain_rebind=None,
        ),
    )
    pending_path = publish_research(
        Runner(root, offline=True).store, "writing", domain.context.research, render_writing(pending)
    )
    backend = WritingBackend()
    result = create_draft(Runner(root, offline=True, backend=backend), pending_path)
    assert result["status"] == "needs_input" and backend.jobs == []
    assert result["gaps"][0]["argv"] == ["academic-agent", "writing-review", str(pending_path)]

    manuscript_result = create_draft(
        Runner(root, offline=True, backend=WritingBackend()), root / complete["output"]
    )
    manuscript = load_writing_version(root / manuscript_result["output"])
    pending_manuscript = manuscript.model_copy(update={
        "review": WritingReview(items=[]),
        "revision": manuscript.revision.model_copy(update={"stage": "pending"}),
    })
    pending_manuscript_path = publish_research(
        Runner(root, offline=True).store,
        "writing",
        domain.context.research,
        render_writing(pending_manuscript),
    )
    manuscript_backend = WritingBackend()
    invalid = create_draft(
        Runner(root, offline=True, backend=manuscript_backend), pending_manuscript_path
    )
    assert invalid["status"] == "failed" and manuscript_backend.jobs == []
    assert invalid["gaps"][0]["reason"] == "load-outline"


def test_draft_rejects_mutable_writing_collection_alias_before_dispatch(project):
    root, selection, domain_path = project
    complete = create_outline(
        Runner(root, offline=True, backend=WritingBackend()), selection, domain_path
    )
    concrete_path = root / complete["output"]
    latest_before = load_research(root / "analyses" / "writing")["manifest"]["version"]
    assert load_writing_version(concrete_path).revision.extent == "outline"

    backend = WritingBackend()
    result = create_draft(
        Runner(root, offline=True, backend=backend), root / "analyses" / "writing"
    )
    assert result["status"] == "failed" and backend.jobs == []
    assert load_research(root / "analyses" / "writing")["manifest"]["version"] == latest_before


def test_over_limit_paused_after_outline_and_status_is_read_only(project, monkeypatch):
    root, selection, domain_path = project
    config = load_config(root)
    config.max_input_chars = 100
    atomic_json(root / "project.json", config.model_dump())
    backend = WritingBackend()
    oversized = create_outline(Runner(root, offline=True, backend=backend), selection, domain_path)
    assert oversized["status"] == "input_too_large" and backend.jobs == []

    config.max_input_chars = 100_000
    config.max_model_calls = 9  # 4 reading/review + 4 domain + outline only.
    atomic_json(root / "project.json", config.model_dump())
    paused_backend = WritingBackend()
    paused = create_outline(Runner(root, offline=True, backend=paused_backend), selection, domain_path)
    assert paused["status"] == "paused_budget"
    assert [job.payload["task"] for job in paused_backend.jobs] == ["writing-outline-v1"]

    config.max_model_calls = 12
    atomic_json(root / "project.json", config.model_dump())
    completed = create_outline(Runner(root, offline=True, backend=WritingBackend()), selection, domain_path)
    assert completed["status"] == "complete"
    monkeypatch.setattr("academic_agent.workflow.Runner", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr("academic_agent.store.Store", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    status = writing_status(root, selection.document_id)
    assert status["status"] == "complete"
    assert status["documents"][0]["versions"][-1]["current"] is True
    artifact = load_writing_version(root / completed["output"])
    source = root / next(iter(artifact.context.research.documents.values())).source_path
    source.write_bytes(source.read_bytes() + b"intact external change")
    stale = writing_status(root, selection.document_id)
    assert stale["status"] == "stale"
    assert stale["documents"][0]["versions"][-1]["available"] is True
    assert load_writing_version(root / completed["output"]) == artifact
    (domain_path / "domain.json").write_bytes((domain_path / "domain.json").read_bytes() + b" ")
    invalid = writing_status(root, selection.document_id)
    assert invalid["status"] == "failed"
    assert invalid["documents"][0]["versions"][-1]["available"] is False
    assert invalid["documents"][0]["versions"][-1]["gaps"] == [
        {"reason": "writing_artifact_invalid"}
    ]


def test_loader_rejects_missing_review_coverage_even_with_valid_new_manifest(project):
    root, selection, domain_path = project
    result = create_outline(Runner(root, offline=True, backend=WritingBackend()), selection, domain_path)
    path = root / result["output"]
    loaded = load_research(path)
    files = dict(loaded["files"])
    review = json.loads(files["review.json"])
    review["items"].pop()
    files["review.json"] = json.dumps(
        review, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode()
    snapshot = load_writing_version(path).context.research
    republished = publish_research(Runner(root, offline=True).store, "writing", snapshot, files)
    with pytest.raises(ValueError):
        load_writing_version(republished)
