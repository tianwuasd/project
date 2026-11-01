import json
from pathlib import Path

import pytest

from academic_agent.backends import ExchangeBackend
from academic_agent.config import atomic_json, load_config
from academic_agent.demo import seed_demo
from academic_agent.domain import (
    domain_context_payload,
    extract_domain,
    load_domain_version,
    validate_domain_output,
    validate_domain_review,
)
from academic_agent.models import ModelJob, ModelResult
from academic_agent.research_artifacts import load_research, publish_research
from academic_agent.workflow import Runner
from academic_agent.writing_context import load_writing_context
from academic_agent.writing_models import WritingError, WritingSelection


def _evidence(job, phrase):
    return next(item["id"] for item in job.payload["research"]["evidence"]
                if phrase in item["text"])


def _claim_evidence(job, category):
    claim = next(item for item in job.payload["research"]["claims"]
                 if item["category"] == category)
    return claim["evidence_ids"][0]


class DomainBackend:
    name = "controlled-domain"

    def __init__(self, *, fault=None):
        self.jobs = []
        self.fault = fault

    def run(self, job):
        self.jobs.append(job)
        task = job.payload["task"]
        title = job.payload["research"]["work"]["title"]
        is_a = title.endswith("A")
        field_ids = [item["id"] for item in job.payload["domain"]["fields"]]
        if task == "domain-extract-v1":
            if field_ids == ["simulated_jobs", "arrival_pattern", "waiting_time", "baseline"]:
                data_evidence = _evidence(job, "simulated jobs")
                result_evidence = _claim_evidence(job, "result")
                method_evidence = _claim_evidence(job, "method")
                values = {
                    "simulated_jobs": (40 if is_a else 60, [data_evidence]),
                    "arrival_pattern": ((None, []) if is_a else ("bursty", [method_evidence])),
                    "waiting_time": (("10 to 8 simulation units", [result_evidence]) if is_a
                                     else ("10 to 11 simulation units", [result_evidence])),
                    "baseline": (("round-robin", [method_evidence]) if is_a else (None, [])),
                }
            else:
                data_evidence = _evidence(job, "simulated jobs")
                limit_evidence = _evidence(job, "real")
                values = {
                    "observation_unit": ("simulated jobs", [data_evidence]),
                    "transfer_limit": ("cannot establish effects in real systems", [limit_evidence]),
                }
            fields = [{
                "field_id": field_id,
                "status": "reported" if values[field_id][0] is not None else "not_reported",
                "value": values[field_id][0],
                "evidence_ids": values[field_id][1],
            } for field_id in field_ids]
            if self.fault == "fictional_evidence":
                fields[0]["evidence_ids"] = ["evidence_fictional"]
            elif self.fault == "unknown_field":
                fields[0]["field_id"] = "unknown"
            elif self.fault == "wrong_type":
                fields[0]["value"] = "forty"
            data = {"fields": fields}
        else:
            items = [{"field_id": field["field_id"], "decision": "supported",
                      "reason": "Checked independently against the complete original source."}
                     for field in job.payload["fields"]]
            if self.fault == "omitted_review":
                items.pop()
            data = {"items": items}
        return ModelResult(status="succeeded", backend=self.name, data=data,
                           usage={"mode": "controlled_test", "model_calls": 1})


@pytest.fixture
def prepared(tmp_path):
    seed_demo(tmp_path)
    Runner(tmp_path, offline=True).run()
    return tmp_path


def _selection(root, *, domain_path="builtin:scheduling"):
    work_ids = [work.id for work in sorted(
        Runner(root, offline=True).store.works(), key=lambda work: work.title)]
    return WritingSelection(
        document_id="doc_" + "2" * 32,
        question="How do the scheduling reports differ?",
        work_ids=work_ids,
        domain_path=domain_path,
        trend_directory=None,
        candidate_directory=None,
        candidate_id=None,
        assessment_directory=None,
    )


def test_extracts_configured_fields_reviews_full_sources_and_loads_closed_artifact(prepared):
    backend = DomainBackend()
    result = extract_domain(Runner(prepared, offline=True, backend=backend), _selection(prepared))
    assert set(result) == {"status", "input_hash", "output", "gaps"}
    assert result["status"] == "complete" and result["gaps"] == []
    artifact = load_domain_version(prepared / result["output"])
    work_a = next(work_id for work_id, work in artifact.context.research.works.items()
                  if work.title.endswith("A"))
    work_b = next(work_id for work_id, work in artifact.context.research.works.items()
                  if work.title.endswith("B"))
    assert [field.field_id for field in artifact.fields[work_a]] == [
        "simulated_jobs", "arrival_pattern", "waiting_time", "baseline"]
    assert [field.value for field in artifact.fields[work_a]][:2] == [40, None]
    assert [field.value for field in artifact.fields[work_b]][:2] == [60, "bursty"]
    assert artifact.fields[work_b][-1].value is None
    assert {item.field_id for item in artifact.review[work_a]} == {
        field.field_id for field in artifact.fields[work_a]}
    assert set(domain_context_payload(artifact)) == {"context", "fields", "review"}

    assert [job.payload["task"] for job in backend.jobs] == [
        "domain-extract-v1", "domain-extract-v1", "domain-review-v1", "domain-review-v1"]
    assert all(job.kind == "synthesize" for job in backend.jobs)
    assert all(set(job.payload) == ({"task", "domain", "research", "work_id"}
                                    | ({"fields"} if job.payload["task"] == "domain-review-v1" else set()))
               for job in backend.jobs)
    for job in backend.jobs:
        source = job.payload["research"]
        assert source["base_context_input_hash"] == artifact.context.input_hash
        assert source["domain_pack_identity"] == {
            "source_path": artifact.context.pack.source_path,
            "sha256": artifact.context.pack.sha256,
        }
        assert source["document"]["pages"]
        assert source["evidence"] and source["claims"] and source["effective_reviews"]
        assert source["scope"]
        if job.payload["task"] == "domain-review-v1":
            assert {item["field_id"] for item in job.payload["fields"]} == {
                "simulated_jobs", "arrival_pattern", "waiting_time", "baseline"}

    comparison = (prepared / result["output"] / "comparison.md").read_text(encoding="utf-8")
    header = next(line for line in comparison.splitlines() if line.startswith("| work_id"))
    assert [cell.strip() for cell in header.strip("|").split("|")] == [
        "work_id", "simulated_jobs", "arrival_pattern", "waiting_time", "baseline"]
    assert "40 [supported]" in comparison and "60 [supported]" in comparison
    assert "not reported [supported]" in comparison
    progress = Runner(prepared, offline=True).store.logs("domain_progress")
    assert progress and all(set(item["data"]) == {
        "document_id", "input_hash", "stage", "status", "output", "error"} for item in progress)


def test_external_pack_changes_field_set_and_column_order_without_core_changes(prepared):
    external = prepared / "inputs" / "external-domain.json"
    external.write_text(json.dumps({
        "schema_version": 1,
        "contract": "writing-domain-1",
        "name": "external",
        "version": "1",
        "terminology": [{"term": "observation unit", "meaning": "A simulated workload unit."}],
        "search_hints": ["simulation scope"],
        "screening_hints": ["Retain explicit transfer limits."],
        "fields": [
            {"id": "observation_unit", "label": "Observation unit",
             "description": "Reported unit of observation.", "value_type": "text", "unit": None},
            {"id": "transfer_limit", "label": "Transfer limit",
             "description": "Limit on transfer to real systems.", "value_type": "text", "unit": None},
        ],
        "comparison_columns": ["transfer_limit", "observation_unit"],
        "evaluation_criteria": [{"id": "scope_transfer", "text": "Preserve simulation scope."}],
        "writing_structure": [{"id": "boundary_argument", "title": "Boundary argument",
                               "purpose": "Compare transfer limits."}],
    }, ensure_ascii=False), encoding="utf-8")
    selection = _selection(prepared, domain_path=str(external.resolve()))
    exchange = prepared / "exchange-domain"
    for _ in range(4):
        pending = extract_domain(
            Runner(prepared, offline=True, backend=ExchangeBackend(exchange)), selection)
        assert pending["status"] == "needs_input"
        request_path = next(path for path in exchange.glob("*.request.json")
                            if not path.with_name(path.name.replace(".request.", ".response.")).exists())
        request = json.loads(request_path.read_text(encoding="utf-8"))
        job = ModelJob.model_validate(request["job"])
        data = DomainBackend().run(job).data
        response_path = request_path.with_name(request_path.name.replace(".request.", ".response."))
        response_path.write_text(json.dumps({"fingerprint": request["fingerprint"], "data": data},
                                            ensure_ascii=False), encoding="utf-8")
    result = extract_domain(
        Runner(prepared, offline=True, backend=ExchangeBackend(exchange)), selection)
    artifact = load_domain_version(prepared / result["output"])
    assert result["status"] == "complete"
    for work_id in artifact.context.selection.work_ids:
        extracted = {field.field_id: field for field in artifact.fields[work_id]}
        assert list(extracted) == ["observation_unit", "transfer_limit"]
        assert extracted["observation_unit"].value == "simulated jobs"
        assert extracted["transfer_limit"].value == "cannot establish effects in real systems"
        for field in extracted.values():
            assert field.evidence_ids
            assert all(artifact.context.research.evidence[evidence_id].work_id == work_id
                       for evidence_id in field.evidence_ids)
    comparison = (prepared / result["output"] / "comparison.md").read_text(encoding="utf-8")
    header = next(line for line in comparison.splitlines() if line.startswith("| work_id"))
    assert [cell.strip() for cell in header.strip("|").split("|")] == [
        "work_id", "transfer_limit", "observation_unit"]
    assert comparison.count("cannot establish effects in real systems [supported]") == 2
    assert comparison.count("simulated jobs [supported]") == 2


@pytest.mark.parametrize("fault", [
    "fictional_evidence", "unknown_field", "wrong_type", "omitted_review",
])
def test_structural_failures_do_not_publish_or_enter_success_cache(prepared, fault):
    runner = Runner(prepared, offline=True, backend=DomainBackend(fault=fault))
    result = extract_domain(runner, _selection(prepared))
    assert result["status"] == "failed" and result["output"] is None and result["gaps"]
    synthesize = [job for job in runner.store.jobs() if job["kind"] == "synthesize"]
    assert any(job["status"] == "failed" for job in synthesize)
    assert not (prepared / "analyses" / "domain" / "latest.json").exists()


def test_direct_validators_translate_model_status_and_reject_cross_work_evidence(prepared):
    context = load_writing_context(prepared, _selection(prepared))
    work_a, work_b = context.selection.work_ids
    source_a = next(e for e in context.research.evidence.values() if e.work_id == work_a)
    result = ModelResult(status="succeeded", backend="controlled", data={"fields": [{
        "field_id": definition.id,
        "status": "reported" if definition.id == "simulated_jobs" else "not_reported",
        "value": 40 if definition.id == "simulated_jobs" else None,
        "evidence_ids": [source_a.id] if definition.id == "simulated_jobs" else [],
    } for definition in context.pack.pack.fields]})
    fields = validate_domain_output(context, work_a, result)
    foreign = next(e for e in context.research.evidence.values() if e.work_id == work_b)
    result.data["fields"][0]["evidence_ids"] = [foreign.id]
    with pytest.raises(WritingError):
        validate_domain_output(context, work_a, result)
    with pytest.raises(WritingError) as caught:
        validate_domain_output(context, work_a,
                               ModelResult(status="needs_input", backend="exchange"))
    assert caught.value.status == "needs_input"
    with pytest.raises(WritingError) as caught:
        validate_domain_review(context, work_a, fields,
                               ModelResult(status="failed", backend="controlled", error="bad"))
    assert caught.value.status == "failed"


def test_budget_pause_resumes_after_cached_first_work_and_usage_logs_do_not_invalidate(prepared):
    config = load_config(prepared)
    config.max_model_calls = 5  # Four reading/review reservations plus work A extraction.
    atomic_json(prepared / "project.json", config.model_dump())
    first_backend = DomainBackend()
    paused = extract_domain(Runner(prepared, offline=True, backend=first_backend), _selection(prepared))
    assert paused["status"] == "paused_budget"
    assert [job.payload["task"] for job in first_backend.jobs] == ["domain-extract-v1"]

    config.max_model_calls = 8
    atomic_json(prepared / "project.json", config.model_dump())
    runner = Runner(prepared, offline=True, backend=DomainBackend())
    runner.store.log("usage_note", {"detail": "does not alter domain semantics"})
    complete = extract_domain(runner, _selection(prepared))
    assert complete["status"] == "complete"
    assert [job.payload["task"] for job in runner.backend.backend.jobs] == [
        "domain-extract-v1", "domain-review-v1", "domain-review-v1"]
    assert runner.store.budget()["consumed_or_reserved_calls"] == 8


def test_changed_pack_hash_disables_reuse_but_intact_history_remains_loadable(prepared):
    config = load_config(prepared)
    config.max_model_calls = 20
    atomic_json(prepared / "project.json", config.model_dump())
    external = prepared / "external.json"
    builtin = json.loads((Path(__file__).parents[1] / "src" / "academic_agent" /
                          "domain_packs" / "scheduling.json").read_text(encoding="utf-8"))
    external.write_text(json.dumps(builtin, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    selection = _selection(prepared, domain_path=str(external.resolve()))
    first = extract_domain(Runner(prepared, offline=True, backend=DomainBackend()), selection)
    old = load_domain_version(prepared / first["output"])
    original_hash = old.context.pack.sha256
    external.write_text(json.dumps(builtin, ensure_ascii=False, indent=2), encoding="utf-8")
    backend = DomainBackend()
    second = extract_domain(Runner(prepared, offline=True, backend=backend), selection)
    assert second["status"] == "complete" and len(backend.jobs) == 4
    assert load_domain_version(prepared / second["output"]).context.pack.sha256 != original_hash

    relocated = prepared / "same-bytes-relocated.json"
    relocated.write_bytes(external.read_bytes())
    relocated_backend = DomainBackend()
    relocated_result = extract_domain(
        Runner(prepared, offline=True, backend=relocated_backend),
        _selection(prepared, domain_path=str(relocated.resolve())),
    )
    assert relocated_result["status"] == "complete" and len(relocated_backend.jobs) == 4
    relocated_artifact = load_domain_version(prepared / relocated_result["output"])
    assert relocated_artifact.context.pack.sha256 == load_domain_version(
        prepared / second["output"]).context.pack.sha256
    assert relocated_artifact.context.pack.source_path != load_domain_version(
        prepared / second["output"]).context.pack.source_path
    assert load_domain_version(Path(old.pin.version_path)) == old


def test_loader_rejects_host_payload_mismatch_even_when_manifest_hashes_are_valid(prepared):
    result = extract_domain(Runner(prepared, offline=True, backend=DomainBackend()), _selection(prepared))
    artifact = load_domain_version(prepared / result["output"])
    loaded = load_research(Path(artifact.pin.version_path))
    files = dict(loaded["files"])
    files["comparison.md"] += b"\nfictional row\n"
    republished = publish_research(Runner(prepared, offline=True).store, "domain",
                                   artifact.context.research, files)
    with pytest.raises(ValueError, match="closure"):
        load_domain_version(republished)


def test_input_too_large_and_source_only_stale_vs_malformed_failed_are_precise(prepared):
    config = load_config(prepared)
    config.max_input_chars = 100
    atomic_json(prepared / "project.json", config.model_dump())
    runner = Runner(prepared, offline=True, backend=DomainBackend())
    oversized = extract_domain(runner, _selection(prepared))
    assert oversized["status"] == "input_too_large" and not runner.backend.backend.jobs
    assert runner.store.budget()["consumed_or_reserved_calls"] == 4

    config.max_input_chars = 100_000
    atomic_json(prepared / "project.json", config.model_dump())
    external = prepared / "status-domain.json"
    builtin = json.loads((Path(__file__).parents[1] / "src" / "academic_agent" /
                          "domain_packs" / "scheduling.json").read_text(encoding="utf-8"))
    external.write_text(json.dumps(builtin, ensure_ascii=False), encoding="utf-8")

    class ChangesIntactPack(DomainBackend):
        def run(self, job):
            result = super().run(job)
            if len(self.jobs) == 1:
                external.write_text(json.dumps({**builtin, "version": "2"}, ensure_ascii=False),
                                    encoding="utf-8")
            return result

    stale = extract_domain(
        Runner(prepared, offline=True, backend=ChangesIntactPack()),
        _selection(prepared, domain_path=str(external.resolve())),
    )
    assert stale["status"] == "stale"

    context = load_writing_context(prepared, _selection(prepared))
    source = prepared / next(iter(context.research.documents.values())).source_path

    class ChangesIntactSource(DomainBackend):
        def run(self, job):
            result = super().run(job)
            if len(self.jobs) == 1:
                source.write_text("changed replacement source bytes", encoding="utf-8")
            return result

    source_stale = extract_domain(
        Runner(prepared, offline=True, backend=ChangesIntactSource()), _selection(prepared),
    )
    assert source_stale["status"] == "stale"
    assert not (prepared / "analyses" / "domain" / "latest.json").exists()

    with Runner(prepared, offline=True).store.connect() as connection:
        connection.execute("UPDATE logs SET data='{' WHERE kind='review'")
    invalid_backend = DomainBackend()
    invalid = extract_domain(
        Runner(prepared, offline=True, backend=invalid_backend), _selection(prepared),
    )
    assert invalid["status"] == "failed"
    assert not invalid_backend.jobs
