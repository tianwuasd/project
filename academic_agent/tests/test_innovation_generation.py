import copy
import hashlib
import json

import pytest

from academic_agent.backends import DemoBackend, ExchangeBackend, job_prompt
from academic_agent.config import atomic_json, load_config
from academic_agent.demo import FIXTURES, seed_demo
from academic_agent.innovation_analysis import InnovationError, generate_candidates, review_candidates
from academic_agent.innovation_models import CandidateBatch, GroundingReview
from academic_agent.innovations import create_innovations
from academic_agent.models import ModelResult
from academic_agent.research_context import freeze_research, research_fingerprint
from academic_agent.workflow import Runner


def _generation(job):
    research = job.payload["research"]
    chains = [chain for work in research["works"] for chain in work["reviewed_claims"]
              if chain["decision"]["verdict"] == "supported"]
    contributions = [chain for chain in chains if chain["claim"]["category"] == "contribution"]
    chosen = contributions or chains[:1]
    return {
        "candidates": [{
            "statement_type": "system_hypothesis",
            "question": "在不同合成负载下，按负载切换调度策略能否降低平均等待时间？",
            "closest_work_ids": [work["work"]["id"] for work in research["works"]],
            "closest_claim_ids": [chain["claim"]["id"] for chain in chosen],
            "claimed_difference": "提出按负载特征选择策略，而不是把固定优先级规则视为普遍改进。",
            "grounding_claim_ids": [chain["claim"]["id"] for chain in chains[:2]],
            "grounding_evidence_ids": sorted({evidence["id"] for chain in chains[:2]
                                                for evidence in chain["evidence"]}),
            "possible_counterexamples": ["突发到达负载可能使固定优先级策略增加等待时间。"],
            "falsifiable_hypothesis": "预注册的负载感知切换规则在留出负载上降低平均等待时间。",
            "minimum_validation": "在两种已给负载及一组留出负载上与两种固定策略比较。",
            "data_requirements": "带到达时间和服务时间的原创合成作业轨迹。",
            "compute_requirements": "可重复运行的小型离散事件模拟。",
            "risks": ["规则可能过拟合两种已知负载。"],
            "targeted_queries": ["workload adaptive scheduling mean waiting time counterexample"],
            "material_cutoff": research["material_cutoff"],
        }],
    }


def _review(job, *, verdict="partial"):
    candidate = job.payload["batch"]["candidates"][0]
    return {"items": [{
        "candidate_id": candidate["id"],
        "verdict": verdict,
        "rationale": "所引材料支持负载相关差异，但不证明新颖性或未运行实验的成功。",
        "claim_ids": candidate["grounding_claim_ids"],
        "evidence_ids": candidate["grounding_evidence_ids"],
    }]}


class RecordingBackend:
    name = "recording"

    def __init__(self, *, bad_generation=False, review_items=None, mutate=None):
        self.jobs = []
        self.bad_generation = bad_generation
        self.review_items = review_items
        self.mutate = mutate

    def run(self, job):
        self.jobs.append(job)
        if self.mutate:
            self.mutate(job)
        if job.payload["task"] == "innovation-generation-v1":
            data = _generation(job)
            if self.bad_generation:
                data["candidates"][0]["grounding_claim_ids"] = ["claim_not_supplied"]
        else:
            data = _review(job)
            if self.review_items is not None:
                data["items"] = self.review_items(job, data["items"])
        return ModelResult(status="succeeded", backend=self.name, data=data,
                           usage={"mode": "controlled_test", "model_calls": 1})


@pytest.fixture
def prepared(tmp_path):
    seed_demo(tmp_path)
    Runner(tmp_path, offline=True).run()
    return tmp_path


def test_generation_and_review_are_distinct_closed_jobs_with_complete_selected_context(prepared):
    snapshot = freeze_research(Runner(prepared, offline=True).store)
    backend = RecordingBackend()
    batch = generate_candidates(snapshot, backend, timeout_seconds=5, max_input_chars=100_000)
    review = review_candidates(snapshot, batch, backend, timeout_seconds=5, max_input_chars=100_000)

    assert isinstance(batch, CandidateBatch)
    assert isinstance(review, GroundingReview)
    assert len(backend.jobs) == 2
    assert [job.kind for job in backend.jobs] == ["synthesize", "synthesize"]
    assert [job.payload["task"] for job in backend.jobs] == [
        "innovation-generation-v1", "innovation-grounding-review-v1"]
    assert backend.jobs[0].id != backend.jobs[1].id
    assert all(job.output_schema["additionalProperties"] is False for job in backend.jobs)
    assert all(set(job.output_schema["properties"]) == set(job.output_schema["required"])
               for job in backend.jobs)
    contribution_chains = [chain for work in backend.jobs[0].payload["research"]["works"]
                           for chain in work["reviewed_claims"]
                           if chain["claim"]["category"] == "contribution"]
    assert contribution_chains
    assert all(chain["statement_type"] == "author_contribution" for chain in contribution_chains)
    expected_text = {page.text for document in snapshot.documents.values() for page in document.pages}
    for job in backend.jobs:
        sent_text = {page["text"] for work in job.payload["research"]["works"]
                     for page in work["document"]["pages"]}
        assert sent_text == expected_text
        assert job.payload["research"]["input_hash"] == research_fingerprint(snapshot)

    candidate = batch.candidates[0]
    assert candidate.statement_type == "system_hypothesis"
    assert candidate.id.startswith("innovation_")
    assert not any(claim.category == "result" and claim.text == candidate.falsifiable_hypothesis
                   for card in snapshot.cards.values() for claim in card.claims)
    assert {item.candidate_id for item in review.items} == {candidate.id}
    assert review.items[0].verdict == "partial"


def test_candidate_ids_are_host_owned_stable_and_references_are_exact(prepared):
    snapshot = freeze_research(Runner(prepared, offline=True).store)
    backend = RecordingBackend()
    first = generate_candidates(snapshot, backend, timeout_seconds=5, max_input_chars=100_000)
    second = generate_candidates(snapshot, backend, timeout_seconds=5, max_input_chars=100_000)
    assert first.candidates[0].id == second.candidates[0].id
    candidate = first.candidates[0]
    assert set(candidate.closest_work_ids) <= set(snapshot.work_ids)
    assert set(candidate.closest_claim_ids + candidate.grounding_claim_ids) <= set(snapshot.supported_claim_ids)
    assert set(candidate.grounding_evidence_ids) <= set(snapshot.evidence)
    assert len(candidate.targeted_queries) <= 2

    bad = RecordingBackend(bad_generation=True)
    with pytest.raises(InnovationError, match="references"):
        generate_candidates(snapshot, bad, timeout_seconds=5, max_input_chars=100_000)

    class ReorderedBackend(RecordingBackend):
        def run(self, job):
            result = super().run(job)
            if job.payload["task"] == "innovation-generation-v1":
                candidate = result.data["candidates"][0]
                for field in ("closest_work_ids", "closest_claim_ids", "grounding_claim_ids",
                              "grounding_evidence_ids"):
                    candidate[field].reverse()
                candidate["question"] = "  " + candidate["question"] + "  "
            return result

    reordered = generate_candidates(snapshot, ReorderedBackend(), timeout_seconds=5, max_input_chars=100_000)
    assert reordered.candidates[0].id == first.candidates[0].id


def test_projection_omits_raw_and_unselected_bodies_but_hashes_complete_event(prepared):
    runner = Runner(prepared, offline=True)
    snapshot = freeze_research(runner.store)
    raw_sentinel = "RAW-UNSELECTED-SENTINEL-" + "z" * 200_000
    snapshot.search_events.append({
        "id": "event-raw", "kind": "search", "created": "2026-09-09T00:00:00+00:00",
        "data": {
            "query_key": "query-key", "retained_work_ids": [snapshot.work_ids[0], "unselected-work"],
            "summary": {"query": "counterexample", "retained": 2, "dropped_at_project_cap": 1,
                        "sources": [{"source": "fixture-index", "status": "partial", "records": 2,
                                     "limit": 10, "truncated": True, "errors": ["limited"]}]},
            "responses": [{"records": [{"id": "unselected-work", "abstract": raw_sentinel}],
                           "raw_responses": [{"provider_body": raw_sentinel}]}],
        },
    })
    first_backend = RecordingBackend()
    batch = generate_candidates(snapshot, first_backend, timeout_seconds=5, max_input_chars=100_000)
    review_candidates(snapshot, batch, first_backend, timeout_seconds=5, max_input_chars=100_000)
    first_job = first_backend.jobs[0]
    for job in first_backend.jobs:
        serialized = job_prompt(job)
        assert raw_sentinel not in serialized and "unselected-work" in serialized
        sent_texts = {page["text"] for work in job.payload["research"]["works"]
                      for page in work["document"]["pages"]}
        assert sent_texts == {text for _, text in FIXTURES}
    event = first_job.payload["research"]["retrieval"]["events"][0]
    assert event["query"] == "counterexample"
    assert event["sources"][0] == {
        "source": "fixture-index", "status": "partial", "count": 2, "limit": 10,
        "truncated": True, "errors": ["limited"],
    }
    expected_hash = hashlib.sha256(json.dumps(snapshot.search_events[0], ensure_ascii=False,
                                              sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert event["full_event_hash"] == expected_hash

    changed = copy.deepcopy(snapshot)
    changed.search_events[0]["data"]["responses"][0]["raw_responses"].append({"duplicate": True})
    second_backend = RecordingBackend()
    generate_candidates(changed, second_backend, timeout_seconds=5, max_input_chars=100_000)
    assert second_backend.jobs[0].id != first_job.id


def test_create_innovations_keeps_generation_cache_across_budget_pause(prepared):
    config = load_config(prepared)
    config.max_model_calls = 5  # four original reading/review calls plus generation
    atomic_json(prepared / "project.json", config.model_dump())
    first_backend = RecordingBackend()
    paused = create_innovations(Runner(prepared, offline=True, backend=first_backend))
    assert paused["status"] == "budget_paused"
    assert len(paused["candidate_ids"]) == 1
    assert len(first_backend.jobs) == 1
    assert not (prepared / "analyses" / "innovation" / "latest.json").exists()

    config.max_model_calls = 6
    atomic_json(prepared / "project.json", config.model_dump())
    resumed_backend = RecordingBackend()
    completed = create_innovations(Runner(prepared, offline=True, backend=resumed_backend))
    assert completed["status"] == "complete"
    assert [job.payload["task"] for job in resumed_backend.jobs] == ["innovation-grounding-review-v1"]
    directory = prepared / completed["output"]
    assert {path.name for path in directory.iterdir()} == {
        "manifest.json", "candidates.json", "grounding-review.json", "report.md",
        "references.bib", "evidence.json",
    }
    report = (directory / "report.md").read_text(encoding="utf-8")
    assert "候选" in report and "不证明" in report and "partial" in report
    assert CandidateBatch.model_validate_json((directory / "candidates.json").read_text(encoding="utf-8"))
    assert GroundingReview.model_validate_json(
        (directory / "grounding-review.json").read_text(encoding="utf-8"))


def test_evidence_appendix_closes_every_exported_claim_and_decision_dependency(prepared):
    # The controlled proposal grounds in work A while naming contribution claims
    # from A and B as closest work, so B's contribution span is not otherwise cited.
    result = create_innovations(Runner(prepared, offline=True, backend=RecordingBackend()))
    assert result["status"] == "complete"
    directory = prepared / result["output"]
    appendix = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    candidates = CandidateBatch.model_validate_json(
        (directory / "candidates.json").read_text(encoding="utf-8"))
    exported_evidence = set(appendix["evidence"])
    for claim in appendix["claims"].values():
        assert set(claim["evidence_ids"]) <= exported_evidence
    for decision in appendix["decisions"].values():
        assert set(decision["evidence_ids"]) <= exported_evidence
    closest = set(candidates.candidates[0].closest_claim_ids)
    assert closest
    assert all(appendix["claims"][claim_id]["category"] == "contribution" for claim_id in closest)
    assert all(appendix["claims"][claim_id]["evidence_ids"] for claim_id in closest)


def test_exchange_response_resume_does_not_repeat_completed_generation(prepared):
    exchange = prepared / "exchange-innovation"
    runner = Runner(prepared, offline=True, backend=ExchangeBackend(exchange))
    pending_generation = create_innovations(runner)
    assert pending_generation["status"] == "needs_input"
    generation_request = next(exchange.glob("*.request.json"))
    request = json.loads(generation_request.read_text(encoding="utf-8"))
    generation_job = type("Job", (), {"payload": request["job"]["payload"]})()
    response = generation_request.with_name(generation_request.name.replace(".request.", ".response."))
    response.write_text(json.dumps({"fingerprint": request["fingerprint"], "data": _generation(generation_job)}),
                        encoding="utf-8")

    pending_review = create_innovations(Runner(prepared, offline=True, backend=ExchangeBackend(exchange)))
    assert pending_review["status"] == "needs_input"
    assert pending_review["candidate_ids"]
    requests = sorted(exchange.glob("*.request.json"))
    assert len(requests) == 2
    review_request = next(path for path in requests if path != generation_request)
    request = json.loads(review_request.read_text(encoding="utf-8"))
    review_job = type("Job", (), {"payload": request["job"]["payload"]})()
    response = review_request.with_name(review_request.name.replace(".request.", ".response."))
    response.write_text(json.dumps({"fingerprint": request["fingerprint"], "data": _review(review_job)}),
                        encoding="utf-8")
    complete = create_innovations(Runner(prepared, offline=True, backend=ExchangeBackend(exchange)))
    assert complete["status"] == "complete"
    assert len(list(exchange.glob("*.request.json"))) == 2


def test_oversize_stale_and_invalid_outputs_never_publish_or_cache(prepared):
    config = load_config(prepared)
    config.max_input_chars = 100
    atomic_json(prepared / "project.json", config.model_dump())
    runner = Runner(prepared, offline=True, backend=RecordingBackend())
    oversized = create_innovations(runner)
    assert oversized["status"] == "failed"
    assert oversized["gaps"][-1] == {
        "reason": "model_input_too_large",
        "guidance": "Choose fewer works with work_ids and retry; selected documents remain complete.",
    }
    assert not runner.backend.backend.jobs
    assert runner.store.budget()["consumed_or_reserved_calls"] == 4

    config.max_input_chars = 100_000
    atomic_json(prepared / "project.json", config.model_dump())
    invalid_backend = RecordingBackend(bad_generation=True)
    invalid = create_innovations(Runner(prepared, offline=True, backend=invalid_backend))
    assert invalid["status"] == "failed"
    assert not any(job["status"] == "succeeded" and job["kind"] == "synthesize"
                   for job in runner.store.jobs())
    assert not (prepared / "analyses" / "innovation" / "latest.json").exists()


def test_review_only_overflow_preserves_generation_cache_and_selection_guidance(prepared):
    snapshot = freeze_research(Runner(prepared, offline=True).store)
    probe = RecordingBackend()
    batch = generate_candidates(snapshot, probe, timeout_seconds=5, max_input_chars=100_000)
    review_candidates(snapshot, batch, probe, timeout_seconds=5, max_input_chars=100_000)
    generation_size = len(job_prompt(probe.jobs[0]))
    assert len(job_prompt(probe.jobs[1])) > generation_size

    config = load_config(prepared)
    config.max_input_chars = generation_size
    atomic_json(prepared / "project.json", config.model_dump())
    first_backend = RecordingBackend()
    first = create_innovations(Runner(prepared, offline=True, backend=first_backend))
    expected_gap = {
        "reason": "model_input_too_large",
        "guidance": "Choose fewer works with work_ids and retry; selected documents remain complete.",
    }
    assert first["status"] == "failed" and first["gaps"][-1] == expected_gap
    assert first["candidate_ids"]
    assert [job.payload["task"] for job in first_backend.jobs] == ["innovation-generation-v1"]

    resumed_backend = RecordingBackend()
    resumed = create_innovations(Runner(prepared, offline=True, backend=resumed_backend))
    assert resumed["status"] == "failed" and resumed["gaps"][-1] == expected_gap
    assert resumed["candidate_ids"] == first["candidate_ids"]
    assert not resumed_backend.jobs
    synthesize = [job for job in Runner(prepared, offline=True).store.jobs() if job["kind"] == "synthesize"]
    assert [job["status"] for job in synthesize] == ["succeeded"]
    assert not (prepared / "analyses" / "innovation" / "latest.json").exists()
    assert "model_artifact_invalid" not in json.dumps(resumed)


def test_invalid_review_keeps_only_domain_valid_generation_cache(prepared):
    def duplicate(job, items):
        return [items[0], items[0]]

    backend = RecordingBackend(review_items=duplicate)
    result = create_innovations(Runner(prepared, offline=True, backend=backend))
    assert result["status"] == "failed"
    synthesize = [job for job in Runner(prepared, offline=True).store.jobs() if job["kind"] == "synthesize"]
    assert [job["status"] for job in synthesize].count("succeeded") == 1
    assert [job["status"] for job in synthesize].count("failed") == 1
    assert not (prepared / "analyses" / "innovation" / "latest.json").exists()


def test_source_change_before_domain_commit_is_stale_and_not_cached(prepared):
    runner = Runner(prepared, offline=True)
    snapshot = freeze_research(runner.store)
    source = prepared / next(iter(snapshot.documents.values())).source_path

    def mutate(job):
        if job.payload["task"] == "innovation-generation-v1":
            source.write_text("changed original source", encoding="utf-8")

    backend = RecordingBackend(mutate=mutate)
    result = create_innovations(Runner(prepared, offline=True, backend=backend))
    assert result["status"] == "stale"
    assert not any(job["status"] == "succeeded" and job["kind"] == "synthesize"
                   for job in runner.store.jobs())
    assert not (prepared / "analyses" / "innovation" / "latest.json").exists()


@pytest.mark.parametrize("fault", ["missing", "duplicate", "foreign", "foreign_claim", "foreign_evidence"])
def test_review_requires_exact_candidate_set_and_known_chain(prepared, fault):
    snapshot = freeze_research(Runner(prepared, offline=True).store)
    generation = RecordingBackend()
    batch = generate_candidates(snapshot, generation, timeout_seconds=5, max_input_chars=100_000)

    def broken(job, items):
        if fault == "missing":
            return []
        if fault == "duplicate":
            return [items[0], items[0]]
        if fault == "foreign":
            items[0]["candidate_id"] = "innovation_foreign"
        elif fault == "foreign_claim":
            items[0]["claim_ids"] = ["claim_foreign"]
        else:
            items[0]["evidence_ids"] = ["evidence_foreign"]
        return items

    with pytest.raises(InnovationError, match="review"):
        review_candidates(snapshot, batch, RecordingBackend(review_items=broken),
                          timeout_seconds=5, max_input_chars=100_000)


def test_independent_contradiction_can_cite_supported_work_outside_closest_subset(prepared):
    snapshot = freeze_research(Runner(prepared, offline=True).store)

    class ClosestSubsetBackend(RecordingBackend):
        def run(self, job):
            if job.payload["task"] == "innovation-generation-v1":
                data = _generation(job)
                candidate = data["candidates"][0]
                closest_work = candidate["closest_work_ids"][0]
                candidate["closest_work_ids"] = [closest_work]
                candidate["closest_claim_ids"] = [claim_id for claim_id in candidate["closest_claim_ids"]
                                                   if any(chain["claim"]["id"] == claim_id
                                                          and chain["claim"]["work_id"] == closest_work
                                                          for work in job.payload["research"]["works"]
                                                          for chain in work["reviewed_claims"])]
            else:
                candidate = job.payload["batch"]["candidates"][0]
                outside = next(
                    chain for work in job.payload["research"]["works"]
                    if work["work"]["id"] not in candidate["closest_work_ids"]
                    for chain in work["reviewed_claims"]
                    if chain["decision"]["verdict"] == "supported"
                )
                data = {"items": [{
                    "candidate_id": candidate["id"],
                    "verdict": "contradicted",
                    "rationale": "另一篇已提供工作的支持证据构成反例。",
                    "claim_ids": [outside["claim"]["id"]],
                    "evidence_ids": outside["decision"]["evidence_ids"],
                }]}
            self.jobs.append(job)
            return ModelResult(status="succeeded", backend=self.name, data=data)

    backend = ClosestSubsetBackend()
    batch = generate_candidates(snapshot, backend, timeout_seconds=5, max_input_chars=100_000)
    assert len(batch.candidates[0].closest_work_ids) == 1
    review = review_candidates(snapshot, batch, backend, timeout_seconds=5, max_input_chars=100_000)
    assert review.items[0].verdict == "contradicted"
    cited_claim = review.items[0].claim_ids[0]
    claims = {claim.id: claim for card in snapshot.cards.values() for claim in card.claims}
    assert claims[cited_claim].work_id not in batch.candidates[0].closest_work_ids

    result = create_innovations(Runner(prepared, offline=True, backend=ClosestSubsetBackend()))
    assert result["status"] == "complete"
    appendix = json.loads((prepared / result["output"] / "evidence.json").read_text(encoding="utf-8"))
    cited_work = claims[cited_claim].work_id
    assert cited_work in appendix["works"]
    assert cited_claim in appendix["claims"] and cited_claim in appendix["decisions"]
    assert set(appendix["decisions"][cited_claim]["evidence_ids"]) <= set(appendix["evidence"])


@pytest.mark.parametrize("verdict", ["supported", "partial", "contradicted", "insufficient"])
def test_grounding_verdicts_remain_visible_without_novelty_upgrade(prepared, verdict):
    snapshot = freeze_research(Runner(prepared, offline=True).store)
    backend = RecordingBackend()
    batch = generate_candidates(snapshot, backend, timeout_seconds=5, max_input_chars=100_000)
    if verdict == "insufficient":
        backend.review_items = lambda job, items: [{**items[0], "verdict": verdict,
                                                     "claim_ids": [], "evidence_ids": []}]
    else:
        backend.review_items = lambda job, items: [{**items[0], "verdict": verdict}]
    review = review_candidates(snapshot, batch, backend, timeout_seconds=5, max_input_chars=100_000)
    assert review.items[0].verdict == verdict
    assert "novelty" not in review.items[0].model_dump()


def test_demo_synthesis_requires_exact_original_pair_and_preserves_old_protocol(prepared):
    snapshot = freeze_research(Runner(prepared, offline=True).store)
    batch = generate_candidates(snapshot, DemoBackend(), timeout_seconds=5, max_input_chars=100_000)
    review = review_candidates(snapshot, batch, DemoBackend(), timeout_seconds=5, max_input_chars=100_000)
    assert len(batch.candidates) == 1
    assert review.items[0].verdict in {"supported", "partial"}

    mixed = copy.deepcopy(snapshot)
    next(iter(mixed.works.values())).title = "A real paper"
    with pytest.raises(InnovationError):
        generate_candidates(mixed, DemoBackend(), timeout_seconds=5, max_input_chars=100_000)
