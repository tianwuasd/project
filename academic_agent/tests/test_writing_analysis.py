import hashlib
import json

import pytest
from pydantic import TypeAdapter, ValidationError
from test_domain_analysis import DomainBackend

from academic_agent.demo import seed_demo
from academic_agent.domain import extract_domain, load_domain_version
from academic_agent.models import ModelResult
from academic_agent.trends import build_trends
from academic_agent.workflow import Runner
from academic_agent.writing_analysis import (
    DRAFT_SCHEMA,
    OUTLINE_SCHEMA,
    REVIEW_SCHEMA,
    SEGMENT_SCHEMA,
    generate_draft,
    generate_outline,
    review_text,
)
from academic_agent.writing_context import writing_context_payload, writing_currentness
from academic_agent.writing_models import (
    Draft,
    Outline,
    OutlineReviewState,
    Reference,
    TrendPin,
    WritingError,
    WritingReview,
    WritingSelection,
    assign_generated_ids,
    assign_segment_ids,
)


def _selection(root):
    work_ids = [work.id for work in sorted(Runner(root, offline=True).store.works(), key=lambda work: work.title)]
    return WritingSelection(
        document_id="doc_" + "7" * 32,
        question="How do the scheduling reports differ?",
        work_ids=work_ids,
        domain_path="builtin:scheduling",
        trend_directory=None,
        candidate_directory=None,
        candidate_id=None,
        assessment_directory=None,
    )


@pytest.fixture
def domain(tmp_path):
    seed_demo(tmp_path)
    Runner(tmp_path, offline=True).run()
    result = extract_domain(Runner(tmp_path, offline=True, backend=DomainBackend()), _selection(tmp_path))
    assert result["status"] == "complete"
    return load_domain_version(tmp_path / result["output"])


def _source(payload, work_id):
    research = payload["context"]["research"]
    card = research["cards"][work_id]
    claim = next(claim for claim in card["claims"] if research["decisions"][claim["id"]]["verdict"] == "supported")
    decision = research["decisions"][claim["id"]]
    return {
        "type": "source",
        "work_id": work_id,
        "document_id": card["document_id"],
        "claim_id": claim["id"],
        "evidence_ids": decision["evidence_ids"],
    }


def _sources(payload, work_id):
    research = payload["context"]["research"]
    card = research["cards"][work_id]
    return [
        {
            "type": "source",
            "work_id": work_id,
            "document_id": card["document_id"],
            "claim_id": claim["id"],
            "evidence_ids": research["decisions"][claim["id"]]["evidence_ids"],
        }
        for claim in card["claims"]
        if research["decisions"][claim["id"]]["verdict"] == "supported"
    ]


class WritingBackend:
    name = "controlled-writing"

    def __init__(self, *, fault=None, review_decision="supported"):
        self.jobs = []
        self.fault = fault
        self.review_decision = review_decision

    def run(self, job):
        self.jobs.append(job)
        task = job.payload["task"]
        work_ids = job.payload["context"]["selection"]["work_ids"]
        sources = [_source(job.payload, work_id) for work_id in work_ids]
        if task == "writing-outline-v1":
            data = {"blocks": [
                {"role": "heading", "level": 1, "units": [{
                    "text": "Bounded scheduling comparison", "kind": "connector", "references": [],
                    "purpose": "Name the bounded comparison.",
                }]},
                {"role": "paragraph", "level": None, "units": [{
                    "text": "Compare the opposite waiting-time results and their workload limits.",
                    "kind": "synthesis", "references": [{"type": "synthesis", "sources": sources}],
                    "purpose": "Plan a cross-paper comparison.",
                }]},
            ]}
        elif task == "writing-draft-v1":
            data = {"blocks": [
                {"role": "heading", "level": 1, "units": [{
                    "text": "Results under different simulated workloads", "kind": "connector", "references": [],
                    "purpose": "Identify the scope of the comparison.",
                }]},
                {"role": "paragraph", "level": None, "units": [{
                    "text": (
                        "Study A reports waiting time falling from 10 to 8 simulation units under its single "
                        "workload, whereas Study B reports an increase from 10 to 11 under bursty arrivals; "
                        "these simulated workloads do not establish effectiveness in real systems."
                    ),
                    "kind": "synthesis", "references": [{"type": "synthesis", "sources": sources}],
                    "purpose": "Combine the result, condition difference, counterexample, and transfer limit.",
                }]},
            ]}
        else:
            data = {"items": [
                {"unit_id": unit_id, "decision": self.review_decision,
                 "reason": "Independently checked against the complete supplied context."}
                for unit_id in job.payload["target_ids"]
            ]}
        if self.fault == "extra_text" and task == "writing-outline-v1":
            data["blocks"][1]["text"] = "Unreviewed paragraph text"
        elif self.fault == "unknown_evidence" and task == "writing-outline-v1":
            data["blocks"][1]["units"][0]["references"][0]["sources"][0]["evidence_ids"] = ["missing"]
        elif self.fault == "mismatched_chain" and task == "writing-outline-v1":
            data["blocks"][1]["units"][0]["references"][0]["sources"][0]["document_id"] = sources[1]["document_id"]
        elif self.fault == "unknown_claim" and task == "writing-outline-v1":
            data["blocks"][1]["units"][0]["references"][0]["sources"][0]["claim_id"] = "missing"
        elif self.fault == "foreign_hypothesis" and task == "writing-outline-v1":
            data["blocks"][1]["units"][0].update({
                "kind": "hypothesis",
                "references": [{
                    "type": "hypothesis",
                    "candidate_id": "innovation_" + "0" * 32,
                    "candidate_semantic_hash": "0" * 64,
                }],
            })
        elif self.fault == "card_list" and task == "writing-draft-v1":
            data["blocks"][1]["units"] = [
                {"text": "One source card.", "kind": "source", "references": [source], "purpose": None}
                for source in sources
            ]
        elif self.fault == "missing_review" and task == "writing-review-v1":
            data["items"].pop()
        elif self.fault == "unknown_citekey" and task == "writing-draft-v1":
            data["blocks"][1]["units"][0]["text"] += " [@not_in_catalog]"
        elif self.fault in {"wrong_unit_citekey", "cross_work_source", "same_work_sources"} \
                and task == "writing-outline-v1":
            first_work, second_work = work_ids
            first_sources = _sources(job.payload, first_work)
            citekeys = {
                work_id: job.payload["context"]["research"]["works"][work_id]["citekey"]
                for work_id in work_ids
            }
            unit = data["blocks"][1]["units"][0]
            unit.update({"kind": "source", "references": first_sources[:2]})
            if self.fault == "wrong_unit_citekey":
                unit["text"] = f"This is visibly attributed to the other selected paper [@{citekeys[second_work]}]."
            elif self.fault == "cross_work_source":
                unit["text"] = "This source-labelled unit combines two papers."
                unit["references"] = [first_sources[0], _source(job.payload, second_work)]
            else:
                unit["text"] = f"Two supported claims come from one paper [@{citekeys[first_work]}]."
        return ModelResult(
            status="succeeded", backend=self.name, data=data,
            usage={"mode": "controlled_test", "model_calls": 1},
        )


def _outline(domain, backend=None):
    return generate_outline(
        domain.context, domain, backend or WritingBackend(), timeout_seconds=30, max_input_chars=100_000
    )


def _review(domain, outline, backend=None):
    return review_text(
        domain.context, domain, outline, Draft(blocks=[]),
        [unit.id for block in outline.blocks for unit in block.units],
        backend or WritingBackend(), timeout_seconds=30, max_input_chars=100_000,
    )


def test_generated_ids_follow_exact_canonical_algorithm_and_segmentation_preserves_unicode():
    wire = [{"role": "paragraph", "level": None, "units": [{
        "text": "A", "kind": "connector", "references": [], "purpose": "Join",
    }]}]
    canonical = lambda value: json.dumps(  # noqa: E731
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    document_id, input_hash = "doc_" + "1" * 32, "2" * 64
    seed = hashlib.sha256(canonical(["writing-ids-1", document_id, "writing-outline-v1", input_hash, wire])).hexdigest()
    blocks = assign_generated_ids(document_id, "writing-outline-v1", input_hash, wire)
    assert blocks[0].id == "b_" + hashlib.sha256(canonical([seed, 0])).hexdigest()[:32]
    assert blocks[0].units[0].id == "u_" + hashlib.sha256(canonical([seed, 0, 0])).hexdigest()[:32]
    with pytest.raises(ValueError):
        assign_generated_ids(
            document_id, "writing-outline-v1", input_hash, [{**wire[0], "text": "outside units"}]
        )

    raw = "A🙂界"
    spans = [
        {"start": 0, "end": 2, "kind": "connector", "references": [], "purpose": "Join"},
        {"start": 2, "end": 3, "kind": "connector", "references": [], "purpose": "Close"},
    ]
    units = assign_segment_ids(blocks[0].id, "3" * 64, spans, raw)
    assert [unit.text for unit in units] == ["A🙂", "界"]
    with pytest.raises(ValueError):
        assign_segment_ids(blocks[0].id, "3" * 64, [{**spans[0], "end": 1}], raw)
    with pytest.raises(ValueError):
        assign_segment_ids(blocks[0].id, "3" * 64, [{**spans[0], "start": False}], raw)


def test_outline_and_draft_payloads_are_exact_and_review_state_changes_identity(domain):
    outline_backend = WritingBackend()
    outline = _outline(domain, outline_backend)
    outline_job = outline_backend.jobs[0]
    assert set(outline_job.payload) == {"task", "context", "domain"}
    assert outline_job.payload["task"] == "writing-outline-v1"
    assert all(unit.id.startswith("u_") for block in outline.blocks for unit in block.units)

    review = _review(domain, outline)
    complete = OutlineReviewState(status="complete", review=review)
    first, second = WritingBackend(), WritingBackend()
    draft = generate_draft(
        domain.context, domain, outline, complete, first,
        timeout_seconds=30, max_input_chars=100_000,
    )
    changed = complete.model_copy(deep=True)
    changed.review.items[0].decision = "contradicted"
    changed.review.items[0].reason = "The heading overstates the source scope."
    generate_draft(
        domain.context, domain, outline, changed, second,
        timeout_seconds=30, max_input_chars=100_000,
    )
    assert set(first.jobs[0].payload) == {"task", "context", "domain", "outline", "outline_review_state"}
    assert first.jobs[0].payload["outline_review_state"] == complete.model_dump(mode="json")
    assert second.jobs[0].payload["outline_review_state"]["review"]["items"][0]["decision"] == "contradicted"
    assert first.jobs[0].id != second.jobs[0].id
    body = "".join(unit.text for block in draft.blocks for unit in block.units)
    assert all(fragment in body for fragment in ("10 to 8", "10 to 11", "bursty", "real systems"))


def test_review_receives_complete_ordered_artifact_and_covers_headings_connectors_and_body(domain):
    outline = _outline(domain)
    outline_review = _review(domain, outline)
    draft = generate_draft(
        domain.context, domain, outline,
        OutlineReviewState(status="complete", review=outline_review), WritingBackend(),
        timeout_seconds=30, max_input_chars=100_000,
    )
    targets = [unit.id for block in [*outline.blocks, *draft.blocks] for unit in block.units]
    backend = WritingBackend()
    review = review_text(
        domain.context, domain, outline, draft, targets, backend,
        timeout_seconds=30, max_input_chars=100_000,
    )
    assert set(backend.jobs[0].payload) == {"task", "context", "domain", "outline", "draft", "target_ids"}
    assert backend.jobs[0].payload["target_ids"] == targets
    assert [item.unit_id for item in review.items] == targets
    assert len(targets) == 4


@pytest.mark.parametrize("fault", [
    "extra_text", "unknown_evidence", "mismatched_chain", "unknown_claim", "foreign_hypothesis",
])
def test_outline_rejects_unreviewed_or_foreign_material(domain, fault):
    with pytest.raises(WritingError):
        _outline(domain, WritingBackend(fault=fault))


@pytest.mark.parametrize("fault", ["card_list", "unknown_citekey"])
def test_draft_rejects_card_lists_and_unknown_bibtex_keys(domain, fault):
    outline = _outline(domain)
    state = OutlineReviewState(status="complete", review=_review(domain, outline))
    with pytest.raises(WritingError):
        generate_draft(
            domain.context, domain, outline, state, WritingBackend(fault=fault),
            timeout_seconds=30, max_input_chars=100_000,
        )


@pytest.mark.parametrize("fault, location, error_type", [
    ("connector_purpose", "blocks.0.units.0", "value_error"),
    ("paragraph_level", "blocks.1", "value_error"),
    ("reference_kind", "blocks.1.units.0", "value_error"),
    ("extra_field", "blocks.1.<field>", "extra_forbidden"),
])
def test_draft_schema_failure_locates_rejected_output_without_echoing_it(
        domain, fault, location, error_type):
    marker = "PRIVATE_MODEL_TEXT_DO_NOT_LOG"

    class InvalidDraftBackend(WritingBackend):
        def run(self, job):
            result = super().run(job)
            if job.payload["task"] == "writing-draft-v1":
                blocks = result.data["blocks"]
                blocks[1]["units"][0]["text"] = marker
                if fault == "connector_purpose":
                    blocks[0]["units"][0]["purpose"] = None
                elif fault == "paragraph_level":
                    blocks[1]["level"] = 1
                elif fault == "reference_kind":
                    blocks[1]["units"][0]["kind"] = "source"
                else:
                    blocks[1][marker] = marker
            return result

    outline = _outline(domain)
    state = OutlineReviewState(status="complete", review=_review(domain, outline))
    backend = InvalidDraftBackend()
    with pytest.raises(WritingError) as caught:
        generate_draft(
            domain.context, domain, outline, state, backend,
            timeout_seconds=30, max_input_chars=100_000,
        )
    assert caught.value.status == "failed"
    assert location in str(caught.value) and error_type in str(caught.value)
    assert marker not in str(caught.value)
    assert len(backend.jobs) == 1


def test_unit_citations_and_source_kind_close_over_the_units_own_work_chains(domain):
    for fault in ("wrong_unit_citekey", "cross_work_source"):
        with pytest.raises(WritingError):
            _outline(domain, WritingBackend(fault=fault))
    assert _outline(domain, WritingBackend(fault="same_work_sources")).blocks

    class CitedSynthesisBackend(WritingBackend):
        def run(self, job):
            response = super().run(job)
            if job.payload["task"] == "writing-outline-v1":
                citekeys = [
                    job.payload["context"]["research"]["works"][work_id]["citekey"]
                    for work_id in job.payload["context"]["selection"]["work_ids"]
                ]
                response.data["blocks"][1]["units"][0]["text"] = (
                    f"The selected papers differ in bounded conditions [@{citekeys[0]}; @{citekeys[1]}]."
                )
            return response

    assert _outline(domain, CitedSynthesisBackend()).blocks


def test_pending_outline_and_incomplete_review_fail_before_dispatch(domain):
    outline = _outline(domain)
    backend = WritingBackend()
    with pytest.raises(WritingError) as caught:
        generate_draft(
            domain.context, domain, outline,
            OutlineReviewState(status="pending", review=WritingReview(items=[])), backend,
            timeout_seconds=30, max_input_chars=100_000,
        )
    assert caught.value.status == "needs_input" and backend.jobs == []
    targets = [unit.id for block in outline.blocks for unit in block.units]
    with pytest.raises(WritingError):
        review_text(
            domain.context, domain, outline, Draft(blocks=[]), targets,
            WritingBackend(fault="missing_review"), timeout_seconds=30, max_input_chars=100_000,
        )


def test_input_limit_is_checked_before_dispatch_without_truncating(domain):
    backend = WritingBackend()
    with pytest.raises(WritingError) as caught:
        generate_outline(domain.context, domain, backend, timeout_seconds=30, max_input_chars=100)
    assert caught.value.status == "input_too_large" and backend.jobs == []


def test_frozen_statistic_requires_exact_host_pin_and_catalog_path(tmp_path):
    seed_demo(tmp_path)
    Runner(tmp_path, offline=True).run()
    trend = build_trends(Runner(tmp_path, offline=True).store)
    selection = _selection(tmp_path).model_copy(update={"trend_directory": str(trend)})
    result = extract_domain(Runner(tmp_path, offline=True, backend=DomainBackend()), selection)
    domain = load_domain_version(tmp_path / result["output"])
    assert domain.context.trend is not None
    serialized_trend = domain.context.trend.model_dump(mode="json")
    assert TrendPin.model_validate(serialized_trend, strict=True) == domain.context.trend
    assert writing_currentness(tmp_path, domain.context) == {"current": True, "gaps": []}
    invalid_snapshots = [
        [1, 2, 3],
        [True, 2, 3, "1" * 64],
        [1, "two", 3, "1" * 64],
        [1, 2, 3, "not-a-valid-hash"],
    ]
    for source_snapshot in invalid_snapshots:
        changed = {
            **serialized_trend,
            "snapshot": {**serialized_trend["snapshot"], "source_snapshot": source_snapshot},
        }
        with pytest.raises(ValidationError):
            TrendPin.model_validate(changed, strict=True)

    class StatisticBackend(WritingBackend):
        def __init__(self, change=None):
            super().__init__()
            self.change = change

        def run(self, job):
            response = super().run(job)
            unit = response.data["blocks"][1]["units"][0]
            unit.update({
                "text": "The frozen eligible count is reported with its recorded computation identity.",
                "kind": "frozen_statistic",
                "references": [{
                    "type": "frozen_statistic",
                    "pin_hash": domain.context.trend.artifact.semantic_hash,
                    "computation_path": "counts.eligible",
                }],
                "purpose": "Report a frozen corpus statistic.",
            })
            if self.change == "pin":
                unit["references"][0]["pin_hash"] = "0" * 64
            elif self.change == "path":
                unit["references"][0]["computation_path"] = "counts.not_in_catalog"
            return response

    assert _outline(domain, StatisticBackend()).blocks
    for change in ("pin", "path"):
        with pytest.raises(WritingError):
            _outline(domain, StatisticBackend(change))


def test_independent_review_receives_seven_actual_wrong_assertions_with_full_context(domain):
    payload = {"context": writing_context_payload(domain.context)}
    work_ids = domain.context.selection.work_ids
    sources = [_source(payload, work_id) for work_id in work_ids]
    synthesis = {"type": "synthesis", "sources": sources}
    wrong = [
        "Study A reports waiting time falling from 10 to 7 simulation units.",
        "Study B reports that waiting time did not increase.",
        "Study B's result applies without its bursty-arrival condition.",
        "Priority scheduling improves every workload.",
        "Study A enrolled 40 human participants.",
        "The proposed scheduling hypothesis was observed to succeed in deployment.",
        "Therefore the studies establish real-world effectiveness.",
    ]
    outline_wire = [
        {"role": "heading", "level": 1, "units": [{
            "text": wrong[0], "kind": "source", "references": [sources[0]], "purpose": "Wrong number.",
        }]},
        {"role": "paragraph", "level": None, "units": [{
            "text": wrong[1], "kind": "source", "references": [sources[1]], "purpose": "Omitted negation.",
        }, {
            "text": wrong[2], "kind": "source", "references": [sources[1]], "purpose": "Concealed condition.",
        }]},
        {"role": "heading", "level": 2, "units": [{
            "text": wrong[3], "kind": "synthesis", "references": [synthesis], "purpose": "Universalized heading.",
        }]},
    ]
    draft_wire = [
        {"role": "heading", "level": 1, "units": [{
            "text": "Unsupported interpretation", "kind": "connector", "references": [],
            "purpose": "Name the unsupported section.",
        }]},
        {"role": "paragraph", "level": None, "units": [{
            "text": wrong[4], "kind": "source", "references": [sources[0]], "purpose": "Simulation as participants.",
        }, {
            "text": wrong[5], "kind": "synthesis", "references": [synthesis],
            "purpose": "Hypothesis as observed result.",
        }, {
            "text": wrong[6], "kind": "connector", "references": [synthesis],
            "purpose": "False connector conclusion.",
        }]},
    ]
    outline = Outline(blocks=assign_generated_ids(
        domain.context.selection.document_id, "controlled-negative-outline", "4" * 64, outline_wire
    ))
    draft = Draft(blocks=assign_generated_ids(
        domain.context.selection.document_id, "controlled-negative-draft", "5" * 64, draft_wire
    ))
    targets = [unit.id for block in [*outline.blocks, *draft.blocks] for unit in block.units]

    class ControlledNegativeReviewBackend:
        name = "controlled-negative-review"

        def __init__(self):
            self.jobs = []

        def run(self, job):
            self.jobs.append(job)
            return ModelResult(
                status="succeeded",
                backend=self.name,
                data={"items": [
                    {
                        "unit_id": unit_id,
                        "decision": "contradicted",
                        "reason": "Controlled non-supported protocol decision; this does not measure accuracy.",
                    }
                    for unit_id in job.payload["target_ids"]
                ]},
                usage={"mode": "controlled_test", "model_calls": 1},
            )

    backend = ControlledNegativeReviewBackend()
    review = review_text(
        domain.context, domain, outline, draft, targets, backend,
        timeout_seconds=30, max_input_chars=100_000,
    )
    job = backend.jobs[0]
    delivered_units = [
        unit
        for block in [*job.payload["outline"]["blocks"], *job.payload["draft"]["blocks"]]
        for unit in block["units"]
    ]
    delivered_wrong = [unit for unit in delivered_units if unit["text"] in wrong]
    expected = {
        unit.text: [reference.model_dump(mode="json") for reference in unit.references]
        for block in [*outline.blocks, *draft.blocks]
        for unit in block.units
        if unit.text in wrong
    }
    assert job.payload["outline"] == outline.model_dump(mode="json")
    assert job.payload["draft"] == draft.model_dump(mode="json")
    assert job.payload["target_ids"] == targets
    assert [unit["text"] for unit in delivered_wrong] == wrong
    assert {unit["text"]: unit["references"] for unit in delivered_wrong} == expected
    assert [item.unit_id for item in review.items] == targets
    assert {item.decision for item in review.items} == {"contradicted"}


@pytest.mark.parametrize("schema", [OUTLINE_SCHEMA, DRAFT_SCHEMA, REVIEW_SCHEMA, SEGMENT_SCHEMA],
                         ids=["outline", "draft", "review", "segment"])
def test_writing_schemas_fit_documented_structured_output_subset(schema):
    # External contract: https://developers.openai.com/api/docs/guides/structured-outputs
    # This checks the subset used here; it is not a general provider validator.
    def check(node):
        assert "oneOf" not in node, "Structured Outputs supports anyOf unions"
        if "anyOf" in node:
            for branch in node["anyOf"]:
                check(branch)
            return
        assert node.get("type") in {"object", "array", "string", "integer", "number", "boolean", "null"}
        if node["type"] == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for child in node["properties"].values():
                check(child)
        elif node["type"] == "array":
            check(node["items"])

    assert schema["type"] == "object"
    check(schema)


def test_supported_reference_union_remains_disjoint_and_host_validation_stays_strict():
    reference_schema = DRAFT_SCHEMA["properties"]["blocks"]["items"]["properties"]["units"]["items"][
        "properties"]["references"]["items"]
    branches = reference_schema["anyOf"]
    tags = [branch["properties"]["type"]["enum"] for branch in branches]
    assert all(len(tag) == 1 for tag in tags)
    assert {tag[0] for tag in tags} == {"source", "synthesis", "frozen_statistic", "hypothesis"}
    assert len(tags) == 4
    assert all(branch["properties"]["type"]["type"] == "string" for branch in branches)

    source = {"type": "source", "work_id": "a", "document_id": "doc_a",
              "claim_id": "claim_a", "evidence_ids": ["evidence_a"]}
    references = [
        source,
        {"type": "synthesis", "sources": [source, {**source, "work_id": "b"}]},
        {"type": "frozen_statistic", "pin_hash": "a" * 64, "computation_path": "counts.eligible"},
        {"type": "hypothesis", "candidate_id": "innovation_" + "a" * 32,
         "candidate_semantic_hash": "b" * 64},
    ]
    adapter = TypeAdapter(Reference)
    for reference in references:
        assert adapter.validate_python(reference, strict=True).model_dump(mode="json") == reference
        # A tagged reference cannot silently accept fields from another branch.
        with pytest.raises(ValidationError):
            adapter.validate_python({**reference, "unexpected_reference_field": "x"}, strict=True)
    with pytest.raises(ValidationError):
        adapter.validate_python({**source, "type": "synthesis"}, strict=True)
    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "synthesis", "sources": [source, source]}, strict=True)
