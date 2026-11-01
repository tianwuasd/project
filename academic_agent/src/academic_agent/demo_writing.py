"""Exact writing protocol for the bundled original scheduling fixtures."""

from __future__ import annotations

import hashlib
import re
from importlib import resources
from typing import Any

from pydantic import ValidationError

from .demo import FIXTURES
from .documents import verify_span
from .models import Claim, Document, Evidence, ModelJob, ModelResult, ReadingCard, Work, digest
from .research_context import ResearchDecision
from .writing_models import DomainPack

TASKS = {
    "domain-extract-v1",
    "domain-review-v1",
    "writing-outline-v1",
    "writing-draft-v1",
    "writing-segment-v1",
    "writing-review-v1",
}


def _failure(message: str) -> ModelResult:
    return ModelResult(status="failed", backend="demo", error=message)


def _builtin_pack() -> tuple[bytes, dict[str, Any]]:
    raw = resources.files("academic_agent.domain_packs").joinpath("scheduling.json").read_bytes()
    pack = DomainPack.model_validate_json(raw, strict=True)
    return raw, pack.model_dump(mode="json")


def _pack_is_exact(job: ModelJob) -> bool:
    raw, expected = _builtin_pack()
    expected_hash = hashlib.sha256(raw).hexdigest()
    task = job.payload.get("task")
    if task in {"domain-extract-v1", "domain-review-v1"}:
        research = job.payload.get("research", {})
        identity = research.get("domain_pack_identity", {}) if isinstance(research, dict) else {}
        return (
            job.payload.get("domain") == expected
            and identity == {"source_path": "builtin:scheduling", "sha256": expected_hash}
        )
    context = job.payload.get("context", {})
    pin = context.get("pack", {}) if isinstance(context, dict) else {}
    return (
        pin.get("source_path") == "builtin:scheduling"
        and pin.get("sha256") == expected_hash
        and pin.get("pack") == expected
        and job.payload.get("domain", {}).get("context", {}).get("pack") == pin
    )


def _metadata_is_exact(metadata: dict[str, Any], title: str) -> bool:
    expected_source = f"study-{1 if title.endswith('A') else 2}"
    return (
        metadata.get("authors") == ["Academic Agent original fixture"]
        and metadata.get("year") == 2026
        and metadata.get("source_ids") == {"synthetic": expected_source}
        and metadata.get("screening") == "include"
        and metadata.get("screening_reason") == "Explicitly included by the synthetic demo command"
        and metadata.get("abstract") == ""
        and metadata.get("doi") is None
        and metadata.get("arxiv_id") is None
        and metadata.get("urls") == []
        and metadata.get("open_access_url") is None
        and metadata.get("venue") == ""
        and metadata.get("provenance") == []
        and metadata.get("relations") == []
        and metadata.get("citekey") == "ref_" + str(metadata.get("id", "")).removeprefix("work_")
        and [
            {key: item.get(key) for key in ("source", "status", "detail")}
            for item in metadata.get("publication_checks", [])
        ] == [
            {"source": "synthetic", "status": "unknown", "detail": "Not a real publication"},
            {"source": "offline", "status": "unknown", "detail": "pre_export: network check skipped explicitly"},
        ]
    )


def _expected_claims(title: str) -> list[tuple[str, str]]:
    text = dict(FIXTURES)[title]
    return [
        (match.group(1), match.group(2).strip())
        for match in re.finditer(
            r"(?m)^(problem|method|data|result|limitation|contribution):\s*(\S[^\n]*)$",
            text,
        )
    ]


def _validate_fixture_chain(
    title: str,
    document: Document,
    evidence: list[Evidence],
    claims: list[Claim],
    reviews: list[ResearchDecision],
    *,
    card: ReadingCard | None = None,
) -> tuple[dict[str, Evidence], dict[str, Claim], dict[str, ResearchDecision]]:
    fixture_text = dict(FIXTURES)[title]
    if (
        document.level != "full_text"
        or document.parser != "utf8-text"
        or document.diagnostics != []
        or len(document.pages) != 1
        or document.pages[0].number != 1
        or document.pages[0].text != fixture_text
        or not document.source_path
        or not document.source_path.casefold().endswith(".txt")
    ):
        raise ValueError("Demo requires the canonical original fixture document")
    if (
        len(evidence) != 1
        or evidence[0].page != 1
        or evidence[0].start != 0
        or evidence[0].end != len(fixture_text)
        or evidence[0].text != fixture_text
        or not verify_span(evidence[0], document)
    ):
        raise ValueError("Demo requires the canonical complete fixture evidence span")
    if [(claim.category, claim.text) for claim in claims] != _expected_claims(title):
        raise ValueError("Demo requires the canonical fixture claims")
    if card is None:
        card = ReadingCard(
            work_id=document.work_id,
            claims=claims,
            missing=[],
            backend="demo",
            document_id=document.id,
        )
    if (
        card.work_id != document.work_id
        or card.document_id != document.id
        or card.backend != "demo"
        or card.missing != []
        or card.claims != claims
    ):
        raise ValueError("Demo requires the canonical complete reading card")
    evidence_by_id = {item.id: item for item in evidence}
    review_by_claim = {item.claim_id: item for item in reviews}
    if (
        len(evidence_by_id) != len(evidence)
        or len(review_by_claim) != len(reviews)
        or set(review_by_claim) != {item.id for item in claims}
    ):
        raise ValueError("Demo requires one effective review for every canonical claim")
    for claim in claims:
        decision = review_by_claim[claim.id]
        if (
            claim.work_id != document.work_id
            or claim.evidence_ids != [evidence[0].id]
            or decision.work_id != document.work_id
            or decision.document_id != document.id
            or decision.card_hash != digest(card.model_dump_json())
            or decision.verdict != "supported"
            or decision.rationale
            != "Demo: exact labelled fixture sentence; no independent AI evaluation"
            or decision.reviewer != "demo"
            or decision.evidence_ids != claim.evidence_ids
            or decision.type != "model"
        ):
            raise ValueError("Demo fixture evidence and review chain is not canonical")
    return evidence_by_id, {item.category: item for item in claims}, review_by_claim


def _validated_domain_fixture(job: ModelJob) -> dict[str, Any]:
    material = job.payload.get("research")
    if not isinstance(material, dict) or not isinstance(material.get("work"), dict):
        raise ValueError("Demo domain input is malformed")
    title = material["work"].get("title")
    expected = dict(FIXTURES)
    if title not in expected or not _metadata_is_exact(material["work"], title):
        raise ValueError("Demo requires exact original fixture metadata")
    document = Document.model_validate(material.get("document"))
    if (
        material["work"].get("id") != document.work_id
        or "".join(page.text for page in document.pages) != expected[title]
    ):
        raise ValueError("Demo requires exact original fixture bytes")
    evidence = [Evidence.model_validate(item) for item in material.get("evidence", [])]
    claims = [Claim.model_validate(item) for item in material.get("claims", [])]
    reviews = [ResearchDecision.model_validate(item) for item in material.get("effective_reviews", [])]
    evidence_by_id, claims_by_category, review_by_claim = _validate_fixture_chain(
        title, document, evidence, claims, reviews
    )
    return {
        "title": title,
        "work": material["work"],
        "document": document,
        "evidence": evidence_by_id,
        "claims": claims_by_category,
        "reviews": review_by_claim,
    }


def _validated_writing_fixtures(job: ModelJob) -> dict[str, dict[str, Any]]:
    context = job.payload.get("context")
    if not isinstance(context, dict) or not isinstance(context.get("research"), dict):
        raise ValueError("Demo writing context is malformed")
    research = context["research"]
    works = research.get("works")
    documents = research.get("documents")
    evidence = research.get("evidence")
    cards = research.get("cards")
    decisions = research.get("decisions")
    if not all(isinstance(value, dict) for value in (works, documents, evidence, cards, decisions)):
        raise ValueError("Demo writing context lacks complete source maps")
    parsed_works = {key: Work.model_validate(value) for key, value in works.items()}
    parsed_documents = {key: Document.model_validate(value) for key, value in documents.items()}
    parsed_evidence = {key: Evidence.model_validate(value) for key, value in evidence.items()}
    parsed_cards = {key: ReadingCard.model_validate(value) for key, value in cards.items()}
    parsed_decisions = {
        key: ResearchDecision.model_validate(value) for key, value in decisions.items()
    }
    if (
        any(key != value.id for key, value in parsed_works.items())
        or any(key != value.id for key, value in parsed_documents.items())
        or any(key != value.id for key, value in parsed_evidence.items())
        or any(key != value.work_id for key, value in parsed_cards.items())
        or any(key != value.claim_id for key, value in parsed_decisions.items())
    ):
        raise ValueError("Demo writing source map identity changed")
    expected = dict(FIXTURES)
    if (
        len(parsed_works) != 2
        or len(parsed_documents) != 2
        or set(parsed_cards) != set(parsed_works)
        or {item.title for item in parsed_works.values()} != set(expected)
    ):
        raise ValueError("Demo writing requires the exact two original fixtures")
    result: dict[str, dict[str, Any]] = {}
    consumed_evidence: set[str] = set()
    consumed_decisions: set[str] = set()
    for work_id, work in parsed_works.items():
        title = work.title
        if title not in expected or not _metadata_is_exact(work.model_dump(mode="json"), title):
            raise ValueError("Demo writing fixture metadata changed")
        document_values = [
            value for value in parsed_documents.values() if value.work_id == work_id
        ]
        if len(document_values) != 1:
            raise ValueError("Demo writing fixture document identity changed")
        document = document_values[0]
        if "".join(page.text for page in document.pages) != expected[title]:
            raise ValueError("Demo writing fixture text changed")
        card = parsed_cards[work_id]
        work_evidence = [
            value for value in parsed_evidence.values() if value.work_id == work_id
        ]
        consumed_evidence.update(item.id for item in work_evidence)
        work_reviews = [
            value
            for claim in card.claims
            if (value := parsed_decisions.get(claim.id)) is not None
        ]
        consumed_decisions.update(item.claim_id for item in work_reviews)
        _, claims_by_category, _ = _validate_fixture_chain(
            title, document, work_evidence, list(card.claims), work_reviews, card=card
        )
        result[title] = {
            "work_id": work_id,
            "document": document,
            "claims": claims_by_category,
        }
    if consumed_evidence != set(parsed_evidence) or consumed_decisions != set(parsed_decisions):
        raise ValueError("Demo writing source maps contain unconsumed entries")
    return result


def _evidence_for(entry: dict[str, Any], category: str) -> list[str]:
    claim = entry["claims"][category]
    return list(claim.evidence_ids)


def _domain_extract(job: ModelJob, fixture: dict[str, Any]) -> dict[str, Any]:
    title = fixture["title"]
    is_a = title.endswith("A")
    evidence = {
        "data": _evidence_for(fixture, "data"),
        "method": _evidence_for(fixture, "method"),
        "result": _evidence_for(fixture, "result"),
    }
    values = {
        "simulated_jobs": (40 if is_a else 60, evidence["data"]),
        "arrival_pattern": ((None, []) if is_a else ("bursty", evidence["method"])),
        "waiting_time": (
            ("10 to 8 simulation units", evidence["result"])
            if is_a
            else ("10 to 11 simulation units", evidence["result"])
        ),
        "baseline": (("round-robin", evidence["method"]) if is_a else (None, [])),
    }
    field_ids = [item["id"] for item in job.payload["domain"]["fields"]]
    if field_ids != ["simulated_jobs", "arrival_pattern", "waiting_time", "baseline"]:
        raise ValueError("Demo domain fields do not match the builtin protocol")
    return {"fields": [
        {
            "field_id": field_id,
            "status": "reported" if values[field_id][0] is not None else "not_reported",
            "value": values[field_id][0],
            "evidence_ids": values[field_id][1],
        }
        for field_id in field_ids
    ]}


def _domain_review(job: ModelJob, fixture: dict[str, Any]) -> dict[str, Any]:
    expected = _domain_extract(job.model_copy(update={"payload": {
        **job.payload,
        "task": "domain-extract-v1",
    }}), fixture)["fields"]
    if job.payload.get("fields") != expected:
        raise ValueError("Demo domain review requires the exact extracted fields")
    return {"items": [
        {
            "field_id": field["field_id"],
            "decision": "supported",
            "reason": (
                "The complete original fixture explicitly reports this value and its same-work evidence."
                if field["status"] == "reported"
                else "The complete original fixture does not report this field; no value or evidence was invented."
            ),
        }
        for field in expected
    ]}


def _source(fixtures: dict[str, dict[str, Any]], context: dict[str, Any], title: str,
            category: str) -> dict[str, Any]:
    entry = fixtures[title]
    claim_id = entry["claims"][category].id
    sources = context["reference_catalog"]["sources"]
    matches = [item for item in sources if item["work_id"] == entry["work_id"]
               and item["claim_id"] == claim_id]
    if len(matches) != 1:
        raise ValueError("Demo writing reference catalog is incomplete")
    return matches[0]


def _synthesis(fixtures: dict[str, dict[str, Any]], context: dict[str, Any],
               category: str) -> dict[str, Any]:
    return {
        "type": "synthesis",
        "sources": [_source(fixtures, context, title, category) for title, _ in FIXTURES],
    }


def _outline(fixtures: dict[str, dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    result_ref = _synthesis(fixtures, context, "result")
    limitation_ref = _synthesis(fixtures, context, "limitation")
    return {"blocks": [
        {"role": "heading", "level": 1, "units": [{
            "text": "Workload-bounded scheduling evidence",
            "kind": "connector",
            "references": [],
            "purpose": "Name the scope of the reviewed outline.",
        }]},
        {"role": "paragraph", "level": None, "units": [{
            "text": "Compare Study A's 40-job 10-to-8 result with Study B's 60-job 10-to-11 counterexample.",
            "kind": "synthesis",
            "references": [result_ref],
            "purpose": "Plan a conditional cross-paper comparison.",
        }]},
        {"role": "heading", "level": 2, "units": [{
            "text": "Simulation and transfer limits",
            "kind": "connector",
            "references": [],
            "purpose": "Keep simulation scope visible in the structure.",
        }]},
        {"role": "paragraph", "level": None, "units": [{
            "text": "Retain the single-workload and real-system transfer limits instead of claiming general accuracy.",
            "kind": "synthesis",
            "references": [limitation_ref],
            "purpose": "Plan the limitations argument.",
        }]},
    ]}


def _draft(fixtures: dict[str, dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
    result_ref = _synthesis(fixtures, context, "result")
    limitation_ref = _synthesis(fixtures, context, "limitation")
    return {"blocks": [
        {"role": "heading", "level": 1, "units": [{
            "text": "Conditional results in two synthetic workloads",
            "kind": "connector",
            "references": [],
            "purpose": "State that the comparison is conditional and synthetic.",
        }]},
        {"role": "paragraph", "level": None, "units": [{
            "text": (
                "Study A reports mean waiting time falling from 10 to 8 simulation units for 40 simulated jobs, "
                "whereas Study B reports an increase from 10 to 11 for 60 simulated jobs under bursty arrivals."
            ),
            "kind": "synthesis",
            "references": [result_ref],
            "purpose": "Synthesize the opposed results with their workload conditions.",
        }]},
        {"role": "heading", "level": 2, "units": [{
            "text": "Limits and a proposed next test",
            "kind": "connector",
            "references": [],
            "purpose": "Separate source findings from a future research proposal.",
        }]},
        {"role": "paragraph", "level": None, "units": [{
            "text": (
                "These fixed-corpus simulations do not establish effectiveness in real systems or general model "
                "accuracy; they motivate a future, preregistered comparison across held-out synthetic workloads."
            ),
            "kind": "synthesis",
            "references": [limitation_ref],
            "purpose": "Preserve transfer limits and label the next test as proposed.",
        }]},
    ]}


_GENERATED_HEADINGS = {
    "Workload-bounded scheduling evidence",
    "Simulation and transfer limits",
    "Conditional results in two synthetic workloads",
    "Limits and a proposed next test",
}
_GENERATED_PARAGRAPHS = {
    "Compare Study A's 40-job 10-to-8 result with Study B's 60-job 10-to-11 counterexample.": "result",
    "Retain the single-workload and real-system transfer limits instead of claiming general accuracy.": "limitation",
    (
        "Study A reports mean waiting time falling from 10 to 8 simulation units for 40 simulated jobs, "
        "whereas Study B reports an increase from 10 to 11 for 60 simulated jobs under bursty arrivals."
    ): "result",
    (
        "These fixed-corpus simulations do not establish effectiveness in real systems or general model "
        "accuracy; they motivate a future, preregistered comparison across held-out synthetic workloads."
    ): "limitation",
}
_KNOWN_ERROR_HEADING = "All schedulers always improve"
_KNOWN_ERROR_PREFIX = "Study A used 400 simulated participants.\r\n"
_KNOWN_ERROR_CONNECTOR = "因此两种策略总会改进。🙂 e\u0301"
_KNOWN_ERROR_PARAGRAPH = _KNOWN_ERROR_PREFIX + _KNOWN_ERROR_CONNECTOR
_KNOWN_NEUTRAL_HEADINGS = {
    "Scheduling evidence scope",
    "条件限定的调度证据",
    "反例与局限",
}


def _neutral_heading(text: str) -> bool:
    return text in _KNOWN_NEUTRAL_HEADINGS


def _segmentation(job: ModelJob, fixtures: dict[str, dict[str, Any]],
                  context: dict[str, Any]) -> dict[str, Any]:
    blocks = []
    for block in job.payload.get("raw_blocks", []):
        text = block["text"]
        if block["role"] == "heading":
            if text not in _GENERATED_HEADINGS | {_KNOWN_ERROR_HEADING} and not _neutral_heading(text):
                raise ValueError("Demo cannot classify an unknown submitted heading")
            spans = [{"start": 0, "end": len(text), "kind": "connector", "references": [],
                      "purpose": "Preserve the exact submitted heading for independent review."}]
        elif text == _KNOWN_ERROR_PARAGRAPH:
            cut = len(_KNOWN_ERROR_PREFIX)
            spans = [
                {"start": 0, "end": cut, "kind": "synthesis",
                 "references": [_synthesis(fixtures, context, "result")],
                 "purpose": "Classify the submitted cross-fixture statement without rewriting it."},
                {"start": cut, "end": len(text), "kind": "connector", "references": [],
                 "purpose": "Retain the submitted connector and any concealed claim for review."},
            ]
        elif text in _GENERATED_PARAGRAPHS:
            spans = [{"start": 0, "end": len(text), "kind": "synthesis",
                      "references": [_synthesis(fixtures, context, _GENERATED_PARAGRAPHS[text])],
                      "purpose": "Classify the exact submitted paragraph without rewriting it."}]
        else:
            raise ValueError("Demo cannot classify unknown submitted paragraph text")
        blocks.append({"block_id": block["id"], "spans": spans})
    if [item["block_id"] for item in blocks] != job.payload.get("target_ids"):
        raise ValueError("Demo segmentation requires the exact pending target order")
    return {"blocks": blocks}


def _review(job: ModelJob) -> dict[str, Any]:
    units: dict[str, tuple[str, str, str]] = {}
    for section in ("outline", "draft"):
        for block in job.payload.get(section, {}).get("blocks", []):
            for unit in block.get("units", []):
                units[unit["id"]] = (unit["text"], unit["kind"], block["role"])
    target_ids = job.payload.get("target_ids")
    if not isinstance(target_ids, list) or target_ids != list(units):
        raise ValueError("Demo review requires every unit in exact order")
    items = []
    for unit_id in target_ids:
        text, kind, role = units[unit_id]
        if text in {_KNOWN_ERROR_HEADING, _KNOWN_ERROR_PREFIX, _KNOWN_ERROR_CONNECTOR}:
            decision = "contradicted"
            location = "heading" if role == "heading" else "connector" if kind == "connector" else "unit"
            reason = (
                f"The {location} conflicts with the exact A=40, 10→8 and B=60, 10→11 synthetic evidence, "
                "or turns a bounded simulation/proposal into a general observed result."
            )
        elif text in _GENERATED_HEADINGS or text in _GENERATED_PARAGRAPHS or (
            role == "heading" and _neutral_heading(text)
        ):
            decision = "supported"
            reason = (
                "Checked against the complete original A/B fixture, including workload conditions, "
                "the 10→8 and 10→11 counterexample, and simulation transfer limits."
            )
        else:
            raise ValueError("Demo cannot review unknown writing text")
        items.append({"unit_id": unit_id, "decision": decision, "reason": reason})
    return {"items": items}


def run(job: ModelJob) -> ModelResult:
    """Handle only the six exact writing/domain tasks and exact builtin fixture protocol."""
    task = job.payload.get("task")
    if task not in TASKS:
        return _failure("Demo does not support this writing task protocol")
    try:
        if not _pack_is_exact(job):
            raise ValueError("Demo writing supports only the exact builtin scheduling pack bytes")
        if task in {"domain-extract-v1", "domain-review-v1"}:
            fixture = _validated_domain_fixture(job)
            data = _domain_extract(job, fixture) if task == "domain-extract-v1" else _domain_review(job, fixture)
        else:
            fixtures = _validated_writing_fixtures(job)
            context = job.payload["context"]
            if task == "writing-outline-v1":
                data = _outline(fixtures, context)
            elif task == "writing-draft-v1":
                data = _draft(fixtures, context)
            elif task == "writing-segment-v1":
                data = _segmentation(job, fixtures, context)
            else:
                data = _review(job)
    except (AttributeError, KeyError, TypeError, ValueError, ValidationError):
        return _failure(
            "Demo writing input failed the exact builtin-pack and original-fixture protocol; "
            "use Exchange or a configured model for external domains."
        )
    return ModelResult(
        status="succeeded",
        backend="demo",
        data=data,
        usage={"mode": "synthetic_demo", "model_calls": 0},
    )
