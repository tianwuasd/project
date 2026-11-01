import copy

import pytest

from academic_agent.analysis import AnalysisError, read, review
from academic_agent.backends import DemoBackend
from academic_agent.documents import make_evidence
from academic_agent.models import Document, ModelResult, Page, PublicationCheck, Work, digest


@pytest.fixture
def source():
    text = (
        "ORIGINAL_SYNTHETIC_FIXTURE\n"
        "problem: Can a toy scheduler reduce delay?\n"
        "method: Compare FIFO and shortest-job-first in an invented simulation.\n"
        "data: The synthetic sample contains 12 jobs.\n"
        "result: Median delay fell from 9 to 6 ticks; reliability did not improve.\n"
        "limitation: This invented experiment has no real-world validation.\n"
        "contribution: This is a reproducible teaching example.\n"
        "Appendix: The simulation does not establish causality in operational systems.\n"
    )
    work = Work(id="work-1", title="Original scheduler fixture")
    document = Document(id="doc-1", work_id=work.id, content_hash=digest(text),
                        level="full_text", pages=[Page(number=1, text=text)])
    return work, document, make_evidence(document, chunk_size=200)


class FakeBackend:
    name = "fake"

    def __init__(self, data, status="succeeded"):
        self.data = data
        self.status = status
        self.jobs = []

    def run(self, job):
        self.jobs.append(job)
        data = self.data(job) if callable(self.data) else self.data
        return ModelResult(backend=self.name, status=self.status, data=data, error="untrusted secret prompt")


def test_demo_reads_six_categories_and_independently_reviews(source):
    work, document, evidence = source
    card = read(work, document, evidence, DemoBackend())
    assert len(card.claims) == 6
    assert not card.missing
    assert card.backend == "demo"
    report = review(card, document, evidence, DemoBackend())
    assert len(report.verdicts) == 6
    assert all(v.verdict == "supported" for v in report.verdicts)
    assert "full_text" in report.scope
    assert all("no independent AI" in v.rationale for v in report.verdicts)


def test_claim_ids_are_stable_and_host_owned(source):
    work, document, evidence = source
    data = {"claims": [{"text": "The sample has 12 jobs.", "category": "data",
                         "evidence_ids": [evidence[0].id]}], "missing": []}
    backend = FakeBackend(data)
    first = read(work, document, evidence, backend)
    second = read(work, document, evidence, backend)
    assert first.claims[0].id == second.claims[0].id
    assert first.claims[0].work_id == work.id
    assert set(first.missing) == {"problem", "method", "result", "limitation", "contribution"}
    assert backend.jobs[0].id == backend.jobs[1].id
    data["claims"][0]["text"] = "The sample has 13 jobs."
    assert read(work, document, evidence, backend).claims[0].id != first.claims[0].id


@pytest.mark.parametrize("field", ["id", "work_id", "document_id"])
def test_reader_cannot_supply_identifiers(source, field):
    work, document, evidence = source
    data = {"claims": [{"text": "Claim", "category": "data", "evidence_ids": [evidence[0].id],
                         field: "forged"}], "missing": []}
    with pytest.raises(AnalysisError, match="schema"):
        read(work, document, evidence, FakeBackend(data))


def test_nonexistent_evidence_is_rejected_without_repair_call(source):
    work, document, evidence = source
    backend = FakeBackend({"claims": [{"text": "Claim", "category": "result",
                                       "evidence_ids": ["invented"]}], "missing": []})
    with pytest.raises(AnalysisError, match="nonexistent"):
        read(work, document, evidence, backend)
    assert len(backend.jobs) == 1


@pytest.mark.parametrize("change", ["hash", "span", "document", "work", "duplicate"])
def test_invalid_provenance_never_reaches_backend(source, change):
    work, document, evidence = copy.deepcopy(source)
    if change == "hash":
        evidence[0].text_hash = "invalid"
    elif change == "span":
        evidence[0].text = "tampered"
    elif change == "document":
        evidence[0].document_id = "other-doc"
    elif change == "work":
        evidence[0].work_id = "other-work"
    else:
        evidence.append(evidence[0])
    backend = FakeBackend({})
    with pytest.raises(AnalysisError):
        read(work, document, evidence, backend)
    assert not backend.jobs


def test_independent_review_receives_full_context_not_just_writer_excerpts(source):
    work, document, evidence = source
    card = read(work, document, evidence, DemoBackend())
    backend = FakeBackend({"verdicts": []})
    report = review(card, document, evidence, backend)
    assert len(report.verdicts) == len(card.claims)
    assert all(v.verdict == "insufficient" for v in report.verdicts)
    payload = backend.jobs[0].payload
    assert payload["work"]["id"] == work.id
    assert payload["document"]["id"] == document.id
    assert payload["full_available_context"][0]["text"] == document.pages[0].text
    assert "does not establish causality" in payload["full_available_context"][0]["text"]


@pytest.mark.parametrize("fault", ["no_evidence", "unknown_evidence", "unrelated_evidence", "duplicate",
                                   "invalid_verdict", "no_rationale"])
def test_invalid_supported_verdicts_become_insufficient(source, fault):
    work, document, evidence = source
    card = read(work, document, evidence, DemoBackend())
    claim = card.claims[0]
    unrelated = next(e.id for e in evidence if e.id not in claim.evidence_ids)
    verdict = {"claim_id": claim.id, "verdict": "supported", "rationale": "Looks supported",
               "evidence_ids": claim.evidence_ids}
    if fault == "no_evidence":
        verdict["evidence_ids"] = []
    elif fault == "unknown_evidence":
        verdict["evidence_ids"] = ["made-up"]
    elif fault == "unrelated_evidence":
        verdict["evidence_ids"] = [unrelated]
    elif fault == "invalid_verdict":
        verdict["verdict"] = "probably"
    elif fault == "no_rationale":
        verdict["rationale"] = ""
    data = {"verdicts": [verdict, verdict] if fault == "duplicate" else [verdict]}
    report = review(card, document, evidence, FakeBackend(data))
    assert all(v.verdict == "insufficient" for v in report.verdicts)


@pytest.mark.parametrize("text, verdict", [
    ("Median delay fell from 9 to 3 ticks; reliability did not improve.", "contradicted"),
    ("Median delay fell from 9 to 6 ticks; reliability improved.", "contradicted"),
    ("Median delay fell from 9 to 6 ticks and this generalizes to every scheduler.", "partial"),
])
def test_semantic_verdict_is_separate_from_valid_exact_evidence(source, text, verdict):
    work, document, evidence = source
    card = read(work, document, evidence, DemoBackend())
    result_claim = next(c for c in card.claims if c.category == "result")
    result_claim.text = text
    backend = FakeBackend({"verdicts": [{"claim_id": result_claim.id, "verdict": verdict,
                                        "rationale": "Claim differs from available result sentence",
                                        "evidence_ids": result_claim.evidence_ids}]})
    report = review(card, document, evidence, backend)
    assert next(v for v in report.verdicts if v.claim_id == result_claim.id).verdict == verdict
    demo_report = review(card, document, evidence, DemoBackend())
    assert next(v for v in demo_report.verdicts if v.claim_id == result_claim.id).verdict == "insufficient"


def test_unknown_review_claim_is_rejected(source):
    work, document, evidence = source
    card = read(work, document, evidence, DemoBackend())
    backend = FakeBackend({"verdicts": [{"claim_id": "invented", "verdict": "supported",
                                        "rationale": "OK", "evidence_ids": [evidence[0].id]}]})
    with pytest.raises(AnalysisError, match="unknown claim"):
        review(card, document, evidence, backend)


def test_question_uses_ask_job_and_citations(source):
    work, document, evidence = source
    backend = FakeBackend(lambda job: DemoBackend().run(job).data)
    card = read(work, document, evidence, backend, question="What was the result?")
    assert backend.jobs[0].kind == "ask"
    assert backend.jobs[0].payload["question"] == "What was the result?"
    assert card.claims and all(c.category == "result" for c in card.claims)


@pytest.mark.parametrize("level", ["metadata_only", "needs_ocr"])
def test_inadequate_document_is_explicit(source, level):
    work, document, evidence = source
    document.level = level
    with pytest.raises(AnalysisError, match=level):
        read(work, document, evidence, FakeBackend({}))


def test_input_limit_applies_before_any_model_call_for_read_and_review(source):
    work, document, evidence = source
    card = read(work, document, evidence, DemoBackend())
    backend = FakeBackend({})
    with pytest.raises(AnalysisError, match="no text was truncated"):
        read(work, document, evidence, backend, max_input_chars=100)
    with pytest.raises(AnalysisError, match="no text was truncated"):
        review(card, document, evidence, backend, max_input_chars=100)
    assert not backend.jobs


def test_abstract_remains_abstract_and_retraction_rejected(source):
    work, document, evidence = source
    document.level = "abstract_only"
    card = read(work, document, evidence, DemoBackend())
    assert "abstract_only" in review(card, document, evidence, DemoBackend()).scope
    work.publication_checks = [PublicationCheck(source="test", status="retracted")]
    with pytest.raises(AnalysisError, match="Retracted"):
        read(work, document, evidence, DemoBackend())


def test_backend_failure_does_not_echo_external_diagnostic(source):
    work, document, evidence = source
    with pytest.raises(AnalysisError) as caught:
        read(work, document, evidence, FakeBackend({}, status="needs_input"))
    assert caught.value.status == "needs_input"
    assert "secret" not in str(caught.value)
