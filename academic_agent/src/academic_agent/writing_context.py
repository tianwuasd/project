"""Capture semantic writing inputs and revalidate their explicit immutable pins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .challenges import candidate_version_currentness, load_candidate_version, read_challenges
from .domain import load_domain_pack
from .innovation_analysis import assessment_context_payload
from .innovation_models import AssessmentContext
from .innovations import assessment_version_currentness, load_assessment_version
from .research_context import (
    ResearchSnapshot,
    read_research,
    research_fingerprint,
    research_state_currentness,
)
from .trends import VerifiedTrends, load_verified_trends
from .writing_models import (
    ArtifactPin,
    AssessmentPin,
    CandidatePin,
    TrendPin,
    WritingContext,
    WritingError,
    WritingSelection,
)


def _canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hash(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _research_payload(snapshot: ResearchSnapshot) -> dict:
    return snapshot.stable_payload()


def _trend_payload(value: VerifiedTrends | TrendPin) -> dict:
    return {"snapshot": value.snapshot.model_dump(mode="json"),
            "computation": value.computation.model_dump(mode="json")}


def _candidate_payload(loaded: dict | CandidatePin) -> dict:
    get = (lambda name: getattr(loaded, name)) if isinstance(loaded, CandidatePin) else loaded.__getitem__
    return {
        "candidate": get("candidate").model_dump(mode="json"),
        "batch": get("batch").model_dump(mode="json"),
        "review": get("review").model_dump(mode="json"),
        "snapshot": _research_payload(get("snapshot")),
    }


def _assessment_payload(loaded: dict | AssessmentPin) -> dict:
    get = (lambda name: getattr(loaded, name)) if isinstance(loaded, AssessmentPin) else loaded.__getitem__
    return {
        "candidate": get("candidate").model_dump(mode="json"),
        "context": assessment_context_payload(get("context")),
        "assessment": get("assessment").model_dump(mode="json", exclude={"input_hash"}),
        "review": get("review").model_dump(mode="json", exclude={"input_hash", "backend"}),
        "snapshot": _research_payload(get("snapshot")),
    }


def _pin(kind: str, version_path: Path, manifest_hash: str, semantic) -> ArtifactPin:
    return ArtifactPin(kind=kind, version_path=str(Path(version_path).resolve()),
                       manifest_hash=manifest_hash, semantic_hash=_hash(semantic))


def _load_trend(directory: str) -> TrendPin:
    verified = load_verified_trends(Path(directory))
    artifact = _pin("trend", Path(directory), verified.manifest.manifest_hash, _trend_payload(verified))
    return TrendPin(artifact=artifact, snapshot=verified.snapshot, computation=verified.computation)


def _load_candidate(directory: str, candidate_id: str) -> CandidatePin:
    loaded = load_candidate_version(Path(directory), candidate_id)
    artifact = _pin("innovation", loaded["version_path"], loaded["manifest_hash"], _candidate_payload(loaded))
    return CandidatePin(artifact=artifact, candidate=loaded["candidate"], batch=loaded["batch"],
                        review=loaded["review"], snapshot=loaded["snapshot"])


def _load_assessment(root: Path, directory: str) -> AssessmentPin:
    loaded = load_assessment_version(root, Path(directory))
    artifact = _pin("assessment", loaded["version_path"], loaded["manifest_hash"],
                    _assessment_payload(loaded))
    return AssessmentPin(artifact=artifact, candidate=loaded["candidate"], context=loaded["context"],
                         assessment=loaded["assessment"], review=loaded["review"],
                         snapshot=loaded["snapshot"])


def _challenge_context(
    root: Path,
    candidate: CandidatePin,
    *,
    verified_research: ResearchSnapshot | None = None,
) -> AssessmentContext:
    current = candidate_version_currentness(root, {
        "candidate": candidate.candidate, "batch": candidate.batch, "review": candidate.review,
        "snapshot": candidate.snapshot, "version_path": Path(candidate.artifact.version_path),
        "manifest_hash": candidate.artifact.manifest_hash,
    })
    coverage = read_challenges(
        root,
        candidate.candidate.id,
        verified_research=verified_research,
    )
    return AssessmentContext(
        candidate_manifest_hash=candidate.artifact.manifest_hash,
        candidate_current=current["current"], premise_gaps=current["gaps"],
        rounds=coverage["rounds"], hits=coverage["hits"],
        unresolved_tasks=coverage["unresolved_tasks"], coverage_status=coverage["status"],
    )


def _assert_candidate_references(research: ResearchSnapshot, candidate: CandidatePin) -> None:
    claims = {claim.id: claim for card in research.cards.values() for claim in card.claims}
    if not set(candidate.candidate.closest_work_ids) <= set(research.work_ids):
        raise ValueError("Candidate references a work outside the writing selection")
    claim_ids = set(candidate.candidate.closest_claim_ids + candidate.candidate.grounding_claim_ids)
    if not claim_ids <= set(claims):
        raise ValueError("Candidate references an unknown writing claim")
    if not set(candidate.candidate.grounding_evidence_ids) <= set(research.evidence):
        raise ValueError("Candidate references unknown writing evidence")
    for claim_id in claim_ids:
        claim = claims[claim_id]
        if claim.work_id not in research.work_ids or any(eid not in research.evidence for eid in claim.evidence_ids):
            raise ValueError("Candidate reference chain is foreign to the writing selection")


def writing_context_payload(context: WritingContext) -> dict:
    """Return the semantic cache projection, excluding publication-only artifact pins."""
    selection = context.selection.model_dump(
        mode="json", exclude={"trend_directory", "candidate_directory", "assessment_directory"})
    return {
        "schema_version": context.schema_version,
        "selection": selection,
        "research": _research_payload(context.research),
        "pack": context.pack.model_dump(mode="json"),
        "trend": _trend_payload(context.trend) if context.trend else None,
        "candidate": _candidate_payload(context.candidate) if context.candidate else None,
        "assessment": _assessment_payload(context.assessment) if context.assessment else None,
        "challenge_context": (assessment_context_payload(context.challenge_context)
                              if context.challenge_context else None),
    }


def load_writing_context(root: Path, selection: WritingSelection) -> WritingContext:
    """Capture research and every explicit dependency, then close the read fence once."""
    root = Path(root).resolve()
    selection = WritingSelection.model_validate(selection.model_dump(mode="json"), strict=True)
    before = read_research(root, work_ids=selection.work_ids, require_supported=True)
    pack = load_domain_pack(selection.domain_path)
    trend = _load_trend(selection.trend_directory) if selection.trend_directory else None
    candidate = (_load_candidate(selection.candidate_directory, selection.candidate_id)
                 if selection.candidate_directory and selection.candidate_id else None)
    assessment = _load_assessment(root, selection.assessment_directory) if selection.assessment_directory else None
    if candidate:
        _assert_candidate_references(before, candidate)
    if assessment and (candidate is None or assessment.candidate != candidate.candidate
                       or assessment.context.candidate_manifest_hash != candidate.artifact.manifest_hash):
        raise ValueError("Assessment does not belong to the selected candidate version")
    challenge_context = _challenge_context(root, candidate) if candidate else None
    after = read_research(root, work_ids=selection.work_ids, require_supported=True)
    if research_fingerprint(before) != research_fingerprint(after):
        raise WritingError("Research inputs changed while capturing writing context", status="stale")
    placeholder = WritingContext(
        schema_version=1, selection=selection, research=before, pack=pack, trend=trend,
        candidate=candidate, assessment=assessment, challenge_context=challenge_context,
        input_hash="0" * 64,
    )
    context = placeholder.model_copy(update={"input_hash": _hash(writing_context_payload(placeholder))})
    state = writing_currentness(root, context)
    if not state["current"]:
        status = "failed" if any(gap["reason"].endswith("invalid") for gap in state["gaps"]) else "stale"
        raise WritingError("Writing dependencies changed while capturing context", status=status)
    return context


def writing_currentness(root: Path, context: WritingContext) -> dict:
    """Check the same explicit selection; never substitute any latest pointer."""
    root = Path(root).resolve()
    try:
        context = WritingContext.model_validate_json(context.model_dump_json(), strict=True)
    except (AttributeError, TypeError, ValueError):
        return {"current": False, "gaps": [{"reason": "writing_context_invalid"}]}
    if _hash(writing_context_payload(context)) != context.input_hash:
        return {"current": False, "gaps": [{"reason": "writing_context_identity_invalid"}]}
    gaps = []
    gaps.extend(research_state_currentness(root, context.research)["gaps"])
    try:
        if load_domain_pack(context.pack.source_path) != context.pack:
            gaps.append({"reason": "domain_pack_changed"})
    except (OSError, ValueError, KeyError, TypeError):
        gaps.append({"reason": "domain_pack_invalid"})
    if context.trend:
        try:
            loaded = _load_trend(context.trend.artifact.version_path)
            if loaded != context.trend:
                gaps.append({"reason": "trend_pin_invalid"})
        except (OSError, ValueError, KeyError, TypeError):
            gaps.append({"reason": "trend_pin_invalid"})
    if context.candidate:
        try:
            loaded = _load_candidate(context.candidate.artifact.version_path, context.candidate.candidate.id)
            if loaded != context.candidate:
                gaps.append({"reason": "candidate_pin_invalid"})
            state = candidate_version_currentness(root, {
                "candidate": loaded.candidate, "batch": loaded.batch, "review": loaded.review,
                "snapshot": loaded.snapshot, "version_path": Path(loaded.artifact.version_path),
                "manifest_hash": loaded.artifact.manifest_hash,
            })
            gaps.extend(state["gaps"])
            if context.challenge_context and assessment_context_payload(_challenge_context(
                    root, loaded, verified_research=loaded.snapshot)) != \
                    assessment_context_payload(context.challenge_context):
                gaps.append({"reason": "challenge_context_changed"})
        except (OSError, ValueError, KeyError, TypeError):
            gaps.append({"reason": "candidate_pin_invalid"})
    if context.assessment:
        try:
            loaded = _load_assessment(root, context.assessment.artifact.version_path)
            if loaded != context.assessment:
                gaps.append({"reason": "assessment_pin_invalid"})
            state = assessment_version_currentness(root, {
                "candidate": loaded.candidate, "context": loaded.context, "assessment": loaded.assessment,
                "review": loaded.review, "snapshot": loaded.snapshot,
                "version_path": Path(loaded.artifact.version_path),
                "manifest_hash": loaded.artifact.manifest_hash,
            })
            gaps.extend(state["gaps"])
        except (OSError, ValueError, KeyError, TypeError):
            gaps.append({"reason": "assessment_pin_invalid"})
    unique = []
    for gap in gaps:
        if gap not in unique:
            unique.append(gap)
    return {"current": not unique, "gaps": unique}


def assert_writing_current(root: Path, context: WritingContext) -> None:
    state = writing_currentness(root, context)
    if state["current"]:
        return
    status = "failed" if any(gap["reason"].endswith("invalid") for gap in state["gaps"]) else "stale"
    raise WritingError("Writing dependencies are no longer current", status=status)
