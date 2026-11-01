"""Strict public contracts for bounded research innovation candidates."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class InnovationCandidate(_Strict):
    schema_version: Literal[1] = 1
    id: str = Field(min_length=1)
    statement_type: Literal["system_hypothesis"]
    question: str = Field(min_length=1)
    closest_work_ids: list[str] = Field(min_length=1)
    closest_claim_ids: list[str] = Field(min_length=1)
    claimed_difference: str = Field(min_length=1)
    grounding_claim_ids: list[str] = Field(min_length=1)
    grounding_evidence_ids: list[str] = Field(min_length=1)
    possible_counterexamples: list[str] = Field(min_length=1)
    falsifiable_hypothesis: str = Field(min_length=1)
    minimum_validation: str = Field(min_length=1)
    data_requirements: str = Field(min_length=1)
    compute_requirements: str = Field(min_length=1)
    risks: list[str] = Field(min_length=1)
    targeted_queries: list[str] = Field(min_length=1, max_length=2)
    material_cutoff: str = Field(min_length=1)

    @field_validator(
        "question", "claimed_difference", "falsifiable_hypothesis", "minimum_validation",
        "data_requirements", "compute_requirements", "material_cutoff",
    )
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Candidate text must be nonblank")
        return value

    @field_validator(
        "closest_work_ids", "closest_claim_ids", "grounding_claim_ids", "grounding_evidence_ids",
        "possible_counterexamples", "risks", "targeted_queries",
    )
    @classmethod
    def unique_nonblank_items(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("Candidate lists require unique nonblank values")
        return normalized

    @field_validator("id")
    @classmethod
    def host_id(cls, value: str) -> str:
        if not re.fullmatch(r"innovation_[0-9a-f]{32}", value):
            raise ValueError("Candidate ID must be host-derived")
        return value


class CandidateBatch(_Strict):
    schema_version: Literal[1] = 1
    task_version: Literal["innovation-generation-v1"] = "innovation-generation-v1"
    input_hash: str
    candidates: list[InnovationCandidate] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_candidates(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.input_hash):
            raise ValueError("Candidate input hash is invalid")
        if len({candidate.id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("Candidate IDs must be unique")
        return self


GroundingVerdict = Literal["supported", "partial", "contradicted", "insufficient"]


class GroundingReviewItem(_Strict):
    schema_version: Literal[1] = 1
    candidate_id: str
    verdict: GroundingVerdict
    rationale: str = Field(min_length=1)
    claim_ids: list[str]
    evidence_ids: list[str]

    @field_validator("candidate_id", "rationale")
    @classmethod
    def nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Review text and identity must be nonblank")
        return value

    @field_validator("claim_ids", "evidence_ids")
    @classmethod
    def unique_nonblank_items(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("Review references require unique nonblank values")
        return normalized


class GroundingReview(_Strict):
    schema_version: Literal[1] = 1
    task_version: Literal["innovation-grounding-review-v1"] = "innovation-grounding-review-v1"
    input_hash: str
    backend: str = Field(min_length=1)
    items: list[GroundingReviewItem] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_items(self):
        if not re.fullmatch(r"[0-9a-f]{64}", self.input_hash):
            raise ValueError("Review input hash is invalid")
        if len({item.candidate_id for item in self.items}) != len(self.items):
            raise ValueError("Review candidate IDs must be unique")
        return self


ChallengeStatus = Literal["not_searched", "source_failed", "needs_reading", "needs_assessment"]
CoverageReason = Literal[
    "not_searched", "empty", "partial", "unavailable", "rate_limited", "truncated",
    "dropped_at_cap", "pending_screen", "excluded_with_reason", "excluded_without_reason",
    "unread", "unreviewed", "ready", "missing_work", "retracted", "publication_unknown",
    "retained_ids_unavailable", "abstract_only",
]


class ChallengeTask(_Strict):
    reason: CoverageReason
    round: Literal[1, 2] | None = None
    work_id: str | None = None
    query: str | None = None
    source: str | None = None
    detail: str = ""


class ChallengeSourceSummary(_Strict):
    source: str = Field(min_length=1)
    status: Literal["ok", "empty", "partial", "unavailable", "rate_limited"]
    records: int = Field(ge=0)
    truncated: bool
    errors: list[str]


class ChallengeQueryReceipt(_Strict):
    query: str = Field(min_length=1)
    query_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_event_id: str = Field(min_length=1)
    search_event_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, max_length=128)
    created: str = Field(min_length=1)
    effective_limit: int = Field(ge=1, le=10)
    project_candidate_cap: int = Field(ge=1)
    sources: list[ChallengeSourceSummary]
    retained_work_ids: list[str]
    retained_ids_available: bool
    dropped_at_project_cap: int = Field(ge=0)
    complete: bool

    @model_validator(mode="after")
    def consistent_receipt(self):
        if self.retained_work_ids != sorted(set(self.retained_work_ids)):
            raise ValueError("Receipt retained IDs must be unique and ordered")
        if not self.retained_ids_available and self.retained_work_ids:
            raise ValueError("Unavailable retained IDs cannot be fabricated")
        healthy = bool(self.sources) and all(s.status in {"ok", "empty"} for s in self.sources)
        if self.complete != (healthy and self.retained_ids_available):
            raise ValueError("Receipt completion disagrees with source coverage")
        return self


class ChallengeHit(_Strict):
    work_id: str
    statuses: list[CoverageReason]
    screening_reason: str
    access_level: Literal["metadata_only", "abstract_only", "full_text", "needs_ocr"]
    publication_status: Literal["unknown", "no_notice_found", "retracted", "corrected"]


class ChallengeRound(_Strict):
    schema_version: Literal[1] = 1
    task_version: Literal["counterexample-search-v1"] = "counterexample-search-v1"
    identity: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^innovation_[0-9a-f]{32}$")
    candidate_manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    round: Literal[1, 2]
    queries: list[str] = Field(min_length=1, max_length=2)
    sources: list[str]
    effective_limit: int = Field(ge=1, le=10)
    start_event_id: str = Field(min_length=1)
    receipts: list[ChallengeQueryReceipt] = Field(max_length=2)
    search_attempts: list[ChallengeQueryReceipt]
    status: ChallengeStatus
    retained_work_ids: list[str]
    unresolved_tasks: list[ChallengeTask]

    @field_validator("round", mode="before")
    @classmethod
    def integer_round(cls, value):
        if type(value) is not int:
            raise ValueError("Challenge round must be an integer")
        return value

    @model_validator(mode="after")
    def consistent_round(self):
        if (len(set(self.queries)) != len(self.queries)
                or any(not q.strip() or q != q.strip() for q in self.queries)
                or len(set(self.sources)) != len(self.sources) or any(not s.strip() for s in self.sources)
                or len({r.query for r in self.receipts}) != len(self.receipts)
                or any(r.query not in self.queries or r.effective_limit != self.effective_limit
                       or [s.source for s in r.sources] != self.sources for r in self.receipts)):
            raise ValueError("Challenge receipts do not match round scope")
        if (len({r.search_event_id for r in self.search_attempts}) != len(self.search_attempts)
                or any(r not in self.search_attempts for r in self.receipts)
                or any(r.query not in self.queries or r.effective_limit != self.effective_limit
                       or [s.source for s in r.sources] != self.sources for r in self.search_attempts)):
            raise ValueError("Challenge attempts disagree with current receipts or scope")
        retained = sorted({wid for r in self.search_attempts for wid in r.retained_work_ids})
        if self.retained_work_ids != retained:
            raise ValueError("Challenge round retained IDs disagree with receipts")
        return self


class AssessmentStatement(_Strict):
    text: str = Field(min_length=1)
    claim_ids: list[str]
    evidence_ids: list[str]

    _text = field_validator('text')(GroundingReviewItem.nonblank.__func__)
    _refs = field_validator('claim_ids', 'evidence_ids')(GroundingReviewItem.unique_nonblank_items.__func__)


class AssessmentDimension(_Strict):
    judgment: str = Field(min_length=1)
    claim_ids: list[str]
    evidence_ids: list[str]
    reasons: list[AssessmentStatement] = Field(min_length=1)

    _text = field_validator('judgment')(GroundingReviewItem.nonblank.__func__)
    _refs = field_validator('claim_ids', 'evidence_ids')(GroundingReviewItem.unique_nonblank_items.__func__)


class MinimumValidation(_Strict):
    proposal: str = Field(min_length=1)
    falsification_condition: str = Field(min_length=1)

    _text = field_validator('proposal', 'falsification_condition')(GroundingReviewItem.nonblank.__func__)


class AssessmentContent(_Strict):
    novelty_clues: AssessmentDimension
    research_value: AssessmentDimension
    feasibility: AssessmentDimension
    evidence_strength: AssessmentDimension
    gaps: list[AssessmentStatement]
    minimum_validation: MinimumValidation


class AssessmentContext(_Strict):
    schema_version: Literal[1] = 1
    candidate_manifest_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    candidate_current: bool
    premise_gaps: list[dict[str, JsonValue]]
    rounds: list[ChallengeRound]
    hits: list[ChallengeHit]
    unresolved_tasks: list[ChallengeTask]
    coverage_status: ChallengeStatus


class InnovationAssessment(AssessmentContent):
    schema_version: Literal[1] = 1
    task_version: Literal['innovation-assessment-v1'] = 'innovation-assessment-v1'
    input_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    candidate_id: str = Field(pattern=r'^innovation_[0-9a-f]{32}$')
    statement_type: Literal['scope_limited_assessment'] = 'scope_limited_assessment'
    material_cutoff: str
    material_scope: dict[str, JsonValue]
    host_gaps: list[dict[str, JsonValue]]
    evidence_sufficiency: Literal['insufficient', 'available_within_scope']


class AssessmentReviewItem(_Strict):
    unit_id: str = Field(min_length=1)
    verdict: GroundingVerdict
    rationale: str = Field(min_length=1)
    claim_ids: list[str]
    evidence_ids: list[str]

    _text = field_validator('unit_id', 'rationale')(GroundingReviewItem.nonblank.__func__)
    _refs = field_validator('claim_ids', 'evidence_ids')(GroundingReviewItem.unique_nonblank_items.__func__)


class AssessmentReview(_Strict):
    schema_version: Literal[1] = 1
    task_version: Literal['innovation-assessment-review-v1'] = 'innovation-assessment-review-v1'
    input_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    candidate_id: str = Field(pattern=r'^innovation_[0-9a-f]{32}$')
    backend: str = Field(min_length=1)
    items: list[AssessmentReviewItem] = Field(min_length=1)

    @model_validator(mode='after')
    def unique_units(self):
        if len({item.unit_id for item in self.items}) != len(self.items):
            raise ValueError('Assessment review requires unique units')
        return self
