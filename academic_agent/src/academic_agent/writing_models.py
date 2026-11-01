"""Strict, JSON-safe contracts shared by the writing and domain stages."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .innovation_models import (
    AssessmentContext,
    AssessmentReview,
    CandidateBatch,
    GroundingReview,
    InnovationAssessment,
    InnovationCandidate,
)
from .research_context import ResearchSnapshot
from .trends import ComputationRecord, CorpusSnapshot

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_DOCUMENT = re.compile(r"doc_[0-9a-f]{32}\Z")
_ID = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_STATISTIC_PATH = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\Z")
_STATUS = {"complete", "paused_budget", "needs_input", "input_too_large", "stale", "failed"}

ShortText = Annotated[str, StringConstraints(min_length=1, max_length=4096, strip_whitespace=True)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
Hash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
BlockId = Annotated[str, StringConstraints(pattern=r"^b_[0-9a-f]{32}$")]
UnitId = Annotated[str, StringConstraints(pattern=r"^u_[0-9a-f]{32}$")]
LongText = Annotated[str, StringConstraints(min_length=1, max_length=200000)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DomainTerm(_Strict):
    term: ShortText
    meaning: ShortText


class DomainFieldDefinition(_Strict):
    id: Identifier
    label: ShortText
    description: ShortText
    value_type: Literal["text", "number"]
    unit: Annotated[str, StringConstraints(max_length=256, strip_whitespace=True)] | None


class EvaluationCriterion(_Strict):
    id: Identifier
    text: ShortText


class WritingStructureItem(_Strict):
    id: Identifier
    title: ShortText
    purpose: ShortText


class DomainPack(_Strict):
    schema_version: Literal[1]
    contract: Literal["writing-domain-1"]
    name: Identifier
    version: ShortText
    terminology: list[DomainTerm] = Field(max_length=64)
    search_hints: list[ShortText] = Field(max_length=64)
    screening_hints: list[ShortText] = Field(max_length=64)
    fields: list[DomainFieldDefinition] = Field(min_length=1, max_length=24)
    comparison_columns: list[Identifier] = Field(min_length=1, max_length=12)
    evaluation_criteria: list[EvaluationCriterion] = Field(max_length=12)
    writing_structure: list[WritingStructureItem] = Field(min_length=1, max_length=12)

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_schema_version(cls, value):
        if type(value) is not int:
            raise ValueError("Domain pack schema version must be an integer")
        return value

    @model_validator(mode="after")
    def closed_references(self):
        for values in (self.terminology, self.fields, self.evaluation_criteria, self.writing_structure):
            keys = [getattr(value, "id", getattr(value, "term", None)) for value in values]
            if len(keys) != len(set(keys)):
                raise ValueError("Domain pack IDs and terms must be unique")
        for values in (self.search_hints, self.screening_hints, self.comparison_columns):
            if len(values) != len(set(values)):
                raise ValueError("Domain pack lists must not contain duplicates")
        field_ids = {field.id for field in self.fields}
        if not set(self.comparison_columns) <= field_ids:
            raise ValueError("Comparison columns must reference declared fields")
        return self


class PackPin(_Strict):
    source_path: str = Field(min_length=1)
    sha256: Hash
    pack: DomainPack


class ArtifactPin(_Strict):
    kind: Literal["trend", "innovation", "assessment", "domain", "writing"]
    version_path: str = Field(min_length=1)
    manifest_hash: Hash
    semantic_hash: Hash


class TrendPin(_Strict):
    artifact: ArtifactPin
    snapshot: CorpusSnapshot
    computation: ComputationRecord

    @field_validator("snapshot", mode="before")
    @classmethod
    def restore_json_snapshot_tuple(cls, value):
        # JSON has no tuple type. Restore the one fixed-width tuple before the
        # nested strict model checks it, while retaining strict values inside it.
        if isinstance(value, dict) and isinstance(value.get("source_snapshot"), list):
            value = {**value, "source_snapshot": tuple(value["source_snapshot"])}
        return value

    @model_validator(mode="after")
    def trend_kind(self):
        if self.artifact.kind != "trend":
            raise ValueError("Trend pin requires a trend artifact")
        return self


class CandidatePin(_Strict):
    artifact: ArtifactPin
    candidate: InnovationCandidate
    batch: CandidateBatch
    review: GroundingReview
    snapshot: ResearchSnapshot

    @model_validator(mode="after")
    def candidate_kind(self):
        if self.artifact.kind != "innovation":
            raise ValueError("Candidate pin requires an innovation artifact")
        return self


class AssessmentPin(_Strict):
    artifact: ArtifactPin
    candidate: InnovationCandidate
    context: AssessmentContext
    assessment: InnovationAssessment
    review: AssessmentReview
    snapshot: ResearchSnapshot

    @model_validator(mode="after")
    def assessment_kind(self):
        if self.artifact.kind != "assessment":
            raise ValueError("Assessment pin requires an assessment artifact")
        return self


class WritingSelection(_Strict):
    document_id: str
    question: ShortText
    work_ids: list[str] = Field(min_length=1)
    domain_path: str = Field(min_length=1)
    trend_directory: str | None
    candidate_directory: str | None
    candidate_id: str | None
    assessment_directory: str | None

    @field_validator("document_id")
    @classmethod
    def valid_document_id(cls, value: str) -> str:
        if not _DOCUMENT.fullmatch(value):
            raise ValueError("Writing document ID must be host-derived")
        return value

    @model_validator(mode="after")
    def explicit_scope_and_versions(self):
        if len(self.work_ids) != len(set(self.work_ids)) or any(not value.strip() for value in self.work_ids):
            raise ValueError("Writing work IDs must be unique and nonblank")
        if bool(self.candidate_directory) != bool(self.candidate_id):
            raise ValueError("Candidate directory and ID must be selected together")
        if self.assessment_directory and not self.candidate_directory:
            raise ValueError("Assessment selection requires its candidate selection")
        for value in (self.trend_directory, self.candidate_directory, self.assessment_directory):
            if value is None:
                continue
            path = Path(value)
            if not path.is_absolute() or path.name == "latest" or not path.name.startswith("v-"):
                raise ValueError("Writing dependencies require absolute concrete version directories")
        return self


class WritingContext(_Strict):
    schema_version: Literal[1]
    selection: WritingSelection
    research: ResearchSnapshot
    pack: PackPin
    trend: TrendPin | None
    candidate: CandidatePin | None
    assessment: AssessmentPin | None
    challenge_context: AssessmentContext | None
    input_hash: Hash

    @model_validator(mode="after")
    def selected_dependency_closure(self):
        if set(self.selection.work_ids) != set(self.research.work_ids):
            raise ValueError("Writing research does not match the selected work scope")
        source_path = self.selection.domain_path
        if source_path != "builtin:scheduling":
            source_path = str(Path(source_path).resolve())
        if source_path != self.pack.source_path:
            raise ValueError("Writing domain pack does not match the selected source")
        selected = (
            (self.selection.trend_directory, self.trend),
            (self.selection.candidate_directory, self.candidate),
            (self.selection.assessment_directory, self.assessment),
        )
        for version_path, pin in selected:
            if bool(version_path) != bool(pin):
                raise ValueError("Writing dependency selection and pin presence disagree")
            if version_path and pin and str(Path(version_path).resolve()) != pin.artifact.version_path:
                raise ValueError("Writing dependency pin does not match its selected version")
        if self.candidate and self.selection.candidate_id != self.candidate.candidate.id:
            raise ValueError("Writing candidate pin does not match the selected candidate")
        if bool(self.challenge_context) != bool(self.candidate):
            raise ValueError("Writing challenge context must accompany a candidate pin")
        if self.candidate and self.challenge_context and (
                self.challenge_context.candidate_manifest_hash != self.candidate.artifact.manifest_hash):
            raise ValueError("Writing challenge context does not match its candidate pin")
        if self.assessment and self.candidate and (
                self.assessment.candidate != self.candidate.candidate
                or self.assessment.context.candidate_manifest_hash != self.candidate.artifact.manifest_hash):
            raise ValueError("Writing assessment pin does not match its candidate pin")
        return self


class DomainField(_Strict):
    field_id: Identifier
    status: Literal["reported", "not_reported"]
    value: str | int | float | None
    evidence_ids: list[str]

    @field_validator("value", mode="before")
    @classmethod
    def finite_json_value(cls, value):
        if isinstance(value, bool):
            raise ValueError("Domain field values cannot be booleans")
        if isinstance(value, float) and not (-float("inf") < value < float("inf")):
            raise ValueError("Domain field values must be finite")
        return value

    @model_validator(mode="after")
    def status_matches_value(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Domain field evidence IDs must be unique")
        if self.status == "reported" and (self.value is None or not self.evidence_ids):
            raise ValueError("Reported domain fields require a value and evidence")
        if self.status == "not_reported" and (self.value is not None or self.evidence_ids):
            raise ValueError("Unreported domain fields require null and no evidence")
        return self


class DomainReviewItem(_Strict):
    field_id: Identifier
    decision: Literal["supported", "partial", "contradicted", "insufficient"]
    reason: ShortText


class DomainArtifact(_Strict):
    pin: ArtifactPin
    context: WritingContext
    fields: dict[str, list[DomainField]]
    review: dict[str, list[DomainReviewItem]]

    @model_validator(mode="after")
    def domain_pin_and_local_identities(self):
        if self.pin.kind != "domain":
            raise ValueError("Domain artifacts require a domain pin")
        for values in self.fields.values():
            if len({item.field_id for item in values}) != len(values):
                raise ValueError("Domain field IDs must be unique per work")
        for values in self.review.values():
            if len({item.field_id for item in values}) != len(values):
                raise ValueError("Domain review IDs must be unique per work")
        return self


class SourceRef(_Strict):
    type: Literal["source"]
    work_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_evidence(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("Source evidence IDs must be unique")
        return self


class SynthesisRef(_Strict):
    type: Literal["synthesis"]
    sources: list[SourceRef] = Field(min_length=2)

    @model_validator(mode="after")
    def multiple_works(self):
        if len({source.work_id for source in self.sources}) < 2:
            raise ValueError("Synthesis references require at least two distinct works")
        return self


class StatisticRef(_Strict):
    type: Literal["frozen_statistic"]
    pin_hash: Hash
    computation_path: str

    @field_validator("computation_path")
    @classmethod
    def valid_computation_path(cls, value: str) -> str:
        if not _STATISTIC_PATH.fullmatch(value):
            raise ValueError("Statistic path must be a host-provided dotted field path")
        return value


class HypothesisRef(_Strict):
    type: Literal["hypothesis"]
    candidate_id: str = Field(pattern=r"^innovation_[0-9a-f]{32}$")
    candidate_semantic_hash: Hash


Reference = Annotated[SourceRef | SynthesisRef | StatisticRef | HypothesisRef, Field(discriminator="type")]


UnitKind = Literal["source", "synthesis", "frozen_statistic", "hypothesis", "connector", "unclassified"]
WireUnitKind = Literal["source", "synthesis", "frozen_statistic", "hypothesis", "connector"]


def _validate_unit(kind: UnitKind, references: list[Reference], purpose: str | None):
    expected = {
        "source": SourceRef,
        "synthesis": SynthesisRef,
        "frozen_statistic": StatisticRef,
        "hypothesis": HypothesisRef,
    }
    if kind in expected and (not references or any(not isinstance(ref, expected[kind]) for ref in references)):
        raise ValueError("Claim unit kind must match every nonempty reference")
    if kind == "connector" and purpose is None:
        raise ValueError("Connector units require a purpose")
    if kind == "unclassified" and (references or purpose != "pending segmentation"):
        raise ValueError("Unclassified units must preserve the pending segmentation marker")
    if kind != "unclassified" and purpose == "pending segmentation":
        raise ValueError("Pending segmentation purpose is reserved for unclassified units")


class TextUnit(_Strict):
    id: UnitId
    text: LongText
    kind: UnitKind
    references: list[Reference]
    purpose: ShortText | None

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text unit content must be nonblank")
        return value

    @model_validator(mode="after")
    def reference_contract(self):
        _validate_unit(self.kind, self.references, self.purpose)
        return self


class WireTextUnit(_Strict):
    text: LongText
    kind: WireUnitKind
    references: list[Reference]
    purpose: ShortText | None

    _nonblank_text = field_validator("text")(TextUnit.nonblank_text.__func__)

    @model_validator(mode="after")
    def reference_contract(self):
        _validate_unit(self.kind, self.references, self.purpose)
        return self


class TextBlock(_Strict):
    id: BlockId
    role: Literal["heading", "paragraph"]
    level: int | None
    units: list[TextUnit] = Field(min_length=1)

    @field_validator("level", mode="before")
    @classmethod
    def strict_level(cls, value):
        if value is not None and type(value) is not int:
            raise ValueError("Heading level must be an integer or null")
        return value

    @model_validator(mode="after")
    def role_level_and_unit_identity(self):
        if self.role == "heading" and (self.level is None or not 1 <= self.level <= 3):
            raise ValueError("Heading blocks require level 1..3")
        if self.role == "paragraph" and self.level is not None:
            raise ValueError("Paragraph blocks require a null level")
        if len({unit.id for unit in self.units}) != len(self.units):
            raise ValueError("Text unit IDs must be unique within a block")
        return self


class WireTextBlock(_Strict):
    role: Literal["heading", "paragraph"]
    level: int | None
    units: list[WireTextUnit] = Field(min_length=1)

    _strict_level = field_validator("level", mode="before")(TextBlock.strict_level.__func__)

    @model_validator(mode="after")
    def role_level(self):
        if self.role == "heading" and (self.level is None or not 1 <= self.level <= 3):
            raise ValueError("Heading blocks require level 1..3")
        if self.role == "paragraph" and self.level is not None:
            raise ValueError("Paragraph blocks require a null level")
        return self


def _validate_blocks(blocks: list[TextBlock]) -> None:
    if len({block.id for block in blocks}) != len(blocks):
        raise ValueError("Text block IDs must be unique")
    unit_ids = [unit.id for block in blocks for unit in block.units]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("Text unit IDs must be unique across blocks")


class Outline(_Strict):
    blocks: list[TextBlock] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self):
        _validate_blocks(self.blocks)
        return self


class Draft(_Strict):
    blocks: list[TextBlock]

    @model_validator(mode="after")
    def unique_ids(self):
        _validate_blocks(self.blocks)
        return self


class WireOutline(_Strict):
    blocks: list[WireTextBlock] = Field(min_length=1)


class WireDraft(_Strict):
    blocks: list[WireTextBlock] = Field(min_length=1)


class WireTextSpan(_Strict):
    start: int
    end: int
    kind: WireUnitKind
    references: list[Reference]
    purpose: ShortText | None

    @field_validator("start", "end", mode="before")
    @classmethod
    def strict_offset(cls, value):
        if type(value) is not int:
            raise ValueError("Text span offsets must be integers")
        return value

    @model_validator(mode="after")
    def reference_contract(self):
        _validate_unit(self.kind, self.references, self.purpose)
        return self


class EditableBlock(_Strict):
    id: str
    role: Literal["heading", "paragraph"]
    level: int | None
    text: LongText

    @field_validator("id")
    @classmethod
    def editable_id(cls, value: str) -> str:
        if (not re.fullmatch(r"b_[0-9a-f]{32}|new:[a-z][a-z0-9_-]{0,63}", value)
                or "\x00" in value):
            raise ValueError("Editable block ID must be a parent ID or unique new:<slug> marker")
        return value

    @field_validator("level", mode="before")
    @classmethod
    def strict_level(cls, value):
        return TextBlock.strict_level.__func__(cls, value)

    @field_validator("text")
    @classmethod
    def exact_nonblank_text(cls, value: str) -> str:
        if not value.strip() or "\x00" in value or any(0xD800 <= ord(char) <= 0xDFFF for char in value):
            raise ValueError("Editable text must be nonblank Unicode without NUL or surrogate code points")
        return value

    @model_validator(mode="after")
    def role_level(self):
        if self.role == "heading" and (self.level is None or not 1 <= self.level <= 3):
            raise ValueError("Editable heading blocks require level 1..3")
        if self.role == "paragraph" and self.level is not None:
            raise ValueError("Editable paragraph blocks require a null level")
        return self


class EditableWriting(_Strict):
    schema_version: Literal[1]
    document_id: str
    parent_version: str
    parent_manifest_hash: Hash
    outline: list[EditableBlock] = Field(min_length=1)
    paragraphs: list[EditableBlock]

    _valid_document_id = field_validator("document_id")(WritingSelection.valid_document_id.__func__)

    @field_validator("parent_version")
    @classmethod
    def concrete_parent_version(cls, value: str) -> str:
        if not re.fullmatch(r"v-[0-9a-f]{32}", value):
            raise ValueError("Editable parent must be a concrete immutable version")
        return value

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [block.id for block in [*self.outline, *self.paragraphs]]
        if len(ids) != len(set(ids)):
            raise ValueError("Editable block IDs and new markers must be unique")
        return self


class ReviewItem(_Strict):
    unit_id: UnitId
    decision: Literal["supported", "partial", "contradicted", "insufficient"]
    reason: ShortText


class WritingReview(_Strict):
    items: list[ReviewItem]

    @model_validator(mode="after")
    def unique_units(self):
        if len({item.unit_id for item in self.items}) != len(self.items):
            raise ValueError("Writing review unit IDs must be unique")
        return self


class OutlineReviewState(_Strict):
    status: Literal["pending", "complete"]
    review: WritingReview

    @model_validator(mode="after")
    def status_matches_local_review_state(self):
        if self.status == "pending" and self.review.items:
            raise ValueError("Pending outline review state must have no decisions")
        if self.status == "complete" and not self.review.items:
            raise ValueError("Complete outline review state requires decisions")
        return self


class DomainRebind(_Strict):
    old: ArtifactPin
    new: ArtifactPin

    @model_validator(mode="after")
    def domain_pins(self):
        if self.old.kind != "domain" or self.new.kind != "domain":
            raise ValueError("Domain rebind requires domain artifact pins")
        return self


class Revision(_Strict):
    """Revision-local metadata; artifact loaders enforce body and exact review coverage."""

    schema_version: Literal[1]
    document_id: str
    extent: Literal["outline", "manuscript"]
    stage: Literal["pending", "reviewed"]
    parent_version: ShortText | None
    parent_manifest_hash: Hash | None
    origin: Literal["model", "human"]
    author: ShortText
    reason: ShortText
    created_at: ShortText
    input_sha256: Hash | None
    changed_ids: list[BlockId]
    added_ids: list[BlockId]
    deleted_ids: list[BlockId]
    reordered: bool
    id_map: dict[str, BlockId]
    pending_ids: list[BlockId]
    reused_from: dict[BlockId, ArtifactPin]
    domain_rebind: DomainRebind | None

    _valid_document_id = field_validator("document_id")(WritingSelection.valid_document_id.__func__)

    @field_validator("author", mode="before")
    @classmethod
    def validate_original_model_author(cls, value, info: ValidationInfo):
        if info.data.get("origin") == "model" and isinstance(value, str):
            has_control = any(unicodedata.category(character) == "Cc" for character in value)
            if value != value.strip() or has_control:
                raise ValueError("Model revision author cannot contain outer whitespace or controls")
        return value

    @model_validator(mode="after")
    def local_revision_contract(self):
        if bool(self.parent_version) != bool(self.parent_manifest_hash):
            raise ValueError("Revision parent version and manifest hash must be paired")
        if self.origin == "human" and (self.parent_version is None or self.parent_manifest_hash is None):
            raise ValueError("Human revisions require an exact parent receipt")
        if self.origin == "model" and (self.input_sha256 is not None or self.domain_rebind is not None):
            raise ValueError("Model revisions cannot claim raw edits or domain rebinding")
        if self.origin == "human" and self.input_sha256 is None:
            raise ValueError("Human revisions require the exact raw edit hash")
        if self.origin == "model":
            backend, separator, model = self.author.partition("/")
            if (not separator or not backend.strip() or backend != backend.strip()
                    or not model.strip() or model != model.strip()
                    or any(unicodedata.category(character) == "Cc" for character in self.author)):
                raise ValueError("Model revision author must be the actual backend/model identity")
        if self.stage == "reviewed" and self.pending_ids:
            raise ValueError("Reviewed revisions cannot retain pending block IDs")
        lists = (self.changed_ids, self.added_ids, self.deleted_ids, self.pending_ids)
        if any(len(values) != len(set(values)) for values in lists):
            raise ValueError("Revision block ID lists must not contain duplicates")
        if (set(self.changed_ids) & set(self.added_ids)
                or set(self.changed_ids) & set(self.deleted_ids)
                or set(self.added_ids) & set(self.deleted_ids)
                or set(self.pending_ids) & set(self.deleted_ids)):
            raise ValueError("Revision change and pending ID classes are inconsistent")
        if self.origin == "human" and set(self.id_map.values()) != set(self.added_ids):
            raise ValueError("Revision ID map must account for every added block")
        if len(self.id_map.values()) != len(set(self.id_map.values())):
            raise ValueError("Revision edit markers must map one-to-one to added blocks")
        if self.origin == "model" and self.id_map:
            raise ValueError("Model revisions cannot claim human edit markers")
        if any(not re.fullmatch(r"new:[a-z][a-z0-9_-]{0,63}", marker) for marker in self.id_map):
            raise ValueError("Revision ID map keys must be unique new:<slug> markers")
        if any(pin.kind != "writing" for pin in self.reused_from.values()):
            raise ValueError("Reused block provenance requires writing artifact pins")
        return self


class WritingArtifact(_Strict):
    context: WritingContext
    domain_pin: ArtifactPin
    outline: Outline
    draft: Draft
    review: WritingReview
    revision: Revision

    @model_validator(mode="after")
    def complete_artifact_contract(self):
        if self.domain_pin.kind != "domain":
            raise ValueError("Writing artifacts require a domain artifact pin")
        if self.revision.document_id != self.context.selection.document_id:
            raise ValueError("Writing revision and context document IDs disagree")
        if self.revision.extent == "outline" and self.draft.blocks:
            raise ValueError("Outline extent requires an empty manuscript body")
        if self.revision.extent == "manuscript" and not self.draft.blocks:
            raise ValueError("Manuscript extent requires a nonempty body")
        blocks = [*self.outline.blocks, *self.draft.blocks]
        _validate_blocks(blocks)
        units = [unit for block in blocks for unit in block.units]
        if self.revision.stage == "pending":
            if self.review.items:
                raise ValueError("Pending writing artifacts cannot retain current review decisions")
        else:
            if any(unit.kind == "unclassified" for unit in units):
                raise ValueError("Reviewed writing artifacts cannot contain unclassified text")
            if [item.unit_id for item in self.review.items] != [unit.id for unit in units]:
                raise ValueError("Reviewed writing artifacts require ordered full-unit review coverage")
        return self


def _identity_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _identity_hash(value: Any) -> str:
    return hashlib.sha256(_identity_bytes(value)).hexdigest()


def assign_generated_ids(document_id: str, task: str, input_hash: str,
                         wire_blocks: list[dict]) -> list[TextBlock]:
    """Validate model blocks, then assign their exact deterministic host identities."""
    WritingSelection.valid_document_id(document_id)
    if not _HASH.fullmatch(input_hash):
        raise ValueError("Writing generation input hash is invalid")
    try:
        checked = [WireTextBlock.model_validate(block, strict=True) for block in wire_blocks]
    except ValidationError:
        raise ValueError("Generated writing blocks failed strict wire validation") from None
    canonical_blocks = [block.model_dump(mode="json") for block in checked]
    seed = _identity_hash(["writing-ids-1", document_id, task, input_hash, canonical_blocks])
    blocks = []
    for block_index, block in enumerate(checked):
        block_id = "b_" + _identity_hash([seed, block_index])[:32]
        units = []
        for unit_index, unit in enumerate(block.units):
            unit_id = "u_" + _identity_hash([seed, block_index, unit_index])[:32]
            units.append(TextUnit(id=unit_id, **unit.model_dump(mode="json")))
        blocks.append(TextBlock(id=block_id, role=block.role, level=block.level, units=units))
    _validate_blocks(blocks)
    return blocks


def assign_segment_ids(block_id: str, segmentation_input_hash: str,
                       spans: list[dict], raw_text: str) -> list[TextUnit]:
    """Assign stable IDs to exact, contiguous Python-code-point spans of human text."""
    if not re.fullmatch(r"b_[0-9a-f]{32}", block_id):
        raise ValueError("Writing block ID is invalid")
    if not _HASH.fullmatch(segmentation_input_hash):
        raise ValueError("Writing segmentation input hash is invalid")
    if not isinstance(raw_text, str) or not raw_text:
        raise ValueError("Writing segmentation requires nonempty raw text")
    try:
        checked = [WireTextSpan.model_validate(span, strict=True) for span in spans]
    except ValidationError:
        raise ValueError("Writing segmentation spans failed strict wire validation") from None
    if not checked or checked[0].start != 0 or checked[-1].end != len(raw_text):
        raise ValueError("Writing segmentation must cover the complete original text")
    for index, span in enumerate(checked):
        if span.end <= span.start or (index and checked[index - 1].end != span.start):
            raise ValueError("Writing segmentation spans must be positive, contiguous, and ordered")
    units = []
    for index, span in enumerate(checked):
        wire = span.model_dump(mode="json")
        start, end = wire.pop("start"), wire.pop("end")
        unit_id = "u_" + _identity_hash(
            ["writing-segment-ids-1", block_id, segmentation_input_hash, index,
             span.model_dump(mode="json")]
        )[:32]
        units.append(TextUnit(id=unit_id, text=raw_text[start:end], **wire))
    if len({unit.id for unit in units}) != len(units):
        raise ValueError("Writing segmentation unit IDs collided")
    return units


class WritingError(ValueError):
    """Operation failure with a stable status and an optional immutable late output."""

    def __init__(self, message: str, *, status: str, output: Path | None = None):
        if status not in _STATUS:
            raise ValueError("Unknown writing operation status")
        super().__init__(message)
        self.status = status
        self.output = output
