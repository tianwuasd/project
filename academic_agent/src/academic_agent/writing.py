"""Immutable structured writing workflow, rendering, loading, and read-only status."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, ValidationError

from .bibliography import export_bibtex
from .domain import domain_context_payload, load_domain_version
from .filesystem import io_path
from .models import utcnow
from .research_artifacts import load_research, publish_research
from .research_context import research_fingerprint
from .store import BudgetExceeded
from .writing_analysis import generate_draft, generate_outline, review_text, validate_structured_text
from .writing_context import load_writing_context, writing_context_payload, writing_currentness
from .writing_models import (
    ArtifactPin,
    Draft,
    Outline,
    OutlineReviewState,
    Revision,
    SourceRef,
    StatisticRef,
    SynthesisRef,
    WritingArtifact,
    WritingContext,
    WritingError,
    WritingReview,
    WritingSelection,
)

if TYPE_CHECKING:
    from .workflow import Runner


class _ContextDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    context: WritingContext
    domain_pin: ArtifactPin


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_compact(value).encode("utf-8")).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")


def _unit_ids(outline: Outline, draft: Draft) -> list[str]:
    return [unit.id for block in [*outline.blocks, *draft.blocks] for unit in block.units]


def _writing_semantic_hash(artifact: WritingArtifact) -> str:
    revision = artifact.revision.model_dump(
        mode="json", exclude={"created_at", "parent_version", "parent_manifest_hash"}
    )
    return _hash({
        "context": writing_context_payload(artifact.context),
        "domain_semantic_hash": artifact.domain_pin.semantic_hash,
        "outline": artifact.outline.model_dump(mode="json"),
        "draft": artifact.draft.model_dump(mode="json"),
        "review": artifact.review.model_dump(mode="json"),
        "revision": revision,
    })


def _render_blocks(blocks) -> list[str]:
    lines = []
    for block in blocks:
        text = "".join(unit.text for unit in block.units)
        audit_markers = "".join(f"[^{unit.id}]" for unit in block.units)
        if block.role == "heading":
            lines.extend(["#" * block.level + " " + text + audit_markers, ""])
        else:
            lines.extend([text + audit_markers, ""])
    return lines


def _source_locator(context: WritingContext, reference: SourceRef) -> str:
    work = context.research.works[reference.work_id]
    document = context.research.documents[reference.document_id]
    evidence = ", ".join(
        f"`{evidence_id}` (page {context.research.evidence[evidence_id].page}, "
        f"chars {context.research.evidence[evidence_id].start}:"
        f"{context.research.evidence[evidence_id].end})"
        for evidence_id in reference.evidence_ids
    )
    return (
        f"{_document_locator(work, document)}, claim `{reference.claim_id}`, "
        f"evidence {evidence}"
    )


def _document_locator(work, document) -> str:
    if document.source_path:
        source_path = quote(Path(document.source_path).as_posix(), safe="/")
        return f"[@{work.citekey}](../../../{source_path})"
    level = document.level.replace("_", "-")
    return f"[@{work.citekey}; {level} document {document.id}](context.json)"


def _statistic_source_locators(context: WritingContext) -> list[str]:
    if context.trend is None:
        return []
    work_ids = context.trend.computation.work_ids.get("eligible", [])
    locators = []
    for work_id in work_ids:
        work = context.research.works.get(work_id)
        document = next(
            (item for item in context.research.documents.values() if item.work_id == work_id),
            None,
        )
        if work is None or document is None:
            continue
        locators.append(_document_locator(work, document))
    return locators


def _unit_reference_audit(context: WritingContext, unit) -> str:
    details = []
    for reference in unit.references:
        if isinstance(reference, SourceRef):
            details.append("Source: " + _source_locator(context, reference))
        elif isinstance(reference, SynthesisRef):
            details.append(
                "Synthesis sources: "
                + "; ".join(_source_locator(context, source) for source in reference.sources)
            )
        elif isinstance(reference, StatisticRef):
            detail = (
                f"Frozen statistic `{reference.computation_path}` from trend pin "
                f"`{reference.pin_hash}`"
            )
            sources = _statistic_source_locators(context)
            if sources:
                detail += "; sources: " + ", ".join(sources)
            details.append(detail)
    return " | ".join(details)


def _report(artifact: WritingArtifact) -> bytes:
    review = {item.unit_id: item for item in artifact.review.items}
    units = {
        unit.id: unit
        for block in [*artifact.outline.blocks, *artifact.draft.blocks]
        for unit in block.units
    }
    lines = ["# Structured writing artifact", "", "## Outline", ""]
    lines.extend(_render_blocks(artifact.outline.blocks))
    if artifact.draft.blocks:
        lines.extend(["## Manuscript", ""])
        lines.extend(_render_blocks(artifact.draft.blocks))
    lines.extend(["## Independent review audit", ""])
    for unit_id in _unit_ids(artifact.outline, artifact.draft):
        unit = units[unit_id]
        item = review.get(unit_id)
        audit = _unit_reference_audit(artifact.context, unit)
        suffix = f" | {audit}" if audit else ""
        if item is None:
            lines.append(f"[^{unit_id}]: pending independent review{suffix}")
        else:
            lines.append(f"[^{unit_id}]: {item.decision} — {item.reason}{suffix}")
    lines.extend([
        "",
        "Independent model review is not human validation. Partial, contradicted, and insufficient decisions "
        "remain limitations that require source inspection or revision.",
        "",
        "Repair action: run `academic-agent writing-review <version>` after correcting or segmenting changed text.",
        "",
    ])
    return "\n".join(lines).encode("utf-8")


def _references(context: WritingContext) -> bytes:
    return export_bibtex(
        [context.research.works[work_id] for work_id in context.selection.work_ids]
    ).encode("utf-8")


def _review_input_hash(artifact: WritingArtifact) -> str:
    return _hash({
        "outline": artifact.outline.model_dump(mode="json"),
        "draft": artifact.draft.model_dump(mode="json"),
        "target_ids": _unit_ids(artifact.outline, artifact.draft),
    })


def _current_block_provenance(artifact: WritingArtifact, block_id: str) -> dict:
    backend, separator, model = artifact.revision.author.partition("/")
    is_model = artifact.revision.origin == "model"
    return {
        "block_id": block_id,
        "source": "current_revision",
        "origin": artifact.revision.origin,
        "author": artifact.revision.author,
        "created_at": artifact.revision.created_at,
        "parent_version": artifact.revision.parent_version,
        "parent_manifest_hash": artifact.revision.parent_manifest_hash,
        "source_version_path": None,
        "source_manifest_hash": None,
        "backend": backend if is_model and separator else None,
        "model_identifier": model if is_model and separator and model != "unavailable" else None,
    }


def _resolve_block_provenance(artifact: WritingArtifact,
                              stack: tuple[str, ...] = ()) -> list[dict]:
    if len(stack) > 64:
        raise ValueError("Writing block provenance chain is too deep")
    resolved = []
    parents: dict[str, _WritingVersionBundle] = {}
    for block in [*artifact.outline.blocks, *artifact.draft.blocks]:
        pin = artifact.revision.reused_from.get(block.id)
        if pin is None:
            resolved.append(_current_block_provenance(artifact, block.id))
            continue
        parent = parents.get(pin.version_path)
        if parent is None:
            parent = _load_writing_bundle(Path(pin.version_path), _provenance_stack=stack)
            parents[pin.version_path] = parent
        verified_pin = ArtifactPin(
            kind="writing",
            version_path=str(Path(pin.version_path).resolve()),
            manifest_hash=parent.manifest["manifest_hash"],
            semantic_hash=_writing_semantic_hash(parent.artifact),
        )
        if verified_pin != pin:
            raise ValueError("Reused writing provenance receipt is invalid")
        parent_blocks = {
            candidate.id: candidate
            for candidate in [*parent.artifact.outline.blocks, *parent.artifact.draft.blocks]
        }
        source_block = parent_blocks.get(block.id)
        if source_block is None or (
            source_block.role,
            source_block.level,
            "".join(unit.text for unit in source_block.units),
        ) != (
            block.role,
            block.level,
            "".join(unit.text for unit in block.units),
        ):
            raise ValueError("Reused writing block does not match its verified source")
        parent_provenance = {
            item["block_id"]: item for item in parent.block_provenance
        }
        if len(parent_provenance) != len(parent.block_provenance) or block.id not in parent_provenance:
            raise ValueError("Reused writing block provenance is incomplete")
        item = dict(parent_provenance[block.id])
        if item["source_version_path"] is None:
            item.update({
                "source": "reused_revision",
                "source_version_path": pin.version_path,
                "source_manifest_hash": pin.manifest_hash,
            })
        resolved.append(item)
    return resolved


def _evidence(artifact: WritingArtifact, review_provenance: dict | None = None,
              block_provenance: list[dict] | None = None) -> bytes:
    review = {item.unit_id: item.model_dump(mode="json") for item in artifact.review.items}
    units = []
    for extent, blocks in (("outline", artifact.outline.blocks), ("manuscript", artifact.draft.blocks)):
        for block in blocks:
            for unit in block.units:
                units.append({
                    **unit.model_dump(mode="json"),
                    "block_id": block.id,
                    "block_role": block.role,
                    "extent": extent,
                    "review": review.get(unit.id),
                })
    author_backend, _, author_model = artifact.revision.author.partition("/")
    unavailable = author_model == "unavailable"
    if artifact.revision.origin == "human":
        generation_provenance = {
            "author": artifact.revision.author,
            "backend": None,
            "model_identifier": None,
            "model_identifier_basis": "human",
            "observed_model_identifier": None,
            "observation_limitation": (
                "Human revision metadata identifies the importer; block_provenance records which blocks "
                "retain earlier model or human authorship."
            ),
        }
    else:
        generation_provenance = {
            "author": artifact.revision.author,
            "backend": author_backend,
            "model_identifier": None if unavailable else author_model,
            "model_identifier_basis": "unavailable" if unavailable else "configured_or_requested",
            "observed_model_identifier": None,
            "observation_limitation": (
                "The backend result does not expose an observed model identifier; revision.reason records "
                "whether the stored identifier was configured, requested, or unavailable."
            ),
        }
    if review_provenance is None:
        review_provenance = {
            "reviewer": artifact.revision.author if artifact.revision.stage == "reviewed" else None,
            "source": ("independent_model_call" if artifact.revision.stage == "reviewed"
                       else "pending_independent_model_call"),
            "review_input_hash": _review_input_hash(artifact),
            "created_at": artifact.revision.created_at,
        }
    if block_provenance is None:
        block_provenance = _resolve_block_provenance(artifact)
    payload = {
        "schema_version": 1,
        "document_id": artifact.revision.document_id,
        "domain_pin": artifact.domain_pin.model_dump(mode="json"),
        "units": units,
        "source_context": {
            "work_ids": artifact.context.selection.work_ids,
            "documents": {key: value.model_dump(mode="json")
                          for key, value in sorted(artifact.context.research.documents.items())},
            "evidence": {key: value.model_dump(mode="json")
                         for key, value in sorted(artifact.context.research.evidence.items())},
            "claims": {
                claim.id: claim.model_dump(mode="json")
                for work_id in artifact.context.selection.work_ids
                for claim in artifact.context.research.cards[work_id].claims
            },
            "decisions": {key: value.model_dump(mode="json")
                          for key, value in sorted(artifact.context.research.decisions.items())},
        },
        "generation_provenance": generation_provenance,
        "block_provenance": block_provenance,
        "review_provenance": review_provenance,
    }
    return _json_bytes(payload)


def render_writing(artifact: WritingArtifact) -> dict[str, bytes]:
    """Render the exact immutable writing payload set from structured content only."""
    try:
        artifact = WritingArtifact.model_validate(artifact.model_dump(mode="json"), strict=True)
        validate_structured_text(artifact.context, artifact.outline, artifact.draft)
    except (AttributeError, TypeError, ValidationError, WritingError):
        raise ValueError("Writing artifact failed complete host validation") from None
    if artifact.revision.origin != "model":
        raise ValueError("Human revision byte rendering is owned by the revision workflow")
    return {
        "context.json": _json_bytes({
            "context": artifact.context.model_dump(mode="json"),
            "domain_pin": artifact.domain_pin.model_dump(mode="json"),
        }),
        "outline.json": _json_bytes(artifact.outline.model_dump(mode="json")),
        "draft.json": _json_bytes(artifact.draft.model_dump(mode="json")),
        "review.json": _json_bytes(artifact.review.model_dump(mode="json")),
        "revision.json": _json_bytes(artifact.revision.model_dump(mode="json")),
        "raw-edit.bin": b"",
        "report.md": _report(artifact),
        "references.bib": _references(artifact.context),
        "evidence.json": _evidence(artifact),
    }


def _render_human_writing(artifact: WritingArtifact, raw_edit: bytes,
                          review_provenance: dict | None = None,
                          block_provenance: list[dict] | None = None) -> dict[str, bytes]:
    """Render one host-validated human artifact with its exact imported bytes."""
    try:
        artifact = WritingArtifact.model_validate(artifact.model_dump(mode="json"), strict=True)
        validate_structured_text(artifact.context, artifact.outline, artifact.draft)
    except (AttributeError, TypeError, ValidationError, WritingError):
        raise ValueError("Writing artifact failed complete host validation") from None
    if artifact.revision.origin != "human" or not isinstance(raw_edit, bytes):
        raise ValueError("Human writing rendering requires exact imported bytes")
    if hashlib.sha256(raw_edit).hexdigest() != artifact.revision.input_sha256:
        raise ValueError("Human raw edit bytes do not match the revision receipt")
    return {
        "context.json": _json_bytes({
            "context": artifact.context.model_dump(mode="json"),
            "domain_pin": artifact.domain_pin.model_dump(mode="json"),
        }),
        "outline.json": _json_bytes(artifact.outline.model_dump(mode="json")),
        "draft.json": _json_bytes(artifact.draft.model_dump(mode="json")),
        "review.json": _json_bytes(artifact.review.model_dump(mode="json")),
        "revision.json": _json_bytes(artifact.revision.model_dump(mode="json")),
        "raw-edit.bin": raw_edit,
        "report.md": _report(artifact),
        "references.bib": _references(artifact.context),
        "evidence.json": _evidence(artifact, review_provenance, block_provenance),
    }


def _parse_json(data: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError):
        raise ValueError(f"Writing {description} is malformed") from None
    if not isinstance(value, dict):
        raise ValueError(f"Writing {description} must be an object")
    return value


def _domain_matches_context(domain, context: WritingContext) -> bool:
    return (
        domain.context.selection.question == context.selection.question
        and domain.context.selection.work_ids == context.selection.work_ids
        and research_fingerprint(domain.context.research) == research_fingerprint(context.research)
        and domain.context.pack.source_path == context.pack.source_path
        and domain.context.pack == context.pack
    )


def _validate_generic_payload_closure(artifact: WritingArtifact, files: dict[str, bytes],
                                      review_provenance: dict | None = None,
                                      block_provenance: list[dict] | None = None) -> None:
    structural = {
        "context.json": _json_bytes({
            "context": artifact.context.model_dump(mode="json"),
            "domain_pin": artifact.domain_pin.model_dump(mode="json"),
        }),
        "outline.json": _json_bytes(artifact.outline.model_dump(mode="json")),
        "draft.json": _json_bytes(artifact.draft.model_dump(mode="json")),
        "review.json": _json_bytes(artifact.review.model_dump(mode="json")),
        "revision.json": _json_bytes(artifact.revision.model_dump(mode="json")),
        "report.md": _report(artifact),
        "references.bib": _references(artifact.context),
    }
    if any(files[name] != value for name, value in structural.items()):
        raise ValueError("Writing body, references, review, or report closure is invalid")
    try:
        actual_evidence = _parse_json(files["evidence.json"], "evidence")
        expected_evidence = json.loads(_evidence(
            artifact, review_provenance, block_provenance
        ))
    except (KeyError, ValueError, json.JSONDecodeError):
        raise ValueError("Writing evidence appendix is invalid") from None
    if actual_evidence != expected_evidence:
        raise ValueError("Writing evidence appendix does not close over the structured artifact")


@dataclass(frozen=True)
class _WritingVersionBundle:
    artifact: WritingArtifact
    raw_edit: bytes
    manifest: dict
    files: dict[str, bytes]
    review_provenance: dict
    block_provenance: list[dict]


def _human_review_provenance(artifact: WritingArtifact, files: dict[str, bytes]) -> dict:
    try:
        evidence = _parse_json(files["evidence.json"], "evidence")
        provenance = evidence["review_provenance"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Writing review provenance is invalid") from None
    if not isinstance(provenance, dict) or set(provenance) != {
            "reviewer", "source", "review_input_hash", "created_at"}:
        raise ValueError("Writing review provenance is invalid")
    expected_source = (
        "independent_model_call" if artifact.revision.stage == "reviewed"
        else "pending_independent_model_call"
    )
    reviewer = provenance["reviewer"]
    if (provenance["source"] != expected_source
            or provenance["review_input_hash"] != _review_input_hash(artifact)
            or not isinstance(provenance["created_at"], str)
            or not provenance["created_at"].strip()
            or (artifact.revision.stage == "pending" and reviewer is not None)
            or (artifact.revision.stage == "reviewed"
                and (not isinstance(reviewer, str) or not reviewer.strip()))):
        raise ValueError("Writing review provenance is invalid")
    return provenance


def _load_writing_bundle(directory: Path, *,
                         _provenance_stack: tuple[str, ...] = ()) -> _WritingVersionBundle:
    """Load artifact, raw bytes, evidence and receipt from one verified historical read."""
    logical_path = str(Path(directory).resolve())
    if logical_path in _provenance_stack or len(_provenance_stack) > 64:
        raise ValueError("Writing block provenance contains a cycle or excessive depth")
    provenance_stack = (*_provenance_stack, logical_path)
    loaded = load_research(Path(directory))
    manifest, files = loaded["manifest"], loaded["files"]
    if manifest["kind"] != "writing":
        raise ValueError("Research artifact is not a writing version")
    try:
        envelope = _ContextDocument.model_validate(
            _parse_json(files["context.json"], "context"), strict=True
        )
        artifact = WritingArtifact(
            context=envelope.context,
            domain_pin=envelope.domain_pin,
            outline=Outline.model_validate(_parse_json(files["outline.json"], "outline"), strict=True),
            draft=Draft.model_validate(_parse_json(files["draft.json"], "draft"), strict=True),
            review=WritingReview.model_validate(_parse_json(files["review.json"], "review"), strict=True),
            revision=Revision.model_validate(_parse_json(files["revision.json"], "revision"), strict=True),
        )
    except (KeyError, ValidationError, ValueError):
        raise ValueError("Writing artifact failed strict aggregate validation") from None
    if (_hash(writing_context_payload(artifact.context)) != artifact.context.input_hash
            or manifest["snapshot"] != artifact.context.research.model_dump(mode="json")):
        raise ValueError("Writing context does not close over its verified research manifest")
    try:
        domain = load_domain_version(Path(artifact.domain_pin.version_path))
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("Writing domain pin is no longer an intact historical artifact") from None
    if domain.pin != artifact.domain_pin:
        raise ValueError("Writing domain receipt does not match its verified artifact")
    if not _domain_matches_context(domain, artifact.context):
        raise ValueError("Writing domain artifact does not match its captured semantic selection")
    if artifact.revision.origin == "human":
        review_provenance = _human_review_provenance(artifact, files)
    else:
        try:
            review_provenance = _parse_json(files["evidence.json"], "evidence")[
                "review_provenance"
            ]
        except (KeyError, TypeError, ValueError):
            raise ValueError("Writing review provenance is invalid") from None
    try:
        evidence = _parse_json(files["evidence.json"], "evidence")
        block_provenance = evidence["block_provenance"]
    except (KeyError, TypeError, ValueError):
        raise ValueError("Writing block provenance is invalid") from None
    expected_block_provenance = _resolve_block_provenance(artifact, provenance_stack)
    if block_provenance != expected_block_provenance:
        raise ValueError("Writing block provenance does not match verified history")
    _validate_generic_payload_closure(
        artifact, files, review_provenance, expected_block_provenance
    )
    if artifact.revision.origin == "model":
        if render_writing(artifact) != files:
            raise ValueError("Writing host payload closure is invalid")
    else:
        raw = files["raw-edit.bin"]
        if hashlib.sha256(raw).hexdigest() != artifact.revision.input_sha256:
            raise ValueError("Writing raw edit bytes do not match the revision receipt")
        validate_structured_text(artifact.context, artifact.outline, artifact.draft)
        if _render_human_writing(
                artifact, raw, review_provenance, expected_block_provenance) != files:
            raise ValueError("Writing human host payload closure is invalid")
    return _WritingVersionBundle(
        artifact=artifact,
        raw_edit=files["raw-edit.bin"],
        manifest=manifest,
        files=files,
        review_provenance=review_provenance,
        block_provenance=expected_block_provenance,
    )


def load_writing_version(directory: Path) -> WritingArtifact:
    """Load one intact historical writing artifact without checking today's source freshness."""
    return _load_writing_bundle(directory).artifact


def writing_dependencies_current(root: Path, context: WritingContext, domain_pin: ArtifactPin) -> dict:
    """Check the full saved base context and exact saved domain artifact and selection."""
    root = Path(root)
    base = writing_currentness(root, context)
    gaps = list(base["gaps"])
    try:
        domain_pin = ArtifactPin.model_validate(domain_pin.model_dump(mode="json"), strict=True)
        if domain_pin.kind != "domain":
            raise ValueError("Not a domain pin")
        domain = load_domain_version(Path(domain_pin.version_path))
    except (AttributeError, OSError, ValueError, KeyError, TypeError, ValidationError):
        gaps.append({"reason": "domain_artifact_invalid"})
        unique = []
        for gap in gaps:
            if gap not in unique:
                unique.append(gap)
        return {"current": False, "gaps": unique}
    if domain.pin != domain_pin:
        gaps.append({"reason": "domain_artifact_invalid"})
    try:
        selection_matches = _domain_matches_context(domain, context)
        domain_context_payload(domain)
    except (ValueError, KeyError, TypeError):
        gaps.append({"reason": "domain_artifact_invalid"})
        selection_matches = True
    if not selection_matches:
        gaps.append({"reason": "domain_selection_invalid"})
    unique = []
    for gap in gaps:
        if gap not in unique:
            unique.append(gap)
    return {"current": not unique, "gaps": unique}


def assert_writing_dependencies_current(root: Path, context: WritingContext,
                                        domain_pin: ArtifactPin) -> None:
    state = writing_dependencies_current(root, context, domain_pin)
    if state["current"]:
        return
    status = "failed" if any(gap["reason"].endswith("invalid") for gap in state["gaps"]) else "stale"
    raise WritingError("Writing and saved domain dependencies are no longer current", status=status)


def _progress(runner: Runner, document_id: str, input_hash: str, stage: str,
              status: str, *, output: str | None = None, error: str | None = None) -> None:
    runner.store.log("writing_progress", {
        "document_id": document_id,
        "input_hash": input_hash,
        "stage": stage,
        "status": status,
        "output": output,
        "error": error,
    })


def _result(status: str, input_hash: str, output: str | None, gaps: list[dict]) -> dict[str, Any]:
    return {"status": status, "input_hash": input_hash, "output": output, "gaps": gaps}


def _model_author(runner: Runner, phase: str) -> tuple[str, str]:
    backend_name = runner.backend.name
    configured = runner.config.model
    requested = getattr(runner.backend.backend, "model", None)
    if configured is not None:
        identifier, basis = configured, "configured"
    elif isinstance(requested, str) and requested:
        identifier, basis = requested, "requested"
    else:
        identifier, basis = "unavailable", "unavailable"
    author = f"{backend_name}/{identifier}"
    reason = (
        f"initial {phase} generation; model identifier basis={basis}; "
        "observed model identifier unavailable from backend result"
    )
    return author, reason


def _domain_stage_hash(context: WritingContext, domain) -> str:
    return _hash({
        "task": "writing-outline-v1",
        "context": writing_context_payload(context),
        "domain": domain_context_payload(domain),
    })


def _operation_failure(runner: Runner, document_id: str, input_hash: str, stage: str,
                       exc: Exception, output_path: Path | None = None) -> dict:
    if isinstance(exc, BudgetExceeded):
        status = "paused_budget"
    elif isinstance(exc, WritingError):
        status = exc.status
        output_path = exc.output or output_path
    else:
        status = "failed"
    output = str(output_path.relative_to(runner.root)) if output_path else None
    _progress(runner, document_id, input_hash, stage, status, output=output, error=str(exc))
    guidance = {
        "paused_budget": "Increase the explicit model-call budget, then rerun to reuse validated cached work.",
        "needs_input": "Complete the requested writing review or manual Exchange response, then retry.",
        "input_too_large": "Choose fewer work_ids and retry; no selected content was truncated.",
        "stale": "Recapture the changed base or domain dependency before writing again.",
        "failed": "Inspect the invalid dependency, structured model output, or publication error before retrying.",
    }[status]
    return _result(status, input_hash, output, [{"reason": stage, "guidance": guidance}])


def create_outline(runner: Runner, selection: WritingSelection, domain_version: Path) -> dict:
    """Generate and independently review a complete outline before immutable publication."""
    input_hash, stage, published = "0" * 64, "context", None
    document_id = getattr(selection, "document_id", "doc_" + "0" * 32)
    try:
        selection = WritingSelection.model_validate(selection.model_dump(mode="json"), strict=True)
        document_id = selection.document_id
        domain = load_domain_version(Path(domain_version))
        base_context = domain.context
        assert_writing_dependencies_current(runner.root, base_context, domain.pin)
        if (
            selection.document_id != base_context.selection.document_id
            or selection.question != base_context.selection.question
            or selection.work_ids != base_context.selection.work_ids
            or selection.domain_path != base_context.selection.domain_path
        ):
            raise WritingError("Writing selection does not match the saved domain selection", status="failed")
        context = (
            base_context
            if selection == base_context.selection
            else load_writing_context(runner.root, selection)
        )
        if not _domain_matches_context(domain, context):
            raise WritingError(
                "Explicit writing dependencies do not match the saved base domain",
                status="failed",
            )
        assert_writing_dependencies_current(runner.root, context, domain.pin)
        input_hash = _domain_stage_hash(context, domain)
        _progress(runner, document_id, input_hash, "identity", "complete")
        assert_writing_dependencies_current(runner.root, context, domain.pin)

        stage = "writing-outline-v1"
        assert_writing_dependencies_current(runner.root, context, domain.pin)
        with runner.backend.validated() as backend:
            outline = generate_outline(
                context, domain, backend,
                timeout_seconds=runner.config.timeout_seconds,
                max_input_chars=runner.config.max_input_chars,
            )
            assert_writing_dependencies_current(runner.root, context, domain.pin)
        _progress(runner, document_id, input_hash, stage, "complete")

        stage = "writing-review-v1:outline"
        targets = _unit_ids(outline, Draft(blocks=[]))
        assert_writing_dependencies_current(runner.root, context, domain.pin)
        with runner.backend.validated() as backend:
            review = review_text(
                context, domain, outline, Draft(blocks=[]), targets, backend,
                timeout_seconds=runner.config.timeout_seconds,
                max_input_chars=runner.config.max_input_chars,
            )
            assert_writing_dependencies_current(runner.root, context, domain.pin)
        _progress(runner, document_id, input_hash, stage, "complete")

        author, reason = _model_author(runner, "outline")
        artifact = WritingArtifact(
            context=context,
            domain_pin=domain.pin,
            outline=outline,
            draft=Draft(blocks=[]),
            review=review,
            revision=Revision(
                schema_version=1,
                document_id=document_id,
                extent="outline",
                stage="reviewed",
                parent_version=None,
                parent_manifest_hash=None,
                origin="model",
                author=author,
                reason=reason,
                created_at=utcnow(),
                input_sha256=None,
                changed_ids=[],
                added_ids=[],
                deleted_ids=[],
                reordered=False,
                id_map={},
                pending_ids=[],
                reused_from={},
                domain_rebind=None,
            ),
        )
        stage = "publish:outline"
        assert_writing_dependencies_current(runner.root, context, domain.pin)
        published = publish_research(
            runner.store,
            "writing",
            context.research,
            render_writing(artifact),
            dependency_check=lambda: assert_writing_dependencies_current(
                runner.root, context, domain.pin
            ),
        )
        assert_writing_dependencies_current(runner.root, context, domain.pin)
        output = str(published.relative_to(runner.root))
        _progress(runner, document_id, input_hash, stage, "complete", output=output)
        return _result("complete", input_hash, output, [])
    except Exception as exc:
        return _operation_failure(runner, document_id, input_hash, stage, exc, published)


def _writing_pin(directory: Path, artifact: WritingArtifact) -> ArtifactPin:
    loaded = load_research(directory)
    return ArtifactPin(
        kind="writing",
        version_path=str(Path(directory).resolve()),
        manifest_hash=loaded["manifest"]["manifest_hash"],
        semantic_hash=_writing_semantic_hash(artifact),
    )


def create_draft(runner: Runner, outline_version: Path) -> dict:
    """Generate a manuscript from one exact reviewed outline and review the complete result."""
    input_hash, stage, published = "0" * 64, "load-outline", None
    document_id = "doc_" + "0" * 32
    try:
        outline_path = Path(outline_version).resolve()
        writing_directory = (runner.root / "analyses" / "writing").resolve()
        if (outline_path.parent != writing_directory
                or re.fullmatch(r"v-[0-9a-f]{32}", outline_path.name) is None):
            raise WritingError(
                "Draft creation requires an explicit immutable writing version",
                status="failed",
            )
        parent = load_writing_version(outline_path)
        document_id = parent.context.selection.document_id
        input_hash = _hash({
            "task": "writing-draft-v1",
            "context": writing_context_payload(parent.context),
            "domain_semantic_hash": parent.domain_pin.semantic_hash,
            "outline": parent.outline.model_dump(mode="json"),
            "outline_review": parent.review.model_dump(mode="json"),
        })
        if parent.revision.extent != "outline":
            raise WritingError("Draft creation requires a reviewed outline-only version", status="failed")
        if parent.revision.stage != "reviewed":
            if parent.revision.stage == "pending":
                gap = {
                    "reason": "outline_review_required",
                    "argv": ["academic-agent", "writing-review", str(outline_path)],
                }
                _progress(runner, document_id, input_hash, stage, "needs_input",
                          error="Outline requires review before drafting")
                return _result("needs_input", input_hash, None, [gap])
            raise WritingError("Draft creation requires a reviewed outline-only version", status="failed")
        assert_writing_dependencies_current(runner.root, parent.context, parent.domain_pin)
        domain = load_domain_version(Path(parent.domain_pin.version_path))
        state = OutlineReviewState(status="complete", review=parent.review)

        stage = "writing-draft-v1"
        assert_writing_dependencies_current(runner.root, parent.context, parent.domain_pin)
        with runner.backend.validated() as backend:
            draft = generate_draft(
                parent.context, domain, parent.outline, state, backend,
                timeout_seconds=runner.config.timeout_seconds,
                max_input_chars=runner.config.max_input_chars,
            )
            assert_writing_dependencies_current(runner.root, parent.context, parent.domain_pin)
        _progress(runner, document_id, input_hash, stage, "complete")

        stage = "writing-review-v1:manuscript"
        targets = _unit_ids(parent.outline, draft)
        assert_writing_dependencies_current(runner.root, parent.context, parent.domain_pin)
        with runner.backend.validated() as backend:
            review = review_text(
                parent.context, domain, parent.outline, draft, targets, backend,
                timeout_seconds=runner.config.timeout_seconds,
                max_input_chars=runner.config.max_input_chars,
            )
            assert_writing_dependencies_current(runner.root, parent.context, parent.domain_pin)
        _progress(runner, document_id, input_hash, stage, "complete")

        parent_pin = _writing_pin(outline_path, parent)
        author, reason = _model_author(runner, "manuscript")
        artifact = WritingArtifact(
            context=parent.context,
            domain_pin=parent.domain_pin,
            outline=parent.outline,
            draft=draft,
            review=review,
            revision=Revision(
                schema_version=1,
                document_id=document_id,
                extent="manuscript",
                stage="reviewed",
                parent_version=outline_path.name,
                parent_manifest_hash=parent_pin.manifest_hash,
                origin="model",
                author=author,
                reason=reason,
                created_at=utcnow(),
                input_sha256=None,
                changed_ids=[],
                added_ids=[],
                deleted_ids=[],
                reordered=False,
                id_map={},
                pending_ids=[],
                reused_from={block.id: parent_pin for block in parent.outline.blocks},
                domain_rebind=None,
            ),
        )
        stage = "publish:manuscript"
        assert_writing_dependencies_current(runner.root, parent.context, parent.domain_pin)
        published = publish_research(
            runner.store,
            "writing",
            parent.context.research,
            render_writing(artifact),
            dependency_check=lambda: assert_writing_dependencies_current(
                runner.root, parent.context, parent.domain_pin
            ),
        )
        assert_writing_dependencies_current(runner.root, parent.context, parent.domain_pin)
        output = str(published.relative_to(runner.root))
        _progress(runner, document_id, input_hash, stage, "complete", output=output)
        return _result("complete", input_hash, output, [])
    except Exception as exc:
        return _operation_failure(runner, document_id, input_hash, stage, exc, published)


def _document_id_from_invalid(directory: Path) -> str | None:
    try:
        loaded = load_research(directory)
        value = _parse_json(loaded["files"]["revision.json"], "revision")
        document_id = value.get("document_id")
        return document_id if isinstance(document_id, str) else None
    except Exception:
        return None


def writing_status(root: Path, document_id: str | None = None) -> dict:
    """Inspect writing history without constructing workflow or storage objects."""
    root = Path(root).resolve()
    base = root / "analyses" / "writing"
    if not io_path(base).is_dir():
        return {"status": "needs_input", "documents": [], "gaps": [{"reason": "no_writing_versions"}]}
    documents: dict[str, list[dict]] = {}
    global_gaps = []
    directories = [
        base / entry.name
        for entry in io_path(base).iterdir()
        if entry.is_dir() and entry.name.startswith("v-")
    ]
    for directory in sorted(directories):
        try:
            artifact = load_writing_version(directory)
            currentness = writing_dependencies_current(root, artifact.context, artifact.domain_pin)
            item = {
                "version": directory.name,
                "available": True,
                "current": currentness["current"],
                "extent": artifact.revision.extent,
                "stage": artifact.revision.stage,
                "review_complete": artifact.revision.stage == "reviewed",
                "gaps": currentness["gaps"],
                "output": str(directory.relative_to(root)),
            }
            key = artifact.revision.document_id
        except Exception:
            key = _document_id_from_invalid(directory) or "unknown"
            item = {
                "version": directory.name,
                "available": False,
                "current": False,
                "extent": None,
                "stage": None,
                "review_complete": False,
                "gaps": [{"reason": "writing_artifact_invalid"}],
                "output": str(directory.relative_to(root)),
            }
        if document_id is None or key == document_id:
            documents.setdefault(key, []).append(item)
            global_gaps.extend(item["gaps"])
    values = [{"document_id": key, "versions": versions} for key, versions in sorted(documents.items())]
    if not values:
        status = "needs_input"
    elif any(version["available"] and version["current"] for value in values for version in value["versions"]):
        status = "complete"
    elif any(not version["available"] for value in values for version in value["versions"]):
        status = "failed"
    else:
        status = "stale"
    unique_gaps = []
    for gap in global_gaps:
        if gap not in unique_gaps:
            unique_gaps.append(gap)
    return {"status": status, "documents": values, "gaps": unique_gaps}
