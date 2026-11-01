"""Exact, immutable human writing import, segmentation, and independent review."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .domain import load_domain_version
from .filesystem import io_path
from .models import utcnow
from .research_artifacts import publish_research
from .research_context import research_fingerprint
from .writing import (
    _hash,
    _load_writing_bundle,
    _model_author,
    _operation_failure,
    _progress,
    _render_human_writing,
    _result,
    _review_input_hash,
    _writing_semantic_hash,
    _WritingVersionBundle,
    assert_writing_dependencies_current,
)
from .writing_analysis import review_text, segment_text, validate_structured_text
from .writing_context import load_writing_context, writing_context_payload
from .writing_models import (
    ArtifactPin,
    DomainArtifact,
    DomainRebind,
    Draft,
    EditableBlock,
    EditableWriting,
    Outline,
    Revision,
    TextBlock,
    TextUnit,
    WritingArtifact,
    WritingContext,
    WritingError,
    WritingReview,
)

_BLOCK_ID = re.compile(r"b_[0-9a-f]{32}\Z")
_BRACKET_CITATION = re.compile(r"\[([^\]]*@[A-Za-z0-9_-]+[^\]]*)\]")
_CITATION_KEY = re.compile(r"@([A-Za-z0-9_-]+)")
_TEX_CITATION = re.compile(r"\\cite\{([^{}]+)\}")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _writing_pin(path: Path, bundle: _WritingVersionBundle) -> ArtifactPin:
    return ArtifactPin(
        kind="writing",
        version_path=str(Path(path).resolve()),
        manifest_hash=bundle.manifest["manifest_hash"],
        semantic_hash=_writing_semantic_hash(bundle.artifact),
    )


def _concrete_writing_path(root: Path, version: Path) -> Path:
    root = Path(root).resolve()
    path = Path(version).resolve()
    if (path.parent != (root / "analyses" / "writing").resolve()
            or re.fullmatch(r"v-[0-9a-f]{32}", path.name) is None):
        raise WritingError("A concrete immutable writing version is required", status="failed")
    return path


def _plain_blocks(blocks: list[TextBlock]) -> list[dict[str, Any]]:
    return [
        {
            "id": block.id,
            "role": block.role,
            "level": block.level,
            "text": "".join(unit.text for unit in block.units),
        }
        for block in blocks
    ]


def export_edit(root: Path, version: Path, destination: Path) -> Path:
    """Export one intact historical version to an exclusive-create editable JSON copy."""
    parent_path = _concrete_writing_path(root, version)
    bundle = _load_writing_bundle(parent_path)
    artifact = bundle.artifact
    editable = {
        "schema_version": 1,
        "document_id": artifact.revision.document_id,
        "parent_version": parent_path.name,
        "parent_manifest_hash": bundle.manifest["manifest_hash"],
        "outline": _plain_blocks(artifact.outline.blocks),
        "paragraphs": _plain_blocks(artifact.draft.blocks),
    }
    destination = Path(destination).resolve()
    encoded = json.dumps(
        editable, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8")
    with io_path(destination).open("xb") as stream:
        stream.write(encoded)
    return destination


def _reject_constant(value: str):
    raise ValueError(f"Non-finite JSON value {value} is forbidden")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_edit(path: Path) -> tuple[bytes, EditableWriting]:
    try:
        raw = io_path(path).read_bytes()
        payload = raw[3:] if raw.startswith(b"\xef\xbb\xbf") else raw
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        document = EditableWriting.model_validate(value, strict=True)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise WritingError(f"Editable writing JSON is invalid: {exc}", status="failed") from None
    return raw, document


def _validate_author_reason(author: str, reason: str) -> tuple[str, str]:
    if (not isinstance(author, str) or not author.strip() or author != author.strip()
            or not isinstance(reason, str) or not reason.strip() or reason != reason.strip()
            or any(ord(character) < 32 and character not in "\t\r\n" for character in author + reason)):
        raise WritingError("Human revision author and reason must be nonblank exact text", status="failed")
    return author, reason


def _validate_citations(document: EditableWriting, artifact: WritingArtifact) -> None:
    admitted = {
        artifact.context.research.works[work_id].citekey
        for work_id in artifact.context.selection.work_ids
    }
    for block in [*document.outline, *document.paragraphs]:
        supplied = {
            key
            for citation in _BRACKET_CITATION.findall(block.text)
            for key in _CITATION_KEY.findall(citation)
        }
        supplied.update(
            key.strip() for match in _TEX_CITATION.findall(block.text) for key in match.split(",")
        )
        if not supplied <= admitted:
            raise WritingError("Editable text contains an unknown citation token", status="failed")


def _rebind_domain(root: Path, parent: WritingArtifact,
                   domain_version: Path | None) -> tuple[WritingContext, DomainArtifact, DomainRebind | None]:
    if domain_version is None:
        assert_writing_dependencies_current(root, parent.context, parent.domain_pin)
        domain = load_domain_version(Path(parent.domain_pin.version_path))
        return parent.context, domain, None
    try:
        domain = load_domain_version(Path(domain_version))
    except (OSError, ValueError, KeyError, TypeError):
        raise WritingError("Replacement domain version is not an intact artifact", status="failed") from None
    replacement = domain.context
    old = parent.context
    if (replacement.selection.question != old.selection.question
            or replacement.selection.work_ids != old.selection.work_ids
            or replacement.pack.source_path != old.pack.source_path
            or replacement.pack.pack.name != old.pack.pack.name
            or replacement.pack.pack.contract != old.pack.pack.contract):
        raise WritingError(
            "Replacement domain must match the exact question, ordered works, and pack lineage",
            status="failed",
        )
    assert_writing_dependencies_current(root, replacement, domain.pin)
    try:
        recaptured = load_writing_context(root, old.selection)
    except WritingError:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        raise WritingError(
            "The parent's explicit writing selections could not be recaptured",
            status="failed",
        ) from None
    if (recaptured.selection != old.selection
            or research_fingerprint(recaptured.research)
            != research_fingerprint(replacement.research)
            or recaptured.pack != replacement.pack):
        raise WritingError(
            "Replacement domain does not match the recaptured research and current pack bytes",
            status="failed",
        )
    assert_writing_dependencies_current(root, recaptured, domain.pin)
    return recaptured, domain, DomainRebind(old=parent.domain_pin, new=domain.pin)


def _stale_rebind_result(parent_path: Path, edit_file: Path, parent: WritingArtifact,
                         author: str, reason: str, delete_ids: list[str]) -> dict:
    selection = parent.context.selection
    extract_argv = ["academic-agent", "domain-extract", "--question", selection.question]
    for work_id in selection.work_ids:
        extract_argv.extend(["--work-id", work_id])
    extract_argv.extend(["--domain", selection.domain_path])
    reimport_argv = [
        "academic-agent", "writing-import", str(parent_path), str(Path(edit_file).resolve()),
        "--author", author, "--reason", reason,
    ]
    for block_id in delete_ids:
        reimport_argv.extend(["--delete-id", block_id])
    reimport_argv.extend(["--domain-version", "<DOMAIN_VERSION_FROM_DOMAIN_EXTRACT_OUTPUT>"])
    return _result("stale", parent.context.input_hash, None, [{
        "reason": "domain_rebind_required",
        "question": selection.question,
        "work_ids": selection.work_ids,
        "domain_path": selection.domain_path,
        "extract_argv": extract_argv,
        "reimport_argv": reimport_argv,
    }])


def _unclassified(block_id: str, text: str, input_sha256: str) -> list[TextUnit]:
    unit_id = "u_" + _sha(["unclassified-1", block_id, input_sha256])[:32]
    return [TextUnit(
        id=unit_id,
        text=text,
        kind="unclassified",
        references=[],
        purpose="pending segmentation",
    )]


def _block_from_edit(block: EditableBlock, block_id: str, units: list[TextUnit]) -> TextBlock:
    return TextBlock(id=block_id, role=block.role, level=block.level, units=units)


def import_revision(runner, parent_version: Path, edit_file: Path, *, author: str, reason: str,
                    delete_ids: list[str], domain_version: Path | None = None) -> dict:
    """Validate and publish exact human bytes as a new immutable revision without model calls."""
    author, reason = _validate_author_reason(author, reason)
    parent_path = _concrete_writing_path(runner.root, parent_version)
    bundle = _load_writing_bundle(parent_path)
    parent = bundle.artifact
    raw, document = _load_edit(Path(edit_file))
    input_sha256 = hashlib.sha256(raw).hexdigest()
    if (document.document_id != parent.revision.document_id
            or document.parent_version != parent_path.name
            or document.parent_manifest_hash != bundle.manifest["manifest_hash"]):
        raise WritingError("Editable document does not match the exact parent receipt", status="failed")
    if (not isinstance(delete_ids, list) or len(delete_ids) != len(set(delete_ids))
            or any(not isinstance(value, str) or not _BLOCK_ID.fullmatch(value)
                   for value in delete_ids)):
        raise WritingError("Explicit delete IDs must be unique parent block IDs", status="failed")
    if parent.revision.extent == "outline" and document.paragraphs:
        raise WritingError("An outline revision cannot add a manuscript body", status="failed")
    if parent.revision.extent == "manuscript" and not document.paragraphs:
        raise WritingError("A manuscript revision cannot silently become an outline", status="failed")
    _validate_citations(document, parent)

    parent_outline = {block.id: block for block in parent.outline.blocks}
    parent_body = {block.id: block for block in parent.draft.blocks}
    outline_old_ids = set(parent_outline)
    body_old_ids = set(parent_body)
    outline_ids = {block.id for block in document.outline if _BLOCK_ID.fullmatch(block.id)}
    body_ids = {block.id for block in document.paragraphs if _BLOCK_ID.fullmatch(block.id)}
    if not outline_ids <= outline_old_ids or not body_ids <= body_old_ids:
        raise WritingError("Existing editable IDs must stay in the exact parent array", status="failed")
    missing = (outline_old_ids | body_old_ids) - outline_ids - body_ids
    if set(delete_ids) != missing:
        raise WritingError("Delete IDs must exactly acknowledge every omitted parent block", status="failed")

    try:
        context, domain, domain_rebind = _rebind_domain(runner.root, parent, domain_version)
    except WritingError as exc:
        if domain_version is None and exc.status == "stale":
            _progress(
                runner, parent.revision.document_id, parent.context.input_hash,
                "writing-import-v1", "stale", error=str(exc),
            )
            return _stale_rebind_result(parent_path, Path(edit_file), parent, author, reason, delete_ids)
        raise
    context_rebind = writing_context_payload(context) != writing_context_payload(parent.context)
    semantic_rebind = (
        domain.pin.semantic_hash != parent.domain_pin.semantic_hash or context_rebind
    )
    parent_pin = _writing_pin(parent_path, bundle)

    markers = [
        block.id for block in [*document.outline, *document.paragraphs] if block.id.startswith("new:")
    ]
    id_map = {
        marker: "b_" + _sha([
            "writing-edit-ids-1", parent.revision.document_id,
            bundle.manifest["manifest_hash"], input_sha256, marker,
        ])[:32]
        for marker in markers
    }
    if len(set(id_map.values())) != len(id_map):
        raise WritingError("Stable added block IDs collided", status="failed")

    def resolved_id(block: EditableBlock) -> str:
        return id_map.get(block.id, block.id)

    def metadata(block: TextBlock) -> tuple[str, int | None, str]:
        return block.role, block.level, "".join(unit.text for unit in block.units)

    changed_ids = []
    for editable in [*document.outline, *document.paragraphs]:
        if editable.id.startswith("new:"):
            continue
        original = parent_outline.get(editable.id) or parent_body.get(editable.id)
        if original is None or metadata(original) != (editable.role, editable.level, editable.text):
            changed_ids.append(editable.id)
    added_ids = [id_map[marker] for marker in markers]
    old_outline_order = [block.id for block in parent.outline.blocks if block.id not in missing]
    old_body_order = [block.id for block in parent.draft.blocks if block.id not in missing]
    new_outline_order = [block.id for block in document.outline if _BLOCK_ID.fullmatch(block.id)]
    new_body_order = [block.id for block in document.paragraphs if _BLOCK_ID.fullmatch(block.id)]
    reordered = old_outline_order != new_outline_order or old_body_order != new_body_order
    outline_content_changed = (
        any(block_id in outline_old_ids for block_id in [*changed_ids, *delete_ids])
        or any(block.id.startswith("new:") for block in document.outline)
    )
    semantic_change = bool(changed_ids or added_ids or delete_ids or reordered or semantic_rebind)
    force_all = semantic_rebind or outline_content_changed
    parent_pending = parent.revision.stage == "pending"
    parent_pending_ids = set(parent.revision.pending_ids)
    pending_ids = []
    reused_from = {}

    def build(editable: EditableBlock, original: TextBlock | None) -> TextBlock:
        block_id = resolved_id(editable)
        content_unchanged = original is not None and metadata(original) == (
            editable.role, editable.level, editable.text
        )
        if content_unchanged:
            reused_from[block_id] = parent_pin
        needs_segment = (
            force_all
            or original is None
            or block_id in changed_ids
            or block_id in parent_pending_ids
        )
        if needs_segment:
            pending_ids.append(block_id)
            return _block_from_edit(editable, block_id, _unclassified(block_id, editable.text, input_sha256))
        return _block_from_edit(editable, block_id, list(original.units))

    outline = Outline(blocks=[build(block, parent_outline.get(block.id)) for block in document.outline])
    draft = Draft(blocks=[build(block, parent_body.get(block.id)) for block in document.paragraphs])
    stage = "pending" if semantic_change or parent_pending else "reviewed"
    review = WritingReview(items=[]) if stage == "pending" else parent.review
    revision = Revision(
        schema_version=1,
        document_id=parent.revision.document_id,
        extent=parent.revision.extent,
        stage=stage,
        parent_version=parent_path.name,
        parent_manifest_hash=bundle.manifest["manifest_hash"],
        origin="human",
        author=author,
        reason=reason,
        created_at=utcnow(),
        input_sha256=input_sha256,
        changed_ids=changed_ids,
        added_ids=added_ids,
        deleted_ids=delete_ids,
        reordered=reordered,
        id_map=id_map,
        pending_ids=pending_ids,
        reused_from=reused_from,
        domain_rebind=domain_rebind,
    )
    artifact = WritingArtifact(
        context=context,
        domain_pin=domain.pin,
        outline=outline,
        draft=draft,
        review=review,
        revision=revision,
    )
    validate_structured_text(context, outline, draft)
    provenance = bundle.review_provenance if stage == "reviewed" else None
    input_hash = _hash({
        "task": "writing-import-v1",
        "parent_manifest_hash": bundle.manifest["manifest_hash"],
        "input_sha256": input_sha256,
        "domain_pin": domain.pin.model_dump(mode="json"),
    })
    published = None
    try:
        assert_writing_dependencies_current(runner.root, context, domain.pin)
        published = publish_research(
            runner.store,
            "writing",
            context.research,
            _render_human_writing(artifact, raw, provenance),
            dependency_check=lambda: assert_writing_dependencies_current(
                runner.root, context, domain.pin
            ),
        )
        assert_writing_dependencies_current(runner.root, context, domain.pin)
        output = str(published.relative_to(runner.root))
        _progress(runner, parent.revision.document_id, input_hash, "writing-import-v1", "complete",
                  output=output)
        return _result("complete", input_hash, output, [])
    except Exception as exc:
        return _operation_failure(
            runner, parent.revision.document_id, input_hash, "writing-import-v1", exc, published
        )


def segment_revision(context: WritingContext, domain: DomainArtifact, artifact: WritingArtifact,
                     target_ids: list[str], backend, *, timeout_seconds: int,
                     max_input_chars: int) -> list[TextBlock]:
    """Segment only pending blocks while slicing all unit text from the submitted original."""
    try:
        artifact = WritingArtifact.model_validate(artifact.model_dump(mode="json"), strict=True)
    except (AttributeError, TypeError, ValidationError):
        raise WritingError("Pending writing artifact is invalid", status="failed") from None
    if artifact.context != context or artifact.domain_pin != domain.pin:
        raise WritingError("Segmentation context or domain does not match the revision", status="failed")
    if target_ids != artifact.revision.pending_ids:
        raise WritingError("Segmentation targets must equal the exact pending block list", status="failed")
    blocks = [*artifact.outline.blocks, *artifact.draft.blocks]
    by_id = {block.id: block for block in blocks}
    raw_blocks = []
    for block_id in target_ids:
        block = by_id.get(block_id)
        if (block is None or len(block.units) != 1
                or block.units[0].kind != "unclassified"):
            raise WritingError("Every segmentation target must be exact unclassified text", status="failed")
        raw_blocks.append({
            "id": block.id,
            "role": block.role,
            "level": block.level,
            "text": block.units[0].text,
        })
    return segment_text(
        context, domain, artifact.outline, raw_blocks, target_ids, backend,
        timeout_seconds=timeout_seconds, max_input_chars=max_input_chars,
    )


def review_writing(runner, version: Path) -> dict:
    """Segment pending human text and independently review the complete ordered artifact."""
    input_hash, stage, published = "0" * 64, "load-revision", None
    document_id = "doc_" + "0" * 32
    try:
        version_path = _concrete_writing_path(runner.root, version)
        bundle = _load_writing_bundle(version_path)
        artifact = bundle.artifact
        document_id = artifact.revision.document_id
        input_hash = _hash({
            "task": "writing-review-revision-v1",
            "context": artifact.context.model_dump(mode="json"),
            "domain_pin": artifact.domain_pin.model_dump(mode="json"),
            "outline": artifact.outline.model_dump(mode="json"),
            "draft": artifact.draft.model_dump(mode="json"),
            "input_sha256": artifact.revision.input_sha256,
        })
        if artifact.revision.origin != "human":
            raise WritingError("Revision review requires a human writing version", status="failed")
        if artifact.revision.stage == "reviewed":
            assert_writing_dependencies_current(runner.root, artifact.context, artifact.domain_pin)
            return _result("complete", input_hash, str(version_path.relative_to(runner.root)), [])
        domain = load_domain_version(Path(artifact.domain_pin.version_path))
        assert_writing_dependencies_current(runner.root, artifact.context, artifact.domain_pin)

        replacements = {}
        if artifact.revision.pending_ids:
            stage = "writing-segment-v1"
            assert_writing_dependencies_current(runner.root, artifact.context, artifact.domain_pin)
            with runner.backend.validated() as backend:
                segmented = segment_revision(
                    artifact.context, domain, artifact, artifact.revision.pending_ids, backend,
                    timeout_seconds=runner.config.timeout_seconds,
                    max_input_chars=runner.config.max_input_chars,
                )
                assert_writing_dependencies_current(runner.root, artifact.context, artifact.domain_pin)
            replacements = {block.id: block for block in segmented}
            _progress(runner, document_id, input_hash, stage, "complete")
        outline = Outline(blocks=[replacements.get(block.id, block) for block in artifact.outline.blocks])
        draft = Draft(blocks=[replacements.get(block.id, block) for block in artifact.draft.blocks])
        validate_structured_text(artifact.context, outline, draft)

        stage = "writing-review-v1:revision"
        targets = [unit.id for block in [*outline.blocks, *draft.blocks] for unit in block.units]
        assert_writing_dependencies_current(runner.root, artifact.context, artifact.domain_pin)
        with runner.backend.validated() as backend:
            review = review_text(
                artifact.context, domain, outline, draft, targets, backend,
                timeout_seconds=runner.config.timeout_seconds,
                max_input_chars=runner.config.max_input_chars,
            )
            assert_writing_dependencies_current(runner.root, artifact.context, artifact.domain_pin)
        _progress(runner, document_id, input_hash, stage, "complete")

        reviewed_revision = artifact.revision.model_copy(update={
            "stage": "reviewed",
            "pending_ids": [],
        })
        reviewed = WritingArtifact(
            context=artifact.context,
            domain_pin=artifact.domain_pin,
            outline=outline,
            draft=draft,
            review=review,
            revision=reviewed_revision,
        )
        reviewer, _ = _model_author(runner, "independent revision review")
        review_provenance = {
            "reviewer": reviewer,
            "source": "independent_model_call",
            "review_input_hash": _review_input_hash(reviewed),
            "created_at": utcnow(),
        }
        stage = "publish:revision-review"
        assert_writing_dependencies_current(runner.root, reviewed.context, reviewed.domain_pin)
        published = publish_research(
            runner.store,
            "writing",
            reviewed.context.research,
            _render_human_writing(reviewed, bundle.raw_edit, review_provenance),
            dependency_check=lambda: assert_writing_dependencies_current(
                runner.root, reviewed.context, reviewed.domain_pin
            ),
        )
        assert_writing_dependencies_current(runner.root, reviewed.context, reviewed.domain_pin)
        output = str(published.relative_to(runner.root))
        _progress(runner, document_id, input_hash, stage, "complete", output=output)
        return _result("complete", input_hash, output, [])
    except Exception as exc:
        return _operation_failure(runner, document_id, input_hash, stage, exc, published)
