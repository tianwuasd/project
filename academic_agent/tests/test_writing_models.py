import pytest
from pydantic import ValidationError

import academic_agent.writing_models as models

HASH = "a" * 64
BLOCK_1 = "b_" + "1" * 32
BLOCK_2 = "b_" + "2" * 32
BLOCK_3 = "b_" + "3" * 32
UNIT_1 = "u_" + "1" * 32
UNIT_2 = "u_" + "2" * 32


def _source_reference():
    return {
        "type": "source",
        "work_id": "work_1",
        "document_id": "doc_source_1",
        "claim_id": "claim_1",
        "evidence_ids": ["evidence_1"],
    }


def _unit(**changes):
    value = {
        "id": UNIT_1,
        "text": "The source reports a bounded result.",
        "kind": "source",
        "references": [_source_reference()],
        "purpose": None,
    }
    value.update(changes)
    return value


def _block(**changes):
    value = {"id": BLOCK_1, "role": "paragraph", "level": None, "units": [_unit()]}
    value.update(changes)
    return value


def _writing_pin(version: str = "1"):
    return models.ArtifactPin(
        kind="writing",
        version_path=f"C:/project/analyses/writing/v-{version * 32}",
        manifest_hash=HASH,
        semantic_hash="b" * 64,
    )


def _domain_pin(version: str = "3"):
    return models.ArtifactPin(
        kind="domain",
        version_path=f"C:/project/analyses/domain/v-{version * 32}",
        manifest_hash="c" * 64,
        semantic_hash="d" * 64,
    )


def _revision(**changes):
    value = {
        "schema_version": 1,
        "document_id": "doc_" + "e" * 32,
        "extent": "outline",
        "stage": "reviewed",
        "parent_version": None,
        "parent_manifest_hash": None,
        "origin": "model",
        "author": "demo/academic-agent-v1",
        "reason": "initial generation",
        "created_at": "2026-09-10T00:00:00+00:00",
        "input_sha256": None,
        "changed_ids": [],
        "added_ids": [],
        "deleted_ids": [],
        "reordered": False,
        "id_map": {},
        "pending_ids": [],
        "reused_from": {},
        "domain_rebind": None,
    }
    value.update(changes)
    return value


def test_required_persistent_and_wire_contracts_exist():
    required = {
        "TextUnit", "TextBlock", "Outline", "Draft", "ReviewItem", "WritingReview",
        "OutlineReviewState", "DomainRebind", "Revision", "WireTextUnit", "WireTextBlock",
        "WireOutline", "WireDraft",
    }
    assert required <= set(vars(models))


def test_text_units_enforce_claim_reference_and_pending_segmentation_invariants():
    text_unit = getattr(models, "TextUnit")
    assert text_unit.model_validate(_unit(), strict=True).kind == "source"
    with pytest.raises(ValidationError):
        text_unit.model_validate(_unit(references=[]), strict=True)
    with pytest.raises(ValidationError):
        text_unit.model_validate(_unit(kind="hypothesis"), strict=True)
    with pytest.raises(ValidationError):
        text_unit.model_validate(_unit(kind="connector", references=[], purpose=None), strict=True)
    connector = text_unit.model_validate(
        _unit(kind="connector", references=[], purpose="Connect the bounded findings"), strict=True)
    assert connector.purpose == "Connect the bounded findings"
    pending = text_unit.model_validate(
        _unit(kind="unclassified", references=[], purpose="pending segmentation"), strict=True)
    assert pending.kind == "unclassified"
    with pytest.raises(ValidationError):
        text_unit.model_validate(
            _unit(kind="unclassified", references=[_source_reference()], purpose="pending segmentation"),
            strict=True,
        )


def test_blocks_outlines_drafts_and_reviews_enforce_local_invariants():
    text_block = getattr(models, "TextBlock")
    outline = getattr(models, "Outline")
    draft = getattr(models, "Draft")
    review_item = getattr(models, "ReviewItem")
    writing_review = getattr(models, "WritingReview")
    review_state = getattr(models, "OutlineReviewState")

    assert text_block.model_validate(_block(role="heading", level=2), strict=True).level == 2
    with pytest.raises(ValidationError):
        text_block.model_validate(_block(role="heading", level=None), strict=True)
    with pytest.raises(ValidationError):
        text_block.model_validate(_block(role="paragraph", level=1), strict=True)

    assert len(outline.model_validate({"blocks": [_block()]}, strict=True).blocks) == 1
    assert draft.model_validate({"blocks": []}, strict=True).blocks == []
    with pytest.raises(ValidationError):
        outline.model_validate({"blocks": [_block(), _block()]}, strict=True)
    duplicate_unit = _block(id=BLOCK_2, units=[_unit()])
    with pytest.raises(ValidationError):
        outline.model_validate({"blocks": [_block(), duplicate_unit]}, strict=True)

    item = review_item(unit_id=UNIT_1, decision="supported", reason="Bounded by the cited source")
    complete = writing_review(items=[item])
    assert review_state(status="complete", review=complete).status == "complete"
    assert review_state(status="pending", review=writing_review(items=[])).review.items == []
    with pytest.raises(ValidationError):
        review_state(status="pending", review=complete)
    with pytest.raises(ValidationError):
        writing_review(items=[item, item])


def test_wire_contracts_exclude_host_ids_and_preserve_unit_invariants():
    wire_unit = getattr(models, "WireTextUnit")
    wire_block = getattr(models, "WireTextBlock")
    wire_outline = getattr(models, "WireOutline")
    wire_draft = getattr(models, "WireDraft")
    unit = {key: value for key, value in _unit().items() if key != "id"}
    block = {"role": "paragraph", "level": None, "units": [unit]}
    assert wire_outline.model_validate({"blocks": [block]}, strict=True).blocks
    assert wire_draft.model_validate({"blocks": [block]}, strict=True).blocks
    with pytest.raises(ValidationError):
        wire_draft.model_validate({"blocks": []}, strict=True)
    with pytest.raises(ValidationError):
        wire_unit.model_validate({**unit, "id": UNIT_1}, strict=True)
    with pytest.raises(ValidationError):
        wire_unit.model_validate(
            {**unit, "kind": "unclassified", "references": [], "purpose": "pending segmentation"},
            strict=True,
        )
    with pytest.raises(ValidationError):
        wire_block.model_validate({**block, "id": BLOCK_1}, strict=True)


def test_revision_and_domain_rebind_enforce_only_locally_decidable_invariants():
    revision = getattr(models, "Revision")
    domain_rebind = getattr(models, "DomainRebind")
    valid = revision.model_validate(_revision(), strict=True)
    assert valid.extent == "outline" and valid.stage == "reviewed"
    rebind = domain_rebind(old=_domain_pin("3"), new=_domain_pin("4"))
    human = revision.model_validate(_revision(
        origin="human",
        stage="pending",
        parent_version="v-" + "5" * 32,
        parent_manifest_hash=HASH,
        input_sha256="f" * 64,
        added_ids=[BLOCK_2],
        id_map={"new:boundary": BLOCK_2},
        pending_ids=[BLOCK_2],
        reused_from={BLOCK_1: _writing_pin()},
        domain_rebind=rebind,
    ), strict=True)
    assert human.domain_rebind == rebind

    with pytest.raises(ValidationError):
        revision.model_validate(_revision(parent_version="v-" + "5" * 32), strict=True)
    with pytest.raises(ValidationError):
        revision.model_validate(_revision(origin="human", input_sha256=None), strict=True)
    with pytest.raises(ValidationError):
        revision.model_validate(_revision(stage="reviewed", pending_ids=[BLOCK_1]), strict=True)
    with pytest.raises(ValidationError):
        revision.model_validate(_revision(
            origin="human", input_sha256="f" * 64, added_ids=[BLOCK_2], id_map={}), strict=True)
    with pytest.raises(ValidationError):
        domain_rebind(old=_writing_pin(), new=_domain_pin())

    # Extent/body emptiness, unclassified-unit allowance, and exact review coverage need
    # Outline + Draft + WritingReview together and belong to the producing artifact loader.


def test_human_revision_requires_raw_input_and_exact_parent_receipt():
    revision = models.Revision
    valid = revision.model_validate(_revision(
        origin="human",
        author=" Researcher ",
        parent_version="v-" + "5" * 32,
        parent_manifest_hash=HASH,
        input_sha256="f" * 64,
    ), strict=True)
    assert valid.parent_manifest_hash == HASH
    assert valid.author == "Researcher"
    with pytest.raises(ValidationError):
        revision.model_validate(_revision(
            origin="human", author="Researcher", input_sha256="f" * 64), strict=True)


def test_model_revision_requires_actual_backend_and_model_author_parts():
    revision = models.Revision
    valid = revision.model_validate(_revision(author="codex/gpt-5.3-codex"), strict=True)
    assert valid.author == "codex/gpt-5.3-codex"
    namespaced = revision.model_validate(_revision(author="openrouter/openai/gpt-5"), strict=True)
    assert namespaced.author == "openrouter/openai/gpt-5"
    assert revision.model_validate(_revision(author="backend/model"), strict=True).author == "backend/model"
    invalid_authors = (
        " arbitrary/backend",
        "arbitrary/backend ",
        "\tarbitrary/backend",
        "arbitrary/backend\t",
        "\narbitrary/backend",
        "arbitrary/backend\n",
        "arbitrary/mod\u0085el",
        "arbitrary free form author",
        "codex/model\nforged",
        "codex/",
    )
    for author in invalid_authors:
        with pytest.raises(ValidationError):
            revision.model_validate(_revision(author=author), strict=True)
    with pytest.raises(ValidationError):
        revision.model_validate(_revision(author=42), strict=True)


def test_human_revision_id_map_values_are_one_to_one_with_added_blocks():
    revision = models.Revision
    common = {
        "origin": "human",
        "author": "Researcher",
        "parent_version": "v-" + "5" * 32,
        "parent_manifest_hash": HASH,
        "input_sha256": "f" * 64,
    }
    valid = revision.model_validate(_revision(
        **common, added_ids=[BLOCK_2, BLOCK_3],
        id_map={"new:first": BLOCK_2, "new:second": BLOCK_3}), strict=True)
    assert len(set(valid.id_map.values())) == len(valid.id_map) == 2
    with pytest.raises(ValidationError):
        revision.model_validate(_revision(
            **common, added_ids=[BLOCK_2],
            id_map={"new:first": BLOCK_2, "new:second": BLOCK_2}), strict=True)


def test_pending_revision_ids_cannot_refer_to_deleted_blocks():
    revision = models.Revision
    valid = revision.model_validate(_revision(
        stage="pending", deleted_ids=[BLOCK_1], pending_ids=[BLOCK_2]), strict=True)
    assert set(valid.pending_ids).isdisjoint(valid.deleted_ids)
    with pytest.raises(ValidationError):
        revision.model_validate(_revision(
            stage="pending", deleted_ids=[BLOCK_1], pending_ids=[BLOCK_1]), strict=True)
