import hashlib
import json

import pytest
from test_challenges import adapters, explicit_candidate
from test_challenges import project as _candidate_project
from test_domain_analysis import DomainBackend
from test_innovation_assessment import AssessmentBackend
from test_writing_analysis import WritingBackend

from academic_agent.challenges import challenge_candidate, load_candidate
from academic_agent.config import atomic_json, load_config
from academic_agent.demo import FIXTURES, seed_demo
from academic_agent.domain import _payloads as render_domain
from academic_agent.domain import extract_domain, load_domain_pack, load_domain_version
from academic_agent.innovations import assess_candidate
from academic_agent.models import PublicationCheck, SearchPage, Work
from academic_agent.research_artifacts import load_research, publish_research
from academic_agent.research_context import research_fingerprint
from academic_agent.trends import build_trends
from academic_agent.workflow import Runner
from academic_agent.writing import create_draft, create_outline, load_writing_version
from academic_agent.writing_context import writing_context_payload
from academic_agent.writing_models import WritingError, WritingSelection
from academic_agent.writing_revisions import (
    export_edit,
    import_revision,
    review_writing,
    segment_revision,
)

candidate_project = _candidate_project


def _selection(root):
    work_ids = [
        work.id
        for work in sorted(Runner(root, offline=True).store.works(), key=lambda work: work.title)
    ]
    return WritingSelection(
        document_id="doc_" + "9" * 32,
        question="How do the scheduling reports differ?",
        work_ids=work_ids,
        domain_path="builtin:scheduling",
        trend_directory=None,
        candidate_directory=None,
        candidate_id=None,
        assessment_directory=None,
    )


@pytest.fixture
def revision_project(tmp_path):
    seed_demo(tmp_path)
    config = load_config(tmp_path)
    config.max_model_calls = 40
    atomic_json(tmp_path / "project.json", config.model_dump())
    Runner(tmp_path, offline=True).run()
    selection = _selection(tmp_path)
    domain_result = extract_domain(
        Runner(tmp_path, offline=True, backend=DomainBackend()), selection
    )
    outline_result = create_outline(
        Runner(tmp_path, offline=True, backend=WritingBackend()),
        selection,
        tmp_path / domain_result["output"],
    )
    draft_result = create_draft(
        Runner(tmp_path, offline=True, backend=WritingBackend()),
        tmp_path / outline_result["output"],
    )
    return (
        tmp_path,
        selection,
        tmp_path / domain_result["output"],
        tmp_path / outline_result["output"],
        tmp_path / draft_result["output"],
    )


def _editable(root, parent, name="edit.json"):
    destination = root / name
    export_edit(root, parent, destination)
    return destination, json.loads(destination.read_bytes())


def _write_edit(path, document, *, bom=True, crlf=True):
    raw = json.dumps(
        document, ensure_ascii=False, sort_keys=False, indent=2, allow_nan=False
    ).encode("utf-8")
    if crlf:
        raw = raw.replace(b"\n", b"\r\n")
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)
    return raw


def _provenance(root, version):
    evidence = json.loads(load_research(version)["files"]["evidence.json"])
    return {item["block_id"]: item for item in evidence["block_provenance"]}


class RevisionBackend(WritingBackend):
    def __init__(self, *, span_fault=None, **kwargs):
        super().__init__(**kwargs)
        self.span_fault = span_fault

    def run(self, job):
        if job.payload["task"] != "writing-segment-v1":
            return super().run(job)
        self.jobs.append(job)
        blocks = []
        catalog = job.payload["context"]["reference_catalog"]["sources"]
        work_ids = job.payload["context"]["selection"]["work_ids"]
        synthesis_sources = [
            next(source for source in catalog if source["work_id"] == work_id)
            for work_id in work_ids
        ]
        for block in job.payload["raw_blocks"]:
            end = len(block["text"])
            span = {
                "start": 0,
                "end": end,
                "kind": "connector" if block["role"] == "heading" else "synthesis",
                "references": [] if block["role"] == "heading" else [{
                    "type": "synthesis", "sources": synthesis_sources,
                }],
                "purpose": "Preserve the submitted statement for independent review.",
            }
            spans = [span]
            connector_start = block["text"].find("连接处")
            if block["role"] == "paragraph" and connector_start > 0:
                span["end"] = connector_start
                spans.append({
                    "start": connector_start,
                    "end": end,
                    "kind": "connector",
                    "references": [],
                    "purpose": "Retain the human connector and its number for review.",
                })
            if self.span_fault == "bool":
                span["start"] = False
            elif self.span_fault == "gap":
                span["start"] = 1
            elif self.span_fault == "bounds":
                span["end"] = end + 1
            elif self.span_fault == "overlap":
                span["end"] = 1
                spans.append({
                    **span,
                    "start": 0,
                    "end": end,
                })
            elif self.span_fault == "replacement_text":
                span["text"] = "rewritten"
            blocks.append({"block_id": block["id"], "spans": spans})
        from academic_agent.models import ModelResult

        return ModelResult(
            status="succeeded",
            backend=self.name,
            data={"blocks": blocks},
            usage={"mode": "controlled_test", "model_calls": 1},
        )


def test_exact_human_import_is_model_free_immutable_and_preserves_raw_bytes(revision_project):
    root, _, _, _, parent = revision_project
    edit_file, document = _editable(root, parent)
    old_files = load_research(parent)["files"]
    changed_id = document["paragraphs"][1]["id"]
    original = "A 的40个模拟任务未证明普遍改善。\r\n🙂 e\u0301，连接处错误为999。"
    document["paragraphs"][1]["text"] = original.replace("40", "400")
    raw = _write_edit(edit_file, document)
    backend = RevisionBackend()

    result = import_revision(
        Runner(root, offline=True, backend=backend),
        parent,
        edit_file,
        author="Researcher 张",
        reason="Correct the simulated workload and retain the caveat.",
        delete_ids=[],
    )

    assert result["status"] == "complete" and backend.jobs == []
    pending_path = root / result["output"]
    pending = load_writing_version(pending_path)
    files = load_research(pending_path)["files"]
    assert files["raw-edit.bin"] == raw
    assert pending.revision.input_sha256 == hashlib.sha256(raw).hexdigest()
    assert pending.revision.changed_ids == [changed_id]
    assert pending.revision.pending_ids == [changed_id]
    changed = next(block for block in pending.draft.blocks if block.id == changed_id)
    assert "400" in changed.units[0].text and "\r\n🙂 e\u0301" in changed.units[0].text
    assert changed.units[0].kind == "unclassified" and changed.units[0].references == []
    assert pending.review.items == [] and pending.revision.stage == "pending"
    assert load_research(parent)["files"] == old_files


def test_edit_grammar_enforces_parent_arrays_deletions_and_stable_new_ids(revision_project):
    root, _, _, _, parent = revision_project
    edit_file, document = _editable(root, parent)
    deleted = document["paragraphs"].pop(1)["id"]
    document["paragraphs"].append(
        {"id": "new:boundary", "role": "paragraph", "level": None, "text": "New boundary text."}
    )
    raw = _write_edit(edit_file, document, bom=False, crlf=False)
    with pytest.raises(WritingError):
        import_revision(
            Runner(root, offline=True), parent, edit_file,
            author="Author", reason="Edit", delete_ids=[],
        )
    first = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Author", reason="Edit", delete_ids=[deleted],
    )
    second = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Author", reason="Edit", delete_ids=[deleted],
    )
    one = load_writing_version(root / first["output"])
    two = load_writing_version(root / second["output"])
    assert one.revision.id_map == two.revision.id_map
    new_id = one.revision.id_map["new:boundary"]
    new_block = next(block for block in one.draft.blocks if block.id == new_id)
    assert new_block.units[0].kind == "unclassified" and new_block.units[0].references == []
    expected = hashlib.sha256(json.dumps(
        ["writing-edit-ids-1", one.revision.document_id, one.revision.parent_manifest_hash,
         hashlib.sha256(raw).hexdigest(), "new:boundary"],
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    assert new_id == "b_" + expected[:32]

    wrong_array = json.loads(edit_file.read_bytes())
    wrong_array["outline"].append(wrong_array["paragraphs"].pop(0))
    _write_edit(edit_file, wrong_array, bom=False, crlf=False)
    with pytest.raises(WritingError):
        import_revision(
            Runner(root, offline=True), parent, edit_file,
            author="Author", reason="Edit", delete_ids=[deleted],
        )

    duplicate = b'{"schema_version":1,"schema_version":1}'
    edit_file.write_bytes(duplicate)
    with pytest.raises(WritingError):
        import_revision(
            Runner(root, offline=True), parent, edit_file,
            author="Author", reason="Edit", delete_ids=[deleted],
        )
    document["unsupported"] = True
    _write_edit(edit_file, document, bom=False, crlf=False)
    with pytest.raises(WritingError):
        import_revision(
            Runner(root, offline=True), parent, edit_file,
            author="Author", reason="Edit", delete_ids=[deleted],
        )
    document.pop("unsupported")
    document["paragraphs"][-1]["text"] += " [@unknown_key]"
    _write_edit(edit_file, document, bom=False, crlf=False)
    with pytest.raises(WritingError):
        import_revision(
            Runner(root, offline=True), parent, edit_file,
            author="Author", reason="Edit", delete_ids=[deleted],
        )


def test_formatting_only_outline_revision_remains_reviewed_and_can_create_draft(revision_project):
    root, _, _, outline_parent, _ = revision_project
    edit_file, document = _editable(root, outline_parent, "outline-edit.json")
    raw = _write_edit(edit_file, document, bom=True, crlf=True)
    result = import_revision(
        Runner(root, offline=True), outline_parent, edit_file,
        author="Human author", reason="JSON formatting only", delete_ids=[],
    )
    artifact = load_writing_version(root / result["output"])
    assert artifact.revision.extent == "outline"
    assert artifact.revision.stage == "reviewed"
    assert artifact.revision.pending_ids == [] and artifact.draft.blocks == []
    assert load_research(root / result["output"])["files"]["raw-edit.bin"] == raw
    draft = create_draft(
        Runner(root, offline=True, backend=WritingBackend()), root / result["output"]
    )
    assert draft["status"] == "complete"


@pytest.mark.parametrize("fault", ["bool", "gap", "overlap", "bounds", "replacement_text"])
def test_segmentation_rejects_nonexact_codepoint_spans(revision_project, fault):
    root, _, domain_path, _, parent = revision_project
    edit_file, document = _editable(root, parent, f"{fault}.json")
    document["paragraphs"][1]["text"] = "A 的40个模拟任务未证明普遍改善。\r\n🙂"
    _write_edit(edit_file, document)
    imported = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Author", reason="Exact text", delete_ids=[],
    )
    artifact = load_writing_version(root / imported["output"])
    domain = load_domain_version(domain_path)
    with pytest.raises(WritingError):
        segment_revision(
            artifact.context, domain, artifact, artifact.revision.pending_ids,
            RevisionBackend(span_fault=fault), timeout_seconds=30, max_input_chars=100_000,
        )


def test_review_segments_changed_blocks_then_rereviews_every_ordered_unit_and_caches(revision_project):
    root, _, _, _, parent = revision_project
    edit_file, document = _editable(root, parent)
    changed_id = document["paragraphs"][1]["id"]
    document["paragraphs"][1]["text"] = "A 的400个模拟任务未证明普遍改善。\r\n🙂 连接处错误为999。"
    raw = _write_edit(edit_file, document)
    imported = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Human author", reason="Change a number", delete_ids=[],
    )
    pending_path = root / imported["output"]
    pending_evidence = json.loads(load_research(pending_path)["files"]["evidence.json"])
    parent_units = {
        unit.id for block in load_writing_version(parent).outline.blocks for unit in block.units
    }
    backend = RevisionBackend(review_decision="contradicted")
    reviewed = review_writing(Runner(root, offline=True, backend=backend), pending_path)
    assert reviewed["status"] == "complete", (
        reviewed, Runner(root, offline=True).store.logs("writing_progress")[-1]["data"]["error"]
    )
    assert [job.payload["task"] for job in backend.jobs] == [
        "writing-segment-v1", "writing-review-v1"
    ]
    review_job = backend.jobs[-1]
    assert parent_units <= set(review_job.payload["target_ids"])
    artifact = load_writing_version(root / reviewed["output"])
    assert artifact.revision.author == "Human author"
    assert artifact.revision.input_sha256 == hashlib.sha256(raw).hexdigest()
    assert artifact.revision.pending_ids == [] and artifact.revision.stage == "reviewed"
    changed = next(block for block in artifact.draft.blocks if block.id == changed_id)
    assert "".join(unit.text for unit in changed.units) == document["paragraphs"][1]["text"]
    assert any(unit.kind == "connector" and "999" in unit.text for unit in changed.units)
    assert [item.unit_id for item in artifact.review.items] == review_job.payload["target_ids"]
    evidence = json.loads(load_research(root / reviewed["output"])["files"]["evidence.json"])
    assert evidence["block_provenance"] == pending_evidence["block_provenance"]
    provenance = {item["block_id"]: item for item in evidence["block_provenance"]}
    assert provenance[changed_id]["origin"] == "human"
    assert provenance[changed_id]["author"] == "Human author"
    unchanged_id = artifact.draft.blocks[0].id
    assert provenance[unchanged_id]["origin"] == "model"
    assert provenance[unchanged_id]["author"].startswith("controlled-writing/")
    assert provenance[unchanged_id]["source_version_path"] == str(parent.resolve())
    inherited_outline_id = artifact.outline.blocks[0].id
    parent_artifact = load_writing_version(parent)
    assert provenance[inherited_outline_id]["source_version_path"] == \
        parent_artifact.revision.reused_from[inherited_outline_id].version_path
    assert evidence["review_provenance"]["reviewer"].startswith("controlled-writing/")
    files = dict(load_research(root / reviewed["output"])["files"])
    evidence["block_provenance"][0]["author"] = "forged author"
    files["evidence.json"] = json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode()
    forged = publish_research(
        Runner(root, offline=True).store, "writing", artifact.context.research, files
    )
    with pytest.raises(ValueError):
        load_writing_version(forged)


def test_human_outline_title_edit_reviews_then_creates_draft(revision_project):
    root, _, _, outline_parent, _ = revision_project
    edit_file, document = _editable(root, outline_parent, "human-outline.json")
    document["outline"][0]["text"] = "Revised bounded scheduling title"
    raw = _write_edit(edit_file, document)
    imported = import_revision(
        Runner(root, offline=True), outline_parent, edit_file,
        author="Outline author", reason="Clarify title scope", delete_ids=[],
    )
    pending_path = root / imported["output"]
    pending = load_writing_version(pending_path)
    assert pending.revision.extent == "outline" and pending.revision.stage == "pending"
    assert pending.draft.blocks == []
    changed_id = document["outline"][0]["id"]
    unchanged_id = document["outline"][1]["id"]
    pending_provenance = _provenance(root, pending_path)
    assert pending_provenance[changed_id]["origin"] == "human"
    assert pending_provenance[changed_id]["author"] == "Outline author"
    assert pending_provenance[unchanged_id]["origin"] == "model"
    assert pending_provenance[unchanged_id]["author"].startswith("controlled-writing/")
    assert pending_provenance[unchanged_id]["source_version_path"] == str(
        outline_parent.resolve()
    )
    reviewed = review_writing(
        Runner(root, offline=True, backend=RevisionBackend()), pending_path
    )
    reviewed_path = root / reviewed["output"]
    artifact = load_writing_version(reviewed_path)
    assert artifact.revision.extent == "outline" and artifact.revision.stage == "reviewed"
    assert artifact.revision.author == "Outline author"
    assert artifact.revision.input_sha256 == hashlib.sha256(raw).hexdigest()
    assert _provenance(root, reviewed_path) == pending_provenance
    draft = create_draft(
        Runner(root, offline=True, backend=WritingBackend()), reviewed_path
    )
    assert draft["status"] == "complete"


def test_pure_reorder_reuses_segmentation_but_still_runs_whole_review(revision_project):
    root, _, _, _, parent = revision_project
    edit_file, document = _editable(root, parent)
    document["paragraphs"] = list(reversed(document["paragraphs"]))
    _write_edit(edit_file, document)
    imported = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Human author", reason="Reorder body", delete_ids=[],
    )
    pending = load_writing_version(root / imported["output"])
    assert pending.revision.reordered is True
    assert pending.revision.pending_ids == []
    assert pending.revision.stage == "pending" and pending.review.items == []
    backend = RevisionBackend()
    reviewed = review_writing(Runner(root, offline=True, backend=backend), root / imported["output"])
    assert reviewed["status"] == "complete"
    assert [job.payload["task"] for job in backend.jobs] == ["writing-review-v1"]
    assert backend.jobs[0].payload["draft"]["blocks"][0]["id"] == document["paragraphs"][0]["id"]


def test_second_import_keeps_pending_pure_reorder_until_whole_review(revision_project):
    root, _, _, _, parent = revision_project
    edit_file, document = _editable(root, parent, "first-reorder.json")
    document["paragraphs"] = list(reversed(document["paragraphs"]))
    _write_edit(edit_file, document)
    first = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="First author", reason="Reorder body", delete_ids=[],
    )
    first_path = root / first["output"]
    first_artifact = load_writing_version(first_path)
    assert first_artifact.revision.stage == "pending"
    assert first_artifact.revision.pending_ids == []

    second_edit, unchanged = _editable(root, first_path, "second-reorder.json")
    _write_edit(second_edit, unchanged, bom=False, crlf=False)
    second = import_revision(
        Runner(root, offline=True), first_path, second_edit,
        author="Second author", reason="Formatting-only follow-up", delete_ids=[],
    )
    assert second["status"] == "complete"
    second_path = root / second["output"]
    second_artifact = load_writing_version(second_path)
    assert second_artifact.revision.stage == "pending"
    assert second_artifact.revision.pending_ids == []
    assert second_artifact.review.items == []

    backend = RevisionBackend()
    reviewed = review_writing(Runner(root, offline=True, backend=backend), second_path)
    assert reviewed["status"] == "complete"
    assert [job.payload["task"] for job in backend.jobs] == ["writing-review-v1"]
    final = load_writing_version(root / reviewed["output"])
    assert final.revision.stage == "reviewed"
    assert [item.unit_id for item in final.review.items] == [
        unit.id for block in [*final.outline.blocks, *final.draft.blocks] for unit in block.units
    ]


def test_second_import_carries_all_surviving_pending_targets_through_review(revision_project):
    root, _, _, _, parent = revision_project
    first_edit, first_document = _editable(root, parent, "first-multiple.json")
    first_ids = [block["id"] for block in first_document["paragraphs"]]
    for index, block in enumerate(first_document["paragraphs"], 1):
        block["text"] += f" First pending change {index}."
    _write_edit(first_edit, first_document)
    first = import_revision(
        Runner(root, offline=True), parent, first_edit,
        author="First author", reason="Change two paragraphs", delete_ids=[],
    )
    first_path = root / first["output"]
    assert set(load_writing_version(first_path).revision.pending_ids) == set(first_ids)

    second_edit, second_document = _editable(root, first_path, "second-multiple.json")
    second_document["paragraphs"][0]["text"] += " Second pending change."
    _write_edit(second_edit, second_document)
    second = import_revision(
        Runner(root, offline=True), first_path, second_edit,
        author="Second author", reason="Revise one unfinished paragraph", delete_ids=[],
    )
    assert second["status"] == "complete"
    second_path = root / second["output"]
    second_artifact = load_writing_version(second_path)
    assert second_artifact.revision.stage == "pending"
    assert set(second_artifact.revision.pending_ids) == set(first_ids)

    backend = RevisionBackend()
    reviewed = review_writing(Runner(root, offline=True, backend=backend), second_path)
    assert reviewed["status"] == "complete"
    assert [job.payload["task"] for job in backend.jobs] == [
        "writing-segment-v1", "writing-review-v1",
    ]
    assert set(backend.jobs[0].payload["target_ids"]) == set(first_ids)
    final = load_writing_version(root / reviewed["output"])
    assert final.revision.stage == "reviewed"
    assert [item.unit_id for item in final.review.items] == [
        unit.id for block in [*final.outline.blocks, *final.draft.blocks] for unit in block.units
    ]


def test_review_retry_after_publication_failure_reuses_segmentation_and_review(
        revision_project, monkeypatch):
    root, _, _, _, parent = revision_project
    edit_file, document = _editable(root, parent)
    document["paragraphs"][1]["text"] = "A 的400个模拟任务未证明普遍改善。\r\n🙂"
    _write_edit(edit_file, document)
    imported = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Human author", reason="Change workload", delete_ids=[],
    )
    pending_path = root / imported["output"]
    import academic_agent.writing_revisions as revisions

    actual_publish = revisions.publish_research
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("controlled failure after successful review")
        return actual_publish(*args, **kwargs)

    monkeypatch.setattr(revisions, "publish_research", fail_once)
    first_backend = RevisionBackend()
    first = review_writing(Runner(root, offline=True, backend=first_backend), pending_path)
    assert first["status"] == "failed"
    assert [job.payload["task"] for job in first_backend.jobs] == [
        "writing-segment-v1", "writing-review-v1"
    ]
    first_targets = first_backend.jobs[-1].payload["target_ids"]
    second_backend = RevisionBackend()
    second = review_writing(Runner(root, offline=True, backend=second_backend), pending_path)
    assert second["status"] == "complete" and second_backend.jobs == []
    artifact = load_writing_version(root / second["output"])
    assert [item.unit_id for item in artifact.review.items] == first_targets


def test_semantic_domain_rebind_preserves_human_text_and_resegments_all_blocks(revision_project):
    root, _, domain_path, _, parent = revision_project
    domain = load_domain_version(domain_path)
    fields = {work_id: list(values) for work_id, values in domain.fields.items()}
    work_id = domain.context.selection.work_ids[0]
    fields[work_id][0] = fields[work_id][0].model_copy(update={"value": 400})
    replacement_path = publish_research(
        Runner(root, offline=True).store,
        "domain",
        domain.context.research,
        render_domain(domain.context, fields, domain.review),
    )
    replacement = load_domain_version(replacement_path)
    assert replacement.pin.semantic_hash != domain.pin.semantic_hash
    edit_file, document = _editable(root, parent)
    raw = _write_edit(edit_file, document)
    result = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Human author", reason="Rebind current domain", delete_ids=[],
        domain_version=replacement_path,
    )
    artifact = load_writing_version(root / result["output"])
    all_blocks = [*artifact.outline.blocks, *artifact.draft.blocks]
    assert set(artifact.revision.pending_ids) == {block.id for block in all_blocks}
    assert artifact.revision.domain_rebind.old == domain.pin
    assert artifact.revision.domain_rebind.new == replacement.pin
    assert load_research(root / result["output"])["files"]["raw-edit.bin"] == raw
    assert all(block.units[0].kind == "unclassified" for block in all_blocks)
    provenance = _provenance(root, root / result["output"])
    assert all(item["origin"] == "model" for item in provenance.values())
    assert all(item["author"].startswith("controlled-writing/") for item in provenance.values())
    parent_provenance = _provenance(root, parent)
    for block_id, item in provenance.items():
        source = parent_provenance[block_id]
        assert item["created_at"] == source["created_at"]
        assert item["parent_version"] == source["parent_version"]
        assert item["parent_manifest_hash"] == source["parent_manifest_hash"]
        assert item["source_version_path"] == (
            source["source_version_path"] or str(parent.resolve())
        )
        assert item["source_manifest_hash"] == (
            source["source_manifest_hash"]
            or load_research(parent)["manifest"]["manifest_hash"]
        )


def test_recaptured_challenge_context_change_with_same_domain_hash_requires_fresh_review(
        candidate_project, monkeypatch):
    root, candidate_id = candidate_project
    candidate_id, snapshot = explicit_candidate(root, candidate_id)
    config = load_config(root)
    config.max_model_calls = 60
    config.max_input_chars = 500_000
    atomic_json(root / "project.json", config.model_dump())
    candidate = load_candidate(root, candidate_id)
    full_selection = WritingSelection(
        document_id="doc_" + "5" * 32,
        question="How do the scheduling reports differ?",
        work_ids=snapshot.work_ids,
        domain_path="builtin:scheduling",
        trend_directory=None,
        candidate_directory=str(candidate["version_path"].resolve()),
        candidate_id=candidate_id,
        assessment_directory=None,
    )
    base_selection = full_selection.model_copy(update={
        "candidate_directory": None,
        "candidate_id": None,
    })
    adapters(monkeypatch, lambda source, query, count: SearchPage(source=source, records=[
        Work(title="Later retained unread challenge", doi="10.1234/unread-challenge"),
    ]))
    runner = Runner(root)
    work_id, = challenge_candidate(runner, candidate_id)["retained_work_ids"]
    work = runner.store.work(work_id)
    work.publication_checks = [
        PublicationCheck(source="original-fixture", status="no_notice_found"),
    ]
    runner.store.update_work(work)
    source = root / "later-unread-challenge.txt"
    source.write_text(FIXTURES[0][1], encoding="utf-8")
    runner.attach(work_id, source)
    runner.store.screen(work_id, "include", "Retain for challenge reading.")
    domain_result = extract_domain(
        Runner(root, offline=True, backend=DomainBackend()), base_selection
    )
    domain_path = root / domain_result["output"]
    outline_result = create_outline(
        Runner(root, offline=True, backend=WritingBackend()), full_selection, domain_path
    )
    parent_path = root / outline_result["output"]
    parent = load_writing_version(parent_path)
    assert parent.context.challenge_context is not None
    parent_hit, = parent.context.challenge_context.hits
    assert parent_hit.work_id == work_id and "unread" in parent_hit.statuses

    assert runner.read_selected(work_id=work_id)["results"][0]["status"] == "ok"
    assert runner.review_selected(work_id=work_id)["results"][0]["status"] == "ok"
    replacement_result = extract_domain(
        Runner(root, offline=True, backend=DomainBackend()), base_selection
    )
    replacement_path = root / replacement_result["output"]
    assert load_domain_version(replacement_path).pin.semantic_hash == parent.domain_pin.semantic_hash
    edit_file, document = _editable(root, parent_path, "challenge-recapture.json")
    raw = _write_edit(edit_file, document)
    imported = import_revision(
        Runner(root, offline=True), parent_path, edit_file,
        author="Challenge author", reason="Recapture challenge coverage", delete_ids=[],
        domain_version=replacement_path,
    )
    assert imported["status"] == "complete"
    pending_path = root / imported["output"]
    pending = load_writing_version(pending_path)
    assert pending.domain_pin.semantic_hash == parent.domain_pin.semantic_hash
    assert writing_context_payload(pending.context) != writing_context_payload(parent.context)
    assert pending.context.challenge_context != parent.context.challenge_context
    recaptured_hit, = pending.context.challenge_context.hits
    assert recaptured_hit.work_id == work_id and recaptured_hit.statuses == ["ready"]
    assert pending.revision.stage == "pending" and pending.review.items == []
    all_blocks = [*pending.outline.blocks, *pending.draft.blocks]
    assert set(pending.revision.pending_ids) == {block.id for block in all_blocks}
    assert all(block.units[0].kind == "unclassified" for block in all_blocks)
    assert load_research(pending_path)["files"]["raw-edit.bin"] == raw
    provenance = _provenance(root, pending_path)
    assert all(item["origin"] == "model" for item in provenance.values())
    assert all(item["source_version_path"] == str(parent_path.resolve())
               for item in provenance.values())

    backend = RevisionBackend()
    reviewed = review_writing(Runner(root, offline=True, backend=backend), pending_path)
    assert reviewed["status"] == "complete"
    assert [job.payload["task"] for job in backend.jobs] == [
        "writing-segment-v1", "writing-review-v1",
    ]
    reviewed_artifact = load_writing_version(root / reviewed["output"])
    assert _provenance(root, root / reviewed["output"]) == provenance
    assert [item.unit_id for item in reviewed_artifact.review.items] == [
        unit.id
        for block in [*reviewed_artifact.outline.blocks, *reviewed_artifact.draft.blocks]
        for unit in block.units
    ]


def test_semantically_identical_domain_republication_changes_only_audit_pin(revision_project):
    root, _, domain_path, _, parent = revision_project
    domain = load_domain_version(domain_path)
    replacement_path = publish_research(
        Runner(root, offline=True).store,
        "domain",
        domain.context.research,
        render_domain(domain.context, domain.fields, domain.review),
    )
    replacement = load_domain_version(replacement_path)
    assert replacement.pin.semantic_hash == domain.pin.semantic_hash
    edit_file, document = _editable(root, parent)
    _write_edit(edit_file, document)
    result = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Human author", reason="Audit pin update", delete_ids=[],
        domain_version=replacement_path,
    )
    artifact = load_writing_version(root / result["output"])
    assert artifact.revision.stage == "reviewed" and artifact.revision.pending_ids == []
    assert artifact.revision.domain_rebind.old == domain.pin
    assert artifact.revision.domain_rebind.new == replacement.pin
    assert artifact.review == load_writing_version(parent).review

    changed_selection = domain.context.selection.model_copy(update={"question": "Different question"})
    placeholder = domain.context.model_copy(update={
        "selection": changed_selection,
        "input_hash": "0" * 64,
    })
    changed_context = placeholder.model_copy(update={
        "input_hash": hashlib.sha256(json.dumps(
            writing_context_payload(placeholder), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
    })
    wrong_path = publish_research(
        Runner(root, offline=True).store,
        "domain",
        changed_context.research,
        render_domain(changed_context, domain.fields, domain.review),
    )
    with pytest.raises(WritingError):
        import_revision(
            Runner(root, offline=True), parent, edit_file,
            author="Human author", reason="Wrong selection", delete_ids=[],
            domain_version=wrong_path,
        )


def test_fresh_base_domain_rebind_recaptures_all_explicit_optional_pins(candidate_project):
    root, candidate_id = candidate_project
    config = load_config(root)
    config.max_model_calls = 50
    config.max_input_chars = 500_000
    atomic_json(root / "project.json", config.model_dump())
    candidate = load_candidate(root, candidate_id)
    assessment_result = assess_candidate(
        Runner(root, offline=True, backend=AssessmentBackend()), candidate_id
    )
    assert assessment_result["status"] == "complete"
    trend_path = build_trends(Runner(root, offline=True).store)
    full_selection = WritingSelection(
        document_id="doc_" + "6" * 32,
        question="How do the scheduling reports differ?",
        work_ids=candidate["snapshot"].work_ids,
        domain_path="builtin:scheduling",
        trend_directory=str(trend_path.resolve()),
        candidate_directory=str(candidate["version_path"].resolve()),
        candidate_id=candidate_id,
        assessment_directory=str((root / assessment_result["output"]).resolve()),
    )
    full_domain_result = extract_domain(
        Runner(root, offline=True, backend=DomainBackend()), full_selection
    )
    outline_result = create_outline(
        Runner(root, offline=True, backend=WritingBackend()),
        full_selection,
        root / full_domain_result["output"],
    )
    assert outline_result["status"] == "complete", outline_result
    parent_path = root / outline_result["output"]
    parent = load_writing_version(parent_path)
    base_selection = full_selection.model_copy(update={
        "trend_directory": None,
        "candidate_directory": None,
        "candidate_id": None,
        "assessment_directory": None,
    })
    base_domain_result = extract_domain(
        Runner(root, offline=True, backend=DomainBackend()), base_selection
    )
    edit_file, document = _editable(root, parent_path, "optional-pins-edit.json")
    raw = _write_edit(edit_file, document)
    imported = import_revision(
        Runner(root, offline=True), parent_path, edit_file,
        author="Pinned author", reason="Rebind a fresh base domain", delete_ids=[],
        domain_version=root / base_domain_result["output"],
    )
    assert imported["status"] == "complete"
    artifact = load_writing_version(root / imported["output"])
    assert artifact.context.selection == full_selection
    assert artifact.context.trend == parent.context.trend
    assert artifact.context.candidate == parent.context.candidate
    assert artifact.context.assessment == parent.context.assessment
    assert artifact.domain_pin == load_domain_version(root / base_domain_result["output"]).pin
    assert load_research(root / imported["output"])["files"]["raw-edit.bin"] == raw


@pytest.mark.parametrize("refresh", ["source", "manual_review"])
def test_source_and_manual_review_refresh_rebind_exact_human_text(revision_project, refresh):
    root, selection, _, _, parent_path = revision_project
    parent = load_writing_version(parent_path)
    edit_file, document = _editable(root, parent_path, f"{refresh}-refresh.json")
    raw = _write_edit(edit_file, document)
    runner = Runner(root, offline=True)
    if refresh == "source":
        work_id = selection.work_ids[0]
        old_document = parent.context.research.documents[parent.context.research.cards[work_id].document_id]
        source = root / old_document.source_path
        replacement = root / "refreshed-source.txt"
        replacement.write_bytes(source.read_bytes() + b"\n")
        runner.attach(work_id, replacement)
        assert runner.read_selected(work_id=work_id)["results"][0]["status"] == "ok"
        assert runner.review_selected(work_id=work_id)["results"][0]["status"] == "ok"
    else:
        claim_id = parent.context.research.supported_claim_ids[0]
        runner.manual_review(
            claim_id, "supported", "Refreshed human rationale.", "Manual reviewer"
        )
    refreshed_domain = extract_domain(
        Runner(root, offline=True, backend=DomainBackend()), selection
    )
    result = import_revision(
        Runner(root, offline=True), parent_path, edit_file,
        author="Refresh author", reason=f"Apply {refresh} refresh", delete_ids=[],
        domain_version=root / refreshed_domain["output"],
    )
    assert result["status"] == "complete"
    artifact = load_writing_version(root / result["output"])
    assert research_fingerprint(artifact.context.research) != research_fingerprint(parent.context.research)
    assert artifact.revision.domain_rebind.old == parent.domain_pin
    assert artifact.revision.domain_rebind.new == load_domain_version(
        root / refreshed_domain["output"]
    ).pin
    assert load_research(root / result["output"])["files"]["raw-edit.bin"] == raw


def test_pack_refresh_rebinds_exact_human_text(tmp_path):
    seed_demo(tmp_path)
    config = load_config(tmp_path)
    config.max_model_calls = 40
    atomic_json(tmp_path / "project.json", config.model_dump())
    Runner(tmp_path, offline=True).run()
    external = tmp_path / "external-domain.json"
    pack_value = load_domain_pack("builtin:scheduling").pack.model_dump(mode="json")
    external.write_text(json.dumps(pack_value, ensure_ascii=False), encoding="utf-8")
    selection = _selection(tmp_path).model_copy(update={"domain_path": str(external.resolve())})
    domain_result = extract_domain(
        Runner(tmp_path, offline=True, backend=DomainBackend()), selection
    )
    outline_result = create_outline(
        Runner(tmp_path, offline=True, backend=WritingBackend()),
        selection,
        tmp_path / domain_result["output"],
    )
    parent_path = tmp_path / outline_result["output"]
    edit_file, document = _editable(tmp_path, parent_path, "pack-refresh.json")
    raw = _write_edit(edit_file, document)
    pack_value["version"] = "2"
    external.write_text(json.dumps(pack_value, ensure_ascii=False), encoding="utf-8")
    refreshed = extract_domain(
        Runner(tmp_path, offline=True, backend=DomainBackend()), selection
    )
    result = import_revision(
        Runner(tmp_path, offline=True), parent_path, edit_file,
        author="Pack author", reason="Refresh pack version", delete_ids=[],
        domain_version=tmp_path / refreshed["output"],
    )
    assert result["status"] == "complete"
    artifact = load_writing_version(tmp_path / result["output"])
    assert artifact.context.pack.pack.version == "2"
    assert load_research(tmp_path / result["output"])["files"]["raw-edit.bin"] == raw


@pytest.mark.parametrize("case", [
    "duplicate_block_id",
    "unknown_block_id",
    "wrong_document",
    "wrong_version",
    "wrong_manifest",
    "heading_without_level",
    "paragraph_with_level",
    "invalid_role",
])
def test_import_rejects_exact_id_receipt_and_role_level_grammar(revision_project, case):
    root, _, _, _, parent = revision_project
    edit_file, document = _editable(root, parent, f"negative-{case}.json")
    if case == "duplicate_block_id":
        document["outline"][1]["id"] = document["outline"][0]["id"]
    elif case == "unknown_block_id":
        document["outline"][0]["id"] = "b_" + "0" * 32
    elif case == "wrong_document":
        document["document_id"] = "doc_" + "0" * 32
    elif case == "wrong_version":
        document["parent_version"] = "v-" + "0" * 32
    elif case == "wrong_manifest":
        document["parent_manifest_hash"] = "0" * 64
    elif case == "heading_without_level":
        document["outline"][0]["level"] = None
    elif case == "invalid_role":
        document["outline"][0]["role"] = "title"
    else:
        document["outline"][1]["level"] = 1
    _write_edit(edit_file, document)
    with pytest.raises(WritingError):
        import_revision(
            Runner(root, offline=True), parent, edit_file,
            author="Author", reason="Invalid edit", delete_ids=[],
        )


def test_stale_parent_requires_explicit_rebind_but_export_remains_available(revision_project):
    root, selection, _, _, parent = revision_project
    edit_file, document = _editable(root, parent)
    raw = _write_edit(edit_file, document)
    artifact = load_writing_version(parent)
    source = root / next(iter(artifact.context.research.documents.values())).source_path
    source.write_bytes(source.read_bytes() + b"changed")
    result = import_revision(
        Runner(root, offline=True), parent, edit_file,
        author="Human author", reason="Keep exact edit", delete_ids=[],
    )
    assert result["status"] == "stale" and result["output"] is None
    gap = result["gaps"][0]
    assert gap["extract_argv"] == [
        "academic-agent", "domain-extract", "--question", selection.question,
        "--work-id", selection.work_ids[0], "--work-id", selection.work_ids[1],
        "--domain", selection.domain_path,
    ]
    assert str(parent) in gap["reimport_argv"] and str(edit_file) in gap["reimport_argv"]
    assert edit_file.read_bytes() == raw
    exported = root / "stale-export.json"
    assert export_edit(root, parent, exported) == exported


def test_export_is_exclusive_and_historical_hash_tampering_is_rejected(revision_project):
    root, _, _, _, parent = revision_project
    destination = root / "immutable-edit.json"
    export_edit(root, parent, destination)
    with pytest.raises(FileExistsError):
        export_edit(root, parent, destination)
    (parent / "raw-edit.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        export_edit(root, parent, root / "should-not-exist.json")
