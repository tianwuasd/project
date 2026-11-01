import hashlib
import json
from pathlib import Path

import pytest
from test_challenges import (
    explicit_candidate,
    ready_later_challenge_work,
)
from test_challenges import project as _candidate_project
from test_innovation_assessment import AssessmentBackend

from academic_agent.challenges import (
    candidate_version_currentness,
    load_candidate,
    load_candidate_version,
)
from academic_agent.domain import load_domain_pack
from academic_agent.innovations import (
    assess_candidate,
    assessment_version_currentness,
    load_assessment_version,
)
from academic_agent.research_artifacts import load_research, publish_research
from academic_agent.trends import build_trends
from academic_agent.workflow import Runner
from academic_agent.writing_context import (
    assert_writing_current,
    load_writing_context,
    writing_context_payload,
    writing_currentness,
)
from academic_agent.writing_models import WritingError, WritingSelection

project = _candidate_project


def _selection(root, candidate_id, *, assessment=None):
    candidate = load_candidate(root, candidate_id)
    return WritingSelection(
        document_id="doc_" + "1" * 32,
        question="How do the scheduling reports differ?",
        work_ids=candidate["snapshot"].work_ids,
        domain_path="builtin:scheduling",
        trend_directory=None,
        candidate_directory=str(candidate["version_path"]),
        candidate_id=candidate_id,
        assessment_directory=str(assessment) if assessment else None,
    )


def _refresh_context_hash(context):
    payload = writing_context_payload(context)
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8")
    return context.model_copy(update={"input_hash": hashlib.sha256(value).hexdigest()})


def test_explicit_candidate_version_is_loaded_and_current_without_latest_substitution(project):
    root, candidate_id = project
    selected = load_candidate(root, candidate_id)
    loaded = load_candidate_version(selected["version_path"], candidate_id)
    assert set(loaded) == {"candidate", "batch", "review", "snapshot", "version_path", "manifest_hash"}
    assert loaded["version_path"] == selected["version_path"]
    assert candidate_version_currentness(root, loaded) == {
        "candidate_id": candidate_id, "current": True, "gaps": [],
    }

    context = load_writing_context(root, _selection(root, candidate_id))
    assert context.candidate is not None
    assert context.candidate.artifact.version_path == str(selected["version_path"].resolve())
    assert writing_currentness(root, context) == {"current": True, "gaps": []}
    assert writing_context_payload(context)["candidate"]["candidate"]["id"] == candidate_id


def test_writing_context_keeps_later_ready_challenge_work_current(project, monkeypatch):
    root, candidate_id = project
    candidate_id, _ = explicit_candidate(root, candidate_id)
    work_id, _ = ready_later_challenge_work(root, candidate_id, monkeypatch)

    context = load_writing_context(root, _selection(root, candidate_id))
    hit, = context.challenge_context.hits
    assert hit.work_id == work_id
    assert hit.statuses == ['ready']
    assert context.challenge_context.coverage_status == 'needs_assessment'
    assert writing_currentness(root, context) == {"current": True, "gaps": []}


def test_identical_candidate_republication_changes_audit_pin_not_semantic_identity(project):
    root, candidate_id = project
    first_selection = _selection(root, candidate_id)
    first = load_writing_context(root, first_selection)
    original = load_research(Path(first_selection.candidate_directory))
    duplicate = publish_research(
        Runner(root, offline=True).store, "innovation", first.candidate.snapshot, original["files"])
    reloaded_first = load_candidate_version(Path(first.candidate.artifact.version_path), candidate_id)
    assert reloaded_first["manifest_hash"] == first.candidate.artifact.manifest_hash
    assert writing_currentness(root, first) == {"current": True, "gaps": []}
    second = load_writing_context(
        root, first_selection.model_copy(update={"candidate_directory": str(duplicate.resolve())}))
    assert second.candidate.artifact.version_path != first.candidate.artifact.version_path
    assert second.candidate.artifact.manifest_hash != first.candidate.artifact.manifest_hash
    assert second.candidate.artifact.semantic_hash == first.candidate.artifact.semantic_hash
    assert second.input_hash == first.input_hash


def test_assessment_loader_binds_complete_explicit_version_and_context(project):
    root, candidate_id = project
    result = assess_candidate(Runner(root, offline=True, backend=AssessmentBackend()), candidate_id)
    assert result["status"] == "complete"
    directory = root / result["output"]
    loaded = load_assessment_version(root, directory)
    assert set(loaded) == {
        "candidate", "context", "assessment", "review", "snapshot", "version_path", "manifest_hash",
    }
    assert assessment_version_currentness(root, loaded) == {
        "candidate_id": candidate_id, "current": True, "gaps": [],
    }
    artifact = load_research(directory)
    duplicate = publish_research(
        Runner(root, offline=True).store, "assessment", loaded["snapshot"], artifact["files"])
    assert duplicate != directory
    assert load_assessment_version(root, directory)["version_path"] == directory.resolve()
    context = load_writing_context(root, _selection(root, candidate_id, assessment=directory))
    assert context.assessment is not None
    assert context.assessment.artifact.version_path == str(directory.resolve())


def test_writing_context_detects_tampered_source_and_keeps_historical_context(project):
    root, candidate_id = project
    context = load_writing_context(root, _selection(root, candidate_id))
    document = next(iter(context.research.documents.values()))
    source = root / document.source_path
    source.write_bytes(source.read_bytes() + b"tamper")
    state = writing_currentness(root, context)
    assert state["current"] is False
    assert {gap["reason"] for gap in state["gaps"]} >= {"research_inputs_changed"}
    with pytest.raises(WritingError) as caught:
        assert_writing_current(root, context)
    assert caught.value.status == "stale"

    with Runner(root, offline=True).store.connect() as connection:
        connection.execute("UPDATE logs SET data='{' WHERE kind='review'")
    invalid = writing_currentness(root, context)
    assert {gap["reason"] for gap in invalid["gaps"]} >= {"research_inputs_invalid"}
    assert {gap["reason"] for gap in invalid["gaps"]} != {"research_inputs_changed"}
    with pytest.raises(WritingError) as invalid_error:
        assert_writing_current(root, context)
    assert invalid_error.value.status == "failed"


def test_pack_path_bytes_source_path_and_unrelated_logs_have_precise_identity(project):
    root, candidate_id = project
    external = root / "external-domain.json"
    external.write_text(json.dumps(load_domain_pack("builtin:scheduling").pack.model_dump(mode="json"),
                                   ensure_ascii=False), encoding="utf-8")
    selection = _selection(root, candidate_id).model_copy(update={"domain_path": str(external.resolve())})
    before = load_writing_context(root, selection)
    runner = Runner(root, offline=True)
    runner.store.log("unrelated_progress", {"detail": "does not change evidence"})
    runner.store.log("usage_note", {"detail": "does not change evidence"})
    after = load_writing_context(root, selection)
    assert after.input_hash == before.input_hash
    external.write_bytes(external.read_bytes() + b" ")
    assert writing_currentness(root, before)["gaps"] == [{"reason": "domain_pack_changed"}]


def test_source_path_migration_is_semantic_even_when_bytes_match(project):
    root, candidate_id = project
    context = load_writing_context(root, _selection(root, candidate_id))
    document = next(iter(context.research.documents.values()))
    replacement = root / "migrated-source.txt"
    replacement.write_bytes((root / document.source_path).read_bytes())
    runner = Runner(root, offline=True)
    migrated = document.model_copy(update={"source_path": replacement.name})
    blob, sha = runner.store.put_blob(migrated.model_dump_json().encode("utf-8"))
    with runner.store.connect() as connection:
        connection.execute("UPDATE documents SET path=?,sha=? WHERE id=?", (blob, sha, document.id))
    state = writing_currentness(root, context)
    assert state["current"] is False
    assert {gap["reason"] for gap in state["gaps"]} & {
        "research_inputs_changed", "candidate_evidence_changed", "candidate_source_changed",
    }


def test_currentness_revalidates_supplied_context_and_its_semantic_hash(project):
    root, candidate_id = project
    context = load_writing_context(root, _selection(root, candidate_id))
    changed_selection = context.model_copy(
        update={"selection": context.selection.model_copy(update={"question": "Altered question"})})
    state = writing_currentness(root, changed_selection)
    assert state == {"current": False, "gaps": [{"reason": "writing_context_identity_invalid"}]}
    with pytest.raises(WritingError) as caught:
        assert_writing_current(root, changed_selection)
    assert caught.value.status == "failed"

    changed_version = context.model_copy(update={
        "selection": context.selection.model_copy(
            update={"candidate_directory": str((root / ("analyses/innovation/v-" + "0" * 32)).resolve())}),
    })
    assert writing_currentness(root, changed_version) == {
        "current": False, "gaps": [{"reason": "writing_context_invalid"}],
    }

    invalid = context.model_copy(deep=True)
    invalid.selection.question = 7
    with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
        assert writing_currentness(root, invalid) == {
            "current": False, "gaps": [{"reason": "writing_context_invalid"}],
        }


def test_currentness_compares_full_embedded_candidate_pin_after_hash_refresh(project):
    root, candidate_id = project
    context = load_writing_context(root, _selection(root, candidate_id))
    changed_candidate = context.candidate.candidate.model_copy(update={"question": "Altered candidate"})
    changed_pin = context.candidate.model_copy(update={"candidate": changed_candidate})
    tampered = _refresh_context_hash(context.model_copy(update={"candidate": changed_pin}))
    state = writing_currentness(root, tampered)
    assert state["current"] is False
    assert {gap["reason"] for gap in state["gaps"]} == {"candidate_pin_invalid"}


def test_currentness_compares_full_embedded_assessment_pin_after_hash_refresh(project):
    root, candidate_id = project
    result = assess_candidate(Runner(root, offline=True, backend=AssessmentBackend()), candidate_id)
    context = load_writing_context(root, _selection(root, candidate_id, assessment=root / result["output"]))
    changed_assessment = context.assessment.assessment.model_copy(update={"material_cutoff": "altered"})
    changed_pin = context.assessment.model_copy(update={"assessment": changed_assessment})
    tampered = _refresh_context_hash(context.model_copy(update={"assessment": changed_pin}))
    state = writing_currentness(root, tampered)
    assert state["current"] is False
    assert {gap["reason"] for gap in state["gaps"]} == {"assessment_pin_invalid"}


def test_currentness_compares_full_embedded_trend_pin_after_hash_refresh(project):
    root, candidate_id = project
    trend_directory = build_trends(Runner(root, offline=True).store)
    selection = _selection(root, candidate_id).model_copy(update={"trend_directory": str(trend_directory)})
    context = load_writing_context(root, selection)
    changed_counts = {**context.trend.computation.counts,
                      "eligible": context.trend.computation.counts["eligible"] + 1}
    changed_computation = context.trend.computation.model_copy(update={"counts": changed_counts})
    changed_pin = context.trend.model_copy(update={"computation": changed_computation})
    tampered = _refresh_context_hash(context.model_copy(update={"trend": changed_pin}))
    state = writing_currentness(root, tampered)
    assert state["current"] is False
    assert {gap["reason"] for gap in state["gaps"]} == {"trend_pin_invalid"}


@pytest.mark.parametrize("change", [
    {"candidate_directory": None, "candidate_id": "innovation_" + "0" * 32},
    {"candidate_directory": "latest", "candidate_id": "innovation_" + "0" * 32},
    {"assessment_directory": "latest"},
    {"work_ids": []},
])
def test_writing_selection_rejects_missing_pins_latest_and_empty_scope(change):
    values = {
        "document_id": "doc_" + "2" * 32,
        "question": "Bounded question",
        "work_ids": ["work_1"],
        "domain_path": "builtin:scheduling",
        "trend_directory": None,
        "candidate_directory": None,
        "candidate_id": None,
        "assessment_directory": None,
    }
    values.update(change)
    with pytest.raises(ValueError):
        WritingSelection(**values)
