import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from academic_agent import writing_usage
from academic_agent.config import create_project
from academic_agent.demo import seed_demo
from academic_agent.innovation_analysis import generate_candidates
from academic_agent.models import ModelJob, ModelResult, ProjectConfig
from academic_agent.research_context import freeze_research
from academic_agent.workflow import Runner


class _CountingBackend:
    name = "controlled-live"

    def __init__(self):
        self.jobs = []

    def run(self, job):
        self.jobs.append(job)
        return ModelResult(
            status="succeeded",
            backend=self.name,
            data={"sequence": len(self.jobs)},
            usage={"mode": "controlled_test", "model_calls": 1},
        )


def _job(index):
    return ModelJob(
        id=f"job-{index}",
        kind="synthesize",
        instruction="Return the controlled result.",
        payload={"task": "controlled-live-cap", "index": index},
        output_schema={"type": "object"},
        timeout_seconds=2,
    )


def test_live_call_cap_survives_validated_and_blocks_before_fifth_reservation(tmp_path):
    create_project(tmp_path, ProjectConfig(
        title="Bounded live acceptance", backend="exchange", max_model_calls=60,
    ))
    backend = _CountingBackend()
    runner = Runner(tmp_path, offline=True, backend=backend)
    ledger = writing_usage.install_actual_call_cap(runner, max_new_calls=4)

    for index in range(4):
        with runner.backend.validated() as validated:
            assert validated.run(_job(index)).status == "succeeded"

    assert ledger.reservations == 4
    assert ledger.attempts == 4
    assert len(backend.jobs) == 4
    assert runner.store.budget()["consumed_or_reserved_calls"] == 4

    # Budget metadata and audit logging do not change semantic job identity.  A
    # successful cached result remains usable at the cap without a new attempt.
    runner.config.max_model_calls = 59
    runner.store.log("acceptance_probe", {"note": "log-only identity check"})
    with runner.backend.validated() as validated:
        assert validated.run(_job(0)).status == "succeeded"
    assert ledger.reservations == 4
    assert ledger.attempts == 4

    with pytest.raises(writing_usage.ActualCallCapExceeded, match="four-call acceptance cap"):
        with runner.backend.validated() as validated:
            validated.run(_job(4))
    assert runner.store.budget()["consumed_or_reserved_calls"] == 4
    assert len(backend.jobs) == 4


def test_external_pack_variant_has_real_missingness_field_and_boundary_structure(tmp_path):
    path = writing_usage.write_external_pack(tmp_path)
    pack = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path.resolve()
    assert [field["id"] for field in pack["fields"]] == [
        "observation_unit", "arrival_pattern", "transfer_limit",
    ]
    assert pack["comparison_columns"] == [
        "transfer_limit", "arrival_pattern", "observation_unit",
    ]
    assert pack["writing_structure"] == [{
        "id": "boundary_argument",
        "title": "Boundary argument",
        "purpose": "Compare reported scope and transfer limits before proposing transfer.",
    }]


def test_live_module_rejects_non_four_limit_and_existing_output(tmp_path):
    project = tmp_path / "project"
    external_pack = tmp_path / "external.json"
    domain = tmp_path / "domain"
    outline = tmp_path / "outline"
    existing = tmp_path / "existing"
    existing.mkdir()
    prefix = [sys.executable, "-m", "academic_agent.writing_usage"]
    common = [
        "--project", str(project), "--output", str(existing),
        "--external-pack", str(external_pack), "--domain-version", str(domain),
        "--outline-version", str(outline),
    ]

    wrong = subprocess.run(
        [*prefix, *common, "--max-new-calls", "3"],
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert wrong.returncode != 0
    assert "exactly 4" in (wrong.stdout + wrong.stderr)

    repeated = subprocess.run(
        [*prefix, *common, "--max-new-calls", "4"],
        capture_output=True, text=True, encoding="utf-8", timeout=30, check=False,
    )
    assert repeated.returncode != 0
    assert "must not already exist" in (repeated.stdout + repeated.stderr)


def test_live_report_preserves_completed_stage_when_later_stage_and_inspection_fail(
        tmp_path, monkeypatch):
    project = tmp_path / "project"
    create_project(project, ProjectConfig(
        title="Live evidence preservation", backend="codex", max_model_calls=60,
    ))
    external_pack = tmp_path / "external.json"
    external_pack.write_text('{}', encoding="utf-8")
    domain = tmp_path / "domain"
    outline = tmp_path / "outline"
    domain.mkdir()
    outline.mkdir()
    output = tmp_path / "live-output"
    selection = SimpleNamespace(
        question="Question", work_ids=["work_a", "work_b"],
    )
    research = SimpleNamespace(documents={
        "work_a": SimpleNamespace(work_id="work_a", pages=["page"]),
        "work_b": SimpleNamespace(work_id="work_b", pages=["page"]),
    })
    prepared = SimpleNamespace(context=SimpleNamespace(selection=selection, research=research))

    import academic_agent.domain as domain_module
    import academic_agent.writing as writing_module

    monkeypatch.setattr(domain_module, "load_domain_version", lambda path: prepared)
    monkeypatch.setattr(writing_module, "load_writing_version", lambda path: prepared)
    monkeypatch.setattr(
        domain_module, "extract_domain",
        lambda runner, selected: {
            "status": "complete", "output": "analyses/domain/v-live", "gaps": [],
        },
    )

    def fail_draft(runner, version):
        raise RuntimeError("controlled draft boundary")

    def fail_inspection(path, work_id):
        raise ValueError("controlled inspection boundary")

    monkeypatch.setattr(writing_module, "create_draft", fail_draft)
    monkeypatch.setattr(writing_usage, "_inspect_live_domain", fail_inspection)

    report = writing_usage.run_live_acceptance(
        project=project, output=output, external_pack=external_pack,
        domain_version=domain, outline_version=outline, max_new_calls=4,
    )

    assert report["status"] == "incomplete"
    assert report["results"]["domain"]["status"] == "complete"
    assert report["results"]["draft"]["gaps"] == [{
        "reason": "bounded_live_draft_exception", "detail": "controlled draft boundary",
    }]
    assert report["observations"]["domain"]["status"] == "inspection_failed"
    assert report["calls"]["reservations"] == report["calls"]["attempts"] == 0
    assert json.loads((output / "report.json").read_text(encoding="utf-8")) == report


def test_required_writing_scenarios_are_explicit_and_report_detection_layers():
    assert writing_usage.WRITING_SCENARIO_NAMES == (
        "writing_full_journey",
        "writing_revision_recovery",
        "writing_external_pack",
        "writing_stale_rebind_cache",
        "writing_readonly_tamper",
    )
    assert writing_usage.NEGATIVE_DETECTION_LAYERS == {
        "W01": "controlled_exchange_semantic_review",
        "W02": "controlled_exchange_semantic_review",
        "W03": "controlled_exchange_semantic_review",
        "W04": "controlled_exchange_semantic_review",
        "W05": "controlled_exchange_semantic_review",
        "W06": "controlled_exchange_semantic_review",
        "W07": "host_capture_then_controlled_exchange_review",
        "formal_artifact_tamper": "host_manifest_validation",
        "unknown_demo_pack": "demo_protocol_refusal",
    }


def test_mixed_provenance_oracle_requires_model_and_human_blocks_in_child():
    parent = {
        "outline": {"origin": "model", "author": "outline/model", "created_at": "t0",
                    "backend": "exchange", "model_identifier": "preset",
                    "source_version_path": "C:/verified/outline",
                    "source_manifest_hash": "outline-hash"},
        "body-a": {"origin": "model", "author": "draft/model", "created_at": "t1",
                   "backend": "exchange", "model_identifier": "preset",
                   "source_version_path": None, "source_manifest_hash": None},
        "body-b": {"origin": "model", "author": "draft/model", "created_at": "t1",
                   "backend": "exchange", "model_identifier": "preset",
                   "source_version_path": None, "source_manifest_hash": None},
    }
    revision = {
        "changed_ids": ["body-b"],
        "reused_from": {
            "outline": {"version_path": "C:/verified/parent", "manifest_hash": "parent-hash"},
            "body-a": {"version_path": "C:/verified/parent", "manifest_hash": "parent-hash"},
        },
    }
    false_positive_child = {
        block_id: {**value, "origin": "human", "author": "Acceptance Researcher"}
        for block_id, value in parent.items()
    }
    assert not writing_usage._mixed_provenance_matches(
        false_positive_child, parent, revision,
        parent_path="C:/verified/parent", parent_manifest_hash="parent-hash",
        human_author="Acceptance Researcher",
    )

    child = {
        "outline": dict(parent["outline"]),
        "body-a": {
            **parent["body-a"], "source_version_path": "C:/verified/parent",
            "source_manifest_hash": "parent-hash",
        },
        "body-b": {
            "origin": "human", "author": "Acceptance Researcher", "created_at": "t2",
            "backend": None, "model_identifier": None,
            "source_version_path": None, "source_manifest_hash": None,
        },
    }
    assert writing_usage._mixed_provenance_matches(
        child, parent, revision,
        parent_path="C:/verified/parent", parent_manifest_hash="parent-hash",
        human_author="Acceptance Researcher",
    )


def test_pure_reorder_oracle_rejects_add_delete_and_pending_segmentation():
    parent_blocks = [
        {"id": "outline", "units": [{"id": "u-outline"}]},
        {"id": "body-a", "units": [{"id": "u-a"}]},
        {"id": "body-b", "units": [{"id": "u-b1"}, {"id": "u-b2"}]},
    ]
    reordered_blocks = [parent_blocks[0], parent_blocks[2], parent_blocks[1]]
    pending = {
        "changed_ids": [], "added_ids": [], "deleted_ids": [], "pending_ids": [],
        "reordered": True, "stage": "pending",
        "reused_from": {block_id: {} for block_id in ("outline", "body-a", "body-b")},
    }
    reviewed = {**pending, "stage": "reviewed"}
    review = {"items": [{"unit_id": unit_id} for unit_id in (
        "u-outline", "u-b1", "u-b2", "u-a",
    )]}
    assert writing_usage._pure_reorder_matches(
        parent_blocks, reordered_blocks, pending, reviewed, {"items": []}, review,
    )
    assert not writing_usage._pure_reorder_matches(
        parent_blocks, reordered_blocks,
        {**pending, "added_ids": ["new"], "pending_ids": ["new"]},
        reviewed, {"items": []}, review,
    )


def test_complete_project_snapshot_detects_changes_outside_analyses(tmp_path):
    project = tmp_path / "project"
    (project / "analyses").mkdir(parents=True)
    (project / "logs").mkdir()
    (project / "research.db").write_bytes(b"database-before")
    before = writing_usage._project_tree_snapshot(project)

    (project / "research.db").write_bytes(b"database-after")
    (project / "runtime-empty-directory").mkdir()
    after = writing_usage._project_tree_snapshot(project)

    assert before != after
    assert writing_usage._readonly_snapshot_result(before, after)["passed"] is False
    assert before["files"]["research.db"] != after["files"]["research.db"]
    assert "runtime-empty-directory" not in before["directories"]
    assert "runtime-empty-directory" in after["directories"]


def test_readonly_snapshot_oracle_allows_only_narrow_sqlite_housekeeping(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "research.db").write_bytes(b"database")
    before = writing_usage._project_tree_snapshot(project)

    (project / "research.db-shm").write_bytes(b"lock-index")
    (project / "research.db-wal").write_bytes(b"")
    housekeeping = writing_usage._project_tree_snapshot(project)
    result = writing_usage._readonly_snapshot_result(before, housekeeping)
    assert result["passed"] is True
    assert result["sqlite_housekeeping"]["added"] == [
        "research.db-shm", "research.db-wal",
    ]

    (project / "research.db-wal").write_bytes(b"application change")
    nonempty_wal = writing_usage._project_tree_snapshot(project)
    rejected = writing_usage._readonly_snapshot_result(before, nonempty_wal)
    assert rejected["passed"] is False
    assert rejected["unexpected"]["added"] == ["research.db-wal"]


def test_package_payload_snapshot_covers_data_and_excludes_interpreter_cache(tmp_path):
    package = tmp_path / "academic_agent"
    (package / "domains").mkdir(parents=True)
    (package / "__pycache__").mkdir()
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "domains" / "custom.json").write_text('{"version": 1}\n', encoding="utf-8")
    (package / "__pycache__" / "module.pyc").write_bytes(b"cache-one")
    before = writing_usage._package_payload_snapshot(package)

    (package / "domains" / "custom.json").write_text('{"version": 2}\n', encoding="utf-8")
    changed_payload = writing_usage._package_payload_snapshot(package)
    assert before != changed_payload
    assert set(before) == {"module.py", "domains/custom.json"}

    (package / "domains" / "custom.json").write_text('{"version": 1}\n', encoding="utf-8")
    (package / "__pycache__" / "module.pyc").write_bytes(b"cache-two")
    assert writing_usage._package_payload_snapshot(package) == before


def test_controlled_innovation_response_satisfies_the_real_host_validator(tmp_path):
    seed_demo(tmp_path)
    runner = Runner(tmp_path, offline=True)
    assert runner.run()["complete"] is True
    snapshot = freeze_research(runner.store)

    class Backend:
        name = "exchange"

        def run(self, job):
            return ModelResult(status="succeeded", backend=self.name,
                               data=writing_usage._external_response(job))

    batch = generate_candidates(
        snapshot, Backend(), timeout_seconds=5, max_input_chars=500_000,
    )
    assert len(batch.candidates) == 1
