import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from test_domain_analysis import DomainBackend
from test_writing_analysis import WritingBackend

from academic_agent.backends import DemoBackend
from academic_agent.config import atomic_json, load_config
from academic_agent.demo import seed_demo
from academic_agent.domain import load_domain_version
from academic_agent.filesystem import io_path
from academic_agent.models import ModelJob
from academic_agent.research_artifacts import load_research
from academic_agent.workflow import Runner
from academic_agent.writing import load_writing_version


def _run_cli(
    project: Path,
    *args: str,
    expected: int = 0,
    environment: dict[str, str] | None = None,
) -> tuple[dict, subprocess.CompletedProcess[str]]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "academic_agent.cli",
            "--project",
            str(project),
            "--offline",
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
        env=environment,
        check=False,
    )
    assert completed.returncode == expected, (completed.stdout, completed.stderr)
    return json.loads(completed.stdout), completed


def _demo_with_budget(project: Path, calls: int = 30) -> list[str]:
    result, _ = _run_cli(project, "demo")
    assert result["synthetic_demo"] is True
    config = load_config(project)
    config.max_model_calls = calls
    atomic_json(project / "project.json", config.model_dump())
    return [
        work.id
        for work in sorted(
            Runner(project, offline=True).store.works(), key=lambda item: item.title
        )
    ]


def _selected_args(question: str, work_ids: list[str]) -> list[str]:
    return [
        "--question",
        question,
        *(part for work_id in work_ids for part in ("--work-id", work_id)),
    ]


def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _all_strings(key)
            yield from _all_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_strings(item)


def _create_domain(project: Path, question: str, work_ids: list[str]) -> Path:
    result, _ = _run_cli(
        project,
        "domain-extract",
        *_selected_args(question, work_ids),
        "--domain",
        "builtin:scheduling",
    )
    assert result["status"] == "complete"
    output = (project / result["output"]).resolve()
    assert output.parent == (project / "analyses" / "domain").resolve()
    comparison = io_path(output / "comparison.md").read_text(encoding="utf-8")
    assert "40 [supported]" in comparison and "60 [supported]" in comparison
    assert "10 to 8 simulation units" in comparison
    assert "10 to 11 simulation units" in comparison
    return output


def _create_outline_and_draft(
    project: Path, question: str, work_ids: list[str], domain: Path
) -> tuple[Path, Path]:
    outline, _ = _run_cli(
        project,
        "writing-outline",
        *_selected_args(question, work_ids),
        "--domain-version",
        str(domain),
    )
    assert outline["status"] == "complete"
    outline_path = (project / outline["output"]).resolve()
    manuscript, _ = _run_cli(project, "writing-draft", str(outline_path))
    assert manuscript["status"] == "complete"
    manuscript_path = (project / manuscript["output"]).resolve()
    return outline_path, manuscript_path


def test_complete_demo_writing_cli_journey_and_exact_human_revision(tmp_path):
    project = tmp_path
    for index in range(4):
        project /= f"深层路径 {index} with spaces #100% abcdefghijklmnopqrstuvwxyz"
    project /= "中文 writing project #100%"
    assert len(str(project.resolve())) > 260
    question = "How do the two scheduling reports differ by workload?"
    work_ids = _demo_with_budget(project)
    domain = _create_domain(project, question, work_ids)
    outline, manuscript = _create_outline_and_draft(project, question, work_ids, domain)

    outline_artifact = load_writing_version(outline)
    manuscript_artifact = load_writing_version(manuscript)
    assert outline_artifact.revision.extent == "outline"
    assert outline_artifact.revision.stage == "reviewed"
    assert manuscript_artifact.revision.extent == "manuscript"
    assert manuscript_artifact.revision.stage == "reviewed"
    report = io_path(manuscript / "report.md").read_text(encoding="utf-8")
    assert all(value in report for value in ("40", "60", "10", "8", "11"))
    assert "real systems" in report

    status, _ = _run_cli(
        project,
        "writing-status",
        "--document-id",
        manuscript_artifact.revision.document_id,
    )
    assert status["status"] == "complete"
    versions = status["documents"][0]["versions"]
    assert {(row["extent"], row["stage"]) for row in versions} >= {
        ("outline", "reviewed"),
        ("manuscript", "reviewed"),
    }

    edit_path = project / "编辑副本 with spaces.json"
    exported, _ = _run_cli(
        project, "writing-export-edit", str(manuscript), "--output", str(edit_path)
    )
    assert exported == {"status": "complete", "output": str(edit_path.resolve())}
    editable = json.loads(io_path(edit_path).read_bytes())
    changed_heading_id = editable["outline"][0]["id"]
    changed_id = editable["paragraphs"][1]["id"]
    editable["outline"][0]["text"] = "All schedulers always improve"
    exact_text = "Study A used 400 simulated participants.\r\n因此两种策略总会改进。🙂 e\u0301"
    editable["paragraphs"][1]["text"] = exact_text
    raw = b"\xef\xbb\xbf" + json.dumps(
        editable, ensure_ascii=False, indent=2, allow_nan=False
    ).replace("\n", "\r\n").encode("utf-8")
    io_path(edit_path).write_bytes(raw)

    imported, _ = _run_cli(
        project,
        "writing-import",
        str(manuscript),
        str(edit_path),
        "--author",
        "Researcher 张",
        "--reason",
        "Preserve this exact human correction for independent review.",
    )
    assert imported["status"] == "complete"
    pending_path = (project / imported["output"]).resolve()
    pending = load_writing_version(pending_path)
    assert pending.revision.stage == "pending"
    assert set(pending.revision.changed_ids) == {changed_heading_id, changed_id}
    assert load_research(pending_path)["files"]["raw-edit.bin"] == raw

    reviewed, _ = _run_cli(project, "writing-review", str(pending_path))
    assert reviewed["status"] == "complete"
    assert reviewed["support_review"]["status"] == "issues_found"
    assert reviewed["support_review"]["decisions"]["contradicted"] >= 1
    reviewed_path = (project / reviewed["output"]).resolve()
    reviewed_artifact = load_writing_version(reviewed_path)
    reviewed_text = "".join(
        unit.text for block in reviewed_artifact.draft.blocks for unit in block.units
    )
    assert exact_text in reviewed_text
    decisions = {item.unit_id: item.decision for item in reviewed_artifact.review.items}
    changed_heading = next(
        block for block in reviewed_artifact.outline.blocks if block.id == changed_heading_id
    )
    changed_paragraph = next(
        block for block in reviewed_artifact.draft.blocks if block.id == changed_id
    )
    assert changed_heading.units[0].kind == "connector"
    assert decisions[changed_heading.units[0].id] == "contradicted"
    connector = next(unit for unit in changed_paragraph.units if unit.kind == "connector")
    assert connector.text.startswith("因此")
    assert decisions[connector.id] == "contradicted"
    evidence = json.loads(io_path(reviewed_path / "evidence.json").read_bytes())
    provenance = next(
        row for row in evidence["block_provenance"] if row["block_id"] == changed_id
    )
    assert provenance["author"] == "Researcher 张"
    assert evidence["review_provenance"]["source"] == "independent_model_call"

    source = project / next(iter(reviewed_artifact.context.research.documents.values())).source_path
    io_path(source).write_bytes(io_path(source).read_bytes() + b"\nsource refresh")
    budget_before = Runner(project, offline=True).store.budget()["consumed_or_reserved_calls"]
    stale_import, _ = _run_cli(
        project,
        "writing-import",
        str(manuscript),
        str(edit_path),
        "--author",
        "Researcher 张",
        "--reason",
        "Preserve this exact human correction for independent review.",
        expected=2,
    )
    extraction_offline = [
        "academic-agent",
        "--project",
        str(project.resolve()),
        "--offline",
        "domain-extract",
        *_selected_args(question, work_ids),
        "--domain",
        "builtin:scheduling",
    ]
    assert stale_import["status"] == "stale"
    assert stale_import["reason"] == "domain_rebind_required"
    assert stale_import["resume_argv"] == extraction_offline
    assert stale_import["actions"] == [{
        "summary": "先执行此命令并取得实际、已验证的领域版本输出目录。",
        "argv": extraction_offline,
    }]
    assert stale_import["next_step"]["parent_version"] == str(manuscript)
    assert stale_import["next_step"]["edit_file"] == str(edit_path.resolve())
    assert "<DOMAIN_VERSION" not in json.dumps(stale_import, ensure_ascii=False)
    assert Runner(project, offline=True).store.budget()["consumed_or_reserved_calls"] == budget_before

    guide, _ = _run_cli(project, "guide")
    extraction = [
        "academic-agent",
        "--project",
        str(project.resolve()),
        "--offline",
        "domain-extract",
        *_selected_args(question, work_ids),
        "--domain",
        "builtin:scheduling",
    ]
    assert extraction in [action["argv"] for action in guide["actions"]]
    recovery = next(
        row for row in guide["state"]["writing"]["recovery"]
        if row["version"] == reviewed_path.name
    )
    assert recovery["domain_extract_argv"] == extraction
    assert recovery["after_extract"] == {
        "summary": (
            "领域提取完成后，将它返回的实际 output 目录作为 --domain-version；"
            "保持下面的 parent、原始编辑 bytes、作者、理由和删除清单不变。"
        ),
        "parent_version": str(manuscript),
        "edit_file": str((reviewed_path / "raw-edit.bin").resolve()),
        "author": "Researcher 张",
        "reason": "Preserve this exact human correction for independent review.",
        "delete_ids": [],
        "then": "对 writing-import 返回的实际 pending output 运行 writing-review。",
    }
    assert "<DOMAIN_VERSION" not in json.dumps(guide, ensure_ascii=False)
    logical_values = [
        status,
        exported,
        imported,
        reviewed,
        stale_import,
        guide,
        outline_artifact.model_dump(mode="python"),
        manuscript_artifact.model_dump(mode="python"),
        reviewed_artifact.model_dump(mode="python"),
    ]
    assert all("\\\\?\\" not in item for item in _all_strings(logical_values))

    from academic_agent.guide import project_guide

    default_guide = project_guide(project)
    assert all("--offline" not in action["argv"] for action in default_guide["actions"])


def test_invalid_writing_argument_combinations_fail_before_project_state(tmp_path):
    project = tmp_path / "must stay absent"
    invalid_cases = [
        (
            "writing-outline",
            "--question",
            "Q",
            "--work-id",
            "work_a",
            "--domain-version",
            str(tmp_path / "domain"),
            "--candidate-id",
            "innovation_" + "1" * 32,
        ),
        (
            "writing-outline",
            "--question",
            "Q",
            "--work-id",
            "work_a",
            "--domain-version",
            str(tmp_path / "domain"),
            "--assessment-version",
            str(tmp_path / "assessment"),
        ),
    ]
    for case in invalid_cases:
        result, completed = _run_cli(project, *case, expected=2)
        assert result["type"] == "ArgumentError"
        assert "candidate" in result["error"].lower()
        assert completed.stderr == ""
        assert not project.exists()

    result, completed = _run_cli(
        project,
        "domain-extract",
        "--question",
        "Q",
        "--work-id",
        "work_a",
        "--domain",
        "relative-domain.json",
        expected=2,
    )
    assert result["type"] == "ArgumentError"
    assert "absolute" in result["error"].lower()
    assert completed.stderr == ""
    assert not project.exists()


def test_outline_rejects_explicit_question_or_work_mismatch_without_model_use(tmp_path):
    project = tmp_path / "mismatch"
    question = "Original bounded question"
    work_ids = _demo_with_budget(project)
    domain = _create_domain(project, question, work_ids)
    before = Runner(project, offline=True).store.budget()["consumed_or_reserved_calls"]
    optional_args = [
        "--trends",
        str(tmp_path / "v-trends"),
        "--candidate-version",
        str(tmp_path / "v-candidate"),
        "--candidate-id",
        "innovation_" + "1" * 32,
        "--assessment-version",
        str(tmp_path / "v-assessment"),
    ]

    mismatch, _ = _run_cli(
        project,
        "writing-outline",
        *_selected_args("Different question", work_ids),
        "--domain-version",
        str(domain),
        *optional_args,
        expected=2,
    )
    after = Runner(project, offline=True).store.budget()["consumed_or_reserved_calls"]
    assert mismatch["status"] == "failed"
    assert mismatch["reason"] == "domain_selection_mismatch"
    assert mismatch["gaps"]
    assert mismatch["resume_argv"] == [
        "academic-agent",
        "--project",
        str(project.resolve()),
        "--offline",
        "writing-outline",
        *_selected_args(question, work_ids),
        "--domain-version",
        str(domain),
        "--trends",
        str((tmp_path / "v-trends").resolve()),
        "--candidate-version",
        str((tmp_path / "v-candidate").resolve()),
        "--candidate-id",
        "innovation_" + "1" * 32,
        "--assessment-version",
        str((tmp_path / "v-assessment").resolve()),
    ]
    assert before == after

    reversed_ids = list(reversed(work_ids))
    work_mismatch, _ = _run_cli(
        project,
        "writing-outline",
        *_selected_args(question, reversed_ids),
        "--domain-version",
        str(domain),
        *optional_args,
        expected=2,
    )
    assert work_mismatch["reason"] == "domain_selection_mismatch"
    assert work_mismatch["resume_argv"] == mismatch["resume_argv"]
    assert Runner(project, offline=True).store.budget()["consumed_or_reserved_calls"] == before


def test_offline_guide_preserves_mode_in_writing_recovery_and_default_call_does_not(tmp_path):
    project = tmp_path / "offline guide"
    question = "How do the two scheduling reports differ by workload?"
    work_ids = _demo_with_budget(project)
    domain = _create_domain(project, question, work_ids)
    _, manuscript = _create_outline_and_draft(project, question, work_ids, domain)
    artifact = load_writing_version(manuscript)
    source = project / next(iter(artifact.context.research.documents.values())).source_path
    source.write_bytes(source.read_bytes() + b"\nsource refresh")

    guide, _ = _run_cli(project, "guide")
    recovery = next(
        action for action in guide["actions"] if "domain-extract" in action["argv"]
    )
    assert recovery["argv"][3:5] == ["--offline", "domain-extract"]

    from academic_agent.guide import project_guide

    direct = project_guide(project)
    assert all("--offline" not in action["argv"] for action in direct["actions"])


def test_outline_cli_retains_all_explicit_optional_version_pins(tmp_path):
    project = tmp_path / "all explicit pins"
    question = "Compare the bounded scheduling evidence and a proposed next test."
    work_ids = _demo_with_budget(project, calls=50)
    domain = _create_domain(project, question, work_ids)
    trends, _ = _run_cli(project, "trends")
    trend_path = Path(trends["trends"]).resolve()
    innovation, _ = _run_cli(project, "innovate")
    candidate_id = innovation["candidate_ids"][0]
    candidate_path = Path(innovation["output"]).resolve()
    assessment, _ = _run_cli(project, "assess-innovation", candidate_id)
    assessment_path = (project / assessment["output"]).resolve()

    result, _ = _run_cli(
        project,
        "writing-outline",
        *_selected_args(question, work_ids),
        "--domain-version",
        str(domain),
        "--trends",
        str(trend_path),
        "--candidate-version",
        str(candidate_path),
        "--candidate-id",
        candidate_id,
        "--assessment-version",
        str(assessment_path),
    )

    assert result["status"] == "complete"
    artifact = load_writing_version(project / result["output"])
    assert artifact.context.selection.trend_directory == str(trend_path)
    assert artifact.context.selection.candidate_directory == str(candidate_path)
    assert artifact.context.selection.candidate_id == candidate_id
    assert artifact.context.selection.assessment_directory == str(assessment_path)
    assert artifact.context.trend.artifact.version_path == str(trend_path)
    assert artifact.context.candidate.artifact.version_path == str(candidate_path)
    assert artifact.context.assessment.artifact.version_path == str(assessment_path)
    assert artifact.domain_pin.version_path == str(domain)


def test_budget_pause_is_nonzero_actionable_and_does_not_raise_budget(tmp_path):
    project = tmp_path / "budget pause"
    question = "Bounded question"
    work_ids = _demo_with_budget(project)
    runner = Runner(project, offline=True)
    baseline = runner.store.budget()["consumed_or_reserved_calls"]
    config = load_config(project)
    config.max_model_calls = baseline
    atomic_json(project / "project.json", config.model_dump())

    paused, _ = _run_cli(
        project,
        "domain-extract",
        *_selected_args(question, work_ids),
        "--domain",
        "builtin:scheduling",
        expected=2,
    )
    assert paused["status"] == "paused_budget"
    assert paused["reason"] == "model_budget_exhausted"
    assert paused["budget"] == {
        "consumed_or_reserved_calls": baseline,
        "max_model_calls": baseline,
    }
    assert paused["resume_argv"] == [
        "academic-agent",
        "--project",
        str(project.resolve()),
        "--offline",
        "domain-extract",
        *_selected_args(question, work_ids),
        "--domain",
        "builtin:scheduling",
    ]


def test_writing_status_export_and_guide_do_not_construct_runner_store_or_backend(tmp_path):
    project = tmp_path / "readonly 中文"
    question = "Bounded question"
    work_ids = _demo_with_budget(project)
    domain = _create_domain(project, question, work_ids)
    _, manuscript = _create_outline_and_draft(project, question, work_ids, domain)
    export_path = project / "readonly-copy.json"
    database = project / "research.db"
    before = (database.stat().st_mtime_ns, hashlib.sha256(database.read_bytes()).hexdigest())

    hook = tmp_path / "forbid-stateful-construction"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        """import academic_agent.backends as backends
import academic_agent.store as store
import academic_agent.workflow as workflow

class Forbidden:
    def __init__(self, *args, **kwargs):
        raise AssertionError('read-only command constructed stateful service')

workflow.Runner = Forbidden
store.Store = Forbidden
backends.DemoBackend = Forbidden
backends.ExchangeBackend = Forbidden
backends.CodexBackend = Forbidden
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(hook) + os.pathsep + environment.get("PYTHONPATH", "")

    status, _ = _run_cli(project, "writing-status", environment=environment)
    assert status["status"] == "complete"
    exported, _ = _run_cli(
        project,
        "writing-export-edit",
        str(manuscript),
        "--output",
        str(export_path),
        environment=environment,
    )
    assert exported["output"] == str(export_path.resolve())
    guide, _ = _run_cli(project, "guide", environment=environment)
    assert guide["state"]["writing"]["documents"]
    after = (database.stat().st_mtime_ns, hashlib.sha256(database.read_bytes()).hexdigest())
    assert before == after


def _answer_next_exchange(project: Path, backend) -> ModelJob:
    request_path = next(
        path
        for path in (project / "exchange").glob("*.request.json")
        if not path.with_name(path.name.replace(".request.", ".response.")).exists()
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    job = ModelJob.model_validate(request["job"])
    data = backend.run(job).data
    response_path = request_path.with_name(request_path.name.replace(".request.", ".response."))
    response_path.write_text(
        json.dumps({"fingerprint": request["fingerprint"], "data": data}, ensure_ascii=False),
        encoding="utf-8",
    )
    return job


def test_exchange_external_data_pack_changes_fields_columns_and_writing_structure(tmp_path):
    project = tmp_path / "external pack"
    seed_demo(project, backend="exchange")
    config = load_config(project)
    config.max_model_calls = 30
    atomic_json(project / "project.json", config.model_dump())
    Runner(project, offline=True, backend=DemoBackend()).run()
    work_ids = [
        work.id
        for work in sorted(Runner(project, offline=True).store.works(), key=lambda item: item.title)
    ]
    question = "What observation units and transfer limits are reported?"
    external = project / "inputs" / "external-domain.json"
    external.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "writing-domain-1",
                "name": "external_scope",
                "version": "1",
                "terminology": [
                    {"term": "observation unit", "meaning": "The unit actually studied."}
                ],
                "search_hints": ["simulation transfer scope"],
                "screening_hints": ["Retain explicit limitations."],
                "fields": [
                    {
                        "id": "observation_unit",
                        "label": "Observation unit",
                        "description": "Reported unit of observation.",
                        "value_type": "text",
                        "unit": None,
                    },
                    {
                        "id": "transfer_limit",
                        "label": "Transfer limit",
                        "description": "Reported limit on comparison across two workloads and real systems.",
                        "value_type": "text",
                        "unit": None,
                    },
                ],
                "comparison_columns": ["transfer_limit", "observation_unit"],
                "evaluation_criteria": [
                    {"id": "scope_transfer", "text": "Keep simulation scope explicit."}
                ],
                "writing_structure": [
                    {
                        "id": "boundary_argument",
                        "title": "Boundary argument",
                        "purpose": "Compare limits before proposing transfer.",
                    }
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    domain_args = (
        "domain-extract",
        *_selected_args(question, work_ids),
        "--domain",
        str(external.resolve()),
    )
    domain_jobs = []
    class ExternalDomainBackend(DomainBackend):
        def run(self, job):
            result = super().run(job)
            if job.payload["task"] == "domain-extract-v1" and job.payload["research"]["work"]["title"].endswith("A"):
                transfer = next(item for item in result.data["fields"] if item["field_id"] == "transfer_limit")
                transfer.update(status="not_reported", value=None, evidence_ids=[])
            return result

    for _ in range(4):
        pending, _ = _run_cli(project, *domain_args, expected=2)
        assert pending["status"] == "needs_input", pending
        domain_jobs.append(_answer_next_exchange(project, ExternalDomainBackend()))
    completed, _ = _run_cli(project, *domain_args)
    domain_path = (project / completed["output"]).resolve()
    artifact = load_domain_version(domain_path)
    assert [field.id for field in artifact.context.pack.pack.fields] == [
        "observation_unit",
        "transfer_limit",
    ]
    assert artifact.context.pack.pack.comparison_columns == [
        "transfer_limit",
        "observation_unit",
    ]
    assert artifact.context.pack.pack.writing_structure[0].id == "boundary_argument"
    work_a = next(work_id for work_id in work_ids if artifact.context.research.works[work_id].title.endswith("A"))
    work_b = next(work_id for work_id in work_ids if artifact.context.research.works[work_id].title.endswith("B"))
    fields_a = {field.field_id: field for field in artifact.fields[work_a]}
    fields_b = {field.field_id: field for field in artifact.fields[work_b]}
    assert fields_a["observation_unit"].value == "simulated jobs"
    assert fields_a["observation_unit"].evidence_ids
    assert fields_a["transfer_limit"].status == "not_reported"
    assert fields_a["transfer_limit"].value is None
    assert fields_a["transfer_limit"].evidence_ids == []
    assert fields_b["observation_unit"].value == "simulated jobs"
    assert fields_b["observation_unit"].evidence_ids
    assert fields_b["transfer_limit"].value == "cannot establish effects in real systems"
    assert fields_b["transfer_limit"].evidence_ids
    comparison = (domain_path / "comparison.md").read_text(encoding="utf-8")
    header = next(line for line in comparison.splitlines() if line.startswith("| work_id"))
    assert [cell.strip() for cell in header.strip("|").split("|")] == [
        "work_id",
        "transfer_limit",
        "observation_unit",
    ]
    row_a = next(line for line in comparison.splitlines() if line.startswith(f"| {work_a} "))
    row_b = next(line for line in comparison.splitlines() if line.startswith(f"| {work_b} "))
    assert "not reported [supported]" in row_a
    assert "simulated jobs [supported]" in row_a
    assert "cannot establish effects in real systems [supported]" in row_b
    assert {job.payload["task"] for job in domain_jobs} == {
        "domain-extract-v1",
        "domain-review-v1",
    }

    outline_args = (
        "writing-outline",
        *_selected_args(question, work_ids),
        "--domain-version",
        str(domain_path),
    )
    writing_jobs = []
    for _ in range(2):
        pending, _ = _run_cli(project, *outline_args, expected=2)
        assert pending["status"] == "needs_input", pending
        writing_jobs.append(_answer_next_exchange(project, WritingBackend()))
    outlined, _ = _run_cli(project, *outline_args)
    outlined_artifact = load_writing_version(project / outlined["output"])
    assert outlined_artifact.context.pack.pack.writing_structure[0].model_dump() == {
        "id": "boundary_argument",
        "title": "Boundary argument",
        "purpose": "Compare limits before proposing transfer.",
    }
    assert [job.payload["task"] for job in writing_jobs] == [
        "writing-outline-v1",
        "writing-review-v1",
    ]
