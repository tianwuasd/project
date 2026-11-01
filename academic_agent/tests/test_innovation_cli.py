import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from academic_agent.config import atomic_json, load_config
from academic_agent.demo import seed_demo
from academic_agent.innovations import innovation_status
from academic_agent.workflow import Runner


def _cli(project: Path, *args: str, expected: int = 0, offline: bool = True,
         environment: dict[str, str] | None = None) -> dict:
    global_options = ["--project", str(project), *(["--offline"] if offline else [])]
    completed = subprocess.run(
        [sys.executable, "-m", "academic_agent.cli", *global_options, *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        env=environment,
        check=False,
    )
    assert completed.returncode == expected, (completed.stdout, completed.stderr)
    return json.loads(completed.stdout)


def _prepared_demo(project: Path) -> None:
    result = _cli(project, "demo")
    assert result["synthetic_demo"] is True


def _controlled_transport(tmp_path: Path, mode: str) -> dict[str, str]:
    hook = tmp_path / f"transport-{mode}"
    hook.mkdir()
    (hook / "sitecustomize.py").write_text(
        """import os
import academic_agent.workflow as workflow
from academic_agent.models import Work
from academic_agent.sources import SearchPage

def controlled(query, source_names, limit, proxy_url):
    mode = os.environ["ACADEMIC_AGENT_CHALLENGE_TRANSPORT"]
    if mode == "interrupted":
        raise RuntimeError("controlled interruption after durable dispatch")
    if mode == "source_failed":
        return [SearchPage(source=name, status="rate_limited", total=0, truncated=True,
                           queries=[query], errors=["controlled 429"])
                for name in source_names]
    records = ([Work(title="Controlled counterexample", year=2025,
                     source_ids={"controlled": "counterexample-1"})]
               if mode == "needs_reading" else [])
    return [SearchPage(source=name, status="ok" if records else "empty", records=records,
                       total=len(records), queries=[query]) for name in source_names]

workflow.collect_sources = controlled
""",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(hook) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["ACADEMIC_AGENT_CHALLENGE_TRANSPORT"] = mode
    return environment


def test_cli_generates_ids_and_resolves_mixed_output_path_contracts(tmp_path):
    project = tmp_path / "中文 项目 #100%"
    _prepared_demo(project)
    work_ids = [work.id for work in Runner(project, offline=True).store.works()]

    generated = _cli(project, "innovate", *(item for work_id in work_ids for item in ("--work-id", work_id)))
    assert generated["status"] == "complete"
    assert 1 <= len(generated["candidate_ids"]) <= 3
    generated_output = Path(generated["output"])
    assert generated_output.is_absolute() and generated_output.is_relative_to(project.resolve())
    assert (generated_output / "manifest.json").is_file()

    candidate_id = generated["candidate_ids"][0]
    guide = _cli(project, "guide")
    commands = [action["argv"] for action in guide["actions"]]
    assert any("历史版本" in blocker and candidate_id in blocker for blocker in guide["blockers"])
    assert ["academic-agent", "--project", str(project.resolve()), "--offline", "challenge", candidate_id,
            "--round", "1"] in commands
    assert ["academic-agent", "--project", str(project.resolve()), "--offline",
            "assess-innovation", candidate_id] in commands

    assessed = _cli(project, "assess-innovation", candidate_id)
    assert assessed["status"] == "complete"
    assert assessed["candidate_ids"] == [candidate_id]
    assessed_output = (project / assessed["output"]).resolve()
    assert assessed_output.is_relative_to(project.resolve())
    assert (assessed_output / "assessment.json").is_file()
    assert (assessed_output / "assessment-review.json").is_file()

    status = _cli(project, "innovation-status")
    row = next(item for item in status["candidates"] if item["candidate_id"] == candidate_id)
    assert row["assessment"]["status"] == "current"


def test_cli_budget_pause_is_nonzero_and_same_command_resumes_without_repeating_completed_attempts(tmp_path):
    project = tmp_path / "budget"
    _prepared_demo(project)
    config = load_config(project)
    baseline = Runner(project, offline=True).store.budget()["consumed_or_reserved_calls"]
    config.max_model_calls = baseline
    atomic_json(project / "project.json", config.model_dump())

    paused = _cli(project, "innovate", expected=2)
    assert paused["status"] == "budget_paused"
    assert paused["reason"] == "model_budget_exhausted"
    assert paused["budget"]["consumed_or_reserved_calls"] == baseline
    assert paused["budget"]["max_model_calls"] == baseline
    assert paused["resume_argv"] == [
        "academic-agent", "--project", str(project.resolve()), "--offline", "innovate",
    ]

    config.max_model_calls = baseline + 2
    atomic_json(project / "project.json", config.model_dump())
    resumed = _cli(project, "innovate")
    assert resumed["status"] == "complete"
    after_first = Runner(project, offline=True).store.budget()["consumed_or_reserved_calls"]
    repeated = _cli(project, "innovate")
    after_second = Runner(project, offline=True).store.budget()["consumed_or_reserved_calls"]
    assert repeated["candidate_ids"] == resumed["candidate_ids"]
    assert after_first == after_second == baseline + 2


@pytest.mark.parametrize(
    ("mode", "status", "expected"),
    [("needs_reading", "needs_reading", 0),
     ("needs_assessment", "needs_assessment", 0),
     ("source_failed", "source_failed", 2)],
)
def test_challenge_exit_semantics_use_controlled_no_network_transport(tmp_path, mode, status, expected):
    project = tmp_path / mode
    _prepared_demo(project)
    config = load_config(project)
    config.sources = ["crossref"]
    atomic_json(project / "project.json", config.model_dump())
    generated = _cli(project, "innovate")
    candidate_id = generated["candidate_ids"][0]
    result = _cli(project, "challenge", candidate_id, "--round", "1", expected=expected,
                  offline=False, environment=_controlled_transport(tmp_path, mode))
    assert result["status"] == status
    if expected == 0:
        assert "reason" not in result and "resume_argv" not in result
    else:
        assert result["reason"] == "source_failed"
        assert "来源" in result["explanation"]
        assert result["resume_argv"] == [
            "academic-agent", "--project", str(project.resolve()),
            "challenge", candidate_id, "--round", "1",
        ]


def test_readonly_corrupt_status_and_offline_challenge_errors_are_actionable(tmp_path):
    corrupt = tmp_path / "corrupt"
    _prepared_demo(corrupt)
    generated = _cli(corrupt, "innovate")
    (Path(generated["output"]) / "candidates.json").write_text("{}", encoding="utf-8")

    status = _cli(corrupt, "innovation-status", expected=2)
    assert status["status"] == "failed"
    assert status["reason"] == "candidate_artifact_invalid"
    assert "候选" in status["explanation"] and "修复" in status["explanation"]
    assert status["suggested_argv"] == [
        "academic-agent", "--project", str(corrupt.resolve()), "--offline", "guide",
    ]

    project = tmp_path / "offline"
    _prepared_demo(project)
    generated = _cli(project, "innovate")
    candidate_id = generated["candidate_ids"][0]
    blocked = _cli(project, "challenge", candidate_id, "--round", "1", expected=2)
    assert blocked["status"] == "failed" and blocked["reason"] == "challenge_offline"
    assert "离线" in blocked["explanation"] and blocked["budget"]["max_model_calls"] >= 1
    assert blocked["resume_argv"] == [
        "academic-agent", "--project", str(project.resolve()), "--offline",
        "challenge", candidate_id, "--round", "1",
    ]
    assert blocked["source_network_action"]["argv"] == [
        "academic-agent", "--project", str(project.resolve()),
        "challenge", candidate_id, "--round", "1",
    ]
    assert "启用" in blocked["source_network_action"]["summary"]


def test_cli_rejects_unknown_candidate_and_third_round_with_actionable_json(tmp_path):
    project = tmp_path / "errors"
    seed_demo(project)

    unknown = _cli(project, "assess-innovation", "candidate_missing", expected=1)
    assert unknown["type"]
    assert "候选" in unknown["error"]
    assert isinstance(unknown["suggested_argv"], list)

    invalid = _cli(project, "challenge", "candidate_missing", "--round", "3", expected=2)
    assert invalid["type"] == "ArgumentError"
    assert "1" in invalid["error"] and "2" in invalid["error"]


def test_innovation_guide_is_readonly_and_uses_complete_argv(tmp_path):
    project = tmp_path / "guide 中文 #100%"
    _prepared_demo(project)
    before = innovation_status(project)
    database = project / "research.db"
    database_mtime = database.stat().st_mtime_ns

    guide = _cli(project, "guide")

    assert database.stat().st_mtime_ns == database_mtime
    assert innovation_status(project) == before
    assert guide["state"]["innovation"]["generation_ready"] is True
    assert ["academic-agent", "--project", str(project.resolve()), "--offline", "innovate"] in [
        action["argv"] for action in guide["actions"]
    ]


def test_guide_recovers_the_actual_unfinished_challenge_round_and_query(tmp_path):
    completed_environment = _controlled_transport(tmp_path, "needs_assessment")
    interrupted_environment = _controlled_transport(tmp_path, "interrupted")
    failed_environment = _controlled_transport(tmp_path, "source_failed")

    interrupted_project = tmp_path / "interrupted-round-two"
    _prepared_demo(interrupted_project)
    config = load_config(interrupted_project)
    config.sources = ["crossref"]
    atomic_json(interrupted_project / "project.json", config.model_dump())
    generated = _cli(interrupted_project, "innovate")
    candidate_id = generated["candidate_ids"][0]
    first = _cli(interrupted_project, "challenge", candidate_id, "--round", "1",
                 offline=False, environment=completed_environment)
    unfinished_query = first["queries"][0]["query"]
    _cli(interrupted_project, "challenge", candidate_id, "--round", "2", expected=1,
         offline=False, environment=interrupted_environment)

    guide = _cli(interrupted_project, "guide")
    action = next(item for item in guide["actions"] if item["argv"][-4:] == [
        "challenge", candidate_id, "--round", "2",
    ])
    assert unfinished_query in action["summary"] and "第 2 轮" in action["summary"]
    assert any(unfinished_query in blocker and "第 2 轮" in blocker for blocker in guide["blockers"])

    failed_project = tmp_path / "failed-round-two"
    _prepared_demo(failed_project)
    config = load_config(failed_project)
    config.sources = ["crossref"]
    atomic_json(failed_project / "project.json", config.model_dump())
    generated = _cli(failed_project, "innovate")
    candidate_id = generated["candidate_ids"][0]
    first = _cli(failed_project, "challenge", candidate_id, "--round", "1",
                 offline=False, environment=completed_environment)
    assert first["unresolved_tasks"][0]["reason"] == "empty"
    failed = _cli(failed_project, "challenge", candidate_id, "--round", "2", expected=2,
                  offline=False, environment=failed_environment)
    failed_query = next(task["query"] for task in failed["unresolved_tasks"]
                        if task["reason"] == "rate_limited")
    guide = _cli(failed_project, "guide")
    action = next(item for item in guide["actions"] if item["argv"][-4:] == [
        "challenge", candidate_id, "--round", "2",
    ])
    assert failed_query in action["summary"] and "rate_limited" in action["summary"]
    assert any(failed_query in blocker and "rate_limited" in blocker for blocker in guide["blockers"])


def test_cli_overflow_explanation_respects_each_command_scope(tmp_path):
    assessment_project = tmp_path / "assessment-overflow"
    _prepared_demo(assessment_project)
    generated = _cli(assessment_project, "innovate")
    candidate_id = generated["candidate_ids"][0]
    config = load_config(assessment_project)
    config.max_input_chars = 100
    atomic_json(assessment_project / "project.json", config.model_dump())

    assessed = _cli(assessment_project, "assess-innovation", candidate_id, expected=2)
    assert assessed["status"] == "failed" and assessed["reason"] == "model_input_too_large"
    assert "--work-id" not in assessed["explanation"]
    assert "候选前提" in assessed["explanation"] and "反例" in assessed["explanation"]
    assert "范围" in assessed["explanation"] and "输入上限" in assessed["explanation"]
    assert assessed["resume_argv"] == [
        "academic-agent", "--project", str(assessment_project.resolve()), "--offline",
        "assess-innovation", candidate_id,
    ]

    generation_project = tmp_path / "generation-overflow"
    _prepared_demo(generation_project)
    config = load_config(generation_project)
    config.max_input_chars = 100
    atomic_json(generation_project / "project.json", config.model_dump())
    generated = _cli(generation_project, "innovate", expected=2)
    assert generated["status"] == "failed" and generated["reason"] == "model_input_too_large"
    assert "--work-id" in generated["explanation"]
