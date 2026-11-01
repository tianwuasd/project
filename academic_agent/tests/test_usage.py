import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from academic_agent import __version__, usage
from academic_agent.config import atomic_json, create_project, load_config
from academic_agent.demo import FIXTURES, seed_demo
from academic_agent.guide import project_guide
from academic_agent.models import ProjectConfig, Work
from academic_agent.store import Store
from academic_agent.usage import run_self_test
from academic_agent.workflow import Runner


def _action_commands(guide: dict) -> list[list[str]]:
    return [action["argv"] for action in guide["actions"]]


@pytest.mark.skipif(os.name != "nt", reason="Windows MAX_PATH regression")
def test_usage_json_reader_handles_exact_windows_max_path_after_directory_publish(tmp_path):
    staging = tmp_path / "s"
    staging.mkdir()
    (staging / "computation.json").write_text('{"count":5}', encoding="utf-8")
    fixed_length = len(str(tmp_path.resolve())) + 1 + len("\\computation.json")
    component_length = 260 - fixed_length
    assert 1 <= component_length <= 255
    destination = tmp_path / ("v" + "x" * (component_length - 1))
    staging.rename(destination)
    published = destination / "computation.json"
    assert len(str(published.resolve())) == 260

    assert usage._json_file(published) == {"count": 5}


def test_guide_for_missing_and_unopened_project_is_read_only(tmp_path):
    missing = tmp_path / "尚未创建 项目"
    result = project_guide(missing)
    assert result["blockers"] == ["项目尚未创建"]
    assert _action_commands(result) == [[
        "academic-agent", "--project", str(missing.resolve()), "init", "--title", "研究标题"
    ]]
    assert not missing.exists()

    unopened = tmp_path / "只有配置 项目"
    create_project(unopened, ProjectConfig(title="空项目", backend="demo", sources=[], queries=[]))
    before = sorted(path.relative_to(unopened) for path in unopened.rglob("*"))
    result = project_guide(unopened)
    after = sorted(path.relative_to(unopened) for path in unopened.rglob("*"))
    assert before == after == [Path("project.json")]
    assert "尚无文献" in result["summary"]
    assert all(isinstance(part, str) for command in _action_commands(result) for part in command)
    assert not (unopened / "research.db").exists()


def test_guide_uses_individual_safe_argv_and_tracks_read_review_export_staleness(tmp_path):
    root = tmp_path / "中文 空格"
    create_project(root, ProjectConfig(title="Guide", backend="demo", sources=[], queries=["graph \"model\""]))
    store = Store(root)
    pending = store.upsert_work(Work(title="Pending paper"))
    pending_guide = project_guide(root)
    commands = _action_commands(pending_guide)
    assert ["academic-agent", "--project", str(root.resolve()), "screen", pending.id,
            "--decision", "include", "--reason", "请填写逐篇筛选理由"] in commands
    assert all("--all" not in command for command in commands)

    store.screen(pending.id, "include", "matches")
    no_text = project_guide(root)
    assert any(command[-3:] == ["attach", pending.id, "文献全文.pdf"] for command in _action_commands(no_text))

    document = root / "original.txt"
    document.write_text(FIXTURES[0][1], encoding="utf-8")
    Runner(root, offline=True).attach(pending.id, document)
    unread = project_guide(root)
    assert any(command[-3:] == ["read", "--work-id", pending.id] for command in _action_commands(unread))

    runner = Runner(root, offline=True)
    assert runner.read_selected(work_id=pending.id)["results"][0]["status"] == "ok"
    unreviewed = project_guide(root)
    assert any(command[-3:] == ["review", "--work-id", pending.id] for command in _action_commands(unreviewed))

    assert runner.review_selected(work_id=pending.id)["results"][0]["status"] == "ok"
    current = project_guide(root)
    assert any(command[-1] == "export" for command in _action_commands(current))
    assert any(command[-1] == "trends" for command in _action_commands(current))
    runner.export()
    assert project_guide(root)["state"]["latest_export_current"] is True

    changed_document = root / "changed.txt"
    changed_document.write_text(FIXTURES[0][1].replace("40 simulated", "41 simulated"), encoding="utf-8")
    runner.attach(pending.id, changed_document)
    stale = project_guide(root)
    assert stale["state"]["latest_export_current"] is False
    assert any("过期" in blocker for blocker in stale["blockers"])
    assert any(command[-3:] == ["read", "--work-id", pending.id] for command in _action_commands(stale))


def test_guide_reports_budget_and_exchange_input_without_running_backend(tmp_path):
    exchange = tmp_path / "exchange"
    seed_demo(exchange, backend="exchange")
    config = load_config(exchange)
    config.max_model_calls = 1
    atomic_json(exchange / "project.json", config.model_dump())
    result = Runner(exchange, offline=True).read_selected()
    assert any(item["status"] == "failed" for item in result["results"])
    before = (exchange / "research.db").stat().st_mtime_ns
    guide = project_guide(exchange)
    assert (exchange / "research.db").stat().st_mtime_ns == before
    assert any("人工响应" in blocker for blocker in guide["blockers"])

    budget = tmp_path / "budget"
    seed_demo(budget)
    config = load_config(budget)
    config.max_model_calls = 1
    atomic_json(budget / "project.json", config.model_dump())
    Runner(budget, offline=True).read_selected()
    assert any("预算" in blocker for blocker in project_guide(budget)["blockers"])


def test_guide_discloses_failed_sources_and_suggests_explicit_query_retry(tmp_path):
    create_project(tmp_path, ProjectConfig(title="Sources", backend="demo", queries=["graph methods"]))
    store = Store(tmp_path)
    store.log("search", {
        "query_key": "fixture",
        "summary": {
            "query": "graph methods", "retained": 0, "dropped_at_project_cap": 0,
            "sources": [{"source": "crossref", "status": "rate_limited", "records": 0,
                         "truncated": True, "errors": ["429"]}],
        },
        "responses": [],
    })
    guide = project_guide(tmp_path)
    assert any("来源" in blocker and "Crossref" in blocker and "rate_limited" in blocker
               for blocker in guide["blockers"])
    assert ["academic-agent", "--project", str(tmp_path.resolve()), "search", "graph methods"] \
        in _action_commands(guide)


def test_guide_rejects_card_for_replaced_source_abstract_without_writing(tmp_path):
    create_project(tmp_path, ProjectConfig(title="Abstract freshness", backend="demo", sources=[]))
    runner = Runner(tmp_path, offline=True)
    work = runner.store.upsert_work(Work(
        title="Abstract study", arxiv_id="2601.00001v1", abstract=FIXTURES[0][1],
        screening="include", screening_reason="范围匹配",
    ))
    assert runner.run()["complete"] is True
    old_document = runner.store.active_document(work.id)
    runner.store.upsert_work(Work(
        title="Abstract study", arxiv_id="2601.00001v2",
        abstract=FIXTURES[0][1].replace("40 simulated", "41 simulated"),
    ))

    guide = project_guide(tmp_path)

    assert runner.store.active_document(work.id) == old_document
    assert any(command[-3:] == ["read", "--work-id", work.id] for command in _action_commands(guide))
    assert any("阅读卡" in blocker for blocker in guide["blockers"])


def test_guide_uses_current_abstract_in_memory_without_preparing_or_replacing_documents(tmp_path):
    fresh = tmp_path / "fresh"
    create_project(fresh, ProjectConfig(title="Fresh abstract", backend="demo", sources=[]))
    fresh_store = Store(fresh)
    fresh_work = fresh_store.upsert_work(Work(
        title="Fresh", abstract=FIXTURES[0][1], screening="include", screening_reason="范围匹配",
    ))
    fresh_before = {path.relative_to(fresh): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in fresh.rglob("*") if path.is_file()}

    fresh_guide = project_guide(fresh)

    assert fresh_store.active_document(fresh_work.id) is None
    assert fresh_store.evidence(fresh_work.id) == []
    assert fresh_before == {path.relative_to(fresh): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in fresh.rglob("*") if path.is_file()}
    assert any(command[-3:] == ["read", "--work-id", fresh_work.id]
               for command in _action_commands(fresh_guide))
    assert not any(command[-3:] == ["attach", fresh_work.id, "文献全文.pdf"]
                   for command in _action_commands(fresh_guide))

    updated = tmp_path / "updated"
    create_project(updated, ProjectConfig(title="Updated abstract", backend="demo", sources=[]))
    runner = Runner(updated, offline=True)
    updated_work = runner.store.upsert_work(Work(
        title="Updated", arxiv_id="2601.00002v1", screening="include", screening_reason="范围匹配",
    ))
    assert runner.prepare()["documents"][updated_work.id] == "metadata_only"
    old_document = runner.store.active_document(updated_work.id)
    runner.store.upsert_work(Work(title="Updated", arxiv_id="2601.00002v2", abstract=FIXTURES[0][1]))

    updated_guide = project_guide(updated)

    assert runner.store.active_document(updated_work.id) == old_document
    assert runner.store.evidence(updated_work.id) == []
    assert any(command[-3:] == ["read", "--work-id", updated_work.id]
               for command in _action_commands(updated_guide))
    assert not any(command[-3:] == ["attach", updated_work.id, "文献全文.pdf"]
                   for command in _action_commands(updated_guide))

    blank = updated / "blank.pdf"
    usage._write_pdf(blank, None)
    runner.attach(updated_work.id, blank)
    ocr_guide = project_guide(updated)
    assert any(command[-3:] == ["attach", updated_work.id, "文献全文.pdf"]
               for command in _action_commands(ocr_guide))
    assert not any(command[-3:] == ["read", "--work-id", updated_work.id]
                   for command in _action_commands(ocr_guide))


def test_guide_readonly_uri_preserves_literal_fragment_and_percent_project_path(tmp_path):
    project = tmp_path / "chapter#1%2Fresearch"
    create_project(project, ProjectConfig(title="URI path", backend="demo", sources=[]))
    store = Store(project)
    work = store.upsert_work(Work(title="URI record", year=2025))

    def tree() -> dict[Path, str]:
        return {path.relative_to(tmp_path): "directory" if path.is_dir()
                else hashlib.sha256(path.read_bytes()).hexdigest()
                for path in tmp_path.rglob("*")}

    before = tree()
    guide = project_guide(project)
    after = tree()

    changed = {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}
    allowed = {
        Path(project.name) / "research.db-wal",
        Path(project.name) / "research.db-shm",
    }
    assert changed <= allowed
    assert not (tmp_path / "chapter").exists()
    assert guide["state"]["works"] == 1
    assert guide["state"]["pending"] == 1
    assert any(work.id in action["argv"] for action in guide["actions"])


def test_guide_reads_latest_committed_wal_without_mutating_research_rows(tmp_path):
    create_project(tmp_path, ProjectConfig(title="Live WAL", backend="demo", sources=[]))
    store = Store(tmp_path)

    def research_rows(connection) -> dict[str, list[tuple]]:
        tables = ("meta", "works", "documents", "evidence", "logs", "jobs", "attempts")
        return {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")]
                for table in tables}

    with store.connect() as keeper:
        keeper.execute("SELECT count(*) FROM works").fetchone()
        keeper.commit()
        assert keeper.in_transaction is False
        work = store.upsert_work(Work(title="Committed in WAL", year=2026))
        assert (tmp_path / "research.db-wal").stat().st_size > 0
        before = research_rows(keeper)

        guide = project_guide(tmp_path)

        after = research_rows(keeper)
        assert guide["state"]["works"] == 1
        assert guide["state"]["pending"] == 1
        assert any(work.id in action["argv"] for action in guide["actions"])
        assert after == before


def test_guide_offers_trends_for_metadata_only_and_export_for_partial_reviewed_results(tmp_path):
    metadata = tmp_path / "metadata"
    create_project(metadata, ProjectConfig(title="Metadata trends", backend="demo", sources=[]))
    Store(metadata).upsert_work(Work(
        title="Metadata only", year=2024, screening="include", screening_reason="趋势语料",
    ))
    metadata_commands = _action_commands(project_guide(metadata))
    assert any(command[-1] == "trends" for command in metadata_commands)

    mixed = tmp_path / "mixed"
    create_project(mixed, ProjectConfig(title="Mixed progress", backend="demo", sources=[]))
    runner = Runner(mixed, offline=True)
    works = [runner.store.upsert_work(Work(
        title=f"Fixture {index}", screening="include", screening_reason="范围匹配",
    )) for index in range(2)]
    for index, work in enumerate(works):
        document = mixed / f"fixture-{index}.txt"
        document.write_text(FIXTURES[index][1], encoding="utf-8")
        runner.attach(work.id, document)
    assert runner.read_selected(work_id=works[0].id)["results"][0]["status"] == "ok"
    assert runner.review_selected(work_id=works[0].id)["results"][0]["status"] == "ok"

    guide = project_guide(mixed)
    commands = _action_commands(guide)
    assert any(command[-1] == "export" for command in commands)
    assert any(command[-1] == "trends" for command in commands)
    assert any(command[-3:] == ["read", "--work-id", works[1].id] for command in commands)


@pytest.mark.parametrize("journey", ["missing", "readable", "blank"])
def test_mandatory_pdf_and_missing_text_journeys_fail_when_import_is_empty(tmp_path, monkeypatch, journey):
    recorder = usage._Recorder(tmp_path)
    projects = tmp_path / "projects"
    projects.mkdir()

    def empty_import(project, *args, expected=0):
        data = {"imported": []} if args[0] == "import" else {}
        record = {
            "argv": [*recorder.cli, "--project", str(project), *args],
            "expected_returncode": expected, "returncode": expected, "duration_seconds": 0,
            "stdout": json.dumps(data), "stderr": "", "parsed_json": data, "error": "",
            "assertions": [{"name": "退出码符合预期", "passed": True,
                            "expected": expected, "actual": expected}],
        }
        recorder._current_scenario["commands"].append(record)
        return record

    monkeypatch.setattr(recorder, "project_command", empty_import)
    functions = {
        "missing": usage._scenario_missing_document,
        "readable": usage._scenario_readable_pdf,
        "blank": usage._scenario_blank_pdf,
    }
    recorder.scenario(journey, journey, lambda scenario: functions[journey](recorder, scenario, projects))

    result = recorder.scenarios[0]
    assert result["passed"] is False
    assert any(not assertion["passed"] and "导入" in assertion["name"] for assertion in result["assertions"])


def test_stale_export_journey_fails_when_status_has_no_works(tmp_path, monkeypatch):
    recorder = usage._Recorder(tmp_path)
    projects = tmp_path / "projects"
    projects.mkdir()

    def empty_status(project, *args, expected=0):
        record = {"argv": [*recorder.cli, *args], "expected_returncode": expected, "returncode": 0,
                  "duration_seconds": 0, "stdout": '{"works":[]}', "stderr": "",
                  "parsed_json": {"works": []}, "error": "", "assertions": [
                      {"name": "退出码符合预期", "passed": True, "expected": 0, "actual": 0}]}
        recorder._current_scenario["commands"].append(record)
        return record

    monkeypatch.setattr(recorder, "project_command", empty_status)
    recorder.scenario("stale", "stale", lambda scenario: usage._scenario_stale_export(recorder, scenario, projects))
    assert recorder.scenarios[0]["passed"] is False
    assert any(not assertion["passed"] and "状态" in assertion["name"]
               for assertion in recorder.scenarios[0]["assertions"])


def test_innovation_journey_fails_before_false_pass_when_candidate_ids_are_missing(tmp_path, monkeypatch):
    recorder = usage._Recorder(tmp_path)
    projects = tmp_path / "projects"
    projects.mkdir()
    create_project(projects / "创新 中文 空格 #100%", ProjectConfig(title="fixture", backend="demo"))
    calls = 0

    def missing_candidate(project, *args, expected=0):
        nonlocal calls
        calls += 1
        data = ({"works": [{"id": "work_A"}, {"id": "work_B"}],
                 "budget": {"consumed_or_reserved_calls": 4}}
                if args[0] == "status" else
                {"status": "budget_paused", "reason": "model_budget_exhausted", "resume_argv": ["academic-agent"]}
                if calls == 3 else
                {"status": "complete", "candidate_ids": []}
                if args[0] == "innovate" else {"synthetic_demo": True})
        record = {
            "argv": [*recorder.cli, "--project", str(project), *args],
            "expected_returncode": expected, "returncode": expected, "duration_seconds": 0,
            "stdout": json.dumps(data), "stderr": "", "parsed_json": data, "error": "",
            "assertions": [{"name": "退出码符合预期", "passed": True,
                            "expected": expected, "actual": expected}],
        }
        recorder._current_scenario["commands"].append(record)
        return record

    monkeypatch.setattr(recorder, "project_command", missing_candidate)
    recorder.scenario("innovation", "innovation",
                      lambda scenario: usage._scenario_innovation(recorder, scenario, projects))
    result = recorder.scenarios[0]
    assert result["passed"] is False
    assert any(not assertion["passed"] and "一个可核验候选" in assertion["name"]
               for assertion in result["assertions"]
               + [item for command in result["commands"] for item in command["assertions"]])


def test_recorder_preserves_preceding_and_failed_commands_on_spawn_error(tmp_path, monkeypatch):
    recorder = usage._Recorder(tmp_path)
    calls = 0

    def launch(argv, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(argv, 0, stdout='{"ok":true}', stderr="")
        raise PermissionError("synthetic launch denied")

    monkeypatch.setattr(usage.subprocess, "run", launch)

    def journey(scenario):
        recorder.command(["academic-agent", "init"])
        recorder.command(["academic-agent", "import", "input.ris"])

    recorder.scenario("spawn", "spawn", journey)
    commands = recorder.scenarios[0]["commands"]
    assert len(commands) == 2
    assert commands[0]["returncode"] == 0 and commands[0]["parsed_json"] == {"ok": True}
    assert commands[1]["argv"][-2:] == ["import", "input.ris"]
    assert commands[1]["returncode"] is None
    assert "PermissionError" in commands[1]["error"] and "synthetic launch denied" in commands[1]["error"]
    assert recorder.scenarios[0]["passed"] is False


def test_recorder_parses_json_before_bounding_recorded_subprocess_output(tmp_path):
    recorder = usage._Recorder(tmp_path)
    script = (
        "import json; print(json.dumps({"
        "'checkable_before_tail': {'actions': [{'argv': ['academic-agent', 'guide']}]}, "
        "'padding': 'x' * 45000}))"
    )

    record = recorder.command([sys.executable, "-c", script])

    assert record["returncode"] == 0
    assert record["parsed_json"]["checkable_before_tail"] == {
        "actions": [{"argv": ["academic-agent", "guide"]}],
    }
    assert len(record["stdout"]) == 40_000
    assert record["stdout_truncated"] is True
    assert record["stderr_truncated"] is False


def test_cli_trends_and_verify_require_explicit_directory(tmp_path):
    project = tmp_path / "project"
    seed_demo(project)
    prefix = [sys.executable, "-m", "academic_agent.cli"]
    built = subprocess.run(prefix + ["--project", str(project), "trends"], capture_output=True,
                           text=True, encoding="utf-8", timeout=30, check=False)
    assert built.returncode == 0
    directory = Path(json.loads(built.stdout)["trends"])
    verified = subprocess.run(prefix + ["verify-trends", str(directory)], capture_output=True,
                              text=True, encoding="utf-8", timeout=30, check=False)
    assert verified.returncode == 0
    assert json.loads(verified.stdout)["valid"] is True
    missing = subprocess.run(prefix + ["verify-trends"], capture_output=True, text=True,
                             encoding="utf-8", timeout=30, check=False)
    assert missing.returncode != 0


def test_self_test_subprocess_journeys_write_verifiable_reports_and_refuse_repeat(tmp_path):
    output = tmp_path / "验收 输出"
    result = run_self_test(output)
    assert result["passed"] is True
    assert Path(result["json_report"]) == output / "usage-report.json"
    assert Path(result["markdown_report"]) == output / "usage-report.md"
    report = json.loads((output / "usage-report.json").read_text(encoding="utf-8"))
    expected = {
        "demo_export", "unicode_ris", "missing_document", "budget_resume",
        "human_edit_versions", "stale_export", "trend_recompute", "invalid_annotations",
        "empty_project_guide", "readable_pdf", "blank_pdf", "innovation_journey",
        "writing_full_journey", "writing_revision_recovery", "writing_stale_rebind_cache",
        "writing_external_pack", "writing_readonly_tamper",
    }
    assert expected == {scenario["name"] for scenario in report["scenarios"]}
    assert all(scenario["passed"] for scenario in report["scenarios"])
    assert all(scenario["commands"] for scenario in report["scenarios"])
    assert all(command["assertions"] for scenario in report["scenarios"] for command in scenario["commands"])
    assert all("argv" in command and "returncode" in command and "duration_seconds" in command
               for scenario in report["scenarios"] for command in scenario["commands"])
    assert report["observations"]["model_or_network_calls"] == 0
    assert "未执行人工学术准确率评测" in (output / "usage-report.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        run_self_test(output)


def test_self_test_cli_rejects_existing_output_without_creating_default_project(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    prefix = [sys.executable, "-m", "academic_agent.cli", "self-test", "--output", str(output)]
    result = subprocess.run(prefix, cwd=tmp_path, capture_output=True, text=True, encoding="utf-8",
                            timeout=30, check=False)
    assert result.returncode != 0
    assert "must not already exist" in json.loads(result.stdout)["error"]
    assert not (tmp_path / "projects" / "default").exists()


def test_self_test_always_writes_partial_report_after_subprocess_failures(tmp_path, monkeypatch):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(usage.subprocess, "run", timeout)
    output = tmp_path / "partial"
    result = run_self_test(output)
    report = json.loads((output / "usage-report.json").read_text(encoding="utf-8"))
    assert result["passed"] is False and report["passed"] is False
    assert report["scenarios"]
    assert any(not assertion["passed"] for scenario in report["scenarios"]
               for command in scenario["commands"] for assertion in command["assertions"])
    assert (output / "usage-report.md").is_file()


def test_cli_version_comes_from_package_metadata():
    result = subprocess.run([sys.executable, "-m", "academic_agent.cli", "--version"],
                            capture_output=True, text=True, encoding="utf-8", timeout=30, check=False)
    assert result.returncode == 0
    assert result.stdout.strip() == f"academic-agent {__version__}"
    launcher = Path(sys.executable).with_name("academic-agent.exe" if os.name == "nt" else "academic-agent")
    installed = subprocess.run([str(launcher), "--version"], capture_output=True, text=True,
                               encoding="utf-8", timeout=30, check=False)
    assert installed.returncode == 0
    assert installed.stdout.strip() == f"academic-agent {__version__}"
