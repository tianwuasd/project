import json
import subprocess
import sys

from academic_agent.cli import main
from academic_agent.exports import _csv, escape_md, safe_url


def test_cli_installed_module_help():
    result = subprocess.run([sys.executable, "-m", "academic_agent", "--help"], capture_output=True, timeout=30)
    assert result.returncode == 0 and b"resume" in result.stdout and b"demo" in result.stdout


def test_cli_init_status_and_json_errors(tmp_path, capsys):
    prefix = ["--project", str(tmp_path), "--offline"]
    assert main(prefix + ["init", "--title", "研究项目", "--backend", "demo"]) == 0
    capsys.readouterr()
    assert main(prefix + ["status"]) == 0
    assert json.loads(capsys.readouterr().out)["works"] == []
    assert main(prefix + ["screen", "--decision", "include", "--reason", "范围"]) == 1
    assert "Provide work IDs" in json.loads(capsys.readouterr().out)["error"]


def test_cli_demo_refuses_to_overwrite_existing_project(tmp_path, capsys):
    args = ["--project", str(tmp_path), "demo"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["synthetic_demo"]
    assert main(args) == 1
    assert "already exists" in json.loads(capsys.readouterr().out)["error"]


def test_export_escaping_blocks_active_content():
    assert _csv([["=HYPERLINK(1)", "+SUM(1)", "plain"]]).startswith("'=HYPERLINK(1),'+SUM(1),plain")
    escaped = escape_md("<script>evil</script> [click](javascript:bad) | title")
    assert "<script>" not in escaped and "\\[click\\]" in escaped and "\\|" in escaped
    assert safe_url("javascript:alert(1)") == ""
    assert safe_url("https://user:secret@example.com") == ""


def test_ask_review_failure_is_nonzero_and_question_scope_is_current(tmp_path, capsys):
    from academic_agent.config import atomic_json, load_config
    from academic_agent.demo import seed_demo
    from academic_agent.store import Store
    seed_demo(tmp_path)
    work_id = Store(tmp_path).works()[0].id
    config = load_config(tmp_path)
    config.max_model_calls = 1
    atomic_json(tmp_path / "project.json", config.model_dump())
    args = ["--project", str(tmp_path), "--offline", "ask", "result", "--work-id", work_id]
    assert main(args) == 2
    data = json.loads(capsys.readouterr().out)
    assert data["review"]["results"][0]["status"] == "failed"
    assert all(not item["usable"] for item in data["answers"])
