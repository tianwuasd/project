"""Installed-package offline acceptance journeys with inspectable evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from .config import atomic_json, load_config
from .demo import FIXTURES
from .filesystem import io_path

COMMAND_TIMEOUT = 60


def _safe_environment() -> dict[str, str]:
    names = {
        "PATH", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
        "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    }
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update({"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    return environment


def _json_file(path: Path) -> Any:
    return json.loads(io_path(path).read_text(encoding="utf-8"))


def _ris(records: list[tuple[str, int | None, str]]) -> str:
    lines = []
    for title, year, abstract in records:
        lines.extend(["TY  - JOUR", f"TI  - {title}"])
        if year is not None:
            lines.append(f"PY  - {year}")
        if abstract:
            parts = abstract.splitlines()
            lines.append(f"AB  - {parts[0]}")
            lines.extend(f"      {part}" for part in parts[1:])
        lines.extend(["ER  -", ""])
    return "\n".join(lines)


def _write_pdf(path: Path, text: str | None) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text is not None:
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Courier"),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
        })
        commands = []
        for index, line in enumerate(text.splitlines()):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"BT /F1 9 Tf 36 {756 - index * 12} Td ({escaped}) Tj ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


class _Recorder:
    def __init__(self, output: Path):
        self.output = output
        self.cli = [sys.executable, "-m", "academic_agent.cli"]
        self.environment = _safe_environment()
        self.scenarios: list[dict[str, Any]] = []
        self._current_scenario: dict[str, Any] | None = None

    def project_command(self, project: Path, *args: str, expected: int = 0) -> dict[str, Any]:
        return self.command([*self.cli, "--project", str(project), "--offline", *args], expected=expected)

    def command(self, argv: list[str], *, expected: int = 0) -> dict[str, Any]:
        started = time.monotonic()
        record = {
            "argv": argv,
            "expected_returncode": expected,
            "returncode": None,
            "duration_seconds": 0.0,
            "stdout": "",
            "stderr": "",
            "parsed_json": None,
            "error": "",
            "assertions": [],
        }
        if self._current_scenario is not None:
            self._current_scenario["commands"].append(record)
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=COMMAND_TIMEOUT, env=self.environment, check=False,
            )
            returncode: int | None = completed.returncode
            raw_stdout = completed.stdout
            try:
                data = json.loads(raw_stdout)
            except json.JSONDecodeError:
                data = None
            stdout, stderr = raw_stdout[-40_000:], completed.stderr[-20_000:]
            stdout_truncated = len(raw_stdout) > 40_000
            stderr_truncated = len(completed.stderr) > 20_000
            error = ""
        except subprocess.TimeoutExpired as exc:
            returncode, data = None, None
            stdout = (exc.stdout or "")[-40_000:] if isinstance(exc.stdout, str) else ""
            stderr = (exc.stderr or "")[-20_000:] if isinstance(exc.stderr, str) else ""
            stdout_truncated = isinstance(exc.stdout, str) and len(exc.stdout) > 40_000
            stderr_truncated = isinstance(exc.stderr, str) and len(exc.stderr) > 20_000
            error = f"命令超过 {COMMAND_TIMEOUT} 秒限制"
        except OSError as exc:
            returncode, data, stdout, stderr = None, None, "", ""
            stdout_truncated = stderr_truncated = False
            error = f"{type(exc).__name__}: {exc}"
        record.update({
            "returncode": returncode, "duration_seconds": round(time.monotonic() - started, 6),
            "stdout": stdout, "stderr": stderr, "parsed_json": data, "error": error,
            "stdout_truncated": stdout_truncated, "stderr_truncated": stderr_truncated,
        })
        self.check(record, "退出码符合预期", returncode == expected, expected, returncode)
        return record

    @staticmethod
    def check(target: dict[str, Any], name: str, passed: bool, expected: Any, actual: Any) -> None:
        target.setdefault("assertions", []).append({
            "name": name, "passed": bool(passed), "expected": expected, "actual": actual,
        })

    def scenario(self, name: str, title: str, function: Callable[[dict[str, Any]], None]) -> None:
        scenario = {"name": name, "title": title, "passed": False, "commands": [], "assertions": []}
        if self._current_scenario is not None:
            raise RuntimeError("Nested self-test scenarios are unsupported")
        self._current_scenario = scenario
        try:
            function(scenario)
        except Exception as exc:  # A partial report is more useful than aborting the remaining journeys.
            self.check(scenario, "场景执行未发生未处理异常", False, "无异常", f"{type(exc).__name__}: {exc}")
        finally:
            self._current_scenario = None
        assertions = [*scenario["assertions"],
                      *(item for command in scenario["commands"] for item in command["assertions"])]
        if not any(item["name"] != "退出码符合预期" for item in assertions):
            self.check(scenario, "场景完成至少一项结果或产物检查", False, "至少一项", "零项")
            assertions = [*scenario["assertions"],
                          *(item for command in scenario["commands"] for item in command["assertions"])]
        scenario["passed"] = bool(assertions) and all(item["passed"] for item in assertions)
        self.scenarios.append(scenario)


def _imported_ids(command: dict[str, Any]) -> list[str]:
    data = command.get("parsed_json") or {}
    if not isinstance(data, dict) or not isinstance(data.get("imported", []), list):
        return []
    return [item["id"] for item in data.get("imported", []) if isinstance(item, dict) and item.get("id")]


def _export_directory(project: Path) -> Path:
    pointer = _json_file(project / "exports" / "latest.json")
    version = pointer["version"]
    if not isinstance(version, str) or Path(version).name != version:
        raise ValueError("导出 latest.json 的版本路径不安全")
    directory = (project / "exports" / version).resolve()
    if directory.parent != (project / "exports").resolve():
        raise ValueError("导出版本路径越界")
    return directory


def _innovation_directory(project: Path, output: Any) -> Path:
    if not isinstance(output, str) or not output:
        raise ValueError("创新产物没有输出路径")
    supplied = Path(output)
    directory = supplied.resolve() if supplied.is_absolute() else (project / supplied).resolve()
    if not directory.is_relative_to(project.resolve()) or not directory.is_dir():
        raise ValueError("创新产物路径缺失或越界")
    return directory


def _scenario_demo(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "demo"
    command = recorder.project_command(project, "demo")
    data = command["parsed_json"] or {}
    recorder.check(command, "演示明确标记原创合成数据", data.get("synthetic_demo") is True, True,
                   data.get("synthetic_demo"))
    directory = _export_directory(project)
    quality = _json_file(directory / "quality.json")
    ledger = _json_file(directory / "evidence-ledger.json")
    recorder.check(scenario, "演示导出包含两篇阅读卡和十二条支持陈述",
                   quality.get("reading_cards") == 2 and quality.get("supported_claims") == 12,
                   {"reading_cards": 2, "supported_claims": 12},
                   {key: quality.get(key) for key in ("reading_cards", "supported_claims")})
    recorder.check(scenario, "证据台账实际包含两篇原文的精确证据块", len(ledger.get("evidence", [])) == 2,
                   2, len(ledger.get("evidence", [])))


def _scenario_unicode_ris(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "中文 项目"
    inputs = recorder.output / "输入 文件"
    inputs.mkdir(exist_ok=True)
    ris = inputs / "中文 书目.ris"
    ris.write_text(_ris([("中文合成调度研究", 2024, "")]), encoding="utf-8")
    document = inputs / "原创 合成全文.txt"
    document.write_text(FIXTURES[0][1], encoding="utf-8")
    recorder.project_command(project, "init", "--title", "中文 空格项目", "--backend", "demo")
    imported = recorder.project_command(project, "import", str(ris))
    ids = _imported_ids(imported)
    recorder.check(imported, "RIS 中文标题按 UTF-8 导入",
                   bool(ids) and imported["parsed_json"]["imported"][0]["title"] == "中文合成调度研究",
                   "中文合成调度研究", (imported.get("parsed_json") or {}).get("imported"))
    if not ids:
        return
    work_id = ids[0]
    recorder.project_command(project, "screen", work_id, "--decision", "include", "--reason", "逐篇确认")
    attach = recorder.project_command(project, "attach", work_id, str(document))
    read = recorder.project_command(project, "read", "--work-id", work_id)
    recorder.project_command(project, "review", "--work-id", work_id)
    export = recorder.project_command(project, "export")
    recorder.check(attach, "关联全文后层级为可读全文", (attach["parsed_json"] or {}).get("level") == "full_text",
                   "full_text", (attach["parsed_json"] or {}).get("level"))
    results = (read["parsed_json"] or {}).get("results", [])
    recorder.check(read, "实际阅读生成六类有证据陈述",
                   len(results) == 1 and results[0].get("status") == "ok" and results[0].get("claims") == 6,
                   {"status": "ok", "claims": 6}, results)
    quality = _json_file(_export_directory(project) / "quality.json")
    recorder.check(export, "导出包含六条已复核支持陈述", quality.get("supported_claims") == 6,
                   6, quality.get("supported_claims"))


def _scenario_missing_document(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "missing-document"
    ris = recorder.output / "missing.ris"
    ris.write_text(_ris([("Metadata only", 2023, "")]), encoding="utf-8")
    recorder.project_command(project, "init", "--title", "缺全文", "--backend", "demo")
    imported = recorder.project_command(project, "import", str(ris))
    ids = _imported_ids(imported)
    if not ids:
        recorder.check(scenario, "缺全文场景必须导入一篇文献", False, 1, len(ids))
        return
    recorder.project_command(project, "screen", ids[0], "--decision", "include", "--reason", "范围匹配")
    read = recorder.project_command(project, "read", "--work-id", ids[0], expected=2)
    status = recorder.project_command(project, "status")
    results = (read["parsed_json"] or {}).get("results", [])
    recorder.check(read, "缺少可用文本时明确失败而非生成陈述",
                   len(results) == 1 and results[0].get("status") == "failed"
                   and "no usable reading text" in results[0].get("error", "").lower(),
                   "failed: no usable reading text", results)
    budget = (status["parsed_json"] or {}).get("budget", {})
    recorder.check(status, "缺全文失败不消耗模型调用", budget.get("consumed_or_reserved_calls") == 0,
                   0, budget.get("consumed_or_reserved_calls"))


def _scenario_budget_resume(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "budget-resume"
    ris = recorder.output / "budget.ris"
    ris.write_text(_ris([(title, 2025, text) for title, text in FIXTURES]), encoding="utf-8")
    recorder.project_command(project, "init", "--title", "预算恢复", "--backend", "demo",
                             "--max-model-calls", "1")
    imported = recorder.project_command(project, "import", str(ris))
    ids = _imported_ids(imported)
    if len(ids) != 2:
        recorder.check(scenario, "预算场景导入两篇文献", False, 2, len(ids))
        return
    recorder.project_command(project, "screen", *ids, "--decision", "include", "--reason", "逐篇材料范围一致")
    for index, (work_id, fixture) in enumerate(zip(ids, FIXTURES), 1):
        path = recorder.output / f"budget-original-{index}.txt"
        path.write_text(fixture[1], encoding="utf-8")
        recorder.project_command(project, "attach", work_id, str(path))
    paused = recorder.project_command(project, "run", expected=1)
    recorder.check(paused, "预算中断明确保留为可恢复暂停",
                   "budget" in str((paused["parsed_json"] or {}).get("error", "")).lower(),
                   "错误包含 budget", (paused["parsed_json"] or {}).get("error"))
    config = load_config(project)
    config.max_model_calls = 6
    atomic_json(project / "project.json", config.model_dump())
    resumed = recorder.project_command(project, "resume")
    status = recorder.project_command(project, "status")
    recorder.check(resumed, "提高本地预算后恢复完成", (resumed["parsed_json"] or {}).get("complete") is True,
                   True, (resumed["parsed_json"] or {}).get("complete"))
    status_data = status["parsed_json"] or {}
    jobs = status_data.get("jobs", [])
    recorder.check(status, "恢复复用已完成调用且最终仅有四个成功任务",
                   status_data.get("budget", {}).get("consumed_or_reserved_calls") == 4
                   and len(jobs) == 4 and all(job.get("status") == "succeeded" for job in jobs),
                   {"calls": 4, "successful_jobs": 4},
                   {"calls": status_data.get("budget", {}).get("consumed_or_reserved_calls"), "jobs": jobs})


def _scenario_human_edits(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "demo"
    old = _export_directory(project)
    old_report = old / "review.md"
    old_report.write_text("研究者人工改稿，必须保留。", encoding="utf-8")
    exported = recorder.project_command(project, "export")
    new = Path((exported["parsed_json"] or {}).get("export", ""))
    recorder.check(exported, "再次导出创建不同版本并保留人工改稿",
                   new != old and new.is_dir() and old_report.read_text(encoding="utf-8") == "研究者人工改稿，必须保留。",
                   "新版本目录且旧改稿原样保留", {"old": str(old), "new": str(new),
                    "old_text": old_report.read_text(encoding="utf-8")})


def _scenario_stale_export(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "demo"
    status_before = recorder.project_command(project, "status")
    works = (status_before["parsed_json"] or {}).get("works", [])
    if not works:
        recorder.check(scenario, "过期检测场景的状态必须包含文献", False, "至少一篇", len(works))
        return
    recorder.project_command(project, "screen", works[0]["id"], "--decision", "include",
                             "--reason", "人工重新确认后改变研究修订")
    status_after = recorder.project_command(project, "status")
    recorder.check(status_before, "变更前最新导出与研究状态一致",
                   (status_before["parsed_json"] or {}).get("latest_export_current") is True,
                   True, (status_before["parsed_json"] or {}).get("latest_export_current"))
    recorder.check(status_after, "文献筛选修订使旧导出明确过期",
                   (status_after["parsed_json"] or {}).get("latest_export_current") is False,
                   False, (status_after["parsed_json"] or {}).get("latest_export_current"))


def _scenario_trends(recorder: _Recorder, scenario: dict[str, Any], projects: Path,
                     shared: dict[str, Any]) -> None:
    project = projects / "trend-counts"
    ris = recorder.output / "trend-counts.ris"
    records = [("Year 2022 A", 2022, ""), ("Year 2022 B", 2022, ""),
               ("Year 2023", 2023, ""), ("Missing year", None, ""),
               ("Future year", datetime.now().year + 1, "")]
    ris.write_text(_ris(records), encoding="utf-8")
    recorder.project_command(project, "init", "--title", "精确趋势计数", "--backend", "demo")
    imported = recorder.project_command(project, "import", str(ris))
    ids = _imported_ids(imported)
    if len(ids) != 5:
        recorder.check(scenario, "趋势场景导入五篇文献", False, 5, len(ids))
        return
    recorder.project_command(project, "screen", *ids, "--decision", "include", "--reason", "计数夹具")
    annotations = recorder.output / "annotations.json"
    atomic_json(annotations, {
        "version": "1", "annotator": "离线自检",
        "works": {
            ids[0]: {"labels": ["方法甲", "方法甲"], "rationale": "人工固定标签"},
            ids[1]: {"labels": ["方法甲", "方法乙"], "rationale": "人工固定标签"},
            ids[2]: {"labels": ["方法乙"], "rationale": "人工固定标签"},
        },
    })
    built = recorder.project_command(project, "trends", "--date-basis", "record_year",
                                     "--annotations", str(annotations))
    directory = Path((built["parsed_json"] or {}).get("trends", ""))
    computation = _json_file(directory / "computation.json")
    annual = {row["year"]: row["count"] for row in computation["annual"]}
    labels = {row["label"]: row["count"] for row in computation["labels"]}
    expected_counts = {"eligible": 5, "valid_year": 3, "missing_year": 1, "future_year": 1}
    recorder.check(built, "年度、缺失、未来年份和去重标签数量精确",
                   all(computation["counts"].get(key) == value for key, value in expected_counts.items())
                   and annual == {2022: 2, 2023: 1} and labels == {"方法甲": 2, "方法乙": 2},
                   {"counts": expected_counts, "annual": {2022: 2, 2023: 1}, "labels": {"方法甲": 2, "方法乙": 2}},
                   {"counts": {key: computation["counts"].get(key) for key in expected_counts},
                    "annual": annual, "labels": labels})
    verified = recorder.command([*recorder.cli, "verify-trends", str(directory)])
    recorder.check(verified, "冻结语料可在无数据库参数下完整重算并验证六个文件",
                   (verified["parsed_json"] or {}).get("valid") is True
                   and (verified["parsed_json"] or {}).get("files_verified") == 6,
                   {"valid": True, "files_verified": 6}, verified["parsed_json"])
    shared.update({"trend_project": project})


def _scenario_invalid_annotations(recorder: _Recorder, scenario: dict[str, Any], shared: dict[str, Any]) -> None:
    project: Path = shared["trend_project"]
    invalid = recorder.output / "invalid-annotations.json"
    atomic_json(invalid, {
        "version": "bad", "annotator": "离线自检",
        "works": {"work_not_in_project": {"labels": ["未知"], "rationale": "错误 ID"}},
    })
    command = recorder.project_command(project, "trends", "--annotations", str(invalid), expected=1)
    error = str((command["parsed_json"] or {}).get("error", ""))
    recorder.check(command, "不存在的批注文献 ID 被严格拒绝且没有发布结果",
                   "Unknown annotation" in error and "work_not_in_project" in error,
                   "错误包含 Unknown annotation 与错误 ID", error)


def _scenario_empty_guide(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "empty-guide"
    recorder.project_command(project, "init", "--title", "空语料指南", "--backend", "demo")
    database_before = (project / "research.db").exists()
    guide = recorder.project_command(project, "guide")
    data = guide["parsed_json"] or {}
    recorder.check(guide, "空语料指南只给检索或导入 argv 且不创建数据库",
                   not database_before and not (project / "research.db").exists()
                   and "尚无文献" in data.get("summary", "")
                   and all(isinstance(action.get("argv"), list) for action in data.get("actions", [])),
                   "无数据库；中文空语料说明；动作均为 argv 数组",
                   {"database_before": database_before, "database_after": (project / "research.db").exists(),
                    "guide": data})


def _scenario_readable_pdf(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "readable-pdf"
    ris = recorder.output / "readable-pdf.ris"
    pdf = recorder.output / "readable-original.pdf"
    ris.write_text(_ris([("Readable PDF fixture", 2024, "")]), encoding="utf-8")
    _write_pdf(pdf, FIXTURES[0][1])
    recorder.project_command(project, "init", "--title", "可读 PDF", "--backend", "demo")
    imported = recorder.project_command(project, "import", str(ris))
    ids = _imported_ids(imported)
    if not ids:
        recorder.check(scenario, "可读 PDF 场景必须导入一篇文献", False, 1, len(ids))
        return
    recorder.project_command(project, "screen", ids[0], "--decision", "include", "--reason", "PDF 文字层")
    attach = recorder.project_command(project, "attach", ids[0], str(pdf))
    recorder.project_command(project, "read", "--work-id", ids[0])
    recorder.project_command(project, "review", "--work-id", ids[0])
    export = recorder.project_command(project, "export")
    ledger = _json_file(_export_directory(project) / "evidence-ledger.json")
    evidence = ledger.get("evidence", [])
    recorder.check(attach, "原创 PDF 文字层识别为可读全文", (attach["parsed_json"] or {}).get("level") == "full_text",
                   "full_text", (attach["parsed_json"] or {}).get("level"))
    recorder.check(export, "PDF 证据保留物理页码一和精确字符范围",
                   bool(evidence) and all(item.get("page") == 1 and item.get("end", 0) > item.get("start", 0)
                                          and item.get("text_hash") for item in evidence),
                   "所有证据 page=1 且范围、哈希非空", evidence)


def _scenario_blank_pdf(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "blank-pdf"
    ris = recorder.output / "blank-pdf.ris"
    pdf = recorder.output / "blank-original.pdf"
    ris.write_text(_ris([("Blank PDF fixture", 2024, "")]), encoding="utf-8")
    _write_pdf(pdf, None)
    recorder.project_command(project, "init", "--title", "空白 PDF", "--backend", "demo")
    imported = recorder.project_command(project, "import", str(ris))
    ids = _imported_ids(imported)
    if not ids:
        recorder.check(scenario, "空白 PDF 场景必须导入一篇文献", False, 1, len(ids))
        return
    recorder.project_command(project, "screen", ids[0], "--decision", "include", "--reason", "OCR 检查")
    attach = recorder.project_command(project, "attach", ids[0], str(pdf))
    recorder.project_command(project, "read", "--work-id", ids[0], expected=2)
    status = recorder.project_command(project, "status")
    recorder.check(attach, "无文字 PDF 明确标记 needs_ocr 并给出诊断",
                   (attach["parsed_json"] or {}).get("level") == "needs_ocr"
                   and bool((attach["parsed_json"] or {}).get("diagnostics")),
                   "needs_ocr 且诊断非空", attach["parsed_json"])
    budget = (status["parsed_json"] or {}).get("budget", {})
    recorder.check(status, "needs_ocr 阅读失败不消耗模型调用",
                   budget.get("consumed_or_reserved_calls") == 0,
                   0, budget.get("consumed_or_reserved_calls"))


def _scenario_innovation(recorder: _Recorder, scenario: dict[str, Any], projects: Path) -> None:
    project = projects / "创新 中文 空格 #100%"
    recorder.project_command(project, "demo")
    status_before = recorder.project_command(project, "status")
    works = (status_before["parsed_json"] or {}).get("works", [])
    work_ids = [work.get("id") for work in works if isinstance(work, dict) and work.get("id")]
    if len(work_ids) != 2:
        recorder.check(scenario, "创新旅程必须从两篇原创合成材料开始", False, 2, len(work_ids))
        return
    baseline = (status_before["parsed_json"] or {}).get("budget", {}).get("consumed_or_reserved_calls")
    config = load_config(project)
    config.max_model_calls = baseline + 1
    atomic_json(project / "project.json", config.model_dump())
    argv = tuple(part for work_id in work_ids for part in ("--work-id", work_id))
    paused = recorder.project_command(project, "innovate", *argv, expected=2)
    paused_data = paused["parsed_json"] or {}
    recorder.check(paused, "候选生成在复核前遇到预算上限时明确暂停并保留恢复命令",
                   paused_data.get("status") == "budget_paused"
                   and paused_data.get("reason") == "model_budget_exhausted"
                   and isinstance(paused_data.get("resume_argv"), list),
                   {"status": "budget_paused", "reason": "model_budget_exhausted", "resume_argv": "argv"},
                   paused_data)

    config.max_model_calls = baseline + 4
    atomic_json(project / "project.json", config.model_dump())
    generated = recorder.project_command(project, "innovate", *argv)
    generated_data = generated["parsed_json"] or {}
    candidate_ids = generated_data.get("candidate_ids", [])
    recorder.check(generated, "恢复后固定原创协议精确生成一个可核验候选",
                   generated_data.get("status") == "complete" and len(candidate_ids) == 1,
                   {"status": "complete", "candidate_count": 1},
                   {"status": generated_data.get("status"), "candidate_ids": candidate_ids})
    if len(candidate_ids) != 1:
        return
    candidate_id = candidate_ids[0]
    generation_directory = _innovation_directory(project, generated_data.get("output"))
    required_generation = {
        "candidates.json", "grounding-review.json", "evidence.json", "references.bib", "manifest.json", "report.md",
    }
    candidates = _json_file(generation_directory / "candidates.json")
    recorder.check(generated, "候选版本引用的生成、复核、清单与报告文件实际存在",
                   all((generation_directory / name).is_file() for name in required_generation),
                   sorted(required_generation),
                   sorted(path.name for path in generation_directory.iterdir() if path.is_file()))

    assessed = recorder.project_command(project, "assess-innovation", candidate_id)
    assessed_data = assessed["parsed_json"] or {}
    assessment_directory = _innovation_directory(project, assessed_data.get("output"))
    required_assessment = {
        "candidate.json", "assessment.json", "assessment-review.json", "context.json",
        "report.md", "references.bib", "evidence.json", "manifest.json",
    }
    assessment = _json_file(assessment_directory / "assessment.json")
    review = _json_file(assessment_directory / "assessment-review.json")
    dimensions = {"novelty_clues", "research_value", "feasibility", "evidence_strength"}
    recorder.check(assessed, "四维评估版本引用全部八个必需文件",
                   assessed_data.get("status") == "complete"
                   and all((assessment_directory / name).is_file() for name in required_assessment),
                   {"status": "complete", "files": sorted(required_assessment)},
                   {"status": assessed_data.get("status"),
                    "files": sorted(path.name for path in assessment_directory.iterdir() if path.is_file())})
    recorder.check(assessed, "合成标签、证据不足与逐单元复核状态均明确保存",
                   assessment.get("statement_type") == "scope_limited_assessment"
                   and assessment.get("evidence_sufficiency") == "insufficient"
                   and dimensions.issubset(assessment)
                   and review.get("backend") == "demo"
                   and review.get("items")
                   and all(item.get("verdict") == "partial" for item in review["items"]),
                   {"statement_type": "scope_limited_assessment", "evidence_sufficiency": "insufficient",
                    "dimensions": sorted(dimensions), "backend": "demo", "verdicts": ["partial"]},
                   {"statement_type": assessment.get("statement_type"),
                    "evidence_sufficiency": assessment.get("evidence_sufficiency"),
                    "dimensions": sorted(dimensions.intersection(assessment)), "backend": review.get("backend"),
                    "verdicts": sorted({item.get("verdict") for item in review.get("items", [])})})
    serialized = json.dumps({"candidates": candidates, "assessment": assessment, "review": review}, ensure_ascii=False)
    recorder.check(assessed, "离线评估保留范围限制且不宣称全球新颖性",
                   candidates.get("candidates", [{}])[0].get("statement_type") == "system_hypothesis"
                   and "材料不足以确认新颖性" in serialized
                   and "全球首次" not in serialized and "全球首创" not in serialized,
                   "候选为系统假设，包含新颖性不足说明且无全球首次/首创", serialized[:600])

    innovation = recorder.project_command(project, "innovation-status")
    rows = (innovation["parsed_json"] or {}).get("candidates", [])
    row = next((item for item in rows if item.get("candidate_id") == candidate_id), {})
    recorder.check(innovation, "只读状态确认候选前提和独立评估均为当前版本",
                   row.get("premises_current") is True
                   and row.get("assessment", {}).get("status") == "current",
                   {"premises_current": True, "assessment": "current"}, row)

    repeated_generation = recorder.project_command(project, "innovate", *argv)
    repeated_assessment = recorder.project_command(project, "assess-innovation", candidate_id)
    status_after = recorder.project_command(project, "status")
    final_calls = (status_after["parsed_json"] or {}).get("budget", {}).get("consumed_or_reserved_calls")
    recorder.check(status_after, "重复完成的候选与评估命令复用缓存且没有新增模型尝试",
                   (repeated_generation["parsed_json"] or {}).get("candidate_ids") == [candidate_id]
                   and (repeated_assessment["parsed_json"] or {}).get("candidate_ids") == [candidate_id]
                   and final_calls == baseline + 4,
                   {"candidate_ids": [candidate_id], "calls": baseline + 4},
                   {"generation": (repeated_generation["parsed_json"] or {}).get("candidate_ids"),
                    "assessment": (repeated_assessment["parsed_json"] or {}).get("candidate_ids"),
                    "calls": final_calls})

    changed_source = recorder.output / "innovation-changed-original.txt"
    changed_source.write_text(FIXTURES[0][1].replace("40 simulated", "41 simulated"), encoding="utf-8")
    recorder.project_command(project, "attach", work_ids[0], str(changed_source))
    stale = recorder.project_command(project, "innovation-status")
    stale_row = next((item for item in (stale["parsed_json"] or {}).get("candidates", [])
                      if item.get("candidate_id") == candidate_id), {})
    recorder.check(stale, "研究输入变化后历史候选仍可追溯且前提与评估明确过期",
                   stale_row.get("historical_available") is True
                   and stale_row.get("premises_current") is False
                   and stale_row.get("assessment", {}).get("status") == "stale",
                   {"historical_available": True, "premises_current": False, "assessment": "stale"},
                   stale_row)


def _markdown(report: dict[str, Any]) -> str:
    def display(value: Any) -> str:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        text = " ".join(text.split())
        return text if len(text) <= 600 else text[:597] + "..."

    lines = ["# Academic Agent 离线使用自检\n\n",
             f"总体结果：{'通过' if report['passed'] else '失败'}。\n\n",
             "全部材料为原创合成内容；未访问论文网络或真实模型。未执行人工学术准确率评测。\n\n"]
    for scenario in report["scenarios"]:
        lines.append(f"## {'通过' if scenario['passed'] else '失败'} · {scenario['title']}\n\n")
        assertions = [*scenario["assertions"],
                      *(item for command in scenario["commands"] for item in command["assertions"])]
        for assertion in assertions:
            lines.append(f"- {'通过' if assertion['passed'] else '失败'}：{assertion['name']}"
                         f"（预期：{display(assertion['expected'])}；实际：{display(assertion['actual'])}）\n")
        lines.append("\n")
    return "".join(lines)


def run_self_test(output: Path) -> dict:
    """Run bounded real CLI subprocess journeys in a brand-new directory."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Self-test output directory must not already exist")
    output.mkdir(parents=True)
    projects = output / "projects"
    projects.mkdir()
    recorder = _Recorder(output)
    shared: dict[str, Any] = {}
    started = datetime.now(timezone.utc).isoformat()
    scenarios = [
        ("demo_export", "首次演示导出与证据数量", lambda item: _scenario_demo(recorder, item, projects)),
        ("unicode_ris", "中文空格路径的 RIS、筛选、全文与导出", lambda item: _scenario_unicode_ris(recorder, item, projects)),
        ("missing_document", "缺全文失败与零调用", lambda item: _scenario_missing_document(recorder, item, projects)),
        ("budget_resume", "预算中断、提高配置与无重复恢复", lambda item: _scenario_budget_resume(recorder, item, projects)),
        ("human_edit_versions", "人工改稿与版本保留", lambda item: _scenario_human_edits(recorder, item, projects)),
        ("stale_export", "研究修订使导出过期", lambda item: _scenario_stale_export(recorder, item, projects)),
        ("trend_recompute", "年度标签精确计数与离线重算", lambda item: _scenario_trends(recorder, item, projects, shared)),
        ("invalid_annotations", "错误批注文献 ID 被拒绝", lambda item: _scenario_invalid_annotations(recorder, item, shared)),
        ("empty_project_guide", "空语料只读指南", lambda item: _scenario_empty_guide(recorder, item, projects)),
        ("readable_pdf", "原创可读 PDF 物理页码证据", lambda item: _scenario_readable_pdf(recorder, item, projects)),
        ("blank_pdf", "原创无文字 PDF 的 OCR 提示", lambda item: _scenario_blank_pdf(recorder, item, projects)),
        ("innovation_journey", "原创候选、四维评估、缓存恢复与过期检测",
         lambda item: _scenario_innovation(recorder, item, projects)),
    ]
    report: dict[str, Any] = {}
    try:
        for name, title, function in scenarios:
            recorder.scenario(name, title, function)
        from .writing_usage import WRITING_SCENARIO_NAMES, run_writing_scenarios

        run_writing_scenarios(recorder, projects)
    finally:
        planned = [name for name, _, _ in scenarios]
        planned.extend(WRITING_SCENARIO_NAMES)
        passed = ([item["name"] for item in recorder.scenarios] == planned
                  and all(item["passed"] for item in recorder.scenarios))
        json_path = output / "usage-report.json"
        markdown_path = output / "usage-report.md"
        report = {
            "schema_version": 1,
            "passed": passed,
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "python_executable": sys.executable,
            "command_timeout_seconds": COMMAND_TIMEOUT,
            "scenarios": recorder.scenarios,
            "observations": {
                "model_or_network_calls": 0,
                "materials": "original_synthetic_only",
                "human_academic_accuracy_evaluation": "not_performed",
            },
            "json_report": str(json_path),
            "markdown_report": str(markdown_path),
        }
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {"passed": report["passed"], "scenarios": len(report["scenarios"]),
            "json_report": report["json_report"], "markdown_report": report["markdown_report"]}
