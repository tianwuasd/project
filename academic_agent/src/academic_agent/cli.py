"""Small, inspectable command-line interface; all responses are UTF-8 JSON."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import create_project, load_config
from .models import ProjectConfig


def _emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        _emit({"error": f"命令参数无效：{message}", "type": "ArgumentError"})
        raise SystemExit(2)


def _parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(description="Academic Agent — 从多渠道文献发现到带证据的综述工作稿")
    parser.add_argument("--version", action="version", version=f"academic-agent {__version__}")
    parser.add_argument("--project", type=Path, default=Path("projects/default"), help="项目数据目录")
    parser.add_argument("--offline", action="store_true", help="禁用论文来源网络请求；出版检查标为未知")
    commands = parser.add_subparsers(dest="command", required=True, parser_class=_JsonArgumentParser)
    init = commands.add_parser("init", help="创建独立研究项目")
    init.add_argument("--title", required=True)
    init.add_argument("--question", default="")
    init.add_argument("--query", action="append", default=[])
    init.add_argument("--backend", choices=["codex", "demo", "exchange"], default="codex")
    init.add_argument("--model", default=None)
    init.add_argument("--proxy", default="http://127.0.0.1:7897")
    init.add_argument("--max-model-calls", type=int, default=60)
    commands.add_parser("doctor", help="检查本地运行环境，不读取凭据内容")
    search = commands.add_parser("search", help="跨来源检索，保存查询与原始响应")
    search.add_argument("query")
    imported = commands.add_parser("import", help="导入 RIS / BibTeX 文献")
    imported.add_argument("file", type=Path)
    attached = commands.add_parser("attach", help="关联本地 PDF / Markdown / TXT")
    attached.add_argument("work_id")
    attached.add_argument("file", type=Path)
    fetch = commands.add_parser("fetch", help="下载已有开放获取链接指向的 PDF")
    fetch.add_argument("work_id")
    screen = commands.add_parser("screen", help="记录人工纳入/排除/待定决定")
    screen.add_argument("work_ids", nargs="*")
    screen.add_argument("--all", action="store_true", help="显式对所有候选应用同一决定")
    screen.add_argument("--decision", choices=["include", "exclude", "pending"], required=True)
    screen.add_argument("--reason", required=True)
    reading = commands.add_parser("read", help="阅读已纳入文献并提取有依据的陈述")
    reading.add_argument("--work-id")
    review = commands.add_parser("review", help="独立证据复核；或记录具名人工复核")
    review.add_argument("--work-id")
    review.add_argument("--claim-id")
    review.add_argument("--decision", choices=["supported", "partial", "contradicted", "insufficient"])
    review.add_argument("--reason")
    review.add_argument("--reviewer")
    ask = commands.add_parser("ask", help="根据当前文献回答问题并独立复核")
    ask.add_argument("question")
    ask.add_argument("--work-id")
    commands.add_parser("export", help="重新检查出版状态并创建新版本报告")
    commands.add_parser("status", help="查看文献、任务、预算及导出是否过期")
    for name in ("run", "resume"):
        run = commands.add_parser(name, help="执行检查点工作流" if name == "run" else "恢复已有检查点")
        run.add_argument("--run-id", default="main")
        run.add_argument("--stop-after", choices=["discover", "prepare", "read", "review", "export"])
    cancel = commands.add_parser("cancel", help="取消指定任务，阻止迟到结果入库")
    cancel.add_argument("job_id")
    commands.add_parser("demo", help="创建原创建模样例并离线运行，不调用真实模型")
    commands.add_parser("guide", help="只读检查项目状态并给出安全的下一步")
    trends = commands.add_parser("trends", help="生成可离线重算的冻结语料描述统计")
    trends.add_argument("--date-basis", choices=["record_year", "first_publication_year", "formal_publication_year"],
                        default="record_year")
    trends.add_argument("--annotations", type=Path)
    verify = commands.add_parser("verify-trends", help="离线校验并重算指定趋势版本")
    verify.add_argument("directory", type=Path)
    innovate = commands.add_parser("innovate", help="从当前已复核证据生成并复核创新候选")
    innovate.add_argument("--work-id", action="append", default=[])
    challenge = commands.add_parser("challenge", help="按候选检索反例材料；每轮最多两条查询")
    challenge.add_argument("candidate_id")
    challenge.add_argument("--round", type=int, choices=[1, 2], default=1, dest="round_number")
    assess = commands.add_parser("assess-innovation", help="按四个维度评估一个创新候选")
    assess.add_argument("candidate_id")
    commands.add_parser("innovation-status", help="只读检查创新候选、前提与评估是否为当前版本")
    domain_extract = commands.add_parser("domain-extract", help="按数据型领域包提取并独立复核字段")
    domain_extract.add_argument("--question", required=True)
    domain_extract.add_argument("--work-id", action="append", required=True, dest="work_ids")
    domain_extract.add_argument("--domain", required=True)
    writing_outline = commands.add_parser("writing-outline", help="从一个明确领域版本生成并复核大纲")
    writing_outline.add_argument("--question", required=True)
    writing_outline.add_argument("--work-id", action="append", required=True, dest="work_ids")
    writing_outline.add_argument("--domain-version", type=Path, required=True)
    writing_outline.add_argument("--trends", type=Path)
    writing_outline.add_argument("--candidate-version", type=Path)
    writing_outline.add_argument("--candidate-id")
    writing_outline.add_argument("--assessment-version", type=Path)
    writing_draft = commands.add_parser("writing-draft", help="从一个明确且已复核的大纲版本生成正文")
    writing_draft.add_argument("outline_version", type=Path)
    writing_export = commands.add_parser("writing-export-edit", help="导出一个明确写作版本的可编辑 JSON 副本")
    writing_export.add_argument("version", type=Path)
    writing_export.add_argument("--output", type=Path, required=True)
    writing_import = commands.add_parser("writing-import", help="无模型地导入精确的人工作者修订")
    writing_import.add_argument("parent_version", type=Path)
    writing_import.add_argument("file", type=Path)
    writing_import.add_argument("--author", required=True)
    writing_import.add_argument("--reason", required=True)
    writing_import.add_argument("--delete-id", action="append", default=[], dest="delete_ids")
    writing_import.add_argument("--domain-version", type=Path)
    writing_review = commands.add_parser("writing-review", help="划分并独立复核一个明确的人工作者版本")
    writing_review.add_argument("version", type=Path)
    writing_status = commands.add_parser("writing-status", help="只读检查写作版本、内容范围和复核阶段")
    writing_status.add_argument("--document-id")
    self_test = commands.add_parser("self-test", help="在全新目录运行已安装包的离线使用自检")
    self_test.add_argument("--output", type=Path, required=True)
    return parser


_WRITING_COMMANDS = {
    "domain-extract", "writing-outline", "writing-draft", "writing-export-edit",
    "writing-import", "writing-review", "writing-status",
}


def _validate_writing_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command not in _WRITING_COMMANDS:
        return
    if hasattr(args, "question") and (
        not args.question.strip() or args.question != args.question.strip()
    ):
        parser.error("--question must contain exact nonblank text without outer whitespace")
    if hasattr(args, "work_ids") and (
        len(args.work_ids) != len(set(args.work_ids))
        or any(not value.strip() or value != value.strip() for value in args.work_ids)
    ):
        parser.error("--work-id values must be unique exact nonblank IDs")
    if args.command == "domain-extract" and (
        args.domain != "builtin:scheduling" and not Path(args.domain).is_absolute()
    ):
        parser.error("--domain must be exactly builtin:scheduling or an absolute JSON path")
    if args.command == "writing-outline":
        if bool(args.candidate_version) != bool(args.candidate_id):
            parser.error("--candidate-version and --candidate-id must be supplied together")
        if args.assessment_version and not args.candidate_version:
            parser.error("--assessment-version requires --candidate-version and --candidate-id")
    if args.command == "writing-import" and (
        not args.author.strip() or args.author != args.author.strip()
        or not args.reason.strip() or args.reason != args.reason.strip()
    ):
        parser.error("--author and --reason must contain exact nonblank text")
    if args.command == "writing-import" and len(args.delete_ids) != len(set(args.delete_ids)):
        parser.error("--delete-id values must be unique")
    if args.command == "writing-outline":
        version_paths = [args.domain_version, args.trends, args.candidate_version,
                         args.assessment_version]
    elif args.command == "writing-draft":
        version_paths = [args.outline_version]
    elif args.command == "writing-export-edit":
        version_paths = [args.version]
    elif args.command == "writing-import":
        version_paths = [args.parent_version, args.domain_version]
    elif args.command == "writing-review":
        version_paths = [args.version]
    else:
        version_paths = []
    if any(path is not None and (path.name == "latest" or not path.name.startswith("v-"))
           for path in version_paths):
        parser.error("writing dependencies require concrete v-* version directories, never latest")


def _domain_source(value: str) -> str:
    return value if value == "builtin:scheduling" else str(Path(value).resolve())


def _writing_document_id(question: str, work_ids: list[str], domain_path: str) -> str:
    payload = json.dumps(
        ["writing-cli-document-1", question, work_ids, domain_path],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "doc_" + hashlib.sha256(payload).hexdigest()[:32]


def _writing_prefix(root: Path, offline: bool) -> list[str]:
    return ["academic-agent", "--project", str(root.resolve()), *(["--offline"] if offline else [])]


def _writing_argv(
    root: Path,
    args: argparse.Namespace,
    *,
    selected_question: str | None = None,
    selected_work_ids: list[str] | None = None,
) -> list[str]:
    prefix = _writing_prefix(root, args.offline)
    if args.command == "domain-extract":
        return [
            *prefix, "domain-extract", "--question", args.question,
            *(part for work_id in args.work_ids for part in ("--work-id", work_id)),
            "--domain", _domain_source(args.domain),
        ]
    if args.command == "writing-outline":
        question = args.question if selected_question is None else selected_question
        work_ids = args.work_ids if selected_work_ids is None else selected_work_ids
        command = [
            *prefix, "writing-outline", "--question", question,
            *(part for work_id in work_ids for part in ("--work-id", work_id)),
            "--domain-version", str(args.domain_version.resolve()),
        ]
        if args.trends:
            command.extend(["--trends", str(args.trends.resolve())])
        if args.candidate_version:
            command.extend(["--candidate-version", str(args.candidate_version.resolve()),
                            "--candidate-id", args.candidate_id])
        if args.assessment_version:
            command.extend(["--assessment-version", str(args.assessment_version.resolve())])
        return command
    if args.command == "writing-draft":
        return [*prefix, "writing-draft", str(args.outline_version.resolve())]
    if args.command == "writing-export-edit":
        return [*prefix, "writing-export-edit", str(args.version.resolve()),
                "--output", str(args.output.resolve())]
    if args.command == "writing-import":
        command = [*prefix, "writing-import", str(args.parent_version.resolve()), str(args.file.resolve()),
                   "--author", args.author, "--reason", args.reason]
        for block_id in args.delete_ids:
            command.extend(["--delete-id", block_id])
        if args.domain_version:
            command.extend(["--domain-version", str(args.domain_version.resolve())])
        return command
    if args.command == "writing-review":
        return [*prefix, "writing-review", str(args.version.resolve())]
    command = [*prefix, "writing-status"]
    if args.document_id:
        command.extend(["--document-id", args.document_id])
    return command


def _with_project_prefix(root: Path, offline: bool, argv: list[str]) -> list[str]:
    command = list(argv)
    if command and command[0] == "academic-agent":
        command = command[1:]
    return [*_writing_prefix(root, offline), *command]


def _support_review(root: Path, output: str | None) -> dict | None:
    if not output:
        return None
    try:
        from .writing import load_writing_version

        path = Path(output)
        path = path.resolve() if path.is_absolute() else (root / path).resolve()
        artifact = load_writing_version(path)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    counts = {decision: 0 for decision in ("supported", "partial", "contradicted", "insufficient")}
    gaps = []
    for item in artifact.review.items:
        counts[item.decision] += 1
        if item.decision != "supported":
            gaps.append(item.model_dump(mode="json"))
    return {
        "status": "complete" if not gaps else "issues_found",
        "decisions": counts,
        "gaps": gaps,
        "reviewer": "independent_model_call",
    }


def _writing_succeeded(result: dict) -> bool:
    return result.get("status") == "complete"


def _operational_writing_result(root: Path, args: argparse.Namespace, result: dict, runner=None) -> dict:
    if _writing_succeeded(result):
        if args.command in {"writing-outline", "writing-draft", "writing-review"}:
            review = _support_review(root, result.get("output"))
            if review is not None:
                return {**result, "support_review": review}
        return result
    status = str(result.get("status") or "failed")
    gaps = result.get("gaps") if isinstance(result.get("gaps"), list) else []
    first_reason = next(
        (gap.get("reason") for gap in gaps if isinstance(gap, dict) and gap.get("reason")),
        None,
    )
    actionable_reasons = {"outline_review_required", "domain_rebind_required", "domain_selection_mismatch"}
    explicit_reason = first_reason if first_reason in actionable_reasons else None
    reasons = {
        "paused_budget": "model_budget_exhausted",
        "needs_input": explicit_reason or (
            "writing_not_started" if args.command == "writing-status" else "model_input_required"
        ),
        "input_too_large": "model_input_too_large",
        "stale": explicit_reason or "writing_inputs_changed",
        "failed": explicit_reason or "writing_operation_failed",
    }
    reason = reasons.get(status, first_reason or "writing_operation_failed")
    explanations = {
        "model_budget_exhausted": "模型调用预算已用尽；明确提高预算后可重复同一命令，已验证阶段会复用。",
        "model_input_required": "模型后端正在等待可验证输入；填写 Exchange 响应后重复同一命令。",
        "model_input_too_large": "完整材料超过输入上限；请显式减少 --work-id 范围，不会静默截断。",
        "outline_review_required": "该人工大纲仍待独立复核；先运行 writing-review，再用实际复核输出生成正文。",
        "domain_rebind_required": "父稿的来源、复核或领域包已变化；先按原问题、作品和包重新提取领域字段。",
        "domain_selection_mismatch": "问题或作品范围与明确领域版本不一致；请使用该版本保存的原选择。",
        "writing_not_started": "尚无写作版本；先完成受支持的研究材料和领域提取，再生成大纲。",
    }
    enriched = {
        **result,
        "reason": reason,
        "explanation": explanations.get(
            reason, "写作操作未完成；请检查 writing-status 和 guide 给出的具体修复动作。"
        ),
        "resume_argv": _writing_argv(root, args),
    }
    if reason == "domain_rebind_required":
        enriched["gaps"] = [
            {key: value for key, value in item.items() if key != "reimport_argv"}
            if isinstance(item, dict) else item
            for item in gaps
        ]
        gap = next((item for item in gaps if isinstance(item, dict)
                    and item.get("reason") == reason), None)
        if gap and isinstance(gap.get("extract_argv"), list):
            extraction = _with_project_prefix(root, args.offline, gap["extract_argv"])
            enriched["resume_argv"] = extraction
            enriched["actions"] = [{
                "summary": "先执行此命令并取得实际、已验证的领域版本输出目录。",
                "argv": extraction,
            }]
            enriched["next_step"] = {
                "summary": (
                    "领域提取成功后，把它返回的实际 output 目录传给 --domain-version，"
                    "并保持同一 parent、编辑文件、author、reason 和 delete-id；随后对导入返回的 pending output "
                    "运行 writing-review。"
                ),
                "parent_version": str(args.parent_version.resolve()),
                "edit_file": str(args.file.resolve()),
                "author": args.author,
                "reason": args.reason,
                "delete_ids": args.delete_ids,
            }
    elif reason == "outline_review_required":
        gap = next((item for item in gaps if isinstance(item, dict)
                    and item.get("reason") == reason), None)
        if gap and isinstance(gap.get("argv"), list):
            review = _with_project_prefix(root, args.offline, gap["argv"])
            enriched["resume_argv"] = review
            enriched["actions"] = [{
                "summary": "先独立复核这个 pending 大纲版本。",
                "argv": review,
            }]
            enriched["next_step"] = (
                "复核完成后，把 writing-review 返回的实际 reviewed output 传给 writing-draft。"
            )
    if args.command == "writing-status":
        enriched["suggested_argv"] = [*_writing_prefix(root, args.offline), "guide"]
    if runner is not None:
        budget = runner.store.budget()
        enriched["budget"] = {
            "consumed_or_reserved_calls": budget["consumed_or_reserved_calls"],
            "max_model_calls": runner.config.max_model_calls,
        }
    return enriched


def _innovation_argv(root: Path, args: argparse.Namespace, *, offline: bool | None = None) -> list[str]:
    use_offline = args.offline if offline is None else offline
    prefix = ["academic-agent", "--project", str(root.resolve()), *(["--offline"] if use_offline else [])]
    if args.command == "innovate":
        return [*prefix, "innovate", *(part for work_id in args.work_id for part in ("--work-id", work_id))]
    if args.command == "challenge":
        return [*prefix, "challenge", args.candidate_id, "--round", str(args.round_number)]
    if args.command == "assess-innovation":
        return [*prefix, "assess-innovation", args.candidate_id]
    return [*prefix, "innovation-status"]


def _innovation_succeeded(args: argparse.Namespace, result: dict) -> bool:
    if args.command == "challenge":
        return result.get("status") in {"needs_reading", "needs_assessment"}
    return result.get("status") == "complete"


def _operational_innovation_result(root: Path, args: argparse.Namespace, result: dict, runner=None) -> dict:
    status = str(result.get("status") or "failed")
    if _innovation_succeeded(args, result):
        return result
    gaps = result.get("gaps") if isinstance(result.get("gaps"), list) else result.get("unresolved_tasks", [])
    reason = status if args.command == "challenge" else next(
        (gap.get("reason") for gap in reversed(gaps) if isinstance(gap, dict) and gap.get("reason")),
        "innovation_operation_failed",
    )
    input_overflow_explanation = (
        "完整评估材料超过输入上限；必须保留所有已纳入的候选前提与相关反例。"
        "请更正不应纳入的研究范围，或改用明确兼容当前完整输入上限的模型后端；"
        "若两者均不适用，此评估不能继续。"
        if args.command == "assess-innovation"
        else "完整研究材料超过输入上限；请用 --work-id 显式缩小文献范围。"
    )
    explanations = {
        "model_budget_exhausted": "模型调用预算已用尽；提高项目预算后可用同一命令恢复，已完成调用会复用。",
        "model_input_required": "模型后端正在等待可验证输入；若使用 exchange，请填写对应响应文件后重试。",
        "model_artifact_invalid": "模型返回的结构化结果未通过验证；请检查后端响应后重试。",
        "model_input_too_large": input_overflow_explanation,
        "research_inputs_changed": "研究材料在操作期间发生变化；请检查状态后重试。",
        "research_input_invalid": "当前研究材料缺失或无法验证；请先按 guide 修复阅读与复核状态。",
        "no_current_supported_evidence": "尚无当前且已复核支持的证据；请先完成阅读与复核。",
        "general_search_coverage": "通用检索覆盖仍不完整；评估会保留该证据缺口。",
        "source_failed": "反例检索来源失败或覆盖不完整；修复来源后重复同一轮即可恢复。",
        "not_searched": "反例检索尚未取得当前来源收据；请检查项目来源配置后重复同一轮。",
        "needs_reading": "反例材料仍需筛选、关联可读全文、阅读并复核。",
        "needs_assessment": "反例材料已准备好；请运行 assess-innovation 完成四维评估。",
        "candidate_artifact_invalid": "候选历史文件缺失或校验失败；请从可信备份修复候选产物，再按指南检查。",
        "assessment_artifact_invalid": "评估历史文件缺失或校验失败；请从可信备份修复评估产物，再按指南检查。",
        "assessment_inputs_changed": "评估输入已经变化；请按当前证据重新评估候选。",
        "assessment_inputs_invalid": "评估输入无法验证；请先按指南修复研究材料。",
    }
    enriched = {**result, "reason": reason, "explanation": explanations.get(
        reason, "创新操作未完成；请先查看 innovation-status 与 guide 给出的修复步骤。"
    ), "resume_argv": _innovation_argv(root, args)}
    if args.command == "innovation-status":
        enriched["suggested_argv"] = ["academic-agent", "--project", str(root.resolve()),
                                      *(["--offline"] if args.offline else []), "guide"]
    if runner is not None:
        enriched["budget"] = {**runner.store.budget(), "max_model_calls": runner.config.max_model_calls}
    return enriched


def _challenge_error(root: Path, args: argparse.Namespace, runner, *, reason: str) -> dict:
    explanations = {
        "candidate_not_found": "未找到指定候选；请先运行 innovation-status 检查可用候选 ID。",
        "challenge_offline": "离线模式禁止反例来源检索；可保持离线检查状态，或明确启用配置的论文来源。",
        "challenge_round_order": "第二轮不能先于第一轮开始；请先完成或恢复第一轮。",
        "challenge_scope_changed": "该轮检索范围已变化；请恢复原来源与上限后重试。",
        "challenge_configuration_changed": "项目配置在命令期间变化；请重新运行命令读取当前配置。",
        "challenge_failed": "反例检索无法继续；请运行 guide 检查候选与研究状态。",
    }
    result = {
        "status": "failed",
        "reason": reason,
        "explanation": explanations[reason],
        "resume_argv": _innovation_argv(root, args),
        "suggested_argv": ["academic-agent", "--project", str(root.resolve()),
                           *(["--offline"] if args.offline else []), "guide"],
        "budget": {**runner.store.budget(), "max_model_calls": runner.config.max_model_calls},
    }
    if reason == "challenge_offline":
        result["source_network_action"] = {
            "summary": "此替代命令会明确移除 --offline，并启用项目已配置的论文来源网络检索。",
            "argv": _innovation_argv(root, args, offline=False),
        }
    return result


def _challenge_exception_reason(exc: ValueError) -> str:
    message = str(exc)
    if message == "Challenge network search is unavailable in offline mode":
        return "challenge_offline"
    if message == "Start challenge round 1 before round 2":
        return "challenge_round_order"
    if message == "Challenge round scope changed; existing round cannot be reset":
        return "challenge_scope_changed"
    if message in {"Runner configuration changed; recreate the runner",
                   "Challenge project cap changed; restore it to resume failed queries"}:
        return "challenge_configuration_changed"
    return "challenge_failed"


def doctor(root: Path) -> dict:
    from .backends import CodexBackend
    packages = {}
    for name in ("academic-work-agent", "pydantic", "httpx", "pypdf", "langgraph", "langgraph-checkpoint-sqlite"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    backend = CodexBackend()
    result = {"python": sys.version.split()[0], "packages": packages, "codex_on_path": bool(shutil.which("codex")),
              "project_exists": (root / "project.json").exists(),
              "credentials": "not_read", "codex_login": "not_checked; run codex login status in your user terminal"}
    if hasattr(backend, "doctor"):
        result["codex"] = backend.doctor()
    if result["project_exists"]:
        config = load_config(root)
        result["backend"] = config.backend
        result["proxy_configured"] = bool(config.proxy_url)
    return result


def demo(root: Path) -> dict:
    from .demo import seed_demo
    from .workflow import Runner
    seed_demo(root)
    runner = Runner(root, offline=True)
    return {"synthetic_demo": True, **runner.run()}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_writing_arguments(parser, args)
    try:
        if args.command == "init":
            config = ProjectConfig(title=args.title, question=args.question, queries=args.query, backend=args.backend,
                                   model=args.model, proxy_url=args.proxy or None, max_model_calls=args.max_model_calls)
            create_project(args.project, config)
            _emit({"created": str(args.project.resolve()), "backend": config.backend})
            return 0
        if args.command == "doctor":
            _emit(doctor(args.project))
            return 0
        if args.command == "demo":
            _emit(demo(args.project))
            return 0
        if args.command == "guide":
            from .guide import project_guide
            _emit(project_guide(args.project, offline=args.offline))
            return 0
        if args.command == "writing-status":
            from .writing import writing_status

            result = _operational_writing_result(
                args.project, args, writing_status(args.project, args.document_id)
            )
            _emit(result)
            return 0 if _writing_succeeded(result) else 2
        if args.command == "writing-export-edit":
            from .writing_revisions import export_edit

            output = export_edit(args.project, args.version.resolve(), args.output.resolve())
            _emit({"status": "complete", "output": str(output)})
            return 0
        if args.command == "innovation-status":
            from .innovations import innovation_status
            result = _operational_innovation_result(args.project, args, innovation_status(args.project))
            _emit(result)
            return 0 if _innovation_succeeded(args, result) else 2
        if args.command == "verify-trends":
            from .trends import verify_trends
            _emit(verify_trends(args.directory))
            return 0
        if args.command == "self-test":
            from .usage import run_self_test
            result = run_self_test(args.output)
            _emit(result)
            return 0 if result["passed"] else 2
        if args.command == "trends":
            from .store import Store
            from .trends import CorpusAnnotations, build_trends
            annotations = None
            if args.annotations:
                annotations = CorpusAnnotations.model_validate_json(args.annotations.read_text(encoding="utf-8"))
            directory = build_trends(Store(args.project), annotations=annotations, date_basis=args.date_basis)
            _emit({"trends": str(directory), "date_basis": args.date_basis})
            return 0
        outline_domain = None
        if args.command == "writing-outline":
            from .domain import load_domain_version

            outline_domain = load_domain_version(args.domain_version.resolve())
            saved = outline_domain.context.selection
            if args.question != saved.question or args.work_ids != saved.work_ids:
                result = _operational_writing_result(args.project, args, {
                    "status": "failed",
                    "input_hash": outline_domain.context.input_hash,
                    "output": None,
                    "gaps": [{
                        "reason": "domain_selection_mismatch",
                        "question": saved.question,
                        "work_ids": saved.work_ids,
                    }],
                })
                result["resume_argv"] = _writing_argv(
                    args.project,
                    args,
                    selected_question=saved.question,
                    selected_work_ids=saved.work_ids,
                )
                _emit(result)
                return 2
        from .workflow import Runner
        runner = Runner(args.project, offline=args.offline)
        command = args.command
        if command == "domain-extract":
            from .domain import extract_domain
            from .writing_models import WritingSelection

            domain_path = _domain_source(args.domain)
            selection = WritingSelection(
                document_id=_writing_document_id(args.question, args.work_ids, domain_path),
                question=args.question,
                work_ids=args.work_ids,
                domain_path=domain_path,
                trend_directory=None,
                candidate_directory=None,
                candidate_id=None,
                assessment_directory=None,
            )
            result = extract_domain(runner, selection)
            result = _operational_writing_result(args.project, args, result, runner)
        elif command == "writing-outline":
            from .writing import create_outline
            from .writing_models import WritingSelection

            saved = outline_domain.context.selection
            selection = WritingSelection(
                document_id=saved.document_id,
                question=saved.question,
                work_ids=saved.work_ids,
                domain_path=saved.domain_path,
                trend_directory=str(args.trends.resolve()) if args.trends else None,
                candidate_directory=(str(args.candidate_version.resolve())
                                     if args.candidate_version else None),
                candidate_id=args.candidate_id,
                assessment_directory=(str(args.assessment_version.resolve())
                                      if args.assessment_version else None),
            )
            result = create_outline(runner, selection, args.domain_version.resolve())
            result = _operational_writing_result(args.project, args, result, runner)
        elif command == "writing-draft":
            from .writing import create_draft

            result = create_draft(runner, args.outline_version.resolve())
            result = _operational_writing_result(args.project, args, result, runner)
        elif command == "writing-import":
            from .writing_revisions import import_revision

            result = import_revision(
                runner,
                args.parent_version.resolve(),
                args.file.resolve(),
                author=args.author,
                reason=args.reason,
                delete_ids=args.delete_ids,
                domain_version=args.domain_version.resolve() if args.domain_version else None,
            )
            result = _operational_writing_result(args.project, args, result, runner)
        elif command == "writing-review":
            from .writing_revisions import review_writing

            result = review_writing(runner, args.version.resolve())
            result = _operational_writing_result(args.project, args, result, runner)
        elif command == "innovate":
            from .innovations import create_innovations
            result = create_innovations(runner, work_ids=args.work_id or None)
            result = _operational_innovation_result(args.project, args, result, runner)
        elif command == "challenge":
            from .challenges import challenge_candidate
            from .innovations import innovation_status
            known = {row.get("candidate_id") for row in innovation_status(args.project).get("candidates", [])}
            if args.candidate_id not in known:
                result = _challenge_error(args.project, args, runner, reason="candidate_not_found")
            else:
                try:
                    result = challenge_candidate(runner, args.candidate_id, round_number=args.round_number)
                    result = _operational_innovation_result(args.project, args, result, runner)
                except ValueError as exc:
                    result = _challenge_error(args.project, args, runner,
                                              reason=_challenge_exception_reason(exc))
        elif command == "assess-innovation":
            from .innovations import assess_candidate, innovation_status
            known = {row.get("candidate_id") for row in innovation_status(args.project).get("candidates", [])}
            if args.candidate_id not in known:
                raise ValueError(f"未找到候选 {args.candidate_id}；请先运行 innovation-status 检查可用 ID")
            result = assess_candidate(runner, args.candidate_id)
            result = _operational_innovation_result(args.project, args, result, runner)
        elif command == "search":
            result = runner.search(args.query)
        elif command == "import":
            result = {"imported": [w.model_dump() for w in runner.import_records(args.file)]}
        elif command == "attach":
            result = runner.attach(args.work_id, args.file)
        elif command == "fetch":
            result = runner.fetch(args.work_id)
        elif command == "screen":
            ids = [w.id for w in runner.store.works()] if args.all else args.work_ids
            if not ids:
                raise ValueError("Provide work IDs or explicitly choose --all")
            # Validate the complete selection before mutating any record.
            for work_id in ids:
                runner.store.work(work_id)
            for work_id in ids:
                runner.store.screen(work_id, args.decision, args.reason)
            result = {"screened": ids, "decision": args.decision}
        elif command == "read":
            result = runner.read_selected(work_id=args.work_id)
        elif command == "review":
            if args.claim_id:
                result = runner.manual_review(args.claim_id, args.decision, args.reason or "", args.reviewer or "")
            else:
                if any((args.decision, args.reason, args.reviewer)):
                    raise ValueError("Manual review requires --claim-id")
                result = runner.review_selected(work_id=args.work_id)
        elif command == "ask":
            result = runner.read_selected(work_id=args.work_id, question=args.question)
            result["review"] = runner.review_selected(work_id=args.work_id, answers=True, question=args.question)
            from .exports import current_cards, decisions
            cards = current_cards(runner.store, kind="answer", question=args.question)
            verdicts = decisions(runner.store, cards)
            result["answers"] = [{"work_id": card.work_id, "claim": c.model_dump(), "review": verdicts.get(c.id),
                                  "usable": verdicts.get(c.id, {}).get("verdict") == "supported"
                                  and not runner.store.work(card.work_id).retracted
                                  and runner.store.work(card.work_id).screening == "include"}
                                 for card in cards.values() if not args.work_id or card.work_id == args.work_id
                                 for c in card.claims]
        elif command == "export":
            result = {"export": str(runner.export())}
        elif command == "status":
            result = runner.status()
        elif command in {"run", "resume"}:
            result = runner.run(resume=command == "resume", run_id=args.run_id, stop_after=args.stop_after)
        elif command == "cancel":
            runner.store.cancel_job(args.job_id)
            result = {"cancel_requested": args.job_id,
                      "note": "Late commits are fenced; a dispatched provider request may still consume usage"}
        else:
            raise ValueError("Unknown command")
        _emit(result)
        if command in _WRITING_COMMANDS:
            return 0 if _writing_succeeded(result) else 2
        if command in {"innovate", "challenge", "assess-innovation"}:
            return 0 if _innovation_succeeded(args, result) else 2
        failures = [*result.get("results", []), *result.get("review", {}).get("results", [])]
        return 2 if any(item.get("status") == "failed" for item in failures) else 0
    except KeyboardInterrupt:
        _emit({"error": "Interrupted; completed results remain resumable", "type": "Interrupted"})
        return 130
    except Exception as exc:
        # Library and backend exceptions are deliberately concise and redact provider output.
        if getattr(args, "command", "") in _WRITING_COMMANDS:
            failed = _operational_writing_result(args.project, args, {
                "status": "failed",
                "input_hash": None,
                "output": None,
                "gaps": [{"reason": "writing_command_error", "detail": str(exc)}],
            }, locals().get("runner"))
            failed["error"] = f"写作操作无法继续：{exc}"
            failed["type"] = type(exc).__name__
            failed["suggested_argv"] = [*_writing_prefix(args.project, args.offline), "writing-status"]
            _emit(failed)
        elif getattr(args, "command", "") in {"innovate", "challenge", "assess-innovation", "innovation-status"}:
            _emit({"error": f"创新操作无法继续：{exc}", "type": type(exc).__name__,
                   "suggested_argv": ["academic-agent", "--project", str(args.project.resolve()),
                                      "innovation-status"]})
        else:
            _emit({"error": str(exc), "type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
