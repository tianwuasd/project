"""Panel/CLI publication orchestration; archive review precedes authenticated upload."""

import shlex
from pathlib import Path

from ..recovery import read_json
from . import inspection, jobs
from .launcher import personal_path, write_json
from .prompts import ask


def run_upload(args, profile, interactive):
    from . import upload
    from .resume import reject_live_commands

    package = Path(args.upload_package).expanduser().resolve() if args.upload_package else None
    if package:
        manifest = read_json(package / "manifest.json")
        job = Path(manifest["job_directory"]).resolve()
        runtime = personal_path(manifest["runtime"])
    else:
        runtime = personal_path(args.runtime or profile["runtime"])
        job = (
            inspection.latest(runtime)
            if args.upload_job == "latest"
            else Path(args.upload_job).expanduser().resolve()
        )
        if job is None:
            raise ValueError("没有可上传的历史任务")
        saved = read_json(job / "plan.json")
        if args.runtime is None:
            runtime = personal_path(saved["runtime"])
    plan = read_json(job / "plan.json")
    if Path(plan["runtime"]).resolve() != runtime:
        raise ValueError("上传须使用原任务 runtime")
    if args.runtime and personal_path(args.runtime) != runtime:
        raise ValueError("上传须使用原任务 runtime")
    with jobs.acquire_lock(runtime):
        if inspection.job_active(job):
            raise ValueError("任务仍在运行，完成后再上传")
        reject_live_commands(plan)
        package = package or upload.prepare_upload(job, runtime)
        upload.show_preview(package)
        command = "bash start_server.sh --upload-package " + shlex.quote(str(package)) + " --yes-upload"
        print("上传/失败后重试命令：" + command)
        confirmed = bool(args.yes_upload)
        if not confirmed and interactive:
            confirmed = ask("确认将以上日志、配置、指标及权重发布到 GitHub Release？[y/N]：").lower() == "y"
        if not confirmed:
            print("已准备上传包，尚未发送。需要上传时执行上方命令。")
            return 0
        url = upload.upload_package(package)
        receipts = job / "uploads"
        receipts.mkdir(exist_ok=True)
        write_json(
            receipts / (package.name + ".json"),
            {
                "url": url,
                "package": str(package),
                "uploaded_at": jobs.now(),
            },
        )
        print("上传完成：" + url)
        return 0
