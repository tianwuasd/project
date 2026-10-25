"""CT publication lifecycle adapted to Vista frozen tasks, with preview and same-package retry."""
from pathlib import Path
import shlex
from task_runtime import load_plan, active_processes, runtime_lock, latest, now
from server_data import read_json, write_json
from . import upload
from .prompts import ask


def run_upload(args, profile, interactive):
    package=Path(args.upload_package).expanduser().resolve() if args.upload_package else None
    if package:
        manifest=upload._validate_package(package)[1]
        job=Path(manifest['job_directory']);runtime=Path(manifest['runtime'])
    else:
        runtime=Path(args.runtime or profile['runtime']).expanduser().resolve()
        job=latest(runtime) if args.upload_job=='latest' else Path(args.upload_job).expanduser().resolve()
        if not job:raise ValueError('没有可上传任务')
    plan=load_plan(job)
    if Path(plan['runtime']).resolve()!=runtime or args.runtime and Path(args.runtime).resolve()!=runtime:
        raise ValueError('上传必须使用原任务runtime')
    with runtime_lock(runtime):
        if active_processes(job):raise ValueError('任务仍有活动进程，请等待其退出')
        package=package or upload.prepare_upload(job,runtime)
        upload.show_preview(package)
        print('原包重试：bash start_server.sh --runtime '+shlex.quote(str(runtime))+' --upload-package '+shlex.quote(str(package))+' --yes-upload')
        confirmed=args.yes_upload or interactive and ask('确认通过同一SSH密钥+Git LFS上传此清单？[y/N]：').lower()=='y'
        if not confirmed:
            print('已准备不可变上传包，尚未发送。');return 0
        url=upload.upload_package(package)
        write_json(job/'uploads'/f'{package.name}.json',dict(url=url,package=str(package),uploaded_at=now()))
        print('上传完成：'+url)
        return 0
