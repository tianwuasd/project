"""Whole-heart task panel: stdlib control plane; compute starts only on submission."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from model_catalog import MODELS, FOLDS, SCOPES, parse_models, parse_folds
from server_data import read_json
import task_runtime as jobs
from task_queries import query, follow, result_lines, read_optional


class Back(Exception):
    pass


class Quit(Exception):
    pass


def ask(label, default=None):
    value = input(label + (f' [{default}]' if default is not None else '') + '：').strip()
    if value.lower() in {'b','back','返回'}:
        raise Back
    if value.lower() in {'q','quit','exit','退出'}:
        raise Quit
    value = value or default
    if isinstance(value, str) and len(value) > 1 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value


def choose_job(runtime, allow_none=False):
    records = jobs.history(runtime)
    for i, row in enumerate(records, 1):
        print(f"{i}. {jobs.clean(row['name'])} [{row['status']}] {row['directory']}")
    text = '选择历史编号；d手工填写旧任务/工作区'
    if allow_none:
        text += '；0不使用历史输入'
    value = ask(text, '0' if allow_none else '1' if records else 'd')
    if allow_none and value == '0':
        return None
    if value == 'd':
        return ask('完整路径')
    if not str(value).isdigit() or not 1 <= int(value) <= len(records):
        raise ValueError('历史编号无效')
    return records[int(value)-1]['directory']


def resolve_gpus(value):
    from evaluation_gpu import snapshot, allowed_ids
    rows = snapshot()
    visible = allowed_ids(rows, {'allowed_gpus': None})
    tokens = str(value).replace('，', ',').replace(',', ' ').split()
    if not tokens:
        raise ValueError('请明确填写获准GPU，不能默认使用全部显卡')
    result = []
    for token in tokens:
        matched = [r['uuid'] for r in rows if token in [r['index'], r['uuid']] and r['uuid'] in visible]
        if len(matched) != 1 or matched[0] in result:
            raise ValueError('GPU未知、重复或超出分配范围：' + token)
        result.extend(matched)
    return result


def new_config(args, saved, interactive=True):
    from zmic44_setup import DEFAULT_PATHS
    def value(argument, key, default, label):
        if argument is not None:
            return argument
        suggestion = saved.get(key, default)
        return ask(label, suggestion) if interactive else suggestion
    runtime = str(Path(args.runtime or saved.get('runtime', DEFAULT_PATHS['base'])).expanduser().resolve())
    tasks = jobs.parse_tasks(value(args.tasks, 'tasks', 'environment,prepare,smoke', '步骤：0环境 1预处理 2短测 3训练 4预测评估 5已有评价汇总；可多选'))
    source = args.input_job
    if source is None and interactive and tasks != ['environment']:
        source = choose_job(runtime, allow_none=True)
    if source == 'none':
        source = None
    source_config = None
    if source:
        old = read_optional(Path(source) / 'plan.json')
        if old:
            source_config = old
        else:
            old = read_optional(Path(source) / '00_admin/server_config.json')
            if old:
                source_config = {'data': old['data_root'], 'gpu_memory': old['gpu_memory_gb']}
    default_data = (source_config or {}).get('data', saved.get('data', DEFAULT_PATHS['data']))
    methods = parse_models(args.models)
    folds = parse_folds(args.folds or args.scope or 'ct-first')
    gpus, count = [], 0
    if jobs.GPU_TASKS.intersection(tasks):
        print('\n模型可多选：' + '；'.join(f'{i} {v["label"]}' for i,v in enumerate(MODELS.values(), 1)))
        if interactive and args.models is None:
            methods = parse_models(ask('模型编号或名称，逗号分隔；all全部', ','.join(saved.get('methods', list(MODELS)))))
        if interactive and args.folds is None and args.scope is None:
            folds = parse_folds(ask('范围：ct-first / mr-first / all，或逗号分隔方向', 'ct-first'))
        print(query('gpu', runtime))
        selection = args.gpus if args.gpus is not None else ask('填写本次获准的候选GPU编号/UUID；繁忙时后台等待') if interactive else None
        if selection is None:
            raise ValueError('非交互GPU任务必须提供 --gpus')
        gpus = resolve_gpus(selection)
        count = int(args.gpu_count or (ask('最多并行使用几张GPU', len(gpus)) if interactive else len(gpus)))
    config = dict(runtime=runtime,
                  results=value(args.results, 'results', DEFAULT_PATHS['work'], '结果根目录（每次自动生成独立子目录）'),
                  data=args.data or (ask('原始数据目录', default_data) if interactive and tasks != ['environment'] else default_data),
                  name=value(args.name, 'name', 'whole-heart', '任务显示名称'), tasks=tasks,
                  methods=methods, folds=folds, gpus=gpus, gpu_count=count,
                  gpu_memory=float(args.gpu_memory or (source_config or {}).get('gpu_memory', saved.get('gpu_memory', 5))),
                  input_job=source, preprocess_cache=args.preprocess_cache or 'reuse')
    if interactive and args.gpu_memory is None and any(t in tasks for t in ['prepare','smoke','train']):
        config['gpu_memory'] = float(ask('网络规划显存预算GiB（已有输入需保持一致）', config['gpu_memory']))
    if source and 'prepare' in tasks and interactive and args.preprocess_cache is None:
        choice = ask('预处理：1尝试沿用（缓存不兼容则新建） 2新建', '1')
        if choice not in {'1','2'}:
            raise ValueError('缓存选择无效')
        config['preprocess_cache'] = 'reuse' if choice == '1' else 'new'
    return jobs.validate_request(config)


def result_menu(job):
    sections = {'1': ('summary','分割摘要'), '2': ('cases','病例与七结构'),
                '3': ('training','训练与数据'), '4': ('files','文件与来源')}
    while True:
        print('\n结果中心：' + str(job) + '\n1 分割摘要  2 病例与七结构  3 训练与数据  4 文件与来源\nb/0 返回  q 退出面板')
        try:
            choice = ask('选择', '1')
            if choice == '0':
                return
            if choice not in sections:
                print('请输入有效编号'); continue
            section, title = sections[choice]
            detailed, page = False, 0
            lines = result_lines(job, section, detailed)
            while True:
                pages = max(1, (len(lines)+19)//20)
                page = max(0, min(page, pages-1))
                print(f'\n{title} / {"详细" if detailed else "简洁"} / {page+1}/{pages}')
                print('\n'.join(jobs.clean(v) for v in lines[page*20:(page+1)*20]))
                try:
                    action = ask('d详细 s简洁 n下一页 p上一页 r刷新 b/0返回 q退出', 'b')
                except Back:
                    break
                if action == '0':
                    break
                if action in {'d','s','r'}:
                    detailed = action == 'd' if action != 'r' else detailed
                    lines, page = result_lines(job, section, detailed), 0
                elif action in {'n','p'}:
                    page += 1 if action == 'n' else -1
        except (Back, KeyboardInterrupt):
            return


def panel(args, runtime, saved):
    while True:
        print('\nWhole-heart 服务器任务面板\n运行目录：' + runtime + '\n' + query('status', runtime))
        print('n 新建  r 续跑/追加  k 独立重跑  h 历史  o 结果\ns 状态  g 显卡(1秒刷新)  m 模型  l 日志  p 我的进程\nx 中断指定任务  u 更新程序  0/q 退出（后台继续）')
        try:
            action = ask('选择；回车只看状态', 's')
            if action == '0':
                return 0
            if action in {'s','g','m','l','p','h'}:
                kind = {'s':'status','g':'gpu','m':'models','l':'logs','p':'processes','h':'history'}[action]
                if action == 'g':
                    follow(kind, runtime)
                else:
                    print(query(kind, runtime))
            elif action == 'o':
                result_menu(choose_job(runtime))
            elif action == 'n':
                config = new_config(args, saved)
                print(json.dumps({k:v for k,v in config.items() if k != 'gpus'}, ensure_ascii=False, indent=2))
                if ask('提交以上任务？输入 RUN；其它输入返回', '') == 'RUN':
                    jobs.submit(config, args.foreground)
                    saved = config
            elif action == 'r':
                job = choose_job(runtime)
                state = read_optional(Path(job) / 'state.json') or {}
                check = ask('续跑检查：simple简单 full全面 none跳过历史产物', 'simple')
                if state.get('status') == 'completed':
                    tasks = ask('追加未执行的后续步骤：predict / summarize')
                    jobs.append_tasks(job, tasks, args.foreground, check)
                else:
                    jobs.resume(job, args.foreground, check)
            elif action == 'k':
                job = choose_job(runtime)
                old = jobs.load_plan(job)
                if (read_optional(Path(job) / 'state.json') or {}).get('status') != 'completed':
                    raise ValueError('主动重跑只接受已完成任务；中断任务请续跑')
                config = dict(old, input_job=str(job), tasks=jobs.parse_tasks(ask('重跑步骤', ','.join(old['tasks']))),
                              name=ask('新任务名称', old['name']+' 重跑'), preprocess_cache='new')
                if ask('独立重跑并保留原结果？输入 RUN', '') == 'RUN':
                    jobs.submit(config, args.foreground)
            elif action == 'x':
                job = choose_job(runtime)
                print(query('status', runtime, job))
                if ask('仅中断上方任务，保留最后完整检查点；输入 STOP', '') == 'STOP':
                    jobs.stop(job)
            elif action == 'u':
                from task_update import update
                if ask('检查并切换本项目版本（不自动恢复任务），输入 UPDATE', '') == 'UPDATE':
                    update(runtime)
                    print('更新操作结束。重新打开面板载入选定版本。')
                    return 0
            else:
                print('未知菜单选项。')
        except Back:
            continue
        except Quit:
            return 0
        except KeyboardInterrupt:
            print('已退出当前操作，后台任务继续。')
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            print('操作未完成：' + jobs.clean(error))


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_root', nargs='?')
    p.add_argument('--runtime', '--base-dir', dest='runtime')
    p.add_argument('--results', '--work-dir', dest='results')
    p.add_argument('--data-dir', dest='data')
    p.add_argument('--name')
    p.add_argument('--tasks', help='environment,prepare,smoke,train,predict,summarize，多选')
    p.add_argument('--models', '--methods', dest='models', help='B0、B1或B0,B1；当前两种模型')
    p.add_argument('--scope', choices=list(SCOPES))
    p.add_argument('--folds')
    p.add_argument('--input-job')
    p.add_argument('--preprocess-cache', choices=['reuse','new'])
    p.add_argument('--gpu-count', type=int)
    p.add_argument('--gpus', '--gpu', dest='gpus')
    p.add_argument('--gpu-memory', type=float)
    p.add_argument('--foreground', action='store_true')
    p.add_argument('--non-interactive', '--yes-train', dest='non_interactive', action='store_true')
    p.add_argument('--query', choices=['status','gpu','models','logs','processes','history','results'])
    p.add_argument('--status', action='store_true')
    p.add_argument('--follow', action='store_true')
    p.add_argument('--job', help='指定命名任务或旧工作区，只读查询不会改变最近任务')
    p.add_argument('--result-section', choices=['summary','cases','training','files'], default='summary')
    p.add_argument('--result-detail', choices=['summary','detailed'], default='summary')
    p.add_argument('--resume-job', nargs='?', const='latest')
    p.add_argument('--resume-check', choices=['simple','full','none'], default='simple')
    p.add_argument('--append-tasks')
    p.add_argument('--rerun-job')
    p.add_argument('--stop', action='store_true')
    p.add_argument('--update', action='store_true')
    p.add_argument('--mode', choices=['check','smoke','train','resume','experiment','evaluate','status'])
    p.add_argument('--server-defaults', action='store_true')
    p.add_argument('--loaded-release', action='store_true', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    from zmic44_setup import DEFAULT_PATHS, SETTINGS
    legacy = read_optional(SETTINGS) or {}
    runtime = args.runtime or (None if args.server_defaults else legacy.get('base')) or DEFAULT_PATHS['base']
    runtime = str(Path(runtime).expanduser().resolve())
    args.runtime, args.data = runtime, args.data or args.data_root
    if not args.loaded_release:
        from task_update import selected_entry
        entry = selected_entry(runtime)
        if entry:
            return subprocess.call([sys.executable, '-B', str(entry), *(sys.argv[1:] if argv is None else argv), '--loaded-release'])
    saved = read_optional(Path(runtime) / 'settings.json') or {}
    if args.status or args.mode == 'status':
        args.query = 'status'
    if args.query:
        kwargs = dict(section=args.result_section, detailed=args.result_detail == 'detailed')
        if args.follow:
            follow(args.query, runtime, args.job, **kwargs)
        else:
            print(query(args.query, runtime, args.job, **kwargs))
        return 0
    if args.stop:
        target = args.job or jobs.latest(runtime)
        if not target:
            raise ValueError('未找到最近任务')
        jobs.stop(target); return 0
    if args.update:
        from task_update import update
        update(runtime); return 0
    if args.resume_job or args.mode == 'resume' or args.append_tasks:
        target = args.resume_job or args.job or 'latest'
        target = jobs.latest(runtime) if target == 'latest' else target
        if not target:
            raise ValueError('没有可续跑任务')
        if args.append_tasks:
            return jobs.append_tasks(target, args.append_tasks, args.foreground, args.resume_check)
        return jobs.resume(target, args.foreground, args.resume_check)
    if args.rerun_job:
        old = jobs.load_plan(args.rerun_job)
        if read_json(Path(args.rerun_job) / 'state.json')['status'] != 'completed':
            raise ValueError('只能主动重跑已完成任务')
        config = dict(old, input_job=args.rerun_job, tasks=jobs.parse_tasks(args.tasks or old['tasks']),
                      name=args.name or old['name']+' 重跑', preprocess_cache='new')
        return jobs.submit(config, args.foreground)
    if args.mode:
        args.tasks = {'check':'environment', 'smoke':'environment,prepare,smoke',
                      'train':'environment,prepare,train', 'experiment':'environment,prepare,smoke,train,predict',
                      'evaluate':'predict'}[args.mode]
        args.scope = args.scope or ('all' if args.mode in {'experiment','evaluate'} else 'ct-first')
    if args.tasks:
        return jobs.submit(new_config(args, saved, interactive=not args.non_interactive), args.foreground)
    if args.non_interactive:
        raise ValueError('非交互运行请指定 --tasks 或 --query')
    return panel(args, runtime, saved)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Quit, Back, KeyboardInterrupt):
        print('已退出面板，后台任务继续。')
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print('操作未完成：' + jobs.clean(error), file=sys.stderr)
        raise SystemExit(1)
