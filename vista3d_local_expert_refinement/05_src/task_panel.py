"""Whole-heart task panel: stdlib control plane; compute starts only on submission."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from model_catalog import MODELS, FOLDS, SCOPES, parse_models, parse_folds
from server_data import read_json
import task_runtime as jobs
from task_queries import query, follow, result_lines, read_optional, log_sources
from task_progress import STATES

PANEL_VERSION = '0.3.9'


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
    if allow_none:
        records = [r for r in records if (Path(r.get('workspace', r['directory']))/'04_data/manifests/case_inventory.json').is_file()]
        print('这里只列出已有数据清单的输入任务；仅安装环境的任务无需选作缓存。')
    for i, row in enumerate(records, 1):
        print(f"{i}. {jobs.clean(row['name'])} [{STATES.get(row['status'], row['status'])}] {row['directory']}")
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
    if tokens == ['auto']:
        result = [r['uuid'] for r in rows if r['uuid'] in visible]
        if not result:
            raise ValueError('获准可见范围内没有 GPU；请检查调度分配和 CUDA_VISIBLE_DEVICES')
        return result
    result = []
    for token in tokens:
        matched = [r['uuid'] for r in rows if token in [r['index'], r['uuid']] and r['uuid'] in visible]
        if len(matched) != 1 or matched[0] in result:
            raise ValueError('GPU未知、重复或超出分配范围：' + token)
        result.extend(matched)
    return result


def show_gpu_choices():
    from evaluation_gpu import query as gpu_query, allowed_ids
    rows=[]
    processes=gpu_query('gpu_uuid,pid','compute-apps')
    for index,uuid,name,free,util,total in gpu_query('index,uuid,name,memory.free,utilization.gpu,memory.total'):
        rows.append(dict(index=index,uuid=uuid,name=name,free=free,util=util,total=total))
    visible=allowed_ids(rows,{'allowed_gpus':None})
    print('\n当前显卡状态（空闲不代表已预约；执行前再次检查）：')
    print('编号 / 型号 / 总显存MiB / 空闲显存MiB / 利用率 / 计算进程PID / 本次分配可见范围')
    for row in rows:
        pids=','.join(p[1] for p in processes if p[0]==row['uuid']) or '无'
        print(f"{row['index']} / {row['name']} / {row['total']} / {row['free']} / {row['util']}% / {pids} / " + ('范围内' if row['uuid'] in visible else '范围外'))
    print('auto 表示授权当前分配可见范围为候选；不会超过并发上限，不抢占外部进程。')


def new_config(args, saved, interactive=True):
    import server_profile
    profile=server_profile.load(getattr(args,'profile',None))
    def value(argument, key, default, label):
        if argument is not None:
            return argument
        suggestion = saved.get(key, default)
        displayed = ','.join(map(str, suggestion)) if isinstance(suggestion, (list, tuple)) else suggestion
        return ask(label, displayed) if interactive else suggestion
    runtime = str(Path(args.runtime or saved.get('runtime') or profile['runtime']).expanduser().resolve())
    if interactive:
        print('\n新建任务：命名 → 步骤 → 路径 → 输入 → 步骤配置 → 冻结预览。每页 b 返回、q 退出。')
    name=value(args.name,'name','wholeheart nnU-Net','任务显示名称')
    tasks = jobs.parse_tasks(value(args.tasks, 'tasks', 'environment,prepare,smoke', '步骤：0环境与CPU软件检查 1预处理 2完整尺寸预检 3训练 4预测评估 5已有评价汇总；可多选'))
    results=value(args.results,'results',profile['results'],'结果根目录（每次自动生成独立子目录）')
    data=args.data or (ask('原始 Wholeheart 数据目录',saved.get('data') or profile['data']) if interactive and tasks!=['environment'] else saved.get('data') or profile['data'])
    if interactive:
        server_profile.show(profile,dict(runtime=runtime,results=results,data=data))
    source = args.input_job
    if source is None and interactive and tasks != ['environment']:
        print(f'“{jobs.clean(name)}”尚未提交；下面选择旧任务作为输入，不是新任务列表。完成预览并输入 RUN 后才创建。')
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
    if source_config and args.data is None:
        if interactive and str(source_config.get('data',data)) != str(data):
            print('所选历史输入的冻结原始数据：'+str(source_config.get('data')))
        data=source_config.get('data',data)
    methods = parse_models(args.models or (source_config or {}).get('methods'))
    folds = parse_folds(args.folds or args.scope or (source_config or {}).get('folds') or 'ct-first')
    gpus, count = [], 0
    if jobs.GPU_TASKS.intersection(tasks):
        if interactive:
            print('\n计算范围与获准资源（CPU 前置完成后才等待显卡）')
        print('\n模型可多选：' + '；'.join(f'{i} {v["label"]}' for i,v in enumerate(MODELS.values(), 1)))
        if interactive and args.models is None:
            methods = parse_models(ask('模型编号或名称，逗号分隔；all全部', ','.join(saved.get('methods', list(MODELS)))))
        if interactive and args.folds is None and args.scope is None:
            folds = parse_folds(ask('范围：ct-first / all，或逗号分隔方向', 'ct-first'))
        if interactive:
            show_gpu_choices()
        raw_count=value(args.gpu_count,'gpu_count',profile['gpu_count'],'最多同时使用几张 GPU（1—5；不足时少卡运行，无卡时等待）')
        if isinstance(raw_count,bool) or str(raw_count).strip() not in {'1','2','3','4','5'}:
            raise ValueError('显卡并发上限需为1至5的整数')
        count=int(raw_count)
        selection=value(args.gpus,'gpu_candidates',profile['gpu_candidates'],'允许使用的候选 GPU 编号/UUID，逗号分隔，或 auto（卡数可少于/多于上限）')
        gpus = resolve_gpus(selection)
    elif interactive:
        print('\n本次仅 CPU 步骤，无需设置 GPU。')
    from gpu_budget import DEFAULT_MEMORY_GIB,MEMORY_POLICY
    planning=any(t in tasks for t in ['prepare','smoke','train'])
    if args.gpu_memory is not None:
        memory=float(args.gpu_memory)
    elif source_config and not planning:
        memory=float(source_config.get('gpu_memory',DEFAULT_MEMORY_GIB))
    elif saved.get('gpu_memory_policy')==MEMORY_POLICY:
        memory=float(saved.get('gpu_memory',profile.get('gpu_memory',DEFAULT_MEMORY_GIB)))
    else:
        memory=float(profile.get('gpu_memory',DEFAULT_MEMORY_GIB))
        if saved.get('gpu_memory') and saved['gpu_memory']!=memory:
            print(f'新任务显存默认已按RTX3090设为{memory:g} GiB；旧任务仍保留原计划。')
    config = dict(runtime=runtime,
                  results=results, data=data,name=name,tasks=tasks,profile=profile['name'],conda=profile.get('conda',''),
                  methods=methods, folds=folds, gpus=gpus, gpu_count=count,
                  gpu_memory=memory,gpu_memory_policy=MEMORY_POLICY,
                  input_job=source, preprocess_cache=args.preprocess_cache or 'reuse')
    if gpus:
        config['gpu_candidates']=selection
    if interactive and args.gpu_memory is None and any(t in tasks for t in ['prepare','smoke','train']):
        config['gpu_memory'] = float(ask('网络规划显存预算GiB（24 GiB卡默认20，预留约4 GiB；改预算需新计划）', config['gpu_memory']))
    if source and 'prepare' in tasks and interactive and args.preprocess_cache is None:
        choice = ask('预处理：1尝试沿用（缓存不兼容则新建） 2新建', '1')
        if choice not in {'1','2'}:
            raise ValueError('缓存选择无效')
        config['preprocess_cache'] = 'reuse' if choice == '1' else 'new'
    if source_config and planning and float(source_config.get('gpu_memory',config['gpu_memory']))!=config['gpu_memory']:
        if 'prepare' not in tasks:raise ValueError('历史输入使用另一显存计划；请加入1预处理，为新预算生成独立计划和缓存')
        config['preprocess_cache']='new'
        print('规划显存已改变：独立新建计划和预处理缓存，保留历史任务。')
    return jobs.validate_request(config)


def preview_config(config):
    print('\n提交预览\n任务名称：' + jobs.clean(config['name']))
    print('执行顺序：' + ' → '.join(['独立环境检查', *(jobs.TASKS[t] for t in config['tasks'] if t != 'environment')]))
    print('运行环境：' + config['runtime'] + '\n结果根目录：' + config['results'] + '（自动新建唯一目录）')
    if config['tasks'] != ['environment']:
        print('原始数据：' + config['data'])
        print('模型：' + ', '.join(config['methods']) + '；方向：' + ', '.join(config['folds']))
    if config.get('gpus'):
        print('获准 GPU：' + ', '.join(config['gpus']) + f"；并发上限：{config['gpu_count']}")
    else:
        print('本次仅 CPU 步骤；不会申请显卡。')
    if any(t in config['tasks'] for t in ['prepare','smoke','train']):
        print(f"网络规划显存预算：{config['gpu_memory']:g} GiB（用于生成patch/batch计划，不是显存硬限额）")
    print('提交后在后台执行；退出面板或日志查看不会停止任务。')


def view_menu(kind, runtime, job=None, **kwargs):
    mode = ask('1 单次查看  2 持续刷新（Ctrl+C 返回）  b 返回', '1')
    if mode == '1':
        print(query(kind, runtime, job, **kwargs))
    elif mode == '2':
        follow(kind, runtime, job, **kwargs)
    else:
        print('请输入 1 或 2。')


def log_menu(runtime, job=None):
    target = job or jobs.latest(runtime)
    if not target:
        print('尚无任务日志。先新建环境或计算任务。'); return
    while True:
        sources = log_sources(target)
        print('\n日志中心：' + str(target) + '\n总日志包含环境安装；模型日志包含实际 epoch/loss。')
        for index, (_,label,path) in enumerate(sources):
            print(f'{index}. {label}\n   {path}')
        print('a 全部日志（动态发现新模型）  b 返回  q 退出')
        try:
            choice = ask('选择日志编号', '0' if sources else 'a')
            if choice == 'a':
                key = 'all'
            elif str(choice).isdigit() and 0 <= int(choice) < len(sources):
                key = sources[int(choice)][0]
            else:
                print('日志编号无效，请重选。'); continue
            view_menu('logs', runtime, target, log_source=key)
        except (Back, KeyboardInterrupt):
            return


def job_menu(runtime, job):
    while True:
        print('\n历史任务：' + str(job) + '\ns 概览  m 模型  l 日志  p 进程  o 结果  b/0 返回  q 退出')
        try:
            action = ask('选择', 's')
            if action == '0': return
            if action == 'l': log_menu(runtime, job)
            elif action == 'o': result_menu(job)
            elif action in {'s','m','p'}:
                view_menu({'s':'status','m':'models','p':'processes'}[action], runtime, job)
            else: print('请选择显示的操作。')
        except (Back, KeyboardInterrupt):
            return


def environment_menu(args, runtime, saved):
    print('\n独立计算环境\n目录：' + str(Path(runtime)/'envs/vista_baseline_py311'))
    print('固定 Python 3.11 / PyTorch 2.8.0+cu128 / nnU-Net 2.8.1。\n复用可用 Conda 或校验后安装独立 Miniforge；不修改其它环境。\n不需要数据路径或 GPU，提交后可在 s/t 看安装阶段，在 l 看下载与依赖明细。')
    try:
        action = ask('1 提交环境准备/核验  2 查看最近任务安装日志  b 返回', 'b')
        if action == '2':
            log_menu(runtime); return
        if action != '1':
            print('请选择 1 或 2。'); return
        selected = copy.copy(args)
        selected.tasks, selected.name, selected.input_job = 'environment', args.name or '独立环境准备', None
        config = new_config(selected, saved)
        preview_config(config)
        if ask('输入 RUN 提交环境准备，其它输入取消', '') == 'RUN':
            jobs.submit(config, args.foreground)
    except (Back, KeyboardInterrupt):
        return


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
    import server_profile
    profile=server_profile.load(getattr(args,'profile',None))
    while True:
        active = any(jobs.active_processes(row['directory']) for row in jobs.history(runtime))
        print(f'\nVISTA3D / nnU-Net 服务器任务面板 v{PANEL_VERSION}\n运行目录：' + runtime + '\n' + query('status', runtime))
        server_profile.show(profile,dict(runtime=runtime,results=args.results or saved.get('results') or profile['results'],data=args.data or saved.get('data') or profile['data']))
        print('\ns 任务概览  t 持续进度  g 显卡(1秒刷新)  m 模型 epoch/loss\nl 选择详细日志  p 本任务/我的进程  h 历史任务  o 查看结果')
        print('x 中断指定任务（保留完整检查点）' if active else 'n 新建任务  e 独立环境准备  r 续跑/追加  k 完成任务独立重跑')
        print('u 更新程序  c GitHub网络设置  w 上传已提交代码  v 上传日志/指标/权重（SSH+Git LFS）')
        print('0/q 退出面板（后台任务继续）；普通输入页 b 返回、q 退出。')
        try:
            action = ask('选择；回车只看状态', 's')
            if action == '0':
                return 0
            if active and action in {'n','e','r','k'}:
                print('此运行目录仍有活动任务。先查看进程或明确中断并等它退出，再提交任务。'); continue
            if action in {'s','g','m','l','p','h','t'}:
                kind = {'s':'status','g':'gpu','m':'models','l':'logs','p':'processes','h':'history','t':'status'}[action]
                if action == 'g':
                    follow(kind, runtime)
                elif action == 't':
                    follow(kind, runtime)
                elif action == 'l':
                    log_menu(runtime)
                elif action == 'h':
                    job_menu(runtime, choose_job(runtime))
                elif action == 'm':
                    view_menu(kind, runtime)
                elif action == 'p':
                    scope = ask('1 本任务进程树  2 本用户全部进程  b 返回', '1')
                    if scope not in {'1','2'}:
                        print('请输入 1 或 2。'); continue
                    view_menu(kind, runtime, process_scope='task' if scope == '1' else 'all')
                else:
                    print(query(kind, runtime))
            elif action == 'e':
                environment_menu(args, runtime, saved)
            elif action == 'c':
                from panel_publication.network import configure_interactive
                configure_interactive(runtime,profile.get('network'))
            elif action == 'w':
                from panel_publication.code_sync import run as push_code
                selected=copy.copy(args);selected.yes_upload=False
                push_code(selected,profile,True)
            elif action == 'v':
                from panel_publication.sharing import run_upload
                selected=copy.copy(args)
                selected.upload_job=choose_job(runtime);selected.upload_package=None;selected.yes_upload=False
                run_upload(selected,profile,True)
            elif action == 'o':
                result_menu(choose_job(runtime))
            elif action == 'n':
                config = new_config(args, saved)
                preview_config(config)
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
                config['preprocess_cache']='new' if 'prepare' in config['tasks'] else 'reuse'
                if ask('独立重跑并保留原结果？输入 RUN', '') == 'RUN':
                    jobs.submit(config, args.foreground)
            elif action == 'x':
                job = choose_job(runtime)
                print(query('status', runtime, job))
                if ask('仅中断上方任务，保留最后完整检查点；输入 STOP', '') == 'STOP':
                    jobs.stop(job)
            elif action == 'u':
                from task_update import update
                if active:
                    targets=[r for r in jobs.history(runtime) if jobs.active_processes(r['directory'])]
                    if len(targets)!=1:
                        raise ValueError('存在多个活动任务记录，请先用x逐一确认停止目标')
                    target=targets[0]['directory']
                    print(query('status',runtime,target))
                    if ask('中断上方任务、保留检查点并等待退出后更新？[y/N]','n').lower()!='y':
                        continue
                    jobs.stop(target)
                    deadline=time.monotonic()+60
                    while jobs.active_processes(target) and time.monotonic()<deadline:
                        time.sleep(1)
                    if jobs.active_processes(target):
                        raise ValueError('子进程60秒内仍未退出；已停止更新，锁和检查点保留')
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
    p.add_argument('--profile',choices=['zmic44','generic'],help='Linux默认zmic44已记录路径；原runtime与CLI优先')
    p.add_argument('--results', '--work-dir', dest='results')
    p.add_argument('--data-dir', dest='data')
    p.add_argument('--name')
    p.add_argument('--tasks', help='environment,prepare,smoke,train,predict,summarize，多选')
    p.add_argument('--models', '--methods', dest='models', help='B0；当前仅监督基线')
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
    p.add_argument('--log-source', default='all', help='all全部 / 0总日志 / step:train / model:模型目录名')
    p.add_argument('--process-scope', choices=['task','all'], default='task', help='本任务进程树或本用户全部进程')
    p.add_argument('--result-section', choices=['summary','cases','training','files'], default='summary')
    p.add_argument('--result-detail', choices=['summary','detailed'], default='summary')
    p.add_argument('--resume-job', nargs='?', const='latest')
    p.add_argument('--resume-check', choices=['simple','full','none'], default='simple')
    p.add_argument('--append-tasks')
    p.add_argument('--rerun-job')
    p.add_argument('--stop', action='store_true')
    p.add_argument('--update', action='store_true')
    p.add_argument('--configure-network',action='store_true')
    p.add_argument('--show-network',action='store_true')
    p.add_argument('--github-proxy')
    p.add_argument('--github-transport',choices=['github-ssh','github-https'])
    p.add_argument('--github-ssh-key')
    p.add_argument('--push-code',action='store_true',help='预览/上传已提交的本项目代码到main')
    p.add_argument('--upload-job',help='预览/上传已完成任务的日志、配置、指标和权重')
    p.add_argument('--upload-package',help='按原不可变包重试SSH+LFS发布')
    p.add_argument('--yes-upload',action='store_true',help='明确确认当前预览范围的上传')
    p.add_argument('--mode', choices=['check','smoke','train','resume','experiment','evaluate','status'])
    p.add_argument('--server-defaults', action='store_true')
    p.add_argument('--loaded-release', action='store_true', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = make_parser().parse_args(argv)
    from runtime_setup import DEFAULT_PATHS, SETTINGS
    import server_profile
    profile=server_profile.load(args.profile)
    legacy = read_optional(SETTINGS) or {}
    old_runtime=Path(DEFAULT_PATHS['base'])
    compatible_default=str(old_runtime) if any((old_runtime/p).exists() for p in ['settings.json','last-job.json','release.json']) else profile['runtime']
    runtime = args.runtime or (None if args.server_defaults else legacy.get('base')) or compatible_default
    runtime = str(Path(runtime).expanduser().resolve())
    args.runtime, args.data = runtime, args.data or args.data_root
    if not args.loaded_release:
        from task_update import selected_entry
        entry = selected_entry(runtime)
        if entry:
            return subprocess.call([sys.executable, '-B', str(entry), *(sys.argv[1:] if argv is None else argv),
                                    '--runtime', runtime, '--loaded-release'])
    saved = read_optional(Path(runtime) / 'settings.json') or {}
    if args.configure_network or args.show_network:
        from panel_publication import network
        if args.show_network:
            network.show(network.load(runtime));return 0
        if any(v is not None for v in [args.github_proxy,args.github_transport,args.github_ssh_key]):
            network.show(network.save(runtime,proxy=args.github_proxy,git_transport=args.github_transport,ssh_key=args.github_ssh_key))
        else:
            network.configure_interactive(runtime,profile.get('network'))
        return 0
    if args.push_code:
        from panel_publication.code_sync import run
        return run(args,profile,not args.non_interactive)
    if args.upload_job or args.upload_package:
        from panel_publication.sharing import run_upload
        return run_upload(args,profile,not args.non_interactive)
    if args.status or args.mode == 'status':
        args.query = 'status'
    if args.query:
        kwargs = dict(section=args.result_section, detailed=args.result_detail == 'detailed',
                      log_source=args.log_source, process_scope=args.process_scope)
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
        config['preprocess_cache']='new' if 'prepare' in config['tasks'] else 'reuse'
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
