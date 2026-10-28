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

PANEL_VERSION = '0.6.0'


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
    modern=os.environ.get('VISTA_CATALOG_VERSION')=='2'
    if allow_none:
        if modern:
            from pipeline_catalog import resolve
            checked=[]
            for row in records:
                try:resolve(runtime,row['directory'])
                except (ValueError,OSError):continue
                checked.append(row)
            records=checked
        records = [r for r in records if (Path(r.get('workspace', r['directory']))/'04_data/manifests/case_inventory.json').is_file()]
        print('这里只列出已有数据清单的输入任务；仅安装环境的任务无需选作缓存。')
    for i, row in enumerate(records, 1):
        print(f"{i}. {jobs.clean(row['name'])} [{STATES.get(row['status'], row['status'])}] {row['directory']}")
    if modern and not records:raise ValueError('新版尚无可选择的任务；请先在预处理、训练或完整测评板块创建。')
    text = '选择任务编号' if modern else '选择历史编号；d手工填写旧任务/工作区'
    if allow_none:
        text += '；0不使用历史输入'
    value = ask(text, '0' if allow_none else '1' if records else 'd')
    if allow_none and value == '0':
        return None
    if value == 'd' and not modern:
        return ask('完整路径')
    if not str(value).isdigit() or not 1 <= int(value) <= len(records):
        raise ValueError('历史编号无效')
    job=records[int(value)-1]['directory']
    if modern:
        from pipeline_catalog import resolve
        job=resolve(runtime,job)
    return str(job)


def choose_stage_source(runtime,stage):
    from pipeline_catalog import history
    rows=[r for r in history(runtime) if r.get('stage')==stage and r.get('status')=='completed']
    from task_inputs import source_compatibility
    usable=[]
    for row in rows:
        reason=source_compatibility(row['directory'])
        if reason:print(jobs.clean(row['name'])+'：不可作为新版来源；'+reason)
        else:usable.append(row)
    rows=usable
    if not rows:
        raise ValueError('没有已完成的新版'+{'prepare':'预处理','train':'训练'}[stage]+'任务；请先完成对应板块。')
    for i,row in enumerate(rows,1):
        print(f"{i}. {jobs.clean(row['name'])} / {row['directory']}")
    value=ask('选择来源'+{'prepare':'预处理','train':'训练'}[stage]+'任务','1')
    if not str(value).isdigit() or not 1<=int(value)<=len(rows):raise ValueError('来源任务编号无效')
    return rows[int(value)-1]['directory']


def stage_config(args,saved,stage,interactive=True):
    """Independent boards: raw->prepare, prepare->train, trained model->predict."""
    import server_profile
    from assessment_policy import automatic_request,PREPROCESSORS
    profile=dict(server_profile.load(getattr(args,'profile',None)))
    profile['runtime']=args.runtime or profile['runtime']
    base_results=getattr(args,'results',None) or saved.get('results_base') or profile['results']
    source=None;upstream=None
    if stage in {'train','predict'}:
        source=getattr(args,'input_job',None) or (choose_stage_source(profile['runtime'],'prepare' if stage=='train' else 'train') if interactive else None)
        if not source:raise ValueError('请指定已完成的来源任务')
        from pipeline_catalog import history
        allowed={r['directory']:r for r in history(profile['runtime'])}
        row=allowed.get(str(Path(source).resolve()))
        needed='prepare' if stage=='train' else 'train'
        if not row or row['stage']!=needed or row['status']!='completed':raise ValueError('来源必须为新版已完成的'+needed+'任务')
        from pipeline_catalog import resolve
        source=resolve(profile['runtime'],source)
        upstream=jobs.load_plan(source)
        source=str(source)
        data=upstream['data'];preprocessor=upstream['preprocessor']
        profile['gpu_memory']=upstream['gpu_memory']
        print('来源数据：'+data+'；预处理：'+PREPROCESSORS.get(preprocessor,preprocessor))
    else:
        data=getattr(args,'data',None) or saved.get('data') or profile['data']
        if interactive:data=ask('来源数据目录',data)
        preprocessor=getattr(args,'preprocessor',None) or 'nnunet_default'
        if len(PREPROCESSORS)>1 and interactive:
            print('预处理：'+'；'.join(f'{key} {label}' for key,label in PREPROCESSORS.items()))
            preprocessor=ask('预处理方法',preprocessor)
        else:print('预处理：'+PREPROCESSORS.get(preprocessor,preprocessor)+'（自动）')
    methods=getattr(args,'models',None) or ','.join((upstream or {}).get('methods',['B0']))
    if stage in {'train','assessment'} and interactive:
        print('模型：'+'；'.join(f'{key} {value["label"]}' for key,value in MODELS.items()))
        methods=ask('训练模型' if stage=='train' else '测评模型',methods)
    if stage=='predict':
        allowed_models=upstream['methods']
        if len(allowed_models)>1 and interactive:methods=ask('选择已训练模型（'+','.join(allowed_models)+'）',methods)
        chosen=parse_models(methods)
        if not set(chosen)<=set(allowed_models):raise ValueError('所选模型不属于来源训练任务')
        methods=','.join(chosen)
    config=automatic_request(profile,data,methods,preprocessor=preprocessor,
        kind='assessment' if stage=='assessment' else 'formal',blocks=stage if stage!='assessment' else 'all')
    config.update(catalog_schema=2,stage=stage,tasks=[stage],input_job=source,
        formal_training=stage=='train',
        upstream_jobs=[str(Path(source).resolve())] if source else [],results_base=str(Path(base_results).expanduser().resolve()),
        results=str(Path(base_results).expanduser().resolve()/'pipeline_v2'/stage),
        name={'prepare':'预处理','train':'训练','predict':'预测','assessment':'完整测评'}[stage]+' '+Path(data).name+(' '+','.join(config['methods']) if stage!='prepare' else ''))
    if upstream:config['folds']=upstream['folds'][:]
    return jobs.validate_request(config)


def restart_job(args,runtime,job):
    if os.environ.get('VISTA_CATALOG_VERSION')=='2':
        from pipeline_catalog import resolve
        job=resolve(runtime,job)
    plan=jobs.load_plan(job)
    print(query('status',runtime,job))
    active=jobs.active_processes(job)
    completed=(read_json(Path(job)/'state.json').get('status')=='completed')
    print('将新建同板块独立任务，保留本次结果。' if completed else '将从最近完整保存位置继续；未提交的当前轮可能重算。')
    if ask('重启以上任务，输入 RESTART；其它输入返回','')!='RESTART':return
    if active:
        jobs.stop(job);deadline=time.monotonic()+60
        while jobs.active_processes(job) and time.monotonic()<deadline:time.sleep(1)
        if jobs.active_processes(job):raise ValueError('进程尚未退出，未启动第二份任务；可在进程菜单查看。')
    if completed:
        config=dict(plan,name=plan['name']+' 重启')
        if plan['stage']=='prepare':config.update(input_job=None,upstream_jobs=[],preprocess_cache='new')
        return jobs.submit(config,args.foreground)
    return jobs.resume(job,args.foreground,check='simple')


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
        raw_count=value(args.gpu_count,'gpu_count',profile['gpu_count'],'最多同时使用几张 GPU（正整数；不超过实际服务器和分配范围）')
        from training_policy import validate_gpu_count
        count=validate_gpu_count(raw_count)
        if count is None:
            count=len(resolve_gpus('auto'))
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


def simple_config(args,saved,*,kind='formal',interactive=True):
    """Primary choices only. Actual GPU allocation is deferred until execution."""
    import server_profile
    from assessment_policy import automatic_request,PREPROCESSORS
    profile=dict(server_profile.load(getattr(args,'profile',None)))
    for key in ['runtime','results']:
        profile[key]=getattr(args,key,None) or saved.get(key) or profile[key]
    data=getattr(args,'data',None) or saved.get('data') or profile['data']
    if interactive:data=ask('数据目录',data)
    methods=getattr(args,'models',None) or ','.join(saved.get('methods',['B0']))
    if interactive:
        print('模型：'+'；'.join(f'{key} {value["label"]}' for key,value in MODELS.items()))
        methods=ask('模型',methods)
    preprocessing=getattr(args,'preprocessor',None) or 'nnunet_default'
    if len(PREPROCESSORS)>1 and interactive:
        print('预处理：'+'；'.join(f'{key} {label}' for key,label in PREPROCESSORS.items()))
        preprocessing=ask('预处理方法',preprocessing)
    else:print('预处理：'+PREPROCESSORS.get(preprocessing,preprocessing)+'（自动）')
    blocks=getattr(args,'tasks',None) or 'all'
    if kind=='formal' and interactive:
        blocks=ask('正式任务：1预处理 2训练 3预测；可多选','1,2,3')
    config=automatic_request(profile,data,methods,preprocessor=preprocessing,kind=kind,blocks=blocks)
    if kind=='formal':
        config['selected_blocks']=config['tasks'][:]
        # Select the newest compatible formal input, never an assessment model.
        source=None
        for row in jobs.history(config['runtime']):
            if row.get('status') in {'running','queued','waiting_gpu'}:continue
            path=Path(row['directory'])
            try:
                old=jobs.load_plan(path)
                state=read_json(path/'state.json')
                if (old.get('job_kind','formal')!='formal' or Path(old['data']).resolve()!=Path(config['data'])
                        or old.get('methods')!=config['methods'] or old.get('folds')!=config['folds']
                        or old.get('gpu_memory',20)!=config['gpu_memory']):continue
                needed='train' if 'predict' in config['tasks'] and 'train' not in config['tasks'] else 'prepare'
                if state.get('steps',{}).get(needed,{}).get('status')=='completed':
                    source=str(path);break
            except (OSError,ValueError,KeyError):continue
        config['input_job']=source
        if not source and 'train' in config['tasks'] and 'prepare' not in config['tasks']:
            config['tasks'].insert(0,'prepare')
            config['automatic_prerequisites']=['prepare']
        if not source and 'predict' in config['tasks'] and 'train' not in config['tasks']:
            raise ValueError('没有可用的已训练模型；请先选择训练，或从历史任务选择已有模型')
    return jobs.validate_request(config)


def preview_config(config):
    print('\n提交预览\n任务名称：' + jobs.clean(config['name']))
    print('执行顺序：' + ' → '.join(['独立环境检查', *(jobs.TASKS[t] for t in config['tasks'] if t != 'environment')]))
    print('运行环境：' + config['runtime'] + '\n结果根目录：' + config['results'] + '（自动新建唯一目录）')
    if config['tasks'] != ['environment']:
        print('原始数据：' + config['data'])
        print('模型：' + ', '.join(config['methods']) + '；方向：' + ', '.join(config['folds']))
        print('数据模式：全部来源监督训练，独立留出目标中心；默认ct_holdG为30训练 / 8验证 / 20测试。')
        print('采样：nnU-Net原生随机病例/前景取块，不增加每轮逐例必到机制。')
    if config['tasks']==['prepare']:
        print('本次仅CPU预处理；显存预算只用于网络patch/batch规划。')
    elif config.get('gpus'):
        print('获准 GPU：' + ', '.join(config['gpus']) + f"；并发上限：{config['gpu_count']}")
    elif config.get('gpu_policy')=='automatic':
        print('GPU/CPU：自动探测、现场容量配置；GPU上限按实际服务器和分配范围，不需要填写卡号。')
        if set(config['tasks'])&{'train','assessment'}:
            print('batch允许自动调整；正式250轮、每轮250训练/50验证保持，实际采样量将明确记录。')
        elif config['tasks']==['predict']:
            print('仅使用所选训练任务的冻结模型预测，按主机容量安排病例并发。')
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
        sources = log_sources(target,runtime)
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
            elif action == 'z':
                from diagnostic_export import export
                path=export(choose_job(runtime),Path(runtime)/'diagnostic-exports')
                print('诊断包已保存：'+str(path))
            elif action == 'o': result_menu(job,runtime)
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


def result_menu(job,runtime=None):
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
            lines = result_lines(job, section, detailed,runtime)
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
                    lines, page = result_lines(job, section, detailed,runtime), 0
                elif action in {'n','p'}:
                    page += 1 if action == 'n' else -1
        except (Back, KeyboardInterrupt):
            return


def panel(args, runtime, saved):
    import server_profile
    profile=server_profile.load(getattr(args,'profile',None))
    while True:
        rows=jobs.history(runtime)
        active = any(row['status']=='running' for row in rows)
        print(f'\nVISTA3D / nnU-Net 服务器任务面板 v{PANEL_VERSION}\n运行目录：' + runtime + '\n' + query('status', runtime))
        server_profile.show(profile,dict(runtime=runtime,results=args.results or saved.get('results_base') or profile['results'],data=args.data or saved.get('data') or profile['data']))
        print('\ns 任务概览  t 持续进度  g 显卡(1秒刷新)  m 模型 epoch/loss\nl 选择详细日志  p 本任务/我的进程  h 任务列表  o 查看结果')
        print('1 预处理（来源数据/方法）  2 训练（选择预处理任务/模型）  3 预测（选择训练模型）')
        print('a 完整测评（独立短流程）  r 重启任务  d 删除任务  x 中断任务（保留完整检查点）')
        print('e 独立环境准备/检查（每次计算任务前也会自动执行）')
        print('u 更新程序  c GitHub网络设置  w 上传已提交代码  v 上传日志/指标/权重（SSH+Git LFS）')
        print('z 本地诊断导出（选择任务 → ZIP → 显示路径）')
        print('0/q 退出面板（后台任务继续）；普通输入页 b 返回、q 退出。')
        try:
            action = ask('选择；回车只看状态', 's')
            if action == '0':
                return 0
            if active and action in {'a','1','2','3','e'}:
                print('新版当前仍有活动任务；可查看进度、重启或中断，退出后再提交计算。');continue
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
            elif action == 'z':
                from diagnostic_export import export
                path=export(choose_job(runtime),Path(runtime)/'diagnostic-exports')
                print('诊断包已保存：'+str(path))
            elif action == 'o':
                result_menu(choose_job(runtime),runtime)
            elif action in {'a','1','2','3'}:
                config = stage_config(args,saved,{'a':'assessment','1':'prepare','2':'train','3':'predict'}[action])
                preview_config(config)
                if ask('提交以上任务？输入 RUN；其它输入返回', '') == 'RUN':
                    jobs.submit(config, args.foreground)
                    saved = config
            elif action == 'r':
                job = choose_job(runtime)
                restart_job(args,runtime,job)
            elif action == 'd':
                if active:raise ValueError('当前有新版计算任务运行；待退出后再删除，避免目录与启动操作交错。')
                job = choose_job(runtime)
                from pipeline_catalog import dependents,trash
                dependencies=dependents(runtime,job)
                if jobs.active_processes(job):raise ValueError('任务仍在运行，请先中断并等所有子进程退出。')
                if dependencies:
                    raise ValueError('下游任务仍依赖此来源，请先处理下游：'+'；'.join(r['name'] for r in dependencies))
                print('将从新版任务列表移除，并移入可恢复回收区（暂不释放磁盘空间）：'+str(job))
                if ask('删除以上任务，输入 DELETE；其它输入返回','')=='DELETE':
                    destination=trash(runtime,job)
                    print('已移入回收区：'+str(destination))
            elif action == 'x':
                job = choose_job(runtime)
                print(query('status', runtime, job))
                if ask('仅中断上方任务，保留最后完整检查点；输入 STOP', '') == 'STOP':
                    jobs.stop(job)
            elif action == 'u':
                from task_update import update
                if active:
                    targets=[r for r in jobs.history(runtime) if r['status']=='running']
                    if len(targets)!=1:
                        raise ValueError('存在多个活动任务记录，请先用x逐一确认停止目标')
                    target=targets[0]['directory']
                    if os.environ.get('VISTA_CATALOG_VERSION')=='2':
                        from pipeline_catalog import resolve
                        target=resolve(runtime,target)
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
    p.add_argument('--preprocessor',choices=['nnunet_default'])
    p.add_argument('--assessment',action='store_true',help='独立少量真实数据完整测评；资源自动选择')
    p.add_argument('--stage',choices=['prepare','train','predict'],help='独立板块；训练/预测使用 --input-job 选择新版来源')
    p.add_argument('--legacy-jobs',action='store_true',help=argparse.SUPPRESS)
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


def _main(argv=None):
    args = make_parser().parse_args(argv)
    os.environ['VISTA_CATALOG_VERSION']='1' if args.legacy_jobs else '2'
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
    saved = read_optional(Path(runtime) / ('settings.json' if args.legacy_jobs else 'settings_v2.json')) or {}
    if not args.legacy_jobs:
        if args.append_tasks:raise ValueError('新版不追加旧步骤；请使用独立预处理/训练/预测板块，或重启已有任务。')
        # Explicit CLI paths also belong to the new catalog. Do not probe an old
        # task merely because it was supplied as a convenient default/path.
        targets=[args.job,args.upload_job if args.upload_job!='latest' else None,args.rerun_job,
                 args.resume_job if args.resume_job not in {None,'latest'} else None]
        if any(targets):
            from pipeline_catalog import resolve
            for target in targets:
                if target:resolve(runtime,Path(target).expanduser().absolute())
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
    if args.assessment:
        config=stage_config(args,saved,'assessment',interactive=not args.non_interactive)
        preview_config(config)
        if not args.non_interactive and ask('输入 RUN 开始独立短测，其它输入返回','')!='RUN':return 0
        return jobs.submit(config,args.foreground)
    if args.stage:
        config=stage_config(args,saved,args.stage,interactive=not args.non_interactive)
        preview_config(config)
        if not args.non_interactive and ask('提交以上独立板块？输入 RUN','')!='RUN':return 0
        return jobs.submit(config,args.foreground)
    if args.tasks:
        if not args.legacy_jobs:raise ValueError('新版使用 --stage prepare/train/predict；测评使用 --assessment。')
        return jobs.submit(new_config(args, saved, interactive=not args.non_interactive), args.foreground)
    if args.non_interactive:
        raise ValueError('非交互运行请指定 --tasks 或 --query')
    return panel(args, runtime, saved)


def main(argv=None):
    previous=os.environ.get('VISTA_CATALOG_VERSION')
    try:return _main(argv)
    finally:
        if previous is None:os.environ.pop('VISTA_CATALOG_VERSION',None)
        else:os.environ['VISTA_CATALOG_VERSION']=previous


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Quit, Back, KeyboardInterrupt):
        print('已退出面板，后台任务继续。')
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print('操作未完成：' + jobs.clean(error), file=sys.stderr)
        raise SystemExit(1)
