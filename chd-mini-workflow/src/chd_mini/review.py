import ast
import json
import os
from pathlib import Path
import shutil
import traceback
from time import perf_counter

from .attempts import AttemptLedger
from .contracts import TOOL_ROOT, confined, git, read_json, write_json
from .runner import inventory
from .version_review import full_commit, model_versions
from .geometry_review import source_geometry, verify_output


ROLES = dict(train=['ct_1001'], validation=['ct_1012'], test=[])
MODELS = ('stu', 'q', 'med')


def review_models(handoff):
    mode = handoff.get('review_mode', 'complete')
    available = handoff.get('available_models', list(MODELS) if mode == 'complete' else None)
    unavailable = handoff.get('unavailable_models', {})
    if mode not in ('complete', 'partial') or not isinstance(available, list) or not available:
        raise ValueError('须明确完整或部分验收模式及非空可用模型列表')
    if (any(not isinstance(m,str) for m in available) or len(set(available)) != len(available) or
            not set(available) <= set(MODELS) or not isinstance(unavailable,dict)):
        raise ValueError('模型声明无效')
    if (set(handoff['models']) != set(available) or set(available) & set(unavailable) or
            set(available) | set(unavailable) != set(MODELS)):
        raise ValueError('每个目标模型必须明确可用或不可用，且与实际交接模型一致')
    if mode == 'complete' and (set(available) != set(MODELS) or unavailable):
        raise ValueError('完整验收需要三模型全部交接')
    if mode == 'partial' and (not unavailable or not set(available) <= {'stu','q'}):
        raise ValueError('本轮部分验收仅允许明确交接的STU/Q，Med缺项必须登记')
    if any(not isinstance(reason,str) or not reason.strip() for reason in unavailable.values()):
        raise ValueError('不可用模型必须有具体原因')
    return [m for m in MODELS if m in available]


def flags(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    return {arg.value for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
            for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}


def selected_best(item):
    record_path = Path(item['best_record']).resolve(strict=True)
    record = read_json(record_path)
    checkpoint = Path(record[item['best_path_field']])
    checkpoint = (record_path.parent / checkpoint).resolve(strict=True)
    if not checkpoint.is_file():
        raise ValueError('best记录没有指向可读模型文件')
    return str(checkpoint)


def prediction_args(model, item, checkpoint, output):
    if model == 'stu':
        return ['predict', '--config', item['config'], '--checkpoint', 'best',
                '--preprocessed-source', item['prepared'], '--output', output]
    args = ['models', 'predict', '--config', item['config'], '--checkpoint', checkpoint,
            '--output', output, '--prepared', item['prepared'], '--device', 'cpu']
    if model == 'med':
        args += ['--prompt-mode', item['prompt_mode']]
        if item['prompt_mode'] == 'external':
            args += ['--prompts', item['prompts']]
        elif item['prompt_mode'] != 'label-box':
            raise ValueError('Med提示模式无效')
    return args


def evaluation_args(item, predictions, output):
    return ['evaluate','--config',item['evaluation_config'],'--prepared',item['prepared'],
            '--predictions',predictions,'--output',output]


def evaluation_step(model, item, predictions, output):
    if model == 'med' and item['prompt_mode'] == 'label-box':
        return dict(name='med_evaluate',model=model,kind='producer_evaluation',
                    output=str(Path(predictions)/'evaluation.json'),
                    producer_summary=str(Path(predictions)/'summary.json'),producer_step='med_resume')
    return dict(name=model+'_evaluate',model=model,kind='evaluate',output=output,
                args=evaluation_args(item,predictions,output))


def producer_evaluation(step, reports, records):
    started = perf_counter()
    producer = next(r for r in records if r['step']['name'] == step['producer_step'])
    summary = read_json(step['producer_summary'])
    evaluation = read_json(step['output'])
    if summary['model'] != 'medsam2' or summary['prompt_mode'] != 'label-box':
        raise ValueError('生产评价未对应Med标签框预测')
    if evaluation != summary['evaluation'] or evaluation != producer['production_summary']['evaluation']:
        raise ValueError('生产评价文件与实际预测Job的summary不符')
    if (evaluation['metric'] != 'prompted_known_present_class_dice' or evaluation['scope'] != 'technical'
            or set(evaluation['case_classes']) != {'ct_1001','ct_1012'}):
        raise ValueError('Med评价的指标、范围或病例不符')
    if not producer['process']['all_workers_exited']:
        raise ValueError('Med生产评价的预测Job尚未确认退出')
    receipt = dict(step=step,evaluation=evaluation,source_evaluation_file=step['output'],
        source_prediction_summary=step['producer_summary'],
        source_prediction_process=str(reports/(step['producer_step']+'.process.json')),
        prediction_job=producer['process'],read_seconds=perf_counter()-started,
        evaluation_compute_seconds=None,
        scope='参考标签框提示下已知且存在类别的Dice；不证明自动定位或全类别效果',
        timing_note='评价由生产预测Job生成，未独立测量评价计算耗时，不重复相加')
    write_json(reports/(step['name']+'.json'),receipt)
    return receipt


def build_review(handoff):
    available = review_models(handoff)
    if handoff['roles'] != ROLES:
        raise ValueError('交接必须是ct_1001/ct_1012技术角色')
    if set(handoff['source_images']) != {'ct_1001', 'ct_1012'}:
        raise ValueError('几何来源必须恰好对应两例')
    if Path(handoff['work_root']).resolve() != Path('D:/projects/back/t17/chd-review-20261118').resolve():
        raise ValueError('工作根不符合本轮共享许可')
    if Path(handoff['report_root']).resolve() != Path('D:/projects/back/sync-reports/chd-review-20261118').resolve():
        raise ValueError('报告根不符合本轮共享许可')
    engine = Path(handoff['engine_root']).resolve(strict=True)
    require_clean_source(engine)
    actual = git(engine, 'rev-parse', 'HEAD')
    if actual != full_commit(handoff['engine_commit']):
        raise ValueError('生产目录与最终交接版本不对应')
    ledger = AttemptLedger(handoff['attempt_ledger']).read()
    if any(r['outcome'] == 'started' for r in ledger['attempts']):
        raise ValueError('阶段2仍有未结算尝试，不能开始阶段3')
    steps, versions = [], {}
    for model in available:
        item = handoff['models'][model]
        versions[model] = model_versions(engine, model, item, actual)
        checkpoint = selected_best(item)
        cli = engine / 'src/chd_segmentation' / ('cli.py' if model == 'stu' else 'models_cli.py')
        if '--checkpoint' not in flags(cli):
            raise ValueError(model+'生产CLI尚无显式checkpoint入口，等待助手2交接')
        output = str(Path(handoff['work_root']) / 'assistant3' / model)
        args = prediction_args(model, item, checkpoint, output)
        for label, cases in (('first', ['ct_1001']), ('resume', ['ct_1001', 'ct_1012'])):
            requested = [arg for name in cases for arg in ('--case', name)]
            steps.append(dict(name=model+'_'+label, model=model, output=output,
                              kind='predict',checkpoint=checkpoint, args=args+requested))
        evaluation = str(Path(handoff['work_root'])/'assistant3'/(model+'-evaluation.json'))
        steps.append(evaluation_step(model,item,output,evaluation))
    return dict(steps=steps, ledger=ledger, engine_commit=actual, model_versions=versions,
                review_mode=handoff.get('review_mode','complete'), target_models=list(MODELS),
                available_models=available, unavailable_models=handoff.get('unavailable_models',{}),
                scope='zero_additional_training', roles=ROLES)


def require_clean_source(engine):
    status = git(engine,'status','--porcelain','--untracked-files=all','--',
                 'src','run.py','pyproject.toml')
    if status:
        raise ValueError('生产业务源码存在未提交修改：'+status)
    return status


def geometry(handoff, model, output):
    result = []
    produced = {r['case_id']:r for r in read_json(Path(output)/'summary.json')['cases']}
    for case, source in handoff['source_images'].items():
        metadata = source_geometry(handoff['models'][model],case)
        if produced[case]['roi_xyz'] != metadata['roi_xyz']:
            raise ValueError('实际预测ROI与读取来源记录不符：'+case)
        record = verify_output(source,Path(output)/(case+'.nii.gz'),metadata)
        result.append(dict(case_id=case,**record))
    return result


def prediction_events(summary):
    path = Path(summary['event_log'])
    start, end = summary['start_offset'], summary['end_offset']
    if not path.is_absolute() or type(start) is not int or type(end) is not int:
        raise ValueError('生产事件必须给出绝对路径及整数字节区间')
    if not 0 <= start < end <= path.stat().st_size or end-start > 512*1024:
        raise ValueError('生产事件字节区间无效、为空或超过本步保留额度')
    with path.open('rb') as stream:
        if start:
            stream.seek(start-1)
            if stream.read(1) != b'\n':
                raise ValueError('事件起始偏移不在JSONL行边界')
        stream.seek(start)
        payload = stream.read(end-start)
    if len(payload) != end-start or not payload.endswith(b'\n'):
        raise ValueError('事件区间已截断或末尾记录未提交完整')
    events = []
    position = start
    for line in payload.splitlines(keepends=True):
        next_position = position+len(line)
        if line.strip():
            row = json.loads(line)
            if row.get('event') in ('prediction_reused', 'model_case_reused', 'prediction', 'model_case_completed'):
                if row.get('case_id') in ('ct_1001', 'ct_1012'):
                    events.append(dict(source_event_log=str(path), range_start_offset=start, range_end_offset=end,
                                       record_start_offset=position, record_end_offset=next_position, record=row))
        position = next_position
    if not events:
        raise ValueError('本次生产事件区间缺少真实预测/复用事件')
    return events


def require_prediction_events(step, events):
    records = [item['record'] for item in events]
    completed = {r.get('case_id') for r in records if r.get('event') in ('prediction','model_case_completed')}
    reused = {r.get('case_id') for r in records if r.get('event') in ('prediction_reused','model_case_reused')}
    valid = 'ct_1001' in completed if step['name'].endswith('_first') else (
        'ct_1001' in reused and 'ct_1012' in completed)
    if not valid:
        raise ValueError('本次事件未证明所请求的病例完成或逐例接续，禁止以mtime补齐')


def capacity(handoff):
    rows = inventory(handoff['work_root'])
    reports = inventory(handoff['report_root'])
    own_reports = inventory(Path(handoff['report_root'])/'assistant3')
    return dict(files=len(rows), bytes=sum(r['bytes'] for r in rows),
                report_bytes=sum(r['bytes'] for r in reports),
                own_report_bytes=sum(r['bytes'] for r in own_reports))


def budget(handoff):
    measured = capacity(handoff)
    if (measured['files'] > 400 or measured['bytes'] > 16*1024**3 or
            measured['report_bytes'] > 20*1024**2 or measured['own_report_bytes'] > 5*1024**2):
        raise RuntimeError('共享工作或报告超过批准范围')
    return measured


def optimizer_evidence(handoff, ledger):
    evidence = {}
    for stage in ('stu', 'q_base', 'q_localizer', 'q_local', 'med_sequence'):
        observed, auxiliary = [], []
        for path in handoff.get('training_event_logs', {}).get(stage, []):
            for line in Path(path).read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if stage == 'stu' and row.get('event') == 'train':
                    observed.append(dict(path=str(path), event=row))
                if stage.startswith('q_') and row.get('stage') == stage[2:]:
                    if row.get('event') == 'model_optimizer_completed':
                        observed.append(dict(path=str(path),event=row))
                    if row.get('event') == 'model_update_completed':
                        auxiliary.append(dict(path=str(path),event=row,meaning='保存后更新位置，不能代替总真实更新次数'))
                if stage == 'med_sequence' and row.get('event') in ('model_forward_backward','model_epoch_completed'):
                    auxiliary.append(dict(path=str(path),event=row,
                        meaning='Med计算/epoch记录，单独不足以证明优化器成功更新'))
        successful = sum(r['stage'] == stage and r['outcome'] == 'successful' for r in ledger['attempts'])
        evidence[stage] = dict(ledger_successful=successful, optimizer_update_events=observed, auxiliary_events=auxiliary,
            event_positions=[r['event'].get('updates',r['event'].get('update')) for r in observed],
            total_real_updates=None,
            status='原始已记录位置不等于总真实更新次数；未保存更新及Med事件语义仍需阶段2核对')
    return evidence


def cleanup_review(root, work_root, reports, process_paths):
    receipts = [read_json(path) for path in process_paths if Path(path).is_file()]
    confirmed = len(receipts) == len(process_paths) and all(r['all_workers_exited'] for r in receipts)
    if not confirmed:
        write_json(Path(reports)/'cleanup.json', dict(state='retained_unconfirmed_workers',
            expected_receipts=[str(p) for p in process_paths], work=str(root),
            reason='缺少Job退出证据，须核实后清理；不删除阶段2资产'))
        return
    confined(root, Path(work_root))
    rows = inventory(root)
    write_json(Path(reports)/'cleanup.json', dict(state='pending', files=rows))
    shutil.rmtree(root)
    write_json(Path(reports)/'cleanup.json', dict(state='completed', deleted_files=len(rows),
        released_bytes=sum(r['bytes'] for r in rows), work_removed=not Path(root).exists(),
        shared_assets_removed=False))


class ReviewFinalizer:
    def __init__(self, handoff, plan, root, reports, records, process_paths):
        self.handoff, self.plan, self.root, self.reports = handoff, plan, root, reports
        self.work_root = handoff['work_root']
        self.records, self.process_paths = records, process_paths
        self.started = perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        try:
            summary = dict(state='INCOMPLETE', records=self.records,
                wall_seconds=perf_counter()-self.started,
                error=None if kind is None else ''.join(traceback.format_exception(kind,error,trace)),
                production_commit=self.plan['engine_commit'],
                model_versions=self.plan.get('model_versions', {}),
                target_models=list(MODELS), review_mode=self.plan.get('review_mode','complete'),
                available_models=self.plan.get('available_models',[]),
                unavailable_models=self.plan.get('unavailable_models',{}),
                geometry_scope='两例原图头/ROI外零填充；计算仅ROI，不证明全体积预测效果')
            write_json(self.reports/'review.json', summary)
            # 最终容量只记录，不在此处抛出额度错误阻止清理。
            summary['capacity'] = capacity(self.handoff)
            if 'engine_root' in self.handoff:
                summary['production_status_end'] = require_clean_source(self.handoff['engine_root'])
            current = AttemptLedger(self.handoff['attempt_ledger']).read()
            summary['training_ledger_unchanged'] = current == self.plan['ledger']
            summary['optimizer_evidence'] = optimizer_evidence(self.handoff, self.plan['ledger'])
            summary['tool_commit'] = git(TOOL_ROOT, 'rev-parse', 'HEAD')
            write_json(self.reports/'review.json', summary)
        finally:
            cleanup_review(self.root, self.work_root, self.reports, self.process_paths)
        return False


def execute_review(path):
    from .windows_job import supervise
    handoff = read_json(path)
    if not isinstance(handoff['secretary_stage2_release'], str) or not handoff['secretary_stage2_release'].strip():
        raise ValueError('等待秘书明确阶段2交接')
    plan = build_review(handoff)
    root = Path(handoff['work_root']) / 'assistant3'
    reports = Path(handoff['report_root']) / 'assistant3'
    if root.exists():
        raise FileExistsError('本轮阶段3目录已存在，须核对进度后恢复')
    reports.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True)
    records, before, process_paths = [], {}, []
    with ReviewFinalizer(handoff, plan, root, reports, records, process_paths):
        return _execute_steps(handoff, plan, root, reports, records, before, process_paths, supervise)


def _execute_steps(handoff, plan, root, reports, records, before, process_paths, supervise):
    cache = root/'cache'
    cache.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               PYTHONIOENCODING='utf-8', TEMP=str(cache), TMP=str(cache),
               XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache/'torch'),
               MPLCONFIGDIR=str(cache/'matplotlib'), HF_HOME=str(cache/'huggingface'))
    write_json(reports/'owner.json', dict(work=str(root.resolve()), shared_cleanup_owner=handoff['cleanup_owner']))
    from .simulation import check_resources
    check_resources(dict(engine_root=handoff['engine_root'], engine_commit=plan['engine_commit'],
                             simulator_root=handoff['simulator_root']), root, reports)
    for step in plan['steps']:
        budget(handoff)
        require_clean_source(handoff['engine_root'])
        if step['kind'] == 'producer_evaluation':
            records.append(producer_evaluation(step,reports,records))
            write_json(reports/'execution.json',records)
            if AttemptLedger(handoff['attempt_ledger']).read() != plan['ledger']:
                raise RuntimeError('生产评价读取阶段改变共享训练台账')
            continue
        command = [handoff['python'], '-u', '-B', str(Path(handoff['engine_root'])/'run.py'), *step['args']]
        process_path = reports/(step['name']+'.process.json')
        process_paths.append(process_path)
        receipt = supervise(command, handoff['engine_root'], env, reports/(step['name']+'.log'),
                             1200, 512*1024, 16*1024**3, process_path)
        if step['kind'] == 'evaluate':
            evaluation = read_json(step['output'])
            write_json(reports/(step['name']+'.json'),evaluation)
            records.append(dict(step=step,process=receipt,evaluation=evaluation,
                scope='两例实际ROI技术评价',wall_seconds=receipt['wall_seconds']))
            write_json(reports/'execution.json',records)
            if {r['case_id'] for r in evaluation['cases']} != {'ct_1001','ct_1012'}:
                raise ValueError('生产评价未覆盖本轮两例')
            if evaluation['scope'] != 'technical':
                raise ValueError('本轮评价范围必须标记technical')
            if AttemptLedger(handoff['attempt_ledger']).read() != plan['ledger']:
                raise RuntimeError('评价阶段改变共享训练台账')
            continue
        summary = read_json(Path(step['output'])/'summary.json')
        write_json(reports/(step['name']+'.production-summary.json'), summary)
        item = dict(step=step, process=receipt, production_summary=summary,
                    production_events=prediction_events(summary))
        records.append(item)
        write_json(reports/'execution.json', records)
        require_prediction_events(step,item['production_events'])
        first = Path(step['output'])/'ct_1001.nii.gz'
        stamp = [first.stat().st_size, first.stat().st_mtime_ns]
        if step['name'].endswith('_first'):
            before[step['model']] = stamp
        else:
            item['first_output_unchanged'] = stamp == before[step['model']]
            item['reuse_events'] = [r for r in item['production_events'] if r['record']['event'] in
                                   ('prediction_reused', 'model_case_reused') and r['record']['case_id'] == 'ct_1001']
            item['geometry'] = geometry(handoff,step['model'],step['output'])
        write_json(reports/'execution.json', records)
        if AttemptLedger(handoff['attempt_ledger']).read() != plan['ledger']:
            raise RuntimeError('阶段3改变共享训练台账，立即终止')
    return dict(state='INCOMPLETE', records=records, target_models=list(MODELS),
                review_mode=plan['review_mode'], unavailable_models=plan['unavailable_models'])
