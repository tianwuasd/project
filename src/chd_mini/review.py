import ast
import json
import os
from pathlib import Path
import shutil
import traceback
from time import perf_counter

import nibabel as nib
import numpy as np

from .attempts import AttemptLedger
from .contracts import TOOL_ROOT, confined, git, read_json, write_json
from .runner import inventory
from .version_review import full_commit, model_versions


ROLES = dict(train=['ct_1001'], validation=['ct_1012'], test=[])
MODELS = ('stu', 'q', 'med')


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


def build_review(handoff):
    if handoff['roles'] != ROLES or set(handoff['models']) != set(MODELS):
        raise ValueError('交接必须是ct_1001/ct_1012且包含三模型')
    if set(handoff['source_images']) != {'ct_1001', 'ct_1012'}:
        raise ValueError('几何来源必须恰好对应两例')
    if Path(handoff['work_root']).resolve() != Path('D:/projects/back/t17/chd-review-20261118').resolve():
        raise ValueError('工作根不符合本轮共享许可')
    if Path(handoff['report_root']).resolve() != Path('D:/projects/back/sync-reports/chd-review-20261118').resolve():
        raise ValueError('报告根不符合本轮共享许可')
    engine = Path(handoff['engine_root']).resolve(strict=True)
    actual = git(engine, 'rev-parse', 'HEAD')
    if actual != full_commit(handoff['engine_commit']):
        raise ValueError('生产目录与最终交接版本不对应')
    ledger = AttemptLedger(handoff['attempt_ledger']).read()
    if any(r['outcome'] == 'started' for r in ledger['attempts']):
        raise ValueError('阶段2仍有未结算尝试，不能开始阶段3')
    steps, versions = [], {}
    for model in MODELS:
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
                              checkpoint=checkpoint, args=args+requested))
    return dict(steps=steps, ledger=ledger, engine_commit=actual, model_versions=versions,
                scope='zero_additional_training', roles=ROLES)


def geometry(item, output):
    result = []
    if set(item['geometry_sources']) != {'ct_1001', 'ct_1012'}:
        raise ValueError('必须提供两例实际小区域来源')
    for case, source in item['geometry_sources'].items():
        if case not in ('ct_1001', 'ct_1012'):
            raise ValueError('禁止读取额外病例')
        original, prediction = nib.load(source), nib.load(str(Path(output)/(case+'.nii.gz')))
        matches = original.shape[:3] == prediction.shape[:3] and np.allclose(
            original.affine, prediction.affine, atol=1e-5, rtol=0)
        result.append(dict(case_id=case, shape=list(prediction.shape), matches=bool(matches),
                           source=str(source), scope='实际交接小区域shape/affine，未验证完整原图回填'))
    return result


def prediction_events_from_console(path):
    events = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if line.lstrip().startswith('{') and '"event":' in line:
            row = json.loads(line)
            if row.get('event') in ('prediction_reused', 'model_case_reused', 'prediction', 'model_case_completed'):
                if row.get('case_id') in ('ct_1001', 'ct_1012'):
                    events.append(dict(source_console=str(path), source_line=number, record=row))
    return events


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
        observed = []
        for path in handoff.get('training_event_logs', {}).get(stage, []):
            for line in Path(path).read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                counter = None
                if stage == 'stu' and row.get('event') == 'train':
                    counter = row.get('updates')
                if stage.startswith('q_') and row.get('event') == 'model_update_completed' and row.get('stage') == stage[2:]:
                    counter = row.get('update')
                if type(counter) is int and counter > 0:
                    observed.append(dict(path=str(path), event=row))
        counters = {r['event'].get('updates', r['event'].get('update')) for r in observed}
        successful = sum(r['stage'] == stage and r['outcome'] == 'successful' for r in ledger['attempts'])
        evidence[stage] = dict(ledger_successful=successful, optimizer_update_events=observed,
            observed_update_positions=sorted(counters),
            successful_updates=len(counters) if counters else None,
            status='已取得优化器更新事件，待与完整阶段2记录核对' if counters else '未取得可解释的优化器更新事件，未验证')
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
                geometry_scope='两例实际交接小区域；完整原图回填未验证')
            write_json(self.reports/'review.json', summary)
            # 最终容量只记录，不在此处抛出额度错误阻止清理。
            summary['capacity'] = capacity(self.handoff)
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
        command = [handoff['python'], '-u', '-B', str(Path(handoff['engine_root'])/'run.py'), *step['args']]
        process_path = reports/(step['name']+'.process.json')
        process_paths.append(process_path)
        receipt = supervise(command, handoff['engine_root'], env, reports/(step['name']+'.log'),
                             1200, 512*1024, 16*1024**3, process_path)
        item = dict(step=step, process=receipt,
                    production_events=prediction_events_from_console(reports/(step['name']+'.log')))
        first = Path(step['output'])/'ct_1001.nii.gz'
        stamp = [first.stat().st_size, first.stat().st_mtime_ns]
        if step['name'].endswith('_first'):
            before[step['model']] = stamp
        else:
            item['first_output_unchanged'] = stamp == before[step['model']]
            item['reuse_events'] = [r for r in item['production_events'] if r['record']['event'] in
                                   ('prediction_reused', 'model_case_reused') and r['record']['case_id'] == 'ct_1001']
            item['geometry'] = geometry(handoff['models'][step['model']], step['output'])
            item['production_summary'] = read_json(Path(step['output'])/'summary.json')
        records.append(item)
        write_json(reports/'execution.json', records)
        if AttemptLedger(handoff['attempt_ledger']).read() != plan['ledger']:
            raise RuntimeError('阶段3改变共享训练台账，立即终止')
    return dict(state='INCOMPLETE', records=records)
