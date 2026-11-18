import ast
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import nibabel as nib
import numpy as np

from .attempts import AttemptLedger
from .contracts import TOOL_ROOT, confined, git, read_json, write_json
from .runner import inventory


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
    if actual != handoff['engine_commit'] or actual != handoff['training_engine_commit']:
        raise ValueError('生产版本与阶段2训练证据不对应，须保留版本差异并重新审核')
    ledger = AttemptLedger(handoff['attempt_ledger']).read()
    if any(r['outcome'] == 'started' for r in ledger['attempts']):
        raise ValueError('阶段2仍有未结算尝试，不能开始阶段3')
    steps = []
    for model in MODELS:
        item = handoff['models'][model]
        checkpoint = selected_best(item)
        base = ['predict'] if model == 'stu' else ['models', 'predict']
        cli = engine / 'src/chd_segmentation' / ('cli.py' if model == 'stu' else 'models_cli.py')
        if '--checkpoint' not in flags(cli):
            raise ValueError(model+'生产CLI尚无显式checkpoint入口，等待助手2交接')
        output = str(Path(handoff['work_root']) / 'assistant3' / model)
        args = [*base, '--config', item['config'], '--checkpoint', checkpoint, '--output', output]
        if model != 'stu':
            args += ['--prepared', item['prepared'], '--device', 'cpu']
        if model == 'med':
            args += ['--prompt-mode', item['prompt_mode']]
            if item['prompt_mode'] == 'external':
                args += ['--prompts', item['prompts']]
            elif item['prompt_mode'] != 'label-box':
                raise ValueError('Med提示模式无效')
        for label, cases in (('first', ['ct_1001']), ('resume', ['ct_1001', 'ct_1012'])):
            requested = [arg for name in cases for arg in ('--case', name)]
            steps.append(dict(name=model+'_'+label, model=model, output=output,
                              checkpoint=checkpoint, args=args+requested))
    return dict(steps=steps, ledger=ledger, engine_commit=actual,
                scope='zero_additional_training', roles=ROLES)


def geometry(handoff, output):
    result = []
    for case, source in handoff['source_images'].items():
        if case not in ('ct_1001', 'ct_1012'):
            raise ValueError('禁止读取额外病例')
        original, prediction = nib.load(source), nib.load(str(Path(output)/(case+'.nii.gz')))
        matches = original.shape[:3] == prediction.shape[:3] and np.allclose(
            original.affine, prediction.affine, atol=1e-5, rtol=0)
        result.append(dict(case_id=case, shape=list(prediction.shape), matches=bool(matches)))
    return result


def reuse_events(paths, offsets):
    events = []
    for path in paths:
        with Path(path).open('rb') as stream:
            stream.seek(offsets[path])
            for line in stream:
                if line.strip():
                    events.append(json.loads(line))
    return [r for r in events if r.get('event') in ('prediction_reused', 'model_case_reused')
            and r.get('case_id') == 'ct_1001']


def budget(handoff):
    rows = inventory(handoff['work_root'])
    reports = inventory(handoff['report_root'])
    own_reports = inventory(Path(handoff['report_root'])/'assistant3')
    if len(rows) > 400 or sum(r['bytes'] for r in rows) > 16*1024**3:
        raise RuntimeError('共享工作目录超过批准范围')
    if sum(r['bytes'] for r in reports) > 20*1024**2 or sum(r['bytes'] for r in own_reports) > 5*1024**2:
        raise RuntimeError('共享或助手3报告超过批准范围')
    return dict(files=len(rows), bytes=sum(r['bytes'] for r in rows))


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
    root.mkdir(parents=True)
    reports.mkdir(parents=True, exist_ok=True)
    cache = root/'cache'
    cache.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1',
               PYTHONIOENCODING='utf-8', TEMP=str(cache), TMP=str(cache),
               XDG_CACHE_HOME=str(cache), TORCH_HOME=str(cache/'torch'),
               MPLCONFIGDIR=str(cache/'matplotlib'), HF_HOME=str(cache/'huggingface'))
    records, before = [], {}
    started = perf_counter()
    write_json(reports/'owner.json', dict(work=str(root.resolve()), shared_cleanup_owner=handoff['cleanup_owner']))
    try:
        from .simulation import check_resources
        check_resources(dict(engine_root=handoff['engine_root'], engine_commit=plan['engine_commit'],
                             simulator_root=handoff['simulator_root']), root, reports)
        for step in plan['steps']:
            budget(handoff)
            event_paths = handoff['models'][step['model']]['event_logs']
            offsets = {p: Path(p).stat().st_size if Path(p).exists() else 0 for p in event_paths}
            command = [handoff['python'], '-u', '-B', str(Path(handoff['engine_root'])/'run.py'), *step['args']]
            receipt = supervise(command, handoff['engine_root'], env, reports/(step['name']+'.log'),
                                 1200, 512*1024, 16*1024**3, reports/(step['name']+'.process.json'))
            item = dict(step=step, process=receipt)
            first = Path(step['output'])/'ct_1001.nii.gz'
            stamp = [first.stat().st_size, first.stat().st_mtime_ns]
            if step['name'].endswith('_first'):
                before[step['model']] = stamp
            else:
                item['first_output_unchanged'] = stamp == before[step['model']]
                item['reuse_events'] = reuse_events(event_paths, offsets)
                item['geometry'] = geometry(handoff, step['output'])
                item['production_summary'] = read_json(Path(step['output'])/'summary.json')
            records.append(item)
            write_json(reports/'execution.json', records)
            if AttemptLedger(handoff['attempt_ledger']).read() != plan['ledger']:
                raise RuntimeError('阶段3改变共享训练台账，立即终止')
    finally:
        final_ledger = AttemptLedger(handoff['attempt_ledger']).read()
        write_json(reports/'review.json', dict(state='INCOMPLETE', records=records,
            wall_seconds=perf_counter()-started, training_ledger_unchanged=final_ledger == plan['ledger'],
            tool_commit=git(TOOL_ROOT,'rev-parse','HEAD'), production_commit=plan['engine_commit'],
            cleanup='交由指定所有者核对共享最后消费者后执行', capacity=budget(handoff),
            evidence_note='预测记录与几何为实际消费证据；待逐项审核待办，不能总PASS'))
        process_records = [read_json(p) for p in reports.glob('*.process.json')]
        if process_records and all(r['all_workers_exited'] for r in process_records):
            confined(root, Path(handoff['work_root']))
            rows = inventory(root)
            write_json(reports/'cleanup.json', dict(state='pending', files=rows))
            shutil.rmtree(root)
            write_json(reports/'cleanup.json', dict(state='completed', deleted_files=len(rows),
                released_bytes=sum(r['bytes'] for r in rows), work_removed=not root.exists(),
                shared_model_data_cleanup_owner=handoff['cleanup_owner']))
    return read_json(reports/'review.json')
