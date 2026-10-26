"""Historical inputs are materialized into a private workspace, never mutated in place."""
import os
from pathlib import Path
import shutil
import time
from contextlib import contextmanager
from server_data import digest, read_json, write_json


class CacheMismatch(ValueError):
    pass


@contextmanager
def source_lock(source):
    """Coordinate with legacy FileLock/.run.lock writers, including other runtimes."""
    source = Path(source)
    plan = read_json(source / 'plan.json') if (source / 'plan.json').is_file() else None
    work = Path(plan['workspace']) if plan else source
    if os.name != 'posix':
        # Windows is not an execution host; tests exercise independent copies only.
        yield
        return
    import fcntl
    with (work / '.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('历史工作区仍被使用，不能复用或回退') from error
        yield


def inspect_input(config):
    from task_runtime import active_processes, load_plan
    source = Path(config['input_job']).expanduser().resolve()
    evidence={}
    if (source / 'plan.json').is_file():
        plan = load_plan(source)
        if Path(plan['runtime']).resolve() != Path(config['runtime']).resolve():
            raise ValueError('历史输入必须来自同一runtime；跨runtime请先单独核查共享写入')
        if active_processes(source):
            raise ValueError('历史输入仍在运行，不能复用')
        work = Path(plan['workspace'])
        for name in ['plan.json','state.json','code_manifest.json']:
            evidence[name]=digest(source/name)
        for file in sorted((source/'receipts').glob('*.json')):
            evidence[str(file.relative_to(source))]=digest(file)
        for relative,sha in read_json(source/'code_manifest.json').items():
            if digest(Path(plan['code'])/relative)!=sha:
                raise ValueError('来源任务冻结源码不符：'+relative)
    else:
        work = source
    meta = work / '00_admin/server_config.json'
    old = read_json(meta)
    from cohort import PROJECT_ID
    from server_data import protocol_hash
    if old.get('project_id') != PROJECT_ID or old.get('protocol_sha256') != protocol_hash():
        raise ValueError('输入属于不同工程或数据协议，禁止复用')
    if Path(old['data_root']).resolve() != Path(config['data']).resolve():
        raise ValueError('输入任务的数据来源不同，请选择对应原始数据')
    if old['gpu_memory_gb'] != config.get('gpu_memory', 5):
        raise CacheMismatch('规划显存预算不同，需要新缓存')
    # Do not silently accept a partially prepared source.
    splits = read_json(work / '04_data/manifests/splits_v1.json')['folds']
    fingerprints = {str(meta.relative_to(work)): digest(meta)}
    for relative in ['04_data/manifests/case_inventory.json', '04_data/manifests/splits_v1.json']:
        fingerprints[relative] = digest(work / relative)
    for fold in config['folds']:
        complete = work / f'09_reports/{fold}_preparation.json'
        if not complete.exists() or read_json(complete).get('status') != 'passed':
            raise CacheMismatch(f'{fold} 缓存未完整准备')
        prep = work / '04_data/derived/nnUNet_preprocessed' / f"Dataset{splits[fold]['dataset_id']:03d}_{fold}"
        signature = read_json(prep / 'server_preparation_signature.json')
        current_code = Path(__file__).resolve().parent
        current_implementation = {name: digest(current_code / name) for name in ['server_prepare.py','data_pipeline.py','prepare_nnunet.py']}
        if signature.get('implementation_sha256') != current_implementation:
            raise CacheMismatch('缓存实现版本不兼容')
        plan_file = prep / 'WholeheartBaselinePlans.json'
        if read_json(complete)['plan_sha256'] != digest(plan_file):
            raise CacheMismatch(f'{fold} 计划摘要不符')
        plan_value = read_json(plan_file)
        cache = prep / plan_value['configurations']['3d_fullres']['data_identifier']
        for case in splits[fold]['train'] + splits[fold]['val']:
            if not all((cache / (case + suffix)).is_file() for suffix in ['.b2nd', '_seg.b2nd', '.pkl']):
                raise CacheMismatch(f'{fold} 缓存缺少病例文件')
        for path in [complete, plan_file, prep / 'server_preparation_signature.json']:
            fingerprints[str(path.relative_to(work))] = digest(path)
    if 'predict' in config['tasks'] and 'train' not in config['tasks']:
        for fold in config['folds']:
            for method in config['methods']:
                folder = work / f'07_experiments/formal/{fold}_{method}_seed0'
                run = read_json(folder / 'run.json')
                if run.get('status') != 'completed' or not run.get('formal_training'):
                    raise ValueError(f'{fold}/{method} 尚未正式训练完成')
                for relative in ['run.json', 'fold_0/checkpoint_best.pth', 'fold_0/checkpoint_final.pth']:
                    path = folder / relative
                    fingerprints[str(path.relative_to(work))] = digest(path)
    if 'summarize' in config['tasks'] and 'predict' not in config['tasks']:
        frozen=work/'08_results/evaluation/frozen_models.json'
        fingerprints[str(frozen.relative_to(work))]=digest(frozen)
        for fold in config['folds']:
            for method in config['methods']:
                folder=work/f'08_results/evaluation/{fold}_{method}'
                for case in splits[fold]['target']:
                    path=folder/f'{case}.json'
                    if read_json(path).get('status')!='completed':raise ValueError('来源评价未完成：'+str(path))
                    fingerprints[str(path.relative_to(work))]=digest(path)
    return dict(directory=str(source), workspace=str(work), files=fingerprints,evidence=evidence,
                verification='来源清单/计划/所选权重完整SHA256；缓存大文件存在检查，执行时保留原预处理校验')


def copy_input(plan):
    if not plan.get('source'):
        return
    with source_lock(plan['source']['directory']):
        return _copy_input_locked(plan)


def _copy_input_locked(plan):
    source = plan.get('source')
    if not source:
        return
    work, src = Path(plan['workspace']), Path(source['workspace'])
    if (work / '00_admin/input_receipt.json').exists():
        return  # Resume uses its own committed copy, not the evolving source task.
    for relative,sha in source.get('evidence',{}).items():
        if digest(Path(source['directory'])/relative)!=sha:
            raise ValueError('来源任务证据在选择后发生变化：'+relative)
    for relative, sha in source['files'].items():
        if digest(src / relative) != sha:
            raise ValueError('历史输入在选择后发生变化：' + relative)
    if work.exists() and any(work.iterdir()):
        archive = work.with_name('workspace_partial_input_' + str(__import__('time').time_ns()))
        work.rename(archive)
        print('未提交的输入复制保留于：' + str(archive), flush=True)
    work.mkdir(parents=True, exist_ok=True)
    for relative in source.get('evidence',{}):
        dest=work/'00_admin/input_evidence'/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(source['directory'])/relative,dest)
    # Independent copies intentionally favour correctness across NAS/filesystems.
    # No symlinks/hardlinks: downstream recovery must never alter source bytes.
    for relative in ['04_data/manifests', '06_configs']:
        if (src / relative).exists():
            shutil.copytree(src / relative, work / relative)
    splits = read_json(src / '04_data/manifests/splits_v1.json')['folds']
    for fold in plan['folds']:
        dataset = f"Dataset{splits[fold]['dataset_id']:03d}_{fold}"
        for area in ['nnUNet_raw', 'nnUNet_preprocessed']:
            relative = Path('04_data/derived') / area / dataset
            if (src / relative).exists():
                shutil.copytree(src / relative, work / relative)
        dest = work / f'09_reports/{fold}_preparation.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f'09_reports/{fold}_preparation.json', dest)
    old = read_json(src / '00_admin/server_config.json')
    old['work_root'] = str(work.resolve())
    write_json(work / '00_admin/server_config.json', old)
    from model_catalog import save_selection
    save_selection(work, plan['folds'], plan['methods'])
    if 'predict' in plan['tasks'] and 'train' not in plan['tasks']:
        for fold in plan['folds']:
            for method in plan['methods']:
                relative = f'07_experiments/formal/{fold}_{method}_seed0'
                shutil.copytree(src / relative, work / relative)
    if 'summarize' in plan['tasks'] and 'predict' not in plan['tasks']:
        # Existing records contain original absolute prediction paths, read-only.
        shutil.copytree(src / '08_results/evaluation', work / '08_results/evaluation')
    write_json(work / '00_admin/input_receipt.json', source)


def verify_resume(job, level, pending=None):
    if level not in {'simple', 'full', 'none'}:
        raise ValueError('检查等级应为simple/full/none')
    job = Path(job)
    plan = read_json(job / 'plan.json')
    # Source code and frozen plan are never bypassed by an artifact-check preference.
    for relative, sha in read_json(job / 'code_manifest.json').items():
        if digest(Path(plan['code']) / relative) != sha:
            raise ValueError('任务源码快照发生变化：' + relative)
    receipts = sorted((job / 'receipts').glob('*.json'))
    state=read_json(job/'state.json') if (job/'state.json').exists() else {'steps':{}}
    pending=pending if pending is not None else [t for t,s in state['steps'].items() if s['status']!='completed']
    deps={'environment':[],'prepare':[],'smoke':['prepare'],'train':['prepare'],
          'predict':['prepare','train'],'summarize':['predict']}
    direct={d for task in pending for d in deps.get(task,[])}
    if level=='simple':receipts=[r for r in receipts if r.stem in direct]
    if level=='none':receipts=[]
    report=dict(level=level,pending=pending,scope=[r.stem for r in receipts],
                missing_receipts=sorted(direct-{r.stem for r in receipts}) if level!='none' else [],
                checked_files=0,hashed_bytes=0,large_file_rule='size+mtime' if level=='simple' else level,
                code_manifest='full_sha256',status='running')
    files=[(r.stem,raw,info) for r in receipts for raw,info in read_json(r).get('files',{}).items()]
    total=sum(i['size'] for _,_,i in files if level=='full' or i['size']<=1024*1024)
    started=last=time.monotonic()
    try:
      for stem,raw,info in files:
            path = Path(raw)
            stat = path.stat()
            if stat.st_size != info['size']:
                raise ValueError('历史产物大小发生变化：' + str(path))
            if level == 'full' or info['size'] <= 1024*1024:
                if digest(path) != info['sha256']:
                    raise ValueError('历史产物摘要不符：' + str(path))
                report['hashed_bytes']+=info['size']
            elif stat.st_mtime_ns != info['mtime_ns']:
                raise ValueError('历史大文件修改时间发生变化：' + str(path))
            report['checked_files']+=1
            if time.monotonic()-last>=2 or report['checked_files']==len(files):
                elapsed=max(.001,time.monotonic()-started);speed=report['hashed_bytes']/elapsed
                eta=f'{(total-report["hashed_bytes"])/speed:.1f}s' if speed else '未知'
                print(f'检查 {stem}: {report["checked_files"]}/{len(files)}；已读取 {report["hashed_bytes"]}/{total} bytes；{speed/1024**2:.1f} MiB/s；剩余约 {eta}；{level}',flush=True)
                last=time.monotonic()
      report['status']='passed'
    except BaseException as error:
      report.update(status='failed',error=str(error));raise
    finally:
      report['seconds']=time.monotonic()-started
      write_json(job/'resume-check.json',report)
    if report['missing_receipts']:print('旧任务缺少以下产物凭证，未宣称已验证：'+','.join(report['missing_receipts']),flush=True)


def receipt(job, step):
    plan = read_json(Path(job) / 'plan.json')
    work = Path(plan['workspace'])
    roots = {'environment': [], 'prepare': ['04_data'], 'smoke': ['09_reports/smoke_summary.json'],
             'train': ['07_experiments/formal'], 'predict': ['08_results/evaluation'],
             'summarize': ['09_reports/final_evaluation'],
             'assessment': ['09_reports/assessment.json','00_admin/assessment.json']}[step]
    paths = []
    for relative in roots:
        path = work / relative
        paths.extend([path] if path.is_file() else [p for p in path.rglob('*') if p.is_file()])
    files = {}
    for index, path in enumerate(paths, 1):
        stat = path.stat()
        files[str(path)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest(path))
        if index % 25 == 0:
            print(f'保存{step}产物凭证：{index}/{len(paths)}', flush=True)
    write_json(Path(job) / f'receipts/{step}.json', dict(files=files))
