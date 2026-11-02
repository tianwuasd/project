"""Preprocessing supervisor: contract first, atomic case commits, explicit scope.

Audit is dataset-wide even for --cases/--limit. Fingerprints use the complete
TRAIN subset of the selected fold, never just the short-test target cases.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import nibabel as nib
import numpy as np
from ..config import ROOT
from ..data.analysis import sha256_file
from ..data.author_excel import read_spacing_excel
from ..data.discovery import discover_pairs,descriptor_evidence
from ..observability.locking import run_lock
from ..observability.runlog import RunLog,write_json
from ..acceleration.preprocess_cpu import execute_jobs,estimate_memory
from .registry import get_method
from .cohort import TARGETS,frozen_roles,select_cohort
from .fingerprint import make_plan


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True).encode()).hexdigest()


def producer_sources():
    package = ROOT/'src/chd_stunet'
    paths = [ROOT/'preprocess_worker.py',package/'config.py',package/'__init__.py',
             package/'data/__init__.py',package/'acceleration/__init__.py']
    paths.extend((ROOT/'configs').rglob('*.json'))
    for directory in ('preprocessing','_vendor','observability'):
        paths.extend((package/directory).glob('*.py'))
    paths.extend([package/'acceleration/preprocess_cpu.py',package/'data/author_excel.py',
                  package/'data/discovery.py',package/'data/analysis.py'])
    return {p.relative_to(ROOT).as_posix():sha256_file(p) for p in sorted(set(paths))}


def build_signature(raw, pairs, method, descriptor):
    versions = {name:importlib.metadata.version(name) for name in
                ('nnunetv2','numpy','scipy','nibabel','blosc2','torch','scikit-image')}
    if versions['nnunetv2'] != method['nnunet_version']:
        raise ValueError('This adapter requires audited nnunetv2 '+method['nnunet_version'])
    stats = []
    for pair in pairs:
        for kind in ('image','label'):
            p = Path(pair[kind])
            info = p.stat()
            stats.append(dict(name=p.name,size=info.st_size,mtime_ns=info.st_mtime_ns))
    sources = producer_sources()
    record = dict(method=method,raw_root=str(raw),source_stats=stats,versions=versions,
                  producer_sha256=sources,
                  descriptor_sha256=sha256_file(descriptor) if descriptor else None,
                  spreadsheet_sha256=sha256_file(raw/'imagechd_dataset_image_info.xlsx'))
    return dict(identity=digest(record),**record)


def check_resume_before_writing(output, signature, pairs):
    """A refused resume must not alter old status/log/code/results at all."""
    saved = output/'signature.json'
    if not saved.is_file() or json.loads(saved.read_text(encoding='utf-8')) != signature:
        raise ValueError('Resume contract differs; keep old run and use a new output directory')
    by_id = {p['case_id']:p for p in pairs}
    for path in (output/'audit/cases').glob('*.json'):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('success'):
            for kind,expected in row['source_hashes'].items():
                if sha256_file(by_id[row['case_id']][kind]) != expected:
                    raise ValueError('Resume source content changed: '+row['case_id'])
            if sha256_file(row['sample_path']) != row['sample_sha256']:
                raise ValueError('Resume intensity sample changed: '+row['case_id'])


def cache_valid(foldroot,cid,contract,source_hashes):
    path = foldroot/'completion'/f'{cid}.json'
    if not path.is_file():
        return False
    row = json.loads(path.read_text(encoding='utf-8'))
    if row.get('contract') != contract or row.get('source_hashes') != source_hashes:
        return False
    return all((foldroot/'data'/name).is_file() and sha256_file(foldroot/'data'/name)==sha
               for name,sha in row.get('file_sha256',{}).items()) and len(row.get('file_sha256',{})) == 3


def run_prepare(raw_root,output,*,method_name='stunet-s',descriptor=None,folds=(0,1,2),
                selected_cases=(),limit=None,workers=None,readers=None,resume=False,audit_only=False):
    raw, output = Path(raw_root).resolve(strict=True),Path(output).resolve()
    if not raw.is_dir() or raw==output or output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError('Raw data and derived output must be separate directories')
    method = get_method(method_name)
    discovery = discover_pairs(raw)
    pairs = discovery['pairs']
    if discovery['missing_labels'] or discovery['orphan_labels'] or len(pairs)!=method['expected_source_pairs']:
        raise ValueError('Incomplete/changed source pairing; run read-only analysis before this method')
    signature = build_signature(raw,pairs,method,descriptor)
    if signature['spreadsheet_sha256'] != method['spacing_excel_sha256']:
        raise ValueError('Author spacing spreadsheet differs from audited source; review required')
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError('Output is not empty; use another directory or explicit --resume')
        check_resume_before_writing(output,signature,pairs)
    elif resume:
        raise ValueError('Resume requires an existing signed run')
    with run_lock(output/'.run.lock'):
        # A second simultaneous creator must not slip past the pre-lock check.
        if (output/'signature.json').exists():
            if json.loads((output/'signature.json').read_text(encoding='utf-8')) != signature:
                raise ValueError('Resume contract changed during lock acquisition')
        output.mkdir(parents=True,exist_ok=True)
        log = RunLog(output)
        started = time.monotonic()
        log.status('running',stage='input_contract')
        try:
            write_json(output/'signature.json',signature)
            write_json(output/'effective_config.json',dict(method=method,raw_root=str(raw),
                output=str(output),folds=list(folds),selected_cases=list(selected_cases),limit=limit,
                requested_workers=workers,requested_readers=readers,audit_only=audit_only,
                source_scope='110 pairs audit; 96 clean cohort; historical 97 roles retained'))
            for relative in signature['producer_sha256']:
                target = output/'code'/relative
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/relative,target)
            shutil.copyfile(ROOT/'configs/preprocessing'/f'{method_name}.json',output/'code/method.json')
            spacing = read_spacing_excel(raw/'imagechd_dataset_image_info.xlsx')
            evidence, descriptor_meta = descriptor_evidence(descriptor,raw)
            if descriptor and not descriptor_meta['spreadsheet_matches_descriptor']:
                raise ValueError('Frozen descriptor spacing provenance mismatch')
            write_json(output/'descriptor_evidence.json',descriptor_meta)
            log.event('run.start',f"预处理方法 {method_name}；逐例审计{len(pairs)}对；描述文件{'交叉核验' if descriptor else '不需要，将从原始文件生成清单'}",identity=signature['identity'])
            jobs, rows = [], []
            for pair in pairs:
                cid = pair['case_id']
                previous = output/'audit/cases'/f'{cid}.json'
                if previous.is_file():
                    row = json.loads(previous.read_text(encoding='utf-8'))
                    if row.get('success'):
                        rows.append(row)
                        continue
                if cid not in spacing:
                    raise ValueError('Missing author spacing: '+cid)
                image = nib.load(pair['image'])
                source_spacing = spacing[cid]['spacing_mm']
                task = dict(stage='audit',pair=pair,method=method,spacing_xyz=source_spacing,
                    spacing_evidence=dict(source='author_excel',sha256=signature['spreadsheet_sha256'],
                        row=spacing[cid]['excel_row'],array_axis_mapping='J,K,M -> original x,y,z',
                        independently_verified_dicom_geometry=False),
                    expected_hashes=evidence.get(cid,{}).get('expected_hashes',{}),
                    sample_path=str(output/'audit/samples'/f'{cid}.npy'),gate=str(output/'read_gate.json'),
                    result=str(previous))
                taskfile = output/'tasks/audit'/f'{cid}.json'
                write_json(taskfile,task)
                jobs.append(dict(name='audit_'+cid,task=str(taskfile),result=str(previous),
                    entry=str(output/'code/preprocess_worker.py'),estimate=estimate_memory(image.shape,source_spacing,audit=True)))
            failures = []
            def audited(name,result):
                (rows if result['success'] else failures).append(result)
            reused_audit = len(rows)
            execute_jobs(jobs,output,log,stage_name='audit',workers=workers,readers=readers,completed=audited)
            if failures or len(rows)!=len(pairs):
                write_json(output/'audit/failures.json',failures)
                raise ValueError('Audit failures block cohort fitting; other case results preserved')
            rows.sort(key=lambda r:r['case_id'])
            if len(set(r['source_hashes']['image'] for r in rows)) != len(rows):
                raise ValueError('Duplicate image SHA256; case identities require review')
            complete = [r['case_id'] for r in rows if not r['missing_label_ids']]
            if len(complete) != method['expected_complete_before_exclusions']:
                raise ValueError('Historical complete-annotation cohort changed')
            roles = frozen_roles(complete,method['split_seed'])
            for row in rows:
                cid = row['case_id']
                row['roles'] = roles.get(cid,{})
                if evidence.get(cid,{}).get('roles',{}) != row['roles'] and descriptor:
                    raise ValueError('Historical frozen roles differ: '+cid)
            included, excluded = select_cohort(rows,method)
            manifest = dict(protocol=method['protocol'],method=method_name,source_pairs=len(rows),
                included_pairs=len(included),excluded_pairs=len(excluded),patient_identity_verified=False,
                identity=signature['identity'],cases=included,excluded=excluded,
                historical_complete_count=len(complete),new_split_generated=False)
            write_json(output/'manifests/imagechd_input.json',manifest)
            write_json(output/'manifests/historical_roles_97.json',roles)
            write_json(output/'manifests/excluded.json',excluded)
            fold_counts = {str(f):{role:sum(r['roles'][str(f)]==role for r in included)
                           for role in ('train','val','test')} for f in range(3)}
            write_json(output/'manifests/splits.json',dict(folds=fold_counts,
                roles={r['case_id']:r['roles'] for r in included},patient_level_verified=False))
            log.event('cohort.ready',f"纳入{len(included)}对，排除{len(excluded)}对；各折保留原角色",fold_counts=fold_counts)
            selected = set(selected_cases)
            if selected-set(r['case_id'] for r in included):
                raise ValueError('Requested cases outside this clean cohort: '+str(sorted(selected-set(r['case_id'] for r in included))))
            targets = [r for r in included if not selected or r['case_id'] in selected]
            if limit:
                targets = targets[:limit]
            samples = {r['case_id']:np.load(r['sample_path'],allow_pickle=False) for r in included}
            processed, reused, failed = 0, 0, []
            dataset = dict(channel_names={'0':'CT'},labels=TARGETS,file_ending='.nii.gz',
                           numTraining=0,description='preprocessing only; roles are in splits.json')
            write_json(output/'dataset.json',dataset)
            for fold in (() if audit_only else folds):
                log.event('fingerprint.start',f"fold{fold}：仅使用训练角色拟合强度统计与目标spacing")
                fingerprint,plan = make_plan(included,samples,fold,method)
                foldroot = output/f'fold_{fold}'
                contract = digest(dict(run=signature['identity'],plan=plan))
                write_json(foldroot/'dataset_fingerprint.json',fingerprint)
                write_json(foldroot/'preprocessing_plan.json',plan)
                # Deliberately separate train/validation from outer test roles.
                write_json(foldroot/'splits.json',{role:[r['case_id'] for r in included if r['roles'][str(fold)]==role]
                           for role in ('train','val','test')})
                target_spacing = plan['target_spacing_before_transpose'][::-1]
                jobs = []
                for row in targets:
                    cid = row['case_id']
                    if cache_valid(foldroot,cid,contract,row['source_hashes']):
                        reused += 1
                        continue
                    result_path = foldroot/'results'/f'{cid}.json'
                    task = dict(stage='prepare',pair=row['pair'],method=method,plan=plan,dataset_json=dataset,
                        spacing_xyz=row['spacing_xyz'],spacing_evidence=row['spacing_evidence'],
                        expected_hashes=row['source_hashes'],gate=str(output/'read_gate.json'),
                        result=str(result_path),foldroot=str(foldroot),contract=contract)
                    taskfile = output/'tasks'/f'fold_{fold}'/f'{cid}.json'
                    write_json(taskfile,task)
                    jobs.append(dict(name=f'fold{fold}_{cid}',task=str(taskfile),result=str(result_path),
                        entry=str(output/'code/preprocess_worker.py'),estimate=estimate_memory(row['shape_xyz'],row['spacing_xyz'],target_spacing)))
                def prepared(name,result):
                    nonlocal processed
                    if result['success']:
                        processed += 1
                    else:
                        failed.append(dict(name=name,**result))
                execute_jobs(jobs,output,log,stage_name=f'fold{fold}_prepare',workers=workers,
                             readers=readers,completed=prepared)
            summary = dict(method=method_name,protocol=method['protocol'],source_pairs=len(rows),
                included_pairs=len(included),excluded_pairs=len(excluded),fold_counts=fold_counts,
                audited_reused=reused_audit,requested_folds=list(folds),audit_only=audit_only,
                requested_case_ids=[r['case_id'] for r in targets],processed_cases=processed,
                reused_cases=reused,failed_cases=failed,elapsed_seconds=time.monotonic()-started,
                scope='audit_only' if audit_only else 'selected_real_cases' if len(targets)<len(included) else 'full_clean_cohort',
                model_training_executed=False)
            write_json(output/'summary.json',summary)
            text = (f"# {method_name} 预处理运行\n\n"
                    f"源文件对 {len(rows)}；纳入 {len(included)}；排除 {len(excluded)}。\n\n"
                    f"本次新处理 {processed}；复用 {reused}；失败 {len(failed)}；范围 `{summary['scope']}`。\n\n"
                    "各折拟合病例见 `fold_N/dataset_fingerprint.json` 的 fit_ids；仅 train。"
                    "原角色与排除理由见 manifests；逐例完整日志见 logs。\n\n"
                    "预处理完成项是 completion/*.json；未做训练或预测。没有训练精度或加速倍数结论。\n")
            (output/'report.md').write_text(text,encoding='utf-8')
            state = 'failed' if failed else 'completed'
            log.status(state,stage='done',summary=summary)
            log.event('run.done',f"结束：新处理{processed}，复用{reused}，失败{len(failed)}；范围{summary['scope']}")
            return summary
        except BaseException as error:
            log.event('run.failed',str(error),traceback=traceback.format_exc())
            log.status('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
            raise
