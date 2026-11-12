import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
import unittest

import blosc2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from chd_stunet.data.author_excel import read_spacing_excel
from chd_stunet.data.discovery import discover_pairs
from chd_stunet.observability.runlog import write_json
from chd_stunet.preprocessing.cohort import TARGETS
from chd_stunet.preprocessing.fingerprint import make_plan
from chd_stunet.preprocessing.registry import get_method
from chd_stunet.preprocessing.run import cache_valid
from chd_stunet.preprocessing.shared_cache import MODE, begin_view, case_ids, commit_view, data_directory, digest, lease, ready_view
from chd_stunet.preprocessing.stunet_s import audit_case, hashes, preprocess_case


def compare_properties(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare_properties(left[key], right[key])
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and np.array_equal(left, right)
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            compare_properties(a, b)
    else:
        assert left == right


def compare_case(reference, shared, cid):
    summaries = []
    for suffix in ('.b2nd', '_seg.b2nd'):
        a = blosc2.open(str(reference/(cid+suffix)), mode='r', dparams={'nthreads': 1})
        b = blosc2.open(str(shared/(cid+suffix)), mode='r', dparams={'nthreads': 1})
        assert a.shape == b.shape and a.dtype == b.dtype
        for start in range(0, a.shape[1], 16):
            assert np.array_equal(a[:, start:start+16], b[:, start:start+16])
        summaries.append(dict(suffix=suffix, shape=list(a.shape), dtype=str(a.dtype), full_array_equal=True))
        del a, b
    with (reference/(cid+'.pkl')).open('rb') as stream:
        a = pickle.load(stream)
    with (shared/(cid+'.pkl')).open('rb') as stream:
        b = pickle.load(stream)
    compare_properties(a, b)
    return dict(arrays=summaries, properties_equal=True)


def inventory(root):
    files = [path for path in root.rglob('*') if path.is_file()]
    return dict(files=len(files), logical_bytes=sum(path.stat().st_size for path in files))


def check(raw, work, record, cid):
    started = time.monotonic()
    pair = next(row for row in discover_pairs(raw)['pairs'] if row['case_id'] == cid)
    original_hashes = hashes(pair)
    method = get_method('stunet-s-partial')
    spacing = read_spacing_excel(raw/'imagechd_dataset_image_info.xlsx')[cid]
    gate = work/'read_gate.json'
    write_json(gate, dict(slots=1))
    job = dict(pair=pair, method=method, spacing_xyz=spacing['spacing_mm'],
               spacing_evidence=dict(source='author_excel', excel_row=spacing['excel_row']),
               expected_hashes=original_hashes, gate=str(gate), sample_path=str(work/'samples'/f'{cid}.npy'))
    before = time.monotonic()
    row = audit_case(job)
    row.update(cohort_group='main', roles={str(f): 'train' for f in range(3)})
    sample = np.load(row['sample_path'], allow_pickle=False)
    _, plan0 = make_plan([row], {cid: sample}, 0, method)
    plan1 = copy.deepcopy(plan0)
    plan1['fold'] = 1
    stats = plan1['foreground_intensity_properties_per_channel']['0']
    stats['mean'] += .1*stats['std']
    plan1['target_spacing_before_transpose'] = [v*1.1 for v in plan1['target_spacing_before_transpose']]
    plan1['configurations']['3d_fullres']['spacing'] = [v*1.1 for v in plan1['configurations']['3d_fullres']['spacing']]
    record['audit_and_technical_plan_seconds'] = time.monotonic()-before
    record['parameter_scope'] = 'one real case; second technical plan perturbs mean and spacing; no research-cohort fit'
    shared = work/'shared'
    write_json(shared/'signature.json', dict(identity='single-real-case-storage-check', method={'storage_mode': MODE}))
    records = []
    first_job = None
    first_hashes = None
    for fold, plan in enumerate((plan0, plan1)):
        write_json(shared/f'fold_{fold}/preprocessing_plan.json', plan)
        contract = digest(dict(run='single-real-case-storage-check', plan=plan))
        common = dict(job, plan=plan, contract=contract, dataset_json={'channel_names': {'0': 'CT'}, 'labels': TARGETS},
                      cohort_group='main', roles=row['roles'], cpu_threads=1)
        stages = {}
        before = time.monotonic()
        reference = work/'reference'/f'fold_{fold}'
        independent = preprocess_case(dict(common, foldroot=str(reference), result=str(reference/'result.json')))
        stages['independent_seconds'] = time.monotonic()-before
        before = time.monotonic()
        with lease(shared, enabled=True) as token:
            begin_view(shared, fold, [cid], contract)
            produced = preprocess_case(dict(common, foldroot=str(shared/f'fold_{fold}'), shared_cache=True,
                                           cache_write_token=token, result=str(shared/f'fold_{fold}/result.json')))
            commit_view(shared, fold, [cid], contract)
        stages['shared_seconds'] = time.monotonic()-before
        before = time.monotonic()
        comparison = compare_case(reference/'data', data_directory(shared, fold), cid)
        assert cache_valid(shared/f'fold_{fold}', cid, contract, original_hashes, shared=True)
        assert ready_view(shared, fold)['state'] == 'ready'
        assert len(list((shared/'shared_data').iterdir())) == 3
        stages['compare_seconds'] = time.monotonic()-before
        records.append(dict(technical_view=fold, stages=stages, comparison=comparison,
                            independent_phases=independent['stage_timings'], shared_phases=produced['stage_timings']))
        if fold == 0:
            first_job = common
            first_hashes = produced['file_sha256']
        current = inventory(work)
        assert current['files'] <= 200 and current['logical_bytes'] <= 3*1024**3
    # 切回原参数，先留下 building 状态，再恢复构建，验证旧折记录不会复用另一折数组。
    before = time.monotonic()
    assert not cache_valid(shared/'fold_0', cid, first_job['contract'], original_hashes, shared=True)
    with lease(shared, enabled=True):
        begin_view(shared, 0, [cid], first_job['contract'])
    with unittest.TestCase().assertRaises(ValueError):
        ready_view(shared, 0)
    with lease(shared, enabled=True) as token:
        begin_view(shared, 0, [cid], first_job['contract'])
        restored = preprocess_case(dict(first_job, foldroot=str(shared/'fold_0'), shared_cache=True,
                                       cache_write_token=token, result=str(shared/'fold_0/restore-result.json')))
        commit_view(shared, 0, [cid], first_job['contract'])
    assert restored['file_sha256'] == first_hashes
    assert cache_valid(shared/'fold_0', cid, first_job['contract'], original_hashes, shared=True)
    comparison = compare_case(work/'reference/fold_0/data', data_directory(shared, 0), cid)
    record['restore'] = dict(seconds=time.monotonic()-before, building_rejected=True,
                            earlier_file_sha256_equal=True, comparison=comparison,
                            phases=restored['stage_timings'])
    assert hashes(pair) == original_hashes
    record.update(checks=records, source_unchanged=True, final_inventory=inventory(work),
                  seconds=time.monotonic()-started, completed_transform_calls=5,
                  transforms_per_second=5/(time.monotonic()-started), training_updates=0,
                  memory_peak='not_measured', gpu_peak='not_measured', gpu_used=False, status='passed')


def main():
    parser = argparse.ArgumentParser(description='单个真实病例的共享存储数值对照；不训练模型')
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    raw = args.raw.resolve(strict=True)
    output = args.output.resolve()
    if output == raw or output.is_relative_to(raw) or raw.is_relative_to(output):
        raise ValueError('检查输出与原始数据必须分开')
    cid = case_ids([args.case])[0]
    output.mkdir(parents=True, exist_ok=True)
    (output/'.gitignore').write_text('*\n!.gitignore\n', encoding='utf-8')
    record = dict(case_ids=[cid], scope='single_case_storage_equivalence_only', status='started')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix='work_', dir=output) as temporary:
            work = Path(temporary)
            try:
                check(raw, work, record, cid)
            finally:
                record['removed_work_inventory'] = inventory(work)
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
        record['total_seconds'] = time.monotonic()-started
        record['remaining_before_report'] = inventory(output)
        write_json(output/'acceptance.json', record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
