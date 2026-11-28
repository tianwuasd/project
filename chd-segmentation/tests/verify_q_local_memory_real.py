import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
from time import perf_counter
from unittest import TestCase

import nibabel as nib
import numpy as np
import psutil
import torch

PROJECT = Path(__file__).resolve().parents[1]
TEST_ROOT = Path(os.environ.get('CHD_Q_MEMORY_TEST_ROOT','D:/projects/back/t17/chd-q-local-memory-20261128'))
TEST_ROOT.mkdir(parents=True, exist_ok=True)
(TEST_ROOT/'system-temp').mkdir(exist_ok=True)
os.environ.update(CHD_RUNTIME_DIR=str(TEST_ROOT/'runtime'), PYTHONDONTWRITEBYTECODE='1',
                  TEMP=str(TEST_ROOT/'system-temp'), TMP=str(TEST_ROOT/'system-temp'))
if not os.environ.get('CHD_CODE_SNAPSHOT', '').startswith(str(TEST_ROOT/'runtime')):
    os.environ.pop('CHD_CODE_SNAPSHOT', None)
sys.path.insert(0, str(PROJECT/'src'))

from chd_segmentation.artifacts.store import read_json, write_json, file_digest
from chd_segmentation.config import DEFAULTS
from chd_segmentation.data.dataset import CaseDataset
from chd_segmentation.models.execution import execution_config
from chd_segmentation.models.memory import case_estimates, memory_profile
from chd_segmentation.models.q_route1.adapter import QAdapter
from chd_segmentation.models.q_route1.candidates import select_candidates
from chd_segmentation.models.q_route1.local_cache import LocalCache, FrozenInputs, disk_plan, _readers, _remove
from chd_segmentation.models.q_route1.training_cache import TrainingCache
from chd_segmentation.models.q_route1.sampling import CandidateSampler
from chd_segmentation.models.q_route1.geometry import sample_context
from chd_segmentation.models.workflow import options, save_state, load_state, training_names
from chd_segmentation.training.checkpoint import tensors_equal
from verify_q_low_vram_real import prepare


def bundles(root, prepared, pretrained):
    path = root/'model.json'
    roles = {stage:dict(train=['ct_1003'], validation=['ct_1012'], test=[])
             for stage in ('base', 'localizer', 'local')}
    write_json(path, dict(model='q_route1', run_dir=str(root/'models'), work_root=str(root),
        technical=True, updates=2, patch=64, local_size=32, accumulation=1,
        technical_roles=roles, pretrained=pretrained, cpu_preparation=dict(enabled=False),
        training_cache=dict(enabled=True, local_bytes=256*2**20),
        execution=dict(workers=0, validation_block_bytes=8*2**20),
        logging=dict(heartbeat_seconds=15, case_details=False)))
    config = options(path)
    torch.manual_seed(20261128)
    base = QAdapter(config, prepared, 'base')
    locator = QAdapter(config, prepared, 'localizer')
    parents = {stage:dict(model=b.model.state_dict(), spec=b.spec, stage=stage, prepared=prepared,
        scope='technical', config=config, training_cases=training_names(config, prepared, stage),
        initialization_sha256=None, identity=stage, best=0., history=[])
        for stage, b in (('base', base), ('localizer', locator))}
    local_config = dict(config, parents=dict(base='retained-in-memory', localizer='retained-in-memory'))
    local = QAdapter(local_config, prepared, 'local', parent_states=parents)
    del base, locator
    gc.collect()
    return local, local_config


def dense(bundle, entry):
    with torch.no_grad():
        probability = bundle.base_probability(entry)
        located = bundle.locate(entry)
        candidates = select_candidates(located['heatmap'][0].float().cpu().numpy(),
                                        entry['label'].shape, (4, 4, 8), .2, .8)
        return probability, located['features'].float().cpu(), candidates


def close_errors(a, b, atol=5e-5, rtol=5e-4):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return float((a-b).detach().abs().max())


def gradients(bundle):
    return {key:p.grad.detach().clone() for key,p in bundle.model.named_parameters() if p.grad is not None}


def compare(args, root, prepared):
    bundle, config = bundles(root, prepared, args.pretrained)
    dataset = CaseDataset(prepared, dict(cache_bytes=0))
    result = dict(cases=[], optimizer_updates=0)
    saved_corruption = deepcopy(bundle.corruption_rng.bit_generator.state)
    for name in ('ct_1003', 'ct_1012'):
        entry = dataset.get(name)
        began = perf_counter()
        expected_probability, expected_features, expected_candidates = dense(bundle, entry)
        dense_seconds = perf_counter()-began
        began = perf_counter()
        probability, features, candidates = bundle.frozen(entry)
        disk_seconds = perf_counter()-began
        errors = dict(probability=close_errors(probability, expected_probability),
                      features=close_errors(features, expected_features))
        assert [(c.start,c.source,c.task) for c in candidates] == [
               (c.start,c.source,c.task) for c in expected_candidates]
        np.testing.assert_allclose([c.score for c in candidates], [c.score for c in expected_candidates],
                                   atol=2e-5, rtol=5e-4)
        for candidate in candidates[:2]:
            expected_crop = bundle.crop(entry, candidate, expected_probability, parallel=False)
            actual_crop = bundle.crop(entry, candidate, probability, parallel=False)
            close_errors(actual_crop[0], expected_crop[0])
            assert torch.equal(actual_crop[1], expected_crop[1])
            np.testing.assert_array_equal(actual_crop[2], expected_crop[2])
            close_errors(sample_context(features, candidate.center, samples=4),
                         sample_context(expected_features, candidate.center, samples=4))
        if name == 'ct_1003':
            original_frozen = bundle.current_frozen
            bundle.current_frozen = expected_probability, expected_features, expected_candidates
            bundle.prepare_training(dataset, ['ct_1003'])
            expected_records = [asdict(value) for value in bundle.sampler.records]
            bundle.current_frozen = original_frozen
            bundle.prepare_training(dataset, ['ct_1003'])
            records = bundle.sampler.records
            assert [asdict(value) for value in records] == expected_records
            expected_sampler = CandidateSampler(records, config['seed'])
            assert bundle.sampler.state_dict() == expected_sampler.state_dict()
            entry = bundle.sample(dataset, ['ct_1003'], np.random.default_rng(config['seed']))
            original_frozen = bundle.current_frozen
            bundle.current_frozen = expected_probability, expected_features, expected_candidates
            bundle.corruption_rng.bit_generator.state = deepcopy(saved_corruption)
            bundle.model.zero_grad(set_to_none=True)
            expected_loss = bundle.loss(entry, np.random.default_rng(config['seed']), 0)
            expected_loss.backward()
            expected_gradients = gradients(bundle)
            bundle.current_frozen = original_frozen
            bundle.corruption_rng.bit_generator.state = deepcopy(saved_corruption)
            bundle.model.zero_grad(set_to_none=True)
            actual_loss = bundle.loss(entry, np.random.default_rng(config['seed']), 0)
            actual_loss.backward()
            actual_gradients = gradients(bundle)
            errors['loss'] = close_errors(actual_loss, expected_loss, atol=2e-5)
            errors['gradient'] = max(close_errors(actual_gradients[k], v, atol=8e-5, rtol=2e-3)
                                     for k,v in expected_gradients.items())
            state = deepcopy(bundle.model.state_dict())
            if args.updates:
                ledger = Path('D:/projects/back/sync-reports/chd-q-local-memory-20261128/updates.json')
                used = read_json(ledger)['updates'] if ledger.exists() else 0
                if used+2 > 2:
                    raise RuntimeError('本批次最多两次临时优化器更新，禁止重复使用额度')
                write_json(ledger, dict(updates=used+2))
                optimizer = bundle.optimizer()
                optimizer.step()
                actual_state = deepcopy(bundle.model.state_dict())
                actual_optimizer = deepcopy(optimizer.state_dict())
                bundle.model.load_state_dict(state)
                baseline_optimizer = bundle.optimizer()
                for key, parameter in bundle.model.named_parameters():
                    parameter.grad = expected_gradients.get(key)
                baseline_optimizer.step()
                errors['updated_model'] = max(close_errors(actual_state[k], value, atol=3e-5, rtol=5e-4)
                                              for k,value in bundle.model.state_dict().items())
                bundle.model.load_state_dict(actual_state)
                optimizer.load_state_dict(actual_optimizer)
                saved = dict(model=bundle.model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(), extra=bundle.extra_state())
                save_state(root/'local-latest.pt', saved)
                loaded = load_state(root/'local-latest.pt')
                assert tensors_equal(saved['model'], loaded['model'])
                assert tensors_equal(saved['optimizer'], loaded['optimizer'])
                np.testing.assert_equal(saved['numpy_rng'], loaded['numpy_rng'])
                assert saved['python_rng'] == loaded['python_rng']
                assert torch.equal(saved['torch_rng'], loaded['torch_rng'])
                bundle.model.load_state_dict(loaded['model'])
                optimizer.load_state_dict(loaded['optimizer'])
                bundle.restore_extra(loaded['extra'])
                assert tensors_equal(bundle.model.state_dict(), loaded['model'])
                assert tensors_equal(optimizer.state_dict(), loaded['optimizer'])
                assert tensors_equal(bundle.extra_state(), loaded['extra'])
                result.update(optimizer_updates=2, checkpoint_readback=True)
            if (root/'local-latest.pt').exists():
                loaded = load_state(root/'local-latest.pt')
                bundle.model.load_state_dict(loaded['model'])
                optimizer = bundle.optimizer()
                optimizer.load_state_dict(loaded['optimizer'])
                bundle.restore_extra(loaded['extra'])
                torch.set_rng_state(loaded['torch_rng'])
                random.setstate(loaded['python_rng'])
                np.random.set_state(loaded['numpy_rng'])
                assert tensors_equal(bundle.model.state_dict(), loaded['model'])
                assert tensors_equal(optimizer.state_dict(), loaded['optimizer'])
                assert tensors_equal(bundle.extra_state(), loaded['extra'])
                assert torch.equal(torch.get_rng_state(), loaded['torch_rng'])
                assert random.getstate() == loaded['python_rng']
                np.testing.assert_equal(np.random.get_state(), loaded['numpy_rng'])
                result['checkpoint_readback'] = True
        path = bundle.frozen_store.path
        peer = LocalCache(bundle.training_cache).acquire(entry, lambda *_: (_ for _ in ()).throw(
            AssertionError('同一冻结结果应直接复用')))
        assert peer.path == path
        bundle.clear_frozen()
        peer.release()
        peer.close()
        if path.suffix == '.case':
            assert path.exists()
        result['cases'].append(dict(case_id=name, shape=list(entry['label'].shape), errors=errors,
                                    candidates=len(candidates), dense_seconds=dense_seconds,
                                    disk_seconds=disk_seconds))
        del expected_probability, expected_features, probability, features
        gc.collect()
    bundle.clear_frozen()
    cache = LocalCache(bundle.training_cache)
    for fault in ('entry-pending', 'work-owner-pending'):
        cache.identity['recovery_check'] = fault
        def interrupted(destination, key):
            if fault == 'entry-pending':
                array = np.lib.format.open_memmap(destination/'probability.npy', mode='w+',
                    dtype=np.float32, shape=(1,8,8,8))
                array[:] = np.asarray(entry['image'][:, :8, :8, :8])
                array.flush()
                array._mmap.close()
                (destination/'entry.json.pending').write_text('{', encoding='utf-8')
            else:
                (destination/'work').mkdir()
                (destination/'work'/'owner.json.pending').write_text('{', encoding='utf-8')
            raise InterruptedError('检查原子元数据未完成时的真实目录清理')
        with TestCase().assertRaises(InterruptedError):
            cache.acquire(entry, interrupted)
        assert not list(cache.path.glob('*.pending'))
        assert not list(cache.path.glob('*.reserve.json'))
    result['interrupted_metadata_cleanup'] = True
    common = execution_config(config)
    common['technical'], common['model_stage'] = prepared['technical'], 'local'
    common['resources'] = dict(DEFAULTS['resources'], **common['resources'])
    records = case_estimates(common, prepared, 'local')
    profile = memory_profile(common, prepared, 'local', [2,2], records)
    assert max(profile['rank_case_bytes']) < 8*2**30
    result['local_profile'] = {key:value for key,value in profile.items() if key != 'records'}
    native = []
    manifest = read_json(args.manifest)
    for name in ('ct_1003', 'ct_1012'):
        row = next(r for r in manifest['cases'] if r['case_id'] == name)
        shape = nib.load(Path(args.data_root)/row['image']).shape
        native.append(dict(case_id=name, native_shape_xyz=list(shape),
            disk_projection=disk_plan(shape[::-1], config['patch'])))
    result['native_header_only'] = native
    return result


def measure(args, root, prepared, mode):
    bundle, _ = bundles(root, prepared, args.pretrained)
    entry = CaseDataset(prepared, dict(cache_bytes=0)).get('ct_1003')
    process = psutil.Process()
    baseline = process.memory_info().rss
    peaks = dict(rss=baseline, disk_bytes=0, files=0)
    stop = threading.Event()

    def sample():
        while not stop.wait(.025):
            peaks['rss'] = max(peaks['rss'], process.memory_info().rss)
            files = [p for p in root.rglob('*') if p.is_file()]
            peaks['disk_bytes'] = max(peaks['disk_bytes'], sum(p.stat().st_size for p in files if p.exists()))
            peaks['files'] = max(peaks['files'], len(files))

    executor = ThreadPoolExecutor(max_workers=1)
    monitor = executor.submit(sample)
    began = perf_counter()
    try:
        value = dense(bundle, entry) if mode == 'dense' else bundle.frozen(entry)
        score = float(value[0][0,0,:8,:8,:8].sum())
        peaks['rss'] = max(peaks['rss'], process.memory_info().rss)
    finally:
        stop.set()
        monitor.result()
        executor.shutdown()
        bundle.clear_frozen()
    return dict(mode=mode, seconds=perf_counter()-began, baseline_rss_bytes=baseline,
        peak_rss_bytes=peaks['rss'], incremental_peak_bytes=peaks['rss']-baseline,
        disk_peak_bytes=peaks['disk_bytes'], files_peak=peaks['files'], sample_score=score,
        shape=list(entry['label'].shape), optimizer_updates=0, vram_peak_bytes=None)


def storage_checks(root, prepared):
    config = options(root.parent/'compare'/'model.json')
    store = LocalCache(TrainingCache(config, prepared, 'local'))
    source = next((root.parent/'compare'/'q-training-cache'/'local-blocks').glob('*.case'))
    store.path = source.parent
    key = read_json(source/'entry.json')['key']
    store.enabled = False
    assert store._existing(key) is None
    store.enabled, store.limit = True, 0
    assert store._existing(key) is None
    store.limit = 1
    assert store._existing(key) is None
    store.limit = 256*2**20
    with store.lock():
        existing = store._existing(key)
    assert existing is not None
    reader = FrozenInputs(store, *existing)
    assert torch.isfinite(reader.probability[0, :, :8, :8, :8]).all()
    reader.close()
    active = source/('reader-'+str(os.getpid())+'-atomic-check.json.pending')
    active.write_text('{', encoding='utf-8')
    assert active in _readers(source)
    active.unlink()
    dead_pid = 2**31-1
    assert not psutil.pid_exists(dead_pid)
    abandoned = source/('reader-'+str(dead_pid)+'-atomic-check.json.pending')
    abandoned.write_text('{', encoding='utf-8')
    _readers(source)
    assert not abandoned.exists()
    for fault in ('entry-pending', 'work-owner-pending'):
        path = root/fault
        path.mkdir()
        write_json(path/'owner.json', dict(key=key))
        if fault == 'entry-pending':
            (path/'entry.json.pending').write_text('{', encoding='utf-8')
        else:
            (path/'work').mkdir()
            (path/'work'/'owner.json.pending').write_text('{', encoding='utf-8')
        _remove(path, key)
        assert not path.exists()
    return dict(state='passed', cache_controls=True, mapped_reader=True,
                partial_reader_protected=True, interrupted_metadata_cleanup=True, optimizer_updates=0)


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'report', 'manifest', 'data-root', 'pretrained'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--mode', choices=('all','compare','dense','disk','storage'), default='all')
    parser.add_argument('--updates', type=int, choices=(0,2), default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root).resolve()
    permitted = Path('D:/projects/back/t17/chd-q-local-memory-20261128').resolve()
    if root != permitted and not root.is_relative_to(permitted):
        raise ValueError('检查目录超出本次授权范围')
    root.mkdir(parents=True, exist_ok=True)
    (permitted/'.gitignore').write_text('*\n!.gitignore\n', encoding='utf-8')
    if args.mode == 'all':
        hashes = {str(p):file_digest(p) for p in [Path(args.pretrained), *[
            Path(args.data_root)/(name+suffix) for name in ('ct_1003','ct_1012')
            for suffix in ('_image.nii.gz','_label.nii.gz')]]}
        prepared_path = root/'prepared.json'
        if not prepared_path.exists():
            prepared = prepare(root, args)
            write_json(prepared_path, prepared)
        result = dict(state='running', source_hashes=hashes, optimizer_updates=0)
        write_json(args.report, result)
        for mode in ('compare', 'dense', 'disk'):
            output = root/(mode+'.json')
            command = [sys.executable, '-B', str(Path(__file__)), '--root', str(root/mode),
                '--report', str(output), '--manifest', args.manifest, '--data-root', args.data_root,
                '--pretrained', args.pretrained, '--mode', mode]
            if mode == 'compare':
                command.extend(['--updates', str(args.updates)])
            subprocess.run(command, check=True)
            result[mode] = read_json(output)
            result['optimizer_updates'] += result[mode].get('optimizer_updates', 0)
            write_json(args.report, result)
        assert result['optimizer_updates'] <= 2
        ledger = Path(args.report).parent/'updates.json'
        if ledger.exists():
            result['optimizer_updates'] = read_json(ledger)['updates']
        assert all(file_digest(p) == value for p,value in hashes.items())
        result.update(state='passed', original_files_unchanged=True)
    else:
        prepared = read_json(root.parent/'prepared.json')
        if args.mode == 'storage':
            result = storage_checks(root, prepared)
        else:
            result = compare(args, root, prepared) if args.mode == 'compare' else measure(args, root, prepared, args.mode)
    write_json(args.report, result)
    print(json.dumps(dict(mode=args.mode, state=result.get('state','passed'),
                         report=args.report, updates=result.get('optimizer_updates',0))))


if __name__ == '__main__':
    main()
