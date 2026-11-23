from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import threading
from time import monotonic
from zipfile import ZipFile

import psutil


ROOT = Path(__file__).resolve().parents[1]
TASK = Path(os.environ['AAOCA_TEST_ROOT']).resolve()
REPORT = ROOT / 'reports/checkpoint-zero-20261122.json'
sys.path.insert(0, str(ROOT / '.deps/torchio0196'))
sys.path.insert(0, str(ROOT / 'src'))


def main():
    if TASK.exists() or REPORT.exists():
        raise FileExistsError('本批目录或报告已存在，禁止追加重试')
    started = monotonic()
    deadline = datetime.now(timezone.utc) + timedelta(minutes=10)
    record = {'state': 'running', 'scope': 'zero_update_serialization_component',
              'production_train_resume_verified': False, 'image_cases': 0, 'updates': 0,
              'forward_calls': 0, 'gpu_operations': 0, 'phases': [], 'peak_rss_bytes': 0,
              'deadline_utc': deadline.isoformat()}
    TASK.mkdir(parents=True)
    (TASK / '.gitignore').write_text('*\n', encoding='utf-8')
    temporary = TASK / 'tmp'
    temporary.mkdir()
    for key in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[key] = str(temporary)
    stop = threading.Event()
    process = psutil.Process()

    def guard():
        while not stop.wait(0.05):
            rss = process.memory_info().rss
            record['peak_rss_bytes'] = max(record['peak_rss_bytes'], rss)
            memory = psutil.virtual_memory()
            if rss > 8 * 2**30 or memory.available < memory.total * 0.2 or datetime.now(timezone.utc) >= deadline:
                print(json.dumps({'event': 'hard_stop', 'rss': rss}), flush=True)
                os._exit(75)

    watcher = threading.Thread(target=guard, daemon=True)
    watcher.start()
    print(json.dumps({'event': 'started', 'pid': process.pid, 'deadline_utc': deadline.isoformat()}), flush=True)
    try:
        import random
        import numpy as np
        import torch
        from aaoca_reproduction.classification import LABEL_MAPS, build_model, load_weights, model_metadata, validate_model_metadata
        from aaoca_reproduction.records import RELEASE, identity, signature, write_json
        from aaoca_reproduction.training import restore_random, save_checkpoint, save_update_attempts

        torch.set_num_threads(2)

        def boundary(phase, checks):
            files = [p for p in TASK.rglob('*') if p.is_file()]
            if len(files) > 10 or sum(p.stat().st_size for p in files) > 2 * 2**30:
                raise OSError('本批产物预算超限')
            record['phases'].append({'phase': phase, 'seconds': monotonic() - started, 'checks': checks})
            write_json(REPORT, record)
            print(json.dumps({'event': 'phase_completed', 'phase': phase, 'seconds': monotonic() - started}), flush=True)
            if input().strip() != 'continue':
                raise RuntimeError('阶段边界停止新增计算')

        boundary('imports', {'cuda_initialized': torch.cuda.is_initialized()})
        member = 'classification_models/anomaly_detection/model_1/best_val_model.pt'
        with ZipFile(ROOT / 'data/author_weights/shiri-b134e621/classification_models.zip') as archive:
            if archive.getinfo(member).file_size != 682236342:
                raise ValueError('实际作者成员长度不符')
            archive.extract(member, TASK / 'data')
        path = TASK / 'data' / member
        boundary('one_real_member_extracted', {'bytes': path.stat().st_size})
        memory = psutil.virtual_memory()
        if memory.available - memory.total * 0.2 < 3 * 2**30:
            raise MemoryError('零更新序列化估计需额外3GiB主存余量，当前不足')
        model = build_model('aaoca_present')
        descriptor = {'path': str(path), 'task': 'aaoca_present', 'labels': LABEL_MAPS['aaoca_present'],
                      'author_release': RELEASE, 'source': 'official classification_models.zip/' + member,
                      'format': 'author_state_dict', 'member_index': 0, 'strategy': 'author_pretrained',
                      'experiment_name': 'official_public_anomaly_detection', 'legacy_fc_keys': False}
        initialization = load_weights(model, descriptor, 'aaoca_present')
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)
        assert not optimizer.state
        state = {'scope': 'component_only_not_train_resume', 'attempted_updates': 0, 'successful_updates': 0}
        run_id = identity({'scope': record['scope'], 'source': signature(path)})
        directory = TASK / 'course'
        directory.mkdir()
        checkpoint_path = directory / 'component-zero.pt'
        attempts = directory / 'update-attempts.json'
        boundary('real_model_and_empty_optimizer', {'optimizer_state_entries': len(optimizer.state)})
        save_update_attempts(attempts, run_id, 0)
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, state, run_id,
                        model_metadata('aaoca_present'), initialization)
        expected = (random.random(), float(np.random.random()), torch.rand(4))
        boundary('checkpoint_saved', {'bytes': checkpoint_path.stat().st_size, 'pending_absent': not checkpoint_path.with_suffix('.pt.pending').exists()})
        loaded = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        validate_model_metadata(loaded['model_meta'], 'aaoca_present')
        assert loaded['schema'] == 1 and loaded['identity'] == run_id and loaded['state'] == state
        assert loaded['initialization'] == initialization
        tensors = model.state_dict()
        assert tensors.keys() == loaded['model'].keys()
        assert all(torch.equal(tensor, loaded['model'][name]) for name, tensor in tensors.items())
        model.load_state_dict(loaded['model'], strict=True)
        restored_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.02)
        restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=2, eta_min=0)
        restored_optimizer.load_state_dict(loaded['optimizer'])
        restored_scheduler.load_state_dict(loaded['scheduler'])
        assert restored_optimizer.state_dict() == optimizer.state_dict()
        assert restored_scheduler.state_dict() == scheduler.state_dict()
        restore_random(loaded['random'], torch.device('cpu'))
        actual = (random.random(), float(np.random.random()), torch.rand(4))
        assert expected[0] == actual[0] and expected[1] == actual[1] and torch.equal(expected[2], actual[2])
        assert json.loads(attempts.read_text(encoding='utf-8')) == {'schema': 1, 'identity': run_id, 'attempted_updates': 0}
        assert not torch.cuda.is_initialized()
        boundary('zero_update_roundtrip', {'model_tensors_equal': len(tensors), 'optimizer_moments': 0,
                                          'optimizer_groups_equal': True, 'scheduler_equal': True,
                                          'CPU_RNG_equal': True, 'attempt_count': 0})
        record['state'] = 'passed'
    finally:
        stop.set()
        watcher.join()
        error = sys.exc_info()[1]
        if error is not None:
            record.update(state='failed', error=str(error))
        paths = list(TASK.rglob('*'))
        if TASK.resolve() != Path(os.environ['AAOCA_TEST_ROOT']).resolve() or any(p.is_symlink() or not p.resolve().is_relative_to(TASK.resolve()) for p in paths):
            raise ValueError('清理对象超出本批')
        files = [p for p in paths if p.is_file()]
        record['artifacts'] = [{'path': str(p.relative_to(TASK)), 'bytes': p.stat().st_size} for p in files]
        record.update(cleaned_files=len(files), cleaned_bytes=sum(p.stat().st_size for p in files))
        for p in files:
            p.unlink()
        for p in sorted((p for p in paths if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            p.rmdir()
        TASK.rmdir()
        record.update(seconds=monotonic() - started, root_absent=not TASK.exists(), vram_peak='unmeasured; no GPU')
        REPORT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'finished', 'state': record['state'], 'seconds': record['seconds'],
                          'cleaned_files': record['cleaned_files'], 'cleaned_bytes': record['cleaned_bytes']}), flush=True)


if __name__ == '__main__':
    main()
