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
REPORT = ROOT / 'reports/author-classification-20261122-r2.json'
sys.path.insert(0, str(ROOT / '.deps/torchio0196'))
sys.path.insert(0, str(ROOT / 'src'))


def main():
    if TASK.exists() or REPORT.exists():
        raise FileExistsError('本批目录或报告已存在，禁止自动重试')
    started = monotonic()
    deadline = datetime.now(timezone.utc) + timedelta(minutes=30)
    stop = threading.Event()
    record = {'state': 'running', 'deadline_utc': deadline.isoformat(), 'phases': [],
              'updates': 0, 'forward_limit': 6, 'gpu_operations': 0, 'peak_rss_bytes': 0}
    TASK.mkdir(parents=True)
    (TASK / '.gitignore').write_text('*\n', encoding='utf-8')
    temporary = TASK / 'course/tmp'
    temporary.mkdir(parents=True)
    for name in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[name] = str(temporary)
    print(json.dumps({'event': 'started', 'pid': os.getpid(), 'deadline_utc': deadline.isoformat()}), flush=True)
    process = psutil.Process()

    def guard():
        while not stop.wait(0.05):
            rss = process.memory_info().rss
            record['peak_rss_bytes'] = max(record['peak_rss_bytes'], rss)
            memory = psutil.virtual_memory()
            if (rss > 8 * 2**30 or memory.available < memory.total * 0.2
                    or datetime.now(timezone.utc) >= deadline):
                print(json.dumps({'event': 'hard_stop', 'rss': rss}), flush=True)
                os._exit(75)

    watcher = threading.Thread(target=guard, daemon=True)
    watcher.start()
    try:
        import torch
        from aaoca_reproduction.audit import load_frozen
        from aaoca_reproduction.classification import LABEL_MAPS
        from aaoca_reproduction.prediction import predict
        from aaoca_reproduction.preprocessing import prepare_classification
        from aaoca_reproduction.records import RELEASE, load_config, read_json, write_json

        config = load_config(ROOT / 'project.local.json')
        manifest = load_frozen(config)
        keys = read_json(ROOT / 'reports/classification-preprocessing-r2-20261121.json')['selected_case_ids']
        cases = [next(c for c in manifest['cases'] if c['case_id'] == key) for key in keys]
        if len(cases) != 2 or {c['dataset'] for c in cases} != {'aaoca', 'imagechd'} or any(c['role'] != 'predict' for c in cases):
            raise ValueError('只允许r2两例真实预测角色')
        config.update(data_root=TASK / 'data', course_root=TASK / 'course', classification_recipe='author_native_index',
                      cpu_threads=2, max_memory_bytes=8 * 2**30, host_memory_fraction=0.8,
                      max_work_bytes=3 * 2**30, max_work_files=30, diagnoses=None)
        config['prepared_inputs'] = {}
        record.update(selected_case_ids=keys, production_source='8cd4ffc', scope='two real cases; three author model_1 members')

        def boundary(phase, result):
            # 文件更新结束后核对产物，避免与原子改名及临时硬链接删除并发。
            files = [p for p in TASK.rglob('*') if p.is_file()]
            if len(files) > 30 or sum(p.stat().st_size for p in files) > 3 * 2**30:
                raise OSError('阶段产物超过本批批准预算')
            record['phases'].append({'phase': phase, 'seconds_total': monotonic() - started, 'result': result})
            write_json(REPORT, record)
            print(json.dumps({'event': 'phase_completed', 'phase': phase, 'seconds': monotonic() - started}), flush=True)
            # 主助手在每个阶段查询账户额度后显式发送continue。
            if input().strip() != 'continue':
                raise RuntimeError('阶段边界停止新增计算')

        write_json(REPORT, record)
        receipts = []
        for case in cases:
            directory = TASK / 'data' / case['case_id']
            receipts.append(prepare_classification(case, config, directory))
            config['prepared_inputs'][case['case_id']] = str(directory)
        boundary('preprocessing', receipts)

        groups = [('aaoca_present', 'anomaly_detection'), ('origin_left', 'origin_classification'), ('high_risk', 'risk_classification')]
        with ZipFile(ROOT / 'data/author_weights/shiri-b134e621/classification_models.zip') as archive:
            for task, group in groups:
                member = f'classification_models/{group}/model_1/best_val_model.pt'
                if archive.getinfo(member).file_size != 682236342:
                    raise ValueError('作者成员长度与实际目录证据不符')
                archive.extract(member, TASK / 'data/weights')
        boundary('three_members_extracted', {'members': 3, 'bytes': 2046709026})
        for task, group in groups:
            path = TASK / 'data/weights/classification_models' / group / 'model_1/best_val_model.pt'
            value = torch.load(path, map_location='cpu', weights_only=True)
            if not isinstance(value, dict) or not all(isinstance(t, torch.Tensor) for t in value.values()):
                raise ValueError('实物权重不是裸tensor state_dict')
            legacy = 'loss_func_layer.fc.weight' in value and 'loss_func_layer.fc.bias' in value
            del value
            descriptor = {'path': str(path), 'task': task, 'labels': LABEL_MAPS[task], 'author_release': RELEASE,
                          'source': 'official public classification_models.zip/' + group + '/model_1',
                          'format': 'author_state_dict', 'member_index': 0, 'strategy': 'author_pretrained',
                          'experiment_name': 'official_public_' + group, 'legacy_fc_keys': legacy}
            config['classification'] = {'task': task, 'device': 'cpu', 'prediction_case_ids': keys,
                                        'ensemble_mode': 'single_model', 'weights': [descriptor],
                                        'prediction_directory': str(TASK / 'course' / task), 'artifact_budget_bytes': 2 * 2**20}
            result = predict(config, manifest)
            if result['metrics']['n'] != 0 or result['unknown_or_inapplicable'] != 2 or result['paper_performance_validated']:
                raise AssertionError('未知临床参考不能成为已确认性能')
            boundary(task, result)
        record['state'] = 'passed'
    finally:
        stop.set()
        watcher.join()
        error = sys.exc_info()[1]
        if error is not None:
            record.update(state='failed', error=str(error))
        paths = list(TASK.rglob('*'))
        if TASK.resolve() != Path(os.environ['AAOCA_TEST_ROOT']).resolve() or any(p.is_symlink() or not p.resolve().is_relative_to(TASK.resolve()) for p in paths):
            raise ValueError('清理路径不属于本批')
        files = [p for p in paths if p.is_file()]
        record['artifacts_before_cleanup'] = [{'path': str(p.relative_to(TASK)), 'bytes': p.stat().st_size} for p in files]
        record['cleaned_files'] = len(files)
        record['cleaned_bytes'] = sum(p.stat().st_size for p in files)
        for p in files:
            p.unlink()
        for p in sorted((p for p in paths if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            p.rmdir()
        TASK.rmdir()
        record.update(seconds=monotonic() - started, temporary_root_absent=not TASK.exists(), vram_peak='unmeasured; CPU only')
        REPORT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'finished', 'state': record['state'], 'seconds': record['seconds'],
                          'cleaned_files': record['cleaned_files'], 'cleaned_bytes': record['cleaned_bytes']}), flush=True)


if __name__ == '__main__':
    main()
