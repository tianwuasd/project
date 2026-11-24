import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import psutil
import yaml


def main():
    parser = argparse.ArgumentParser()
    for key in ('manifest', 'cohort', 'data-root', 'train-case', 'val-case',
                'official-root', 'weights', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    deadline = time.time()+1200 if args.deadline is None else args.deadline
    if not time.time() < deadline <= time.time()+1200:
        raise ValueError('固定截止必须位于未来20分钟以内')
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project/'src'))
    from chd_segmentation.artifacts.store import read_json, write_json, file_digest
    from chd_segmentation.models.medsam3.config import load_config
    from chd_segmentation.models.medsam3.training_data import entries, read_slice, datapoint, stream
    root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError('局部测试目录已存在，禁止覆盖')
    root.mkdir(parents=True)
    (root/'temporary').mkdir()
    for key, name in dict(TMP='temporary', TEMP='temporary', TMPDIR='temporary',
            MPLCONFIGDIR='matplotlib', TRITON_CACHE_DIR='triton',
            TORCHINDUCTOR_CACHE_DIR='inductor', TORCH_HOME='torch',
            HF_HOME='huggingface', CUDA_CACHE_PATH='cuda').items():
        os.environ[key] = str(root/name)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                      PYTHONIOENCODING='utf-8')
    began = time.monotonic()
    report = dict(state='running', scope=dict(cases=2, slices_per_case=1, classes=[5], updates=2),
        stages=[], peak_group_rss_bytes=0, sampled_gpu_used_bytes=0,
        peak_artifact_bytes=0, peak_artifact_files=0, cases_read=0, completed_updates=0)

    def budget():
        directories, count, size = [root], 0, 0
        while directories:
            with os.scandir(directories.pop()) as listing:
                for entry in listing:
                    if entry.is_symlink():
                        raise ValueError('局部测试目录不能包含链接')
                    if entry.is_dir(follow_symlinks=False):
                        directories.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):
                        count += 1
                        size += entry.stat(follow_symlinks=False).st_size
        report['peak_artifact_bytes'] = max(size, report['peak_artifact_bytes'])
        report['peak_artifact_files'] = max(count, report['peak_artifact_files'])
        if time.time() >= deadline or size > 4*2**30 or count > 500:
            raise RuntimeError('达到局部测试时间、磁盘或文件数量上限')

    def stage(name, arguments):
        budget()
        started = time.monotonic()
        with (root/(name+'.log')).open('w', encoding='utf-8') as output:
            process = subprocess.Popen([sys.executable, '-B', str(project/'run.py'),
                'models', 'medsam3', *arguments], stdout=output, stderr=subprocess.STDOUT)
            try:
                last = 0.
                while process.poll() is None:
                    rows = [p.info for p in psutil.process_iter(['pid','ppid','memory_info'])]
                    members = {process.pid}
                    for _ in range(len(rows)):
                        added = {r['pid'] for r in rows if r['ppid'] in members}-members
                        if not added:
                            break
                        members.update(added)
                    rss = sum(r['memory_info'].rss for r in rows
                              if r['pid'] in members and r['memory_info'] is not None)
                    report['peak_group_rss_bytes'] = max(rss, report['peak_group_rss_bytes'])
                    if rss > 16*2**30:
                        raise MemoryError('局部测试进程组超过16GiB')
                    if time.monotonic()-last >= 3:
                        budget()
                        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                            '--format=csv,noheader,nounits'], text=True, timeout=10)
                        used = max(int(r[0])*2**20 for r in csv.reader(gpu.splitlines()))
                        report['sampled_gpu_used_bytes'] = max(used, report['sampled_gpu_used_bytes'])
                        last = time.monotonic()
                    time.sleep(.5)
                report['stages'].append(dict(name=name, seconds=time.monotonic()-started,
                                             exit_code=process.returncode))
                write_json(root/'report.json', report)
                if process.returncode:
                    raise RuntimeError(name+'失败，见对应日志')
            finally:
                if process.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], check=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                    process.wait(timeout=10)

    try:
        mark = time.monotonic()
        manifest = read_json(args.manifest)
        names = [args.train_case, args.val_case]
        selected = [r for r in manifest['cases'] if r['case_id'] in names]
        assert len(selected) == len(set(names)) == 2
        data_root = Path(args.data_root).resolve(strict=True)
        records, slices = [], {}
        for row in selected:
            for kind in ('image','label'):
                path = (data_root/row['preferred_files'][kind]).resolve(strict=True)
                assert path.is_relative_to(data_root) and path.stat().st_size == row['source_bytes'][kind]
                assert file_digest(path) == row['sha256'][kind]
                row[kind] = path.relative_to(data_root).as_posix()
            image = nib.load(data_root/row['image'])
            slices[row['case_id']] = [image.shape[2]//2]
            records.append(dict(case_id=row['case_id'], shape=list(image.shape), slice=slices[row['case_id']][0],
                spacing_xyz=row['spacing_xyz'], spacing_evidence=row['spacing_evidence'], sha256=row['sha256']))
        manifest.update(cases=selected, data_root=str(data_root), binding_required=False)
        write_json(root/'bound_manifest.json', manifest)
        report['inputs'] = records
        base = yaml.safe_load((project/'configs/models/medsam3.yaml').read_text(encoding='utf-8'))
        official, weights = Path(args.official_root).resolve(), Path(args.weights).resolve()
        base.update(manifest=str(root/'bound_manifest.json'), cohort=str(Path(args.cohort).resolve()),
            data_root=str(data_root), work_root=str(root/'work'), registry_root=str(root/'resources'),
            python=sys.executable, official_root=str(official),
            official_config=str(official/'configs/full_lora_config.yaml'),
            base_weights=str(weights/'sam3.pt'), lora_weights=str(weights/'best_lora_weights.pt'),
            bpe_path=str(official/'sam3/assets/bpe_simple_vocab_16e6.txt.gz'), precision='bf16',
            orientation_evidence='沿用CHD1282的RAS研究约定；保留原始affine，仅按共享数组索引核对',
            intensity_evidence='沿用CHD1282缩放后NIfTI为HU的项目假设，未认证扫描校准；来源为冻结迁移清单',
            coverage='selected', slices=slices, fixed_ram_bytes=8*2**30, workspace_bytes=512*2**20,
            estimated_vram_bytes=6*2**30, worker_timeout_seconds=1200)
        base['resources'].update(cpu_threads=4, wait_seconds=90, reserve_ram_bytes=4*2**30,
            vram_reserve_bytes=2**30, memory_limit_bytes=16*2**30, share=True)
        base_path = root/'base.yaml'
        base_path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding='utf-8')
        training = dict(base_config='base.yaml', run_dir=str(root/'work/train'),
            train_cases=[args.train_case], val_cases=[args.val_case], classes=[5], updates=2,
            validate_every=1, warmup_steps=0, learning_rate=.00005, seed=42, workers=1,
            prefetch=2, prepared=None, max_seconds=1200, ram_limit_bytes=16*2**30, vram_limit_bytes=7*2**30)
        config = load_config(base_path)
        samples = entries(config, training)
        from chd_segmentation.models.medsam3.weights import import_official
        from chd_segmentation.models.medsam3.adapter import input_transform
        from torch import set_num_threads
        set_num_threads(4)
        official_api, _, _ = import_official(config)
        transform = input_transform(official_api)
        prepared, preprocessed = [], []
        for role, items in samples.items():
            for item in items:
                start = time.monotonic()
                item, rgb, labels = read_slice(item, config)
                path = root/(item['row']['case_id']+'.npz')
                np.savez(path, rgb=rgb, labels=labels, valid=np.ones(labels.shape, dtype=bool))
                prepared.append(dict(case_id=item['row']['case_id'], slice=item['slice'],
                    file=path.name, source_stamp=item['row']['source_stamp']))
                stored = dict(item, prepared=str(path), prepared_stamp=[path.stat().st_size,path.stat().st_mtime_ns])
                _, restored_rgb, restored_labels = read_slice(stored, config)
                assert np.array_equal(rgb, restored_rgb) and np.array_equal(labels, restored_labels)
                point = datapoint(config, transform, (item, rgb, labels))
                batch = official_api.collate_fn_api([point], dict_key='input', with_seg_masks=True)['input']
                assert tuple(batch.img_batch.shape) == (1,3,1008,1008)
                assert batch.find_text_batch == ['aorta']
                assert batch.find_targets[0].segments.shape[-2:] == (1008,1008)
                preprocessed.append(dict(role=role, case_id=item['row']['case_id'],
                    slice=item['slice'], shape=list(rgb.shape), seconds=time.monotonic()-start,
                    foreground_pixels=int((labels == 5).sum()), prepared_exact=True,
                    native_input_shape=list(batch.img_batch.shape), text=batch.find_text_batch,
                    target_boxes=int(batch.find_targets[0].num_boxes.sum())))
        write_json(root/'prepared.json', dict(format='medsam3_training_slices_v1', samples=prepared))
        training['prepared'] = 'prepared.json'
        supplied = entries(config, dict(training, prepared=str(root/'prepared.json')))
        with stream(supplied['train']+supplied['val'], config, 1, 2, deadline) as queue:
            consumed = [item['row']['case_id'] for item, _, _ in queue]
        assert consumed == names
        report['bounded_reader'] = dict(workers=1, prefetch=2, cases=consumed)
        train_path = root/'training.yaml'
        train_path.write_text(yaml.safe_dump(training), encoding='utf-8')
        report.update(cases_read=2, preprocessing=preprocessed)
        report['stages'].append(dict(name='input_and_preprocessing', seconds=time.monotonic()-mark, exit_code=0))
        write_json(root/'report.json', report)
        stage('check_training', ['check-training','--config',str(train_path)])
        from chd_segmentation.models.medsam3.runtime import request
        from chd_segmentation.acceleration.registry import capacity_key
        native_request = request(dict(config, training=training), root/'work/train')
        report['capacity_identity'] = capacity_key(dict(resources=native_request.resources,
            training=dict(precision=config['precision'])), native_request.resources['gpu_spec'], 1)
        if args.prepare_only:
            report['state'] = 'preprocessing_passed'
            return
        limit = ['--deadline',str(deadline)]
        stage('train_first', ['train','--config',str(train_path),'--stop-after','1',*limit])
        import torch
        first = torch.load(root/'work/train/latest.pt', map_location='cpu', weights_only=True)
        assert first['step'] == 1 and first['optimizer']['state']
        initial = torch.load(weights/'best_lora_weights.pt', map_location='cpu', weights_only=True)
        delta1 = sum((first['lora'][k]-v).abs().sum().item() for k,v in initial.items())
        assert delta1 > 0
        del initial
        stage('train_resume', ['train','--config',str(train_path),'--resume','--stop-after','1',*limit])
        second = torch.load(root/'work/train/latest.pt', map_location='cpu', weights_only=True)
        assert second['step'] == 2 and second['optimizer']['state']
        delta2 = sum((second['lora'][k]-v).abs().sum().item() for k,v in first['lora'].items())
        assert delta2 > 0 and all(float(v['step']) == 2 for v in second['optimizer']['state'].values())
        report.update(completed_updates=2, lora_delta_first=delta1, lora_delta_resume=delta2,
                      optimizer_states=len(second['optimizer']['state']), checkpoint_identity=second['identity'])
        del first, second
        stage('validate', ['validate','--config',str(train_path),'--checkpoint','best',*limit])
        base['lora_weights'] = str(root/'work/train/best_lora_weights.pt')
        prediction_path = root/'prediction.yaml'
        prediction_path.write_text(yaml.safe_dump(base, allow_unicode=True), encoding='utf-8')
        prediction = root/'work/prediction'
        stage('predict', ['predict','--config',str(prediction_path),'--case',args.val_case,'--output',str(prediction)])
        stage('evaluate', ['evaluate','--config',str(prediction_path),'--case',args.val_case,'--output',str(prediction)])
        report.update(state='passed', training=read_json(root/'work/train/training.json'),
            validation=read_json(root/'work/train/validation.json'),
            prediction=read_json(prediction/'summary.json'), evaluation=read_json(prediction/'evaluation.json'))
    finally:
        if report['state'] not in ('passed','preprocessing_passed'):
            report['state'] = 'failed'
            report['failure'] = str(sys.exc_info()[1])
        report['seconds'] = time.monotonic()-began
        write_json(root/'report.json', report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
