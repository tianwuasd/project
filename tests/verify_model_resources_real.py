import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT/'tests'))

from verify_delivery_real import MetadataWorkspace
from verify_model_offline_real import source_config, model_config, signature
from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.models.offline import preprocess_model
from chd_segmentation.models.memory import case_estimates


def run(args):
    sys.path.insert(0, args.simulator)
    from common import clean_owned, files_under
    item = yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments'][0]
    assert item['case']['case_id'] == 'ct_1003'
    baseline = read_json(args.baseline)
    with MetadataWorkspace(args.work_root, args.report, clean_owned, files_under) as workspace:
        config, source, source_path = source_config(workspace, item, baseline['plan'])
        config['resources'].update(ram_fraction=.95, reserve_ram_bytes=1024**3)
        if args.q_generated:
            from chd_segmentation.delivery.configs import model_configs
            from chd_segmentation.models.workflow import options
            from chd_segmentation.models.offline import quick_check_model
            from chd_segmentation.models.execution import execution_config
            from chd_segmentation.models.memory import memory_profile
            generated = model_configs(workspace.root, PROJECT, dict(stunet=item['pretrained'],
                medsam2=PROJECT/'data/model_weights/MedSAM2_latest.pt', official=PROJECT/'vendor/MedSAM2',
                dependencies=PROJECT/'env/medsam2'))
            for stage in ('localizer', 'local'):
                raw = generated[f'configs/CHD1282-Q/20261115/{stage}.yaml']
                assert raw['pretrained'] is None
                raw.update(technical=True, patch=32, local_size=32, updates=2, device='cpu', parents={})
                raw['execution'].update(cpu_processes=1, ram_fraction=.95, reserve_ram_bytes=1024**3)
                path = workspace.root/(stage+'.json')
                write_json(path, raw)
                model = options(path)
                if args.q_generated == 'preflight':
                    assert quick_check_model(model, source, 'train')['ok']
                    result = dict(quick_check=True)
                else:
                    common = execution_config(model)
                    result = memory_profile(common, source, stage, [4])
                    assert result['checkpoint_reserve_bytes'] >= 64*1024**2
                workspace.results.append(dict(stage=stage, no_initialization_weight=True,
                    check=args.q_generated, result=result, optimizer_updates=0))
            return
        config['preprocessing'].update(block_bytes=256*1024, read_block_bytes=256*1024)
        write_json(workspace.root/'source-config.json', config)
        path, model = model_config(workspace, 'medsam2', str(PROJECT/'data/model_weights/MedSAM2_latest.pt'))
        model['execution'].update(ram_fraction=.95, reserve_ram_bytes=1024**3)
        write_json(path, model)
        prepared = preprocess_model(path, workspace.root/'source-config.json', source_path,
                                    workspace.root/'data', workspace.root/'course/native.json')
        from chd_segmentation.models.medsam2.native import NativeCases
        reader = NativeCases(prepared)
        entry = reader.get('ct_1003')
        assert signature(entry['image']) == baseline['medsam2']['image']
        assert signature(entry['label']) == baseline['medsam2']['label']
        assert entry['source_affine'] == baseline['medsam2']['source_affine']
        assert entry['gray_window'] == baseline['medsam2']['gray_window']
        entry.clear()
        reader.clear()
        del entry, reader
        gc.collect()
        log = Path(model['run_dir'])/'preprocess'
        records = [json.loads(line) for line in (log/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        blocks = [r for r in records if r['event'] == 'preprocess_blocks' and r.get('phase') == 'native_gray_window']
        assert blocks and blocks[-1]['block_total'] >= 2
        assert any(r['event'] == 'quick_check' and r['report']['ok'] for r in records)
        progress = read_json(log/'progress-preprocess.json')
        assert progress['done'] == progress['total'] == 1
        assert (log/'memory.json').is_file()
        pointers = deepcopy(prepared['storage']['cases'])
        again = preprocess_model(path, workspace.root/'source-config.json', source_path,
                                 workspace.root/'data', workspace.root/'course/reused.json')
        assert again == prepared and again['storage']['cases'] == pointers
        budget = dict(config, model='medsam2', model_stage='preprocess')
        estimates = case_estimates(budget, source, 'preprocess')
        assert estimates[0]['storage'] == 'bounded_disk_blocks'
        assert estimates[0]['workspace_bytes'] <= 2*1024**3+4*config['preprocessing']['block_bytes']
        workspace.results.append(dict(case_id='ct_1003', roi_xyz=item['roi_xyz'], optimizer_updates=0,
            image_and_labels_equal_frozen_baseline=True, native_blocks=blocks[-1]['block_total'],
            quick_check=True, shared_progress=True, actual_memory_snapshot=True,
            reused_without_rewriting=True, native_preprocess_estimates=estimates))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('spec', 'baseline', 'work-root', 'report'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--simulator', default='D:/projects/test')
    parser.add_argument('--q-generated', choices=('preflight', 'budget'))
    run(parser.parse_args())
