import argparse
from copy import deepcopy
import gc
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT/'tests'))

from verify_delivery_real import MetadataWorkspace
from verify_model_offline_real import source_config, model_config
from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.models.offline import preprocess_model
from chd_segmentation.models.workflow import options, train, predict


def run(args):
    sys.path.insert(0, args.simulator)
    from common import clean_owned, files_under
    item = yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments'][0]
    assert item['case']['case_id'] == 'ct_1003'
    if args.model == 'q-inspect':
        import nibabel as nib
        from chd_segmentation.models.q_route1.adapter import allowed
        from chd_segmentation.models.q_route1.targets import build_heatmap_targets
        with MetadataWorkspace(args.work_root, args.report, clean_owned, files_under) as workspace:
            for width in (32, 96):
                roi = [slice((begin+end)//2-width//2, (begin+end)//2+width//2) for begin,end in item['roi_xyz']]
                label = np.asarray(nib.load(item['case']['label']).dataobj[tuple(roi)]).transpose(2, 1, 0)
                support = allowed(label, item['case']['unknown_classes'], item['dataset'])
                target, mask = build_heatmap_targets(support, label>=0, item['case']['spacing_xyz'][::-1])
                count = mask.sum((0,2,3,4)).tolist()
                assert bool(sum(count)) == (width == 96)
                workspace.results.append(dict(case_id='ct_1003', roi_width=width, supervised_cells=count,
                    target_positive_cells=((target>0)&mask).sum((0,2,3,4)).tolist(),
                    source_label_sha256=item['case']['sha256']['label'], optimizer_updates=0,
                    criterion='至少一个真实任务保留完整8毫米可靠邻域；不修改标签、监督掩膜或损失'))
        return
    if args.model == 'q':
        item = deepcopy(item)
        item['roi_xyz'] = [[(begin+end)//2-48, (begin+end)//2+48] for begin,end in item['roi_xyz']]
    baseline = read_json(args.baseline)
    model_name = 'q_route1' if args.model == 'q' else 'medsam2'
    with MetadataWorkspace(args.work_root, args.report, clean_owned, files_under) as workspace:
        config, source, source_path = source_config(workspace, item, None if args.model == 'q' else baseline['plan'])
        config['resources'].update(ram_fraction=.95, reserve_ram_bytes=1024**3)
        config['preprocessing'].update(block_bytes=256*1024, read_block_bytes=256*1024)
        write_json(workspace.root/'source-config.json', config)
        weight = item['pretrained'] if args.model == 'q' else str(PROJECT/'data/model_weights/MedSAM2_latest.pt')
        path, model = model_config(workspace, model_name, weight)
        model['run_dir'] = str(workspace.root/'course'/('ImageCHD-Q' if args.model == 'q' else 'ImageCHD-MedSAM2'))
        model['execution'].update(ram_fraction=.95, reserve_ram_bytes=1024**3,
            cpu_processes=2 if args.model == 'q' else 1, workers=0, prefetch=1,
            readers=1, max_readers=1, batch_layout=[1, 3] if args.model == 'q' else None)
        model['max_candidates'] = 1
        write_json(path, model)
        prepared_path = workspace.root/'course/offline.json'
        prepared = preprocess_model(path, workspace.root/'source-config.json', source_path,
                                    workspace.root/'data', prepared_path)
        record = dict(case_id='ct_1003', roi_xyz=item['roi_xyz'], model=model_name,
                      stage_results=[], full_medsam2_trainer_performed=False)
        workspace.results.append(record)
        if args.model == 'q':
            for stage in ('base', 'localizer', 'local'):
                if stage == 'local':
                    model['parents'] = {name: str(Path(model['run_dir'])/name/'latest.pt') for name in ('base', 'localizer')}
                    write_json(path, model)
                applied = options(path)
                first = train(applied, prepared_path, stage, stop_after=1)
                assert first['update'] == 1 and first['world_size'] == 2 and first['batch_layout'] == [1, 3]
                second = train(applied, prepared_path, stage, resume=True, stop_after=2)
                assert second['update'] == 2 and second['restored_exact'] and second['replicas_equal']
                progress = read_json(Path(model['run_dir'])/stage/'progress-train.json')
                assert progress['done'] == progress['total'] == 2 and progress['unit'] == 'updates'
                assert (Path(model['run_dir'])/stage/'memory.json').is_file()
                memory = read_json(Path(model['run_dir'])/stage/'memory.json')
                record['stage_results'].append(dict(stage=stage, optimizer_updates=2, first=first, resumed=second,
                    actual_memory_peaks=memory['peaks'], actual_memory_budget=memory['budget']))
            checkpoint = Path(model['run_dir'])/'local/latest.pt'
            applied = options(path)
            prompts = None
        else:
            from chd_segmentation.models.medsam2.native import NativeCases
            from chd_segmentation.models.medsam2.prompts import label_box_prompts
            reader = NativeCases(prepared, with_label=True)
            entry = reader.get('ct_1003')
            classes = sorted(set(map(int, np.unique(entry['label'])))-{0, -1}-set(entry['unknown_classes']))
            prompts, _ = label_box_prompts(entry, item['case']['sha256']['label'], 8, classes[:1])
            for prompt in prompts['ct_1003']:
                prompt['source'] = 'technical_simulated'
            reader.clear()
            entry.clear()
            del reader, entry
            gc.collect()
            checkpoint = weight
            applied = options(path)
        began = perf_counter()
        prediction = predict(applied, prepared_path, checkpoint, workspace.root/'result/prediction',
                             prompts=prompts, official_weights=args.model == 'med')
        assert len(prediction['cases']) == 1 and prediction['cases'][0]['case_id'] == 'ct_1003'
        label = 'ImageCHD-Q' if args.model == 'q' else 'ImageCHD-MedSAM2'
        resources = list((workspace.root/'course/runs'/label/'predictions').rglob('resources.json'))
        assert len(resources) == 1
        progress = read_json(resources[0].parent/'progress-predict.json')
        assert progress['done'] == progress['total'] == 1 and progress['unit'] == 'cases'
        assert (resources[0].parent/'memory.json').is_file()
        record.update(prediction=prediction, prediction_seconds=perf_counter()-began,
                      prediction_resources=read_json(resources[0]), shared_progress=True,
                      prediction_actual_memory=read_json(resources[0].parent/'memory.json'),
                      shared_actual_memory_guard=True, optimizer_updates=6 if args.model == 'q' else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('spec', 'baseline', 'work-root', 'report'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--model', choices=('q', 'med', 'q-inspect'), required=True)
    parser.add_argument('--simulator', default='D:/projects/test')
    run(parser.parse_args())
