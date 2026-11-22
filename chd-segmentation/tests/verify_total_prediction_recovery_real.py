import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic, sleep
from unittest import TestCase

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))


def child(root):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from chd_segmentation.artifacts.store import read_json
    from chd_segmentation.finetuning.total import predict_cases

    torch.set_num_threads(2)
    bundle = read_json(root / 'input.json')
    config = {key: Path(value) for key, value in bundle['config'].items()}
    metadata = torch.load(root / 'model.pt', map_location='cpu', weights_only=False)
    manager = PlansManager(bundle['plans'])
    spec = manager.get_configuration('3d_fullres')
    network = nnUNetTrainer.build_network_architecture(manager, spec, 1, 8, False)
    network.load_state_dict(metadata)
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=False, device=torch.device('cpu'), verbose=False, allow_tqdm=False)
    predictor.manual_initialization(network, manager, spec, [metadata], bundle['dataset'],
                                    'nnUNetTrainer', None)
    predict_cases(config, bundle['prepared'], bundle['groups'], 'test', 'best', root/'model.pt', predictor)


def members(pid):
    import psutil
    from chd_segmentation.acceleration.memory import identity
    owner = psutil.Process(pid)
    return [identity(process) for process in [owner, *owner.children(recursive=True)] if process.is_running()]


def stop_child(process):
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=20)


def run(args, root, result):
    import nibabel as nib
    import numpy as np
    import psutil
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from chd_segmentation.acceleration import ResourceDemand, ResourceSession, RuntimeRequest
    from chd_segmentation.artifacts.store import file_digest, read_json, write_json
    from chd_segmentation.datasets.manifest import LABELS
    from chd_segmentation.finetuning.total import DATASET, memory_budget, pending_predictions
    from chd_segmentation.finetuning.total_recovery import check_monitor, prediction_ready, supervise_child

    torch.set_num_threads(2)
    torch.manual_seed(20261122)
    manifest = read_json(args.manifest)
    groups = dict(train=[], validation=[], test=[])
    config = dict(run_dir=root/'run', work_root=root/'course', dataset_dir=root/'data',
                  data_root=root/'data', result_root=root/'result', model_folder=root/'source')
    raw = root/'data/raw'/DATASET/'imagesTs'
    raw.mkdir(parents=True)
    config['run_dir'].mkdir()
    config['model_folder'].mkdir()
    for name in ('ct_1003', 'ct_1012'):
        row = deepcopy(next(row for row in manifest['cases'] if row['case_id'] == name))
        source = nib.load(Path(args.data_root)/row['image'])
        region = tuple(slice(size//2-32, size//2+32) for size in source.shape)
        values = np.asarray(source.dataobj[region], dtype=np.float32)
        assert values.shape == (64, 64, 64)
        image = nib.Nifti1Image(values, np.diag([*row['spacing_xyz'], 1.]))
        path = raw/(name+'_0000.nii.gz')
        nib.save(image, path)
        row.update(image=str(path), roles=['test'], source_stamp=dict(image=[path.stat().st_size,path.stat().st_mtime_ns]))
        row['sha256']['image'] = file_digest(path)
        groups['test'].append(row)
    prepared = dict(scope='technical', dataset='imagechd', fold=0, technical=dict(roi_xyz=None),
                    manifest=dict(data_root=str(Path(args.data_root)), cases=groups['test']))
    dataset = dict(channel_names={'0':'CT'}, labels={name:i for i,name in enumerate(LABELS['imagechd'])}, file_ending='.nii.gz')
    prepared.update(plan=dict(spacing_xyz=groups['test'][0]['spacing_xyz']), dataset_json=dataset)
    plan = deepcopy(read_json(args.plans))
    manager = PlansManager(deepcopy(plan))
    spec = deepcopy(manager.get_configuration('3d_fullres').configuration)
    spec.update(patch_size=[64]*3, spacing=list(reversed(groups['test'][0]['spacing_xyz'])))
    plan['configurations'] = {'3d_fullres': spec}
    manager = PlansManager(plan)
    network = nnUNetTrainer.build_network_architecture(manager,manager.get_configuration('3d_fullres'),1,8,False)
    torch.save(network.state_dict(), root/'model.pt')
    del network
    write_json(root/'input.json',dict(config={k:str(v) for k,v in config.items()},plans=plan,prepared=prepared,groups=groups,dataset=dataset))
    write_json(root/'prepared.json',prepared)
    write_json(config['model_folder']/'plans.json', plan)
    prediction_config = dict(config, prepared=root/'prepared.json', prediction_ram_bytes=1)
    selected = memory_budget(prediction_config, 'predict', cases=groups['test'])
    assert selected > 0
    result['selected_role_budget_bytes'] = selected
    resources = dict(device='cpu',technical_cpu=True, ram_fraction=1.,cpu_fraction=.8,
        reserve_ram_bytes=0,memory_limit_bytes=6*2**30,memory_policy='project_limit',max_wait=20)
    demand = ResourceDemand(fixed_bytes=3*2**30,cpu_slots=3)

    def request(name):
        return RuntimeRequest(name,'total_predict',root/'resources',config['run_dir'],dict(resources),demand)

    command = [sys.executable,'-u','-B',str(Path(__file__).resolve()),'--child',str(root)]
    mark = monotonic()
    with ResourceSession(request('pressure')) as session:
        assert prediction_ready(session,config['run_dir'])
        with (root/'interrupted.log').open('wb') as stream, subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT) as process:
            try:
                session.request.resources['memory_limit_bytes'] = 1
                deadline = monotonic()+8
                while not session.state['paused'] and monotonic()<deadline:
                    sleep(.1)
                assert session.state['paused']
                assert not supervise_child(config,session,'predict',process,members)
                assert 'project_limit' in session.state['pressure_reasons']
            finally:
                stop_child(process)
    assert read_json(root/'resources/active.json') == []
    result['pressure_release_seconds'] = monotonic()-mark
    result['pressure_released_resources'] = True
    mark = monotonic()
    for name in ('resume','reuse'):
        with ResourceSession(request(name)) as session:
            assert prediction_ready(session,config['run_dir'])
            with (root/(name+'.log')).open('wb') as stream, subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT) as process:
                try:
                    assert supervise_child(config,session,'predict',process,members)
                finally:
                    stop_child(process)
            result['peak_process_group_rss_bytes'] = max(result.get('peak_process_group_rss_bytes',0),session.state['peak_task_rss'])
        output = root/'result/ImageCHD-TotalSegmentator-fold0/test-best'
        receipt = read_json(output/'predictions.json')
        assert len(receipt['cases']) == 2
        current = {path.name:(path.stat().st_mtime_ns,file_digest(path)) for path in output.glob('*.nii.gz')}
        if name == 'resume':
            baseline = current
            assert pending_predictions(prepared,groups['test'],output,file_digest(root/'model.pt')) == []
            with TestCase().assertRaises(ValueError):
                pending_predictions(prepared,groups['test'],output,'changed-checkpoint')
            for row in groups['test']:
                prediction = nib.load(output/(row['case_id']+'.nii.gz'))
                assert prediction.shape == (64,64,64)
                assert np.isin(np.asarray(prediction.dataobj),range(8)).all()
        else:
            assert current == baseline
    result['prediction_and_reuse_seconds'] = monotonic()-mark
    result['reused_cases'] = len(baseline)
    with ResourceSession(request('cancel')) as session:
        session.cancel.set()
        with TestCase().assertRaises(InterruptedError):
            prediction_ready(session,config['run_dir'])
    result['cancel_passed'] = True
    registry = root/'resources/active.json'
    with TestCase().assertRaises(json.JSONDecodeError):
        with ResourceSession(request('monitor-error')) as session:
            saved = registry.read_bytes()
            try:
                registry.write_text('{',encoding='utf-8')
                deadline = monotonic()+8
                while not session.state['monitor_failed'] and monotonic()<deadline:
                    sleep(.1)
                assert session.state['monitor_failed']
                with TestCase().assertRaises(json.JSONDecodeError):
                    check_monitor(session,config['run_dir'],'predict')
                assert session.state['monitor_error']['type'] == 'JSONDecodeError'
            finally:
                registry.write_bytes(saved)
    assert read_json(registry) == []
    result.update(monitor_error_preserved=True, cases=['ct_1003','ct_1012'],shape=[64]*3,
        training_updates=0,predicted_cases=2,gpu_used=False,official_301_weights_used=False,
        peak_parent_rss_bytes=psutil.Process().memory_info().peak_wset,
        linux_process_group_verified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child',type=Path)
    parser.add_argument('--manifest')
    parser.add_argument('--data-root')
    parser.add_argument('--plans')
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    root = Path('D:/projects/back/t17/chd-total-predict-recovery-20261122').resolve()
    report = Path('D:/projects/back/sync-reports/chd-total-predict-recovery-20261122')
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    (root/'.gitignore').write_text('*\n',encoding='utf-8')
    report.mkdir(parents=True,exist_ok=True)
    for key in ('TEMP','TMP','MPLCONFIGDIR'):
        path = root/key
        path.mkdir()
        os.environ[key] = str(path)
    os.environ['nnUNet_compile'] = 'false'
    began = monotonic()
    result = dict(state='failed',training_updates=0)
    try:
        run(args,root,result)
        result['state'] = 'passed'
    finally:
        error = sys.exception()
        if error is not None:
            result['error'] = dict(type=type(error).__name__,message=str(error))
        result['seconds'] = monotonic()-began
        files = [path for path in root.rglob('*') if path.is_file()]
        assert all(root in path.resolve().parents and not path.is_symlink() for path in files)
        total = sum(path.stat().st_size for path in files)
        assert len(files)<=80 and total<=2*2**30
        result['cleanup'] = dict(files=len(files),bytes=total,root=str(root),state='pending')
        for name in ('interrupted.log','resume.log','reuse.log'):
            path = root/name
            if path.exists():
                assert path.stat().st_size < 256*1024
                shutil.copy2(path,report/name)
        if (root/'run/events.jsonl').exists():
            shutil.copy2(root/'run/events.jsonl',report/'events.jsonl')
        (report/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        shutil.rmtree(root)
        result['cleanup'].update(state='complete',root_absent=not root.exists())
        (report/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
