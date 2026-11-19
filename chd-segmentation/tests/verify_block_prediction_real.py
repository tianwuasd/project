import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import threading
from time import perf_counter
import unittest

import psutil


sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def verify(args):
    import nibabel as nib
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from chd_segmentation.artifacts.store import digest, read_json, write_json
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import prepare
    from chd_segmentation.preprocessing.offline import preprocess
    from chd_segmentation.data.dataset import CaseDataset
    from chd_segmentation.models.stunet.adapter import StunetAdapter
    from chd_segmentation.training.checkpoint import save_checkpoint
    from chd_segmentation.prediction.blocks import window_probabilities, restore_blocks
    from chd_segmentation.prediction.geometry import restore_source
    from chd_segmentation.prediction.workspace import PredictionWorkspace
    from chd_segmentation.prediction.infer import predict
    from chd_segmentation.prediction.metrics import evaluate
    from chd_segmentation.acceleration.host import host_capacity
    from chd_segmentation.acceleration.limits import memory_ceiling
    from chd_segmentation.acceleration.planning import cpu_plan
    root, data = Path(args.work), Path(args.data_root)
    manifest = read_json(args.manifest)
    cases = []
    for name,role in zip(args.cases,('train','validation')):
        row = deepcopy(next(r for r in manifest['cases'] if r['case_id']==name))
        row.update(image=str(data/(name+'_image.nii.gz')),label=str(data/(name+'_label.nii.gz')),roles=[role])
        cases.append(row)
    manifest.update(cases=cases,data_root=str(data),scope='technical',plan_scope='technical',binding_required=False)
    course = root/'course'
    write_json(course/'manifest.json',manifest)
    raw = dict(dataset='imagechd',model='stunet',scope='technical',fold=0,
        manifest=str(course/'manifest.json'),pretrained=args.pretrained,
        work_root=str(course),run_dir=str(course/'runs/STU'),result_root=str(root/'result'),
        technical=dict(roi_xyz=[[224,288],[224,288],[71,103]]),
        planning=dict(mode='auto',samples_per_case=1000,max_samples=1000,
                      target_spacing_xyz=cases[0]['spacing_xyz']),
        preprocessing=dict(root=str(root/'data'),statistics_source='technical-train'),
        training=dict(patch=[32]*3,batch=1,attempts=1),
        prediction=dict(block_bytes=1024**2,max_block_bytes=2*1024**2),
        resources=dict(device='cpu',cpu_threads=1,workers=0,max_workers=1,cache_bytes=0,prefetch=1,
            max_readers=1,reserve_ram_bytes=256*1024**2,wait_seconds=30),
        logging=dict(heartbeat_seconds=15,case_details=False))
    write_json(course/'config.json',raw)
    config = load_config(course/'config.json')
    times = {}
    started = perf_counter()
    prepare(config,course/'source.json')
    prepared = preprocess(config,course/'source.json',course/'offline.json')
    times['preprocessing_seconds'] = perf_counter()-started
    torch.set_num_threads(1)
    torch.manual_seed(42)
    spec = asdict(StunetAdapter.spec(config,prepared['plan']))
    model = StunetAdapter.build(spec).eval()
    pretrained_receipt = StunetAdapter.load_pretrained(model,args.pretrained)
    optimizer = torch.optim.SGD(model.parameters(),lr=.001,momentum=.99)
    identity = digest(dict(prepared=prepared['identity'],spec=spec,scope='initialized-untrained'))
    write_json(Path(config['run_dir'])/'run.json',dict(identity=identity,prepared=prepared,
        scientific=dict(spec=spec,prepared_identity=prepared['identity'])))
    save_checkpoint(config['run_dir'],dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
        torch_rng=torch.get_rng_state(),sampler_rng=np.random.default_rng(42).bit_generator.state,
        attempts=0,updates=0,spec=spec,config=config,skipped=0,pretrained_receipt=pretrained_receipt))
    dataset = CaseDataset(prepared,config['resources'],with_label=False)
    differences = []
    started = perf_counter()
    def predictor(image):
        result = model(image)
        return result[0] if isinstance(result,(tuple,list)) else result
    for index,name in enumerate(args.cases):
        entry = dataset.get(name)
        options = dict(config['prediction'],blend='gaussian' if index==0 else 'constant',window_batch=index+1)
        with torch.inference_mode():
            logits = sliding_window_inference(torch.from_numpy(entry['image'])[None],tuple(spec['patch']),
                options['window_batch'],predictor,overlap=options['overlap'],mode=options['blend'])
            expected = logits[0].float().softmax(0).numpy()
        reference = restore_source(expected,entry)
        with PredictionWorkspace(config,dict(comparison=name),1024**2) as workspace:
            actual = window_probabilities(entry['image'],predictor,tuple(spec['patch']),8,options,torch.device('cpu'),workspace)
            difference = float(np.max(np.abs(actual-expected)))
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
            restored = restore_blocks(actual,entry,workspace)
            np.testing.assert_array_equal(restored,reference)
            differences.append(dict(case=name,max_probability_difference=difference,blend=options['blend'],window_batch=options['window_batch']))
            del actual,restored
        # 改变目标间距只检验同一真实病例概率的各向异性恢复。
        alternate = deepcopy({k:v for k,v in entry.items() if k not in ('image','label')})
        alternate['processed_spacing'] = [1.,1.,5.]
        reference_separate = restore_source(expected,alternate)
        with PredictionWorkspace(config,dict(anisotropy=name),1024**2) as workspace:
            actual_separate = restore_blocks(expected,alternate,workspace)
            np.testing.assert_array_equal(actual_separate,reference_separate)
            del actual_separate
        del entry,expected,logits,reference,reference_separate
        dataset.clear()
    times['comparison_seconds'] = perf_counter()-started
    del model,optimizer
    started = perf_counter()
    whole = predict(config,root/'result/whole')
    blocked_config = deepcopy(config)
    blocked_config['prediction']['execution'] = 'blocks'
    blocked = predict(blocked_config,root/'result/blocks')
    for name in args.cases:
        left = np.asanyarray(nib.load(root/'result/whole'/f'{name}.nii.gz').dataobj)
        right = np.asanyarray(nib.load(root/'result/blocks'/f'{name}.nii.gz').dataobj)
        np.testing.assert_array_equal(left,right)
        del left,right
    reused = predict(blocked_config,root/'result/blocks')
    assert reused['new']==0 and reused['reused']==2
    first = evaluate(root/'result/blocks',prepared,root/'result/metrics.json',config)
    second = evaluate(root/'result/blocks',prepared,root/'result/metrics.json',config)
    assert second['reused']
    # 用原ROI整块计数核对逐块评价，保持相同未知类别与忽略标签。
    receipts = read_json(root/'result/blocks/predictions.json')['cases']
    for row in prepared['manifest']['cases']:
        name = row['case_id']
        roi = tuple(slice(a,b) for a,b in receipts[name]['roi_xyz'])
        truth = np.asanyarray(nib.load(row['label']).dataobj[roi]).astype(np.int16)
        truth[np.isin(truth,row.get('ignore_label_ids',[]))] = -1
        prediction = np.asanyarray(nib.load(root/'result/blocks'/f'{name}.nii.gz').dataobj[roi])
        record = next(r for r in first['cases'] if r['case_id']==name)
        for c in range(1,8):
            if c in row['unknown_classes']:
                continue
            target,found = truth==c,(prediction==c)&(truth>=0)
            assert record['classes'][str(c)]['reference_voxels']==int(target.sum())
            assert record['classes'][str(c)]['predicted_voxels']==int(found.sum())
        del truth,prediction,target,found
    assert all(row['prediction_execution']=='whole' for row in cpu_plan(dict(config,resources={**config['resources'],'cache_bytes':0}),prepared,1,stage='prediction')['case_estimates'])
    facts = host_capacity()
    assert memory_ceiling(dict(config['resources'],memory_policy='host_fraction'),facts)==int(facts['total_bytes']*.8)
    with unittest.TestCase().assertRaises(InterruptedError):
        with PredictionWorkspace(config,dict(cancellation='real-workspace'),1024**2) as workspace:
            workspace.create('native',(32,32,32),np.uint8)
            raise InterruptedError('主动取消')
    assert not list((course/'prediction-work').iterdir())
    times['prediction_evaluation_seconds'] = perf_counter()-started
    return dict(cases=args.cases,roi_xyz=config['technical']['roi_xyz'],training_updates=0,
        initialized_untrained=True,comparisons=differences,whole=whole,blocks=blocked,
        reused_predictions=reused['reused'],evaluation_reused=second['reused'],timing=times)


def controls(args):
    import nibabel as nib
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from chd_segmentation.artifacts.store import read_json
    from chd_segmentation.config import load_config
    from chd_segmentation.models.stunet.adapter import StunetAdapter
    from chd_segmentation.data.dataset import CaseDataset
    from chd_segmentation.prediction.blocks import window_probabilities, restore_blocks
    from chd_segmentation.prediction.geometry import restore_source
    from chd_segmentation.prediction.workspace import PredictionWorkspace
    from chd_segmentation.training.stage_memory import prediction_workspace
    from chd_segmentation.acceleration.planning import cpu_plan
    from chd_segmentation.acceleration.host import host_capacity
    from chd_segmentation.acceleration.limits import memory_ceiling
    root = Path(args.work)
    config = load_config(root/'course/config.json')
    prepared = read_json(root/'course/offline.json')
    torch.set_num_threads(1)
    state = torch.load(root/'result/whole/selected-model.pt',map_location='cpu',weights_only=False)
    model = StunetAdapter.build(state['spec']).eval()
    model.load_state_dict(state['model'],strict=True)
    spec = state['spec']
    del state
    def predictor(image):
        values = model(image)
        return values[0] if isinstance(values,(list,tuple)) else values
    dataset = CaseDataset(prepared,config['resources'],with_label=False)
    checks = []
    for index,name in enumerate(args.cases):
        entry = dataset.get(name)
        options = dict(config['prediction'],window_batch=2,blend='gaussian' if index==0 else 'constant')
        with torch.inference_mode():
            expected = sliding_window_inference(torch.from_numpy(entry['image'])[None],tuple(spec['patch']),
                options['window_batch'],predictor,overlap=options['overlap'],mode=options['blend'])[0].float().softmax(0).numpy()
        with PredictionWorkspace(config,dict(controls=name),2*1024**2) as workspace:
            actual = window_probabilities(entry['image'],predictor,tuple(spec['patch']),8,options,torch.device('cpu'),workspace)
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
            native = restore_blocks(actual,entry,workspace)
            np.testing.assert_array_equal(native,restore_source(expected,entry))
            checks.append(dict(case=name,shape=list(entry['image'].shape),difference=float(np.max(np.abs(actual-expected)))))
            del actual,native
        if index==0:
            image = entry['image'][:,:17,:23,:29]
            with torch.inference_mode():
                padded_expected = sliding_window_inference(torch.from_numpy(image)[None],tuple(spec['patch']),2,predictor,
                    overlap=options['overlap'],mode=options['blend'])[0].float().softmax(0).numpy()
            with PredictionWorkspace(config,dict(padded=name),2*1024**2) as workspace:
                actual = window_probabilities(image,predictor,tuple(spec['patch']),8,options,torch.device('cpu'),workspace)
                np.testing.assert_allclose(actual,padded_expected,rtol=2e-5,atol=2e-6)
                del actual
            del image,padded_expected
        del entry,expected
        dataset.clear()
    # 使用同两例的真实完整影像头信息计算较低额度下的模式；不读取完整影像数组。
    budget_config = deepcopy(config)
    budget_config['prediction'].update(block_bytes=64*1024**2,max_block_bytes=512*1024**2)
    reports = []
    order = prepared['plan']['transpose_forward']
    target = np.asarray(prepared['plan']['configurations']['3d_fullres']['spacing'])
    for row in prepared['manifest']['cases']:
        native = list(nib.load(row['image']).shape)
        working = np.round(np.asarray(native)[::-1][order]*np.asarray(row['spacing_xyz'])[::-1][order]/target).astype(int)
        memory = prediction_workspace([1,*map(int,working)],native,[[0,n] for n in native],8,
            budget_config['prediction'],spec['patch'])
        reports.append(dict(case_id=row['case_id'],workspace_bytes=memory['whole_workspace_bytes'],**memory))
    facts = host_capacity()
    budget = min(8*1024**3,memory_ceiling(dict(config['resources'],memory_policy='host_fraction'),facts))
    plan = cpu_plan(budget_config,prepared,1,stage='prediction',estimates=reports,
        bounds=dict(facts=facts,memory_budget_bytes=budget,cpu_budget_slots=facts['cores']))
    assert plan['admitted'] and plan['ram_bytes']<=budget
    assert all(r['prediction_execution']=='blocks' for r in plan['case_estimates'])
    assert all(64*1024**2<=r['prediction_block_bytes']<=512*1024**2 for r in plan['case_estimates'])
    return dict(comparisons=checks,padding_verified=True,automatic_blocks_verified=True,
        planning_only=dict(memory_budget_bytes=budget,granted_bytes=plan['ram_bytes'],cases=plan['case_estimates']),
        training_updates=0)


def main():
    parser = argparse.ArgumentParser()
    for flag in ('work','report','manifest','data-root','pretrained'):
        parser.add_argument('--'+flag,required=True)
    parser.add_argument('--cases',nargs=2,required=True)
    parser.add_argument('--part',choices=('full','controls'),default='full')
    args = parser.parse_args()
    root = Path(args.work).resolve()
    root.mkdir(parents=True,exist_ok=args.part=='controls')
    runtime = root/'runtime'
    runtime.mkdir(exist_ok=args.part=='controls')
    for key in ('TEMP','TMP','TMPDIR','MPLCONFIGDIR','TORCH_HOME','HF_HOME','XDG_CACHE_HOME'):
        os.environ[key] = str(runtime)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    stopped = threading.Event()
    measured = []
    def observe():
        while not stopped.wait(.2):
            process = psutil.Process()
            measured.append(sum(p.memory_info().rss for p in [process,*process.children(recursive=True)] if p.is_running()))
    observer = threading.Thread(target=observe)
    observer.start()
    began = perf_counter()
    try:
        result = verify(args) if args.part=='full' else controls(args)
    finally:
        stopped.set()
        observer.join()
    files = [p for p in root.rglob('*') if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    assert len(files)<=100 and size<=2*1024**3
    result.update(state='passed',seconds=perf_counter()-began,files=len(files),bytes=size,
        peak_sampled_group_rss_bytes=max(measured,default=0),peak_vram_bytes=None)
    from chd_segmentation.artifacts.store import write_json
    write_json(args.report,result)
    print(json.dumps(dict(state='passed',seconds=result['seconds'],files=len(files),bytes=size)),flush=True)


if __name__=='__main__':
    main()
