"""Private-data acceptance for the two whole-scan Xu localizers. Use a research Python."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from dataset1282_analysis.contracts import atomic_json, read_json, sha256_file


def box_of(mask):
    bounds = []
    for axis in range(3):
        positions = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        if not len(positions):
            raise ValueError('Foreground is empty')
        bounds.append((int(positions[0]), int(positions[-1]) + 1))
    return bounds


def plane(volume, axis, index):
    key = [slice(None)] * 3
    key[axis] = index
    return np.asanyarray(volume[tuple(key)]).T


def draw_qc(path, native, mask, spacing, box, inputs, transpose_backward):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    truth = box_of(mask)
    fig, axes = plt.subplots(4, 3, figsize=(10.8, 13.0), layout='constrained')
    for col, (axis, anatomical) in enumerate(((2, 'axial'), (1, 'coronal'), (0, 'sagittal'))):
        index = (truth[axis][0] + truth[axis][1] - 1) // 2
        x, y = [i for i in range(3) if i != axis]
        ct_slice = plane(native, axis, index)
        low, high = np.percentile(ct_slice, [1, 99])
        for row in (0, 3):
            axes[row,col].imshow(ct_slice, cmap='gray', origin='lower', vmin=low, vmax=high,
                                 aspect=spacing[y]/spacing[x], interpolation='nearest')
        axes[0,col].set_title(f'CT | {anatomical}, index {index}')
        for row,(branch, data) in enumerate(inputs.items(),1):
            # Cache axis order -> NIfTI XYZ. Show the actual model input center plane.
            xyz = data.transpose(transpose_backward).transpose(2,1,0)
            cut = plane(xyz,axis,xyz.shape[axis]//2)
            axes[row,col].imshow(cut,cmap='gray',origin='lower',interpolation='nearest')
            axes[row,col].set_title(f'{branch} | {anatomical} center')
        axes[3,col].contour(plane(mask,axis,index),levels=[0.5],colors=['#ffcb63'],
                            linewidths=0.8,origin=None)
        if box[axis][0] <= index < box[axis][1]:
            axes[3,col].add_patch(Rectangle((box[x][0]-.5,box[y][0]-.5),
                                            box[x][1]-box[x][0], box[y][1]-box[y][0],
                                            fill=False,edgecolor='#40c9e6',linewidth=2))
        axes[3,col].set_title('Native grid | reference + prediction')
        for row in range(4):
            axes[row,col].axis('off')
    fig.suptitle('Development case | gold: annotated foreground | cyan: smoke-model box\n'
                 'Technical check only. Native display slice chosen from reference for review.')
    fig.savefig(path,dpi=135)
    plt.close(fig)


def paper_forwards(inputs, device):
    """Measure each full-size network sequentially; weights are randomly initialized."""
    import gc
    import torch
    from dataset1282_analysis.xu_models import build_model
    results = {}
    gpu = torch.device(device).type == 'cuda'
    for branch, cube in inputs.items():
        if gpu:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(2023)
        model = build_model(branch,profile='paper').eval().to(device)
        tensor = torch.from_numpy(np.ascontiguousarray(cube)).to(device)[None,None]
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast(torch.device(device).type, enabled=gpu):
            output = model(tensor)
            if output.shape != (1,11,*cube.shape) or not torch.isfinite(output).all().item():
                raise ValueError(f'Paper {branch} produced an invalid tensor')
        results[branch] = dict(input_shape=list(tensor.shape), output_shape=list(output.shape),
                               parameters=sum(w.numel() for w in model.parameters()),
                               dtype=str(output.dtype), device=device, weights='random_initialization',
                               seconds=time.monotonic()-started,
                               peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if gpu else None)
        print('paper forward', branch, results[branch], flush=True)
        del output, tensor, model
        gc.collect()
    if gpu:
        torch.cuda.empty_cache()
    return results


def paper_step64(cache, name, device):
    """One normal FP32/Adam production-width step on four augmentations of one scan."""
    import torch
    from dataset1282_analysis.xu_models import build_model
    from dataset1282_analysis.xu_training import _epoch, learning_rate
    torch.manual_seed(2023)
    gpu = torch.device(device).type == 'cuda'
    if gpu:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    shape = (64,)*3
    sample = cache.read_image(name,shape), cache.read_label(name,shape)
    model = build_model('crop64','paper').to(device)
    before = model.head.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate(0,model.spec.epochs))
    rng = torch.Generator(device=device).manual_seed(2023)
    start = time.monotonic()
    # Exercise exactly one normal training batch; the repeated source is declared below.
    loss = _epoch(model,[name]*model.spec.batch,{name:[]},lambda _id,_branch: sample,
                  'crop64',device,rng,optimizer)
    update = (model.head.weight.detach()-before).abs().max().item()
    if not np.isfinite(loss) or not np.isfinite(update) or update <= 0:
        raise ValueError('Formal-width step did not produce a finite parameter update')
    return dict(branch='crop64',profile='paper',physical_batch=model.spec.batch,
                unique_training_scans=1, distinct_random_augmentations=model.spec.batch,
                optimizer_steps=1, loss=loss, head_max_abs_update=update, device=device,
                dtype='float32', seconds=time.monotonic()-start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if gpu else None)


def main(args):
    import nibabel as nib
    import torch
    from dataset1282_analysis.xu_geometry import PreparedCache
    from dataset1282_analysis.xu_roi import vote_box, coverage, MIN_CLASS_P05
    from dataset1282_analysis.xu_training import load_predictor, train_branch
    from dataset1282_analysis.xu_workflow import input_id

    run, out = args.prepared.resolve(), args.output.resolve()
    if run == out or run in out.parents:
        raise ValueError('Place private acceptance outputs outside the preprocessing run')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new or empty acceptance directory')
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(2023)
    cache = PreparedCache(run,allow_technical_subset=True)
    ds = run/'preprocessed'/'Dataset282_CHDSource'
    train = cache.case_ids('train',complete_only=True)
    dev = cache.case_ids('development')
    if not train or not dev or set(train).intersection(dev):
        raise ValueError('Requires disjoint complete-label training and development cases')
    # One complete training case and one development case keep this a bounded machine check.
    # Ground truth in dev is used for validation and QC, never to generate prediction boxes.
    train_name, dev_name = train[0], dev[0]
    train_cases = {train_name: []}
    val_cases = {dev_name: cache.supervision[dev_name]['unknown_classes']}
    identity = input_id(cache,'complete')
    inputs = {}
    predictions = {}
    checkpoints = {}
    for branch, side in [('crop64',64), ('crop128',128)]:
        shape = (side,)*3
        def load_sample(name, selected_branch):
            if selected_branch != branch:
                raise ValueError(f'Unexpected branch {selected_branch}')
            return cache.read_image(name,shape=shape),cache.read_label(name,shape=shape)
        print('train and validate smoke',branch,flush=True)
        artifacts = train_branch(branch=branch,output_dir=out/branch,train_cases=train_cases,
                                 validation_cases=val_cases,load_sample=load_sample,
                                 data_identity=identity,profile='smoke',supervision='complete',
                                 device=args.device,seed=2023)
        path = Path(artifacts['best_checkpoint'])
        infer = load_predictor(path,branch=branch,profile='smoke',data_identity=identity,
                               device=args.device)
        cube = np.asarray(cache.read_image(dev_name,shape=shape)).reshape(shape)
        inputs[branch] = cube
        pred = np.asarray(infer(cube))
        if pred.shape!=shape or np.min(pred)<0 or np.max(pred)>10:
            raise ValueError(f'Invalid {branch} prediction')
        predictions[branch] = pred
        checkpoints[branch]=dict(sha256=sha256_file(path), case_ids_used_for_updates=list(train_cases),
                                 development_ids=list(val_cases), predicted_foreground_fraction=float((pred>0).mean()))
        del infer

    geo = cache.geometry(dev_name)
    vote = vote_box(predictions['crop64'], predictions['crop128'], geo)
    cache_box = vote['box']
    if not vote['quality']['accepted']:
        atomic_json(out/'acceptance.json', dict(passed=False, roi=vote, reason=vote['quality']['reason']))
        raise ValueError(f"Localizers failed the ROI check: {vote['quality']['reason']}")
    native_box = tuple((int(a),int(b)) for a,b in geo.cache_box_to_native(cache_box))
    native_label = cache.read_native_label(dev_name)
    target = native_label > 0
    roi = tuple(slice(lo,hi) for lo,hi in vote['quality']['native_support'])
    native = nib.load(run/'staged'/dev_name/('image'+cache.staged_extension))
    if native.shape!=target.shape:
        raise ValueError('Source grid mismatch')
    target_count = int(target.sum())
    inside = int(target[roi].sum())
    roi_fraction = float(np.prod([b-a for a,b in native_box]) / np.prod(target.shape))
    shape128 = (128,)*3
    truth128 = np.asarray(cache.read_label(dev_name,shape=shape128)).reshape(shape128)
    restored = geo.restore_label(np.maximum(truth128,0)) > 0
    if restored.shape != target.shape:
        raise ValueError('Restored shape mismatch')
    overlap = int(np.count_nonzero(restored & target))
    roundtrip = 2*overlap / (target_count + int(restored.sum()))
    if roundtrip<0.90:
        raise ValueError(f'Native -> 128 cube -> native foreground Dice {roundtrip:.4f} < 0.90')
    plans = read_json(ds/'nnUNetPlans.json')
    draw_qc(out/'native-localization.png',native.dataobj,target,geo.spacing_xyz,native_box,
            inputs,plans['transpose_backward'])
    retained = coverage(native_label, {'vote':cache_box}, geo, cache.supervision[dev_name]['present_classes'])['vote']
    report=dict(passed=all(x >= MIN_CLASS_P05 for x in retained.values()), per_structure_coverage=retained,
                kind='real_input_technical_acceptance', prepared_scope=cache.scope,
                fit_cases=list(read_json(run/'fit.json')['fit_cases']), localizer_profile='smoke',
                train_cases=list(train_cases), development_cases=list(val_cases),
                development_unknown_classes=val_cases[dev_name], checkpoints=checkpoints,
                native_shape_xyz=list(native.shape), native_spacing_xyz=list(geo.spacing_xyz),
                cache_shape=list(geo.cache_shape), predicted_box_native_xyz=native_box,
                native_foreground_box_xyz=box_of(target), annotated_foreground_coverage=inside/target_count,
                predicted_box_fraction_of_scan=roi_fraction, truth_foreground_roundtrip_dice_128=roundtrip,
                visualization='native-localization.png', full_architecture_forwards={},
                interpretation='Box and coverage are technical smoke outputs, not trained paper-model performance.')
    if args.paper_forward:
        report['full_architecture_forwards'] = paper_forwards(inputs,args.device)
    if args.paper_step64:
        report['full_width_step_capacity'] = paper_step64(cache,train_name,args.device)
    atomic_json(out/'acceptance.json',report)
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)
    if not report['passed']:
        raise ValueError('Insufficient held-out structure coverage; see acceptance.json')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--paper-forward',action='store_true',
                        help='Also run full-width random-initialized inference on both real cubes.')
    parser.add_argument('--paper-step64',action='store_true',
                        help='Also test one full-width crop64 FP32 Adam batch (four views of one scan).')
    main(parser.parse_args())
