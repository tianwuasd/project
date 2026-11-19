import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import sys
from time import perf_counter

import nibabel as nib
import numpy as np
import psutil
import torch
from monai.inferers import sliding_window_inference

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))

from chd_segmentation.artifacts.store import digest, read_json, write_json
from chd_segmentation.models.q_route1.adapter import QAdapter, allowed
from chd_segmentation.models.q_route1.candidates import select_candidates
from chd_segmentation.models.q_route1.localizer import Localizer
from chd_segmentation.models.q_route1.losses import segmentation_loss
from chd_segmentation.models.q_route1.targets import build_heatmap_targets, localizer_loss
from chd_segmentation.models.q_route1.validation_candidates import bounded_candidates
from chd_segmentation.models.q_route1.validation_loss import segmentation_statistics
from chd_segmentation.models.q_route1.validation import local_probability
from chd_segmentation.models.q_route1.validation_network import LayerInference, localizer_forward
from chd_segmentation.models.q_route1.validation_store import ValidationWorkspace
from chd_segmentation.models.q_route1.validation_targets import localizer_statistics
from chd_segmentation.prediction.blocks import window_probabilities


def entry_from_source(row, root):
    image = nib.load(root/row['image'])
    center = [n//2 for n in image.shape]
    roi = tuple(slice(c-32, c+32) for c in center)
    raw = np.asarray(image.dataobj[roi], dtype=np.float32).transpose(2, 1, 0)
    label = np.asarray(nib.load(root/row['label']).dataobj[roi], dtype=np.int16).transpose(2, 1, 0)
    lower, upper = np.percentile(raw, [.5, 99.5])
    values = np.clip(raw, lower, upper)
    values = (values-values.mean())/max(float(values.std()), 1e-6)
    return dict(case_id=row['case_id'], image=np.ascontiguousarray(values[None], dtype=np.float32),
                label=np.ascontiguousarray(label), unknown_classes=row['unknown_classes'],
                processed_spacing=row['spacing_xyz'][::-1])


def run(args):
    started = perf_counter()
    root, report = Path(args.work_root), Path(args.report)
    root.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261111)
    manifest = read_json(args.manifest)
    results = []
    for name in ('ct_1003', 'ct_1012'):
        mark = perf_counter()
        row = next(row for row in manifest['cases'] if row['case_id']==name)
        entry = entry_from_source(row, Path(args.data_root))
        prepared = dict(dataset='imagechd', scope='technical', fold=0,
            technical=dict(roi_xyz=None, allow_role_overlap=True),
            manifest=dict(dataset='imagechd', cases=[dict(row, roles=['train', 'validation'])]),
            plan=dict(configurations={'3d_fullres': {'spacing': entry['processed_spacing']}}))
        prepared['identity'] = digest(prepared)
        config = dict(model='q_route1', technical=True, device='cpu', seed=20261111,
            patch=32, local_size=32, pretrained=args.pretrained, parents={}, method='M07',
            max_candidates=2, work_root=str(root/'course'), run_dir=str(root/'course'/name),
            logging=dict(heartbeat_seconds=3600, case_details=True), execution={},
            technical_roles={stage:dict(train=[name], validation=[], test=[])
                             for stage in ('base','localizer','local')}, accumulation=4, updates=2)
        bundle = QAdapter(config, prepared, 'base')
        bundle.model.eval()
        image = torch.from_numpy(entry['image'])[None]
        with torch.inference_mode():
            logits = sliding_window_inference(image, [32]*3, 1, lambda x: bundle.model(x)[0],
                                              overlap=.5, mode='gaussian', device='cpu')
            expected = logits.softmax(1)[0]
            support = torch.from_numpy(allowed(entry['label'], entry['unknown_classes'], 'imagechd'))[None]
            valid = torch.from_numpy(entry['label']>=0)[None, None]
            expected_loss = float(segmentation_loss(expected.clamp_min(1e-8).log()[None], support, valid))
        with ValidationWorkspace(config, dict(case=name, stage='reference'), 1024**2) as workspace:
            probability = window_probabilities(entry['image'], bundle.model, [32]*3, 11,
                dict(overlap=.5, blend='gaussian', window_batch=1), 'cpu', workspace)
            difference = float(np.max(np.abs(probability-expected.numpy())))
            actual_loss = segmentation_statistics(probability, entry['label'], entry['unknown_classes'], 'imagechd', workspace)
            assert difference < 2e-5, difference
            assert abs(actual_loss-expected_loss) < 2e-6, (actual_loss, expected_loss)
        locator = Localizer().eval()
        with torch.inference_mode():
            reference = locator(image, valid)
            target, mask = build_heatmap_targets(support[0].numpy(), entry['label']>=0, entry['processed_spacing'])
            target_loss = float(localizer_loss(reference['logits'], target, mask))
        with ValidationWorkspace(config, dict(case=name, stage='localizer'), 1024**2) as workspace:
            out, features = localizer_forward(locator, entry['image'], entry['label']>=0, 'cpu', workspace)
            logit_error = float(np.max(np.abs(out-reference['logits'][0].numpy())))
            feature_error = float(np.max(np.abs(features-reference['features'][0].numpy())))
            assert logit_error < 1e-4, logit_error
            assert feature_error < 1e-4, feature_error
            loss = localizer_statistics(out, entry, 'imagechd', workspace)
            assert abs(loss-target_loss) < 2e-5, (loss, target_loss)
            heat = LayerInference(workspace, 'cpu').unary(out, torch.sigmoid)
            candidates = bounded_candidates(heat, entry['label'].shape, workspace)
            wanted = select_candidates(reference['heatmap'][0].numpy(), entry['label'].shape)
            assert [(c.start, c.source, c.task) for c in candidates]==[(c.start, c.source, c.task) for c in wanted]
        base_state = dict(model=bundle.model.state_dict(), spec=bundle.spec, stage='base',
            prepared=prepared, scope='technical', config=config, training_cases=[name],
            initialization_sha256='f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6',
            identity=digest(dict(case=name, stage='base')), best=expected_loss,
            history=[dict(validation_loss=expected_loss)])
        locator_state = dict(base_state, model=locator.state_dict(), stage='localizer', best=target_loss,
                             history=[dict(validation_loss=target_loss)])
        local_config = dict(config, parents={k:str(root/k) for k in ('base','localizer')})
        local = QAdapter(local_config, prepared, 'local', parent_states=dict(base=base_state, localizer=locator_state))
        local.model.eval()
        with torch.inference_mode():
            original, _ = local.predict(entry, internal=True)
        with ValidationWorkspace(config, dict(case=name, stage='local'), 1024**2) as workspace:
            probability = window_probabilities(entry['image'], local.base, [32]*3, 11,
                dict(overlap=.5, blend='gaussian', window_batch=1), 'cpu', workspace)
            actual = local_probability(local, entry, probability, workspace)
            local_error = float(np.max(np.abs(actual-original)))
            assert local_error < 1e-4, local_error
        results.append(dict(case_id=name, source_roi=[64]*3, processed_shape=list(entry['image'].shape),
            probability_error=difference, loss=actual_loss, reference_loss=expected_loss,
            locator_logit_error=logit_error, locator_feature_error=feature_error,
            locator_loss=loss, locator_reference_loss=target_loss, candidates=len(candidates),
            local_probability_error=local_error,
            seconds=perf_counter()-mark))
        del bundle, local, locator, reference, probability, features, out, expected, logits, base_state, locator_state
        gc.collect()
    work = root/'course/prediction-work'
    assert not work.exists() or not list(work.iterdir())
    record = dict(state='passed', cases=results, seconds=perf_counter()-started,
                  peak_rss_bytes=psutil.Process().memory_info().peak_wset if sys.platform=='win32' else None,
                  training_updates=0, gpu_used=False, temporary_files_remaining=0)
    write_json(report/'numeric.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def run_padding(args):
    started = perf_counter()
    torch.set_num_threads(4)
    row = next(row for row in read_json(args.manifest)['cases'] if row['case_id']=='ct_1003')
    entry = entry_from_source(row, Path(args.data_root))
    entry['image'] = entry['image'][:, :61, :57, :49].copy()
    entry['label'] = entry['label'][:61, :57, :49].copy()
    config = dict(work_root=str(Path(args.work_root)/'course'), run_dir=str(Path(args.work_root)/'padding'),
                  logging=dict(heartbeat_seconds=3600))
    locator = Localizer().eval()
    image, valid = torch.from_numpy(entry['image'])[None], torch.from_numpy(entry['label']>=0)[None,None]
    with torch.inference_mode():
        reference = locator(image, valid)
        target, mask = build_heatmap_targets(allowed(entry['label'], entry['unknown_classes'], 'imagechd'),
                                            entry['label']>=0, entry['processed_spacing'])
        expected = float(localizer_loss(reference['logits'], target, mask))
    with ValidationWorkspace(config, dict(case='ct_1003', stage='padding'), 1024**2) as workspace:
        logits, _ = localizer_forward(locator, entry['image'], entry['label']>=0, 'cpu', workspace)
        error = float(np.max(np.abs(logits-reference['logits'][0].numpy())))
        actual = localizer_statistics(logits, entry, 'imagechd', workspace)
        assert error < 1e-4 and abs(actual-expected) < 2e-5, (error, actual, expected)
    record = dict(state='passed', case_id='ct_1003', shape=[61,57,49], logit_error=error,
                  loss=actual, reference_loss=expected, seconds=perf_counter()-started, training_updates=0)
    write_json(Path(args.report)/'padding.json', record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('work-root', 'report', 'manifest', 'data-root', 'pretrained'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--padding-only', action='store_true')
    args = parser.parse_args()
    (run_padding if args.padding_only else run)(args)
