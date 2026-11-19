from pathlib import Path

import numpy as np
import torch

from ...observability.events import event
from ...prediction.blocks import window_probabilities
from ..stunet.streaming import tiles
from .comparisons import infer_local
from .fusion import core_weight
from .geometry import crop_local_input, sample_context
from .validation_candidates import bounded_candidates
from .validation_loss import segmentation_statistics
from .validation_network import LayerInference, localizer_forward
from .validation_store import ValidationWorkspace
from .validation_targets import localizer_statistics


def validity(entry, workspace, use_label):
    result = workspace.create('layer', entry['image'].shape[1:], np.bool_)
    for section in tiles(result.shape, max(1, workspace.block_bytes // 16)):
        with workspace.boundary():
            result[section] = entry['label'][section] >= 0 if use_label else True
        workspace.release(entry.get('label'))
    return result


def local_probability(bundle, entry, probability, workspace):
    from .adapter import SOLVER

    valid = validity(entry, workspace, False)
    logits, features = localizer_forward(bundle.locator, entry['image'], valid, bundle.device, workspace)
    workspace.drop(valid)
    heat = LayerInference(workspace, bundle.device).unary(logits, torch.sigmoid)
    workspace.drop(logits)
    candidates = bounded_candidates(heat, entry['image'].shape[1:], workspace)[:bundle.config['max_candidates']]
    workspace.drop(heat)
    total = workspace.create('layer', probability.shape, np.float32)
    weights = workspace.create('layer', probability.shape[1:], np.float32)
    image = torch.from_numpy(entry['image'])[None]
    base = torch.from_numpy(probability)[None]
    context_map = torch.from_numpy(features)[None]
    size = bundle.config['local_size']
    margin, core_size = size//8, size-2*(size//8)
    for index, candidate in enumerate(candidates, 1):
        with workspace.boundary(), torch.inference_mode():
            start = tuple(int(np.floor(c-(size-1)/2+.5)) for c in candidate.center)
            inputs, editable = crop_local_input(image, base, start, size)
            context = sample_context(context_map, candidate.center, samples=size//8)
            output = bundle.model(inputs.to(bundle.device), context.to(bundle.device), editable.to(bundle.device))
            final, _ = infer_local(output, editable.to(bundle.device), bundle.config['method'], SOLVER)
            core = (slice(None), slice(None), *output['core'])
            local_weight = core_weight(core_size, width=min(8, core_size//3))*editable[core][0, 0]
            scores = final.softmax(1)[0].cpu().numpy()
            weight = local_weight.numpy()
            low = tuple(s+margin for s in start)
            dst = tuple(slice(max(0, s), min(n, s+core_size)) for s, n in zip(low, weights.shape))
            src = tuple(slice(t.start-s, t.stop-s) for t, s in zip(dst, low))
            if all(t.stop > t.start for t in dst):
                total[(slice(None), *dst)] += scores[(slice(None), *src)]*weight[src]
                weights[dst] += weight[src]
            del inputs, editable, context, output, final, scores, weight
        workspace.release(entry['image'])
        workspace.progress('q_local_validation', index, len(candidates))
    del context_map
    workspace.drop(features)
    for section in tiles(weights.shape, max(1, workspace.block_bytes // 256)):
        with workspace.boundary():
            w = np.array(weights[section], copy=True)
            p = np.array(probability[(slice(None), *section)], copy=True)
            sums = np.array(total[(slice(None), *section)], copy=True)
            result = (np.maximum(0, 1-w)*p+sums)/np.maximum(1, w)
            probability[(slice(None), *section)] = np.where(w[None] == 0, p, result)
        workspace.release()
    workspace.drop(total)
    workspace.drop(weights)
    return probability


def validate(bundle, dataset, names):
    patients = {}
    config = bundle.config
    rows = {row['case_id']: row for row in bundle.prepared['manifest']['cases']}
    aggregation = 'patient' if all(rows[name].get('patient') for name in names) else 'case'
    block_bytes = config['execution'].get('validation_block_bytes', 64*1024**2)
    bundle.current_name, bundle.current_frozen = None, None
    for completed, name in enumerate(names, 1):
        entry = dataset.get(name)
        entry = dict(entry, processed_spacing=entry.get('processed_spacing',
                     bundle.prepared['plan']['configurations']['3d_fullres']['spacing']))
        identity = dict(prepared=bundle.prepared['identity'], stage=bundle.stage, case_id=name,
                        run_dir=config['run_dir'])
        with ValidationWorkspace(config, identity, block_bytes, getattr(bundle, 'validation_guard', None)) as workspace:
            if bundle.stage == 'localizer':
                valid = validity(entry, workspace, True)
                logits, features = localizer_forward(bundle.model, entry['image'], valid, bundle.device, workspace)
                workspace.drop(features)
                value = localizer_statistics(logits, entry, bundle.prepared['dataset'], workspace)
            else:
                model = bundle.model if bundle.stage == 'base' else bundle.base
                probability = window_probabilities(entry['image'], model, bundle.spec['patch'], 11,
                    dict(overlap=.5, blend='gaussian', window_batch=1), bundle.device, workspace)
                if bundle.stage == 'local':
                    probability = local_probability(bundle, entry, probability, workspace)
                value = segmentation_statistics(probability, entry['label'], entry['unknown_classes'],
                                                bundle.prepared['dataset'], workspace)
            if not np.isfinite(value):
                raise FloatingPointError('Q验证损失非有限')
            workspace.release(entry['image'], entry['label'])
        key = rows[name]['patient'] if aggregation == 'patient' else name
        patients.setdefault(key, []).append(value)
        if config['logging']['case_details']:
            event(Path(config['run_dir'])/bundle.stage, 'q_validation_case_completed', case_id=name,
                  cases_completed=completed, cases_total=len(names), loss=value,
                  aggregation_unit=aggregation, execution='disk_blocks', block_bytes=block_bytes)
    return float(np.mean([np.mean(values) for values in patients.values()]))
