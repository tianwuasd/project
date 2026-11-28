from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

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
    image = entry['image']
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


def validate_records(bundle, dataset, names):
    records = []
    config = bundle.config
    settings = config['execution']
    bundle.clear_frozen()
    for completed, name in enumerate(names, 1):
        started = perf_counter()
        entry = dataset.get(name)
        entry = dict(entry, processed_spacing=entry.get('processed_spacing',
                     bundle.prepared['plan']['configurations']['3d_fullres']['spacing']))
        identity = dict(prepared=bundle.prepared['identity'], stage=bundle.stage, case_id=name,
                        run_dir=config['run_dir'])
        policy = getattr(bundle, 'validation_policy', None)
        block_bytes = settings.get('validation_block_bytes', 64*1024**2)
        if bundle.stage == 'localizer' and policy is None:
            block_bytes = settings.get('localizer_block_bytes',2*1024**3)
        array_budget, storage, rank, quota = 0, 'disk', 0, 0
        if policy is not None:
            quota, rank = policy['quota_bytes'], policy['rank']
            demand = policy['demands'][name] if 'demands' in policy else policy['demand']
            if 'handle' in policy:
                handle = policy['handle']
                with handle.condition:
                    finished = handle.state['q_validation_finished']
                available = sum(value for done,value in zip(finished,policy['shared_quotas']) if done)
                quota += available // max(1, finished.count(False))
            storage = settings.get('validation_storage', 'auto')
            if storage == 'auto':
                storage = 'memory' if demand['arrays'] + demand['minimum'] <= quota else 'disk'
            array_budget = demand['arrays'] if storage == 'memory' else 0
            block_bytes = min(demand['transient'], quota - array_budget)
            if block_bytes < demand['minimum']:
                raise MemoryError('验证内存额度不能容纳指定存储方式和最小计算块')
        guard = getattr(bundle, 'validation_guard', None)
        root = Path(config['run_dir']) / bundle.stage
        event(root, 'q_validation_case_started', case_id=name, rank=rank,
              storage=storage, block_bytes=block_bytes, array_budget_bytes=array_budget, quota_bytes=quota)
        claim = guard.admit(quota, label='q_validation_case:'+name, reserve=True) if guard is not None else nullcontext(True)
        with claim as ready, ValidationWorkspace(dict(config, run_dir=str(root)), identity, block_bytes, guard,
                storage=storage, array_budget_bytes=array_budget, rank=rank) as workspace:
            if ready is None:
                raise InterruptedError('验证额度等待已取消')
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
        if config['logging']['case_details']:
            event(root, 'q_validation_case_completed', case_id=name, rank=rank,
                  cases_completed=completed, cases_total=len(names), loss=value,
                  execution=storage, block_bytes=block_bytes, seconds=perf_counter()-started)
        records.append(dict(case_id=name, loss=value))
        if hasattr(dataset, 'clear'):
            dataset.clear()
    return records


def validate(bundle, dataset, names):
    from .validation_runtime import aggregate
    rows = {row['case_id']: row for row in bundle.prepared['manifest']['cases']}
    return aggregate(validate_records(bundle, dataset, names), names, rows)
