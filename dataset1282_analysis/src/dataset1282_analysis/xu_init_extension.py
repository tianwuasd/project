import math
from pathlib import Path

from .contracts import atomic_json, digest, read_json


STAGES = ('auto-targets', 'auto-init64', 'auto-init128', 'auto-six-way')


def bind_extension(args, out, contract):
    path = out/'.runtime/auto-init-extension.json'
    enabled = getattr(args, 'auto_init', False)
    if path.exists() and not enabled:
        raise ValueError('This run requires explicit --auto-init on resume')
    if not enabled:
        if getattr(args, 'stop_after', None) in STAGES:
            raise ValueError('Auto-init stages require --auto-init')
        return None
    radius = getattr(args, 'init_radius_mm', 30.0)
    margin = getattr(args, 'init_margin_percent', 20.0)
    if (not math.isfinite(radius) or radius <= 0 or not math.isfinite(margin) or margin < 0
            or not math.isfinite(radius * (1 + margin / 100))):
        raise ValueError('Auto-init radius must be positive and margin nonnegative, both finite')
    binding = dict(protocol='xu-auto-init-extension-v1', base_contract=digest(contract),
                   request=contract['request'], artifacts=contract['artifacts'],
                   method='xu-auto-proximal-targets-v1', radius_mm=radius, margin_percent=margin,
                   routing=dict(targets='auto-targets', init64='auto-init64', init128='auto-init128',
                                fusion='auto-six-way', blood_targets='targets'))
    if path.exists():
        if read_json(path) != binding:
            raise ValueError('Auto-init extension settings or base run changed')
    else:
        occupied = [out/'result.json']
        for name in ('with-blood', 'vessels', *STAGES):
            occupied += [out/name, out/'receipts'/(name+'.json'), out/'started'/(name+'.json')]
        if any(p.exists() for p in occupied):
            raise ValueError('Auto-init extension requires untouched downstream outputs and fresh extension stages')
        atomic_json(path, binding)
    return binding


def eligible_counts(cache, targets, supervision):
    from .xu_targets import TargetStore
    store = TargetStore(targets, cache)
    return {role: sum(store.has(name, 'init') for name in
                      cache.case_ids(role, complete_only=supervision == 'complete'))
            for role in ('train', 'development')}


def extension_identity(stage, base_id, extension):
    if extension and stage.name in (*STAGES, 'with-blood', 'vessels'):
        return digest(dict(base=base_id, extension=extension))
    return base_id
