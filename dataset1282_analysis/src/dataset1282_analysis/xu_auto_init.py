from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .contracts import atomic_json, digest, read_json, sha256_file
from .xu_targets import _label_grid, _spacing, _write_array, anatomy, cache_coordinates

PROTOCOL = 'xu-auto-proximal-targets-v1'
POLICY = 'nearest-vessel-to-LV-RV-union-connected-physical-sphere-v1'
REQUIRED = ('LV', 'RV', 'LA', 'RA', 'MYO', 'AO', 'PA')


def _radius(radius_mm, margin_percent):
    values = np.asarray([radius_mm, margin_percent], dtype=float)
    if not np.isfinite(values).all() or radius_mm <= 0 or margin_percent < 0:
        raise ValueError('Radius must be finite and positive; margin must be finite and nonnegative')
    realized = float(radius_mm * (1 + margin_percent / 100))
    if not np.isfinite(realized):
        raise ValueError('Expanded radius is not finite')
    return realized


def auto_eligible(supervision, mapping):
    required = {mapping[key] for key in REQUIRED}
    return (required <= set(supervision['present_classes'])
            and not required.intersection(supervision['unknown_classes'])
            and not required.intersection(supervision['known_absent_classes']))


def _xyz(index, axes, origin, step):
    result = np.empty(3, dtype=float)
    result[axes] = origin + np.asarray(index) * step
    return result.tolist()


def auto_proximal_target(seg, geometry, mapping, radius_mm=30., margin_percent=20.):
    radius = _radius(radius_mm, margin_percent)
    seg = _label_grid(seg)
    if seg.ndim != 3 or seg.shape != tuple(geometry.cache_shape):
        raise ValueError('Automatic proximal target requires its original prepared 3D grid')
    if any(not np.any(seg == mapping[key]) for key in REQUIRED):
        raise ValueError('All seven proximal/chamber structures must be observed in the prepared array')
    axes, origin, step = cache_coordinates(geometry)
    step = _spacing(step, 3)
    ventricles = np.isin(seg, [mapping['LV'], mapping['RV']])
    boundary = ventricles & ~ndimage.binary_erosion(ventricles)
    heart_indices = np.argwhere(boundary)
    heart_tree = cKDTree(heart_indices * step)
    target = np.zeros(seg.shape, np.int8)
    target[seg == -1] = -1
    for idx, key in enumerate(REQUIRED[:5], 1):
        target[seg == mapping[key]] = idx
    qc = {}
    for key in ('AO', 'PA'):
        vessel = seg == mapping[key]
        points = np.argwhere(vessel)
        best_distance, seed, root = np.inf, None, None
        for start in range(0, len(points), 65536):
            block = points[start:start + 65536]
            distances, nearest = heart_tree.query(block * step)
            chosen = int(np.argmin(distances))
            if distances[chosen] < best_distance:
                best_distance = float(distances[chosen])
                seed = block[chosen].copy()
                root = heart_indices[int(nearest[chosen])].copy()
        extent = np.ceil(radius / step).astype(np.int64)
        lo = np.maximum(seed - extent, 0)
        hi = np.minimum(seed + extent + 1, seg.shape)
        region = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        squared = sum(((np.arange(a, b) - s) * pitch).reshape(
            tuple(-1 if axis == j else 1 for j in range(3))) ** 2
            for axis, (a, b, s, pitch) in enumerate(zip(lo, hi, seed, step)))
        clipped = vessel[region] & (squared <= radius ** 2 + 1e-8)
        components, _ = ndimage.label(clipped)
        selected = components == components[tuple(seed - lo)]
        target[region][selected] = 6
        selected_points = np.argwhere(selected) + lo
        low, high = selected_points.min(0), selected_points.max(0) + 1
        count = int(selected.sum())
        qc[key] = dict(seed_index=seed.tolist(), seed_xyz_mm=_xyz(seed, axes, origin, step),
                       ventricular_root_index=root.tolist(),
                       ventricular_root_xyz_mm=_xyz(root, axes, origin, step),
                       ventricular_root_class=int(seg[tuple(root)]),
                       root_distance_mm=best_distance, radius_mm=radius,
                       selected_voxels=count, total_vessel_voxels=len(points),
                       selected_fraction=count / len(points),
                       bbox_cache=[[int(a), int(b)] for a, b in zip(low, high)],
                       maximum_selected_seed_distance_mm=float(np.sqrt(squared[selected].max())))
    return target, dict(estimate='automatic approximate proximal range', reviewed_ostium=False,
                        coordinate_frame='local_array_excel_mm', spacing_array_mm=step.tolist(),
                        radius_mm=float(radius_mm), margin_percent=float(margin_percent),
                        realized_radius_mm=radius, vessels=qc)


def _contract(cache, radius_mm, margin_percent):
    from .xu_workflow import artifact_identity
    return dict(protocol=PROTOCOL, request=cache.request_digest, artifacts=artifact_identity(cache),
                anatomy=anatomy(cache), policy=POLICY, reviewed_ostium=False,
                radius_mm=float(radius_mm), margin_percent=float(margin_percent),
                realized_radius_mm=_radius(radius_mm, margin_percent),
                source={name: sha256_file(Path(__file__).with_name(name))
                        for name in ('xu_auto_init.py', 'xu_targets.py', 'xu_geometry.py')})


def validate_auto_contract(contract, cache):
    if contract != _contract(cache, contract['radius_mm'], contract['margin_percent']):
        raise ValueError('Automatic proximal target source or policy differs')


def _build_case(cache, name, mapping, out, radius, margin):
    if not auto_eligible(cache.supervision[name], mapping):
        return dict(available={}, skipped={'init': 'seven required structures are not all observed'})
    seg = cache.read_label(name)
    missing = [key for key in REQUIRED if not np.any(seg == mapping[key])]
    if missing:
        return dict(available={}, skipped={'init': 'structures absent from prepared array: ' + ','.join(missing)})
    target, qc = auto_proximal_target(seg, cache.geometry(name), mapping, radius, margin)
    label_hash = _write_array(out / 'cases' / name / 'init.b2nd', target)
    return dict(available={'init': label_hash}, skipped={}, qc=qc)


def build_auto_targets(cache, args, out):
    out = Path(out).resolve()
    radius, margin = getattr(args, 'radius_mm', 30.), getattr(args, 'margin_percent', 20.)
    contract = _contract(cache, radius, margin)
    mapping = contract['anatomy']
    names = [*cache.case_ids('train'), *cache.case_ids('development')]
    if len(set(names)) != len(names):
        raise ValueError('Training and development cases overlap')
    request, manifest = out / 'target-request.json', out / 'targets.json'
    if (request.exists() != args.resume or (args.resume and read_json(request) != contract)
            or (not args.resume and (manifest.exists() or (out / 'cases').exists()))):
        raise ValueError('Use a fresh directory or --resume with identical automatic target inputs')
    atomic_json(request, contract)
    cases = {}
    for name in names:
        marker = out / 'cases' / name / 'targets.json'
        if args.resume and marker.exists():
            saved = read_json(marker)
            if saved['contract'] != digest(contract):
                raise ValueError('Automatic target marker contract changed')
            record = saved['result']
            for kind, expected in record['available'].items():
                if kind != 'init' or sha256_file(marker.parent / 'init.b2nd') != expected:
                    raise ValueError('Committed automatic target differs')
        else:
            record = _build_case(cache, name, mapping, out, radius, margin)
            atomic_json(marker, dict(contract=digest(contract), result=record))
        cases[name] = record
    atomic_json(manifest, dict(contract=contract, cases=cases))
    report = dict(command='build-auto-targets', cases=len(cases), manifest=str(manifest),
                  estimate='automatic approximate proximal range', reviewed_ostium=False,
                  radius_mm=float(radius), margin_percent=float(margin),
                  realized_radius_mm=contract['realized_radius_mm'],
                  available={'blood': 0, 'init': sum('init' in row['available'] for row in cases.values())},
                  qc={name: row.get('qc', {'skipped': row['skipped']}) for name, row in cases.items()})
    atomic_json(out / 'auto-target-report.json', report)
    return report
