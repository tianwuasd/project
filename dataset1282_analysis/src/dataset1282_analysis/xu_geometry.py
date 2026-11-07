"""Read completed HU caches and map Xu-style network grids to native XYZ indices.

The cache spatial axes are the nnU-Net planner's permutation of reader ZYX.
All boxes are half-open intervals on voxel *edges*. Resizing samples voxel
centres, including for nearest-neighbour labels. Physical coordinates are local
Excel-grid coordinates, as specified by the 0.3.0 preprocessing contract.
"""
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy import ndimage

from .contracts import digest, read_json, sha256_file


def _triple(values, label):
    raw = tuple(values)
    if len(raw) != 3 or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in raw):
        raise ValueError(f'Expected three positive integer dimensions for {label}')
    return tuple(map(int, raw))


def _box_in(box, shape):
    if box is None:
        return tuple((0, n) for n in shape)
    if len(box) != 3:
        raise ValueError('Expected a three-dimensional half-open box')
    result = []
    for pair, n in zip(box, shape):
        if len(pair) != 2 or any(not isinstance(x, (int, np.integer)) for x in pair):
            raise ValueError('Expected integer box bounds')
        lo, hi = map(int, pair)
        if not 0 <= lo < hi <= n:
            raise ValueError('Empty or out-of-grid box')
        result.append((lo, hi))
    return tuple(result)


def _resize(volume, shape, order):
    if shape is None:
        return np.ascontiguousarray(volume)
    target = _triple(shape, 'network grid')
    if volume.shape == target:
        return np.ascontiguousarray(volume)
    return ndimage.zoom(volume, np.divide(target, volume.shape), order=order, mode='nearest',
                        grid_mode=True, prefilter=False)


def _scale_bounds(lo, hi, before, after):
    """Smallest target edge box covering the entire specified source box."""
    return lo * after // before, (hi * after + before - 1) // before


def _prediction(arr):
    arr = np.asarray(arr)
    if arr.ndim != 3 or not arr.size or arr.dtype.kind not in 'biu' or arr.min() < 0 or arr.max() > 10:
        raise ValueError('Expected a 3-D prediction of integral 0..10 class IDs')
    return arr


def prediction_box(pred64, pred128, cache_shape):
    """Enclose foreground predicted on either whole-volume grid. No truth fallback."""
    shape = _triple(cache_shape, 'prepared grid')
    boxes = []
    for pred in (pred64, pred128):
        pred = _prediction(pred)
        if not pred.any():
            continue
        bounds = []
        for axis in range(3):
            extents = np.flatnonzero(np.any(pred, axis=tuple(i for i in range(3) if i != axis)))
            bounds.append(_scale_bounds(int(extents[0]), int(extents[-1]) + 1,
                                        pred.shape[axis], shape[axis]))
        boxes.append(bounds)
    if not boxes:
        raise ValueError('Both localizers predicted empty foreground')
    return tuple((min(axis[0] for axis in pairs), max(axis[1] for axis in pairs))
                 for pairs in zip(*boxes))


@dataclass(frozen=True)
class CaseGeometry:
    """Coordinate map for positive-diagonal RAS working affines used by this dataset."""
    native_shape_xyz: tuple
    spacing_xyz: tuple
    transpose_forward: tuple
    bbox_transposed: tuple
    cache_shape: tuple

    def __post_init__(self):
        _triple(self.native_shape_xyz, 'native grid')
        _triple(self.cache_shape, 'prepared grid')
        if tuple(sorted(self.transpose_forward)) != (0, 1, 2):
            raise ValueError('Invalid planner axis permutation')
        if len(self.spacing_xyz) != 3 or not all(np.isfinite(s) and s > 0 for s in self.spacing_xyz):
            raise ValueError('Invalid native spacing')
        _box_in(self.bbox_transposed, self.native_transposed_shape)

    @property
    def native_transposed_shape(self):
        return tuple(self.native_shape_xyz[2-i] for i in self.transpose_forward)

    def cache_box_to_native(self, box):
        """Conservative enclosing half-open XYZ box, including partial edge voxels."""
        xyz = [None]*3
        for axis, ((lo, hi), (start, stop), count) in enumerate(zip(
                _box_in(box, self.cache_shape), self.bbox_transposed, self.cache_shape)):
            first, last = _scale_bounds(lo, hi, count, stop-start)
            xyz[2-self.transpose_forward[axis]] = (start+first, start+last)
        return tuple(xyz)

    def restore_label(self, pred, box=None):
        """Directly sample a whole-grid or in-ROI network prediction on native centres."""
        pred = _prediction(pred)
        result = np.zeros(self.native_transposed_shape, dtype=np.uint8)
        native_slices, sample = [], []
        for axis, ((low, high), (start, stop), count) in enumerate(zip(
                _box_in(box, self.cache_shape), self.bbox_transposed, self.cache_shape)):
            length = stop - start
            # Native voxel centre -> cache edge coordinate -> network cell index.
            # Integer arithmetic gives deterministic ties at exact cell boundaries.
            offset = (2*np.arange(length, dtype=np.int64)+1)*count - 2*length*low
            indices = (offset*pred.shape[axis]) // (2*length*(high-low))
            valid = np.flatnonzero((indices >= 0) & (indices < pred.shape[axis]))
            if not valid.size:
                break  # subvoxel ROI containing no native centre
            native_slices.append(slice(start+int(valid[0]), start+int(valid[-1])+1))
            sample.append(indices[valid])
        else:
            result[tuple(native_slices)] = pred[np.ix_(*sample)]
        return result.transpose(np.argsort(self.transpose_forward)).T


class PreparedCache:
    """A completed, role-checked 0.3.0 preparation. Large arrays verify on first use."""

    def __init__(self, run_dir, *, allow_technical_subset=False):
        self.run_dir = Path(run_dir).resolve()
        request = read_json(self._path('request.json'))
        complete = read_json(self._path('complete.json'))
        self.request_digest = digest(request)
        self.scope = request['scope']
        if (complete['state'] != 'completed' or complete['request_digest'] != self.request_digest
                or self.scope != complete['scope'] or request['method'] != 'HU-marginal-v1'
                or complete['coordinate_frame'] != 'local_array_excel_mm'
                or request['nnunetv2'] != '2.8.1'):
            raise ValueError('Expected a completed matching HU-marginal-v1 run')
        if self.scope != 'all_eligible_source_index_scans' and not (
                self.scope == 'technical_subset' and allow_technical_subset):
            raise ValueError('Technical subset requires explicit admission')
        self.staged_extension = request['staged_extension']
        if self.staged_extension not in ('.nii', '.nii.gz'):
            raise ValueError('Unsupported staged file type')
        self._records = request['cases']
        if not self._records or any(not re.fullmatch(r'[A-Za-z0-9_-]+', n) for n in self._records):
            raise ValueError('Invalid or empty set of cases')
        self._verified = {}
        self._dataset = 'preprocessed/Dataset282_CHDSource'
        contract = read_json(self._path(f'{self._dataset}/training_contract.json'))
        required = {'splits_final.json','dataset.json','nnUNetPlans.json','partial_supervision.json'}
        if contract['request_digest'] != self.request_digest or set(contract['artifacts']) != required:
            raise ValueError('Invalid training contract')
        for file, expected in contract['artifacts'].items():
            self._check_file(f'{self._dataset}/{file}', expected)
        self.supervision = read_json(self._path(f'{self._dataset}/partial_supervision.json'))
        plans = read_json(self._path(f'{self._dataset}/nnUNetPlans.json'))
        fit = read_json(self._artifact('fit','fit.json'))
        identifier = plans['configurations']['3d_fullres']['data_identifier']
        if (not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or fit['data_identifier'] != identifier
                or fit != complete['fit'] or fit['plans_sha256'] != contract['artifacts']['nnUNetPlans.json']
                or plans['image_reader_writer'] != 'NibabelIOWithReorient'):
            raise ValueError('Unexpected plan, fit or image orientation convention')
        self._folder = f'{self._dataset}/{identifier}'
        self._forward = tuple(plans['transpose_forward'])
        if sorted(self._forward) != [0,1,2] or list(np.argsort(self._forward)) != plans['transpose_backward']:
            raise ValueError('Inconsistent planner axis permutations')
        cohort = read_json(self._path('cohort.json'))
        if digest(cohort) != request['cohort_digest']:
            raise ValueError('Frozen cohort digest mismatch')
        split = read_json(self._path(f'{self._dataset}/splits_final.json'))
        self._validate_roles(cohort['cases'], split, fit, complete)

    def _validate_roles(self, frozen, split, fit, complete):
        eligible = {n for n,r in frozen.items() if r.get('included') and r.get('index_scan')
                    and r['role'] in ('train', 'development')}
        selected = set(self._records)
        if (selected - eligible or set(self.supervision) != selected or
                (self.scope == 'all_eligible_source_index_scans' and selected != eligible)):
            raise ValueError('Selection does not match eligible source index scans')
        patients = set()
        for n,r in self._records.items():
            patient = frozen[n].get('patient')
            if not patient or patient in patients or r['role'] != frozen[n]['role']:
                raise ValueError('Duplicate patient or frozen role mismatch')
            patients.add(patient)
            info = self.supervision[n]
            classes = info['present_classes'] + info['known_absent_classes'] + info['unknown_classes']
            if (sorted(classes) != list(range(1,11)) or not info['present_classes'] or
                    info['ambiguous_background_allowed'] != [0]+info['unknown_classes'] or
                    sorted(info['known_absent_classes']) != sorted(r['known_absent'])):
                raise ValueError('Case supervision is inconsistent')
        train, dev = self.case_ids('train'), self.case_ids('development')
        if (len(split) != 1 or tuple(sorted(split[0]['train'])) != train or tuple(sorted(split[0]['val'])) != dev
                or tuple(sorted(fit['fit_cases'])) != train or complete['train_cases'] != len(train)
                or complete['development_cases'] != len(dev)):
            raise ValueError('Fit, split and frozen roles disagree')

    def _path(self, relative):
        path = self.run_dir/relative
        if path.resolve() != path or self.run_dir not in path.parents:
            raise ValueError('Linked or escaping cache path')
        return path

    def _check_file(self, relative, expected):
        path = self._path(relative)
        stat = path.stat()
        identity = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, expected)
        if self._verified.get(relative) != identity:
            if sha256_file(path) != expected:
                raise ValueError(f'Changed preprocessing artifact: {relative}')
            self._verified[relative] = identity
        return path

    def _artifact(self, job, relative):
        marker = read_json(self._path(f'commits/{job}.json'))
        if marker['contract'] != self.request_digest or marker['job']['id'] != job:
            raise ValueError('Cache artifact belongs to a different run')
        return self._check_file(relative, marker['artifacts'][relative])

    def case_ids(self, role, *, complete_only=False):
        if role not in ('train','development'):
            raise ValueError('Only frozen source train and development roles are admitted')
        return tuple(sorted(n for n,r in self._records.items() if r['role'] == role and
                            (not complete_only or len(self.supervision[n]['present_classes']) == 10)))

    def geometry(self, name):
        record = self._records[name]
        qc = read_json(self._artifact(f'process-{name}', f'qc/{name}.json'))
        shape = qc['prepared_shape']
        if len(shape) != 4 or shape[0] != 1 or qc['supervision'] != self.supervision[name]:
            raise ValueError('Invalid cached geometry or supervision')
        meta = read_json(self._artifact(f'stage-{name}', f'staged/{name}/metadata.json'))
        spacing = tuple(record['spacing'])
        if (meta['case'] != name or tuple(meta['source_spacing_xyz']) != spacing or
                not np.allclose(meta['working_affine'], np.diag([*spacing,1]), rtol=0, atol=1e-6)):
            raise ValueError('Unsupported working orientation or spacing')
        return CaseGeometry(tuple(record['shape']), spacing, self._forward,
                            tuple(map(tuple,qc['bbox_transposed'])), tuple(shape[1:]))

    def _volume(self, name, shape, box, label):
        import blosc2
        geometry = self.geometry(name)
        suffix = '_seg.b2nd' if label else '.b2nd'
        path = self._artifact(f'process-{name}', f'{self._folder}/{name}{suffix}')
        array = blosc2.open(str(path), mode='r')
        if tuple(array.shape) != (1,*geometry.cache_shape):
            raise ValueError('Compressed array disagrees with committed geometry')
        region = _box_in(box, geometry.cache_shape)
        slices = (slice(0,1), *(slice(a,b) for a,b in region))
        values = np.asarray(array[slices])[0]
        if label:
            if values.dtype.kind not in 'iu' or values.min() < -1 or values.max() > 10:
                raise ValueError('Invalid segmentation classes')
        elif values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Invalid prepared intensities')
        return _resize(values, shape, 0 if label else 1)

    def read_image(self, name, shape=None, box=None):
        return self._volume(name,shape,box,False)

    def read_label(self, name, shape=None, box=None):
        return self._volume(name,shape,box,True)

    def read_native_label(self, name):
        import nibabel as nib
        geom = self.geometry(name)
        path = self._artifact(f'stage-{name}',f'staged/{name}/label{self.staged_extension}')
        volume = nib.load(path)
        if volume.shape != geom.native_shape_xyz or not np.allclose(
                volume.affine, np.diag([*geom.spacing_xyz,1]), rtol=1e-6, atol=1e-7):
            raise ValueError('Native evaluation grid mismatch')
        return np.asanyarray(volume.dataobj)
