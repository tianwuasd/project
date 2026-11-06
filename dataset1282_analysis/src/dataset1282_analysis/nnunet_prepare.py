"""Numerical worker for Scheme A. Run from the isolated research interpreter."""
from importlib.metadata import version
import math
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

from .contracts import atomic_json, read_json, sha256_file
from .resources import admit_case, budget, preprocessing_memory, snapshot

DATASET = 'Dataset282_CHDSource'


def map_intensity(data, affine):
    """Estimate robust percentiles from an image-only low-resolution body proxy."""
    # Use canonical axial slices for filling internal air spaces, including lung.
    view = nib.orientations.apply_orientation(data, nib.orientations.io_orientation(affine))
    sample = view[::4, ::4, ::2]
    if not np.isfinite(data).all() or sample.min() == sample.max():
        raise ValueError('Nonfinite or constant image')
    components, count = ndimage.label(sample > threshold_otsu(sample))
    if not count:
        raise ValueError('Empty body proxy')
    sizes = np.bincount(components.ravel())
    mask = components == (1 + sizes[1:].argmax())
    for z in range(mask.shape[2]):
        mask[:, :, z] = ndimage.binary_fill_holes(mask[:, :, z])
    lo, hi = np.percentile(sample[mask], (.5, 99.5))
    if not hi > lo:
        raise ValueError('Body percentile window is degenerate')
    nonzero = data != 0
    np.clip(data, lo, hi, out=data)
    data -= lo
    data *= .999 / (hi - lo)
    data += .001
    data[~nonzero] = 0  # preserve the support used by the automatic nonzero crop
    return [float(lo), float(hi)]


def reviewed_affine(record, image, label, mode):
    if image.ndim != 3 or image.shape != label.shape:
        raise ValueError('Expected same-sized 3-D indexed arrays')
    if mode in ('voxel-smoke', 'excel-pilot'):
        if not np.allclose(image.affine, label.affine, rtol=0, atol=1e-5):
            raise ValueError('Index smoke requires matching source grids')
        if mode == 'excel-pilot':
            # Local array axes with Excel sampling. This does not recover the DICOM patient frame.
            return np.diag([*record['excel_geometry']['source_spacing_xyz_mm'], 1.])
        return np.eye(4)
    affine = np.asarray(record['review']['affine_ras_mm'], dtype=float)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1]) or abs(np.linalg.det(affine[:3, :3])) < 1e-9):
        raise ValueError('Review requires a nonsingular voxel-index to RAS-mm affine')
    axes = affine[:3, :3] / nib.affines.voxel_sizes(affine)
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
        raise ValueError('Sheared input needs independent physical geometry correction')
    return affine


def stage_pair(folder, name, record, mode, intensity):
    for kind in ('image', 'label'):
        if sha256_file(record[kind]) != record['sha256'][kind]:
            raise ValueError('Source hash mismatch: ' + name)
    image, label = [nib.load(record[k]) for k in ('image', 'label')]
    affine = reviewed_affine(record, image, label, mode)
    data = image.get_fdata(dtype=np.float32, caching='unchanged')
    if intensity == 'verified_hu':
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite input HU')
        window = None  # CTNormalization learns one shared window on the train partition.
    else:
        window = map_intensity(data, affine)
    # Read before uint8 conversion; neither scaling nor fractional labels may be lost.
    seg = np.asanyarray(label.dataobj)
    if (not np.isfinite(seg).all() or seg.min() < 0 or seg.max() > 10 or seg.max() == 0
            or (seg.dtype.kind == 'f' and not np.equal(seg, np.floor(seg)).all())):
        raise ValueError('Expected finite integer labels 0..10 and nonempty foreground')
    files = {}
    for key, array, suffix in [('images', data, '_0000'), ('label', seg.astype(np.uint8), '')]:
        path = folder / (name + suffix + '.nii.gz')
        volume = nib.Nifti1Image(array, affine)
        volume.header.set_xyzt_units('unknown' if mode == 'voxel-smoke' else 'mm')
        nib.save(volume, path)
        files[key] = [str(path)] if key == 'images' else str(path)
    return files, {'shape_xyz': image.shape, 'original_image_affine': image.affine.tolist(),
                   'original_label_affine': label.affine.tolist(), 'working_affine': affine.tolist(),
                   'intensity_percentiles': window, 'units': 'index' if mode == 'voxel-smoke' else 'mm',
                   'coordinate_frame': {'voxel-smoke': 'array_index', 'excel-pilot': 'local_array_excel_sampling',
                                        'physical': 'reviewed_patient_RAS'}[mode]}


def roundtrip(case_dir, name, original, plans):
    """Read official arrays; independently return the prepared label to the voxel grid."""
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
    data, seg, _, props = nnUNetDatasetBlosc2(str(case_dir), [name]).load_case(name)
    if data.shape != seg.shape:
        raise ValueError('Prepared CT/label shapes differ')
    for z in range(0, data.shape[1], 8):
        if not np.isfinite(data[:, z:z+8]).all():
            raise ValueError('Nonfinite prepared intensities')
    seg = seg[0, :, :, :]
    if seg.min() < -1 or seg.max() > 10:
        raise ValueError('Unexpected prepared label value')
    crop_shape = props['shape_after_cropping_and_before_resampling']
    if tuple(seg.shape) != tuple(crop_shape):
        seg = ndimage.zoom(seg, np.divide(crop_shape, seg.shape), order=0,
                           mode='nearest', grid_mode=True, prefilter=False)
    restored = np.zeros(props['shape_before_cropping'], dtype=np.uint8)
    restored[tuple(slice(a, b) for a, b in props['bbox_used_for_cropping'])] = np.maximum(seg, 0)
    # Match the reader's canonical RAS, XYZ->ZYX, then the planner's axis permutation.
    native = np.asanyarray(nib.as_closest_canonical(nib.load(original)).dataobj).T
    restored = restored.transpose(plans['transpose_backward'])
    if native.shape != restored.shape:
        raise ValueError('Inverse geometry shape mismatch')
    old, new, same = np.zeros((3, 11), dtype=np.int64)
    for z in range(0, native.shape[0], 8):
        a, b = native[z:z+8], restored[z:z+8]
        old += np.bincount(a.ravel(), minlength=11)
        new += np.bincount(b.ravel(), minlength=11)
        same += np.bincount(a[a == b], minlength=11)
    return {'prepared_shape': list(data.shape), 'bbox_transposed': props['bbox_used_for_cropping'],
            'classes': {str(i): {'original_voxels': int(old[i]), 'returned_voxels': int(new[i]),
                                 'dice': float(2 * same[i] / (old[i]+new[i])) if old[i]+new[i] else None}
                        for i in range(1, 11)}}


def run(output):
    if version('nnunetv2') != '2.8.1':
        raise RuntimeError('This preprocessing contract requires nnunetv2==2.8.1')
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    import torch

    torch.set_num_threads(1)
    req = read_json(output / 'request.json')
    raw = output / 'raw' / DATASET
    dest = output / 'preprocessed' / DATASET
    pairs = raw / 'pairs'
    pairs.mkdir(parents=True)
    (output / 'results').mkdir()
    dest.mkdir(parents=True)
    policy = req['policy']
    ceiling = budget(snapshot(), policy)

    def admit(n, resampled=0):
        estimate = preprocessing_memory(n, resampled)
        decision = admit_case(ceiling, budget(snapshot(), policy), [], estimate)
        if not decision['allowed']:
            raise RuntimeError(f"CPU/RAM admission failed: {decision['reason']}; reserved {estimate} bytes, "
                               f"stage budget {ceiling['memory_bytes']} bytes")

    cases = req['cases']
    train = sorted(k for k, v in cases.items() if v['role'] == 'train')
    val = sorted(k for k, v in cases.items() if v['role'] == 'development')
    probe = sorted(k for k, v in cases.items() if v['role'] == 'pilot_probe' and req['mode'] == 'excel-pilot')
    if not train or (req['mode'] == 'physical' and not val) or len(train) + len(val) + len(probe) != len(cases):
        raise ValueError('Worker accepts source train and development only')
    intensity = req.get('intensity', 'body_percentile')
    descriptor = {'channel_names': {'0': 'CT' if intensity == 'verified_hu' else 'noNorm'}, 'labels': {'background': 0, **{
        'label_' + str(i): i for i in range(1, 11)}}, 'file_ending': '.nii.gz',
        'overwrite_image_reader_writer': 'NibabelIOWithReorient'}
    files, qc = {}, {}
    def stage(name):
        admit(math.prod(nib.load(cases[name]['image']).shape))
        files[name], qc[name] = stage_pair(pairs, name, cases[name], req['mode'], intensity)

    for name in train:
        stage(name)
    fit_files = {name: {'images': [str(Path(files[name]['images'][0]).relative_to(raw).as_posix())],
                        'label': Path(files[name]['label']).relative_to(raw).as_posix()} for name in train}
    fit_descriptor = dict(descriptor, dataset=fit_files, numTraining=len(train))
    atomic_json(raw / 'dataset.json', fit_descriptor)
    extractor = DatasetFingerprintExtractor(DATASET, num_processes=1)
    # Match the full-size default, avoiding 100M repeated samples in miniature cohorts.
    extractor.num_foreground_voxels_for_intensitystats = min(100_000_000, 100_000 * len(train))
    extractor.run()
    plans = ExperimentPlanner(DATASET).plan_experiment()
    manager = PlansManager(plans)
    conf = manager.get_configuration('3d_fullres')
    processor = conf.preprocessor_class(verbose=False)
    target = dest / conf.data_identifier
    target.mkdir()
    fit_hash = sha256_file(dest / 'dataset_fingerprint.json')
    for name in train + val + probe:
        if name not in files:
            stage(name)
        entry = files[name]
        volume = nib.as_closest_canonical(nib.load(entry['images'][0]))
        perm = plans['transpose_forward']
        axes = np.array(volume.shape[::-1])[perm]
        spacing = np.array(volume.header.get_zooms()[::-1])[perm]
        admit(math.prod(volume.shape), int(np.prod(np.maximum(1, np.ceil(axes * spacing / conf.spacing)))))
        processor.run_case_save(str(target / name), entry['images'], entry['label'], manager, conf, descriptor)
        qc[name]['roundtrip'] = roundtrip(target, name, entry['label'], plans)
        atomic_json(output / 'qc' / (name + '.json'), qc[name])
    # Official trainer fold 0, validation GT in the original calibrated index grid.
    import shutil
    gt = dest / 'gt_segmentations'
    gt.mkdir()
    for name in val:
        shutil.copyfile(files[name]['label'], gt / (name + '.nii.gz'))
    if req['mode'] == 'physical':
        atomic_json(dest / 'splits_final.json', [{'train': train, 'val': val}])
    atomic_json(dest / 'dataset.json', dict(descriptor, numTraining=len(cases)))
    atomic_json(output / 'complete.json', {'mode': req['mode'], 'nnunetv2': version('nnunetv2'),
                'configuration': '3d_fullres', 'train_cases': len(train), 'development_cases': len(val),
                'fit_cases': train, 'pilot_probe_cases': probe,
                'units': 'index' if req['mode'] == 'voxel-smoke' else 'mm',
                'spacing_xyz': np.array(conf.spacing)[plans['transpose_backward']][::-1].tolist(),
                'coordinate_frame': qc[train[0]]['coordinate_frame'],
                'spacing_plan_axes': list(conf.spacing), 'transpose_forward': plans['transpose_forward'],
                'intensity': intensity,
                'fingerprint_samples': extractor.num_foreground_voxels_for_intensitystats,
                'worker_sha256': sha256_file(__file__), 'request_sha256': sha256_file(output / 'request.json'),
                'fingerprint_sha256': fit_hash, 'plans_sha256': sha256_file(dest / 'nnUNetPlans.json'),
                'cohort_digest': req['cohort_digest'], 'training_executed': False})


if __name__ == '__main__':
    run(Path(sys.argv[1]))
