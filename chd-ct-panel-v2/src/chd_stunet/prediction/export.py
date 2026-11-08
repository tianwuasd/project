import os
from pathlib import Path
import pickle
import time

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from ..observability.runlog import write_json
from ..preprocessing.cohort import TARGETS, map_labels
from ..preprocessing.stunet_s import corrected_affine
from .model import file_sha, read_json


def surface_points(mask, affine):
    border = mask & ~binary_erosion(mask, border_value=0)
    return nib.affines.apply_affine(affine, np.argwhere(border))


def score_segmentation(prediction, reference, affine, progress=None):
    if prediction.shape != reference.shape:
        raise ValueError('预测与标签体素网格不同')
    valid = reference != 8
    result = {}
    voxel_ml = abs(np.linalg.det(np.asarray(affine)[:3, :3]))/1000
    for name, label in TARGETS.items():
        if label not in range(1, 8):
            continue
        if progress:
            progress('metrics_'+name)
        expected, actual = (reference == label) & valid, (prediction == label) & valid
        n, p = int(expected.sum()), int(actual.sum())
        if not n and not p:
            dice, hd, status = None, None, 'both_empty'
        elif not n or not p:
            dice, hd = 0., None
            status = 'missing_prediction' if n else 'unexpected_prediction'
        else:
            dice = 2*int((actual & expected).sum())/(n+p)
            first, second = surface_points(actual, affine), surface_points(expected, affine)
            forward = cKDTree(second).query(first, workers=1)[0]
            backward = cKDTree(first).query(second, workers=1)[0]
            hd = float(max(np.percentile(forward, 95), np.percentile(backward, 95)))
            status = 'scored'
        result[name] = dict(dice=dice, hd95_mm=hd, status=status,
                            prediction_voxels=p, reference_voxels=n,
                            prediction_volume_ml=p*voxel_ml, reference_volume_ml=n*voxel_ml)
    return result


def save_nifti(path, array, affine):
    path = Path(path)
    temporary = path.with_name(path.name.removesuffix('.nii.gz')+'.partial.nii.gz')
    image = nib.Nifti1Image(array, affine)
    image.header.set_xyzt_units('mm')
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, temporary)
    os.replace(temporary, path)


def restore_probabilities(logits_path, prepared, model, threads=1):
    import torch
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    if file_sha(prepared['properties']) != prepared['properties_sha256']:
        raise ValueError('预测空间记录损坏')
    with Path(prepared['properties']).open('rb') as stream:
        properties = pickle.load(stream)
    manager = PlansManager(model['plan'])
    logits = np.load(logits_path, mmap_mode='r')
    segmentation, probabilities = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, manager, manager.get_configuration('3d_fullres'),
        manager.get_label_manager(model['dataset_json']), properties,
        return_probabilities=True, num_threads_torch=threads)
    if torch.is_tensor(probabilities):
        probabilities = probabilities.numpy()
    # nnU-Net NibabelIO 使用 z,y,x；写入 NiBabel 前恢复 x,y,z。
    return segmentation.transpose(2, 1, 0), probabilities.transpose(0, 3, 2, 1)


def export_case(record, prepared, model, logits_path, output, *, evaluate=True, threads=1, keep_probabilities=False, progress=None):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if progress:
        progress('restore_native_grid')
    segmentation, probabilities = restore_probabilities(logits_path, prepared, model, threads)
    if list(segmentation.shape) != record['shape_xyz'] or not np.isfinite(probabilities).all():
        raise ValueError('原图恢复尺寸或概率无效')
    if int(segmentation.min()) < 0 or int(segmentation.max()) > 7:
        raise ValueError('输出标签超出七结构范围')
    affine = corrected_affine(np.asarray(record['source_affine']), record['spacing_xyz'])
    destination = output/(record['case_id']+'.nii.gz')
    if progress:
        progress('write_segmentation')
    save_nifti(destination, segmentation.astype(np.uint8), affine)
    files = {destination.name: file_sha(destination)}
    native = nib.load(record['pair']['image'])
    reference_image = record['pair']['image']
    if not np.allclose(native.affine, affine, rtol=1e-6, atol=1e-6):
        # 原件只读；修正头信息的展示副本与分割保持相同空间。
        reference = output/(record['case_id']+'_reference.nii.gz')
        if progress:
            progress('write_reference_CT')
        save_nifti(reference, native.get_fdata(dtype=np.float32), affine)
        reference_image = str(reference)
        files[reference.name] = file_sha(reference)
    metrics = None
    if evaluate:
        label_path = Path(record['pair']['label'])
        if file_sha(label_path) != record['source_hashes']['label']:
            raise ValueError('评价标签与来源审计不一致')
        truth_image = nib.load(label_path)
        if truth_image.shape != native.shape or not np.allclose(truth_image.affine, native.affine, atol=1e-6):
            raise ValueError('标签与 CT 原始几何不一致')
        truth = map_labels(np.asanyarray(truth_image.dataobj))
        metrics = score_segmentation(segmentation, truth, affine, progress)
    probability_path = None
    if keep_probabilities:
        probability_path = output/'probabilities.npy'
        np.save(probability_path, probabilities.astype(np.float32, copy=False))
        files[probability_path.name] = file_sha(probability_path)
    result = dict(case_id=record['case_id'], fold=model['fold'], segmentation=str(destination),
                  reference_image=reference_image, affine=affine.tolist(), source_affine=record['source_affine'],
                  shape=record['shape_xyz'], spacing_xyz=record['spacing_xyz'],
                  checkpoint_sha256=model['pointer']['sha256'], metrics=metrics,
                  probabilities=str(probability_path) if probability_path else None,
                  file_sha256=files, export_seconds=time.perf_counter()-started,
                  geometry_evidence=record['spacing_evidence'])
    write_json(output/'result.json', result)
    return result


def valid_result(directory, identity):
    directory = Path(directory)
    marker = directory/'complete.json'
    if not marker.exists():
        return None
    completed = read_json(marker)
    if completed['identity'] != identity:
        raise ValueError('已完成预测的任务身份不一致')
    for name, sha in completed['files'].items():
        path = (directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or file_sha(path) != sha:
            raise ValueError('已完成预测文件损坏：'+name)
    return read_json(directory/'result.json')
