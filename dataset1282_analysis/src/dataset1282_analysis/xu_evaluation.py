"""Native-grid volumetric and surface metrics for an individual Xu segmentation branch."""
import numpy as np
import time
from scipy.ndimage import binary_erosion, find_objects, generate_binary_structure
from scipy.spatial import cKDTree


def surface_points(mask, spacing, offset=None):
    if offset is None:
        boxes=find_objects(mask,max_label=1)
        if not boxes or boxes[0] is None:
            return np.empty((0,3),dtype=float)
        box=boxes[0]
        region=np.ascontiguousarray(mask[box])
        origin=np.asarray([s.start for s in box])
    else:
        region=np.ascontiguousarray(mask)
        origin=offset
    interior=binary_erosion(region,structure=generate_binary_structure(3,1),border_value=0)
    return (np.argwhere(region & ~interior)+origin)*spacing


def class_metrics(reference, prediction, spacing_xyz, supervision, progress=None):
    """Inputs are aligned XYZ arrays. Distances use physical voxel-centre coordinates."""
    if progress is not None:
        progress('metric_stage',stage='validate_labels',label_id=None)
    if reference.ndim != 3 or reference.shape != prediction.shape:
        raise ValueError('Expected native-grid predictions matching the three-dimensional reference')
    spacing = np.asarray(spacing_xyz, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Expected three positive millimetre spacings in XYZ order')
    for array in (reference, prediction):
        if array.dtype.kind not in 'iu' or array.min() < 0 or array.max() > 10:
            raise ValueError('Expected integer labels in 0..10')
    present = set(supervision['present_classes'])
    counts = np.zeros(11, dtype=np.int64)
    predicted_counts=np.zeros(11,dtype=np.int64)
    intersections=np.zeros(11,dtype=np.int64)
    for z in range(0, reference.shape[2], 8):
        left,right=reference[:,:,z:z+8],prediction[:,:,z:z+8]
        counts += np.bincount(left.ravel(), minlength=11)
        predicted_counts += np.bincount(right.ravel(), minlength=11)
        intersections += np.bincount(left[left==right],minlength=11)
    if present != set(np.flatnonzero(counts[1:])+1):
        raise ValueError('Native reference and recorded present structures disagree')
    unknown = set(supervision['unknown_classes'])
    absent = set(supervision['known_absent_classes'])
    if present | unknown | absent != set(range(1, 11)) or present & unknown or present & absent or unknown & absent:
        raise ValueError('Invalid structure supervision partition')

    reference_boxes=find_objects(reference,max_label=10)
    prediction_boxes=find_objects(prediction,max_label=10)
    results = {}
    for label in range(1, 11):
        started = time.monotonic()
        def measured(stage, operation):
            if progress is not None:
                progress('metric_stage',stage=stage,label_id=label)
            before = time.monotonic()
            value = operation()
            if progress is not None:
                progress('metric_stage_completed',stage=stage,label_id=label,
                         stage_elapsed_seconds=time.monotonic()-before)
            return value
        def completed(row):
            if progress is not None:
                progress('metric_label_completed',stage='structure_completed',label_id=label,
                         label_seconds=time.monotonic()-started,metrics=dict(row))
        if progress is not None:
            progress('metric_stage',stage='structure_start',label_id=label,metrics=None,
                     label_seconds=None,stage_elapsed_seconds=None)
        if label in unknown:
            results[label] = dict(status='unannotated')
            completed(results[label])
            continue
        predicted = int(predicted_counts[label])
        if label in absent:
            results[label] = dict(status='verified_absent', predicted_voxels=predicted)
            completed(results[label])
            continue
        size = int(counts[label])
        row = dict(status='scored', reference_voxels=size, predicted_voxels=predicted,
                   dice=2*int(intersections[label])/(size+predicted),
                   hd95_mm=None, asd_mm=None)
        if predicted:
            a,b=reference_boxes[label-1],prediction_boxes[label-1]
            left = measured('reference_surface',lambda: surface_points(reference[a]==label,spacing,np.asarray([s.start for s in a])))
            right = measured('prediction_surface',lambda: surface_points(prediction[b]==label,spacing,np.asarray([s.start for s in b])))
            forward = measured('nearest_forward',lambda: cKDTree(right).query(left)[0])
            backward = measured('nearest_backward',lambda: cKDTree(left).query(right)[0])
            distances = np.concatenate((forward, backward))
            row.update(hd95_mm=float(np.percentile(distances, 95)), asd_mm=float(distances.mean()))
        else:
            row['status'] = 'empty_prediction'
        results[label] = row
        completed(row)
    return results


def summarize(cases):
    """Per-class means and sample SDs; a missing surface has an explicit denominator."""
    summaries = {}
    for label in range(1, 11):
        rows = [case[label] for case in cases]
        item = {name:sum(row['status']==name for row in rows)
                for name in ('unannotated', 'verified_absent', 'empty_prediction')}
        for metric in ('dice','hd95_mm','asd_mm'):
            values = [r[metric] for r in rows if r.get(metric) is not None]
            item[metric] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                                sd=float(np.std(values, ddof=1)) if len(values)>1 else None)
        summaries[label] = item
    macro = {}
    for metric in ('dice','hd95_mm','asd_mm'):
        means = [r[metric]['mean'] for r in summaries.values() if r[metric]['n']]
        macro[metric] = dict(classes=len(means), mean=float(np.mean(means)) if len(means)==10 else None)
    return dict(cases=len(cases), by_label=summaries, macro_ten_classes=macro)
