"""Train-only intensity fitting with an explicit seven-class foreground mask."""
import numpy as np


def sample_foreground(image, segmentation, count, seed=1234):
    """Equal-size deterministic per-case samples, including small structures.

    Upstream 2.8.1 uses seg>0 and therefore includes ignore=8. Restrict to
    1..7 here without modifying the environment's installed nnU-Net package.
    Sampling with replacement gives each TRAIN CASE equal weight. The fixed
    200k/case cap is recorded; this is not claimed bit-identical to upstream's
    dataset-size-dependent 10-million-sample fingerprint.
    """
    values = image[(segmentation >= 1) & (segmentation <= 7)]
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('No finite target foreground for intensity fitting')
    return np.random.RandomState(seed).choice(values, int(count), replace=True).astype(np.float32)


def fit_statistics(rows, samples, fold):
    ids = sorted(row['case_id'] for row in rows if row['roles'][str(fold)] == 'train')
    if not ids:
        raise ValueError('Empty training subset')
    values = np.concatenate([samples[cid] for cid in ids])
    if not np.isfinite(values).all():
        raise ValueError('Non-finite training samples')
    low, median, high = np.percentile(values, [0.5, 50., 99.5])
    stats = dict(mean=float(np.mean(values, dtype=np.float64)), std=float(np.std(values, dtype=np.float64)),
                 min=float(values.min()), max=float(values.max()), median=float(median),
                 percentile_00_5=float(low), percentile_99_5=float(high))
    if stats['std'] <= 0:
        raise ValueError('Constant foreground training distribution')
    return stats, ids


def make_plan(rows, samples, fold, method):
    """Reuse the installed official target-spacing rule, without a network plan.

    This is a PREPROCESSING plan. Compression's 128^3 access hint is not a
    frozen training patch, architecture, batch-size or GPU capacity promise.
    """
    from nnunetv2.experiment_planning.experiment_planners.default_experiment_planner import ExperimentPlanner
    stats, ids = fit_statistics(rows, samples, fold)
    training = [r for r in rows if r['case_id'] in set(ids)]
    fingerprint = dict(spacings=[r['spacing_zyx'] for r in training],
                       shapes_after_crop=[r['cropped_shape_zyx'] for r in training],
                       foreground_intensity_properties_per_channel={'0':stats},
                       fit_ids=ids, fitting_roles=['train'], foreground_label_ids=list(range(1,8)),
                       ignore_excluded=True, samples_per_training_case=method['foreground_samples_per_case'])
    planner = ExperimentPlanner.__new__(ExperimentPlanner)
    planner.overwrite_target_spacing = None
    planner.dataset_fingerprint = fingerprint
    planner.anisotropy_threshold = 3
    spacing = planner.determine_fullres_target_spacing()
    forward = [int(np.argmax(spacing))]
    forward += [axis for axis in range(3) if axis not in forward]
    backward = np.argsort(forward).tolist()
    config = dict(data_identifier=method['protocol'] + '_3d_fullres', preprocessor_name='ProjectBoundedPreprocessor',
                  architecture={}, spacing=spacing[forward].tolist(),
                  normalization_schemes=[method['normalization']], use_mask_for_norm=[False],
                  resampling_fn_data='resample_data_or_seg_to_shape',
                  resampling_fn_data_kwargs=dict(is_seg=False, order=3, order_z=0, force_separate_z=None),
                  resampling_fn_seg='resample_data_or_seg_to_shape',
                  resampling_fn_seg_kwargs=dict(is_seg=True, order=1, order_z=0, force_separate_z=None),
                  patch_size=method['compression_patch_hint'])
    plan = dict(schema=1, scope='preprocessing_only', dataset_name='ImageCHD_clean96',
                plans_name='STUNetSPreprocessing', image_reader_writer='NibabelIO',
                label_manager='LabelManager', transpose_forward=forward, transpose_backward=backward,
                foreground_intensity_properties_per_channel={'0':stats},
                configurations={'3d_fullres':config}, fit_ids=ids, fold=fold,
                training_patch_frozen=False, target_spacing_before_transpose=spacing.tolist())
    return fingerprint, plan
