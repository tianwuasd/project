"""Metadata-only scale preview; no target arrays or alternative scientific plan."""
import math


def case_scale(row,plan):
    """Use nnU-Net compute_new_shape's exact Python rounding expression.

    Shapes/spacings here are before the plan's axis transpose. Python integer
    products avoid int32 overflow for very large targets. This is a prediction
    from the audited crop, not a measured memory peak or capacity PASS.
    """
    old=row['cropped_shape_zyx'];spacing=row['spacing_zyx']
    target=plan['target_spacing_before_transpose']
    if not (len(old)==len(spacing)==len(target)==3):raise ValueError('Expected three spatial axes')
    if any(not math.isfinite(v) or v<=0 for v in [*old,*spacing,*target]):
        raise ValueError('Positive finite shape and spacing required')
    shape=[int(round(i/j*k)) for i,j,k in zip(spacing,target,old)]
    if min(shape)<1:raise ValueError('Native target shape contains an empty axis')
    voxels=math.prod(shape)
    return dict(case_id=row['case_id'],predicted_shape_zyx=shape,
        predicted_transposed_shape=[shape[i] for i in plan['transpose_forward']],
        output_voxels=voxels,cropped_native_voxels=math.prod(old),
        expansion_from_crop=voxels/math.prod(old),image_float32_bytes=4*voxels,
        label_int8_bytes=voxels,capacity_verified=False)
