import numpy as np


def supervised_dice(prediction, truth, classes, unknown=(), selected=None):
    if prediction.shape != truth.shape:
        raise ValueError('预测与参考标签尺寸不同')
    if not np.isfinite(truth).all() or not np.equal(truth, np.rint(truth)).all() or \
            not ((truth >= -1) & (truth < classes)).all():
        raise ValueError('参考标签超出已声明的整数类别范围')
    ids = set(range(1, classes))-set(unknown) if selected is None else set(selected)
    if not ids <= set(range(1, classes))-set(unknown):
        raise ValueError('评价类别包含未知或无效类别')
    valid, result = truth >= 0, {}
    for class_id in sorted(ids):
        target, found = (truth == class_id) & valid, (prediction == class_id) & valid
        reference_count, predicted_count = int(target.sum()), int(found.sum())
        result[str(class_id)] = dict(reference_voxels=reference_count, predicted_voxels=predicted_count,
            dice=float(2*(target & found).sum()/(reference_count+predicted_count))
            if reference_count+predicted_count else None)
    return result
