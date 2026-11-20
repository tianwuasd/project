from pathlib import Path

import nibabel as nib
import numpy as np

from ...acceleration import ResourceSession
from ...acceleration.legacy import SessionGuard
from ...artifacts.store import operation_lock, read_json, require_output, write_json
from ...observability.events import event
from . import PROTOCOL
from .input import open_image, selected_cases, source_record
from .runtime import request
from .storage import mapped_image


def evaluate(config, names, output):
    rows = selected_cases(config, names)
    root = require_output(output, config['work_root'], [config['data_root'], Path(__file__).parents[3]])
    manifest = read_json(config['manifest'])
    labels = {row['case_id']: row['label'] for row in manifest['cases'] if row['case_id'] in names}
    results = []
    with operation_lock(root), ResourceSession(request(config, root, evaluation=True)) as session:
        with SessionGuard(session.handle) as guard:
            for row in rows:
                name = row['case_id']
                case_root = root / name
                state = read_json(case_root / 'case.json')
                original = open_image(row)
                if state['protocol'] != PROTOCOL or state['source'] != source_record(row, original, config):
                    raise ValueError('评价来源和预测来源不同')
                covered = sorted(map(int, state['completed_slices']))
                if covered != state['requested_slices'] or not Path(state['output']).is_file():
                    raise ValueError('预测尚未完整提交所请求的切片')
                label_path = (Path(config['data_root']) / labels[name]).resolve(strict=True)
                if not label_path.is_relative_to(Path(config['data_root'])):
                    raise ValueError('标签路径越出数据根')
                reference = nib.load(label_path, keep_file_open=True)
                if reference.shape != original.shape:
                    raise ValueError('共享数组索引评价需要相同形状')
                reference = mapped_image(reference, label_path, case_root / 'evaluation-cache',
                    dict(path=str(label_path), bytes=label_path.stat().st_size,
                         mtime_ns=label_path.stat().st_mtime_ns), guard, root, name='label')
                counts = {class_id: [0, 0, 0] for class_id in range(1, 11)
                    if class_id not in row['unknown_classes']}
                for index in covered:
                    with guard.admit(16 * 2**20, label='medsam3_evaluation_slice') as ticket:
                        if ticket is None:
                            raise InterruptedError('评价资源等待已取消')
                        truth = np.asarray(reference.dataobj[:, :, index], dtype=np.float64)
                        if not np.isfinite(truth).all() or not np.equal(truth, np.rint(truth)).all():
                            raise ValueError('评价标签必须是有限整数')
                        truth = truth.astype(np.int16)
                        truth[np.isin(truth, row['ignore_label_ids'])] = -1
                        if not ((truth >= -1) & (truth <= 10)).all():
                            raise ValueError('标签超出CHD1282十类范围')
                        prediction = np.load(case_root / f'slice-{index}.npy', allow_pickle=False)
                        if prediction.shape != truth.shape or not ((prediction >= 0) & (prediction <= 10)).all():
                            raise ValueError('已覆盖切片预测形状或类别无效')
                        valid = truth >= 0
                        for class_id, values in counts.items():
                            target = (truth == class_id) & valid
                            found = (prediction == class_id) & valid
                            values[0] += int(target.sum())
                            values[1] += int(found.sum())
                            values[2] += int((target & found).sum())
                        del prediction, truth
                results.append(dict(case_id=name, slices=covered,
                    geometry_policy='shared_index_excel',
                    original_image_affine=original.affine.tolist(),
                    original_label_affine=reference.affine.tolist(),
                    classes={str(k): dict(reference_voxels=a, predicted_voxels=b,
                        dice=2 * overlap / (a + b) if a + b else None)
                        for k, (a, b, overlap) in counts.items()},
                    complete_volume=state['complete_volume']))
                del reference, original
        result = dict(protocol=PROTOCOL, cases=results, label_access='independent_evaluation_only',
            interpretation='仅评价明确记录的覆盖切片；共享索引不证明扫描物理坐标已认证')
        write_json(root / 'evaluation.json', result)
        event(root, 'medsam3_evaluation_completed', cases=len(results))
    return result
