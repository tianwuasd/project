from contextlib import contextmanager
from pathlib import Path

import numpy as np
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero

from ..acceleration.legacy import SessionGuard
from ..models.stunet.streaming import BlockWorkspace, read_sources, fill_and_crop


@contextmanager
def plan_case_data(config, reader, row):
    with SessionGuard(config['resource_handle']) as guard:
        image, label, spacing, roi = reader.read_metadata(row)
        amount = int(np.prod([high-low for low, high in roi]))*32
        if amount <= guard.snapshot()['budget']['headroom_bytes']:
            with guard.admit(amount, label='plan_whole_case'):
                data, segmentation, image, spacing, roi, seconds = reader.read_case(row)
                data, segmentation, _ = crop_to_nonzero(data[None], segmentation[None])
                yield data, segmentation, image, spacing, roi, seconds
            return
        path = Path(config['run_dir']) / ('plan-work-'+row['case_id'])
        explicit = config.get('_explicit_preprocessing', [])
        settings = dict(config['preprocessing'])
        if 'block_bytes' not in explicit:
            settings['block_bytes'] = max(1024, guard.snapshot()['budget']['headroom_bytes']//4)
        if 'max_block_bytes' not in explicit:
            settings['max_block_bytes'] = settings['block_bytes']
        config = {**config, 'preprocessing': settings}
        workspace = BlockWorkspace(path, config, guard, row['case_id'])
        try:
            # 拟合使用原始XYZ顺序；分块存储在ZYX中计算，再提供相同的视图。
            reader.prepared['plan'] = {'transpose_forward': [0, 1, 2]}
            image, spacing, roi, data, segmentation, mask, _, _ = read_sources(reader, row, workspace)
            bbox = fill_and_crop(mask, workspace)
            region = tuple(slice(low, high) for low, high in bbox)
            yield data[region].transpose(2,1,0)[None], segmentation[region].transpose(2,1,0)[None], image, spacing, roi, 0.
        finally:
            workspace.close()
