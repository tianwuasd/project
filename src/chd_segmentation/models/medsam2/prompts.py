import hashlib

import numpy as np


def label_box_prompts(entry, reference_sha256, classes, requested=None, seed=2024, max_shift=20):
    if type(seed) is not int or seed < 0 or type(max_shift) is not int or not 0 <= max_shift <= 20:
        raise ValueError('标签框种子需要非负整数，外扩上限需要0—20像素整数')
    label = entry['label']
    if label is None or label.ndim != 3 or label.shape != entry['image'].shape[1:]:
        raise ValueError('标签框需要与原始DHW网格一致的三维参考标签')
    known = set(range(1, classes))-set(entry['unknown_classes'])
    if requested is not None and (not requested or len(set(requested)) != len(requested) or
                                  any(type(value) is not int for value in requested) or not set(requested) <= known):
        raise ValueError('提示类别需要互不重复的已知前景类别')
    selected = sorted(known if requested is None else requested)
    depth, height, width = label.shape
    items, presence = [], {}
    for class_id in selected:
        areas = (label == class_id).sum((1, 2))
        presence[str(class_id)] = bool(areas.max())
        if not areas.max():
            continue
        frame = int(areas.argmax())
        yy, xx = np.where(label[frame] == class_id)
        tight = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
        token = hashlib.sha256(f'{seed}:{entry["case_id"]}:{class_id}'.encode('utf-8')).digest()
        rng = np.random.default_rng(int.from_bytes(token[:8], 'big'))
        shift = rng.integers(0, max_shift+1, size=4).tolist()
        box = [max(0, tight[0]-shift[0]), max(0, tight[1]-shift[1]),
               min(width, tight[2]+shift[2]), min(height, tight[3]+shift[3])]
        items.append(dict(source='label_box', space='native_dhw', class_id=class_id, frame=frame,
                          box_xyxy=box, tight_box_xyxy=tight, expansion_ltrb=shift,
                          reference_sha256=reference_sha256))
    if not items:
        raise ValueError('所选提示类别在参考标签中均无前景')
    protocol = dict(mode='ground_truth_box_prompted', frame_rule='maximum_label_area_first_tie',
        box_rule='tight_2d_bbox_with_seeded_outward_shift', box_convention='xyxy_upper_exclusive',
        seed=seed, max_shift_pixels=max_shift, reference_sha256=reference_sha256,
        class_presence=presence, prompted_classes=[item['class_id'] for item in items],
        grid_dhw=[depth, height, width], prompts=items,
        propagation='independent_structure_forward_reverse_full_depth',
        interpretation='使用参考标签提供提示框的分割评测；不评价自动定位能力')
    return {entry['case_id']: items}, protocol
