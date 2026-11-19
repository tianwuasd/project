import math

from monai.data.utils import dense_patch_slices
from monai.data.utils import compute_importance_map
from nnunetv2.preprocessing.resampling.default_resampling import determine_do_sep_z_and_axis
import numpy as np
from scipy.ndimage import map_coordinates
import torch

from ..models.stunet.streaming import tiles, tile_extent


def window_probabilities(image, model, patch, classes, options, device, workspace):
    shape = tuple(image.shape[1:])
    padded = tuple(max(a, b) for a, b in zip(shape, patch))
    before = tuple((b-a)//2 for a, b in zip(shape, padded))
    intervals = tuple(size if size == full else max(1, int(size*(1-options['overlap'])))
                      for size, full in zip(patch, padded))
    windows = dense_patch_slices(padded, patch, intervals)
    workspace.check_disk(math.prod(padded)*(classes+1)*4 + math.prod(shape)*4)
    sums = workspace.create('logits', (classes, *padded), np.float32)
    counts = workspace.create('counts', padded, np.float32)
    weight = compute_importance_map(tuple(patch), mode=options['blend'], device=device, dtype=torch.float32)
    cpu_weight = weight.cpu().numpy()
    batch = options['window_batch']
    dirty_bytes = 0
    for start in range(0, len(windows), batch):
        with workspace.boundary(), torch.inference_mode():
            sections = windows[start:start+batch]
            inputs = []
            for section in sections:
                native = tuple(slice(max(0, s.start-b), min(n, s.stop-b)) for s, b, n in zip(section, before, shape))
                padding = tuple((max(0, b-s.start), max(0, s.stop-b-n)) for s, b, n in zip(section, before, shape))
                inputs.append(np.pad(np.array(image[(slice(None), *native)], copy=True), ((0, 0), *padding)))
            inputs = torch.from_numpy(np.stack(inputs)).to(device)
            values = model(inputs)
            values = values[0] if isinstance(values, (tuple, list)) else values
            if tuple(values.shape) != (len(sections), classes, *patch) or not torch.isfinite(values).all():
                raise ValueError('分块预测的模型输出尺寸或数值无效')
            weighted = (values*weight).float().cpu().numpy()
            for section, output in zip(sections, weighted):
                sums[(slice(None), *section)] += output
                counts[section] += cpu_weight
            del weighted, values, inputs
        dirty_bytes += len(sections)*math.prod(patch)*(classes+2)*4
        if dirty_bytes >= workspace.block_bytes or start+batch >= len(windows):
            workspace.release(image)
            dirty_bytes = 0
        workspace.progress('sliding_window', min(start+batch, len(windows)), len(windows))
    voxels = max(1, workspace.block_bytes//(classes*16+32))
    extent = tile_extent(padded, voxels)
    total = math.prod(math.ceil(n/b) for n, b in zip(padded, extent))
    for index, section in enumerate(tiles(padded, voxels), 1):
        with workspace.boundary():
            denominator = np.array(counts[section], copy=True)
            if not np.all(denominator > 0):
                raise ValueError('预测窗口未完整覆盖图像')
            values = np.array(sums[(slice(None), *section)], copy=True)
            values /= denominator
            if not np.isfinite(values).all():
                raise ValueError('预测累积出现非有限数值')
            sums[(slice(None), *section)] = torch.from_numpy(values).softmax(0).numpy()
            del values, denominator
        workspace.release()
        workspace.progress('probabilities', index, total)
    crop = tuple(slice(b, b+n) for b, n in zip(before, shape))
    return sums[(slice(None), *crop)]


def restore_blocks(probabilities, entry, workspace):
    plan, props = entry['source_plan'], entry['properties']
    shape = tuple(map(int, props['shape_after_cropping_and_before_resampling']))
    forward, backward = plan['transpose_forward'], plan['transpose_backward']
    current = entry.get('processed_spacing', plan['configurations']['3d_fullres']['spacing'])
    native_spacing = np.asarray(entry['spacing_xyz'])[::-1][forward]
    separate, axis = determine_do_sep_z_and_axis(None, current, native_spacing)
    workspace.check_disk(math.prod(entry['source_shape_xyz']))
    result = workspace.create('native', entry['source_shape_xyz'], np.uint8, fortran=True)
    box = props['bbox_used_for_cropping']
    before_shape = props['shape_before_cropping']
    roi_shape = [end-begin for begin, end in entry['roi_xyz']]
    if [before_shape[i] for i in backward[::-1]] != roi_shape or [b-a for a, b in box] != list(shape):
        raise ValueError('原图ROI、裁剪框或轴顺序不一致')
    voxels = max(1, workspace.block_bytes//160)
    extent = tile_extent(shape, voxels)
    total = math.prod(math.ceil(n/b) for n, b in zip(shape, extent))
    source_shape = probabilities.shape[1:]
    for index, section in enumerate(tiles(shape, voxels), 1):
        with workspace.boundary():
            indices = np.asarray(np.meshgrid(*(np.arange(s.start, s.stop) for s in section), indexing='ij'))
            points = np.asarray([(values+.5)*(float(old)/new)-.5
                                 for values, old, new in zip(indices, source_shape, shape)])
            if separate and tuple(source_shape) != shape and source_shape[axis] != shape[axis]:
                points[axis] = np.clip(np.floor(points[axis]+.5), 0, source_shape[axis]-1)
            best = np.full(indices.shape[1:], -np.inf, dtype=np.float32)
            labels = np.zeros(indices.shape[1:], dtype=np.uint8)
            for channel in range(probabilities.shape[0]):
                scores = map_coordinates(probabilities[channel], points, output=np.float64,
                                         order=1, mode='nearest', prefilter=False).astype(np.float32)
                if not np.isfinite(scores).all():
                    raise ValueError('原图恢复出现非有限概率')
                take = scores > best
                labels[take], best[take] = channel, scores[take]
                del scores, take
            target = tuple(indices[i]+box[i][0]+entry['roi_xyz'][xyz][0]
                           for xyz, i in enumerate(backward[::-1]))
            result[target] = labels
            del indices, points, best, labels, target
        workspace.release()
        workspace.progress('restore_source', index, total)
    return result
