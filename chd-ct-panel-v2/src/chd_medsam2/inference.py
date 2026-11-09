from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import torch

from chd_stunet.observability.runlog import write_json
from chd_stunet.prediction.export import save_nifti
from chd_stunet.preprocessing.cohort import TARGETS

from .contracts import file_sha, read_json
from .data import frame_tensor, xyxy
from .preprocessing import canonical_image, to_axial, to_native


LABEL_NAMES = {value: key for key, value in TARGETS.items() if 1 <= value <= 7}


class AxialFrames:
    def __init__(self, data, size, window):
        self.data, self.size, self.window = data, size, window
        self.frame = lru_cache(maxsize=2)(self.convert)

    def __len__(self):
        return len(self.data)

    def convert(self, frame):
        return frame_tensor(self.data[frame], self.size, self.window)

    def __getitem__(self, frame):
        return self.frame(frame)


def simulated_prompts(labels):
    maximum = np.zeros(8, dtype=np.int64)
    slices = np.zeros(8, dtype=np.int64)
    for z, frame in enumerate(labels):
        sizes = np.bincount(frame.ravel(), minlength=8)[:8]
        improved = sizes > maximum
        maximum[improved], slices[improved] = sizes[improved], z
    missing = [name for label, name in LABEL_NAMES.items() if maximum[label] == 0]
    if missing:
        raise ValueError('主队列评分病例缺少结构：'+', '.join(missing))
    boxes = [dict(label=label, frame=int(slices[label]), xyxy=xyxy(labels[slices[label]] == label))
             for label in LABEL_NAMES]
    return dict(prompt_source='simulated_gt_single_box', boxes=boxes)


def validate_prompts(prompts, shape, *, source):
    if prompts['prompt_source'] != source:
        raise ValueError('MedSAM2 提示来源与当前任务不一致')
    depth, height, width = shape
    boxes = prompts['boxes']
    if (len(boxes) != 7 or any(type(box['label']) is not int for box in boxes)
            or set(box['label'] for box in boxes) != set(LABEL_NAMES)):
        raise ValueError('MedSAM2 每个目标类别必须有一个框提示')
    for item in boxes:
        frame = item['frame']
        x0, y0, x1, y1 = item['xyxy']
        if (type(frame) is not int or not 0 <= frame < depth or
                any(type(value) not in (int, float) for value in (x0, y0, x1, y1)) or
                not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height)):
            raise ValueError('MedSAM2 框提示坐标越界或无面积')
    return sorted(boxes, key=lambda item: item['label'])


@torch.inference_mode()
def segment(predictor, axial, window, prompts, output, *, source, callback=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    boxes = validate_prompts(prompts, axial.shape, source=source)
    frames = AxialFrames(axial, predictor.image_size, window)
    depth, height, width = axial.shape
    # 保留整段官方传播历史，CPU 卸载；完整概率无需在内存中叠加七份。
    best = np.lib.format.open_memmap(output/'max_logits.npy', mode='w+', dtype=np.float32, shape=axial.shape)
    best[:] = 0
    labels = np.lib.format.open_memmap(output/'prediction_axial.npy', mode='w+', dtype=np.uint8, shape=axial.shape)
    labels[:] = 0
    for box in boxes:
        label, start = box['label'], box['frame']
        if callback:
            callback(label, start, depth)
        state = predictor.init_state(frames, height, width, offload_video_to_cpu=True, offload_state_to_cpu=True)
        predictor.add_new_points_or_box(state, frame_idx=start, obj_id=label,
                                       box=np.asarray(box['xyxy'], dtype=np.float32))
        class_path = output/f'mask_{label}.npy'
        mask = np.lib.format.open_memmap(class_path, mode='w+', dtype=np.bool_, shape=axial.shape)
        seen = set()
        for reverse in (False, True):
            for z, ids, logits in predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                if list(ids) != [label]:
                    raise ValueError('官方预测器返回了错误的类别 ID')
                if z in seen:
                    continue
                layer = logits[0, 0].float().cpu().numpy()
                if not np.isfinite(layer).all():
                    raise ValueError('MedSAM2 传播产生非有限 logit')
                mask[z] = layer > 0
                wins = layer > best[z]
                best[z][wins], labels[z][wins] = layer[wins], label
                seen.add(z)
        if seen != set(range(depth)):
            raise RuntimeError('MedSAM2 双向传播未覆盖全部切片')
        mask.flush()
        del mask, state
    best.flush()
    labels.flush()
    write_json(output/'prompts.json', prompts)
    return labels


def dice_on_native(pred, truth):
    if pred.shape != truth.shape:
        raise ValueError('MedSAM2 预测与真值尺寸不一致')
    # 按切片累计体素计数，内存开销不随整卷层数增长。
    pred_counts, truth_counts, matched_counts = np.zeros((3, 8), dtype=np.int64)
    for pred_frame, truth_frame in zip(pred, truth, strict=True):
        valid = truth_frame != 8
        p, t = pred_frame[valid], truth_frame[valid]
        pred_counts += np.bincount(p, minlength=8)[:8]
        truth_counts += np.bincount(t, minlength=8)[:8]
        matched_counts += np.bincount(t[p == t], minlength=8)[:8]
    result = {}
    for label, name in LABEL_NAMES.items():
        p_count, t_count = int(pred_counts[label]), int(truth_counts[label])
        total = p_count+t_count
        result[name] = dict(dice=2*int(matched_counts[label])/total if total else 1.,
                            both_empty=not total, missing_prediction=not p_count and bool(t_count))
    return result


def evaluate_case(predictor, prepared, cid, meta, window, output, log, *, frame_limit=None, retain_raw=True):
    folder = prepared/'cases'/cid
    axial = np.load(folder/'ct.npy', mmap_mode='r', allow_pickle=False)
    truth = np.load(folder/'labels.npy', mmap_mode='r', allow_pickle=False)
    prompts = simulated_prompts(truth)
    if frame_limit:
        # 仅短测使用：截取涵盖所有七类框提示的一段实际 CT。
        frames = [item['frame'] for item in prompts['boxes']]
        start = max(0, min(min(frames), max(frames)-frame_limit+1))
        end = min(len(axial), max(start+frame_limit, max(frames)+1))
        axial, truth = axial[start:end], truth[start:end]
        for item in prompts['boxes']:
            item['frame'] -= start
        prompts['axial_offset'] = start
    def evaluate(workspace):
        result = segment(predictor, axial, window, prompts, workspace, source='simulated_gt_single_box',
                         callback=lambda label, z, depth: log.event('validation.structure',
                                  f'{cid} {LABEL_NAMES[label]}：{depth}帧，提示第{z}帧'))
        if frame_limit:
            return dice_on_native(result, truth)
        native = to_native(result, meta['geometry'])
        if list(native.shape) != meta['geometry']['native_shape']:
            raise ValueError('MedSAM2 验证分割未恢复到原始网格')
        if retain_raw:
            save_nifti(Path(output)/'segmentation.nii.gz', native, np.asarray(meta['geometry']['affine']))
        # 轴序转换只改变排列，按连续轴位切片评分可减少缓存页的反复读取。
        return dice_on_native(result, truth)

    if retain_raw:
        metrics = evaluate(output)
    else:
        # 每轮验证仅保留提示和指标，避免反复保存七套整卷临时掩膜。
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='medsam2-validation-', dir=output.parent) as workspace:
            metrics = evaluate(workspace)
        write_json(output/'prompts.json', prompts)
    record = dict(case_id=cid, metrics=metrics,
                  mean_dice=sum(row['dice'] for row in metrics.values())/7,
                  scope='axial_smoke_subset' if frame_limit else 'whole_volume',
                  prompt_source=prompts['prompt_source'], includes_prompt_slices=True)
    write_json(Path(output)/'result.json', record)
    return record


def predict_image(predictor, image_path, prompts_path, window, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('MedSAM2 预测输出须为空目录')
    image = nib.load(image_path)
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('新 CT 要求明确的毫米坐标单位')
    geometry = canonical_image(image)
    axial = to_axial(image.get_fdata(dtype=np.float32), geometry)
    if not np.isfinite(axial).all():
        raise ValueError('新 CT 含非有限像素值')
    prompts = read_json(prompts_path)
    if prompts['image_sha256'] != file_sha(image_path):
        raise ValueError('人工框提示与输入 CT 摘要不一致')
    prediction = segment(predictor, axial, window, prompts, output, source='manual_box')
    target = output/'segmentation.nii.gz'
    save_nifti(target, to_native(prediction, geometry), np.asarray(geometry['affine']))
    record = dict(status='completed', segmentation=str(target), sha256=file_sha(target),
                  image_sha256=prompts['image_sha256'], prompt_source='manual_box')
    write_json(output/'summary.json', record)
    return record
