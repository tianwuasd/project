from contextlib import nullcontext
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
from .preprocessing import assert_case_files, canonical_image, file_stamp, to_axial, to_native


LABEL_NAMES = {value: key for key, value in TARGETS.items() if 1 <= value <= 7}
VALIDATION_READ_BYTES = 8*1024**2
VALIDATION_READ_FRAMES = 8


class AxialFrames:
    def __init__(self, data, size, window, reader=None, source_path=None, source_offset=0):
        self.data, self.size, self.window = data, size, window
        self.reader, self.source_path = reader, source_path
        self.source_offset = source_offset
        frame_bytes = data.shape[1]*data.shape[2]*data.dtype.itemsize
        self.block_frames = max(1, min(VALIDATION_READ_FRAMES, VALIDATION_READ_BYTES//frame_bytes))
        self.block_start, self.block = None, None
        self.frame = lru_cache(maxsize=2)(self.convert)

    def __len__(self):
        return len(self.data)

    def convert(self, frame):
        if self.reader is None:
            return frame_tensor(self.data[frame], self.size, self.window)
        start = frame//self.block_frames*self.block_frames
        if self.block_start != start:
            end = min(start+self.block_frames, len(self.data))
            # 将一段相邻切片读到内存，读取监督不覆盖后续图像转换或模型计算。
            self.block = None
            with self.reader as progress:
                progress(self.source_path, self.source_offset+start, self.source_offset+end-1)
                self.block = self.data[start:end].copy()
            self.block_start = start
        return frame_tensor(self.block[frame-start], self.size, self.window)

    def __getitem__(self, frame):
        return self.frame(frame)


def simulated_prompts(labels, progress=None):
    maximum = np.zeros(8, dtype=np.int64)
    slices = np.zeros(8, dtype=np.int64)
    for z, frame in enumerate(labels):
        if progress is not None:
            progress(z)
        sizes = np.bincount(frame.ravel(), minlength=8)[:8]
        improved = sizes > maximum
        maximum[improved], slices[improved] = sizes[improved], z
    missing = [name for label, name in LABEL_NAMES.items() if maximum[label] == 0]
    if missing:
        raise ValueError('主队列评分病例缺少结构：'+', '.join(missing))
    boxes = []
    for label in LABEL_NAMES:
        z = int(slices[label])
        if progress is not None:
            progress(z)
        boxes.append(dict(label=label, frame=z, xyxy=xyxy(labels[z] == label)))
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
def segment(predictor, axial, window, prompts, output, *, source, callback=None,
            reader=None, source_path=None, source_offset=0, frame_progress=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    boxes = validate_prompts(prompts, axial.shape, source=source)
    frames = AxialFrames(axial, predictor.image_size, window, reader, source_path, source_offset)
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
                if frame_progress:
                    frame_progress(label, len(seen), depth)
        if seen != set(range(depth)):
            raise RuntimeError('MedSAM2 双向传播未覆盖全部切片')
        mask.flush()
        del mask, state
    best.flush()
    labels.flush()
    write_json(output/'prompts.json', prompts)
    return labels


def dice_on_native(pred, truth, progress=None):
    if pred.shape != truth.shape:
        raise ValueError('MedSAM2 预测与真值尺寸不一致')
    # 按切片累计体素计数，内存开销不随整卷层数增长。
    pred_counts, truth_counts, matched_counts = np.zeros((3, 8), dtype=np.int64)
    for z, (pred_frame, truth_frame) in enumerate(zip(pred, truth, strict=True)):
        if progress is not None:
            progress(z)
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


def evaluate_case(predictor, prepared, cid, meta, window, output, log, *,
                  frame_limit=None, retain_raw=True, reader=None, export_segmentation=False,
                  frame_progress=None):
    folder = prepared/'cases'/cid
    image_path, label_path = folder/'ct.npy', folder/'labels.npy'
    scope = reader or nullcontext()
    with scope as progress:
        assert_case_files(folder, meta, progress)
        if progress is not None:
            progress(image_path)
        axial = np.load(image_path, mmap_mode='r', allow_pickle=False)
        if progress is not None:
            progress(label_path)
        truth = np.load(label_path, mmap_mode='r', allow_pickle=False)
        prompts = simulated_prompts(truth, (lambda z: progress(label_path, z)) if progress else None)
    offset = 0
    if frame_limit:
        # 仅短测使用：截取涵盖所有七类框提示的一段实际 CT。
        frames = [item['frame'] for item in prompts['boxes']]
        start = max(0, min(min(frames), max(frames)-frame_limit+1))
        end = min(len(axial), max(start+frame_limit, max(frames)+1))
        axial, truth = axial[start:end], truth[start:end]
        offset = start
        for item in prompts['boxes']:
            item['frame'] -= start
        prompts['axial_offset'] = start
    def evaluate(workspace):
        result = segment(predictor, axial, window, prompts, workspace, source='simulated_gt_single_box',
                         reader=reader, source_path=image_path, source_offset=offset,
                         frame_progress=frame_progress,
                         callback=lambda label, z, depth: log.event('validation.structure',
                                  f'{cid} {LABEL_NAMES[label]}：{depth}帧，提示第{z}帧'))
        with scope as progress:
            metrics = dice_on_native(result, truth, (lambda z: progress(label_path, z+offset)) if progress else None)
            assert_case_files(folder, meta, progress)
        if frame_limit:
            return metrics
        native = to_native(result, meta['geometry'])
        if list(native.shape) != meta['geometry']['native_shape']:
            raise ValueError('MedSAM2 验证分割未恢复到原始网格')
        if retain_raw or export_segmentation:
            Path(output).mkdir(parents=True, exist_ok=True)
            save_nifti(Path(output)/'segmentation.nii.gz', native, np.asarray(meta['geometry']['affine']))
        # 轴序转换只改变排列，按连续轴位切片评分可减少缓存页的反复读取。
        return metrics

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


def read_prompted_image(image_path, prompts_path):
    stamp = file_stamp(image_path)
    image = nib.load(image_path)
    if image.header.get_xyzt_units()[0] != 'mm':
        raise ValueError('新 CT 要求明确的毫米坐标单位')
    geometry = canonical_image(image)
    prompts = read_json(prompts_path)
    if prompts['image_sha256'] != file_sha(image_path) or file_stamp(image_path) != stamp:
        raise ValueError('人工框提示与输入 CT 摘要不一致')
    # 提示坐标以转为 RAS 后的轴位数组为准，头部尺寸核验无需读取整卷。
    order = np.argsort(np.asarray(geometry['forward'])[:, 0])[::-1]
    validate_prompts(prompts, tuple(image.shape[axis] for axis in order), source='manual_box')
    return image, geometry, prompts, stamp


def predict_image(predictor, image, geometry, prompts, stamp, window, output, log, *, frame_progress=None):
    image_path = Path(image.get_filename())
    if file_stamp(image_path) != stamp:
        raise ValueError('新 CT 在提示核验后变化')
    axial = to_axial(image.get_fdata(dtype=np.float32), geometry)
    if file_stamp(image_path) != stamp:
        raise ValueError('新 CT 在读取中变化')
    if not np.isfinite(axial).all():
        raise ValueError('新 CT 含非有限像素值')
    prediction = segment(predictor, axial, window, prompts, output, source='manual_box',
                         frame_progress=frame_progress,
                         callback=lambda label, z, depth: log.event('prediction.structure',
                                  f'{LABEL_NAMES[label]}：{depth}帧，提示第{z}帧'))
    target = output/'segmentation.nii.gz'
    save_nifti(target, to_native(prediction, geometry), np.asarray(geometry['affine']))
    record = dict(status='completed', segmentation=str(target), sha256=file_sha(target),
                  image_sha256=prompts['image_sha256'], prompt_source='manual_box')
    return record
