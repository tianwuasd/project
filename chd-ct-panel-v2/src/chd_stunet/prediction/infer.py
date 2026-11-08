import itertools
import time

import numpy as np
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window


def predictor_for(network, device, tta):
    predictor = nnUNetPredictor(device=device, use_mirroring=tta, use_gaussian=True,
                               perform_everything_on_device=False, allow_tqdm=False)
    predictor.network = network
    predictor.allowed_mirroring_axes = (0, 1, 2)
    return predictor


def forward_windows(predictor, batch, device):
    with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == 'cuda'):
        result = predictor._internal_maybe_mirror_and_predict(batch.to(device, non_blocking=True))
    if not bool(torch.isfinite(result).all()):
        raise ValueError('模型产生非有限预测值')
    return result.float().cpu()


def sliding_logits(network, image, patch, device, *, batch=1, tta=True, step=.5, progress=None):
    if batch < 1 or not 0 < step <= 1 or image.ndim != 4 or image.shape[0] != 1:
        raise ValueError('滑窗参数或图像维度无效')
    started = time.perf_counter()
    # 输入只读；stack 在取块时分配独立内存，避免修改原缓存。
    padded, revert = pad_nd_image(torch.from_numpy(np.asarray(image)), new_shape=patch,
                                  mode='constant', kwargs={'value': 0}, return_slicer=True)
    steps = compute_steps_for_sliding_window(padded.shape[1:], patch, step)
    windows = [(slice(None), *(slice(p, p+n) for p, n in zip(position, patch)))
               for position in itertools.product(*steps)]
    gaussian = compute_gaussian(tuple(patch), sigma_scale=1./8, value_scaling_factor=10,
                                dtype=torch.float32, device=torch.device('cpu'))
    # FP32 累积保持批量变化的稳定性，CPU 整体预算包含这些数组。
    logits = torch.zeros((8, *padded.shape[1:]), dtype=torch.float32)
    counts = torch.zeros(padded.shape[1:], dtype=torch.float32)
    predictor = predictor_for(network, device, tta)
    compute_seconds = blend_seconds = 0.
    for start in range(0, len(windows), batch):
        if progress:
            progress(start, len(windows), 'forward')
        slices = windows[start:start+batch]
        data = torch.stack([padded[s] for s in slices])
        if device.type == 'cuda':
            data = data.pin_memory()
        stamp = time.perf_counter()
        outputs = forward_windows(predictor, data, device)
        compute_seconds += time.perf_counter()-stamp
        stamp = time.perf_counter()
        for prediction, index in zip(outputs, slices):
            logits[index] += prediction*gaussian
            counts[index[1:]] += gaussian
        blend_seconds += time.perf_counter()-stamp
        if progress:
            progress(start+len(slices), len(windows), 'blend')
        del data, outputs
    if not bool((counts > 0).all()):
        raise ValueError('滑窗未覆盖所有体素')
    logits /= counts
    result = logits[(slice(None), *revert[1:])].numpy()
    if not np.isfinite(result).all():
        raise ValueError('融合结果包含非有限值')
    return result, dict(windows=len(windows), batch=batch, tta=tta, tile_step=step,
                        elapsed_seconds=time.perf_counter()-started,
                        compute_transfer_seconds=compute_seconds, blending_seconds=blend_seconds,
                        accumulator='cpu_float32')


def tune_batch(network, image, patch, device, admission, maximum=8, *, tta=True, progress=None):
    if device.type == 'cpu':
        return 1, {'selected': 1, 'reason': 'CPU明确串行滑窗', 'trials': []}
    padded, _ = pad_nd_image(torch.from_numpy(np.asarray(image)), new_shape=patch,
                             mode='constant', kwargs={'value': 0}, return_slicer=True)
    sample = padded[(slice(None), *(slice(0, n) for n in patch))].contiguous()
    predictor = predictor_for(network, device, tta)
    trials = []
    candidate = 1
    while candidate <= maximum:
        if progress:
            progress(candidate, maximum, 'capacity_probe')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        data = output = None
        fits = True
        # 已知 CUDA 容量错误属于批准的有界资源调整；其他错误直接传播。
        try:
            data = torch.stack([sample]*candidate).pin_memory()
            output = forward_windows(predictor, data, device)
            stamp = time.perf_counter()
            for _ in range(2):
                output = forward_windows(predictor, data, device)
            seconds = (time.perf_counter()-stamp)/2
            peak = int(torch.cuda.max_memory_reserved(device))
            free, _ = torch.cuda.mem_get_info(device)
            fits = peak <= admission['usable_bytes'] and free >= admission['reserve_bytes']
        except torch.cuda.OutOfMemoryError:
            fits, seconds, peak = False, None, None
        finally:
            del data, output
            torch.cuda.empty_cache()
        trials.append(dict(batch=candidate, fits=fits, peak_reserved_bytes=peak, seconds=seconds,
                           windows_per_second=candidate/seconds if fits else None))
        if not fits:
            break
        candidate *= 2
    valid = [row for row in trials if row['fits']]
    if not valid:
        raise MemoryError('单窗口也不满足显存预算；保持原 patch，等待其他获准设备')
    fastest = max(row['windows_per_second'] for row in valid)
    chosen = next(row for row in valid if row['windows_per_second'] >= .95*fastest)
    return chosen['batch'], {'selected': chosen['batch'], 'trials': trials,
                             'rule': '保留显存余量；选择达到最佳吞吐95%的最小批量'}
