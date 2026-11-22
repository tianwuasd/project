from importlib.metadata import version
from contextlib import redirect_stdout, redirect_stderr
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import psutil

from ...artifacts.store import read_json, write_json
from ...acceleration.legacy import SessionGuard
from ..medsam3.input import open_image, rgb_slice


def run(root, handle):
    with (Path(root) / 'worker.log').open('w', encoding='utf-8') as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            execute(root, handle)


def execute(root, handle):
    import torch
    from transformers import Sam3Model, Sam3Processor

    root = Path(root)
    config = read_json(root / 'worker.json')
    # 官方同步逐张量加载，避免异步预读并存形成额外主存峰值。
    os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
    began = perf_counter()
    guard = SessionGuard(handle)
    guard.check_active()
    torch.set_num_threads(config['resources']['cpu_threads'])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('本次显卡不支持请求的CUDA BF16')
    free, total = torch.cuda.mem_get_info()
    cap = min(config['effective_vram_limit_bytes'], free - 2**30)
    if cap <= 0:
        raise MemoryError('CUDA初始化后显存不足以保留1GiB')
    torch.cuda.set_per_process_memory_fraction(cap / total)
    torch.cuda.reset_peak_memory_stats()
    summary = dict(state='loading', training_updates=0, attempted_queries=0,
                   effective_vram_limit_bytes=cap, versions={n: version(n) for n in
                   ('torch', 'transformers', 'modelscope', 'tokenizers', 'safetensors')},
                   prompt=config['prompt'], completed_slices=[])
    write_json(root / 'summary.json', summary)
    guard.check_active()
    model, info = Sam3Model.from_pretrained(config['weights'], local_files_only=True,
                                           dtype=torch.bfloat16, attn_implementation='sdpa',
                                           output_loading_info=True)
    if info['missing_keys'] or info['mismatched_keys'] or info['error_msgs']:
        raise ValueError('官方SAM3图像权重未完整加载：' + str(info))
    info = {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}
    # 同库总权重包含视频跟踪器；保留官方加载器报告的非图像参数清单供审查。
    write_json(root / 'loading.json', info)
    if any(not k.startswith(('tracker_model.', 'tracker_neck.')) for k in info['unexpected_keys']):
        raise ValueError('总权重包含预期外的未加载参数')
    guard.check_active()
    model = model.to('cuda').eval()
    processor = Sam3Processor.from_pretrained(config['weights'], local_files_only=True)
    summary.update(state='predicting', load_seconds=perf_counter() - began,
                   parameters=sum(p.numel() for p in model.parameters()))
    write_json(root / 'summary.json', summary)
    image = open_image(config['selected']['row'])
    for index in config['selected']['slices']:
        started = perf_counter()
        guard.check_active()
        if psutil.virtual_memory().percent > 80:
            raise MemoryError('整机内存达到80%限制')
        rgb = rgb_slice(image, index, config)
        inputs = processor(images=rgb, text=config['prompt'], return_tensors='pt').to('cuda')
        summary['attempted_queries'] += 1
        write_json(root / 'summary.json', summary)
        guard.check_active()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(**inputs)
        if not all(torch.isfinite(t).all().item() for t in
                   (outputs.pred_masks, outputs.pred_logits, outputs.pred_boxes)):
            raise ValueError('SAM3输出包含非有限值')
        processed = processor.post_process_instance_segmentation(outputs, threshold=.5,
                    mask_threshold=.5, target_sizes=inputs['original_sizes'].tolist())[0]
        masks = processed['masks'].cpu().numpy().astype(np.uint8)
        boxes = processed['boxes'].float().cpu().numpy()
        scores = processed['scores'].float().cpu().numpy()
        if masks.shape[1:] != rgb.shape[:2] or boxes.shape != (len(masks), 4) or scores.shape != (len(masks),):
            raise ValueError('原始切片与官方还原掩膜的形状不同')
        # 模型行列为 j,i；保存时恢复为实例,i,j，不构造未处理切片的背景体积。
        restored = masks.transpose(0, 2, 1).copy()
        guard.check_active()
        np.savez_compressed(root / f'slice-{index}.npz', masks=restored, boxes_xyxy=boxes, scores=scores)
        torch.cuda.synchronize()
        summary['completed_slices'].append(dict(index=index, instances=len(masks),
            original_rgb_shape=list(rgb.shape), restored_mask_shape=list(restored.shape),
            processor_shape=list(inputs['pixel_values'].shape), scores=scores.tolist(),
            seconds=perf_counter() - started, foreground_voxels=int(restored.sum())))
        write_json(root / 'summary.json', summary)
        del inputs, outputs, processed, masks, restored, rgb
    elapsed = perf_counter() - began
    summary.update(state='completed', total_seconds=elapsed,
                   slices_per_second=len(summary['completed_slices']) / elapsed,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                   peak_worker_rss_bytes=psutil.Process().memory_info().peak_wset,
                   coverage=dict(kind='selected_slices', indices=config['selected']['slices'],
                                 source_shape=config['selected']['source']['shape'],
                                 uncomputed='未推断，不能作为背景', array_axes='instance,i,j', slice_axis='k'))
    write_json(root / 'summary.json', summary)


