import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml
from filelock import FileLock

from ...artifacts.store import read_json, require_output, write_json
from .config import load_config
from .materials import FILES, REVISION


def settings(path, prepared=None):
    path = Path(path).resolve(strict=True)
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    required = {'base_config', 'prepared', 'run_dir', 'label_evidence', 'updates',
                'validate_every', 'learning_rate', 'seed'}
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError('微调配置字段不同于正式模板')
    if not isinstance(raw['label_evidence'], str) or not raw['label_evidence'].strip():
        raise ValueError('需要真实标签名称和类别映射的核对来源')
    for key in ('updates', 'validate_every'):
        if type(raw[key]) is not int or raw[key] < 1:
            raise ValueError('更新次数与验证间隔必须为正整数')
    if type(raw['seed']) is not int or not math.isfinite(raw['learning_rate']) or raw['learning_rate'] <= 0:
        raise ValueError('需要整数随机种子及正有限学习率')
    config = load_config(path.parent / raw['base_config'])
    source = Path(prepared).resolve(strict=True) if prepared else (path.parent / raw['prepared']).resolve(strict=True)
    data = read_json(source)
    if data.get('format') != 'sam3_semantic_slices_v1' or not data.get('samples'):
        raise ValueError('需要显式SAM3真实切片登记，格式sam3_semantic_slices_v1')
    patients, ids = {'train': set(), 'val': set()}, set()
    rows = []
    for original in data['samples']:
        row = dict(original)
        if (row['role'] not in patients or not row['patient'] or row['id'] in ids or
                Path(row['id']).name != row['id'] or row['id'] in ('.', '..')):
            raise ValueError('样本身份、患者或训练验证角色无效')
        if type(row['label_id']) is not int or row['label_id'] <= 0 or not row['prompt'].strip():
            raise ValueError('需要经过核对的前景类别及预先固定文本')
        if row['label_id'] in row.get('unknown_classes', []):
            raise ValueError('目标类别未被真实标注，禁止作为阴性样本')
        if type(row['slice']) is not int or row['slice'] < 0:
            raise ValueError('需要非负零基切片索引')
        fields = ('prepared',) if row.get('prepared') else ('image', 'label')
        for key in fields:
            row[key] = str((source.parent / row[key]).resolve(strict=True))
        row['source_files'] = {key: dict(bytes=Path(row[key]).stat().st_size,
            mtime_ns=Path(row[key]).stat().st_mtime_ns) for key in fields}
        ids.add(row['id'])
        patients[row['role']].add(row['patient'])
        rows.append(row)
    if not all(patients.values()) or patients['train'] & patients['val']:
        raise ValueError('训练验证必须均存在且按患者完全分离')
    run = (path.parent / raw['run_dir']).resolve()
    require_output(run, config['work_root'], [config['weights'], config['data_root'], source])
    config.update(finetune=raw, samples=rows, run_dir=str(run), prepared=str(source))
    # 身份不依赖输出目录；续训仍严格匹配数据、划分、提示与优化设置。
    identity = dict(config=raw, samples=rows, revision=REVISION,
        input_semantics={key: config[key] for key in ('intensity_semantics', 'source_mean',
                        'source_std', 'gray_window', 'orientation_evidence', 'intensity_evidence')})
    identity['config'] = {k: v for k, v in raw.items() if k not in ('run_dir', 'prepared', 'base_config')}
    config['training_identity'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for name, size in FILES.items():
        if (Path(config['weights']) / name).stat().st_size != size:
            raise ValueError('固定原模型材料缺失或大小不同：' + name)
    return config


def sample(row, config):
    from ..medsam3.input import open_image, rgb_slice
    import nibabel as nib

    for key, expected in row['source_files'].items():
        actual = Path(row[key]).stat()
        if (actual.st_size, actual.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
            raise ValueError('切片输入在登记后发生变化')

    if row.get('prepared'):
        with np.load(row['prepared'], allow_pickle=False) as item:
            rgb, labels, valid = item['rgb'].copy(), item['labels'].copy(), item['valid'].copy()
    else:
        image, label = open_image(row), nib.load(row['label'], keep_file_open=True)
        if image.shape != label.shape or not np.allclose(image.affine, label.affine, atol=1e-6, rtol=0):
            raise ValueError('真实影像标签网格不匹配，禁止自动修改几何')
        if row['slice'] >= image.shape[2]:
            raise ValueError('训练切片超出原始体积')
        rgb = rgb_slice(image, row['slice'], config)
        labels = np.asarray(label.dataobj[:, :, row['slice']]).T
        valid = ~np.isin(labels, row.get('ignore_label_ids', []))
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or
            labels.shape != rgb.shape[:2] or valid.shape != labels.shape or valid.dtype != np.bool_):
        raise ValueError('切片合同需要uint8 RGB、同网格labels和bool valid')
    if not np.isfinite(labels).all() or not np.equal(labels, np.rint(labels)).all() or not valid.any():
        raise ValueError('标签必须为有限整数且存在有效像素')
    valid &= ~np.isin(labels, row.get('ignore_label_ids', []))
    if not valid.any():
        raise ValueError('排除忽略标签后没有有效监督像素')
    return rgb, labels == row['label_id'], valid


def execute(config, operation, deadline, resume=False, checkpoint='best', output=None, stop_after=None):
    sys.path.insert(0, config['dependencies'])
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DEACTIVATE_ASYNC_LOAD='1')
    import torch
    import torch.nn.functional as functional
    from transformers import Sam3Model, Sam3Processor
    from ...acceleration import ResourceDemand, ResourceSession, RuntimeRequest
    from ...observability.events import event

    if not math.isfinite(deadline) or deadline <= time.time():
        raise ValueError('训练操作需要未来的固定Unix截止时间')
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('stop_after需要正整数')
    if operation not in ('train', 'validate', 'segment') or checkpoint not in ('best', 'latest'):
        raise ValueError('操作或检查点名称无效')
    run = Path(config['run_dir'])
    if operation == 'train' and not resume and run.exists():
        raise FileExistsError('新训练目录已存在；续训须显式--resume')
    if operation != 'train' or resume:
        (run / (('latest' if resume else checkpoint) + '.pt')).resolve(strict=True)
    destination = None
    if operation == 'segment':
        if output is None:
            raise ValueError('语义预测需要显式独占输出目录')
        destination = require_output(output, config['work_root'], [run, config['data_root'], config['weights']])
        destination.mkdir(parents=True, exist_ok=False)
    run.mkdir(parents=True, exist_ok=operation != 'train' or resume)
    torch.manual_seed(config['finetune']['seed'])
    torch.set_num_threads(config['resources']['cpu_threads'])
    resources = dict(config['resources'])
    resources['max_wait'] = min(resources['max_wait'], max(0, deadline-time.time()))
    resources['vram_reserve_bytes'] += config['estimated_vram_bytes']
    request = RuntimeRequest(str(run), 'sam3_semantic_'+operation, config['registry_root'], str(run), resources,
                              ResourceDemand(fixed_bytes=config['estimated_ram_bytes'], cpu_slots=resources['cpu_threads'],
                                             gpu_count=1, gpu_ids=resources['gpu_ids'], local_batches=[1]))
    state = dict(step=0, best_loss=float('inf'), identity=config['training_identity'])

    def load(path=None):
        model, info = Sam3Model.from_pretrained(config['weights'], local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True)
        if (info['missing_keys'] or info['mismatched_keys'] or info['error_msgs'] or
                any(not k.startswith(('tracker_model.', 'tracker_neck.')) for k in info['unexpected_keys'])):
            raise ValueError('完整SAM3图像权重键不匹配')
        model.requires_grad_(False)
        head = model.mask_decoder.semantic_projection
        head.float().requires_grad_(True)
        model = model.to('cuda').eval()
        optim = torch.optim.AdamW(head.parameters(), lr=config['finetune']['learning_rate'])
        if path:
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['identity'] != config['training_identity'] or saved['revision'] != REVISION:
                raise ValueError('检查点与数据、提示、类别、划分或原权重身份不同')
            head.load_state_dict(saved['head'], strict=True)
            optim.load_state_dict(saved['optimizer'])
            state.update(step=saved['step'], best_loss=saved['best_loss'])
            torch.set_rng_state(saved['rng'])
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        return model, optim

    def save(name):
        pending = run / (name+'.pt.pending')
        torch.save(dict(identity=config['training_identity'], revision=REVISION,
            head={k: v.detach().cpu() for k, v in model.mask_decoder.semantic_projection.state_dict().items()},
            optimizer=optimizer.state_dict(), step=state['step'], best_loss=state['best_loss'],
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), pending)
        os.replace(pending, run / (name+'.pt'))

    def boundary():
        if time.time() >= deadline:
            raise TimeoutError('SAM3微调达到固定截止，latest保留最近完成更新')
        snap = client.snapshot()
        if snap.cancelled or snap.sample_age_seconds > 30:
            raise InterruptedError('公共监控取消或采样过期')
        import psutil
        if psutil.Process().memory_info().rss > config['ram_limit_bytes']:
            raise MemoryError('SAM3微调超过显式进程内存上限')
        if torch.cuda.mem_get_info()[0] < 2**30:
            raise MemoryError('显卡实际空闲量低于1GiB预留')
        return snap.paused

    def forward(row):
        rgb, target, valid = sample(row, config)
        inputs = processor(images=rgb, text=row['prompt'], return_tensors='pt').to('cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(**inputs).semantic_seg
        logits = functional.interpolate(logits.float(), size=target.shape, mode='bilinear', align_corners=False)[0, 0]
        truth = torch.as_tensor(target, device='cuda', dtype=torch.float32)
        mask = torch.as_tensor(valid, device='cuda')
        selected, reference = logits[mask], truth[mask]
        probability = selected.sigmoid()
        loss = functional.binary_cross_entropy_with_logits(selected, reference)
        loss = loss + 1 - (2*(probability*reference).sum()+1)/(probability.sum()+reference.sum()+1)
        if not torch.isfinite(loss):
            raise ValueError('真实语义分割损失非有限值')
        return loss, logits, truth, mask

    def active():
        nonlocal model, optimizer
        if boundary():
            if operation == 'train':
                save('latest')
            del model, optimizer
            torch.cuda.empty_cache()
            client.handle.resources['max_wait'] = min(resources['max_wait'], max(0, deadline-time.time()))
            client.wait_active()
            model, optimizer = load(run/(('latest' if operation == 'train' else checkpoint)+'.pt'))
            event(run, 'sam3_checkpoint_recovered', step=state['step'])
            boundary()

    def validate():
        values, dice = [], []
        with torch.inference_mode():
            for row in val:
                active()
                loss, logits, truth, mask = forward(row)
                prediction = logits[mask] > 0
                reference = truth[mask] > .5
                denom = int(prediction.sum()+reference.sum())
                values.append(loss.item())
                dice.append(2*int((prediction & reference).sum())/denom if denom else 1.)
                if destination is not None:
                    np.savez_compressed(destination/(row['id']+'.npz'),
                        masks=(logits > 0).cpu().numpy().T, valid=mask.cpu().numpy().T)
                del loss, logits, truth, mask
        result = dict(loss=sum(values)/len(values), dice=sum(dice)/len(dice), samples=len(values),
                      coverage=[dict(id=r['id'], slice=r['slice']) for r in val], array_axes='i,j')
        event(run, 'sam3_validation', step=state['step'], **result)
        return result

    train = [row for row in config['samples'] if row['role'] == 'train']
    val = [row for row in config['samples'] if row['role'] == 'val']
    with FileLock(str(run/'operation.lock'), timeout=0), ResourceSession(request) as session:
        from ...acceleration.runtime import attach
        client = attach(session.handle)
        client.wait_active()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('语义头配方需要CUDA BF16')
        gpu = session.allocation.gpus[0]
        torch.cuda.set_device(gpu['index'])
        free, total = torch.cuda.mem_get_info()
        cap = min(config['vram_limit_bytes'], free-2**30)
        if cap <= 0:
            raise MemoryError('CUDA初始化后无法保留1GiB')
        torch.cuda.set_per_process_memory_fraction(cap/total)
        model, optimizer = load(run/'latest.pt' if resume else run/(checkpoint+'.pt') if operation != 'train' else None)
        processor = Sam3Processor.from_pretrained(config['weights'], local_files_only=True)
        event(run, 'sam3_semantic_started', operation=operation, step=state['step'], revision=REVISION,
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        if operation != 'train':
            result = validate()
            write_json((destination or run)/('semantic_'+operation+'.json'), result)
            return result
        end = min(config['finetune']['updates'], state['step']+stop_after) if stop_after else config['finetune']['updates']
        while state['step'] < end:
            active()
            began = time.monotonic()
            row = train[state['step'] % len(train)]
            optimizer.zero_grad(set_to_none=True)
            loss, logits, truth, mask = forward(row)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.mask_decoder.semantic_projection.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            state['step'] += 1
            event(run, 'sam3_semantic_update', step=state['step'], loss=loss.item(), seconds=time.monotonic()-began)
            del loss, logits, truth, mask
            save('latest')
            if state['step'] % config['finetune']['validate_every'] == 0 or state['step'] == end:
                result = validate()
                if result['loss'] < state['best_loss']:
                    state['best_loss'] = result['loss']
                    save('best')
                save('latest')
        result = dict(step=state['step'], best_loss=state['best_loss'], strategy='semantic_projection_bce_dice',
                      revision=REVISION, completed=state['step'] == config['finetune']['updates'])
        write_json(run/'finetune.json', result)
        return result
