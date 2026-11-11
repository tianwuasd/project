from contextlib import nullcontext
from functools import lru_cache
from itertools import product
from math import prod
import pickle
import os
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..observability.status import write_status
from ..training.marginal import MarginalBatchLoss
from ..xu_training import _save
from .stunet_architecture import STUNet


WEIGHT_SHA256 = 'f440f401bf4cac1d1d6f7c4635542ef057193a5f96c0b3c661a6179afeab7be6'


def topology(spacing, patch):
    from nnunetv2.experiment_planning.experiment_planners.network_topology import get_pool_and_conv_props

    _, pools, _, adjusted, divisors = get_pool_and_conv_props(spacing, patch, 4, 5)
    if list(adjusted) != list(patch) or np.prod(np.asarray(patch)//divisors) <= 1:
        raise ValueError('训练块尺寸不满足 STU-Net 步长或归一化要求')
    strides = [list(map(int, values)) for values in pools[1:]]
    return strides + [[1,1,1]]*(5-len(strides))


def network(strides, deep_supervision):
    return STUNet(1,11,depth=[1]*6,dims=[16,32,64,128,256,256],
                  pool_op_kernel_sizes=strides,conv_kernel_sizes=[[3]*3]*6,
                  enable_deep_supervision=deep_supervision)


def load_pretrained(model, path):
    if sha256_file(path) != WEIGHT_SHA256:
        raise ValueError('STU-Net-S 官方权重 SHA256 不匹配')
    allowed = [(np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'), np.dtype,
               type(np.dtype(np.float32)), type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed):
        source = torch.load(path, map_location='cpu', weights_only=True)['state_dict']
    own = model.state_dict()
    names = {key for key in own if not key.startswith('seg_outputs.')}
    if names != {key for key in source if not key.startswith('seg_outputs.')}:
        raise ValueError('预训练骨干层名不匹配')
    missing, unexpected = model.load_state_dict({key:source[key] for key in names}, strict=False)
    if set(missing) != own.keys()-names or unexpected or not all(
            torch.equal(model.state_dict()[key],source[key]) for key in names):
        raise ValueError('预训练骨干权重未完整加载')
    return dict(loaded_tensors=len(names), reset_head_tensors=len(missing), sha256=WEIGHT_SHA256)


class PatchReader:
    def __init__(self, cache, patch):
        self.cache, self.patch = cache, np.asarray(patch)

    @lru_cache(maxsize=64)
    def locations(self, name):
        relative = f'{self.cache._folder}/{name}.pkl'
        # 仅加载已完成的本项目预处理产物，读取前核验原提交摘要。
        with self.cache._artifact(f'process-{name}',relative).open('rb') as stream:
            props = pickle.load(stream)
        classes = self.cache.supervision[name]['present_classes']
        locations = props['class_locations']
        return [locations[c] for c in classes if c in locations and len(locations[c])]

    def read(self, name, rng, foreground):
        shape = np.asarray(self.cache.geometry(name).cache_shape)
        extent = np.minimum(shape,self.patch)
        available = self.locations(name) if foreground else []
        if available:
            points = available[rng.integers(len(available))]
            center = points[rng.integers(len(points))]
            if len(center) != 4 or center[0] != 0:
                raise ValueError('预处理取样点必须遵循 channel + 三个空间轴的约定')
            start = np.clip(center[1:]-extent//2,0,shape-extent)
        else:
            start = rng.integers(0,shape-extent+1)
        box = tuple((int(a),int(b)) for a,b in zip(start,start+extent))
        pad = tuple((0,int(n)) for n in self.patch-extent)
        image = np.pad(self.cache.read_image(name,box=box),pad)
        label = np.pad(self.cache.read_label(name,box=box).astype(np.int8,copy=False),
                       pad,constant_values=-1)
        return image,label


def arithmetic(device, precision):
    if precision == 'fp32':
        return nullcontext()
    if device.type!='cuda' or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise ValueError('此设备不支持原生 BF16，请使用 --precision fp32 开新任务')
    return torch.autocast('cuda',dtype=torch.bfloat16)


def run_epoch(model, reader, names, settings, device, loss, optimizer, seed):
    training = optimizer is not None
    model.train(training)
    rng = np.random.default_rng(seed)
    total = 0.
    steps = settings['train_steps'] if training else settings['development_steps']
    with torch.set_grad_enabled(training):
        for step in range(steps):
            keys = rng.choice(names,size=settings['batch']).tolist()
            pairs = [reader.read(name,rng,step == 0 or rng.random() < .33) for name in keys]
            data, target = (torch.from_numpy(np.stack(values))[:,None].to(device) for values in zip(*pairs))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with arithmetic(device,settings['precision']):
                outputs = model(data)
                targets = [F.interpolate(target.float(),size=out.shape[2:],mode='nearest').long()
                           for out in outputs]
                loss.bind(keys,reader.cache.supervision)
                value = loss(outputs,targets)
            if not bool(torch.isfinite(value)):
                raise ValueError('STU-Net 损失为非有限值')
            if training:
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),12.,error_if_nonfinite=True)
                optimizer.step()
            total += value.item()
            if (step+1) % 20 == 0 or step+1 == steps:
                print(dict(phase='train' if training else 'development',step=step+1,
                           steps=steps,updates=step+1 if training else 0,loss=total/(step+1)),flush=True)
    return total/steps


def _owned_checkpoint(path):
    path = Path(path)
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError('STU-Net 检查点必须是本运行目录中的普通文件')
    if hasattr(os,'getuid') and path.stat().st_uid != os.getuid():
        raise ValueError('STU-Net 检查点属于其他用户')


def _prune_best_checkpoints(output, identity):
    output = Path(output).absolute()
    _owned_checkpoint(output/'run.json')
    _owned_checkpoint(output/'training.json')
    job = read_json(output/'run.json')
    if (output.resolve() != output or digest(job) != identity
            or job['model'] != 'stunet' or job['command'] != 'train'):
        raise ValueError('STU-Net 清理目录与原运行身份不一致')
    summary = read_json(output/'training.json')
    last_path = output/'last.pt'
    _owned_checkpoint(last_path)
    last_hash = sha256_file(last_path)
    last = torch.load(last_path,map_location='cpu',weights_only=True,mmap=True)
    epoch = last['epoch']
    if (last['job'] != identity or summary['job'] != identity or summary['epoch'] != epoch
            or summary['history'] != last['history'] or len(last['history']) != epoch
            or [row['epoch'] for row in last['history']] != list(range(1,epoch+1))
            or not isinstance(last['optimizer'],dict) or not {'state','param_groups'} <= last['optimizer'].keys()
            or not last['optimizer']['param_groups']):
        raise ValueError('STU-Net 最新优化器与已提交训练摘要不一致')
    best_epoch = last['best_epoch']
    best_name = f'best-{best_epoch}.pt'
    if (not 1 <= best_epoch <= epoch or summary['best_file'] != best_name
            or summary['best_sha256'] != last['best_sha256']
            or min(last['history'],key=lambda row:row['development_loss'])['epoch'] != best_epoch):
        raise ValueError('STU-Net 已选最佳轮次不一致')
    selected = output/best_name
    _owned_checkpoint(selected)
    selected_hash = sha256_file(selected)
    state = torch.load(selected,map_location='cpu',weights_only=True,mmap=True)
    if (selected_hash != last['best_sha256'] or state['job'] != identity
            or state['epoch'] != best_epoch or not isinstance(state['model'],dict) or not state['model']):
        raise ValueError('STU-Net 已选推理权重不完整')
    del state, last
    planned = []
    for path in sorted(output.glob('best-*.pt')):
        match = re.fullmatch(r'best-([1-9][0-9]*)\.pt',path.name)
        if match is None or path == selected:
            continue
        _owned_checkpoint(path)
        checksum = sha256_file(path)
        state = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        candidate_epoch = int(match.group(1))
        if state['job'] != identity or state['epoch'] != candidate_epoch or not isinstance(state['model'],dict) or not state['model']:
            raise ValueError('待清理 STU-Net 权重的归属或轮次不一致')
        del state
        if candidate_epoch <= epoch:
            planned.append((path,checksum))
    # 全部候选验证完成后，再确认最新状态和选定权重仍然完整。
    if sha256_file(last_path) != last_hash or sha256_file(selected) != selected_hash or read_json(output/'training.json') != summary:
        raise ValueError('STU-Net 状态在清理检查期间发生变化')
    for path,checksum in planned:
        _owned_checkpoint(path)
        if sha256_file(path) != checksum:
            raise ValueError('待清理 STU-Net 权重发生变化')
    for path,_ in planned:
        path.unlink()
    return [path.name for path,_ in planned]


def train(cache, job, output, epochs, device, weights_path=None):
    settings = job['settings']
    identity = digest(job)
    last_path = output/'last.pt'
    torch.manual_seed(settings['seed'])
    model = network(settings['strides'],True)
    previous = None
    if last_path.exists():
        previous = torch.load(last_path,map_location='cpu',weights_only=True)
        if previous['job'] != identity or previous['epoch'] > epochs:
            raise ValueError('检查点与训练设置或目标轮数不兼容')
        model.load_state_dict(previous['model'],strict=True)
    else:
        atomic_json(output/'pretrained.json',load_pretrained(model,weights_path or job['weights']))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),lr=settings['learning_rate'],
                                momentum=.99,nesterov=True,weight_decay=3e-5)
    history = []
    if previous:
        optimizer.load_state_dict(previous['optimizer'])
        history = previous['history']
        best_path = output/f'best-{previous["best_epoch"]}.pt'
        if not best_path.exists() or sha256_file(best_path) != previous['best_sha256']:
            raise ValueError('已选开发集权重摘要不一致')
        atomic_json(output/'training.json',dict(job=identity,epoch=previous['epoch'],history=history,
                    best_file=best_path.name,best_sha256=previous['best_sha256']))
        del previous
        _prune_best_checkpoints(output,identity)
    reader = PatchReader(cache,settings['patch'])
    loss = MarginalBatchLoss()
    best = min((row['development_loss'] for row in history),default=float('inf'))
    for epoch in range(len(history)+1,epochs+1):
        write_status(output,state='running',stage='stunet-train',epoch=epoch,epochs=epochs)
        tr = run_epoch(model,reader,cache.case_ids('train'),settings,device,loss,optimizer,
                       settings['seed']+epoch)
        val = run_epoch(model,reader,cache.case_ids('development'),settings,device,loss,None,
                        settings['seed'])
        row = dict(epoch=epoch,training_loss=tr,development_loss=val,updates=settings['train_steps'])
        history.append(row)
        state = dict(job=identity,epoch=epoch,model=model.state_dict())
        if val < best:
            _save(output/f'best-{epoch}.pt',state)
            best = val
            best_epoch = epoch
        else:
            best_epoch = min(history,key=lambda item:item['development_loss'])['epoch']
        best_file = output/f'best-{best_epoch}.pt'
        best_hash = sha256_file(best_file)
        _save(last_path,dict(state,optimizer=optimizer.state_dict(),
                             history=history,best_sha256=best_hash,best_epoch=best_epoch))
        atomic_json(output/'training.json',dict(job=identity,epoch=epoch,history=history,
                    best_file=best_file.name,best_sha256=best_hash))
        _prune_best_checkpoints(output,identity)
        print(row,flush=True)


def load_predictor(training, cache, device):
    manifest = read_json(training/'run.json')
    summary = read_json(training/'training.json')
    if (manifest['model'] != 'stunet' or manifest['command'] != 'train'
            or manifest['prepared_digest'] != cache.request_digest or summary['job'] != digest(manifest)):
        raise ValueError('训练权重与现有预处理或模型不一致')
    path = training/summary['best_file']
    if path.parent != training or sha256_file(path) != summary['best_sha256']:
        raise ValueError('已选 STU-Net 检查点摘要不匹配')
    state = torch.load(path,map_location='cpu',weights_only=True)
    if state['job'] != summary['job']:
        raise ValueError('STU-Net 检查点归属不一致')
    model = network(manifest['settings']['strides'],False)
    model.load_state_dict(state['model'],strict=True)
    precision = manifest['settings']['precision'] if device.type=='cuda' else 'fp32'
    return model.to(device).eval(),tuple(manifest['settings']['patch']),precision


@torch.inference_mode()
def predict(model, image, patch, device, precision):
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window

    padding = [(0,max(0,p-n)) for n,p in zip(image.shape,patch)]
    padded = np.pad(image,padding) if any(end for _,end in padding) else image
    steps = compute_steps_for_sliding_window(padded.shape,patch,.5)
    weight = compute_gaussian(patch,dtype=torch.float32,device=torch.device('cpu')).numpy()
    combined = np.zeros((11,*padded.shape),dtype=np.float32)
    total = prod(map(len,steps))
    started = perf_counter()
    for step,(i,j,k) in enumerate(product(*steps),1):
        window = tuple(slice(start,start+size) for start,size in zip((i,j,k),patch))
        data = torch.from_numpy(np.ascontiguousarray(padded[window]))[None,None].to(device)
        with arithmetic(device,precision):
            logits = model(data)[0].float().cpu().numpy()
        if not np.isfinite(logits).all():
            raise ValueError('STU-Net 滑窗输出非有限数值')
        combined[(slice(None),*window)] += logits*weight
        if step % 50 == 0 or step == total:
            print(dict(stage='stunet-sliding-window',windows=step,total=total,
                       seconds=round(perf_counter()-started,2)),flush=True)
    x,y,z = image.shape
    result = np.empty(image.shape,dtype=np.uint8)
    for start in range(0,x,8):
        end = min(start+8,x)
        result[start:end] = combined[:,start:end,:y,:z].argmax(0)
    return result
