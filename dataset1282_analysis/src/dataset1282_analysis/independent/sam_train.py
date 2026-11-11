import argparse
from datetime import timedelta
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from ..contracts import atomic_json, digest, read_json, sha256_file
from ..observability.status import operation_lock, write_status
from ..xu_dispatch import Children, run_wave, select_devices
from ..xu_geometry import PreparedCache
from ..xu_gpu_isolation import configure_coordinator, validate_entry
from ..xu_pipeline import ROOT, _execute
from ..xu_resources import ResourceUnavailable, apply_env_limit, combined_environment, gpu_budget, reserve_wave, stage_resources
from ..xu_training import _save
from .sam_med3d import SPACING_MM, WIDTH, WEIGHT_SHA256, load_model, prompt_patch


CUDA_BYTES = 3*1024**3
CASE_SEED = 2023


class PromptHead(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.prompt_encoder = model.prompt_encoder
        self.mask_decoder = model.mask_decoder

    def forward(self, embedding, point, point_type, previous):
        sparse, dense = self.prompt_encoder(points=(point, point_type), boxes=None, masks=previous)
        low, _ = self.mask_decoder(image_embeddings=embedding, image_pe=self.prompt_encoder.get_dense_pe(),
                                   sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense)
        return F.interpolate(low, size=(WIDTH,)*3, mode='trilinear', align_corners=False)


def sample(cache, name, label, seed):
    geometry = cache.geometry(name)
    reference = cache.read_native_label(name)
    foreground = np.flatnonzero(reference == label)
    if not foreground.size:
        raise ValueError('训练病例缺少指定的已标注结构')
    rng = np.random.default_rng(seed)
    point = [int(x) for x in np.unravel_index(rng.choice(foreground), reference.shape)]
    relative = f'staged/{name}/image{cache.staged_extension}'
    image = nib.load(cache._artifact(f'stage-{name}', relative))
    if image.shape != reference.shape or not np.allclose(image.affine, np.diag([*geometry.spacing_xyz, 1.])):
        raise ValueError('训练影像与标签的原网格不匹配')
    values = image.get_fdata(dtype=np.float32, caching='unchanged')
    patch, center, scale, offset = prompt_patch(values, geometry.spacing_xyz, point)
    target = affine_transform(reference, np.diag(scale), offset=offset, output_shape=patch.shape,
                              order=0, mode='grid-constant', prefilter=False) == label
    if not target[tuple(center)] or not np.isfinite(patch).all():
        raise ValueError('提示点变换或训练块灰度无效')
    return patch, target, center


def fingerprint(head):
    result = sha256()
    for name, value in head.named_parameters():
        result.update(name.encode('utf-8'))
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def forward_patch(encoder, head, cache, name, label, seed, device):
    patch, target, center = sample(cache, name, label, seed)
    image = torch.from_numpy(patch)[None,None].to(device)
    target = torch.from_numpy(target)[None,None].to(device,dtype=torch.float32)
    point = torch.as_tensor(center,device=device,dtype=torch.float32)[None,None]
    point_type = torch.ones((1,1),device=device,dtype=torch.long)
    previous = torch.zeros((1,1,*((WIDTH//4,)*3)),device=device)
    with torch.no_grad():
        embedding = encoder(image)
    return head(embedding,point,point_type,previous),target


def update_model(encoder, head, cache, case, device, optimizer, settings):
    optimizer.zero_grad(set_to_none=True)
    logits,target = forward_patch(encoder,head,cache,*case,device)
    probability = logits.sigmoid()
    smooth = settings['dice_smoothing']
    dice = (2*(probability*target).sum()+smooth)/(probability.sum()+target.sum()+smooth)
    loss = F.binary_cross_entropy_with_logits(logits,target) + 1-dice
    if not bool(torch.isfinite(loss)):
        raise ValueError('SAM-Med3D 损失非有限')
    loss.backward()
    nn.utils.clip_grad_norm_(head.parameters(),settings['gradient_clip'],error_if_nonfinite=True)
    optimizer.step()
    return loss.item()


def training_epoch(cache, names, epoch, size, rank):
    order = np.random.default_rng(CASE_SEED+epoch).permutation(names).tolist()
    padding = (-len(order)) % size
    order.extend(order[:padding])
    for index in range(rank,len(order),size):
        name = order[index]
        classes = cache.supervision[name]['present_classes']
        offset = int.from_bytes(sha256(name.encode('utf-8')).digest()[:4],'little')
        yield name,classes[(offset+epoch-1)%len(classes)],CASE_SEED+epoch*len(order)+index


def development_cases(cache, names):
    counts = dict.fromkeys(range(1,11),0)
    for index,name in enumerate(names):
        label = min(cache.supervision[name]['present_classes'],key=lambda x:(counts[x],x))
        counts[label] += 1
        yield name,label,CASE_SEED+index


@torch.no_grad()
def validate_epoch(encoder, head, cache, cases, device, rank, size):
    head.eval()
    scores = torch.zeros((2,10),device=device,dtype=torch.float64)
    for name,label,seed in cases[rank::size]:
        logits,target = forward_patch(encoder,head,cache,name,label,seed,device)
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('SAM-Med3D 开发集输出非有限')
        mask = logits > 0
        scores[0,label-1] += 2*(mask*target).sum()/(mask.sum()+target.sum())
        scores[1,label-1] += 1
    if size > 1:
        dist.all_reduce(scores)
    totals,counts = scores.cpu().tolist()
    by_label = {label+1:dict(dice=total/count, cases=int(count))
                for label,(total,count) in enumerate(zip(totals,counts)) if count}
    mean = sum(row['dice'] for row in by_label.values())/len(by_label)
    return mean,by_label


def save_epoch(output, model, optimizer, history, contract, best):
    row = history[-1]
    epoch = row['epoch']
    old_best = best
    if best is None or row['development_dice'] > best['dice']:
        best_path = output/f'best-{epoch}.pt'
        _save(best_path,dict(model_state_dict=model.state_dict(),epoch=epoch))
        best = dict(epoch=epoch,dice=row['development_dice'],sha256=sha256_file(best_path))
    _save(output/'last.pt',dict(model_state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
                               contract=contract,history=history,best=best))
    atomic_json(output/'training.json',dict(epoch=epoch,history=history,best=best,
                best_file=f"best-{best['epoch']}.pt"))
    if old_best and old_best['epoch'] != best['epoch']:
        (output/f"best-{old_best['epoch']}.pt").unlink(missing_ok=True)
    return best


def train_formal(output, request, cache, model, head, training, optimizer, device, rank, size):
    history, best = [],None
    if (output/'last.pt').is_file():
        previous = torch.load(output/'last.pt',map_location='cpu',weights_only=True)
        if previous['contract'] != request['contract']:
            raise ValueError('正式训练的检查点与输入合同不符')
        model.load_state_dict(previous['model_state_dict'],strict=True)
        optimizer.load_state_dict(previous['optimizer'])
        history,best = previous['history'],previous['best']
        del previous
        if sha256_file(output/f"best-{best['epoch']}.pt") != best['sha256']:
            raise ValueError('开发集最优检查点的摘要不符')
    if len(history) > request['epochs']:
        raise ValueError('训练轮数小于检查点已经完成的轮数')
    if history and rank == 0:
        atomic_json(output/'training.json',dict(epoch=len(history),history=history,best=best,
                    best_file=f"best-{best['epoch']}.pt"))
    if size > 1:
        dist.barrier()
    before = fingerprint(head)
    encoder = model.image_encoder
    dev = list(development_cases(cache,request['development']))
    updates = 0
    total = (len(request['train'])+size-1)//size
    for epoch in range(len(history)+1,request['epochs']+1):
        head.train()
        losses = []
        for step,case in enumerate(training_epoch(cache,request['train'],epoch,size,rank),1):
            losses.append(update_model(encoder,training,cache,case,device,optimizer,request['training']))
            if step == 1 or step % 25 == 0 or step == total:
                print(dict(rank=rank,epoch=epoch,step=step,total=total,loss=losses[-1]),flush=True)
        mean,by_label = validate_epoch(encoder,head,cache,dev,device,rank,size)
        summary = dict(epoch=epoch,updates=len(losses),development_dice=mean,by_label=by_label)
        history.append(summary)
        updates += len(losses)
        if rank == 0:
            best = save_epoch(output,model,optimizer,history,request['contract'],best)
            print(summary,flush=True)
        if size > 1:
            dist.barrier()
    return before,updates,len(history)


def train_rank(output, rank, device):
    request = read_json(output/'request.json')
    settings = request['training']
    card = request['cards'][rank]
    mapping, = json.loads(os.environ['XU_GPU_ISOLATION'])
    if mapping['uuid'] != card['uuid'] or mapping['device'] != device:
        raise ValueError('本训练进程未获得计划中的唯一显卡')
    limits = apply_env_limit(device)
    torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
    torch.cuda.set_device(device)
    size = len(request['cards'])
    if size > 1:
        timeout = timedelta(minutes=4)
        store = dist.TCPStore('127.0.0.1', request['port'], None, False, timeout, use_libuv=False)
        dist.init_process_group('nccl', store=store, rank=rank, world_size=size, timeout=timeout)
    cache = PreparedCache(request['prepared'], allow_technical_subset=request['technical_subset'])
    if cache.request_digest != request['prepared_digest']:
        raise ValueError('训练输入预处理摘要已变化')
    model = load_model(request['weights'], device)
    model.image_encoder.requires_grad_(False)
    head = PromptHead(model).train()
    training = (nn.parallel.DistributedDataParallel(head, device_ids=[0], find_unused_parameters=True)
                if size > 1 else head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings['learning_rate'],
                                  weight_decay=settings['weight_decay'])
    losses = []
    torch.cuda.reset_peak_memory_stats(device)
    epochs = request['epochs']
    if epochs:
        before,updates,completed = train_formal(output,request,cache,model,head,training,optimizer,device,rank,size)
    else:
        before = fingerprint(head)
        for step,name in enumerate(request['cases'][rank],1):
            case = name,request['label'],settings['seed']+size*step+rank
            losses.append(update_model(model.image_encoder,training,cache,case,device,optimizer,settings))
            print(dict(rank=rank,step=step,total=request['steps'],loss=losses[-1]),flush=True)
        updates,completed = len(losses),0
    after = fingerprint(head)
    if updates and after == before:
        raise ValueError('训练后没有检测到参数变化')
    if not epochs and rank == 0:
        _save(output/'last.pt', dict(model_state_dict=model.state_dict(),
                                    optimizer=optimizer.state_dict(), steps=len(losses)))
    atomic_json(output/f'rank-{rank}.json', dict(rank=rank, physical_index=limits['physical_index'],
                uuid=card['uuid'], before=before, after=after, losses=losses,
                optimizer_steps=updates, epochs_completed=completed,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(device)))
    if size > 1:
        dist.barrier()
        dist.destroy_process_group()


def wait_for_cards(args, output, children, host_bytes):
    candidates = [card for card in select_devices(None,args.devices,torch)
                  if card['device'] != 'cpu'
                  and gpu_budget(card['total_bytes'],card['total_bytes'],'v2') >= CUDA_BYTES]
    if len(candidates) < args.gpus:
        raise ValueError(f'当前授权范围只有 {len(candidates)} 张容量符合要求的显卡，需要 {args.gpus} 张')
    while True:
        cards, plans, denied = [], [], []
        try:
            reserve = reserve_wave(args.threads,'v2') if args.gpus==2 else None
        except ResourceUnavailable as error:
            denied.append(str(error))
        else:
            for card in candidates:
                try:
                    plan = stage_resources([card['device']], concurrent=args.gpus, threads=args.threads,
                        wave_reservation=reserve, require_pid_evidence=True,
                        host_demand_bytes=host_bytes, cuda_demand_bytes=CUDA_BYTES, owned_pids=children.pids())
                except ResourceUnavailable as error:
                    denied.append(str(error))
                    continue
                cards.append(card)
                plans.append(plan)
                if len(cards) == args.gpus:
                    return cards,plans
        reason = '；'.join(denied)
        message = f'可用显卡 {len(cards)}/{args.gpus}。{reason}'
        if args.wait_seconds == 0:
            raise ResourceUnavailable(message)
        write_status(output, state='waiting_resources', stage='sam-med3d-admission', active=[],
                     eligible=len(cards), required=args.gpus, next_check_seconds=args.wait_seconds, reason=reason)
        print(f'{message} {args.wait_seconds} 秒后重新检查。',flush=True)
        time.sleep(args.wait_seconds)


def separate_output(args):
    output = args.output.resolve()
    paths = (ROOT, args.prepared.resolve(), args.weights.resolve())
    if any(output == path or path in output.parents or output in path.parents for path in paths):
        raise ValueError('训练输出须与代码、预处理和权重分开')
    return output


def run(args):
    if args.threads < 1 or not 0 <= args.wait_seconds <= 300:
        raise ValueError('线程数须为正数，等待间隔须在 0 至 300 秒之间')
    if args.epochs is not None and (args.epochs < 1 or args.label is not None):
        raise ValueError('正式训练需要正数轮数，并按病例轮换已标注结构')
    if args.resume and args.epochs is None:
        raise ValueError('只有正式训练支持从完整轮次续跑')
    configure_coordinator()
    output = separate_output(args)
    if sha256_file(args.weights) != WEIGHT_SHA256:
        raise ValueError('需要固定版本的官方 Turbo 权重')
    cache = PreparedCache(args.prepared, allow_technical_subset=args.technical_subset)
    train = list(cache.case_ids('train'))
    development = list(cache.case_ids('development')) if args.epochs else []
    label = None if args.epochs else args.label or 1
    if args.epochs:
        if len(train) < args.gpus or not development:
            raise ValueError('正式训练需要足够的独立训练病例和开发病例')
        names = train+development
    else:
        eligible = [name for name in train if label in cache.supervision[name]['present_classes']]
        if len(eligible) < args.gpus*args.steps:
            raise ValueError('短测需要足够的互不重复且标有目标结构的训练病例')
        names = np.random.default_rng(CASE_SEED).choice(eligible,size=args.gpus*args.steps,replace=False).tolist()
    gib = 1024**3
    host_bytes = 3*gib + 10*max(prod(cache.geometry(name).native_shape_xyz) for name in names)
    if not args.resume:
        output.mkdir(mode=0o700, parents=True, exist_ok=False)
    if args.resume and not output.is_dir():
        raise ValueError('续跑需要已有的正式训练目录')
    stage = 'sam-med3d-train' if args.epochs else 'sam-med3d-trial'
    training = dict(optimizer='AdamW',learning_rate=8e-5,weight_decay=.1,gradient_clip=1.,
                    loss='BCEWithLogits + soft Dice',dice_smoothing=1.,seed=CASE_SEED,
                    precision='float32',patch_voxels=[WIDTH]*3,spacing_mm=[SPACING_MM]*3,
                    batch_per_gpu=1,global_batch=args.gpus,frozen=['image_encoder'],
                    trainable=['prompt_encoder','mask_decoder'])
    contract = dict(protocol='sam-med3d-epoch-v1',prepared_digest=cache.request_digest,
                    weights_sha256=WEIGHT_SHA256,train=train,development=development,training=training)
    with operation_lock(output,stage,output) as owner:
        children = Children(output, owner)
        try:
            if args.epochs:
                if args.resume:
                    if read_json(output/'contract.json') != contract:
                        raise ValueError('续跑的病例划分、权重或训练参数已变化')
                else:
                    atomic_json(output/'contract.json',contract)
            cards,plans = wait_for_cards(args,output,children,host_bytes)
            store = (dist.TCPStore('127.0.0.1', 0, None, True, timedelta(minutes=4), False, use_libuv=False)
                     if args.gpus==2 else None)
            request = dict(prepared=str(cache.run_dir), prepared_digest=cache.request_digest,
                           technical_subset=args.technical_subset, weights=str(args.weights.resolve()),
                           weights_sha256=WEIGHT_SHA256,label=label,steps=args.steps,cards=cards,
                           epochs=args.epochs,train=train if args.epochs else None,development=development,
                           contract=digest(contract) if args.epochs else None,
                           training=training, wait_seconds=args.wait_seconds,
                           port=store.port if store else None,
                           cases=None if args.epochs else [names[rank::args.gpus] for rank in range(args.gpus)])
            atomic_json(output/'request.json', request)
            children.configure_launch(args.gpus)

            def launch(rank):
                device = cards[rank]['device']
                plan = plans[rank]
                settings = SimpleNamespace(threads=plan['threads'], children=children, device=device,
                                           overlay=combined_environment(plan,[device]), assigned_devices=[device])
                _execute('sam_med3d_train.py', ['worker','--output',str(output),'--rank',str(rank),'--device',device],
                         f'sam-train-rank{rank}',output,owner,settings)

            run_wave(range(args.gpus), launch, children)
            ranks = [read_json(output/f'rank-{i}.json') for i in range(args.gpus)]
            if len({r['before'] for r in ranks}) != 1 or len({r['after'] for r in ranks}) != 1:
                raise ValueError('两张显卡的模型参数未同步')
            updates = args.steps
            formal = {}
            if args.epochs:
                summary = read_json(output/'training.json')
                if summary['epoch'] != args.epochs or any(r['epochs_completed']!=args.epochs for r in ranks):
                    raise ValueError('正式训练尚未完成目标轮数')
                updates = sum(row['updates'] for row in summary['history'])
                formal = dict(epochs_completed=args.epochs,best=summary['best'],best_file=summary['best_file'])
            receipt = dict(state='passed',ranks=ranks,label=label,optimizer_steps=updates,training=training,
                           checkpoint_sha256=sha256_file(output/'last.pt'),**formal)
            atomic_json(output/'result.json', receipt)
            write_status(output,state='complete',stage=stage,active=[])
            print(output/'result.json', flush=True)
        except (Exception, KeyboardInterrupt) as error:
            write_status(output, state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                         error=str(error), active=[])
            raise
        finally:
            children.stop()


def chain(args):
    output = separate_output(args)
    if args.epochs < 1:
        raise ValueError('正式训练轮数须为正数')
    output.mkdir(mode=0o700,parents=True,exist_ok=False)
    with operation_lock(output,'sam-med3d-chain',output):
        for stage,epochs in (('short',None),('train',args.epochs)):
            child = output/stage
            write_status(output,state='running',stage=stage,output=str(child))
            print(f'{stage}: {child}',flush=True)
            options = argparse.Namespace(**(vars(args) | dict(output=child,epochs=epochs,steps=2,
                                                               label=None,resume=False)))
            try:
                run(options)
            except (Exception,KeyboardInterrupt) as error:
                write_status(output,state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                             stage=stage,error=str(error))
                raise
    write_status(output,state='complete',stage='train',output=str(output/'train'))


def interrupt(_signum,_frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description='SAM-Med3D Turbo prompt-head training')
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('run','chain'):
        p = commands.add_parser(command)
        for name in ('prepared','weights','output'):
            p.add_argument('--'+name,type=Path,required=True)
        p.add_argument('--devices',default='all-visible')
        p.add_argument('--gpus',type=int,choices=(1,2),default=2)
        p.add_argument('--threads',type=int,default=4)
        p.add_argument('--wait-seconds',type=int,default=30,
                       help='资源不足时的复查间隔，0 表示立即退出')
        p.add_argument('--technical-subset',action='store_true')
        if command=='run':
            mode = p.add_mutually_exclusive_group()
            mode.add_argument('--steps',type=int,choices=range(1,9),default=2)
            mode.add_argument('--epochs',type=int)
            p.add_argument('--label',type=int,choices=range(1,11))
            p.add_argument('--resume',action='store_true')
        else:
            p.add_argument('--epochs',type=int,required=True)
    p = commands.add_parser('worker')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--device', required=True)
    args = parser.parse_args()
    if args.command == 'worker':
        validate_entry(sys.argv[1:], os.environ)
        train_rank(args.output, args.rank, args.device)
    else:
        signal.signal(signal.SIGTERM,interrupt)
        (chain if args.command=='chain' else run)(args)
