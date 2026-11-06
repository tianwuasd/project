"""DDP safe points and full-state commits shared by the parallel engine/tests."""
import torch
import torch.distributed as dist
from .checkpoints import save_checkpoint,rng_state,restore_rng
from .collectives import all_objects,any_rank,synchronize_overflow,sum_detached
from .resources import transfer
from ..observability.runlog import write_json


def restore_rank_rng(payload,members,rank,device,seed,allow_partial=False):
    record=payload.get('distributed')
    if not record:raise ValueError('Missing per-rank RNG/membership checkpoint; use its original frozen entry')
    old=record['members'];boundary=payload['state']['phase']=='train' and payload['state']['step']==0
    if old!=members and not boundary and not allow_partial:
        raise ValueError('Membership may change only at a complete epoch boundary')
    if len(record['rng_by_rank'])!=len(old) or len(set(old))!=len(old):
        raise ValueError('Invalid rank RNG checkpoint inventory')
    if members==old:
        restore_rng(record['rng_by_rank'][rank],device)
    else:
        # Deliberate deterministic reseeding at an elastic boundary. This is
        # NOT a claim of bitwise equivalence across member/collective changes.
        import random,numpy as np
        if allow_partial:
            # Sampling/augmentation already uses global slot+cursor seeds.
            # A changed group has no one-to-one rank RNG mapping: derive a
            # deterministic model RNG from the saved cursor and new members.
            # Different collective orders need not be bitwise identical.
            import hashlib,json
            cursor=payload['state']
            body=[seed,cursor['epoch'],cursor['phase'],cursor['step'],members,rank]
            value=int.from_bytes(hashlib.sha256(json.dumps(body).encode()).digest()[:4],'little')
        else:value=seed+100003*payload['state']['epoch']+rank
        random.seed(value);np.random.seed(value);torch.manual_seed(value)


def commit(folder,model,optimizer,scaler,state,signature,device,members):
    rank=dist.get_rank()
    rng=all_objects(rng_state(device))
    pointer=None
    if rank==0:
        core=model.module if hasattr(model,'module') else model
        pointer=save_checkpoint(folder,core,optimizer,scaler,state,signature,device,
                                distributed=dict(members=members,rng_by_rank=rng))
        if state['phase']=='train' and state['step']==0:
            # latest may later point at a manual mid-epoch pause. Retain a
            # separate complete boundary for memory-driven group rollback.
            write_json(folder/'epoch.json',pointer)
    values=[pointer];dist.broadcast_object_list(values,src=0)
    return values[0]


def train_attempt(model,optimizer,scaler,criterion,batch,settings,device):
    data,targets,available=transfer(batch,device)
    model.train();optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type,enabled=settings['amp'] and device.type=='cuda'):
        loss=criterion(model(data),targets,available)
    if any_rank(not bool(torch.isfinite(loss)),device):raise ValueError('Non-finite global training loss')
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    overflow=synchronize_overflow(optimizer,scaler,device)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip'])
    # A finite gradient collection may still overflow its norm calculation.
    # Stop before stepping, rather than turning all gradients silently to 0.
    if not overflow and any_rank(not bool(torch.isfinite(norm)),device):
        raise ValueError('Non-finite global gradient norm with finite gradients')
    scaler.step(optimizer);scaler.update()
    return dict(loss=float(loss.detach()),overflow=overflow,update_applied=not overflow,
                scaler_scale=scaler.get_scale(),gradient_norm=float(norm) if not overflow else None)


@torch.no_grad()
def validate_attempt(model,criterion,batch,settings,device):
    # No buffers exist in STU's InstanceNorm. Using the core avoids DDP forward
    # reducer bookkeeping during no-grad validation; loss still sums globally.
    core=model.module if hasattr(model,'module') else model
    data,targets,available=transfer(batch,device);core.eval()
    with torch.autocast(device.type,enabled=settings['amp'] and device.type=='cuda'):
        outputs=core(data);loss=criterion(outputs,targets,available)
    if any_rank(not bool(torch.isfinite(loss)),device):raise ValueError('Non-finite global validation loss')
    truth=targets[0][:,0];pred=outputs[0].argmax(1);valid=(truth>=0)&(truth<8)
    rows=[]
    for label in range(1,8):
        mask=valid&available[:,label-1].view(-1,*([1]*(truth.ndim-1)))
        p=(pred==label)&mask;t=(truth==label)&mask
        rows.append(torch.stack(((p&t).sum(),(p&~t).sum(),(~p&t).sum())))
    counts=sum_detached(torch.stack(rows)).cpu().tolist()
    return dict(loss=float(loss),**{key:[row[i] for row in counts] for i,key in enumerate(('tp','fp','fn'))})
