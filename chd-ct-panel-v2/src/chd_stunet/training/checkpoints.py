"""Atomic full-state checkpoints containing only safe tensor/primitive payloads."""
import hashlib
import os
from pathlib import Path
import random
from uuid import uuid4
import numpy as np
import torch
from ..observability.runlog import write_json


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return value


def rng_state(device):
    state=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(kind=state[0],keys=state[1].tolist(),
        position=state[2],gauss=state[3],cached=state[4]),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(state,device):
    random.setstate(state['python'])
    n=state['numpy'];np.random.set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['gauss'],n['cached']))
    torch.set_rng_state(state['torch'])
    if device.type=='cuda' and state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'],device)


def save_checkpoint(folder,model,optimizer,scaler,state,signature,device,distributed=None):
    """Generation files + atomic pointer keep the previous complete commit valid.

    Replacing last.pt and its checksum separately has a crash window. Here the
    pointer moves only after the new immutable generation and checksum exist.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    generation=f"state-{state['attempts']:09d}-{state['epoch']:04d}-{state['phase']}-{state['step']:04d}-{uuid4().hex[:12]}.pt"
    target=folder/generation;temporary=target.with_suffix('.partial')
    payload=dict(schema=1,signature=signature,state=state,model=cpu_tree(model.state_dict()),
        optimizer=cpu_tree(optimizer.state_dict()),scaler=scaler.state_dict(),rng=rng_state(device))
    if distributed is not None:payload['distributed']=cpu_tree(distributed)
    with temporary.open('wb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,target)
    with target.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    pointer=dict(file=generation,sha256=sha,signature_identity=signature['identity'])
    write_json(folder/'latest.json',pointer)
    return pointer


def read_checkpoint(folder,signature,pointer_name='latest.json'):
    folder=Path(folder)
    import json
    if pointer_name not in ('latest.json','epoch.json'):raise ValueError('Unknown checkpoint pointer role')
    pointer=json.loads((folder/pointer_name).read_text(encoding='utf-8'))
    name=pointer['file']
    if Path(name).name!=name:raise ValueError('Checkpoint pointer must name a local generation')
    if pointer['signature_identity']!=signature['identity']:raise ValueError('Checkpoint scientific contract differs')
    target=folder/name
    with target.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if sha!=pointer['sha256']:raise ValueError('Checkpoint checksum differs')
    payload=torch.load(target,map_location='cpu',weights_only=True)
    if payload['signature']!=signature:raise ValueError('Checkpoint provenance differs')
    return payload


def restore_checkpoint(payload,model,optimizer,scaler,device):
    model.load_state_dict(payload['model'],strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scaler.load_state_dict(payload['scaler'])
    restore_rng(payload['rng'],device)
    return payload['state']
