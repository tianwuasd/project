"""Visible-device admission and a real fixed-shape forward/backward probe."""
import os
import re
import shutil
import subprocess
import torch
from ..acceleration.preprocess_cpu import GIB
from .checkpoints import rng_state,restore_rng


class GPUUnavailable(RuntimeError):
    """Verified contention/free-memory race; eligible placement may be retried."""


def parse_compute_processes(text,uuid,own_pid):
    if re.fullmatch(r'[0-9a-fA-F-]{36}',uuid):uuid='GPU-'+uuid
    if not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',uuid):
        raise RuntimeError('Cannot verify a physical GPU UUID; MIG/unidentified devices need a separate adapter')
    external=[]
    for line in text.splitlines():
        if not line.strip():continue
        values=[v.strip() for v in line.split(',')]
        if (len(values)!=2 or not values[0].isdigit() or int(values[0])<1
                or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',values[1])):
            raise RuntimeError('Unparseable compute-process evidence; GPU admission refused')
        if values[1].lower()==uuid.lower() and int(values[0])!=own_pid:external.append(int(values[0]))
    return external


def admit_device(requested):
    device=torch.device(requested)
    if device.type=='cpu':return device,dict(device='cpu',gpu_requested=False)
    if device.type!='cuda' or not torch.cuda.is_available():raise ValueError('Requested CUDA device unavailable')
    index=device.index if device.index is not None else 0
    if index<0 or index>=torch.cuda.device_count():raise ValueError('Device is outside the visible CUDA scope')
    torch.cuda.set_device(index)
    properties=torch.cuda.get_device_properties(index)
    # nvidia-smi is read-only; physical UUIDs disambiguate CUDA_VISIBLE_DEVICES
    # remapping. WDDM may omit compute-process evidence; report that explicitly.
    external=[];evidence='nvidia-smi unavailable'
    if shutil.which('nvidia-smi'):
        command=['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits']
        result=subprocess.run(command,capture_output=True,text=True,timeout=15)
        uuid=str(getattr(properties,'uuid',''))
        if re.fullmatch(r'[0-9a-fA-F-]{36}',uuid):uuid='GPU-'+uuid
        if result.returncode==0 and uuid:
            inventory=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],
                capture_output=True,text=True,timeout=15)
            if inventory.returncode or uuid.lower() not in {line.strip().lower() for line in inventory.stdout.splitlines()}:
                raise RuntimeError('CUDA UUID does not match a physical GPU inventory entry')
            evidence='compute-apps by UUID'
            external=parse_compute_processes(result.stdout,uuid,os.getpid())
        else:evidence='compute PID/UUID evidence unavailable'
    if evidence!='compute-apps by UUID':
        raise RuntimeError('Missing reliable compute-process evidence; use CPU or repair GPU query support')
    if external:raise GPUUnavailable('Selected GPU has external compute processes; no process was changed: '+str(external))
    free,total=torch.cuda.mem_get_info(index)
    reserve=max(4*GIB,int(total*.1))
    usable=free-reserve
    if usable<=0:raise GPUUnavailable('GPU lacks the required 4GiB/10% reserve')
    return torch.device('cuda',index),dict(device=f'cuda:{index}',name=properties.name,
        uuid=uuid,free_bytes=free,total_bytes=total,reserve_bytes=reserve,
        usable_bytes=usable,external_compute_evidence=evidence)


def transfer(batch,device):
    def tensor(array):
        value=torch.from_numpy(array)
        if device.type=='cuda':value=value.pin_memory()
        return value.to(device,non_blocking=device.type=='cuda')
    return tensor(batch['data']),[tensor(t).long() for t in batch['target']],tensor(batch['available'])


def capacity_probe(model,loss,batch,settings,device,admission,optimizer=None):
    """No optimizer update or data consumption; restore RNG/buffers afterward."""
    saved_rng=rng_state(device)
    buffers={name:value.detach().clone() for name,value in model.named_buffers()}
    was_training=model.training;model.train()
    data=target=available=outputs=value=None;momentum=[]
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        data,target,available=transfer(batch,device)
        with torch.autocast(device.type,enabled=settings['amp'] and device.type=='cuda'):
            outputs=model(data);value=loss(outputs,target,available)
        if not bool(torch.isfinite(value)):raise ValueError('Non-finite fixed-shape probe loss')
        value.backward()
        # Simulate SGD momentum's first-step resident allocation too. This does
        # not initialize or advance the live optimizer/scaler.
        existing=optimizer.state if optimizer is not None else {}
        parameters=[p for p in model.parameters() if p.grad is not None]
        momentum=[torch.zeros_like(p) for p in parameters if 'momentum_buffer' not in existing.get(p,{})]
        existing_count=sum('momentum_buffer' in existing.get(p,{}) for p in parameters)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
            peak=torch.cuda.max_memory_reserved(device)
            free,_=torch.cuda.mem_get_info(device)
            if peak>admission['usable_bytes'] or free<admission['reserve_bytes']:
                raise MemoryError('Selected batch/patch exceeds GPU admission; no automatic batch/patch reduction')
        else:peak=0
        return dict(passed=True,shape=list(data.shape),peak_reserved_bytes=peak,
                    optimizer_updated=False,consumed_batches=0,
                    existing_momentum_buffers=existing_count,simulated_momentum_buffers=len(momentum))
    finally:
        model.zero_grad(set_to_none=True)
        for name,value in model.named_buffers():value.copy_(buffers[name])
        model.train(was_training);restore_rng(saved_rng,device)
        data=target=available=outputs=value=None;momentum=[]
        if device.type=='cuda':torch.cuda.empty_cache()
