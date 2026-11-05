"""Read-only CUDA inventory in a disposable process, never a GPU keeper.

UUIDs come from CUDA's visible ordinal mapping, not guessed nvidia-smi indices.
The scanner exits before ranks are launched, releasing its CUDA contexts.
"""
import json
import os
import re
import subprocess
import sys
import torch
from ..acceleration.preprocess_cpu import GIB,interpreter_command
from .resources import parse_compute_processes


def scan():
    if not torch.cuda.is_available():return []
    query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
    inventory=subprocess.run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader,nounits'],
                             capture_output=True,text=True,timeout=20)
    if query.returncode or inventory.returncode:raise RuntimeError('Reliable GPU/PID query failed')
    physical={s.strip().lower() for s in inventory.stdout.splitlines()}
    rows=[]
    for ordinal in range(torch.cuda.device_count()):
        properties=torch.cuda.get_device_properties(ordinal)
        uuid=str(getattr(properties,'uuid',''))
        if re.fullmatch(r'[0-9a-fA-F-]{36}',uuid):uuid='GPU-'+uuid
        if uuid.lower() not in physical:raise ValueError('Visible CUDA UUID missing in physical inventory')
        external=parse_compute_processes(query.stdout,uuid,os.getpid())
        free,total=torch.cuda.mem_get_info(ordinal);reserve=max(4*GIB,int(total*.1))
        rows.append(dict(ordinal=ordinal,uuid=uuid,name=properties.name,free_bytes=free,total_bytes=total,
                         reserve_bytes=reserve,usable_bytes=free-reserve,external_pids=external))
    return rows


def query_visible(source,timeout=60):
    env=dict(os.environ,PYTHONPATH=str(source))
    result=subprocess.run([interpreter_command(env),'-u','-m','chd_stunet.training.fleet'],
                          env=env,capture_output=True,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError('GPU inventory refused: '+result.stderr[-3000:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def eligible(rows):
    return sorted((r for r in rows if not r['external_pids'] and r['usable_bytes']>0),
                  key=lambda r:(-r['usable_bytes'],r['ordinal']))


def query_live(mapping):
    """Refresh known visible UUIDs without importing/initializing another CUDA context."""
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,timeout=20)
    memory=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.free,memory.total','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,timeout=20)
    if apps.returncode or memory.returncode:raise RuntimeError('Live GPU ownership/memory evidence unavailable')
    inventory={}
    for line in memory.stdout.splitlines():
        if not line.strip():continue
        fields=[s.strip() for s in line.split(',')]
        if len(fields)!=3 or not all(s.isdigit() for s in fields[1:]):raise ValueError('Unparseable GPU memory evidence')
        inventory[fields[0].lower()]=(int(fields[1])*1024**2,int(fields[2])*1024**2)
    rows=[]
    for row in mapping:
        if row['uuid'].lower() not in inventory:raise ValueError('An authorized CUDA UUID disappeared')
        free,total=inventory[row['uuid'].lower()];reserve=max(4*GIB,int(total*.1))
        rows.append(dict(row,free_bytes=free,total_bytes=total,reserve_bytes=reserve,usable_bytes=free-reserve,
                         external_pids=parse_compute_processes(apps.stdout,row['uuid'],os.getpid())))
    return rows


if __name__=='__main__':print(json.dumps(scan()))
