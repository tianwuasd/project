"""Private, bounded runtime evidence; never reads credentials or patient voxels."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from server_data import write_json


def redact(value):
    text=str(value)
    text=re.sub(r'(?i)(https?://)[^/@\s]+:[^/@\s]+@',r'\1[redacted]@',text)
    text=re.sub(r'(?i)(authorization\s*[:=]\s*(?:bearer\s+)?|(?:token|password|secret|api_key)\s*[:=]\s*)[^\s,;]+',r'\1[redacted]',text)
    return text


def event(job, kind, **values):
    root=Path(job)/'diagnostics/events'
    root.mkdir(parents=True,exist_ok=True)
    record=dict(schema_version=1,time=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),kind=kind,**values)
    # Each process has its own append stream: no NFS multiwriter assumption.
    line=redact(json.dumps(record,ensure_ascii=False,default=str))
    with (root/f'{os.getpid()}.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(line+'\n')


def probe_command(name, command, *, timeout=10):
    started=time.monotonic()
    result=dict(name=name,command=list(command),status='failed')
    try:
        completed=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        result.update(status='passed' if completed.returncode==0 else 'failed',exit_code=completed.returncode,
                      stdout=redact(completed.stdout[-24000:]),stderr=redact(completed.stderr[-6000:]))
    except FileNotFoundError as error:
        result.update(status='unavailable',error=redact(error))
    except subprocess.TimeoutExpired as error:
        result.update(status='timeout',error=redact(error))
    except OSError as error:
        result.update(error=redact(error))
    result['seconds']=time.monotonic()-started
    return result


def collect(job, paths=None):
    """Current-host observation; historical server notes never count as probes."""
    from task_resources import probe
    commands=[('hostname',['hostname']),('cpu',['lscpu','--json']),
              ('numa',['numactl','--hardware']),
              ('gpu_topology',['nvidia-smi','topo','-m']),
              ('gpu_state',['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader,nounits']),
              ('gpu_processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])]
    observations=[probe_command(name,command) for name,command in commands]
    versions={}
    for package in ['torch','torchvision','nnunetv2','numpy','batchgenerators','batchgeneratorsv2']:
        try:versions[package]=dict(status='available',version=importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:versions[package]=dict(status='unavailable')
    disks={}
    for name,value in (paths or {}).items():
        path=Path(value)
        try:
            import shutil
            usage=shutil.disk_usage(path)
            disks[name]=dict(path=str(path),exists=path.exists(),readable=os.access(path,os.R_OK),
                             writable=os.access(path,os.W_OK),free_bytes=usage.free,total_bytes=usage.total)
        except OSError as error:disks[name]=dict(path=str(path),status='unavailable',error=redact(error))
    result=dict(schema_version=1,recorded_at=datetime.now(timezone.utc).isoformat(),
                actual_host=True,cpu_memory=probe(),commands=observations,packages=versions,paths=disks)
    missing=[dict(name=r['name'],status=r['status'],reason=r.get('error') or r.get('stderr',''),
                  command=r['command'],impact='该项不能作为资源选择或服务器验收依据')
             for r in observations if r['status']!='passed']
    write_json(Path(job)/'diagnostics/resource-snapshot.json',result)
    write_json(Path(job)/'diagnostics/missing-evidence.json',dict(items=missing))
    event(job,'resources.observed',missing=len(missing),snapshot='diagnostics/resource-snapshot.json')
    return result


def failure(job, stage, error, **context):
    record=dict(schema_version=1,recorded_at=datetime.now(timezone.utc).isoformat(),
                stage=stage,pid=os.getpid(),error_type=type(error).__name__,error=redact(error),
                traceback=redact(''.join(traceback.format_exception(type(error),error,error.__traceback__))[-40000:]),
                context=context)
    write_json(Path(job)/'diagnostics/failure.json',record)
    event(job,'stage.failed',stage=stage,error_type=record['error_type'],error=record['error'])
    return record
