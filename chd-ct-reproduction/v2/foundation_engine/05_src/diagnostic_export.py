"""Local text-only diagnostic ZIP, independent of publication and model uploads."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import zipfile
from uuid import uuid4
from panel_publication.upload import _safe_path,_fingerprint,_blocked

TEXT={'.json','.jsonl','.log','.txt','.csv','.md'}
ROOT_FILES={'plan.json','state.json','launcher.log','resource-plan.json','software-smoke.json','resume-check.json'}
WORK_ROOTS=('diagnostics','00_admin/stage-decisions','00_admin/case-observations','00_admin/preparation_progress',
    '00_admin/preparation_logs','00_admin/preflight','09_reports','07_experiments','08_results/evaluation')
ADMIN_FILES=('cpu-case-budget.json','prediction-host-budget.json','assessment-progress.json','assessment.json')
SENSITIVE=re.compile(r'(?i)password|passwd|credential|secret|token|api[_-]?key|authorization|private[_-]?key')


def scrub(text):
    # Handles JSON, JSONL and free text without interpreting any logged command.
    def clean(value):
        if isinstance(value,dict):return {k:'[redacted]' if SENSITIVE.search(k) else clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,str):return redact_text(value)
        return value
    def redact_text(value):
        value=re.sub(r'(?s)-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----','[redacted private key]',value)
        value=re.sub(r'(?i)(https?://)[^/@\s]+:[^/@\s]+@',r'\1[redacted]@',value)
        value=re.sub(r"(?i)((?:password|passwd|secret|token|api[_-]?key|authorization)\s*[\"']?\s*[:=]\s*[\"']?)(?:bearer\s+)?[^\s,;\"']+",r'\1[redacted]',value)
        value=re.sub(r'\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{12,})\b','[redacted]',value)
        return value
    try:return json.dumps(clean(json.loads(text)),ensure_ascii=False,indent=2)+'\n'
    except (ValueError,TypeError):return '\n'.join(scrub(line).rstrip('\n') if line.lstrip().startswith('{') and _json_line(line) else redact_text(line) for line in redact_text(text).splitlines())+'\n'


def _json_line(line):
    try:json.loads(line);return True
    except ValueError:return False


def sources(job):
    job=_safe_path(job)
    plan=json.loads((job/'plan.json').read_text(encoding='utf-8'))
    work=_safe_path(plan['workspace'])
    if not work.is_relative_to(job):raise ValueError('诊断工作区超出任务目录')
    found=set()
    for name in ROOT_FILES:
        path=_safe_path(job/name)
        if path.is_file():found.add(path)
    for name in ADMIN_FILES:
        path=_safe_path(work/'00_admin'/name)
        if path.is_file():found.add(path)
    for root in [job/'diagnostics',job/'logs',job/'receipts',*(work/p for p in WORK_ROOTS)]:
        root=_safe_path(root)
        if not root.exists():continue
        for current,dirs,files in os.walk(root,followlinks=False):
            # Inspect links before pruning: no indirect reads through junctions.
            for name in dirs+files:_safe_path(Path(current)/name)
            dirs[:]=[d for d in dirs if not _blocked(d) and d not in {'intermediate','staging','nnUNet_raw','nnUNet_preprocessed'}]
            for name in files:
                path=Path(current)/name
                if not _blocked(name) and path.suffix.lower() in TEXT and path.is_file():found.add(path)
    return job,sorted(found)


def export(job,destination):
    job,paths=sources(job);destination=_safe_path(destination)
    if destination.is_relative_to(job):raise ValueError('诊断输出目录须在所选任务之外')
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/f'diagnostics-{time.strftime("%Y%m%d-%H%M%S")}-{uuid4().hex[:8]}.zip'
    temporary=target.with_suffix('.zip.tmp');manifest=[]
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                before=_fingerprint(path)
                if before[0]>64*2**20:raise ValueError('单份诊断超过64MiB，请单独核查：'+str(path))
                raw=path.read_text(encoding='utf-8',errors='replace')
                if _fingerprint(path)!=before:raise ValueError('诊断日志写入中，请稍后重试：'+str(path))
                data=scrub(raw).encode('utf-8');name=path.relative_to(job).as_posix()
                archive.writestr(name,data)
                manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
            archive.writestr('diagnostic-manifest.json',json.dumps(dict(schema_version=1,files=manifest,
                scope='local text diagnostics; sanitized; excludes images, weights, environments, credentials and caches',
                snapshot='individual stable files; a live task is not a transactionally frozen snapshot'),ensure_ascii=False,indent=2))
        os.replace(temporary,target)
    except BaseException:temporary.unlink(missing_ok=True);raise
    return target
