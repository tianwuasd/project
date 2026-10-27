"""Structured causes for humans; logs are evidence, never executable instructions."""
from pathlib import Path
from server_data import read_json
from diagnostic_export import scrub


def summarize(work,error,log=None,since=0):
    candidates=list((Path(work)/'diagnostics').glob('failure.json'))
    candidates+=list((Path(work)/'07_experiments').rglob('failure*.json'))
    candidates+=list((Path(work)/'08_results/evaluation').rglob('failure.json'))
    details=[]
    for path in candidates:
        try:
            if path.stat().st_mtime<since:continue
            item=read_json(path)
            cause=item.get('message') or item.get('error')
            if cause:details.append((path.stat().st_mtime,str(cause),item.get('stage',item.get('classification','子阶段')),path))
        except (OSError,ValueError,TypeError,AttributeError):continue
    if details:
        _,cause,stage,path=max(details,key=lambda row:row[0])
        return scrub(f'{stage}: {cause}').strip()[:600]+'；详情 '+str(path)+('；日志 '+str(log) if log else '')
    return scrub(str(error)).strip()[:400]+'；具体子阶段原因未记录'+('；日志 '+str(log) if log else '')
