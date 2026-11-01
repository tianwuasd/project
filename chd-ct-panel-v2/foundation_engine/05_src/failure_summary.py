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
            if cause:details.append(dict(time=path.stat().st_mtime,cause=str(cause),
                stage=item.get('stage',item.get('classification','子阶段')),path=path,record=item))
        except (OSError,ValueError,TypeError,AttributeError):continue
    if details:
        outer=max(details,key=lambda row:row['time']);selected=outer
        context=outer['record'].get('context') or {}
        refs=context.get('rank_failures',[]) if isinstance(context,dict) else []
        invocation_ids={str(row['invocation_id']) for row in refs if isinstance(row,dict) and row.get('invocation_id')}
        # Relate rank causes to this enclosing failure, never to an unrelated
        # newer/older recovered attempt merely because it is a rank file.
        message=outer['cause'].replace('\\','/').casefold()
        linked=[]
        for row in details:
            if row is outer or not row['path'].parent.name.startswith('rank-'):continue
            invocation=str(row['record'].get('invocation_id') or row['path'].parent.parent.name)
            if invocation_ids:
                matches=invocation in invocation_ids
            else:
                matches=(outer['record'].get('error_type')=='RankGroupFailure' and
                    str(row['path'].parent.parent).replace('\\','/').casefold() in message)
            if matches:linked.append(row)
        if linked:selected=max(linked,key=lambda row:row['time'])
        text=scrub(f"{selected['stage']}: {selected['cause']}").strip()[:600]+'；详情 '+str(selected['path'])
        if selected is not outer:text+='；外层记录 '+str(outer['path'])
        return text+('；日志 '+str(log) if log else '')
    return scrub(str(error)).strip()[:400]+'；具体子阶段原因未记录'+('；日志 '+str(log) if log else '')
