import argparse
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4


def run(args):
    entry = Path(args.entry).resolve(strict=True)
    sys.path.insert(0,str(entry))
    from common import clean_owned,read,sha,source_identity,supervise,write
    project = Path(__file__).resolve().parents[1]
    work,report = Path(args.work_root).resolve(),Path(args.report).resolve()
    if work==report or work.is_relative_to(report) or report.is_relative_to(work):
        raise ValueError('检查工作目录与报告目录需要独立')
    work.mkdir(parents=True,exist_ok=False)
    report.mkdir(parents=True,exist_ok=True)
    token = uuid4().hex
    write(work/'owner.json',dict(token=token))
    identity = source_identity(project)
    external = {name:sha(entry/name) for name in ('common.py','usage.py','validate.py')}
    record = dict(state='running',device='cpu',optimizer_updates=0,source=identity,
        supervisor_files=external,entry=str(entry),stages=[])
    started = perf_counter()
    try:
        for name,script,extra in (
                ('budget','verify_model_budget_real.py',[]),
                ('admission','verify_model_guards_real.py',['--mode','admission','--bundle',args.bundle])):
            stage = work/name
            stage.mkdir()
            command = [sys.executable,'-u','-B',str(project/'tests'/script),'--spec',args.spec,
                '--work-root',str(stage/'work'),'--report',str(report/name),*extra]
            result = supervise(command,project,stage,report/name)
            acceptance = read(report/name/'acceptance.json')
            if acceptance['state']!='passed' or acceptance['cleanup']['work_root_exists']:
                raise ValueError('生产检查未通过或未完成清理')
            record['stages'].append(dict(name=name,supervision=result,acceptance=acceptance))
        if identity!=source_identity(project) or external!={name:sha(entry/name) for name in external}:
            raise ValueError('检查期间项目或监督程序发生变化')
        record['state'] = 'passed'
    finally:
        record['seconds'] = perf_counter()-started
        if record['state']=='running':
            record['state'] = 'failed'
        write(report/'integration.json',record)
        record['cleanup'] = clean_owned(work,work.parent,token,report)
        write(report/'integration.json',record)
        print(record['state'],report/'integration.json',flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    for name in ('entry','spec','bundle','work-root','report'):
        parser.add_argument('--'+name,required=True)
    run(parser.parse_args())
