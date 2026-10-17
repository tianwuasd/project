"""Incremental project entry point used by both the local panel and command line."""
import argparse
from pathlib import Path
import traceback
import uuid

from common import write_json
from patient_pipeline.project_model import ProjectOptions, now
from patient_pipeline.project_store import ProjectStore
from patient_pipeline.project_ingest import ingest
from patient_pipeline.project_identity import resolve_identity
from patient_pipeline.project_process import process_units
from patient_pipeline.project_reports import build_snapshot, publish_project
from patient_pipeline.runtime import Control, RunLock, StopRequested


def run_project(sources,output,options=None,control=None):
    options=options or ProjectOptions();control=control or Control();out=Path(output).resolve()
    sources=[str(s) for s in sources]
    for raw in sources:
        src=Path(raw).resolve()
        if src==out or src.is_relative_to(out) or out.is_relative_to(src):
            raise ValueError('输入与输出目录不能相互包含')
    if options.action!='report' and not sources and not (out/'project.json').exists() and not (out/'processing_manifest.json').exists():
        raise ValueError('新项目至少需要添加一种资料')
    # Validate before creating the lock in an unrelated populated directory.
    if out.exists() and any(out.iterdir()) and not any((out/p).exists() for p in ['project.json','processing_manifest.json']):
        raise ValueError('输出目录不是既有项目；请选择空目录或原结果目录')
    out.mkdir(parents=True,exist_ok=True)
    batch=now().replace(':','').replace('+','_')+'_'+uuid.uuid4().hex[:8]
    status={'batch':batch,'started':now(),'status':'running','options':options.as_dict(),'sources':sources}
    with RunLock(out), ProjectStore(out) as store:
        store.put('meta','current_batch',batch);store.put('batches',batch,status)
        write_json(out/'project_run_status.json',status)
        try:
            store.load_legacy(control=control)
            if options.action!='report':
                if sources:ingest(store,sources,control)
                resolve_identity(store,options,control)
                process_units(store,options,control)
            control.progress('reports',0,1,'更新分类、三类问题和数据一览',True)
            snapshot=build_snapshot(store);publish_project(store,snapshot)
            errors=any(i['severity'] in {'error','warning'} for i in snapshot['issues'])
            status.update(status='completed_with_issues' if errors else 'completed',completed=now())
            store.put('batches',batch,status);write_json(out/'project_run_status.json',status)
            control.progress('reports',1,1,'报告已更新；缺少其他模态不会阻止处理',True)
            return 2 if errors else 0
        except StopRequested:
            status.update(status='interrupted',completed=now());store.put('batches',batch,status)
            write_json(out/'project_run_status.json',status)
            # Completed unit checkpoints remain durable. Report refresh can be requested independently.
            raise
        except BaseException as exc:
            status.update(status='failed',completed=now(),error=str(exc))
            store.put('batches',batch,status);write_json(out/'project_run_status.json',status)
            (out/'details/project_error.log').write_text(traceback.format_exc(),encoding='utf-8')
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',action='append',default=[],help='可多次填写，每项为本批次目录或文件')
    p.add_argument('--output',required=True)
    p.add_argument('--action',choices=['append','retry','report','audit'],default='append')
    p.add_argument('--match-names',action='store_true');p.add_argument('--recover-copies',action='store_true')
    p.add_argument('--split-phases',action='store_true');p.add_argument('--policy',default='')
    p.add_argument('--patient',action='append',default=[])
    a=p.parse_args(argv)
    opts=ProjectOptions(a.action,a.match_names,a.recover_copies,a.split_phases,tuple(a.patient),a.policy)
    try:return run_project(a.input,a.output,opts,Control(lambda e:print(e.get('message',''),flush=True) if e.get('stage') else None))
    except StopRequested:return 130


if __name__=='__main__':raise SystemExit(main())
