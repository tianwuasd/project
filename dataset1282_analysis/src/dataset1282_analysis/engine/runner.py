"""Local/server runner with frozen worker code, bounded admission and case-level commits."""
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import psutil
from ..contracts import atomic_json, digest, read_json, sha256_file
from ..inputs import load_cases,inventory
from ..metadata import read_metadata,link_metadata,summarize_metadata
from ..resources import snapshot,budget,admit_case,estimate_case,thread_environment
from ..observability.logging import event
from ..observability.status import operation_lock,write_status,register_workers
from .recovery import reusable_result,commit_result


def code_identity(root):
    root=Path(root)
    files=sorted(list((root/'src').rglob('*.py'))+[root/'configs/default.json',root/'requirements-lock.txt'])
    return {p.relative_to(root).as_posix():sha256_file(p) for p in files}


def stop_owned(item):
    process=item['process']
    sample_owned(item)
    owned=item.get('owned',{})
    # Windows venv launchers can host the actual interpreter in a child process.
    for pid,born in reversed(list(owned.items())):
        if pid==process.pid: continue
        try:
            child=psutil.Process(pid)
            if child.create_time()!=born: continue
            child.terminate()
            try: child.wait(timeout=5)
            except psutil.TimeoutExpired: child.kill();child.wait(timeout=5)
        except psutil.NoSuchProcess: pass
    if process.poll() is not None: return
    try:
        if psutil.Process(process.pid).create_time()!=item['born']: return
        process.terminate()
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: process.kill();process.wait(timeout=10)
    except psutil.NoSuchProcess: pass


def sample_owned(item):
    try:
        root=psutil.Process(item['process'].pid)
        if root.create_time()!=item['born']: return
        processes=[root]+root.children(recursive=True)
        rss=private=0
        for child in processes:
            try:
                born=child.create_time()
                memory=child.memory_info()
                unique=child.memory_full_info().uss
                if psutil.Process(child.pid).create_time()!=born: continue
                item.setdefault('owned',{})[child.pid]=born
                rss+=memory.rss
                private+=unique
            except (psutil.Error,AttributeError): pass
        item['private_resident']=private
        item['peak_rss']=max(item.get('peak_rss',0),rss)
    except psutil.Error: item['private_resident']=0


def prepare_run(config,root,run_dir,stage,case_ids,resume):
    cases=load_cases(config)
    if case_ids is not None:
        wanted=set(case_ids)
        if wanted-set(c.case_id for c in cases): raise ValueError('Unknown requested case_id')
        cases=[c for c in cases if c.case_id in wanted]
    rows=read_metadata(config['metadata_excel'],config['metadata_sheet'])
    links=link_metadata(cases,rows)
    identities={'manifest':sha256_file(Path(config['data_root'])/'manifest.json'),
                'excel':sha256_file(config['metadata_excel'])}
    params={k:v for k,v in config.items() if k not in ('work_root','environment','max_workers','read_concurrency','resource_wait_seconds')}
    basis={'protocol':config['protocol'],'parameters':params,'sources':identities,'stage':stage,
           'case_ids':[c.case_id for c in cases]}
    if resume:
        stored=read_json(run_dir/'provenance.json')
        if stored['basis']!=basis: raise ValueError('Resume contract changed: inputs, selection, parameters or stage')
        frozen=run_dir/'code_snapshot'
        if code_identity(frozen)!=stored['code_files']: raise ValueError('Frozen run code was changed')
        provenance=stored
    else:
        run_dir.mkdir(parents=True,exist_ok=False)
        frozen=run_dir/'code_snapshot'
        hashes=code_identity(root)
        for relative in hashes:
            target=frozen/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(root)/relative,target)
        provenance={'basis':basis,'code_files':hashes,'code_source':str(root),
                    'contract':digest({'basis':basis,'code_files':hashes}),
                    'created_at':datetime.now(timezone.utc).isoformat(), 'python':sys.executable}
        atomic_json(run_dir/'provenance.json',provenance)
        atomic_json(run_dir/'resolved_config.json',config)
        atomic_json(run_dir/'inventory.json',inventory(cases))
        atomic_json(run_dir/'metadata_records.json',rows)
        atomic_json(run_dir/'metadata_links.json',links)
        atomic_json(run_dir/'metadata_summary.json',summarize_metadata(rows))
    for directory in ('cases','jobs','worker_logs'): (run_dir/directory).mkdir(exist_ok=True)
    return cases,rows,links,provenance,frozen


def run_analysis(config,code_root,home,run_id=None,stage='full',case_ids=None,resume=False):
    run_id=run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+stage
    if not re.fullmatch(r'[A-Za-z0-9_-]+',run_id): raise ValueError('Invalid run identifier')
    run_dir=Path(config['work_root'])/'runs'/run_id
    if resume:
        old=read_json(run_dir/'provenance.json')['basis']
        stage=old['stage'];case_ids=old['case_ids']
    with operation_lock(home,'analysis',run_dir) as owner:
        cases,rows,links,provenance,frozen=prepare_run(config,code_root,run_dir,stage,case_ids,resume)
        contract=provenance['contract'];started=time.monotonic()
        pending=[];complete=failed=reused=0;peak=0;active=[];adjustments=0
        link_map={row['case_id']:row for row in links['case_rows']}
        event(run_dir,'run_start',run_id=run_id,stage=stage,resume=resume,total=len(cases),contract=contract)
        if stage=='metadata':
            result={'run_id':run_id,'run_dir':str(run_dir),'total':len(cases),'complete':len(cases),'failed':0,'reused':0,
                    'state':'completed_metadata_only','seconds':time.monotonic()-started}
            atomic_json(run_dir/'summary.json',result);write_status(run_dir,**result);return result
        for case in cases:
            output=run_dir/'cases'/(case.case_id+'.json')
            if resume and reusable_result(output,contract): complete+=1;reused+=1;continue
            link=link_map[case.case_id]
            job={'case':case.to_dict(),'config':config,'contract':contract,'stage':stage,'output':str(output),
                 'owner':owner.copy(),'project_home':str(home),
                 'metadata_link_status':link['status'],
                 'metadata':rows[link['excel_index']] if link['excel_index'] is not None else None}
            job_path=run_dir/'jobs'/(case.case_id+'.json');atomic_json(job_path,job)
            pending.append({'case':case,'job':job_path,'output':output,
                            'reserved':estimate_case(case.image,case.label,config,stage),'threads':1,'private_resident':0})
        facts=snapshot();stage_budget=budget(facts,config)
        event(run_dir,'stage_budget',requested={k:config[k] for k in ('cpu_fraction','ram_fraction','reserve_gib','max_workers','read_concurrency')},
              applied=stage_budget,facts=facts)
        atomic_json(run_dir/'resource_plan.json',{'facts':facts,'budget':stage_budget,
                    'estimates':{p['case'].case_id:p['reserved'] for p in pending},'threads_per_worker':1})
        last_progress=time.monotonic();last_status=0;last_reason=None;stop=False
        old_handler=signal.getsignal(signal.SIGTERM)
        def stop_request(*_):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,stop_request)
        child_env=os.environ.copy();child_env.update(thread_environment())
        child_env.update(PYTHONPATH=str(frozen/'src'),PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='')
        try:
            while pending or active:
                if stop: raise KeyboardInterrupt('Stop requested; completed cases are retained')
                for item in list(active):
                    process=item['process'];code=process.poll()
                    if code is None:
                        sample_owned(item)
                        continue
                    active.remove(item);item['stream'].close();peak=max(peak,item['peak_rss'])
                    register_workers(home,owner,[a['process'].pid for a in active])
                    if code or not item['output'].exists():
                        commit_result(item['output'],{'case_id':item['case'].case_id,'source_key':item['case'].source_key,
                                      'collection':item['case'].collection,'status':'failed','contract':contract,'stage':stage,
                                      'errors':[{'stage':'worker','error':f'worker_exit_{code}'}], 'source_sha256':{},
                                      'input_paths':{k:getattr(item['case'],k) for k in ('image','label')}})
                    result=read_json(item['output'])
                    item['peak_rss']=max(item['peak_rss'],result.get('peak_process_rss_bytes',0))
                    peak=max(peak,item['peak_rss'])
                    complete+=result['status']=='complete';failed+=result['status']!='complete'
                    event(run_dir,'case_done',case_id=item['case'].case_id,status=result['status'],
                          seconds=result.get('seconds'),peak_rss_bytes=item['peak_rss'],exit_code=code)
                    last_progress=time.monotonic()
                fresh_facts=snapshot();fresh=budget(fresh_facts,config)
                reserve=fresh.get('reserve_bytes');available=fresh.get('available_memory_bytes')
                if active and (available is None or available < reserve):
                    adjustments+=1
                    event(run_dir,'memory_pressure',fresh=fresh,adjustment=adjustments)
                    for item in active:
                        stop_owned(item);item['stream'].close()
                        pending.insert(0,{k:v for k,v in item.items() if k not in ('process','stream','born','peak_rss','owned')})
                    active=[]
                    register_workers(home,owner,[])
                    stage_budget['workers']=max(1,stage_budget['workers']//2)
                    stage_budget['memory_bytes']=int(stage_budget['memory_bytes']*.75)
                    if adjustments>3: raise MemoryError('Memory pressure recovery limit reached')
                # A worker slot bounds both I/O and computation: no hidden prefetch queue.
                if pending:
                    decision=admit_case(stage_budget,fresh,active,pending[0]['reserved'])
                    if decision['allowed']:
                        item=pending.pop(0)
                        stream=(run_dir/'worker_logs'/(item['case'].case_id+'.log')).open('a',encoding='utf-8')
                        try:
                            process=subprocess.Popen([sys.executable,'-B','-m','dataset1282_analysis.engine.worker',str(item['job'])],
                                env=child_env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                start_new_session=os.name!='nt',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                        except BaseException: stream.close();raise
                        item.update(process=process,stream=stream,born=psutil.Process(process.pid).create_time(),peak_rss=0)
                        active.append(item)
                        register_workers(home,owner,[a['process'].pid for a in active])
                        event(run_dir,'case_admitted',case_id=item['case'].case_id,pid=process.pid,decision=decision,
                              stage_budget=stage_budget,fresh_budget=fresh,threads=1)
                        last_progress=time.monotonic();last_reason=None
                    elif decision['reason']!=last_reason:
                        event(run_dir,'admission_wait',decision=decision,fresh_budget=fresh);last_reason=decision['reason']
                    if not active and time.monotonic()-last_progress>config['resource_wait_seconds']:
                        raise MemoryError('No case admitted within bounded wait: '+decision['reason'])
                if time.monotonic()-last_status>=1:
                    write_status(run_dir,state='running',run_id=run_id,total=len(cases),complete=complete,failed=failed,reused=reused,
                                 pending=len(pending),active=[a['case'].case_id for a in active],seconds=time.monotonic()-started,
                                 fresh_budget=fresh,peak_worker_rss_bytes=max([peak]+[a['peak_rss'] for a in active]))
                    last_status=time.monotonic()
                if active or pending: time.sleep(.5)
        except BaseException as e:
            write_status(run_dir,state='interrupted',run_id=run_id,total=len(cases),complete=complete,failed=failed,reused=reused,error=str(e))
            event(run_dir,'run_interrupted',error=type(e).__name__+': '+str(e))
            raise
        finally:
            for item in active: stop_owned(item);item['stream'].close()
            register_workers(home,owner,[])
            signal.signal(signal.SIGTERM,old_handler)
        result={'run_id':run_id,'run_dir':str(run_dir),'total':len(cases),'complete':complete,'failed':failed,'reused':reused,
                'state':'completed_with_failures' if failed else 'completed', 'seconds':time.monotonic()-started,
                'peak_worker_rss_bytes':peak,'memory_adjustments':adjustments,'contract':contract}
        atomic_json(run_dir/'summary.json',result);write_status(run_dir,**result);event(run_dir,'run_done',**result)
        return result
