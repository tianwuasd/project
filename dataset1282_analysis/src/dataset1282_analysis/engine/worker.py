"""One case per process; no input writes and no library state survives a completed case."""
import gc
import os
from pathlib import Path
import sys
import time
import traceback
import threading
from ..contracts import read_json
from ..observability.status import owner_alive
from .recovery import commit_result


def analyze_case(job):
    from ..geometry import inspect_geometry
    from ..voxels import analyze_image
    from ..labels import analyze_label
    started=time.monotonic();case=job['case'];policy=job['config'];errors=[]
    result={'case_id':case['case_id'],'source_key':case['source_key'],'collection':case['collection'],
            'contract':job['contract'],'stage':job['stage'], 'metadata_link_status':job['metadata_link_status'],
            'input_paths':{k:case[k] for k in ('image','label')},'source_sha256':{},
            'source_expected_hash_match':{},'errors':errors,'status':'complete'}
    before={}
    for kind in ('image','label'):
        try:
            stat=Path(case[kind]).stat(); before[kind]=(stat.st_size,stat.st_mtime_ns)
        except OSError as e: errors.append({'stage':kind,'error':str(e)})
    try: result['geometry']=inspect_geometry(case['image'],case['label'],job.get('metadata'))
    except Exception as e: errors.append({'stage':'geometry','error':f'{type(e).__name__}: {e}'})
    if job['stage']=='full':
        for kind,function in [('image',analyze_image),('label',analyze_label)]:
            try:
                result[kind]=function(case[kind],policy)
                actual=result[kind]['sha256']; result['source_sha256'][kind]=actual
                expected=case['expected_hashes'].get(kind)
                result['source_expected_hash_match'][kind]=actual==expected if expected else None
            except Exception as e:
                errors.append({'stage':kind,'error':f'{type(e).__name__}: {e}'})
                traceback.print_exc()
            gc.collect()
    for kind,signature in before.items():
        stat=Path(case[kind]).stat()
        if (stat.st_size,stat.st_mtime_ns)!=signature: errors.append({'stage':kind,'error':'source_changed_during_analysis'})
    result['source_stat_before']={k:list(v) for k,v in before.items()}
    result['readability_status']='failed' if errors else ('full_voxels_read' if job['stage']=='full' else 'headers_only')
    result['status']='failed' if errors else 'complete'
    result['seconds']=time.monotonic()-started
    if os.name=='nt':
        import psutil
        result['peak_process_rss_bytes']=psutil.Process().memory_info().peak_wset
    else:
        import resource
        result['peak_process_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if sys.platform=='darwin' else 1024)
    return result


def start_owner_watchdog(home,owner):
    """Hard supervisor exits cannot leave a worker writing into a resumed run."""
    if not owner_alive(home,owner): raise SystemExit(143)
    def watch():
        while True:
            if not owner_alive(home,owner): os._exit(143)
            time.sleep(.2)
    thread=threading.Thread(target=watch,name='owner-watchdog',daemon=True)
    thread.start()
    return thread


def main():
    job=read_json(sys.argv[1])
    start_owner_watchdog(job['project_home'],job['owner'])
    result=analyze_case(job)
    if not owner_alive(job['project_home'],job['owner']): raise SystemExit(143)
    commit_result(job['output'],result)
    return 0


if __name__=='__main__': raise SystemExit(main())
