"""One resident model per GPU; at most one next-case CPU preparation slot."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile
from server_data import read_json,write_json,digest
from gpu_queue import stop_owned_process


def run(records,models,sha,settings,work):
    from task_resources import prediction_budget
    from evaluation_gpu import idle_gpus
    from evaluation_contract import cached
    work=Path(work);records=[dict(r) for r in records]
    pending=[r for r in records if r['status']!='completed'];active={}
    root=work/'08_results/evaluation/queue_logs'/str(time.time_ns());root.mkdir(parents=True,exist_ok=True)
    status=work/'08_results/evaluation/status.json';state=dict(phase='gpu',status='running',jobs=records,frozen_sha256=sha,settings=settings,started=time.time())
    from read_admission import create_gate,update_gate
    gate=create_gate(settings['max_gpus'])
    stopping=[False];previous=signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM,lambda *_:stopping.__setitem__(0,True))
    try:
        while pending or any(row['assigned'] for row in active.values()):
            if stopping[0]:raise InterruptedError('预测队列收到停止请求')
            for gpu,row in active.items():
                rc=row['process'].poll()
                prep_rc=row['preparation'].poll()
                if prep_rc is not None:raise RuntimeError(f'CPU preparation exited {prep_rc}; log {row["log"].parent / "preparation.log"}')
                failure=row['directory']/'failure.json'
                if rc is not None or failure.exists():
                    cause=read_json(failure).get('error') if failure.exists() else f'worker exited {rc}; cause unavailable'
                    raise RuntimeError(f'GPU预测失败：{cause}；日志 {row["log"]}')
                done=row['directory']/'response.json'
                if row['assigned'] and done.exists() and read_json(done).get('id')==row['sequence']:
                    for record in row['assigned']:
                        if not cached(work,models[record['index']],record['case_id'],sha,'gpu'):
                            raise RuntimeError('GPU病例未提交：'+record['case_id']+'；日志 '+str(row['log']))
                        record['status']='completed'
                        from evaluation_contract import folder
                        observation=folder(work,models[record['index']])/'intermediate'/f"{record['case_id']}.json"
                        if observation.exists():
                            facts=read_json(observation);record['seconds']=facts.get('seconds',0)+facts.get('prepare_seconds',0)
                    row['assigned']=[]
            if pending:
                budget=prediction_budget(work,pending+[r for row in active.values() for r in row['assigned']],settings['max_gpus'])
                state['host_budget']=budget
                update_gate(gate,budget.get('read_concurrency',budget['workers']))
                if not active and budget['workers']<1:raise MemoryError('predict capacity: '+budget['capacity_status'])
                if os.environ.get('VISTA_GPU_LEASES'):
                    from cooperative_control import devices
                    candidates=idle_gpus(settings,candidates=[g for g in devices(False) if g not in active])
                else:candidates=idle_gpus(settings)
                for gpu in candidates:
                    if gpu in active:continue
                    idle_workers=sum(not row['assigned'] for row in active.values())
                    if len(active)>=min(budget['workers'],settings['max_gpus']) or idle_workers>=len(pending):continue
                    logs=root/f'worker{len(active)}';logs.mkdir()
                    directory=Path(tempfile.mkdtemp(prefix='vista-predict-'))
                    log=logs/'worker.log';stream=log.open('w',encoding='utf-8')
                    env=os.environ.copy();env.update(WHOLE_HEART_WORKSPACE=str(work),CUDA_VISIBLE_DEVICES=gpu,VISTA_READ_GATE=str(gate),VISTA_PREDICTION_LOG=str(log))
                    env.pop('WHOLE_HEART_GPU_UUIDS',None)
                    # Main GPU process and next-case process run concurrently.
                    threads=max(1,budget['threads_per_worker']);env['WHOLE_HEART_CPU_THREADS']='1'
                    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]='1'
                    extra={}
                    if env.get('WHOLE_HEART_MANAGED')=='1':
                        from task_worker import child_options
                        extra=child_options()
                    preparation=None;process=None
                    prep_stream=(logs/'preparation.log').open('w',encoding='utf-8')
                    try:
                        prep_env=dict(env,CUDA_VISIBLE_DEVICES='')
                        preparation=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),str(directory),sha,'--prepare'],
                            env=prep_env,stdin=subprocess.DEVNULL,stdout=prep_stream,stderr=subprocess.STDOUT,start_new_session=True,**extra)
                        from task_worker import register_child
                        register_child(preparation,'predict_preparation')
                        process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),str(directory),sha],
                            env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,**extra)
                        register_child(process,'predict_persistent')
                        from task_runtime import identity
                        write_json(directory/'owned-processes.json',dict(
                            gpu=dict(pid=process.pid,identity=identity(process.pid)),
                            preparation=dict(pid=preparation.pid,identity=identity(preparation.pid))))
                    except BaseException:
                        if process:stop_owned_process(process)
                        if preparation:stop_owned_process(preparation)
                        stream.close();prep_stream.close();raise
                    active[gpu]=dict(process=process,preparation=preparation,prep_stream=prep_stream,stream=stream,
                        log=log,directory=directory,assigned=[],sequence=0,threads=threads)
                # No ongoing model is evicted just because another process is busy.
                # Only idle workers receive a fresh natural-dispatch allocation.
                busy_slots=sum(row['threads'] for row in active.values() if row['assigned'])
                for gpu,row in active.items():
                    if row['assigned'] or not pending:continue
                    if sum(bool(r['assigned']) for r in active.values())>=budget['workers']:break
                    available=max(0,int(budget.get('cpu_budget',budget['workers']))-busy_slots)
                    threads=min(max(1,budget['threads_per_worker']),available)
                    if threads<1:continue
                    row['threads']=threads;busy_slots+=threads
                    record=pending.pop(0);batch=[record]
                    # Sequential disk IPC, one next case, same fold/model only.
                    waiting_others=sum(not other['assigned'] for other in active.values())-1
                    if threads>=2 and budget.get('prefetch',1) and len(pending)>waiting_others and pending[0]['index']==record['index']:
                        batch.append(pending.pop(0))
                    row['sequence']+=1;row['assigned']=batch
                    for r in batch:r.update(status='running',gpu_uuid=gpu,pid=row['process'].pid,log=str(row['log']))
                    write_json(row['directory']/'request.json',dict(id=row['sequence'],records=batch,threads=threads))
            if pending and active and not any(row['assigned'] for row in active.values()):
                raise MemoryError('predict dispatch capacity is zero; completed cases preserved')
            write_json(status,state)
            if pending or any(row['assigned'] for row in active.values()):time.sleep(.2)
        # Worker must acknowledge stop AND exit before caller releases keepers.
        for row in active.values():write_json(row['directory']/'stop.json',dict(stop=True))
        for row in active.values():
            row['process'].wait(timeout=60)
            if row['process'].returncode:raise RuntimeError('预测进程退出失败；日志 '+str(row['log']))
            row['preparation'].wait(timeout=60)
            if row['preparation'].returncode:raise RuntimeError('CPU准备进程退出失败')
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=str(error))
        # Completed intermediate files are validated/reused on restart.
        for r in records:
            if cached(work,models[r['index']],r['case_id'],sha,'gpu'):r['status']='completed'
            elif r['status']=='running':r['status']='interrupted'
        raise
    finally:
        try:stop_workers(active.values())
        finally:
            # Preserve the bounded control exchange in durable queue logs.
            import shutil
            for row in active.values():
                for name in ('owned-processes.json','request.json','response.json','prep-request.json','prep-response.json','failure.json'):
                    source=row['directory']/name
                    if source.exists():shutil.copy2(source,row['log'].parent/name)
            write_json(status,state);signal.signal(signal.SIGTERM,previous)


def worker(directory,sha):
    from prepare_nnunet import configure_paths
    configure_paths()
    import torch
    from evaluation import make_predictor
    from evaluation_phases import preprocess_case,infer
    from runtime_observations import failure,event
    directory=Path(directory);work=Path(os.environ['WHOLE_HEART_WORKSPACE'])
    frozen=work/'08_results/evaluation/frozen_models.json'
    if digest(frozen)!=sha:raise ValueError('冻结清单发生变化')
    models=read_json(frozen)['models'];model_index=None;predictor=None;sequence=0
    torch.set_num_threads(1)
    # Preparation is a direct child of the queue supervisor, never this GPU worker.
    try:
        while not (directory/'stop.json').exists():
            request=directory/'request.json'
            if not request.exists() or read_json(request)['id']==sequence:time.sleep(.1);continue
            command=read_json(request);sequence=command['id'];records=command['records'];threads=command['threads']
            index=records[0]['index'];model=models[index]
            if any(r['index']!=index or r['case_id'] not in model['targets'] for r in records):raise ValueError('预测请求不在冻结范围')
            if index!=model_index:
                if predictor is not None:del predictor;torch.cuda.empty_cache()
                if digest(model['checkpoint'])!=model['checkpoint_sha256']:raise ValueError('模型权重发生变化')
                predictor,_=make_predictor(model);model_index=index
                event(work,'predict.model_loaded',model=index,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'))
            prepared_id=0
            def submit(position):
                nonlocal prepared_id
                prepared_id+=1
                token=f'{sequence}:{prepared_id}'
                write_json(directory/'prep-request.json',dict(id=token,model=model,case_id=records[position]['case_id'],threads=max(1,threads-1)))
                return token
            def result(token):
                while True:
                    path=directory/'prep-response.json'
                    if path.exists():
                        value=read_json(path)
                        if value['id']==token:return value['prepared']
                    time.sleep(.05)
            token=submit(0)
            for position,record in enumerate(records):
                prepared=result(token)
                if threads>=2 and position+1<len(records):token=submit(position+1)
                infer(work,model,record['case_id'],sha,predictor=predictor,prepared=prepared)
                if threads<2 and position+1<len(records):token=submit(position+1)
            write_json(directory/'response.json',dict(id=sequence,status='completed'))
    except BaseException as error:
        write_json(directory/'failure.json',dict(error=str(error),type=type(error).__name__))
        failure(work,'persistent_prediction',error,log=os.environ.get('VISTA_PREDICTION_LOG',str(directory/'worker.log')));raise
    finally:
        if predictor is not None:del predictor;torch.cuda.empty_cache()


def stop_workers(rows):
    """Every preparation writer is a directly owned Popen, even after GPU death.

    Do not signal an exited leader's recycled process group. Each live direct
    child is checked by its retained process handle and separately stopped.
    """
    for row in rows:
        try:
            try:stop_owned_process(row['process'])
            finally:stop_owned_process(row['preparation'])
        finally:row['stream'].close();row['prep_stream'].close()


def preparation_worker(directory):
    from prepare_nnunet import configure_paths
    configure_paths()
    from evaluation_phases import preprocess_case
    directory=Path(directory);sequence=None
    while not (directory/'stop.json').exists():
        request=directory/'prep-request.json'
        if not request.exists() or read_json(request)['id']==sequence:time.sleep(.05);continue
        command=read_json(request);sequence=command['id']
        prepared=preprocess_case(Path(os.environ['WHOLE_HEART_WORKSPACE']),command['model'],command['case_id'],command['threads'])
        write_json(directory/'prep-response.json',dict(id=sequence,prepared=prepared))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');parser.add_argument('sha')
    parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:preparation_worker(args.directory)
    else:worker(args.directory,args.sha)
