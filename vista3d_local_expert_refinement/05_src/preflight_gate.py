"""Per model/actual GPU full-size gate; CPU-only helpers, no ML imports."""
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from server_data import digest,read_json,write_json

CODE=Path(__file__).resolve().parents[1]


def signature(work,fold,method,gpu,hardware):
    split=read_json(work/'04_data/manifests/splits_v1.json')['folds'][fold]
    plan=work/f"04_data/derived/nnUNet_preprocessed/Dataset{split['dataset_id']:03d}_{fold}/WholeheartBaselinePlans.json"
    return dict(fold=fold,method=method,gpu=gpu,hardware=hardware,plan_sha256=digest(plan),
                splits_sha256=digest(work/'04_data/manifests/splits_v1.json'),
                environment={f:digest(work/'06_configs'/f) for f in ['pip_freeze_training.txt','conda_explicit_training.txt']},
                code={p.name:digest(p) for p in sorted((CODE/'05_src').glob('*.py'))},
                cpu_threads=os.environ.get('WHOLE_HEART_CPU_THREADS','1'),cpu_selection=os.environ.get('WHOLE_HEART_CPU_SELECTION','bounded_tune'))


def ensure(work,fold,method,gpu,hardware,run):
    work=Path(work)
    if not re.fullmatch(r'GPU-[A-Za-z0-9-]+',gpu):raise ValueError('预检必须绑定实际获准GPU UUID')
    expected=signature(work,fold,method,gpu,hardware)
    gate=work/f'00_admin/preflight/{fold}_{method}_{gpu}.json'
    if gate.exists() and read_json(gate).get('signature')==expected:
        record=read_json(gate)
        if digest(record['report'])==record['report_sha256']:
            for trial in read_json(record['report']).get('cpu_tuning',{}).get('candidates',[]):
                if digest(trial['report'])!=trial['sha256']:raise ValueError('CPU候选预检报告摘要改变')
            print('复用当前模型/计划/实际GPU的完整尺寸预检：'+str(gate),flush=True);return record.get('effective_cpu',{})
        raise ValueError('正式尺寸预检报告被改动，不复用')
    report=gate.with_name(gate.stem+'_'+str(time.time_ns())+'.report.json')
    report.parent.mkdir(parents=True,exist_ok=True)
    print('正式训练前进行实际patch/batch训练、验证和检查点恢复预检；日志：'+str(report.with_suffix('.log')),flush=True)
    state=dict(status='running',fold=fold,method=method,gpu=gpu,report=str(report),log=str(report.with_suffix('.log')))
    status=gate.with_suffix('.state.json');write_json(status,state)
    try:
        run(report)
        value=read_json(report)
        if value.get('status')!='passed' or not value.get('checkpoint_weights_equal') or not value.get('optimizer_state_restored') or not value.get('one_resumed_training_step_passed'):
            raise ValueError('正式尺寸预检未通过；未启动正式训练')
        state['status']='passed'
    except BaseException as error:
        state.update(status='failed',error=str(error));raise
    finally:write_json(status,state)
    effective=value.get('effective_cpu',dict(threads=int(expected.get('cpu_threads','1')),augmentation_workers=0))
    write_json(gate,dict(signature=expected,report=str(report),report_sha256=digest(report),status='passed',effective_cpu=effective))
    return effective


def before_training(work,fold,method,frozen_cpu=None):
    initial_budget=max(1,int(os.environ.get('WHOLE_HEART_CPU_THREADS','1')))
    if frozen_cpu:
        if not 1<=int(frozen_cpu['threads'])<=initial_budget:
            raise ValueError('当前CPU共享预算低于原训练冻结线程数；请恢复原配额后续跑，或选择独立重跑')
        os.environ['WHOLE_HEART_CPU_THREADS']=str(frozen_cpu['threads'])
    os.environ['WHOLE_HEART_CPU_SELECTION']='frozen_resume' if frozen_cpu else 'bounded_tune'
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],text=True,timeout=15)
    hardware=[row for row in csv.reader(raw.splitlines()) if row and row[0].strip()==gpu]
    if len(hardware)!=1:raise ValueError('无法核对正式训练实际GPU硬件；请从受管队列选择UUID')
    def run(report):
      budget=max(1,int(os.environ.get('WHOLE_HEART_CPU_THREADS','1')))
      candidates=[budget] if frozen_cpu else sorted({1,min(2,budget),budget});measurements=[]
      for threads in candidates:
        trial=report.with_name(report.stem+f'.threads{threads}.json')
        command=[sys.executable,'-B','-u',str(CODE/'05_src/smoke_train.py'),fold,method,'--steps','3','--report-file',str(trial)]
        env=os.environ.copy();env['WHOLE_HEART_CPU_THREADS']=str(threads)
        for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:env[key]=str(threads)
        print(f'完整尺寸预检CPU候选：{threads}线程；串行增强，3步训练+验证+恢复；日志 {report.with_suffix(".log")}',flush=True)
        with report.with_suffix('.log').open('a',encoding='utf-8') as stream:
            if os.environ.get('WHOLE_HEART_MANAGED')=='1':
                from task_worker import run_child
                run_child(command,env,cwd=CODE,output=stream,timeout=1800)
            else:subprocess.run(command,env=env,cwd=CODE,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1800)
        value=read_json(trial)
        if value.get('status')!='passed':raise ValueError('CPU候选实际预检未通过')
        measurements.append(dict(threads=threads,augmentation_workers=0,step_seconds=value['median_step_seconds'],report=str(trial),sha256=digest(trial)))
      selected=min(measurements,key=lambda item:item['step_seconds']);value=read_json(selected['report'])
      value.update(effective_cpu=dict(threads=selected['threads'],augmentation_workers=0),cpu_tuning=dict(candidates=measurements,
          selection='minimum measured steady step time after two warmup steps; bounded short sample, not global optimum',
          serial_loader_reason='preserve B0 serial augmentation and checkpoint global RNG semantics; augmentation worker search deliberately disabled'))
      write_json(report,value)
    effective=ensure(Path(work),fold,method,gpu,hardware[0],run)
    if effective:
        os.environ['WHOLE_HEART_CPU_THREADS']=str(effective['threads'])
        os.environ['nnUNet_n_proc_DA']=str(effective['augmentation_workers'])
    return effective
