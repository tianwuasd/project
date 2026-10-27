"""Independent real-data engineering workflow; never supplies formal weights."""
import argparse
import os
from pathlib import Path
import sys
import time


def run(plan):
    from assessment_prepare import prepare
    from cooperative_control import tune,identity,release_devices
    from server_data import read_json,write_json,digest
    from runtime_observations import collect,event,failure
    from task_worker import run_child
    work=Path(plan['workspace']);started=time.monotonic()
    report=dict(schema_version=2,status='running',job_kind='assessment',formal_training=False,
        started=time.time(),stages=[],models=[],formal_budget_unchanged=plan['formal_budget'],
        limits=plan['assessment_budget'])
    target=work/'09_reports/assessment.json'
    fleet=None
    previous_lease=os.environ.get('VISTA_GPU_LEASES')
    def stage(name,status,**values):
        report['stages'].append(dict(name=name,status=status,time=time.time(),**values))
        write_json(target,report);write_json(work/'00_admin/assessment-progress.json',report)
        print(f'完整测评：{name} / {status}',flush=True)
        event(work,'assessment.stage',stage=name,status=status,**values)
    try:
        collect(work,dict(data=plan['data'],workspace=str(work),runtime=plan['runtime']))
        stage('环境与实况采集','passed',software_smoke=str(Path(plan['workspace']).parent/'software-smoke.json'))
        for fold in plan['folds']:
            stage('数据与预处理','running',fold=fold)
            scope=prepare(work,fold,plan['gpu_memory'])
            stage('数据与预处理','passed',fold=fold,scope=scope)
            for method in plan['methods']:
                from gpu_fleet import GPUFleet
                fleet=GPUFleet(Path(plan['workspace']).parent,plan['runtime'],plan['gpu_memory']).start()
                os.environ['VISTA_GPU_LEASES']=str(fleet.path)
                fleet.devices(wait=True)
                stage('完整尺寸协同训练、容量验证与检查点恢复','running',fold=fold,model=method)
                chosen=tune(work,fold,method,'assessment')
                stage('完整尺寸协同训练、容量验证与检查点恢复','passed',chosen=chosen)
                provenance=identity(work,fold,method)  # Must precede final release: identity waits for keepers.
                case=scope['prediction_ids'][0]
                fleet.suspend_discovery(chosen['gpu_uuids'][:1])
                from task_resources import prediction_budget
                host=prediction_budget(work,[dict(fold=fold,case_id=case)],1)
                if host['workers']<1:raise MemoryError('assessment predict capacity: '+host['capacity_status'])
                env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=chosen['gpu_uuids'][0]
                from task_resources import thread_environment
                env.update(thread_environment(host['threads_per_worker'],env))
                logfile=work/f'09_reports/assessment_prediction_{fold}_{method}.log'
                with logfile.open('a',encoding='utf-8') as stream:
                    run_child([sys.executable,'-u','-B',str(Path(__file__).with_name('assessment_predict.py')),
                        fold,method,chosen['output'],case,'--phase','gpu'],env,output=stream,timeout=900)
                fleet.close();fleet=None
                stage('GPU预测进程退出与全部保留卡释放','passed',fold=fold,model=method)
                from task_resources import cpu_tail_budget
                budget=cpu_tail_budget(work,[dict(fold=fold,method=method,case_id=case)],assessment=True)
                if budget['workers']<1:raise MemoryError('assessment CPU tail capacity: '+budget['capacity_status'])
                env['CUDA_VISIBLE_DEVICES']='';env.pop('VISTA_GPU_LEASES',None)
                env.update(thread_environment(budget['threads_per_worker'],env))
                with logfile.open('a',encoding='utf-8') as stream:
                    run_child([sys.executable,'-u','-B',str(Path(__file__).with_name('assessment_predict.py')),
                        fold,method,chosen['output'],case,'--phase','cpu'],env,output=stream,timeout=900)
                prediction=read_json(work/f'09_reports/assessment_prediction_{fold}_{method}.json')
                if prediction.get('status')!='passed':raise ValueError('测评预测/指标未完成')
                report['models'].append(dict(fold=fold,method=method,chosen=chosen,
                    prediction=prediction,identity=provenance))
                stage('原网格预测、评价与写出','passed',fold=fold,model=method,case_id=case,log=str(logfile))
        if fleet:
            fleet.close();fleet=None
        stage('GPU保留进程退出与资源释放','passed')
        report.update(status='passed',completed=time.time(),seconds=time.monotonic()-started,
            acceptance_scope='this real-data source subset and measured GPU groups; not all possible hardware/configurations',
            missing_evidence='diagnostics/missing-evidence.json',logs='diagnostics/events;07_experiments/tuning/*/ranks')
        write_json(target,report)
        report.update(function_status='passed',effect_status='insufficient_long_window_evidence',
                      resource_recommendation=None,resource_receipt_reusable=False,
                      performance_sufficiency='short functional/capacity observation; formal stages independently probe current resources')
        write_json(target,report)
        print(f'完整测评完成。独立报告：{target}；训练权重不会用于正式任务。',flush=True)
        return report
    except BaseException as error:
        report.update(function_status='failed',effect_status='not_evaluated',status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
            error=str(error),seconds=time.monotonic()-started)
        failure(work,'assessment',error,plan=plan,rank_failures=getattr(error,'failures',[]))
        raise
    finally:
        try:
            if fleet:fleet.close()
        finally:
            if previous_lease is None:os.environ.pop('VISTA_GPU_LEASES',None)
            else:os.environ['VISTA_GPU_LEASES']=previous_lease
            write_json(target,report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('plan');args=parser.parse_args()
    from server_data import read_json
    run(read_json(args.plan))
