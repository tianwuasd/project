import argparse
import json
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'05_src'),str(Path(__file__).resolve().parent)]
from ddp_runtime import launch_ranks,RankGroupFailure

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    row=dict(main_threads=1,worker_threads=1,train_workers=1,val_workers=0,val_worker_threads=1,
        prefetch=2,val_prefetch=0,train_cpu_slots=2,val_cpu_slots=1,total_cpu_slots=2,
        resident_pool_bytes=128,base_memory_bytes=320,estimated_ram_bytes=448,
        cpu_share=4,memory_budget_bytes=2**30,train_worker_limit=3,worker_private_bytes=128,max_shared_bytes=256,val_max_shared_bytes=0)
    resource=dict(schema_version=2,status='ready',rank_count=2,rank_settings=[dict(row,rank=r) for r in range(2)],
        cpu_budget=8,memory_budget_bytes=2*2**30,source_shared_bytes=0,total_cpu_slots=4,estimated_ram_bytes=896,
        batch_bytes=64,limits=dict(train_workers=[3,3]))
    values=dict(output=str(out),resource=resource,gpu_uuids=['cpu0','cpu1'])
    success=launch_ranks('coordinator_rank_fixture:run',values,values['gpu_uuids'],out/'success',backend='gloo',timeout_seconds=120,init_timeout_seconds=30)
    try:
        launch_ranks('coordinator_rank_fixture:run',dict(values,fail=True),values['gpu_uuids'],out/'failure',backend='gloo',timeout_seconds=90,init_timeout_seconds=30)
        raise AssertionError('expected group failure')
    except RankGroupFailure as error:
        failure=dict(classification=error.classification,failures=error.failures)
    report=dict(status='passed',success=[v['result'] for v in success],failed_boundary=failure)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
