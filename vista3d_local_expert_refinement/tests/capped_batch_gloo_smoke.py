"""Pinned nnU-Net B0 unequal-rank gradients, no CUDA or real images."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'05_src'));sys.path.insert(0,str(ROOT/'tests'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-file')
    parser.add_argument('--global-batch',type=int,default=8)
    parser.add_argument('--world-sizes',default='4,6')
    parser.add_argument('--no-batch-dice',action='store_true');args=parser.parse_args()
    from ddp_runtime import launch_ranks
    results=[]
    with tempfile.TemporaryDirectory(prefix='capped_batch_gloo_') as temporary:
        for world in map(int,args.world_sizes.split(',')):
            ids=[f'GPU-CPU-{n}' for n in range(world)]
            records=launch_ranks('capped_batch_rank_fixture:compare',dict(global_batch=args.global_batch,batch_dice=not args.no_batch_dice),ids,Path(temporary)/str(world),
                backend='gloo',timeout_seconds=240,init_timeout_seconds=120)
            results.append([r['result'] for r in records])
    result=dict(status='passed',device='cpu',real_ct=False,nccl=False,groups=results)
    if args.report_file:Path(args.report_file).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
