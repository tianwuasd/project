"""CPU-only actual AMP scaler and Gloo agreement; no CUDA/NCCL claim."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
os.environ['CUDA_VISIBLE_DEVICES']='-1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-file',required=True);args=parser.parse_args()
    from ddp_runtime import launch_ranks
    with tempfile.TemporaryDirectory(prefix='vista_amp_') as temp:
        try:
            rows=launch_ranks('test_amp_updates:amp_rank',dict(workspace=temp),['GPU-cpu-0','GPU-cpu-1'],
                Path(temp)/'ranks',backend='gloo',timeout_seconds=180,init_timeout_seconds=90)
        finally:
            import shutil
            for p in Path(temp).rglob('*'):
                if p.is_file() and p.suffix in ('.json','.log','.txt'):
                    target=Path(args.report_file).with_suffix('.ranks')/p.relative_to(temp)
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    result=dict(status='passed',backend='gloo',device='cpu',ranks=rows,cuda_compute=False)
    Path(args.report_file).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
