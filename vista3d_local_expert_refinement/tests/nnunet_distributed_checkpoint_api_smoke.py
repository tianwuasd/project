"""Real CPU Gloo + actual pinned checkpoint overrides; not CUDA trainer acceptance."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'05_src'))
sys.path.insert(0,str(CODE/'tests'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-file');args=parser.parse_args()
    import importlib.metadata
    if importlib.metadata.version('nnunetv2')!='2.8.1':raise ValueError('Pinned2.8.1 required')
    from ddp_runtime import launch_ranks
    with tempfile.TemporaryDirectory(prefix='vista_rank_checkpoint_') as temp:
        try:
            rows=launch_ranks('nnunet_checkpoint_rank_fixture:run',dict(workspace=temp),
                             ['GPU-cpu-test-0','GPU-cpu-test-1'],Path(temp)/'ranks',backend='gloo',
                             timeout_seconds=180,init_timeout_seconds=90)
            fallback=launch_ranks('nnunet_checkpoint_rank_fixture:compile_fallback',dict(workspace=temp),
                             ['GPU-cpu-test-0','GPU-cpu-test-1'],Path(temp)/'compile-ranks',backend='gloo',
                             timeout_seconds=180,init_timeout_seconds=90)
        finally:
            # Preserve synthetic rank diagnostics, NEVER checkpoint/weights.
            if args.report_file:
                import shutil
                evidence=Path(args.report_file).with_suffix('.ranks')
                for path in Path(temp).rglob('*'):
                    if path.is_file() and path.suffix in ('.json','.log','.txt'):
                        target=evidence/path.relative_to(temp)
                        target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(path,target)
    result=dict(status='passed',nnunet='2.8.1',device='cpu',real_ct=False,
        full_size_gpu=False,nccl=False,pinned_constructor_bypassed_for_cpu_only=True,
        checks=[r['result'] for r in rows],compile_fallback=[r['result'] for r in fallback])
    if args.report_file:Path(args.report_file).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
