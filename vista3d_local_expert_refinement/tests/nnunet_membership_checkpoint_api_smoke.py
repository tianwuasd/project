"""Actual CPU Gloo/pinned checkpoint UUID transition; explicit linear API fixture."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'05_src'));sys.path.insert(0,str(CODE/'tests'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report-file');args=parser.parse_args()
    if importlib.metadata.version('nnunetv2')!='2.8.1':raise ValueError('Pinned2.8.1 required')
    from ddp_runtime import launch_ranks
    with tempfile.TemporaryDirectory(prefix='vista_membership_checkpoint_') as temp:
        root=Path(temp);saved_ids=['GPU-CPU-A','GPU-CPU-B'];resumed_ids=['GPU-CPU-B','GPU-CPU-C','GPU-CPU-A']
        saved=launch_ranks('nnunet_checkpoint_rank_fixture:membership',
            dict(workspace=temp,phase='save',gpu_uuids=saved_ids),saved_ids,root/'saved',backend='gloo',
            timeout_seconds=180,init_timeout_seconds=90)
        resumed=launch_ranks('nnunet_checkpoint_rank_fixture:membership',
            dict(workspace=temp,phase='resume',gpu_uuids=resumed_ids),resumed_ids,root/'resumed',backend='gloo',
            timeout_seconds=180,init_timeout_seconds=90)
    result=dict(status='passed',nnunet='2.8.1',device='cpu',membership='2 -> 3; old UUIDs reordered; new UUID stream',
        saved=[row['result'] for row in saved],resumed=[row['result'] for row in resumed],
        pinned_constructor_bypassed_for_cpu_api_fixture=True,network='tiny Linear CPU DDP',
        training_budget_executed=dict(train_updates=250,validation_data_batches=50),
        real_ct=False,full_size_gpu=False,nccl=False,
        limitation='Actual project/upstream checkpoint and epoch-iterator APIs, not the production full nnU-Net trainer lifecycle; validation data batches have no segmentation model metrics')
    if args.report_file:Path(args.report_file).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
