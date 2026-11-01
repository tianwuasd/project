"""Short CPU-only synthetic supply before/after; excludes spawn from steady mean.

This is diagnostic evidence, never a real-CT or CUDA throughput claim. Run once
with the old serial/thread1 supply shape and once with persistent workers and
configured CPU threads. No candidate ranking or production recommendation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'05_src'))
from batch_pool import EpochBatchIterator


class SyntheticCPUAugment:
    def __iter__(self): return self

    def __next__(self):
        import torch
        image=torch.rand((2,1,64,64,64))
        for _ in range(5):
            image=torch.nn.functional.avg_pool3d(image,3,stride=1,padding=1)
        return dict(data=image,target=[(image>.4).to(torch.int16)])


def measure(workers,threads,shared_bytes):
    import torch
    torch.set_num_threads(1)
    iterator=EpochBatchIterator(SyntheticCPUAugment(),workers=workers,worker_threads=threads,
        prefetch=4,phase_steps=6,max_shared_bytes=shared_bytes,timeout_seconds=120)
    epochs=[];digests=[]
    try:
        for epoch in range(2):
            reset=time.perf_counter();iterator.set_epoch(epoch);reset=time.perf_counter()-reset
            waits=[]
            for _ in range(6):
                start=time.perf_counter();batch=next(iterator);waits.append(time.perf_counter()-start)
                digests.append(hashlib.sha256(batch['data'].numpy().tobytes()).hexdigest())
                # Synthetic consumer work lets supply progress between calls.
                time.sleep(.02)
            iterator.pause()
            epochs.append(dict(epoch=epoch,reset_seconds=reset,batch_wait_seconds=waits,
                               pids=iterator.observations()['worker_pids']))
        result=dict(workers=workers,worker_threads=threads,epochs=epochs,
            startup_first_batch_seconds=epochs[0]['batch_wait_seconds'][0],
            steady_epoch1_mean_wait_seconds=statistics.mean(epochs[1]['batch_wait_seconds']),
            observations=iterator.observations())
    finally:iterator.close()
    result['post_close_shared_bytes']=iterator.observations()['shared_allocated_bytes']
    result['post_close_pids']=iterator.observations()['worker_pids']
    return result,digests


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-file',required=True)
    args=parser.parse_args()
    import torch
    before,a=measure(0,1,0)
    after,b=measure(2,2,32*1024*1024)
    assert a==b,'scientific values changed across supply settings'
    assert after['epochs'][0]['pids']==after['epochs'][1]['pids']
    assert after['observations']['shared_reuses']>0
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',workload='synthetic CPU random volume + repeated avg_pool3d',
        pid=os.getpid(),torch=torch.__version__,real_ct=False,cuda_initialized=False,
        timing_scope='first-batch spawn/import warmup separate; steady mean uses epoch1 only',
        deterministic_exact_sha256=True,before=before,after=after,
        limitations=['synthetic CPU only; no real CT storage/augmentation throughput claim',
                     'not Linux, CUDA, NCCL, full-size patch or target-server validation'])
    Path(args.report_file).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
