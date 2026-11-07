import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import torch

from dataset1282_analysis.contracts import atomic_json
from dataset1282_analysis.xu_geometry import PreparedCache
from dataset1282_analysis.xu_sample_cache import CachedPrepared
from dataset1282_analysis.xu_training import _epoch
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description='真实预处理病例的 CPU 供数与输入缓存验收')
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=3)
    parser.add_argument('--side',type=int,choices=(64,128),default=64)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--budget-mib',type=int,default=4096)
    args = parser.parse_args()
    if args.cases < 1 or args.workers < 1 or args.budget_mib < 1:
        raise ValueError('病例数、worker 和内存必须为正数')
    out = args.output.resolve()
    if out == args.prepared.resolve() or args.prepared.resolve() in out.parents:
        raise ValueError('验收输出必须位于预处理目录之外')
    out.mkdir(parents=True,exist_ok=False)
    cache = PreparedCache(args.prepared,allow_technical_subset=True)
    reader = CachedPrepared(cache,out/'inputs')
    ids = list(cache.case_ids('train'))+list(cache.case_ids('development'))
    ids = ids[:args.cases]
    torch.set_num_threads(2)
    torch.manual_seed(2023)
    model = torch.nn.Conv3d(1,11,1)
    model.spec = SimpleNamespace(side=args.side,batch=1)
    shape = (args.side,)*3
    groups = {n:cache.supervision[n]['unknown_classes'] for n in ids}
    results, states = [], []
    for mode in ('serial','parallel-cold','serial-warm','parallel-warm','parallel-warm-repeat'):
        chosen = cache if mode.startswith('serial') else reader
        def load(name, branch):
            return chosen.read_image(name,shape),chosen.read_label(name,shape)
        load.source_voxels = lambda name:int(np.prod(cache.geometry(name).cache_shape))
        os.environ.update(XU_PREFETCH=str(int(not mode.startswith('serial'))),XU_PREFETCH_WORKERS=str(args.workers),
                          XU_PREFETCH_BYTES=str(args.budget_mib*1024**2))
        current = copy.deepcopy(model)
        optimizer = torch.optim.Adam(current.parameters(),lr=2e-4)
        stats = {}
        started = time.perf_counter()
        loss = _epoch(current,ids,groups,load,'crop64','cpu',torch.Generator().manual_seed(2023),optimizer,stats)
        state = copy.deepcopy(current.state_dict())
        if states:
            for name,value in state.items():
                torch.testing.assert_close(value,states[0][name],rtol=0,atol=0)
            if loss != results[0]['loss']:
                raise AssertionError('供数策略改变了实际损失')
        states.append(state)
        row = dict(mode=mode,seconds=time.perf_counter()-started,loss=loss,supply=stats,
                   cache_cumulative=dict(reader.arrays.stats))
        results.append(row)
        print(json.dumps(row),flush=True)
    report = dict(scope='真实病例的供数和数值验收，网络为 Conv3d 探针',cases=len(ids),side=args.side,
                  prepared_request=cache.request_digest,results=results,exact_weights_and_loss=True,
                  speedup=results[2]['seconds']/results[-1]['seconds'])
    atomic_json(out/'report.json',report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
