import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verify_medsam3_attention import load_attention


def objective(module,features,padding,index):
    x = features[index % 2:index % 2+1]
    if index == 0:
        return module.q_proj(x).square().mean()
    if index == 1:
        return module.v_proj(x).square().mean()
    return module(x,x,x,key_padding_mask=padding[index % 2:index % 2+1])[0].square().mean()


def worker(rank,args,port):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
    from chd_segmentation.acceleration.distributed import synchronize_gradients,all_states,distributed_backend
    from chd_segmentation.models.medsam3.distributed import rank_indices,rng_state,restore_rng,require_topology,validation_rounds
    from chd_segmentation.artifacts.store import write_json
    dist.init_process_group(distributed_backend('cpu'),init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2,
        timeout=timedelta(seconds=45))
    if args.prepared_attention:
        sys.path.insert(0,args.official_root)
        from lora_layers import MultiheadAttentionLoRA,LoRALinear
        saved = torch.load(args.prepared_attention,map_location='cpu',weights_only=True)
        assert saved['source'] == 'official_medical_lora_first_text_attention'
        module = MultiheadAttentionLoRA(1024,16,batch_first=True)
        for name in ('q_proj','k_proj','v_proj','out_proj'):
            setattr(module,name,LoRALinear(getattr(module,name),rank=16,alpha=32,dropout=0.))
        module.load_state_dict(saved['state'],strict=True)
        module.eval()
        features,padding = saved['features'],saved['padding']
        torch.set_num_threads(2)
    else:
        module,features,padding = load_attention(args.official_root,args.weights)
    random.seed(42+rank)
    np.random.seed(42+rank)
    torch.manual_seed(42+rank)
    original_rng = rng_state()
    states = all_states(original_rng)
    rng_path = Path(args.output)/f'rng.rank{rank}.pt'
    torch.save(states,rng_path)
    stored_states = torch.load(rng_path,map_location='cpu',weights_only=True)
    draws = (random.random(),float(np.random.random()),float(torch.rand(())))
    restore_rng(stored_states[rank])
    assert draws == (random.random(),float(np.random.random()),float(torch.rand(())))
    assert not torch.equal(states[0]['torch'],states[1]['torch'])
    actual_prompts = ['aorta','pulmonary artery']
    empty_rounds = list(validation_rounds(actual_prompts[:1],2,rank))
    assert len(empty_rounds) == 1 and empty_rounds[0] == ('aorta' if rank == 0 else None)
    checks = []
    for shares in ([1,1],[2,1],[2,2]):
        batch = sum(shares)
        module.zero_grad(set_to_none=True)
        for index in range(batch):
            (objective(module,features,padding,index)/batch).backward()
        expected = {name:p.grad.clone() if p.grad is not None else None for name,p in module.named_parameters()}
        module.zero_grad(set_to_none=True)
        for index in rank_indices(0,batch,shares,rank):
            (objective(module,features,padding,index)/batch).backward()
        synchronize_gradients(module)
        differences = []
        for name,p in module.named_parameters():
            if expected[name] is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad,expected[name],atol=2e-6,rtol=5e-4)
                differences.append(float((p.grad-expected[name]).abs().max()))
        assert list(rank_indices(1,batch,shares,0))+list(rank_indices(1,batch,shares,1)) == list(range(batch,2*batch))
        checks.append(dict(shares=shares,max_gradient_difference=max(differences)))
    shape = dict(world=2,batch=3,shares=[2,1])
    require_topology(shape,shape)
    gathered = all_states(dict(rank=rank,checks=checks,rng_restored=True))
    if rank == 0:
        write_json(Path(args.output)/'distributed.json',dict(state='passed',backend='gloo',world=2,
            model='official_medical_lora_first_text_attention',prompts=['aorta','pulmonary artery'],
            full_model_updates=0,ranks=gathered,cuda_nccl_tested=False,torch=torch.__version__,
            topology_change_rejected=True,validation_rounds_with_empty_rank=True))
    dist.destroy_process_group()


def launched_worker(args,handle,rank,world,coordinator,port):
    from chd_segmentation.acceleration.runtime import attach
    attach(handle)
    coordinator.register(rank)
    if args.fail_rank == rank:
        (Path(args.output)/'missing-checkpoint.pt').resolve(strict=True)
    coordinator.wait(rank,0,'startup_ready')
    from chd_segmentation.models.medsam3.distributed import claim_prediction_slice
    claimed = claim_prediction_slice(handle,1)
    with handle.condition:
        handle.state[f'quota_rank_{rank}'] = int(claimed)
    coordinator.wait(rank,0,'consumed')
    assert handle.state['medsam3_prediction_claimed'] == 1
    assert sum(handle.state[f'quota_rank_{r}'] for r in range(world)) == 1
    from chd_segmentation.models.medsam3.distributed import validation_rounds
    for _ in validation_rounds(['aorta'],world,rank):
        coordinator.wait(rank,0,'memory_boundary')
    worker(rank,args,port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official-root',required=True)
    parser.add_argument('--weights',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--prepared-attention')
    parser.add_argument('--reject-topology',action='store_true')
    parser.add_argument('--public-launcher',action='store_true')
    parser.add_argument('--fail-rank',type=int)
    parser.add_argument('--input-config')
    args = parser.parse_args()
    if args.reject_topology:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        from chd_segmentation.models.medsam3.distributed import require_topology
        require_topology(dict(world=2,batch=3,shares=[2,1]),dict(world=2,batch=3,shares=[1,2]))
        raise AssertionError('不相容拓扑未拒绝')
    root = Path(args.output).resolve(strict=True)
    os.environ.update(USE_LIBUV='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        TEMP=str(root/'temporary'),TMP=str(root/'temporary'),MPLCONFIGDIR=str(root/'matplotlib'))
    began = perf_counter()
    if args.input_config:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        from chd_segmentation.models.medsam3.finetune import settings,check,training_contract
        from chd_segmentation.artifacts.store import digest,file_digest,write_json
        config = settings(args.input_config)
        checked = check(config)
        assert checked['ready']
        contract = training_contract(config,checked)
        changed = training_contract(dict(config,precision='fp32' if config['precision']=='bf16' else 'bf16'),checked)
        assert digest(contract) != digest(changed)
        assert contract['implementation']['gradient_sync'] == file_digest(
            Path(__file__).resolve().parents[1]/'src/chd_segmentation/acceleration/distributed.py')
        write_json(root/'identity-check.json',dict(state='passed',precision_change_rejected=True,
            gradient_sync_digest=contract['implementation']['gradient_sync'],identity=digest(contract)))
    invalid = subprocess.run([sys.executable,'-B',__file__,'--official-root',args.official_root,
        '--weights',args.weights,'--output',args.output,'--reject-topology'],capture_output=True,text=True,encoding='utf-8')
    assert invalid.returncode != 0 and 'ValueError' in invalid.stderr and '不相容拓扑未拒绝' not in invalid.stderr
    if args.public_launcher:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
        from chd_segmentation.config import DEFAULTS
        from chd_segmentation.acceleration import ResourceSession,ResourceDemand,RuntimeRequest
        from chd_segmentation.models.medsam3.launcher import launch
        from chd_segmentation.artifacts.store import read_json,write_json
        from time import time
        resources = dict(DEFAULTS['resources'],device='cpu',cpu_processes=2,cpu_threads=4,
            memory_policy='host_fraction',ram_fraction=.8,memory_limit_bytes=16*2**30,max_wait=60)
        request = RuntimeRequest(str(root),'medsam3_gloo_validation',root/'control-resources',root,
            resources,ResourceDemand(fixed_bytes=4*2**30,cpu_slots=4,local_batches=[1,1]))
        with ResourceSession(request) as session:
            session.state['medsam3_prediction_claimed'] = 0
            launch(dict(python=sys.executable,resources=resources),dict(paths=sys.path),session,
                launched_worker,(args,session.handle),2,time()+60)
            value = read_json(root/'distributed.json')
            value.update(public_launcher=True,peak_group_rss_bytes=session.state['peak_task_rss'],
                global_prediction_quota_passed=True)
            write_json(root/'distributed-launcher.json',value)
    else:
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0))
            port = listener.getsockname()[1]
        mp.spawn(worker,args=(args,port),nprocs=2,join=True)
    print(json.dumps(dict(state='passed',seconds=perf_counter()-began,report=str(root/'distributed.json'))))


if __name__ == '__main__':
    main()
