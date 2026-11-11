import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

from dataset1282_analysis.xu_gpu_isolation import validate_entry

validate_entry(sys.argv[1:],os.environ,default='cpu')

import torch

from dataset1282_analysis.contracts import atomic_json
from dataset1282_analysis.xu_blood import BloodForward, _objective
from dataset1282_analysis.xu_blood_models import BloodSequence, SliceUNet


def benchmark(args):
    from dataset1282_analysis.xu_resources import apply_env_limit
    device = torch.device(args.device)
    apply_env_limit(device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(2023)
    encoder = SliceUNet(args.profile).to(device).eval()
    sequence = BloodSequence(encoder.width).to(device)
    original = copy.deepcopy(sequence.state_dict())
    side = 512 if args.profile == 'paper' else 64
    planes = torch.randn(7+args.steps,1,side,side,device=device)
    labels = torch.randint(0,3,(1,side,side),device=device)
    records = []
    for phase in ('train','development'):
        for mode in ('legacy','auto','auto','legacy'):
            os.environ['XU_BLOOD_EXECUTION'] = mode
            sequence.load_state_dict(original)
            sequence.train(phase == 'train')
            optimizer = torch.optim.Adam(sequence.parameters(),lr=2e-4)
            forward = BloodForward(encoder,device,True,1)
            times, losses = [], []
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for step in range(args.steps+1):
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                images = planes[step:step+7].transpose(0,1)
                keys = [('benchmark',z) for z in range(step,step+7)] if phase == 'development' else None
                with torch.set_grad_enabled(phase == 'train'):
                    scores = forward(images,sequence,keys)
                    loss = _objective(scores,labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('基准测试出现非有限损失')
                    if phase == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter()-started
                if step:
                    times.append(elapsed)
                    losses.append(float(loss.detach()))
            records.append(dict(phase=phase,mode=mode,seconds=times,median_seconds=statistics.median(times),
                                losses=losses,execution=forward.execution,
                                encoded_frames=forward.distribution.stats['cases_by_device'][str(device)],
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None))
            forward.clear_frames()
            del optimizer, forward, scores, loss
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    summary = dict(device=str(device),name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                   torch=str(torch.__version__),profile=args.profile,side=side,steps=args.steps,
                   scope='相同真实网络与优化器的合成输入性能对照，不评估模型准确率',records=records)
    for phase in ('train','development'):
        values = {mode:statistics.median([r['median_seconds'] for r in records if r['phase']==phase and r['mode']==mode])
                  for mode in ('legacy','auto')}
        summary[phase] = dict(**values,speedup=values['legacy']/values['auto'])
    atomic_json(args.output,summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile',choices=('paper','smoke'),default='smoke')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.threads < 1:
        raise ValueError('基准测试步数和线程数必须为正整数')
    benchmark(args)
