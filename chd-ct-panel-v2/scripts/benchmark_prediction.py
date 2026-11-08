import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from chd_stunet.acceleration.prediction import admit_prediction_device, desktop_identity
from chd_stunet.prediction.infer import sliding_logits, tune_batch
from chd_stunet.prediction.model import read_json, restore_network
from chd_stunet.observability.runlog import write_json


def main():
    parser = argparse.ArgumentParser(description='Repeated real-CT sliding-window capacity comparison')
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--share-desktop-pid', type=int)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    source = Path(args.run)
    job = read_json(source/'job.json')
    model = job['models'][0]
    record = job['records'][0]
    prepared = read_json(source/'intermediate'/f"fold_{record['fold']}"/record['case_id']/'prepared.json')
    image = np.load(prepared['image'], mmap_mode='c')
    # 明确只测真实 CT 中的固定区域，避免把局部稳态吞吐当整例耗时。
    shape = [min(n, p*2) for n, p in zip(image.shape[1:], model['patch_size'])]
    image = image[(slice(None), *(slice(0, n) for n in shape))]
    shared = desktop_identity(args.share_desktop_pid, True) if args.share_desktop_pid else None
    device, admission = admit_prediction_device('cuda:0', shared)
    torch.set_num_threads(4)
    network = restore_network(model, device)
    chosen, capacity = tune_batch(network, image, model['patch_size'], device, admission, 8, tta=False)
    for batch in sorted({1, chosen}):
        sliding_logits(network, image, model['patch_size'], device, batch=batch, tta=False)
    rows = []
    baseline = None
    for repeat in range(args.repeats):
        for batch in (1, chosen) if repeat % 2 == 0 else (chosen, 1):
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            logits, performance = sliding_logits(network, image, model['patch_size'], device, batch=batch, tta=False)
            seconds = time.perf_counter()-started
            if baseline is None:
                baseline = logits.copy()
            difference = float(np.max(np.abs(baseline-logits)))
            agreement = float(np.mean(baseline.argmax(0) == logits.argmax(0)))
            if agreement < .999:
                raise AssertionError('批量变化后的标签一致率低于99.9%，需要核查数值行为')
            rows.append(dict(performance, repeat=repeat, seconds=seconds,
                             max_absolute_logit_difference=difference, label_agreement=agreement,
                             peak_reserved_bytes=int(torch.cuda.max_memory_reserved(device))))
            print(json.dumps(rows[-1]), flush=True)
    serial = statistics.median(r['seconds'] for r in rows if r['batch'] == 1)
    adaptive = statistics.median(r['seconds'] for r in rows if r['batch'] == chosen)
    report = dict(scope='real CT subset steady sliding-window; no loading/export; TTA disabled',
                  shape=list(image.shape), checkpoint=model['pointer']['sha256'], chosen_batch=chosen,
                  baseline_median_seconds=serial, adaptive_median_seconds=adaptive,
                  speedup=serial/adaptive, capacity=capacity, trials=rows)
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('capacity', 'trials')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
