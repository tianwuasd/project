import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from dataset1282_analysis.independent.stunet import load_pretrained, network, topology
from dataset1282_analysis.stu_imagechd import STUImageCHDDataset, imagechd_cross_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptor', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    torch.set_num_threads(4)
    dataset = STUImageCHDDataset(args.descriptor, fold=0, role='train')
    sample = dataset[0]
    assert sample['label_space'] == dataset.label_space and dataset.num_classes == 8
    label = sample['label'][0]
    points = np.argwhere((label > 0) & (label < 8))
    if not len(points):
        raise ValueError('真实病例没有可用的前景监督')
    start = np.clip(points[len(points)//2]-16, 0, np.asarray(label.shape)-32)
    region = tuple(slice(int(s), int(s)+32) for s in start)
    image = torch.from_numpy(np.ascontiguousarray(sample['image'][(slice(None), *region)]))[None]
    target = torch.from_numpy(np.ascontiguousarray(sample['label'][(slice(None), *region)]))[None]
    model = network(topology(tuple(sample['spacing_xyz']), (32,)*3), False, num_classes=8)
    receipt = load_pretrained(model, args.weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    model.train()
    parameter = next(p for p in model.parameters() if p.requires_grad)
    before = parameter.detach().clone()
    loss = imagechd_cross_entropy(model(image), target)
    if not bool(torch.isfinite(loss)):
        raise ValueError('ImageCHD短测损失非有限值')
    loss.backward()
    optimizer.step()
    if torch.equal(before, parameter):
        raise ValueError('ImageCHD短测没有更新参数')
    extra_index = next((i for i, c in enumerate(dataset.cases) if c['extra_label_ids']), None)
    checked = [sample['case_id']]
    ignored = 0
    if extra_index is not None and extra_index != 0:
        extra = dataset[extra_index]
        ignored = int((extra['label'] == -1).sum())
        if not ignored:
            raise ValueError('附加标签没有转成忽略区域')
        checked.append(extra['case_id'])
    row = dict(stage='imagechd-eight-class-update', cases=checked, updates=1,
               patch=[32, 32, 32], loss=float(loss.detach()),
               ignored_source_voxels=ignored, seconds=perf_counter()-started,
               peak_ram_bytes=getattr(psutil.Process().memory_info(), 'peak_wset', None),
               peak_cuda_bytes=None, pretrained=receipt['sha256'], generated_files=0)
    with args.log.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
