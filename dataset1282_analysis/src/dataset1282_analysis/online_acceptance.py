from dataclasses import asdict
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import psutil
import torch

from .input_source import open_prepared
from .online_prepare import compact
from .xu_workflow import input_id


def accept(prepared, output, *, sam_weights=None, stunet_weights=None, device='cpu', checks='all',
           models=('xu', 'stunet', 'sam'), targets=None):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    workspace = root/'acceptance-work'
    if workspace.exists():
        raise FileExistsError('验收工作目录必须不存在')
    log = root/'acceptance.log'
    started = time.perf_counter()
    events = []
    def event(**row):
        events.append(dict(elapsed_seconds=time.perf_counter()-started, **row))
        with log.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(events[-1], ensure_ascii=False)+'\n')
        print(json.dumps(events[-1], ensure_ascii=False), flush=True)
    original = open_prepared(prepared, allow_technical_subset=True)
    names = [original.case_ids(role)[0] for role in ('train', 'development')]
    if original.scope != 'technical_subset' or len(set(names)) != 2:
        raise ValueError('验收要求独立技术子集中的一例训练和一例开发数据')
    event(stage='scope', cases=names, updates_per_network=1 if sam_weights else 0, disk_limit_bytes=20*1024**3,
          ram_peak='not_measured', vram_peak='not_measured')
    workspace.mkdir()
    try:
        event(stage='compact', result=compact(prepared, workspace/'prepared', technical_subset=True))
        online = open_prepared(workspace/'prepared', allow_technical_subset=True)
        torch.set_num_threads(4)
        assert input_id(original, 'marginal') == input_id(online, 'marginal')
        from .independent.stunet import PatchReader
        from .independent.sam_train import sample
        for name in names if checks == 'all' else []:
            before = time.perf_counter()
            expected = original.read_image(name, (64,)*3)
            legacy_seconds = time.perf_counter()-before
            before = time.perf_counter()
            actual = online.read_image(name, (64,)*3)
            np.testing.assert_array_equal(actual, expected)
            cold = time.perf_counter()-before
            before = time.perf_counter()
            np.testing.assert_array_equal(online.read_image(name, (64,)*3), actual)
            warm = time.perf_counter()-before
            np.testing.assert_array_equal(original.read_label(name, (64,)*3), online.read_label(name, (64,)*3))
            assert asdict(original.geometry(name)) == asdict(online.geometry(name))
            pairs = [PatchReader(c, (32,)*3).read(name, np.random.default_rng(2023), True)
                     for c in (original, online)]
            for a,b in zip(*pairs):
                np.testing.assert_array_equal(a,b)
            label = original.supervision[name]['present_classes'][0]
            a,b = (sample(c, name, label, 2023) for c in (original, online))
            for u,v in zip(a,b):
                np.testing.assert_array_equal(u,v)
            event(stage='input-equivalence', case=name, exact=True, legacy_seconds=legacy_seconds, cold_seconds=cold,
                  warm_seconds=warm, warm_cases_per_second=1/warm, cache=online.memory.stats,
                  resident_bytes=psutil.Process().memory_info().rss)
            online.clear()
            del expected, actual, pairs, a, b
            gc.collect()
        assert not list((workspace/'prepared').rglob('*.b2nd'))
        assert not list((workspace/'prepared').rglob('*.nii*'))
        if checks == 'all':
            event(stage='accepted-inputs', cases=len(names), request_digest=original.request_digest)
        if checks == 'integration':
            from .online_integration_acceptance import check_integration
            check_integration(original, online, names, workspace, targets, event)
        if sam_weights is not None:
            if stunet_weights is None:
                raise ValueError('模型验收同时需要两个独立模型的正式预训练权重')
            from .online_model_acceptance import check_models
            check_models(online, names[0], workspace, device, sam_weights, stunet_weights, event, models)
        event(stage='accepted', checks=checks, models=list(models) if sam_weights else [], cases=len(names))
    finally:
        kind, error, _ = sys.exc_info()
        if kind is not None:
            event(stage='failed', error_type=kind.__name__, detail=str(error))
        gc.collect()
        files = [p for p in workspace.rglob('*') if p.is_file()]
        count, size = len(files), sum(p.stat().st_size for p in files)
        if workspace.resolve() != root/'acceptance-work' or workspace.is_symlink():
            raise ValueError('验收清理路径改变')
        info = psutil.Process().memory_info()
        event(stage='cleanup', files=count, bytes=size, folder=str(workspace),
              peak_ram_bytes=getattr(info, 'peak_wset', None),
              peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None)
        shutil.rmtree(workspace)
        event(stage='cleaned', exists=workspace.exists(), total_seconds=time.perf_counter()-started)
