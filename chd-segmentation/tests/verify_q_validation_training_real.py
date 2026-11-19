import argparse
from copy import deepcopy
import json
import gc
import random
from pathlib import Path
import sys
from time import perf_counter
from unittest import TestCase

import nibabel as nib
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT/'tests'))

from verify_model_offline_real import model_config
from chd_segmentation.config import DEFAULTS, load_config
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.artifacts.store import read_json, write_json
from chd_segmentation.models.offline import preprocess_model
from chd_segmentation.models.workflow import options, save_state, load_state
from chd_segmentation.models.q_route1.adapter import QAdapter
from chd_segmentation.data.dataset import CaseDataset
from chd_segmentation.training.checkpoint import tensors_equal
from chd_segmentation.models.q_route1.validation_store import ValidationWorkspace


class Work:
    def __init__(self, root):
        self.root = root


def run(args):
    started = perf_counter()
    root = Path(args.work_root)
    root.mkdir(parents=True, exist_ok=True)
    row = next(r for r in read_json(args.manifest)['cases'] if r['case_id']=='ct_1003')
    native = nib.load(Path(args.data_root)/row['image']).shape
    roi = [[n//2-16, n//2+16] for n in native]
    item = dict(case=row, roi_xyz=roi, data_root=args.data_root, pretrained=args.pretrained)
    workspace = Work(root)
    cases = []
    for name, role in (('ct_1003','train'),('ct_1012','validation')):
        source_row = next(r for r in read_json(args.manifest)['cases'] if r['case_id']==name)
        cases.append(dict(source_row, roles=[role]))
    write_json(root/'manifest.json', dict(dataset='imagechd', scope='technical', data_root=args.data_root,
                                         cases=cases, plan_scope='technical'))
    source_options = deepcopy(DEFAULTS)
    source_options.update(dataset='imagechd', scope='technical', manifest=str(root/'manifest.json'),
        pretrained=args.pretrained, work_root=str(root/'course'), run_dir=str(root/'course/source-logs'))
    source_options['training'].update(patch=[32]*3, batch=4, attempts=2)
    source_options['technical'] = dict(roi_xyz=roi, allow_role_overlap=False)
    source_options['resources'].update(workers=0, ram_fraction=1., reserve_ram_bytes=512*1024**2)
    write_json(root/'source-config.json', source_options)
    source = root/'course/source.json'
    if not source.exists():
        prepare(load_config(root/'source-config.json'), source)
    model_path, model = model_config(workspace, 'q_route1', args.pretrained)
    model['execution'].update(cpu_processes=2, workers=0, cache_bytes=16*1024**2,
        validation_block_bytes=1024**2, ram_fraction=1., reserve_ram_bytes=512*1024**2)
    model['logging'] = dict(heartbeat_seconds=15, case_details=True)
    model['technical_roles'] = dict(base=dict(train=['ct_1003'], validation=['ct_1012'], test=[]))
    write_json(model_path, model)
    ready = root/'course/offline.json'
    if not ready.exists():
        preprocess_model(model_path, root/'source-config.json', source, root/'data', ready)
    config = options(model_path)
    prepared = read_json(ready)
    torch.set_num_threads(4)
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    rng = np.random.default_rng(config['seed'])
    dataset = CaseDataset(prepared, dict(cache_bytes=16*1024**2, reserve_ram_bytes=512*1024**2), with_label=True)
    bundle = QAdapter(config, prepared, 'base')
    optimizer = bundle.optimizer()
    history = []
    checkpoint = root/'direct-latest.pt'
    for update in range(2):
        bundle.model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for _ in range(4):
            loss = bundle.loss(bundle.sample(dataset, ['ct_1003'], rng), rng, update)
            assert torch.isfinite(loss)
            (loss/4).backward()
            total += float(loss.detach())/4
        torch.nn.utils.clip_grad_norm_(bundle.model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        bundle.model.eval()
        with torch.no_grad():
            value = bundle.validate(dataset, ['ct_1012'])
        history.append(dict(update=update+1, loss=total, validation_loss=value))
        state = dict(model=bundle.model.state_dict(), optimizer=optimizer.state_dict(),
                     torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                     numpy_rng=np.random.get_state(), data_rng=rng.bit_generator.state, update=update+1)
        save_state(checkpoint, state)
        if update == 0:
            del state, bundle, optimizer, loss
            gc.collect()
            state = load_state(checkpoint)
            bundle = QAdapter(config, prepared, 'base', initialize=False)
            bundle.model.load_state_dict(state['model'])
            optimizer = bundle.optimizer()
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['torch_rng'])
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            rng.bit_generator.state = state['data_rng']
            assert tensors_equal(bundle.model.state_dict(), state['model'])
            assert tensors_equal(optimizer.state_dict(), state['optimizer'])
            assert torch.equal(torch.get_rng_state(), state['torch_rng'])
            del state
    # 实际创建临时数组后主动取消，确认退出清理。
    with TestCase().assertRaises(InterruptedError):
        with ValidationWorkspace(config, dict(task_case='cancel'), 1024**2) as work:
            work.create('layer', (1,32,32,32), np.float32)
            raise InterruptedError('用户主动取消验证')
    assert not list((Path(config['work_root'])/'prediction-work').iterdir())
    dataset.clear()
    record = dict(state='passed', train_case='ct_1003', validation_case='ct_1012', roi_xyz=roi, training_updates=2,
                  process_count=1, history=history, restored_parameters_optimizer_rng=True,
                  two_process_cli='not_run_due_to_insufficient_physical_memory',
                  cancellation_cleaned=True, seconds=perf_counter()-started)
    write_json(Path(args.report)/'training.json', record)
    print(json.dumps(dict(state='passed', updates=2, restored_exact=True, seconds=record['seconds'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('work-root','report','manifest','data-root','pretrained'):
        parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
