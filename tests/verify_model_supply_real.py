from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import sys
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))

from smoke import SmokeWorkspace
from chd_segmentation.artifacts.store import write_json,file_digest
from chd_segmentation.config import DEFAULTS,load_config
from chd_segmentation.preprocessing.prepare import prepare
from chd_segmentation.models.workflow import options
from chd_segmentation.models.supply import CpuReady,OrderedSupply
from chd_segmentation.training.resources import host_capacity


def fingerprint(batch):
    checksum = hashlib.sha256()
    for tensor in (batch.img_batch,batch.masks,batch.obj_to_frame_idx,
                   batch.metadata.unique_objects_identifier,batch.metadata.frame_orig_size):
        checksum.update(str((tensor.dtype,list(tensor.shape))).encode())
        checksum.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return checksum.hexdigest()


def worker(rank,config,recipe,stream,root,port):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),WORLD_SIZE='2',RANK=str(rank),LOCAL_RANK=str(rank),USE_LIBUV='0')
    torch.set_num_threads(1)
    from chd_segmentation.models.medsam2.adapter import activate
    activate(config)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from chd_segmentation.models.medsam2.data import set_read_gate
    from chd_segmentation.models.medsam2.supply import ConsumedLoader
    dist.init_process_group('gloo',rank=rank,world_size=2,timeout=timedelta(seconds=180))
    rows = []
    for ordered in (False,True):
        random.seed(301+rank);np.random.seed(301+rank);torch.manual_seed(301+rank)
        data = instantiate(OmegaConf.create(recipe['trainer']['data']['train']))
        stream.start(rank,'official-data',[1,1])
        set_read_gate(stream if ordered else None,2,rank)
        loader = data.get_loader(0)
        indices = list(loader.dataloaders[0].batch_sampler)
        if ordered:
            loader = ConsumedLoader(loader,stream,rank,0,torch.device('cpu'))
        if len(loader)!=2:
            raise ValueError('真实供数验证必须恰好两个批次')
        before = perf_counter()
        hashes = []
        for batch in loader:
            if list(batch.img_batch.shape[:1])+list(batch.img_batch.shape[-2:])!=[8,512,512]:
                raise ValueError('未保持官方八帧512序列')
            hashes.append(fingerprint(batch))
            if ordered and len(hashes)==1:
                with stream.condition:
                    pending = stream.condition.wait_for(lambda: stream.waiting[rank]>0,timeout=30)
                if not pending:
                    raise ValueError('后续批次未实际进入背压等待')
            value = torch.tensor([float(batch.img_batch.mean())])
            if ordered:
                stream.ready.wait(rank,f'test-transfer:{len(hashes)}',torch.device('cpu'))
            dist.all_reduce(value)
        stream.ready.wait(rank,f'test-finished:{ordered}')
        rows.append(dict(ordered=ordered,hashes=hashes,indices=indices,seconds=perf_counter()-before))
        del loader,data
    plain = [[ticket[2] for ticket in batch] for batch in rows[1]['indices']]
    if rows[0]['indices']!=plain or rows[0]['hashes']!=rows[1]['hashes']:
        raise ValueError('有序供数改变了官方索引、增强或随机调用')
    from chd_segmentation.models.medsam2.data import NativeCases
    from chd_segmentation.preprocessing.prepare import load_prepared
    raw = recipe['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0]['video_dataset']
    stream.start(rank,'rank0-case',[1,0])
    value = torch.tensor(0.)
    if rank==0:
        cases = NativeCases(load_prepared(raw['prepared_path']),log_dir=root)
        with stream.build(rank,0,0):
            stream.reserve(rank,raw['names'][0])
            entry = cases.get(raw['names'][0])
        value = torch.tensor(float(entry['image'].mean()))
        cases.clear();del cases,entry
    stream.finish(rank,'rank0-case',torch.device('cpu'))
    dist.all_reduce(value)
    if rank==0:
        write_json(Path(root)/'rank0-receipt.json',dict(actual_image_mean=float(value)))
    stream.ready.wait(rank,'rank0-file-written')
    dist.barrier()
    write_json(Path(root)/f'rank{rank}.json',dict(rank=rank,rows=rows,exact=True,optimizer_updates=0,
        asymmetric_case_read=True,checkpoint_write_control=True))
    dist.destroy_process_group()


def run(args):
    available = host_capacity()['available_bytes']
    if available<11*2**30:
        raise MemoryError('真实供数验证需为七个CPU进程及四GiB余量保留至少十一GiB可用内存')
    selected = [e for e in yaml.safe_load(Path(args.spec).read_text(encoding='utf-8'))['experiments'] if e['dataset']=='imagechd']
    if len(selected)!=1:
        raise ValueError('供数检查只允许一个真实ImageCHD病例')
    experiment = selected[0]
    with SmokeWorkspace(args.work_root,args.report) as workspace:
        directory = workspace.root
        row = deepcopy(experiment['case']);row['roles']=['train','validation']
        write_json(directory/'manifest.json',dict(dataset='imagechd',scope='technical',plan_scope='technical',
            data_root=experiment['data_root'],cases=[row]))
        base = deepcopy(DEFAULTS)
        base.update(dataset='imagechd',scope='technical',manifest=str(directory/'manifest.json'),
            pretrained=experiment['pretrained'],run_dir=str(directory/'course/stu'),work_root=str(directory/'course'))
        base['technical']=dict(roi_xyz=experiment['roi_xyz'],allow_role_overlap=True)
        write_json(directory/'base.json',base)
        prepared_path = directory/'course/prepared.json'
        prepared = prepare(load_config(directory/'base.json'),prepared_path)
        write_json(directory/'model.json',dict(model='medsam2',run_dir=str(directory/'course/med'),device='cpu',
            precision='fp32',technical=True,updates=2,frames=8,resolution=512,freeze_image_encoder=False,batch=2,workers=2,
            pretrained=str(ROOT/'data/model_weights/MedSAM2_latest.pt'),official_root=str(ROOT/'vendor/MedSAM2'),
            dependency_root=str(ROOT/'env/medsam2')))
        config = options(directory/'model.json')
        from chd_segmentation.models.medsam2.official import effective_config
        recipe = effective_config(config,prepared,prepared_path,directory/'course/med',[row['case_id']])
        recipe['trainer']['data']['train']['batch_sizes']=[1]
        recipe['trainer']['data']['train']['num_workers']=2
        recipe['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0]['multiplier']=4
        context = torch.multiprocessing.get_context('spawn')
        from chd_segmentation.models.memory import case_estimates
        estimates = case_estimates(dict(model='medsam2'),prepared,'sequence')
        stream = OrderedSupply(context,[1,1],1,CpuReady(context,2,directory),directory,
            capacity_bytes=max(v['workspace_bytes'] for v in estimates),estimates=estimates)
        with socket.socket() as rendezvous:
            rendezvous.bind(('127.0.0.1',0));port = rendezvous.getsockname()[1]
        previous = dict(os.environ)
        os.environ.update(PYTHONPATH=os.pathsep.join([str(ROOT/'src'),config['official_root'],config['dependency_root']]),
            TEMP=str(directory),TMP=str(directory),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',USE_LIBUV='0',CUDA_VISIBLE_DEVICES='')
        try:
            torch.multiprocessing.spawn(worker,args=(config,recipe,stream,str(directory),port),nprocs=2,join=True)
        finally:
            os.environ.clear();os.environ.update(previous)
        logs = [json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        starts = [value for value in logs if value['event']=='model_build_started']
        if len(starts)!=5 or any(v['active']>1 or max(v['ready_depth'])>1 for v in starts):
            raise ValueError('实际病例构建并发或队列深度超过界限')
        records = [json.loads((directory/f'rank{rank}.json').read_text(encoding='utf-8')) for rank in range(2)]
        write_json(workspace.report/'supply-events.json',logs)
        workspace.results.append(dict(case_id=row['case_id'],ranks=records,frames=8,resolution=512,
            workers_per_rank=2,batches_per_rank=2,reader_slots=1,ready_depth_max=1,optimizer_updates=0,
            admission_available_bytes=available,data_only_required_bytes=11*2**30,
            trainer_model_created=False,CUDA_used=False,source_unchanged=all(file_digest(row[k])==row['sha256'][k] for k in ('image','label'))))


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec',required=True)
    parser.add_argument('--work-root',required=True)
    parser.add_argument('--report',required=True)
    run(parser.parse_args())
