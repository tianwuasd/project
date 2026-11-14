from copy import deepcopy
import gc
import os
from pathlib import Path
import random
import socket
import sys
from time import perf_counter

import numpy as np
import psutil
import torch
import torch.distributed as dist

from ...artifacts.store import file_digest,read_json,write_json
from .adapter import activate


def main(input_path):
    payload = read_json(input_path)
    config,recipe = payload['config'],payload['recipe']
    root = Path(config['run_dir'])/'sequence'
    torch.set_num_threads(config['threads'])
    activate(config)
    from omegaconf import OmegaConf
    import training.trainer as official
    from training.utils.train_utils import MemMeter
    from ..workflow import save_state

    history = []

    class CheckpointTrainer(official.Trainer):
        def _save_checkpoint(self,checkpoint,checkpoint_path):
            numpy_rng = np.random.get_state()
            # 将恢复所需状态放入同一文件，完整写入后原子替换。
            checkpoint = dict(checkpoint,_chd=dict(identity=payload['identity'],history=deepcopy(history),
                python_rng=random.getstate(),
                numpy_rng=(numpy_rng[0],numpy_rng[1].tolist(),*numpy_rng[2:]),
                torch_rng=torch.get_rng_state(),
                prompt_rng=deepcopy(official.unwrap_ddp_if_wrapped(self.model).rng.bit_generator.state),
                cuda_rng=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None))
            save_state(Path(checkpoint_path),checkpoint)

    class DeviceMemoryMeter(MemMeter):
        def update(self,n=1,reset_peak_usage=True):
            if self.device.type!='cpu':
                return super().update(n,reset_peak_usage)
            self.val = psutil.Process().memory_info().rss/1e9
            self.sum += self.val*n
            self.count += n
            self.avg = self.sum/self.count
            self.peak = max(self.peak,self.val)

    # CPU测量真实RSS；CUDA继续使用官方显存测量。
    official.MemMeter = DeviceMemoryMeter
    with socket.socket() as channel:
        channel.bind(('127.0.0.1',0))
        port = channel.getsockname()[1]
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK='0',LOCAL_RANK='0',WORLD_SIZE='1')
    started = perf_counter()
    arguments = OmegaConf.create(recipe['trainer'])
    del arguments['_target_']
    trainer = CheckpointTrainer(**arguments)
    restored_exact = None
    if payload['resume']:
        checkpoint = root/'checkpoints/checkpoint.pt'
        current = torch.load(checkpoint,map_location='cpu',weights_only=True)
        state = current['_chd']
        if state['identity']!=payload['identity']:
            raise ValueError('MedSAM2官方检查点与运行身份不一致')
        restored_exact = all(torch.equal(value.detach().cpu(),current['model'][key])
                             for key,value in trainer.model.module.state_dict().items())
        if not restored_exact:
            raise ValueError('官方模型参数未完整恢复')
        del current
        random.setstate(state['python_rng'])
        numpy_rng = state['numpy_rng']
        np.random.set_state((numpy_rng[0],np.asarray(numpy_rng[1],dtype=np.uint32),*numpy_rng[2:]))
        torch.set_rng_state(state['torch_rng'])
        trainer.model.module.rng.bit_generator.state = state['prompt_rng']
        if trainer.device.type=='cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        history = state['history']
        del state
    else:
        initialized = torch.load(config['pretrained'],map_location='cpu',weights_only=True)['model']
        if not all(torch.equal(value.detach().cpu(),initialized[key]) for key,value in trainer.model.module.state_dict().items()):
            raise ValueError('官方初始化权重未逐值加载')
        del initialized
    limit = payload['stop_after']
    if config['technical']:
        limit = config['updates'] if limit is None else limit
        if not 1<=limit<=2:
            raise ValueError('MedSAM2短测更新预算必须为1至2')
    model_before = {key:value.detach().cpu().clone() for key,value in trainer.model.module.state_dict().items()} if config['technical'] else None
    while trainer.epoch<trainer.max_epochs:
        if limit is not None and trainer.steps['train']>=limit:
            break
        loader = trainer.train_dataset.get_loader(epoch=trainer.epoch)
        if len(loader)<1 or limit is not None and trainer.steps['train']+len(loader)>limit:
            raise ValueError('停止更新数需要位于官方完整epoch边界')
        mark = perf_counter()
        before = trainer.steps['train']
        result = trainer.train_epoch(loader)
        elapsed = perf_counter()-mark
        trainer.epoch += 1
        history.append(dict(epoch=trainer.epoch,update=trainer.steps['train'],
            optimizer_updates=trainer.steps['train']-before,seconds=elapsed,
            updates_per_second=(trainer.steps['train']-before)/elapsed,
            learning_rates=sorted(set(group['lr'] for group in trainer.optim.optimizer.param_groups)),metrics=result))
        trainer.save_checkpoint(trainer.epoch,checkpoint_names=['checkpoint'])
        del loader
        gc.collect()
    changed = None if model_before is None else any(not torch.equal(value.detach().cpu(),model_before[key])
        for key,value in trainer.model.module.state_dict().items())
    if changed is False:
        raise ValueError('官方训练未改变模型参数')
    state = dict(identity=payload['identity'],model=trainer.model.module.state_dict(),
        config=config,prepared=payload['prepared'],prepared_identity=payload['prepared']['identity'],stage='sequence',
        scope=payload['prepared']['scope'],initialization_sha256=payload['weight']['sha256'],
        update=trainer.steps['train'],history=history,official=True)
    save_state(root/'latest.pt',state)
    memory = psutil.Process().memory_info()
    summary = dict(model='medsam2',stage='sequence',scope=payload['prepared']['scope'],
        update=trainer.steps['train'],history=history,checkpoint=str(root/'latest.pt'),
        checkpoint_sha256=file_digest(root/'latest.pt'),official_checkpoint=str(root/'checkpoints/checkpoint.pt'),
        restored_exact=restored_exact,parameters_changed_this_invocation=changed,
        seconds=perf_counter()-started,official_weight=payload['weight'],
        official_recipe='sam2.1_hiera_tiny512_FLARE_RECIST.yaml',frames=8,max_objects=3,
        gradient_clip_norm=recipe['trainer']['optim']['gradient_clip']['max_norm'],
        epochs=trainer.max_epochs,batch=recipe['scratch']['train_video_batch_size'],
        rss_bytes=memory.rss,peak_rss_bytes=memory.peak_wset if sys.platform=='win32' else None,
        peak_vram_bytes=torch.cuda.max_memory_allocated() if trainer.device.type=='cuda' else None)
    write_json(root/'summary.json',summary)
    dist.destroy_process_group()


if __name__=='__main__':
    main(sys.argv[1])
