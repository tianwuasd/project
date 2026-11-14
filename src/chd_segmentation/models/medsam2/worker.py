from copy import deepcopy
import gc
from pathlib import Path
import random
import sys
from time import perf_counter

import numpy as np
import psutil
import torch
import torch.distributed as dist

from ...artifacts.store import file_digest,write_json
from .adapter import activate


def main(payload,rank,common,allocation,gate):
    config,recipe = payload['config'],deepcopy(payload['recipe'])
    root = Path(config['run_dir'])/'sequence'
    torch.set_num_threads(1)
    activate(config)
    from omegaconf import OmegaConf
    import training.trainer as official
    from training.utils.train_utils import MemMeter
    from ..workflow import save_state
    from ..execution import all_states,pressure,verify_device,rank_measurement
    from ...training.resources import record_peaks
    from .data import ResourceWorkerInit,set_read_gate

    device = torch.device('cuda',rank) if allocation['gpus'] else torch.device('cpu')
    verify_device(device,allocation,rank)
    recipe['scratch']['train_video_batch_size'] = allocation['shares'][rank]
    recipe['scratch']['num_train_workers'] = allocation['supply']['workers'][rank]
    recipe['trainer']['data']['train']['batch_sizes'] = [allocation['shares'][rank]]
    recipe['trainer']['data']['train']['num_workers'] = allocation['supply']['workers'][rank]
    if rank:
        recipe['trainer']['logging']['log_dir'] = str(root/f'logs/rank{rank}')

    history = []
    observed_peak = 0

    class CheckpointTrainer(official.Trainer):
        def _run_step(self,*args,**kwargs):
            if pressure(common,self.device):
                raise MemoryError('MedSAM2在批次边界检测到主机或显存压力，保留最近完整epoch检查点')
            return super()._run_step(*args,**kwargs)

        def _save_checkpoint(self,checkpoint,checkpoint_path):
            # 将恢复所需状态放入同一文件，完整写入后原子替换。
            checkpoint = dict(checkpoint,_chd=dict(identity=payload['identity'],history=deepcopy(history),
                rank_states=rank_states,world_size=allocation['world'],batch_layout=allocation['shares'],
                workers=allocation['supply']['workers']))
            save_state(Path(checkpoint_path),checkpoint)

    class DeviceMemoryMeter(MemMeter):
        def update(self,n=1,reset_peak_usage=True):
            nonlocal observed_peak
            if self.device.type!='cpu':
                observed_peak = max(observed_peak,rank_measurement(self.device,allocation,rank)['peak_vram_bytes'])
                return super().update(n,reset_peak_usage)
            self.val = psutil.Process().memory_info().rss/1e9
            self.sum += self.val*n
            self.count += n
            self.avg = self.sum/self.count
            self.peak = max(self.peak,self.val)

    # CPU测量真实RSS；CUDA继续使用官方显存测量。
    official.MemMeter = DeviceMemoryMeter
    started = perf_counter()
    arguments = OmegaConf.create(recipe['trainer'])
    del arguments['_target_']
    trainer = CheckpointTrainer(**arguments)
    applied_workers = allocation['supply']['workers'][rank]
    set_read_gate(gate,max(common['resources']['prefetch'],applied_workers))
    trainer.train_dataset.worker_init_fn = ResourceWorkerInit(common,allocation['owner'],gate)
    restored_exact = None
    if payload['resume']:
        checkpoint = root/'checkpoints/checkpoint.pt'
        current = torch.load(checkpoint,map_location='cpu',weights_only=True)
        extra = current['_chd']
        if (extra['identity']!=payload['identity'] or extra['world_size']!=allocation['world'] or
            extra['batch_layout']!=allocation['shares'] or extra['workers']!=allocation['supply']['workers']):
            raise ValueError('MedSAM2官方检查点与运行身份不一致')
        state = extra['rank_states'][rank]
        from ...training.checkpoint import tensors_equal
        restored_exact = (tensors_equal(trainer.model.module.state_dict(),current['model']) and
                          tensors_equal(trainer.optim.optimizer.state_dict(),current['optimizer']))
        if not restored_exact:
            raise ValueError('官方模型参数未完整恢复')
        del current
        random.setstate(state['python_rng'])
        numpy_rng = state['numpy_rng']
        np.random.set_state((numpy_rng[0],np.asarray(numpy_rng[1],dtype=np.uint32),*numpy_rng[2:]))
        torch.set_rng_state(state['torch_rng'])
        trainer.model.module.rng.bit_generator.state = state['prompt_rng']
        if trainer.device.type=='cuda':
            torch.cuda.set_rng_state(state['cuda_rng'],trainer.device)
        history = extra['history']
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
    reason = None
    actual_prefetch = None
    initial_update = trainer.steps['train']
    while trainer.epoch<trainer.max_epochs:
        if limit is not None and trainer.steps['train']>=limit:
            break
        if pressure(common,trainer.device):
            reason = 'resource_pressure'
            break
        loader = trainer.train_dataset.get_loader(epoch=trainer.epoch)
        actual_prefetch = loader.dataloaders[0].prefetch_factor
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
        numpy_rng = np.random.get_state()
        rank_states = all_states(dict(python_rng=random.getstate(),
            numpy_rng=(numpy_rng[0],numpy_rng[1].tolist(),*numpy_rng[2:]),torch_rng=torch.get_rng_state(),
            prompt_rng=deepcopy(trainer.model.module.rng.bit_generator.state),
            cuda_rng=torch.cuda.get_rng_state(trainer.device) if trainer.device.type=='cuda' else None))
        trainer.save_checkpoint(trainer.epoch,checkpoint_names=['checkpoint'])
        dist.barrier()
        del loader
        gc.collect()
    changed = None if model_before is None else any(not torch.equal(value.detach().cpu(),model_before[key])
        for key,value in trainer.model.module.state_dict().items())
    if changed is False and trainer.steps['train']>initial_update:
        raise ValueError('官方训练未改变模型参数')
    if reason and not (root/'checkpoints/checkpoint.pt').is_file():
        raise MemoryError('MedSAM2在首次训练前资源不足，未生成检查点')
    state = dict(identity=payload['identity'],model=trainer.model.module.state_dict(),
        config=config,prepared=payload['prepared'],prepared_identity=payload['prepared']['identity'],stage='sequence',
        scope=payload['prepared']['scope'],initialization_sha256=payload['weight']['sha256'],
        update=trainer.steps['train'],history=history,official=True)
    if rank==0:
        save_state(root/'latest.pt',state)
    dist.barrier()
    measurement = rank_measurement(trainer.device,allocation,rank)
    if trainer.device.type=='cuda':
        measurement['peak_vram_bytes'] = max(measurement['peak_vram_bytes'],observed_peak)
    measurements = all_states(measurement)
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
        peak_vram_bytes=max(m['peak_vram_bytes'] for m in measurements) if trainer.device.type=='cuda' else None,
        world_size=allocation['world'],batch_layout=allocation['shares'],ranks=measurements,reason=reason,
        data_workers=allocation['supply']['workers'],prefetch_factor=actual_prefetch)
    if rank==0:
        write_json(root/'summary.json',summary)
        if reason is None:
            record_peaks(common,allocation['spec'],allocation['gpus'],allocation['shares'],
                [m['peak_vram_bytes'] for m in measurements] if allocation['gpus'] else [])
    dist.destroy_process_group()
