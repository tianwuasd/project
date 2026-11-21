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
from ...training.attempts import settle_attempts, finish_attempt


@settle_attempts
def main(payload,rank,common,allocation,gate,memory):
    config,recipe = payload['config'],deepcopy(payload['recipe'])
    root = Path(config['run_dir'])/'sequence'
    torch.set_num_threads(1)
    activate(config)
    from omegaconf import OmegaConf
    import training.trainer as official
    from training.utils.train_utils import MemMeter
    from ..workflow import save_state
    from ..execution import all_states,pressure,verify_device,rank_measurement,record_model_peaks
    from ...acceleration.measurements import complete_peak
    from ...training.resources import record_peaks
    from .data import ResourceWorkerInit,set_read_gate

    device = torch.device('cuda',rank) if allocation['gpus'] else torch.device('cpu')
    verify_device(device,allocation,rank)
    recipe['scratch']['train_video_batch_size'] = allocation['shares'][rank]
    recipe['scratch']['num_train_workers'] = allocation['supply']['workers'][rank]
    recipe['trainer']['data']['train']['batch_sizes'] = [allocation['shares'][rank]]
    recipe['trainer']['data']['train']['num_workers'] = allocation['supply']['workers'][rank]
    raw_dataset = recipe['trainer']['data']['train']['datasets'][0]['dataset']['datasets'][0]['video_dataset']
    raw_dataset['reserve_ram_bytes'] = common['resources']['reserve_ram_bytes']
    if rank:
        recipe['trainer']['logging']['log_dir'] = str(root/f'logs/rank{rank}')
    write_json(root/f'official-config-rank{rank}.json',recipe)

    history = []
    selection = None
    successful_updates = 0
    from ...training.stopping import selection_options, update_selection
    stopping = selection_options(config.get('early_stopping', {}), 'medsam2')
    observed_peak = 0
    vram_complete = True

    class CheckpointTrainer(official.Trainer):
        def _setup_torch_dist_and_backend(self,*args,**kwargs):
            gate.ready.wait(rank,'med-distributed-entry',self.device)
            return super()._setup_torch_dist_and_backend(*args,**kwargs)

        def _setup_ddp_distributed_training(self,*args,**kwargs):
            gate.ready.wait(rank,'med-model-initialized',self.device)
            return super()._setup_ddp_distributed_training(*args,**kwargs)

        def _run_step(self,*args,**kwargs):
            gate.ready.wait(rank,f'med-transfer:{self.epoch}:{self.steps["train"]}',self.device)
            if pressure(common,self.device,gate.ready,rank,f'med-step:{self.epoch}:{self.steps["train"]}',memory,
                        release=lambda: self.optim.optimizer.zero_grad(set_to_none=True)):
                raise MemoryError('MedSAM2在批次边界检测到主机或显存压力，保留最近完整epoch检查点')
            mark = perf_counter()
            result = super()._run_step(*args,**kwargs)
            if rank == 0:
                feedback.update(epoch=self.epoch, step=self.steps['train'])
                feedback.snapshot()
            if self.device.type=='cuda':
                torch.cuda.synchronize(self.device)
            from ...observability.events import event
            event(root,'model_forward_backward',rank=rank,epoch=self.epoch,update=self.steps['train'],
                  seconds=perf_counter()-mark,includes_ddp_communication=True,nccl_wait_seconds=None)
            return result

        def _save_checkpoint(self,checkpoint,checkpoint_path):
            # 将恢复所需状态放入同一文件，完整写入后原子替换。
            checkpoint = dict(checkpoint,_chd=dict(identity=payload['identity'],history=deepcopy(history),
                rank_states=rank_states,world_size=allocation['world'],batch_layout=allocation['shares'],
                workers=allocation['supply']['workers'],selection=selection,
                successful_updates=successful_updates))
            save_state(Path(checkpoint_path),checkpoint)

    class DeviceMemoryMeter(MemMeter):
        def update(self,n=1,reset_peak_usage=True):
            nonlocal observed_peak,vram_complete
            if self.device.type!='cpu':
                observed = rank_measurement(self.device,allocation,rank,common)
                if observed['peak_vram_bytes'] is None:
                    vram_complete = False
                else:
                    observed_peak = max(observed_peak,observed['peak_vram_bytes'])
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
    set_read_gate(gate,max(allocation['supply']['prefetch'],applied_workers),rank)
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
        del current
        random.setstate(state['python_rng'])
        numpy_rng = state['numpy_rng']
        np.random.set_state((numpy_rng[0],np.asarray(numpy_rng[1],dtype=np.uint32),*numpy_rng[2:]))
        torch.set_rng_state(state['torch_rng'])
        trainer.model.module.rng.bit_generator.state = state['prompt_rng']
        if trainer.device.type=='cuda':
            torch.cuda.set_rng_state(state['cuda_rng'],trainer.device)
        history = extra['history']
        selection = None if payload.get('selection_source_changed') else extra.get('selection')
        successful_updates = extra.get('successful_updates', trainer.steps['train'])
        del state
    limit = payload['stop_after']
    active_attempt = None
    step_succeeded = False
    def completed_optimizer_step(optimizer, args, kwargs):
        nonlocal successful_updates, step_succeeded
        successful_updates += 1
        step_succeeded = True
    optimizer_hook = trainer.optim.optimizer.register_step_post_hook(completed_optimizer_step)
    if config['technical']:
        limit = config['updates'] if limit is None else limit
        if not 1<=limit<=2:
            raise ValueError('MedSAM2短测更新预算必须为1至2')
    model_before = {key:value.detach().cpu().clone() for key,value in trainer.model.module.state_dict().items()} if config['technical'] else None
    reason = None
    actual_prefetch = None
    initial_update = trainer.steps['train']
    from ...observability.progress import Progress
    feedback = Progress(dict(common, run_dir=str(root)), 'train', total=trainer.max_epochs, unit='epochs') if rank == 0 else None
    resumed_epoch = trainer.epoch
    if rank == 0:
        feedback.update(done=trainer.epoch, new=0, epoch=trainer.epoch, step=trainer.steps['train'])
        feedback.snapshot(force=True)
    gate.ready.wait(rank,'med-state-restored',device)
    while trainer.epoch<trainer.max_epochs:
        if limit is not None and trainer.steps['train']>=limit:
            break
        gate.ready.wait(rank,f'med-epoch-boundary:{trainer.epoch}',device)
        if pressure(common,trainer.device,gate.ready,rank,f'med-epoch:{trainer.epoch}',memory,
                    release=lambda: trainer.optim.optimizer.zero_grad(set_to_none=True)):
            reason = 'resource_pressure'
            break
        gate.start(rank,f'med-epoch:{trainer.epoch}',allocation['shares'],device)
        from .supply import ConsumedLoader
        loader = ConsumedLoader(trainer.train_dataset.get_loader(epoch=trainer.epoch),gate,rank,trainer.epoch,device)
        actual_prefetch = loader.dataloaders[0].prefetch_factor
        if config['technical']:
            from ...training.attempts import claim_attempt, remaining_attempts
            from .bounded_loader import BoundedLoader
            remaining = remaining_attempts(config, 'sequence') if rank == 0 else None
            remaining = all_states(remaining)[0]
            if remaining == 0:
                reason = 'attempt_limit'
                break
            if applied_workers != 0:
                raise ValueError('技术训练尝试需要workers=0，在读取训练输入前提交累计计数')
            def claim():
                nonlocal active_attempt, step_succeeded
                step_succeeded = False
                value = claim_attempt(config, 'sequence') if rank == 0 else None
                active_attempt = all_states(value)[0]
                return active_attempt
            def finish():
                if rank == 0:
                    finish_attempt(config, 'sequence', active_attempt, 'successful' if step_succeeded else 'skipped')
            loader = BoundedLoader(loader, min(1, remaining), claim, finish)
        from .bounded_loader import require_complete_epoch
        require_complete_epoch(steps=trainer.steps['train'],limit=limit,batches=len(loader))
        mark = perf_counter()
        before = trainer.steps['train']
        result = trainer.train_epoch(loader)
        elapsed = perf_counter()-mark
        trainer.epoch += 1
        history.append(dict(epoch=trainer.epoch,update=trainer.steps['train'], successful_updates=successful_updates,
            attempts=trainer.steps['train']-before,seconds=elapsed,
            updates_per_second=(trainer.steps['train']-before)/elapsed,
            learning_rates=sorted(set(group['lr'] for group in trainer.optim.optimizer.param_groups)),metrics=result))
        if config['technical']:
            from ..roles import technical_names
            validation_names = technical_names(config['technical_roles'], 'sequence', 'validation',
                [row['case_id'] for row in payload['prepared']['manifest']['cases']])
        else:
            validation_names = [row['case_id'] for row in payload['prepared']['manifest']['cases']
                                if 'validation' in row['roles']]
        from .validation import validate_sequences
        validation = validate_sequences(trainer, recipe, payload['prepared'], validation_names, config['seed'],
            gate=gate,rank=rank)
        improved = False
        if validation is not None:
            value = validation['value']
            selection = update_selection(selection, value, trainer.epoch, stopping)
            improved = selection['improved']
            history[-1]['validation'] = validation
        if improved and rank == 0:
            save_state(root/'best.pt', dict(identity=payload['identity'], model=trainer.model.module.state_dict(),
                config=config, prepared=payload['prepared'], stage='sequence', scope=payload['prepared']['scope'],
                update=trainer.steps['train'], history=deepcopy(history), selection=selection,
                initialization_sha256=payload['weight'].get('sha256')))
        from ...observability.events import event
        event(root,'model_epoch_completed',model='medsam2',rank=rank,**history[-1])
        numpy_rng = np.random.get_state()
        gate.ready.wait(rank,f'med-checkpoint-state:{trainer.epoch}',device)
        rank_states = all_states(dict(python_rng=random.getstate(),
            numpy_rng=(numpy_rng[0],numpy_rng[1].tolist(),*numpy_rng[2:]),torch_rng=torch.get_rng_state(),
            prompt_rng=deepcopy(trainer.model.module.rng.bit_generator.state),
            cuda_rng=torch.cuda.get_rng_state(trainer.device) if trainer.device.type=='cuda' else None))
        trainer.save_checkpoint(trainer.epoch,checkpoint_names=['checkpoint'])
        gate.ready.wait(rank,f'med-checkpoint-written:{trainer.epoch}',device)
        if rank == 0:
            feedback.update(done=trainer.epoch, new=trainer.epoch-resumed_epoch,
                epoch=trainer.epoch, step=trainer.steps['train'])
            feedback.snapshot(force=True)
        dist.barrier()
        del loader
        gc.collect()
        if selection and selection['stop']:
            reason = 'early_stopping'
            break
    changed = None if model_before is None else any(not torch.equal(value.detach().cpu(),model_before[key])
        for key,value in trainer.model.module.state_dict().items())
    if changed is False and trainer.steps['train']>initial_update:
        raise ValueError('官方训练未改变模型参数')
    if reason and not (root/'checkpoints/checkpoint.pt').is_file():
        raise MemoryError('MedSAM2在首次训练前资源不足，未生成检查点')
    state = dict(identity=payload['identity'],model=trainer.model.module.state_dict(),
        config=config,prepared=payload['prepared'],prepared_identity=payload['prepared']['identity'],stage='sequence',
        scope=payload['prepared']['scope'],initialization_sha256=payload['weight']['sha256'],
        update=trainer.steps['train'],successful_updates=successful_updates,history=history,official=True)
    if rank==0:
        save_state(root/'latest.pt',state)
    gate.ready.wait(rank,'med-export-written',device)
    dist.barrier()
    measurement = rank_measurement(trainer.device,allocation,rank,common)
    if trainer.device.type=='cuda':
        if not vram_complete or measurement['peak_vram_bytes'] is None:
            measurement.update(peak_vram_bytes=None,vram_complete=False)
        else:
            measurement['peak_vram_bytes'] = max(measurement['peak_vram_bytes'],observed_peak)
    measurements = all_states(measurement)
    memory = psutil.Process().memory_info()
    summary = dict(model='medsam2',stage='sequence',scope=payload['prepared']['scope'],
        update=trainer.steps['train'],successful_updates=successful_updates,
        history=history,checkpoint=str(root/'latest.pt'),
        official_checkpoint=str(root/'checkpoints/checkpoint.pt'),
        restored=bool(payload['resume']),
        restored_exact=restored_exact,parameters_changed_this_invocation=changed,
        seconds=perf_counter()-started,official_weight=payload['weight'],
        official_recipe='sam2.1_hiera_tiny512_FLARE_RECIST.yaml',frames=8,max_objects=3,
        gradient_clip_norm=recipe['trainer']['optim']['gradient_clip']['max_norm'],
        epochs=trainer.max_epochs,batch=recipe['scratch']['train_video_batch_size'],
        rss_bytes=memory.rss,peak_rss_bytes=memory.peak_wset if sys.platform=='win32' else None,
        peak_vram_bytes=complete_peak(measurements),
        world_size=allocation['world'],batch_layout=allocation['shares'],ranks=measurements,reason=reason,
        data_workers=allocation['supply']['workers'],prefetch_factor=actual_prefetch)
    if rank==0:
        feedback.close()
        write_json(root/'summary.json',summary)
        if reason is None:
            record_model_peaks(common,allocation['spec'],allocation['gpus'],allocation['shares'],measurements)
    dist.destroy_process_group()
    optimizer_hook.remove()
