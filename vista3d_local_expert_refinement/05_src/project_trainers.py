"""nnU-Net 官方训练方法；固定少标注划分与训练预算。"""
import copy
import os
from pathlib import Path
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from data_pipeline import ROOT, load_json


class BaselineNNUNet(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        # nnU-Net 2.8 的官方入口注入此字段；直接调用类时也必须补齐。
        # 深拷贝避免父类 pop 改坏供训练和恢复流程共用的原始配置。
        plans = copy.deepcopy(plans)
        plans.setdefault("continue_training", False)
        super().__init__(plans, configuration, fold, dataset_json, device)
        modality = dataset_json["channel_names"]["0"]
        self.num_epochs = 250 if modality == "CT" else 300
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

    def do_split(self):
        # 禁止 nnU-Net 在缺文件/错误折号时自动生成随机划分。
        if self.fold != 0:
            raise ValueError("每个中心留出数据集只允许使用预先冻结的内层 fold=0")
        fold_name = self.plans_manager.dataset_name.split("_", 1)[1]
        expected = load_json(ROOT / "04_data/manifests/splits_v1.json")["folds"][fold_name]
        actual = load_json(Path(self.preprocessed_dataset_folder_base) / "splits_final.json")
        if actual != [{"train": expected["train"], "val": expected["val"]}]:
            raise ValueError("nnU-Net 划分与项目冻结清单不一致")
        provenance = load_json(Path(self.preprocessed_dataset_folder_base) / "fingerprint_provenance.json")
        if provenance["fit_ids"] != expected["train"]:
            raise ValueError("指纹来源与训练子集不一致")
        if getattr(self,'_vista_run_kind','formal')=='assessment':
            record=load_json(ROOT/'00_admin/assessment.json')
            if record.get('job_kind')!='assessment':raise ValueError('测评工作区身份缺失')
            scope=record['scope']
            train,val=scope['train_ids'],scope['validation_ids']
            if (not train or not val or not set(train)<=set(expected['train'])
                    or not set(val)<=set(expected['val']) or set(train)&set(val)):
                raise ValueError('测评来源子集不符合冻结划分')
            self.print_to_log_file(f'独立测评来源子集：{len(train)} train / {len(val)} val；非正式效果实验')
            return train,val
        self.print_to_log_file(f"Frozen source split: {len(expected['train'])} train / {len(expected['val'])} val; target excluded")
        return expected["train"], expected["val"]

    def plot_network_architecture(self):
        # 结构在 plans.json 中完整保存；短测不调用额外的 Graphviz 可执行程序。
        pass

    def get_dataloaders(self):
        resource=getattr(self,'_vista_resource_config',None)
        if resource is None:return super().get_dataloaders()
        from nnunet_batches import build_loaders
        return build_loaders(self,resource)

    def _do_i_compile(self):
        # Upstream uses this flag for both network compilation and Dice weights.
        # Keep its exact eager loss contract, including DDP's nonzero last weight.
        # Our explicit network-only compilation is controlled independently.
        return False

    def _execution_step(self, batch, validation=False):
        from contextlib import nullcontext
        import time
        from training_execution import EagerRetryRequired, preflight_compile
        report = getattr(self, '_vista_execution', None)
        if report and not validation:
            # Pause speculative supply before compilation consumes its CPU share.
            # Keep this already-owned input; pause rewinds only source position.
            if not report.get('preflight_complete'):
                loader = getattr(self, 'dataloader_train', None)
                if hasattr(loader, 'pause'):
                    # Do not discard the current logical batch: preflight only
                    # needs worker quiescence, not outer transfer rewind.
                    source = getattr(loader, 'source', loader)
                    from batch_transfer import BatchTransfer
                    if isinstance(source, BatchTransfer): source.quiesce()
                    elif hasattr(source, 'pause'): source.pause()
                preflight_compile(self, batch, report)
                if self.device.type == 'cuda': torch.cuda.reset_peak_memory_stats(self.device)
        timing = getattr(self, '_vista_timing', None)
        phase = 'validation_step' if validation else 'train_step'
        started, ended = None, None
        if self.device.type == 'cuda':
            started = torch.cuda.Event(enable_timing=True)
            ended = torch.cuda.Event(enable_timing=True)
            started.record()
        host_started = time.perf_counter()
        try:
            with timing.measure(phase) if timing else nullcontext():
                result = super().validation_step(batch) if validation else self._finite_train_step(batch)
        except Exception as error:
            if report and report.get('compile_applied'):
                reason = f'{type(error).__name__}: {error}'
                report['runtime_failure'] = reason
                self.print_to_log_file('Compiled runtime failed; group abort and eager checkpoint retry required: '+reason)
                raise EagerRetryRequired(reason) from error
            raise
        if not np.isfinite(result['loss']).all():
            raise FloatingPointError('训练或开发损失出现 NaN/Inf')
        if report is not None:
            name = 'validation_step_host_seconds' if validation else 'train_step_host_seconds'
            report[name] = report.get(name,0.) + time.perf_counter()-host_started
        if started is not None:
            ended.record(); ended.synchronize()
            if report is not None:
                name = 'validation_compute_seconds' if validation else 'train_compute_seconds'
                report[name] = report.get(name, 0.) + started.elapsed_time(ended)/1000
        loader = getattr(self, 'dataloader_val' if validation else 'dataloader_train', None)
        if hasattr(loader, 'mark_consumed'): loader.mark_consumed()
        return result

    def _finite_train_step(self, batch):
        """Pinned AMP backoff; every rank skips together and consumes this attempt."""
        from contextlib import nullcontext
        self._vista_last_optimizer_step=False
        for name in ('_vista_update_progress','_vista_epoch_update_progress'):
            if not hasattr(self,name):setattr(self,name,dict(attempted=0,successful=0,amp_skipped=0))
            getattr(self,name)['attempted']+=1
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        target = [x.to(self.device, non_blocking=True) for x in target] if isinstance(target,list) else target.to(self.device,non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else nullcontext():
            output = self.network(data)
            loss = self.loss(output, target)
        if not self._all_ranks_finite(torch.isfinite(loss).all()):
            raise FloatingPointError('训练损失出现 NaN/Inf（任一rank）')
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            finite=torch.stack([torch.isfinite(p.grad).all() for p in self.network.parameters() if p.grad is not None]).all()
            if self._all_ranks_finite(finite):
                # Preserve pinned clipping, including finite gradients whose norm
                # overflows. GradScaler checks gradients during unscale, not norm.
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),12,error_if_nonfinite=False)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self._vista_last_optimizer_step=True
            else:
                # Public scaler APIs only. update resets per-optimizer stages;
                # loading canonical backoff state also resets growth tracking.
                state=self.grad_scaler.state_dict()
                if getattr(self,'is_ddp',False):
                    import torch.distributed as dist
                    states=[state];dist.broadcast_object_list(states,src=0);state=states[0]
                state['scale']*=state['backoff_factor'];state['_growth_tracker']=0
                self.grad_scaler.update(new_scale=state['scale'])
                self.grad_scaler.load_state_dict(state)
                self.print_to_log_file(f"AMP overflow: all ranks skipped optimizer update; attempt consumed; scale={state['scale']}")
        else:
            loss.backward()
            finite=torch.stack([torch.isfinite(p.grad).all() for p in self.network.parameters() if p.grad is not None]).all()
            if not self._all_ranks_finite(finite):raise RuntimeError('non-finite non-AMP gradients on a rank')
            norm=torch.nn.utils.clip_grad_norm_(self.network.parameters(),12,error_if_nonfinite=False)
            if not self._all_ranks_finite(torch.isfinite(norm)):raise RuntimeError('non-finite non-AMP gradient norm on a rank')
            self.optimizer.step()
            self._vista_last_optimizer_step=True
        counter='successful' if self._vista_last_optimizer_step else 'amp_skipped'
        for name in ('_vista_update_progress','_vista_epoch_update_progress'):getattr(self,name)[counter]+=1
        return {'loss':loss.detach().cpu().numpy()}

    def _all_ranks_finite(self, finite):
        flag=finite.to(device=self.device,dtype=torch.int32)
        if getattr(self,'is_ddp',False):
            import torch.distributed as dist
            dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def train_step(self, batch):
        return self._execution_step(batch)

    def validation_step(self, batch):
        return self._execution_step(batch, validation=True)

    def save_checkpoint(self, filename):
        import time
        started = time.perf_counter()
        if self.disable_checkpointing:
            if self.local_rank == 0: self.print_to_log_file('No checkpoint written, checkpointing is disabled')
            return
        # Quiesce outer pin/copy AND CPU queues before RNG capture.
        for name in ('dataloader_train', 'dataloader_val'):
            loader = getattr(self, name, None)
            if hasattr(loader, 'pause'): loader.pause()
        temporary = str(filename) + ".tmp"
        distributed_state=None
        if hasattr(self,'_vista_world_size') and not self.disable_checkpointing:
            from rank_checkpoint import capture
            distributed_state=capture(self)
        if self.local_rank == 0 and not self.disable_checkpointing:
            from torch._dynamo import OptimizedModule
            module = self.network.module if self.is_ddp else self.network
            if isinstance(module, OptimizedModule): module = module._orig_mod
            # Exact pinned2.8.1 fields, assembled once before the only save.
            checkpoint = dict(network_weights=module.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                grad_scaler_state=self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                logging=self.logger.get_checkpoint(), _best_ema=self._best_ema,
                current_epoch=self.current_epoch+1, init_args=self.my_init_kwargs,
                trainer_name=self.__class__.__name__,
                inference_allowed_mirroring_axes=self.inference_allowed_mirroring_axes)
            checkpoint['vista_update_progress']=dict(getattr(self,'_vista_update_progress',{}))
            checkpoint['vista_update_progress_origin_known']=getattr(self,'_vista_progress_origin_known',True)
            if distributed_state is not None:
                checkpoint['vista_rank_states']=distributed_state
            else:
                from checkpoint_rng import capture
                checkpoint['vista_rng_state']=capture()
            try:
                with open(temporary, 'wb') as stream:
                    torch.save(checkpoint,stream)
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary, filename)
            finally:
                if os.path.exists(temporary): os.unlink(temporary)
        if distributed_state is not None:
            import torch.distributed as dist
            dist.barrier()
        report = getattr(self, '_vista_execution', None)
        if report is not None:
            report['checkpoint_seconds'] = report.get('checkpoint_seconds',0.) + time.perf_counter()-started

    def load_checkpoint(self, filename_or_checkpoint):
        from checkpoint_compat import load_parent_checkpoint
        checkpoint=load_parent_checkpoint(super().load_checkpoint,filename_or_checkpoint,torch)
        self._vista_update_progress=dict(checkpoint.get('vista_update_progress') or dict(attempted=0,successful=0,amp_skipped=0))
        self._vista_progress_origin_known=bool(checkpoint.get('vista_update_progress')) and checkpoint.get('vista_update_progress_origin_known',True)
        self._vista_resume_rank_states=checkpoint.get('vista_rank_states')
        if self._vista_resume_rank_states is not None:
            if getattr(self,'_vista_inference_only',False):
                self._vista_resume_rank_states=None
                self.rng_state_restored=False
                return
            if not hasattr(self,'_vista_world_size'):
                raise ValueError('协同检查点必须通过协同训练入口恢复')
            from rank_checkpoint import restore
            restore(self,self._vista_resume_rank_states)
            self.rng_state_restored=True
            return
        self._vista_resume_rng=checkpoint.get('vista_rng_state')
        self.rng_state_restored=bool(self._vista_resume_rng)
        if self._vista_resume_rng:
            from checkpoint_rng import restore
            restore(self._vista_resume_rng)

    def on_train_start(self):
        super().on_train_start()
        if getattr(self,'_vista_resume_rank_states',None):
            from rank_checkpoint import restore
            restore(self,self._vista_resume_rank_states,loaders=True)
            self._vista_resume_rank_states=None
        # Initialization/loading pools may consume random draws after load_checkpoint.
        if getattr(self,'_vista_resume_rng',None):
            from checkpoint_rng import restore
            restore(self._vista_resume_rng)
            self._vista_resume_rng=None
        from training_timing import EpochTiming, TimedIterator
        from batch_transfer import BatchTransfer
        self._vista_timing = EpochTiming()
        if getattr(self, '_vista_resource_config', None) is not None:
            self.dataloader_train = BatchTransfer(self.dataloader_train, self.device)
            self.dataloader_val = BatchTransfer(self.dataloader_val, self.device)
        self.dataloader_train = TimedIterator(self.dataloader_train, self._vista_timing, 'train')
        self.dataloader_val = TimedIterator(self.dataloader_val, self._vista_timing, 'validation')

    def on_epoch_start(self):
        self._vista_epoch_update_progress=dict(attempted=0,successful=0,amp_skipped=0)
        for name in ['dataloader_train','dataloader_val']:
            loader=getattr(self,name,None)
            if hasattr(loader,'set_epoch'):loader.set_epoch(self.current_epoch)
        if getattr(self, '_vista_timing', None):
            self._vista_timing.start(self.current_epoch)
        super().on_epoch_start()

    def on_train_epoch_start(self):
        loader = getattr(self, 'dataloader_val', None)
        if hasattr(loader, 'pause'): loader.pause()
        super().on_train_epoch_start()

    def on_validation_epoch_start(self):
        loader = getattr(self, 'dataloader_train', None)
        if hasattr(loader, 'pause'): loader.pause()
        super().on_validation_epoch_start()

    def on_epoch_end(self):
        super().on_epoch_end()
        timing = getattr(self, '_vista_timing', None)
        if timing and timing.active:
            record = timing.finish()
            rank = getattr(self, 'local_rank', 0)
            error = timing.publish(Path(self.output_folder) / f'training_timing.rank{rank}.jsonl', record)
            if rank == 0:
                phases = record['phases']
                def seconds(name):
                    return phases.get(name, {}).get('sum_seconds', 0.)
                self.print_to_log_file(
                    f"分段耗时 epoch={record['epoch']}: 训练取批/增强等待 {seconds('train_batch_wait'):.2f}s；"
                    f"训练步骤（含传输和计算）{seconds('train_step'):.2f}s；"
                    f"验证取批 {seconds('validation_batch_wait'):.2f}s；验证步骤 {seconds('validation_step'):.2f}s")
                if error and not getattr(self, '_vista_timing_io_warning', False):
                    self.print_to_log_file('计时文件写入未完成（不影响已保存检查点）：'+error)
                    self._vista_timing_io_warning = True

    def on_train_end(self):
        from nnunet_batches import close_loaders
        close_loaders(self)
        # The parent uses isinstance to close multiprocessing augmenters.
        # Restore the real objects so observation cannot hide pool cleanup.
        from training_timing import TimedIterator
        for name in ['dataloader_train', 'dataloader_val']:
            loader = getattr(self, name, None)
            if isinstance(loader, TimedIterator):
                setattr(self, name, loader.source)
        super().on_train_end()
