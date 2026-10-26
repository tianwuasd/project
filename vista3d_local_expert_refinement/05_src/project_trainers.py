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

    def train_step(self, batch):
        from contextlib import nullcontext
        timing = getattr(self, '_vista_timing', None)
        with timing.measure('train_step') if timing else nullcontext():
            result = super().train_step(batch)
        if not np.isfinite(result["loss"]).all():
            raise FloatingPointError("训练损失出现 NaN/Inf，停止并保留此前检查点")
        return result

    def validation_step(self, batch):
        from contextlib import nullcontext
        timing = getattr(self, '_vista_timing', None)
        with timing.measure('validation_step') if timing else nullcontext():
            result = super().validation_step(batch)
        if not np.isfinite(result["loss"]).all():
            raise FloatingPointError("开发集损失出现 NaN/Inf")
        return result

    def save_checkpoint(self, filename):
        # 原子替换：写出中断时保留上一份完整检查点。
        temporary = str(filename) + ".tmp"
        distributed_state=None
        if hasattr(self,'_vista_world_size') and not self.disable_checkpointing:
            from rank_checkpoint import capture
            distributed_state=capture(self)
        super().save_checkpoint(temporary)
        if self.local_rank == 0 and not self.disable_checkpointing:
            checkpoint=torch.load(temporary,map_location='cpu',weights_only=False)
            if distributed_state is not None:
                checkpoint['vista_rank_states']=distributed_state
            else:
                from checkpoint_rng import capture
                checkpoint['vista_rng_state']=capture()
            torch.save(checkpoint,temporary)
            os.replace(temporary, filename)
        if distributed_state is not None:
            import torch.distributed as dist
            dist.barrier()

    def load_checkpoint(self, filename_or_checkpoint):
        from checkpoint_compat import load_parent_checkpoint
        checkpoint=load_parent_checkpoint(super().load_checkpoint,filename_or_checkpoint,torch)
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
        self._vista_timing = EpochTiming()
        self.dataloader_train = TimedIterator(self.dataloader_train, self._vista_timing, 'train')
        self.dataloader_val = TimedIterator(self.dataloader_val, self._vista_timing, 'validation')

    def on_epoch_start(self):
        for name in ['dataloader_train','dataloader_val']:
            loader=getattr(self,name,None)
            if hasattr(loader,'set_epoch'):loader.set_epoch(self.current_epoch)
        if getattr(self, '_vista_timing', None):
            self._vista_timing.start(self.current_epoch)
        super().on_epoch_start()

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
