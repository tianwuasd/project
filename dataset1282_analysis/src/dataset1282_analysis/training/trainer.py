"""Explicit nnU-Net Trainer adapter; no edits to the installed nnU-Net package."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from ..contracts import read_json, sha256_file
from .marginal import MarginalBatchLoss, observed_confusion


class nnUNetTrainerMarginal(nnUNetTrainer):
    """Single-device interface. DDP requires separately tested global reductions."""
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        plans=deepcopy(plans)
        # The 2.8.1 official run_training entry injects this runtime-only flag.
        plans.setdefault('continue_training',False)
        super().__init__(plans,configuration,fold,dataset_json,device=device)
        if self.is_ddp:
            raise ValueError('Marginal Trainer DDP reduction has not been validated')
        if self.label_manager.has_regions or self.label_manager.num_segmentation_heads!=11:
            raise ValueError('Expected eleven mutually exclusive softmax channels')
        base=Path(self.preprocessed_dataset_folder_base)
        contract=read_json(base/'training_contract.json')
        required={'partial_supervision.json','splits_final.json','nnUNetPlans.json','dataset.json'}
        if set(contract['artifacts'])!=required or any(sha256_file(base/p)!=h for p,h in contract['artifacts'].items()):
            raise ValueError('Training input contract changed')
        self.partial_metadata=read_json(base/'partial_supervision.json')

    def _build_loss(self):
        return MarginalBatchLoss()

    def do_split(self):
        path=Path(self.preprocessed_dataset_folder_base)/'splits_final.json'
        if self.fold!=0 or not path.is_file():
            raise ValueError('Requires explicit frozen patient split, fold 0')
        split=read_json(path)[0]
        if not split['train'] or not split['val'] or set(split['train']) & set(split['val']):
            raise ValueError('Invalid frozen source split')
        if set(split['train']+split['val'])-self.partial_metadata.keys():
            raise ValueError('Missing case supervision metadata')
        return split['train'],split['val']

    def train_step(self,batch):
        self.loss.bind(batch.get('keys'),self.partial_metadata)
        return super().train_step(batch)

    @torch.no_grad()
    def validation_step(self,batch):
        self.loss.bind(batch.get('keys'),self.partial_metadata)
        data=batch['data'].to(self.device,non_blocking=True)
        target=batch['target']
        target=[x.to(self.device) for x in target] if isinstance(target,(list,tuple)) else target.to(self.device)
        with torch.autocast(self.device.type) if self.device.type=='cuda' else nullcontext():
            output=self.network(data)
            loss=self.loss(output,target)
        if isinstance(output,(list,tuple)):
            output,target=output[0],target[0]
        tp,fp,fn=observed_confusion(output,target,self.loss.unknown)
        return dict(loss=loss.detach().cpu().numpy(),tp_hard=tp,fp_hard=fp,fn_hard=fn)

    def perform_actual_validation(self,*args,**kwargs):
        raise RuntimeError('Default whole-volume evaluation mis-scores unknown labels; use a case-aware evaluator')
