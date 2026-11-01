"""STU-Net-S with the v2 controller's checkpoint, loss and resource hooks."""
import os
from pathlib import Path
import numpy as np
import torch
from project_trainers import BaselineNNUNet
from stunet_architecture import STUNet


def strides_for(configuration):
    strides=[list(map(int,s)) for s in configuration.pool_op_kernel_sizes[1:]][:5]
    return strides+[[1,1,1]]*(5-len(strides))


class STUNetRandom(BaselineNNUNet):
    def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        if dataset_json['labels']!={'background':0,'LV':1,'RV':2,'LA':3,'RA':4,'MYO':5,'AO':6,'PA':7,'ignore':8}:
            raise ValueError('不是ImageCHD七结构标签契约')
        self.initial_lr=1e-3
        self.num_epochs=250

    @staticmethod
    def build_network_architecture(plans_manager,configuration_manager,num_input_channels,
                                   num_output_channels,enable_deep_supervision=True):
        if num_input_channels!=1 or num_output_channels!=8:
            raise ValueError('STU-Net输入应为单通道、8类输出')
        return STUNet(1,8,depth=[1]*6,dims=[16,32,64,128,256,256],
            pool_op_kernel_sizes=strides_for(configuration_manager),conv_kernel_sizes=[[3]*3]*6,
            enable_deep_supervision=enable_deep_supervision)

    def _get_deep_supervision_scales(self):
        if not self.enable_deep_supervision:return None
        return [list(s) for s in np.vstack([np.ones(3),1/np.cumprod(strides_for(self.configuration_manager),axis=0)])[:5]]


class STUNetPretrained(STUNetRandom):
    def initialize(self):
        if getattr(self,'_vista_inference_only',False):
            # A verified, fine-tuned checkpoint is loaded immediately after
            # construction; the pretraining download is not an inference input.
            return super().initialize()
        evidence_path=Path(self.output_folder)/'pretrained_provenance.json'
        path=os.environ.get('IMAGECHD_STUNET_WEIGHTS')
        if not path and evidence_path.exists():
            from server_data import read_json
            path=read_json(evidence_path).get('local_weight_path')
        if not path:
            raise ValueError('请设置IMAGECHD_STUNET_WEIGHTS指向已校验的作者small_ep4k.model；不允许静默随机初始化')
        super().initialize()
        network=self.network.module if self.is_ddp else self.network
        if hasattr(network,'_orig_mod'):network=network._orig_mod
        from stunet_weights import load_pretrained
        evidence=load_pretrained(network,path)
        evidence['local_weight_path']=str(Path(path).expanduser().resolve())
        self.pretrained_provenance=evidence
        if self.local_rank==0:
            from server_data import write_json
            write_json(Path(self.output_folder)/'pretrained_provenance.json',evidence)
        self.print_to_log_file('STU-Net-S官方骨干已校验加载；8类输出头重新初始化。')
