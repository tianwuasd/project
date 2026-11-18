from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

from ...datasets.manifest import LABELS
from ...training.resources import host_capacity
from ...training.checkpoint import tensors_equal
from . import UPSTREAM_COMMIT


def activate(config, *, inference=False):
    root = Path(config['official_root']).resolve(strict=True)
    if not inference:
        sha = subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
        if sha!=UPSTREAM_COMMIT:
            raise ValueError('MedSAM2官方提交不匹配')
        changed = subprocess.run(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],check=True,capture_output=True,text=True).stdout
        if changed:
            raise ValueError('MedSAM2官方源码包含修改')
    sys.path.insert(0,str(root))
    if config['dependency_root']:
        sys.path.insert(0,str(Path(config['dependency_root']).resolve(strict=True)))
    import sam2
    if not inference and Path(sam2.__file__).resolve()!=root/'sam2/__init__.py':
        raise ValueError('MedSAM2导入来源不匹配')
    return root


class FrameSequence:
    def __init__(self,image,resolution,window,reserve_ram_bytes=4*2**30,windowed=False):
        from .data import window_image
        self.image,self.resolution = image[0],resolution
        if host_capacity()['available_bytes'] < reserve_ram_bytes+self.image.nbytes*3:
            raise MemoryError('MedSAM2灰度统计超出主机内存预算')
        if windowed:
            if self.image.dtype != np.uint8:
                raise ValueError('已完成灰度窗的序列必须为uint8')
        else:
            self.image = window_image(self.image,window)

    def __len__(self):
        return len(self.image)

    def __getitem__(self,index):
        plane = torch.from_numpy(np.array(self.image[index],copy=True)).float()
        value = (plane/255.)[None,None]
        value = F.interpolate(value,size=(self.resolution,)*2,mode='bilinear',align_corners=False)[0].repeat(3,1,1)
        mean = value.new_tensor([.485,.456,.406])[:,None,None]
        std = value.new_tensor([.229,.224,.225])[:,None,None]
        return (value-mean)/std


class MedSAM2Adapter:
    def __init__(self,config,prepared,stage,initialize=True):
        if stage!='sequence':
            raise ValueError('MedSAM2阶段必须为sequence')
        root = activate(config, inference=not initialize)
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        self.config,self.prepared = config,prepared
        self.device = torch.device(config['device'])
        official = OmegaConf.load(root/'sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml')
        model_config = OmegaConf.to_container(official.trainer.model,resolve=False)
        model_config['image_size'] = config['resolution']
        self.spec = deepcopy(model_config)
        self.model = instantiate(OmegaConf.create(model_config),_recursive_=True).to(self.device)
        if config['freeze_image_encoder']:
            self.model.image_encoder.requires_grad_(False)
        if config['pretrained'] and initialize:
            source = torch.load(config['pretrained'],map_location='cpu',weights_only=True)
            self.model.load_state_dict(source['model'],strict=True)
        elif initialize and not config['technical']:
            raise ValueError('MedSAM2正式训练需要登记官方预训练权重')

    def predict(self,entry,prompts,label_box=False):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        if not prompts or entry['case_id'] not in prompts:
            raise ValueError('MedSAM2预测必须提供人工提示或图像定位器的提示文件')
        items = prompts[entry['case_id']]
        if not items:
            raise ValueError('病例提示列表为空')
        model_config = deepcopy(self.spec)
        model_config['_target_'] = 'sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ'
        for key in list(model_config):
            if key.startswith(('prob_to_','num_frames_to_','num_init_cond_','rand_frames_','rand_init_','num_correction_','pt_sampling_','use_act_ckpt_','forward_backbone_','freeze_image_encoder')):
                del model_config[key]
        model_config['fill_hole_area'] = 0
        predictor = instantiate(OmegaConf.create(model_config),_recursive_=True).to(self.device).eval()
        predictor.load_state_dict(self.model.state_dict(),strict=True)
        from .native import resolve_window
        window = resolve_window(self.config,self.prepared)
        if entry['gray_window'] != window:
            raise ValueError('MedSAM2预测窗口与离线数据不同')
        image = FrameSequence(entry['image'],self.config['resolution'],window,
            self.config['execution'].get('reserve_ram_bytes',4*2**30),windowed=True)
        depth,height,width = entry['image'].shape[1:]
        state = predictor.init_state(image,height,width,offload_video_to_cpu=True,offload_state_to_cpu=True)
        classes = len(LABELS[self.prepared['dataset']])
        for item in items:
            permitted = ('label_box',) if label_box else (
                ('manual','image_localizer','technical_simulated') if self.config['technical'] else ('manual','image_localizer'))
            if item['source'] not in permitted:
                raise ValueError('正式预测提示来源必须是人工或图像定位器')
            if item['space']!='native_dhw' or not 0<=item['frame']<depth or not 0<item['class_id']<classes:
                raise ValueError('MedSAM2提示坐标或类别无效')
            box = item['box_xyxy']
            if len(box)!=4 or not all(np.isfinite(box)) or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height:
                raise ValueError('MedSAM2提示框超出输入网格')
            if item['source']=='image_localizer' and not item.get('checkpoint_sha256'):
                raise ValueError('自动提示必须登记图像定位器检查点摘要')
            if not label_box:
                predictor.add_new_points_or_box(state,item['frame'],item['class_id'],box=np.asarray(box,dtype=np.float32))
        scores = torch.full((classes,depth,height,width),-1024.,dtype=torch.float32)
        scores[0] = 0
        propagation_steps = []
        if label_box:
            for item in items:
                for reverse in (False,True):
                    completed = 0
                    predictor.reset_state(state)
                    predictor.add_new_points_or_box(state,item['frame'],item['class_id'],
                        box=np.asarray(item['box_xyxy'],dtype=np.float32))
                    for index,ids,logits in predictor.propagate_in_video(state,reverse=reverse):
                        scores[item['class_id'],index] = logits[0,0].float().cpu()
                        completed += 1
                    propagation_steps.append(dict(class_id=item['class_id'],reverse=reverse,frames=completed))
        else:
            for reverse in (False,True):
                for index,ids,logits in predictor.propagate_in_video(state,reverse=reverse):
                    for position,class_id in enumerate(ids):
                        scores[class_id,index] = logits[position,0].float().cpu()
        probability = scores.softmax(0).numpy()
        return probability,dict(prompt_sources=[item['source'] for item in items],prompt_space='native_dhw',
                                upstream_commit=UPSTREAM_COMMIT,frames=depth,postprocess_extension=False,
                                propagation_steps=propagation_steps)
