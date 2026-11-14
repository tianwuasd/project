from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import psutil
import torch
import torch.nn.functional as F

from ...datasets.manifest import LABELS
from . import UPSTREAM_COMMIT


def activate(config):
    root = Path(config['official_root']).resolve(strict=True)
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
    if Path(sam2.__file__).resolve()!=root/'sam2/__init__.py':
        raise ValueError('MedSAM2导入来源不匹配')
    return root


class FrameSequence:
    def __init__(self,image,resolution):
        self.image,self.resolution = image[0],resolution
        if psutil.virtual_memory().available < 4*2**30+self.image.nbytes*3:
            raise MemoryError('MedSAM2灰度统计超出主机内存预算')
        self.low,self.high = np.quantile(self.image,[.005,.995])
        if not np.isfinite(self.image).all() or self.high<=self.low:
            raise ValueError('MedSAM2灰度范围无效')

    def __len__(self):
        return len(self.image)

    def __getitem__(self,index):
        plane = torch.from_numpy(np.array(self.image[index],copy=True)).float()
        value = ((plane-self.low)/(self.high-self.low)).clamp(0,1)[None,None]
        value = F.interpolate(value,size=(self.resolution,)*2,mode='bilinear',align_corners=False)[0].repeat(3,1,1)
        mean = value.new_tensor([.485,.456,.406])[:,None,None]
        std = value.new_tensor([.229,.224,.225])[:,None,None]
        return (value-mean)/std


class MedSAM2Adapter:
    def __init__(self,config,prepared,stage,initialize=True):
        if stage!='sequence':
            raise ValueError('MedSAM2阶段必须为sequence')
        root = activate(config)
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from training.loss_fns import MultiStepMultiMasksAndIous
        self.config,self.prepared = config,prepared
        self.device = torch.device(config['device'])
        official = OmegaConf.load(root/'sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml')
        model_config = OmegaConf.to_container(official.trainer.model,resolve=False)
        model_config['image_size'] = config['resolution']
        model_config['num_init_cond_frames_for_train'] = min(2,config['frames']-1)
        self.spec = deepcopy(model_config)
        self.model = instantiate(OmegaConf.create(model_config),_recursive_=True).to(self.device)
        if config['freeze_image_encoder']:
            self.model.image_encoder.requires_grad_(False)
        if config['pretrained'] and initialize:
            source = torch.load(config['pretrained'],map_location='cpu',weights_only=True)
            self.model.load_state_dict(source['model'],strict=True)
        elif initialize and not config['technical']:
            raise ValueError('MedSAM2正式训练需要登记官方预训练权重')
        self.criterion = MultiStepMultiMasksAndIous(dict(loss_mask=20,loss_dice=1,loss_iou=1,loss_class=1),
            supervise_all_iou=True,iou_use_l1_loss=True,pred_obj_scores=True)

    def optimizer(self):
        groups = []
        for name,parameter in self.model.named_parameters():
            if parameter.requires_grad:
                groups.append(dict(params=[parameter],lr=self.config['learning_rate']*(.6 if name.startswith('image_encoder.') else 1.),
                    weight_decay=0. if parameter.ndim==1 or name.endswith('bias') else .1))
        return torch.optim.AdamW(groups)

    def extra_state(self):
        return dict(prompt_rng=deepcopy(self.model.rng.bit_generator.state),upstream_commit=UPSTREAM_COMMIT)

    def restore_extra(self,state):
        if state['upstream_commit']!=UPSTREAM_COMMIT:
            raise ValueError('MedSAM2恢复来源提交不同')
        self.model.rng.bit_generator.state = state['prompt_rng']

    def prepare_training(self,dataset,names):
        pass

    def sample(self,dataset,names,rng):
        return dataset.get(names[int(rng.integers(len(names)))])

    def schedule(self,optimizer,update):
        factor = .1+.9*(1+np.cos(np.pi*update/self.config['updates']))/2
        for group in optimizer.param_groups:
            if 'peak_lr' not in group:
                group['peak_lr'] = group['lr']
            group['lr'] = group['peak_lr']*factor

    def validate(self,dataset,names):
        saved = self.extra_state()
        rng = np.random.default_rng(self.config['seed'])
        values = [float(self.loss(dataset.get(name),rng,0)) for name in names]
        self.restore_extra(saved)
        return float(np.mean(values))

    def loss(self,entry,rng,update):
        from training.utils.data_utils import Frame,Object,VideoDatapoint,collate_fn
        label = entry['label']
        known = sorted(set(map(int,np.unique(label)))-{0,-1}-set(entry['unknown_classes']))
        if not known or (label<0).any():
            raise ValueError('MedSAM2训练序列需要已知目标与完整有效像素范围')
        target = known[int(rng.integers(len(known)))]
        count = min(self.config['frames'],label.shape[0])
        if count<2:
            raise ValueError('MedSAM2序列至少需要两帧')
        occupied = np.flatnonzero((label==target).any((1,2)))
        first = int(occupied[int(rng.integers(len(occupied)))])
        direction = 1 if first+count<=label.shape[0] else -1
        indices = [first+direction*index for index in range(count)]
        if min(indices)<0:
            raise ValueError('MedSAM2连续序列超出体积范围')
        rgb = FrameSequence(entry['image'],self.config['resolution'])
        masks = F.interpolate(torch.from_numpy(label[indices]==target)[:,None].float(),
                              size=(self.config['resolution'],)*2,mode='nearest')[:,0].bool()
        if not masks[0].any():
            raise ValueError('MedSAM2缩放后首帧目标消失')
        video = VideoDatapoint([Frame(rgb[index],[Object(target,index,masks[t])]) for t,index in enumerate(indices)],
                               video_id=0,size=tuple(label.shape[1:]))
        batch = collate_fn([video],dict_key='heart').to(self.device)
        outputs = self.model(batch)
        return self.criterion(outputs,batch.masks)['core_loss']

    def predict(self,entry,prompts):
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
        image = FrameSequence(entry['image'],self.config['resolution'])
        depth,height,width = entry['image'].shape[1:]
        state = predictor.init_state(image,height,width,offload_video_to_cpu=True,offload_state_to_cpu=True)
        classes = len(LABELS[self.prepared['dataset']])
        for item in items:
            permitted = ('manual','image_localizer','technical_simulated') if self.config['technical'] else ('manual','image_localizer')
            if item['source'] not in permitted:
                raise ValueError('正式预测提示来源必须是人工或图像定位器')
            if item['space']!='prepared_dhw' or not 0<=item['frame']<depth or not 0<item['class_id']<classes:
                raise ValueError('MedSAM2提示坐标或类别无效')
            box = item['box_xyxy']
            if len(box)!=4 or not all(np.isfinite(box)) or not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height:
                raise ValueError('MedSAM2提示框超出输入网格')
            if item['source']=='image_localizer' and not item.get('checkpoint_sha256'):
                raise ValueError('自动提示必须登记图像定位器检查点摘要')
            predictor.add_new_points_or_box(state,item['frame'],item['class_id'],box=np.asarray(box,dtype=np.float32))
        scores = torch.full((classes,depth,height,width),-1024.,dtype=torch.float32)
        scores[0] = 0
        for reverse in (False,True):
            for index,ids,logits in predictor.propagate_in_video(state,reverse=reverse):
                for position,class_id in enumerate(ids):
                    scores[class_id,index] = logits[position,0].float().cpu()
        probability = scores.softmax(0).numpy()
        return probability,dict(prompt_sources=[item['source'] for item in items],prompt_space='prepared_dhw',
                                upstream_commit=UPSTREAM_COMMIT,frames=depth,postprocess_extension=False)
