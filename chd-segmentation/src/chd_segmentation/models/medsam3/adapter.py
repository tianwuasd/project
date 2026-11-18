from pathlib import Path
import sys

import nibabel as nib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.ops import nms
import yaml


class MedSAM3Adapter:
    def __init__(self, config, device, boundary):
        self.config, self.device, self.boundary = config, torch.device(device), boundary
        backend = config['backend']
        for key in ('base_weights','lora_weights','lora_config','tokenizer'):
            if not Path(backend[key]).is_file():
                raise FileNotFoundError(backend[key])
        sys.path.insert(0,backend['official_root'])
        from sam3.model_builder import build_sam3_image_model
        from infer_sam import SAM3LoRAInference, ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
        from lora_layers import LoRAConfig, apply_lora_to_model
        self.datapoint = SAM3LoRAInference.create_datapoint
        self.model = build_sam3_image_model(device='cpu',compile=False,
            checkpoint_path=None,load_from_HF=False,
            bpe_path=backend['tokenizer'],eval_mode=True)
        base = torch.load(backend['base_weights'],map_location='cpu',weights_only=True)
        if 'model' in base and isinstance(base['model'],dict):
            base = base['model']
        image_state = {key.replace('detector.',''):value for key,value in base.items() if 'detector' in key}
        self.model.load_state_dict(image_state,strict=True)
        del base, image_state
        lora = yaml.safe_load(Path(backend['lora_config']).read_text(encoding='utf-8'))['lora']
        self.model = apply_lora_to_model(self.model,LoRAConfig(**dict(lora,dropout=0.)))
        state = torch.load(backend['lora_weights'],map_location='cpu',weights_only=True)
        expected = {k for k in self.model.state_dict() if '.lora.' in k}
        if set(state) != expected:
            raise ValueError('医学v1 LoRA参数集合与作者配置不一致')
        incompatible = self.model.load_state_dict(state,strict=False)
        if incompatible.unexpected_keys or set(incompatible.missing_keys) & expected:
            raise ValueError('医学LoRA实际state加载失败')
        self.model.to(self.device).eval()
        self.transform = ComposeAPI(transforms=[RandomResizeAPI(sizes=1008,max_size=1008,
            square=True,consistent_transform=False),ToTensorAPI(),NormalizeAPI(mean=[.5]*3,std=[.5]*3)])

    @torch.inference_mode()
    def concept(self, image, prompt):
        from sam3.train.data.collator import collate_fn_api
        from sam3.model.utils.misc import copy_data_to_device
        point = self.transform(self.datapoint(self,image,[prompt]))
        batch = copy_data_to_device(collate_fn_api([point],dict_key='input')['input'],self.device,non_blocking=True)
        output = self.model(batch)[-1]
        scores = output['pred_logits'].sigmoid()[0].max(dim=-1)[0]
        keep = scores > self.config['backend']['detection_threshold']
        if not keep.any():
            return
        boxes = output['pred_boxes'][0,keep]
        cx,cy,w,h = boxes.unbind(-1)
        width,height = image.size
        boxes = torch.stack(((cx-w/2)*width,(cy-h/2)*height,(cx+w/2)*width,(cy+h/2)*height),dim=-1)
        scores = scores[keep]
        selected = nms(boxes,scores,self.config['backend']['nms_iou_threshold'])
        if output.get('pred_masks') is None:
            raise RuntimeError('SAM3未返回分割mask')
        kept_indices = keep.nonzero(as_tuple=True)[0][selected]
        # 沿用作者的二值化、双线性插值和再次二值化顺序。
        for index,score in zip(kept_indices,scores[selected]):
            mask = output['pred_masks'][0,index].sigmoid() > .5
            restored = F.interpolate(mask[None,None].float(),size=(height,width),mode='bilinear',align_corners=False)[0,0] > .5
            yield restored.cpu().numpy(),float(score)

    def predict(self, image, labels, native_uint8=None):
        if native_uint8 is None:
            # 病例额度内只解压一次原始gzip，逐片转换在CPU进行。
            pixels = np.asanyarray(image.dataobj).astype(np.float32,copy=False)
            canonical = nib.as_closest_canonical(nib.Nifti1Image(pixels,image.affine))
            native = None
        else:
            canonical = nib.as_closest_canonical(native_uint8)
            native = canonical
        result = np.zeros(canonical.shape,dtype=np.uint8)
        low,high = self.config['backend']['gray_window']
        prompts = self.config['backend']['class_prompts']
        selected = {name:value for name,value in labels.items() if value}
        if set(selected)-set(prompts):
            raise ValueError('解剖类别缺少文本提示：'+','.join(sorted(set(selected)-set(prompts))))
        detections = {name:0 for name in selected}
        for z in range(canonical.shape[2]):
            self.boundary('slice_started',slice=z+1,slices=canonical.shape[2])
            if native is None:
                plane = np.asarray(canonical.dataobj[:,:,z],dtype=np.float32).T
                if not np.isfinite(plane).all():
                    raise ValueError('CT切片包含非有限强度')
                windowed = np.rint(np.clip((plane-low)/(high-low),0,1)*255).astype(np.uint8)
            else:
                windowed = np.asarray(native.dataobj[:,:,z],dtype=np.uint8).T
            rgb = Image.fromarray(windowed).convert('RGB')
            score_plane = np.zeros(windowed.shape,dtype=np.float32)
            label_plane = np.zeros(windowed.shape,dtype=np.uint8)
            for name,label in selected.items():
                self.boundary('concept_started',slice=z+1,slices=canonical.shape[2],concept=name)
                for mask,score in self.concept(rgb,prompts[name]):
                    detections[name] += 1
                    update = mask & (score > score_plane)
                    score_plane[update],label_plane[update] = score,label
            result[:,:,z] = label_plane.T
        orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine),
            nib.orientations.io_orientation(image.affine))
        result = nib.orientations.apply_orientation(result,orientation)
        return result,dict(covered_classes=list(selected.values()),uncovered_classes=[],detections=detections,
            zero_detection_classes=[name for name,count in detections.items() if count==0],
            lifecycle='base+LoRA initialized once per worker',slice_direction='closest-canonical axial',
            overlap='highest detection confidence; equal confidence keeps first dataset class',
            intensity=dict(source='native uint8' if native is not None else 'manifest HU',gray_window=[low,high],
                conversion='reuse fixed linear uint8' if native is not None else 'per-slice RGB uint8'))
