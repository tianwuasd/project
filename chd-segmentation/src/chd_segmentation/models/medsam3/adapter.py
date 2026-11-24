from contextlib import nullcontext
from types import MethodType

import numpy as np
import torch
import yaml
from accelerate import init_empty_weights
from PIL import Image
from torchvision.ops import nms
from torch.nn import functional as F

from .weights import import_official


def input_transform(official):
    return official.ComposeAPI(transforms=[
        official.RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
        official.ToTensorAPI(), official.NormalizeAPI(mean=[.5]*3, std=[.5]*3)])


def lora_sdpa(self, query, key, value, key_padding_mask=None, need_weights=False,
              attn_mask=None, average_attn_weights=True, is_causal=False):
    if need_weights or is_causal:
        raise ValueError('医学图像LoRA只支持非因果注意力且不导出注意力矩阵')
    if not self.batch_first:
        query, key, value = (x.transpose(0, 1) for x in (query, key, value))
    batch, target, _ = query.shape
    source = key.shape[1]
    q, k, v = (projection(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
               for projection, x, length in ((self.q_proj, query, target),
                                            (self.k_proj, key, source), (self.v_proj, value, source)))
    mask = attn_mask
    if mask is not None:
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            if mask.shape[0] == batch:
                mask = mask[:, None]
            elif mask.shape[0] == batch*self.num_heads:
                mask = mask.reshape(batch, self.num_heads, target, source)
            else:
                raise ValueError('LoRA注意力掩膜批次维度无效')
        elif mask.ndim != 4:
            raise ValueError('LoRA注意力掩膜维度无效')
        if mask.dtype == torch.bool:
            mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(mask, float('-inf'))
    if key_padding_mask is not None:
        padding = torch.zeros_like(key_padding_mask, dtype=q.dtype).masked_fill(key_padding_mask, float('-inf'))
        mask = padding[:, None, None] if mask is None else mask+padding[:, None, None]
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
        dropout_p=self.dropout if self.training else 0.)
    output = self.out_proj(output.transpose(1, 2).contiguous().view(batch, target, self.embed_dim))
    return (output if self.batch_first else output.transpose(0, 1)), None


class MedSAM3Adapter:
    def __init__(self, config, *, training=False):
        self.config = config
        official, lora, builder = import_official(config)
        self.official = official
        self.device = torch.device(config['device'])
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('请求的CUDA设备不可用')
        if config['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError('当前CUDA设备不支持BF16')
        with open(config['official_config'], encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)['lora']
        if raw['rank'] != 16 or raw['alpha'] != 32:
            raise ValueError('医学LoRA配置不是固定的v1配方')
        # 使用本地基座及词表，显式禁止官方构建器联网下载。
        with init_empty_weights(include_buffers=False):
            model = builder.build_sam3_image_model(
                device='cpu', checkpoint_path=None, load_from_HF=False,
                bpe_path=config['bpe_path'], eval_mode=not training, compile=False)
        state = torch.load(config['base_weights'], map_location=self.device, weights_only=True)
        state = state['model'] if 'model' in state else state
        detector = {key.removeprefix('detector.'): value for key, value in state.items()
                    if key.startswith('detector.')}
        expected_base = set(model.state_dict())
        extra = set(detector)-expected_base
        if any(not key.startswith('backbone.vision_backbone.sam2_convs.') for key in extra):
            raise ValueError('原生基座包含未知图像模型参数')
        detector = {key:value for key,value in detector.items() if key in expected_base}
        model.load_state_dict(detector, strict=True, assign=True)
        del detector, state
        if any(p.is_meta for p in model.parameters()):
            raise ValueError('原生基座参数没有完整加载')
        settings = {key: value for key, value in raw.items() if key != 'dropout'}
        model = lora.apply_lora_to_model(model, lora.LoRAConfig(**settings, dropout=raw['dropout'] if training else 0.))
        for module in model.modules():
            if isinstance(module, lora.MultiheadAttentionLoRA):
                module.forward = MethodType(lora_sdpa, module)
        weights = torch.load(config['lora_weights'], map_location='cpu', weights_only=True)
        expected = {key for key in model.state_dict() if key.endswith(('.lora_A', '.lora_B'))}
        if not expected or set(weights) != expected:
            raise ValueError('LoRA权重键缺失或包含不匹配结构')
        loaded = model.load_state_dict(weights, strict=False)
        if loaded.unexpected_keys or any(key in expected for key in loaded.missing_keys):
            raise ValueError('LoRA没有完整加载')
        del weights
        self.model = model.to(self.device).train(training)
        self.transform = input_transform(official)
        # 复用官方Datapoint构造，不通过保存图片或更改官方源码传入切片。
        self.datapoint = official.SAM3LoRAInference.create_datapoint

    @torch.inference_mode()
    def predict(self, rgb, boundary):
        image = Image.fromarray(rgb)
        height, width = rgb.shape[:2]
        labels = np.zeros((height, width), dtype=np.uint8)
        scores = np.zeros((height, width), dtype=np.float32)
        counts = {}
        for class_id, prompt in sorted(self.config['prompts'].items()):
            boundary()
            point = self.datapoint(self, image, [prompt])
            batch = self.official.collate_fn_api([self.transform(point)], dict_key='input')['input']
            batch = self.official.copy_data_to_device(batch, self.device, non_blocking=False)
            precision = (torch.autocast('cuda', dtype=torch.bfloat16)
                         if self.config['precision'] == 'bf16' else nullcontext())
            with precision:
                result = self.model(batch)[-1]
            confidence = result['pred_logits'].sigmoid()[0].amax(dim=-1)
            if not torch.isfinite(confidence).all():
                raise ValueError('MedSAM3检测分数包含非有限值')
            keep = confidence > self.config['detection_threshold']
            instances = 0
            if keep.any():
                boxes = result['pred_boxes'][0, keep]
                cx, cy, w, h = boxes.unbind(-1)
                boxes = torch.stack([(cx - w / 2) * width, (cy - h / 2) * height,
                                     (cx + w / 2) * width, (cy + h / 2) * height], dim=-1)
                confidence = confidence[keep].float()
                indices = nms(boxes.float(), confidence, self.config['nms_iou_threshold'])
                masks = result['pred_masks'][0, keep][indices].sigmoid() > .5
                masks = F.interpolate(masks[None].float(), size=(height, width),
                                      mode='bilinear', align_corners=False)[0] > .5
                for mask, score in zip(masks.cpu().numpy(), confidence[indices].cpu().tolist()):
                    replace = mask & (score > scores)
                    labels[replace] = class_id
                    scores[replace] = score
                instances = len(indices)
            counts[str(class_id)] = instances
            del result, batch, point
        return labels.T.copy(),counts
