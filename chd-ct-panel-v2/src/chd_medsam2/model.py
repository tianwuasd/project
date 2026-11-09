import importlib.metadata
from pathlib import Path
import subprocess

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
import sam2
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.utils.misc import concat_points
from training.model.sam2 import SAM2Train
from training.utils.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData

from .contracts import file_sha


def upstream_root(config):
    upstream = Path(sam2.__file__).resolve().parents[1]
    head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != config['source_revision']:
        raise ValueError('MedSAM2 源码提交与配置不一致：'+head)
    modified = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain',
                                       '--untracked-files=no'], text=True)
    if modified.strip():
        raise ValueError('固定 MedSAM2 官方源码有未提交改动')
    return upstream


def versions():
    names = ('torch', 'torchvision', 'numpy', 'nibabel', 'hydra-core', 'omegaconf', 'tensordict', 'fvcore')
    return {name: importlib.metadata.version(name) for name in names}


def build(config, *, training, device, weights=None, shared_image_position_encoding=False):
    upstream = upstream_root(config)
    directory = upstream/'sam2/configs'
    if training:
        recipe = OmegaConf.load(directory/'sam2.1_hiera_tiny512_FLARE_RECIST.yaml')
        recipe.scratch.resolution = config['image_size']
        model = recipe.trainer.model
        model._target_ = 'chd_medsam2.model.BoxTrain'
        model.num_correction_pt_per_frame = config['correction_points']
        if shared_image_position_encoding:
            model.image_encoder.neck.position_encoding._target_ = 'chd_medsam2.model.SharedImagePositionEncoding'
    else:
        model = OmegaConf.load(directory/'sam2.1_hiera_t512.yaml').model
        model._target_ = 'sam2.sam2_video_predictor_npz.SAM2VideoPredictorNPZ'
        model.image_size = config['image_size']
        model.fill_hole_area = 0
    # 记忆注意力的位置网格跟随真实尺寸；短测不改变任何权重张量形状。
    feature_size = config['image_size']//16
    model.memory_attention.layer.self_attention.feat_sizes = [feature_size]*2
    model.memory_attention.layer.cross_attention.feat_sizes = [feature_size]*2
    network = instantiate(model, _recursive_=True)
    if weights is not None:
        if file_sha(weights) != config['weights_sha256']:
            raise ValueError('MedSAM2 医学预训练权重 SHA256 不匹配')
        network.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True)['model'], strict=True)
    return network.to(device).train(training)


def make_batch(clip, device):
    images = clip['images'].to(device)
    masks = clip['masks'].to(device)
    frames, objects = masks.shape[:2]
    indices = torch.arange(frames, dtype=torch.int32, device=device)[:, None].expand(-1, objects)
    return BatchedVideoDatapoint(img_batch=images[:, None],
                                obj_to_frame_idx=torch.stack((indices, torch.zeros_like(indices)), dim=-1),
                                masks=masks, metadata=BatchedVideoMetaData(
                                    unique_objects_identifier=torch.zeros((frames, objects, 3), dtype=torch.int64, device=device),
                                    frame_orig_size=torch.full((frames, objects, 2), images.shape[-1], device=device),
                                    batch_size=[frames]),
                                dict_key='all', batch_size=[frames])


class SharedImagePositionEncoding(PositionEmbeddingSine):
    @torch.no_grad()
    def forward(self, features):
        # 同一尺度的帧使用同一位置网格，先转换精度，再广播到整个序列。
        grid = super().forward(features[:1]).to(dtype=features.dtype)
        return grid.expand(features.shape[0], -1, -1, -1)


class BoxTrain(SAM2Train):
    def forward(self, input, valid, boxes):
        self._valid_first = valid[0]
        self._boxes = boxes
        try:
            return super().forward(input)
        finally:
            self._valid_first = self._boxes = None

    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        # 一段只在首帧提供框；后续帧使用官方记忆传播。
        points = self._boxes.reshape(-1, 2, 2)
        labels = torch.tensor([2, 3], dtype=torch.int32, device=points.device).expand(len(points), -1)
        backbone_out.update(gt_masks_per_frame={t: masks[:, None] for t, masks in enumerate(input.masks)},
                            num_frames=input.num_frames, use_pt_input=True, init_cond_frames=[0],
                            frames_not_in_init_cond=list(range(1, input.num_frames)), mask_inputs_per_frame={},
                            point_inputs_per_frame={0: {'point_coords': points, 'point_labels': labels}},
                            frames_to_add_correction_pt=[0] if self.num_correction_pt_per_frame else [])
        return backbone_out

    def _iter_correct_pt_sampling(self, is_init_cond_frame, point_inputs, gt_masks, high_res_features,
                                  pix_feat_with_mem, low_res_multimasks, high_res_multimasks, ious,
                                  low_res_masks, high_res_masks, object_score_logits, current_out):
        gt = gt_masks[:, 0].bool()
        valid = self._valid_first
        low_history, high_history = [low_res_masks], [high_res_masks]
        for _ in range(self.num_correction_pt_per_frame):
            predicted = high_res_masks[:, 0] > 0
            errors = gt if self.rng.random() < self.prob_to_sample_from_gt_for_train else gt != predicted
            choices = (errors & valid).flatten(1)
            has_error = choices.any(1)
            weights = torch.where(has_error[:, None], choices, valid.flatten()[None]).float()
            offset = torch.multinomial(weights, 1)
            width = gt.shape[-1]
            coordinates = torch.stack((offset % width, offset // width), dim=-1).float()
            labels = torch.where(has_error[:, None], gt.flatten(1).gather(1, offset).int(), -1)
            point_inputs = concat_points(point_inputs, coordinates, labels)
            sam = self._forward_sam_heads(backbone_features=pix_feat_with_mem, point_inputs=point_inputs,
                                          mask_inputs=low_res_masks, high_res_features=high_res_features,
                                          multimask_output=self._use_multimask(is_init_cond_frame, point_inputs))
            low_multi, high_multi, ious, low_res_masks, high_res_masks, _, object_scores = sam
            low_history.append(low_res_masks)
            high_history.append(high_res_masks)
            current_out['multistep_pred_multimasks'].append(low_multi)
            current_out['multistep_pred_multimasks_high_res'].append(high_multi)
            current_out['multistep_pred_ious'].append(ious)
            current_out['multistep_point_inputs'].append(point_inputs)
            current_out['multistep_object_score_logits'].append(object_scores)
        current_out['multistep_pred_masks'] = torch.cat(low_history, dim=1)
        current_out['multistep_pred_masks_high_res'] = torch.cat(high_history, dim=1)
        return point_inputs, sam
