from collections import defaultdict

import torch
from training.loss_fns import MultiStepMultiMasksAndIous


def criterion():
    return MultiStepMultiMasksAndIous(
        weight_dict=dict(loss_mask=20, loss_dice=1, loss_iou=1, loss_class=1),
        supervise_all_iou=True, iou_use_l1_loss=True, pred_obj_scores=True,
        focal_gamma_obj_score=0.0, focal_alpha_obj_score=-1.0)


def masked_loss(loss, outputs, targets, validity, frame_objects):
    # 在官方候选掩膜选择、IoU 及存在性目标之前剔除未知像素。
    if not bool(validity.any()):
        raise ValueError('MedSAM2 批次无有效监督')
    total = defaultdict(float)
    for output, target, valid, objects in zip(outputs, targets, validity, frame_objects, strict=True):
        if not bool(valid.any()):
            continue
        masks = output['multistep_pred_multimasks_high_res']
        if not bool(valid.all()):
            masks = [mask[:, :, valid].unsqueeze(-2) for mask in masks]
            target = target[:, valid].unsqueeze(-2)
        selected = dict(multistep_pred_multimasks_high_res=[m.float() for m in masks],
                        multistep_pred_ious=[m.float() for m in output['multistep_pred_ious']],
                        multistep_object_score_logits=[m.float() for m in output['multistep_object_score_logits']])
        for key, value in loss._forward(selected, target, objects).items():
            total[key] += value
    return total
