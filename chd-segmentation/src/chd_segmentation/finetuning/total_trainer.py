import os
from pathlib import Path

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from chd_segmentation.artifacts.store import read_json
from chd_segmentation.losses.partial import imagechd_loss


class ImageCHDTotalTrainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        settings = dataset_json["imagechd_total_training"]
        self.num_epochs = settings["epochs"]
        self.initial_lr = settings["learning_rate"]
        self.save_every = 1
        self._unknown_by_case = read_json(Path(self.preprocessed_dataset_folder_base) / "unknown_classes.json")
        self._batch_unknown = None

    def _build_loss(self):
        return self._imagechd_loss

    def _imagechd_loss(self, outputs, targets):
        ignore = self.label_manager.ignore_label
        if not isinstance(outputs, (list, tuple)):
            return imagechd_loss(outputs, targets.masked_fill(targets == ignore, -1), self._batch_unknown)
        weights = [2.0 ** -index for index in range(len(outputs))]
        # 保留最粗分割头的梯度连接，兼容双卡训练。
        weights[-1] = 1e-6
        return sum(
            weight * imagechd_loss(output, target.masked_fill(target == ignore, -1), self._batch_unknown)
            for output, target, weight in zip(outputs, targets, weights, strict=True)
        ) / sum(weights)

    def train_step(self, batch):
        self._batch_unknown = [self._unknown_by_case[key] for key in batch["keys"]]
        return super().train_step(batch)

    def validation_step(self, batch):
        self._batch_unknown = [self._unknown_by_case[key] for key in batch["keys"]]
        data = batch["data"].to(self.device, non_blocking=True)
        targets = [target.to(self.device, non_blocking=True) for target in batch["target"]]
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            outputs = self.network(data)
            loss = self.loss(outputs, targets)

        target = targets[0][:, 0]
        predicted = outputs[0].argmax(dim=1)
        valid = target != self.label_manager.ignore_label
        counts = torch.zeros((3, 7), device=target.device, dtype=torch.int64)
        for case_index, unknown in enumerate(self._batch_unknown):
            for label in range(1, 8):
                if label in unknown:
                    continue
                prediction = (predicted[case_index] == label) & valid[case_index]
                reference = (target[case_index] == label) & valid[case_index]
                counts[0, label - 1] += (prediction & reference).sum()
                counts[1, label - 1] += (prediction & ~reference).sum()
                counts[2, label - 1] += (~prediction & reference).sum()
        true_positive, false_positive, false_negative = counts.cpu().numpy()
        return dict(loss=loss.detach().cpu().numpy(), tp_hard=true_positive,
                    fp_hard=false_positive, fn_hard=false_negative)

    def save_checkpoint(self, filename):
        if self.local_rank == 0 and not self.disable_checkpointing:
            pending = str(filename) + ".pending"
            super().save_checkpoint(pending)
            os.replace(pending, filename)

    def perform_actual_validation(self, save_probabilities=False):
        self.print_to_log_file("完整病例验证使用项目的 predict 和 evaluate 入口。")
