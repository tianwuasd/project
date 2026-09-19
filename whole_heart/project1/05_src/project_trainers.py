"""两组共用同一个 nnU-Net 训练器；B1 只额外加入外观增强。"""
import copy
from pathlib import Path
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from appearance import BiasField, MonotoneBezier
from data_pipeline import ROOT, load_json


class Project1B0(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        # nnU-Net 2.8 的官方入口注入此字段；直接调用类时也必须补齐。
        # 深拷贝避免父类 pop 改坏供 B0/B1 和恢复流程共用的原始配置。
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
        self.print_to_log_file(f"Frozen source split: {len(expected['train'])} train / {len(expected['val'])} val; target excluded")
        return expected["train"], expected["val"]

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy, initial, _ = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        self.inference_allowed_mirroring_axes = None
        return rotation, dummy, initial, None

    def plot_network_architecture(self):
        # 结构在 plans.json 中完整保存；短测不调用额外的 Graphviz 可执行程序。
        pass


class Project1B1(Project1B0):
    @staticmethod
    def get_training_transforms(*args, **kwargs):
        base = Project1B0.get_training_transforms(*args, **kwargs)
        return ComposeTransforms(list(base.transforms) + [RandomTransform(BiasField(0.25), apply_probability=0.30), RandomTransform(MonotoneBezier(1000), apply_probability=0.25)])
