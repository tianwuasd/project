"""对已规划的真实 nnU-Net 架构做合成输入显存检查，不读取目标病例。"""
import gc
import time
from pathlib import Path
from data_pipeline import ROOT, load_json, save_json
from prepare_nnunet import configure_paths


def main():
    configure_paths()
    import torch
    from project_trainers import Project1B0
    torch.set_num_threads(2)
    torch.manual_seed(0)
    data = ROOT / "04_data/derived/nnUNet_preprocessed/Dataset703_ct_holdG"
    plans, dataset = load_json(data / "Project1Plans.json"), load_json(data / "dataset.json")
    trainer = Project1B0(plans, "3d_fullres", 0, dataset)
    folder = ROOT / "07_experiments/gpu_probe"
    folder.mkdir(parents=True, exist_ok=True)
    trainer.output_folder_base = str(folder)
    trainer.output_folder = str(folder)
    trainer.log_file = str(folder / "probe.log")
    trainer.initialize()
    shape = list(map(int, trainer.configuration_manager.patch_size))
    batch = trainer.batch_size
    image = torch.randn(batch, 1, *shape)
    labels = torch.randint(0, 8, (batch, 1, *shape)).float()
    targets = [torch.nn.functional.interpolate(labels, size=[round(s*f) for s, f in zip(shape, scale)], mode="nearest").long() for scale in trainer._get_deep_supervision_scales()]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train_step({"data": image, "target": targets})
    torch.cuda.synchronize()
    report = {"status": "passed", "synthetic_only": True, "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "patch": shape, "batch": batch, "loss": float(result["loss"]), "seconds_including_warmup": time.perf_counter()-started, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
    assert torch.isfinite(torch.tensor(report["loss"]))
    save_json(ROOT / "09_reports/gpu_probe.json", report)
    print(report)


if __name__ == "__main__":
    main()
