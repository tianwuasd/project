"""真实来源病例的短测：训练、开发集验证、检查点恢复与可选完整预测。

此脚本只运行有限步数，输出不能作为分割效果证据。正式训练另行启动。
示例：python 05_src/smoke_train.py ct_holdG B0 --steps 12 --predict
"""
import argparse
import gc
import hashlib
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
from data_pipeline import ROOT, CODE_ROOT, FOLDS, load_json, save_json, inverse_labels
from prepare_nnunet import configure_paths


def state_hash(network):
    digest = hashlib.sha256()
    for name, value in network.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", choices=list(FOLDS))
    parser.add_argument("method", choices=["B0", "B1"])
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    if not 3 <= args.steps <= 50:
        parser.error("短测限 3..50 步；不得把正式训练隐藏在短测中")
    configure_paths()
    os.environ["nnUNet_n_proc_DA"] = "0"
    import torch
    import nibabel as nib
    from project_trainers import Project1B0, Project1B1
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(2)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    split = load_json(ROOT / "04_data/manifests/splits_v1.json")["folds"][args.fold]
    name = f"Dataset{split['dataset_id']:03d}_{args.fold}"
    data = Path(os.environ["nnUNet_preprocessed"]) / name
    plans = load_json(data / "Project1Plans.json")
    dataset_json = load_json(data / "dataset.json")
    run_id = f"smoke_{args.fold}_{args.method}_seed0_{time.strftime('%Y%m%d_%H%M%S')}"
    out = ROOT / "07_experiments" / run_id
    out.mkdir(parents=True, exist_ok=False)
    os.environ["nnUNet_results"] = str(out / "native_results")
    # nnunetv2.paths 已在 import 时缓存；单独设置训练器输出路径，避免污染正式结果。
    cls = Project1B0 if args.method == "B0" else Project1B1
    report = {"run_id": run_id, "status": "running", "smoke_only": True, "fold": args.fold, "method": args.method, "seed": 0, "steps": args.steps, "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "plan_sha256": hashlib.sha256((data / "Project1Plans.json").read_bytes()).hexdigest()}
    report["code_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((CODE_ROOT / "05_src").glob("*.py"))}
    report["input_record_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["04_data/manifests/case_inventory.json", "04_data/manifests/splits_v1.json", "06_configs/conda_explicit_training.txt", "06_configs/pip_freeze_training.txt"]}
    report["split"] = split
    save_json(out / "run.json", report)
    started = time.perf_counter()
    try:
        trainer = cls(plans, "3d_fullres", 0, dataset_json)
        trainer.output_folder_base = str(out)
        trainer.output_folder = str(out / "fold_0")
        trainer.log_file = str(out / "training.log")
        trainer.initialize()
        report["initial_weight_sha256"] = state_hash(trainer.network)
        report["patch_size"] = list(map(int, trainer.configuration_manager.patch_size))
        report["batch_size"] = trainer.batch_size
        trainer.on_train_start()
        trainer.on_epoch_start()
        trainer.on_train_epoch_start()
        torch.cuda.reset_peak_memory_stats()
        outputs, times = [], []
        for step in range(args.steps):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            batch = next(trainer.dataloader_train)
            value = trainer.train_step(batch)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - tick)
            if not np.isfinite(value["loss"]).all():
                raise FloatingPointError("训练 loss 非有限数")
            outputs.append(value)
            print(f"{run_id} step {step+1}/{args.steps}: {times[-1]:.3f}s loss={float(value['loss']):.5f}", flush=True)
        trainer.on_train_epoch_end(outputs)
        trainer.on_validation_epoch_start()
        tick = time.perf_counter()
        with torch.no_grad():
            val_outputs = [trainer.validation_step(next(trainer.dataloader_val)) for _ in range(2)]
        torch.cuda.synchronize()
        val_seconds = (time.perf_counter()-tick)/2
        if not all(np.isfinite(v["loss"]).all() for v in val_outputs):
            raise FloatingPointError("开发集 loss 非有限数")
        trainer.on_validation_epoch_end(val_outputs)
        trainer.on_epoch_end()
        checkpoint = out / "fold_0/checkpoint_smoke.pth"
        trainer.save_checkpoint(str(checkpoint))
        trained_hash = state_hash(trainer.network)
        report.update({"train_seconds_excluding_first_two": times[2:], "median_step_seconds": float(np.median(times[2:])), "validation_step_seconds": val_seconds, "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30, "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30, "train_loss_finite": True, "val_loss_finite": True})
        # 释放原网络后建立新实例，实际加载权重与优化器，不仅检查文件存在。
        del outputs, trainer
        gc.collect()
        torch.cuda.empty_cache()
        restored = cls(plans, "3d_fullres", 0, dataset_json)
        restored.output_folder_base, restored.output_folder = str(out), str(out / "fold_0")
        restored.log_file = str(out / "restore.log")
        restored.load_checkpoint(str(checkpoint))
        assert state_hash(restored.network) == trained_hash, "恢复权重不一致"
        report["checkpoint_weights_equal"] = True
        report["optimizer_state_restored"] = bool(restored.optimizer.state)
        assert report["optimizer_state_restored"]
        resumed_step = restored.train_step(batch)
        assert np.isfinite(resumed_step["loss"]).all(), "恢复后的训练步骤失败"
        report["one_resumed_training_step_passed"] = True
        report["actual_optimizer_steps_including_resume"] = args.steps + 1
        del batch
        if args.predict:
            case_id = split["val"][0]
            raw = Path(os.environ["nnUNet_raw"]) / name
            reader = restored.plans_manager.image_reader_writer_class()
            image_path = raw / "imagesTr" / f"{case_id}_0000.nii.gz"
            images, properties = reader.read_images([str(image_path)])
            restored.set_deep_supervision_enabled(False)
            predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=False, device=torch.device("cuda"), verbose=False, allow_tqdm=False)
            predictor.manual_initialization(restored.network, restored.plans_manager, restored.configuration_manager, [restored.network.state_dict()], dataset_json, cls.__name__, None)
            tick = time.perf_counter()
            segmentation = predictor.predict_single_npy_array(images, properties, None, None, False)
            pred_path = out / f"{case_id}_smoke_prediction.nii.gz"
            reader.write_seg(inverse_labels(segmentation), str(pred_path), properties)
            pred, original = nib.load(pred_path), nib.load(image_path)
            assert pred.shape == original.shape and np.allclose(pred.affine, original.affine, atol=1e-4, rtol=0)
            # nnU-Net 的通用写出器不复制全部 NIfTI 头；恢复原图的单位等元数据。
            header = original.header.copy()
            header.set_data_dtype(np.uint16)
            output_image = nib.Nifti1Image(np.asanyarray(pred.dataobj).astype(np.uint16), original.affine, header)
            nib.save(output_image, pred_path)
            assert nib.load(pred_path).header.get_xyzt_units() == original.header.get_xyzt_units()
            report["prediction"] = {"source_val_case": case_id, "seconds": time.perf_counter()-tick, "original_grid_restored": True, "path": str(pred_path), "not_a_performance_result": True}
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["total_seconds"] = time.perf_counter()-started
        save_json(out / "run.json", report)
        print(f"REPORT: {out / 'run.json'}", flush=True)


if __name__ == "__main__":
    main()
