"""正式训练入口：固定250/300轮，仅来源开发集选模，不评价外层目标中心。"""
import argparse
import os
import random
import time
import traceback
from pathlib import Path
import numpy as np
from data_pipeline import ROOT, CODE_ROOT, FOLDS, load_json
from prepare_nnunet import configure_paths
from server_data import digest, write_json


def select_checkpoint(folder):
    """只从同一正式实验继续，绝不误用短测或 best（可能倒退很多轮）。"""
    folder = Path(folder)
    for name in ["checkpoint_final.pth", "checkpoint_latest.pth"]:
        if (folder / name).is_file():
            return folder / name
    raise FileNotFoundError("尚无完整轮次的检查点；请使用新工作区从头运行，不使用短测权重续训")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fold", choices=list(FOLDS))
    parser.add_argument("method", choices=["B0", "B1"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    configure_paths()
    os.environ["nnUNet_n_proc_DA"] = "0"
    import torch
    from project_trainers import Project1B0, Project1B1
    from smoke_train import state_hash
    from nnunetv2.training.logging.nnunet_logger import MetaLogger
    torch.set_num_threads(2)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    split = load_json(ROOT / "04_data/manifests/splits_v1.json")["folds"][args.fold]
    prep = ROOT / "04_data/derived/nnUNet_preprocessed" / f"Dataset{split['dataset_id']:03d}_{args.fold}"
    out = ROOT / "07_experiments/formal" / f"{args.fold}_{args.method}_seed0"
    record = out / "run.json"
    signature = {"fold": args.fold, "method": args.method, "seed": 0, "plan_sha256": digest(prep / "Project1Plans.json"), "splits_sha256": digest(ROOT / "04_data/manifests/splits_v1.json"), "environment_sha256": digest(ROOT / "06_configs/pip_freeze_training.txt"), "code_sha256": {p.name: digest(p) for p in sorted((CODE_ROOT / "05_src").glob("*.py"))}}
    signature["conda_environment_sha256"] = digest(ROOT / "06_configs/conda_explicit_training.txt")
    previous = load_json(record) if record.exists() else None
    if previous is not None:
        if previous["signature"] != signature:
            raise ValueError("续训的代码、环境、划分或计划已改变；禁止混用旧检查点")
        if previous["status"] == "completed":
            if not (out / "fold_0/checkpoint_final.pth").is_file():
                raise FileNotFoundError("完成记录存在但最终检查点缺失")
            print(f"{args.fold}/{args.method} 已完成，跳过。", flush=True)
            return
        if not args.resume:
            raise FileExistsError("实验已存在，请选择‘断点继续’或新的工作区")
    elif out.exists():
        raise FileExistsError("发现无状态记录的已有实验目录，不自动覆盖")
    checkpoint = select_checkpoint(out / "fold_0") if previous else None
    out.mkdir(parents=True, exist_ok=True)
    report = previous or {"signature": signature, "split": split, "history": [], "formal_training": True}
    report["status"] = "running"
    report["history"].append({"started": time.strftime("%Y-%m-%d %H:%M:%S"), "resumed_from": str(checkpoint) if checkpoint else None})
    write_json(record, report)
    try:
        cls = Project1B0 if args.method == "B0" else Project1B1
        trainer = cls(load_json(prep / "Project1Plans.json"), "3d_fullres", 0, load_json(prep / "dataset.json"))
        trainer.output_folder_base = str(out)
        trainer.output_folder = str(out / "fold_0")
        Path(trainer.output_folder).mkdir(exist_ok=True)
        trainer.log_file = str(out / "training.log")
        trainer.logger = MetaLogger(trainer.output_folder, resume=bool(checkpoint))
        trainer.save_every = 1  # 每轮保存，掉线或中断后最多回退约一轮。
        trainer.initialize()
        initial_hash = state_hash(trainer.network)
        other_method = "B1" if args.method == "B0" else "B0"
        other = out.parent / f"{args.fold}_{other_method}_seed0/run.json"
        if other.exists() and load_json(other).get("initial_weight_sha256") != initial_hash:
            raise ValueError("两组初始权重不同，停止以保护对照")
        report["initial_weight_sha256"] = initial_hash
        report["planned_epochs"] = trainer.num_epochs
        if checkpoint:
            trainer.load_checkpoint(str(checkpoint))
        write_json(record, report)
        # 只执行训练和来源开发集验证；这里不调用目标中心推理或选模。
        if trainer.current_epoch < trainer.num_epochs:
            trainer.run_training()
        if not (out / "fold_0/checkpoint_final.pth").exists():
            raise RuntimeError("未得到最终检查点，不能标记完成")
        report.update(status="completed", completed_epochs=trainer.current_epoch)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        report["history"][-1]["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(record, report)


if __name__ == "__main__":
    main()
