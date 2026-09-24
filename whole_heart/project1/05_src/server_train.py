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
    raise FileNotFoundError("尚无完整轮次的检查点；不能用短测或best作为续训权重")


def archive_uncheckpointed(out):
    """用户选择继续后，仅重启没有完整轮次检查点的这一模型；保留失败现场。"""
    root = ROOT.resolve()
    out = Path(out).resolve()
    if not out.is_relative_to(root / '07_experiments/formal'):
        raise ValueError('只能归档本工作区的正式实验目录')
    if (root / '08_results/evaluation/frozen_models.json').exists():
        raise ValueError('模型已经冻结，不能重新启动训练')
    if any((out / 'fold_0' / name).exists() for name in ['checkpoint_latest.pth', 'checkpoint_final.pth']):
        raise ValueError('已有完整检查点，应继续而不是重新初始化')
    destination = root / '07_experiments/recovery_archive' / f'{out.name}_{time.time_ns()}'
    if not destination.resolve().is_relative_to(root):
        raise ValueError('归档路径超出当前工作区')
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.rename(destination)
    print(f'本模型尚无完整轮次检查点，失败现场保留于 {destination}；仅该模型从seed0重新开始。', flush=True)
    return str(destination)


def publish_initial_weight(record, report, other, initial_hash):
    """先发布再比较，兼容另一组正在初始化或归档无检查点的失败记录。"""
    report['initial_weight_sha256'] = initial_hash
    write_json(record, report)
    try:
        other_hash = load_json(other).get('initial_weight_sha256')
    except FileNotFoundError:
        other_hash = None
    if other_hash is not None and other_hash != initial_hash:
        raise ValueError('两组初始权重不同，停止以保护对照')


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
    checkpoint, restarted_from = None, None
    if previous:
        try:
            checkpoint = select_checkpoint(out / "fold_0")
        except FileNotFoundError:
            # --resume且签名相符才到这里。一次调用只启动一次，不在失败后循环重试。
            restarted_from = archive_uncheckpointed(out)
            previous = None
    out.mkdir(parents=True, exist_ok=True)
    report = previous or {"signature": signature, "split": split, "history": [], "formal_training": True}
    if restarted_from:
        report['restarted_from_archive'] = restarted_from
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
        # 多卡时两组可能同时初始化：先原子发布自己的哈希，再读取另一组。
        # 后发布的一组一定能核对先发布的值；评价冻结时仍会再次核对两组。
        publish_initial_weight(record, report, other, initial_hash)
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
