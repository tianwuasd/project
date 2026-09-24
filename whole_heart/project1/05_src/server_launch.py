"""Linux 启动脚本的中文向导；本模块顶层只导入标准库，先设置工作区再导入训练模块。"""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
SCOPES = {"ct-first": ["ct_holdG"], "mr-first": ["mr_holdE"], "all": ["ct_holdG", "ct_holdA", "ct_holdB", "mr_holdE", "mr_holdCD"]}


def child(script, *arguments):
    """参数列表直接调用，不经过 shell，数据目录中的空格或符号不会变成命令。"""
    command = [sys.executable, "-B", "-X", "utf8", str(CODE_ROOT / "05_src" / script), *map(str, arguments)]
    print(f"\n执行阶段：{script} {' '.join(map(str, arguments))}", flush=True)
    subprocess.run(command, check=True, cwd=CODE_ROOT)


def runtime_check(work, gpu_memory):
    import importlib.metadata as metadata
    import torch
    from server_data import write_json
    expected = {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "nnunetv2": "2.8.1"}
    actual = {name: metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(f"依赖版本不符：{actual}；请使用 start_server.sh 建立的独立环境")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 无法使用 CUDA。请检查 NVIDIA 驱动与 CUDA_VISIBLE_DEVICES；不自动退回 CPU")
    torch.set_num_threads(2)
    free, total = torch.cuda.mem_get_info()
    if free / 2**30 < gpu_memory + 0.5:
        raise RuntimeError(f"可用显存只有 {free/2**30:.1f} GiB，无法使用 {gpu_memory:g} GiB 的规划预算；请选择空闲 GPU 或新工作区中的较低预算")
    net = torch.nn.Conv3d(1, 8, 3, padding=1).cuda()
    value = net(torch.randn(1, 1, 24, 24, 24, device="cuda")).float().square().mean()
    value.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(value):
        raise RuntimeError("GPU 计算出现非有限值")
    del net, value
    torch.cuda.empty_cache()
    config_dir = work / "06_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (config_dir / "pip_freeze_training.txt").write_text(frozen, encoding="utf-8")
    conda = os.environ.get("WHOLE_HEART_CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise RuntimeError("无法导出 Conda 环境，请从 start_server.sh 进入")
    explicit = subprocess.check_output([conda, "list", "--prefix", sys.prefix, "--explicit"], text=True)
    (config_dir / "conda_explicit_training.txt").write_text(explicit, encoding="utf-8")
    report = {"checked": time.strftime("%Y-%m-%d %H:%M:%S"), "gpu": torch.cuda.get_device_name(), "total_gib": total/2**30, "free_gib": free/2**30, "packages": actual, "gpu_backward_check": "passed", "platform": sys.platform}
    write_json(config_dir / "server_environment.json", report)
    print(f"GPU 检查通过：{report['gpu']}，总显存 {report['total_gib']:.1f} GiB。", flush=True)


def show_status(work):
    from server_data import read_json
    records = sorted((work / "07_experiments/formal").glob("*/run.json"))
    if not records:
        print("还没有正式训练记录。")
    for record in records:
        value = read_json(record)
        print(f"{record.parent.name}: {value['status']}；已完成轮数 {value.get('completed_epochs', '见训练日志')}")
    print(f"输出目录：{work}")


def preflight_signature(work, gpu_memory):
    """独立短测和完整实验共享凭证，避免同版本重复短测。"""
    from server_data import digest
    return {"code": {p.name: digest(p) for p in sorted((CODE_ROOT / "05_src").glob("*.py"))},
            "splits": digest(work / "04_data/manifests/splits_v1.json"),
            "environment": digest(work / "06_configs/pip_freeze_training.txt"),
            "conda": digest(work / "06_configs/conda_explicit_training.txt"),
            "gpu_memory": gpu_memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", nargs="?", help="原始数据目录；支持包含中心子文件夹")
    parser.add_argument("--work-dir", default=str(CODE_ROOT / "server_work"))
    parser.add_argument("--mode", choices=["check", "smoke", "train", "resume", "experiment", "evaluate", "status"])
    parser.add_argument("--scope", choices=list(SCOPES), default="ct-first")
    parser.add_argument("--gpu-memory", type=float, default=None, help="规划显存预算，默认5 GiB；已有工作区沿用已保存值")
    parser.add_argument("--yes-train", action="store_true", help="仅在明确接受长训练时使用，省略输入TRAIN确认")
    args = parser.parse_args()
    work = Path(args.work_dir).expanduser().resolve()
    from server_data import bind_workspace, read_json, write_json
    config_path = work / "00_admin/server_config.json"
    previous = read_json(config_path) if config_path.exists() else {}
    if args.mode == "status":
        show_status(work)
        return
    data = args.data_root or previous.get("data_root")
    if not data:
        data = input("请输入原始数据集目录（可直接粘贴，不需要加引号）：\n> ").strip()
    data = Path(data).expanduser().resolve()
    if not data.is_dir():
        raise ValueError(f"数据目录不存在：{data}")
    if work == CODE_ROOT or work in CODE_ROOT.parents or work.is_relative_to(data) or data.is_relative_to(work):
        raise ValueError("结果目录必须独立于原始数据，且不能直接使用代码目录")
    gpu_memory = args.gpu_memory if args.gpu_memory is not None else previous.get("gpu_memory_gb", 5)
    if not math.isfinite(gpu_memory) or gpu_memory < 2:
        raise ValueError("规划显存预算必须是至少2 GiB的有限数值")
    mode, scope = args.mode, args.scope
    if mode in {"experiment", "evaluate"}:
        scope = "all"
    if mode is None:
        print("\n1 只检查环境与数据\n2 短测 CT+MRI（推荐首次选择）\n3 正式训练：CT 留出G，基线+增强\n4 正式训练：MRI 留出E，基线+增强\n5 正式训练：完整10次\n6 继续上次正式训练\n")
        choice = input("请选择 [默认2]：").strip() or "2"
        if choice not in {"1", "2", "3", "4", "5", "6"}:
            raise ValueError("无效菜单选项")
        mode, scope = {"1": ("check", scope), "2": ("smoke", scope), "3": ("train", "ct-first"), "4": ("train", "mr-first"), "5": ("train", "all"), "6": ("resume", scope)}[choice]
    queue_file = work / "00_admin/last_queue.json"
    if mode == "resume":
        if not queue_file.exists():
            raise ValueError("没有上次正式训练队列；首次运行请选择短测或正式训练")
        scope = read_json(queue_file)["scope"]
    if mode in {"train", "resume", "experiment"}:
        print(f"将运行 {2*len(SCOPES[scope])} 个正式实验；本机旧预算为全10次约155小时，服务器速度需重新实测。")
        if not args.yes_train and input("确定开始长训练请输入 TRAIN，其余输入取消：").strip() != "TRAIN":
            print("已取消，没有启动训练。")
            return
    if shutil.disk_usage(work.parent if work.parent.exists() else CODE_ROOT).free < 10*2**30:
        raise RuntimeError("可用磁盘不足10 GiB，请选择空间更充足的结果目录")
    work.mkdir(parents=True, exist_ok=True)
    os.environ["WHOLE_HEART_WORKSPACE"] = str(work)
    from filelock import FileLock
    # 防止不同代码副本同时写入同一个输出目录。
    with FileLock(str(work / ".run.lock"), timeout=0):
        bind_workspace(CODE_ROOT, work, data, gpu_memory)
        runtime_check(work, gpu_memory)
        if mode == "check":
            print(f"检查完成：106例匹配，按既定规则排除5例；未启动训练。\n记录目录：{work}")
            return
        if mode in {"experiment", "evaluate"}:
            from evaluation import establish_protocol
            establish_protocol()
        if mode == "evaluate":
            child("evaluation.py")
            return
        if mode == "experiment":
            # 第一次完整运行先做短测。版本/环境改变后拒绝复用旧短测凭证。
            evidence = preflight_signature(work, gpu_memory)
            gate = work / "00_admin/full_experiment_preflight.json"
            if gate.exists():
                if read_json(gate) != evidence:
                    raise ValueError("完整实验短测凭证与当前代码/环境不同，请使用新工作区")
            else:
                for fold in ["ct_holdG", "mr_holdE"]:
                    child("server_prepare.py", fold, "--gpu-memory", gpu_memory)
                child("run_smoke_suite.py")
                child("summarize_smoke.py")
                write_json(gate, evidence)
        folds = ["ct_holdG", "mr_holdE"] if mode == "smoke" else SCOPES[scope]
        if mode != "smoke":
            write_json(queue_file, {"scope": scope, "folds": folds, "methods": ["B0", "B1"]})
        # 按方向逐一准备并训练，不并发占用显卡。
        for fold in folds:
            child("server_prepare.py", fold, "--gpu-memory", gpu_memory)
            if mode != "smoke":
                for method in ["B0", "B1"]:
                    extra = ["--resume"] if mode in {"resume", "experiment"} else []
                    child("server_train.py", fold, method, *extra)
        if mode == "smoke":
            child("run_smoke_suite.py")
            child("summarize_smoke.py")
            write_json(work / "00_admin/full_experiment_preflight.json", preflight_signature(work, gpu_memory))
            print(f"短测通过。请查看 {work / '09_reports/短测与预算.md'}")
        elif mode == "experiment":
            child("evaluation.py")
            print(f"完整实验完成，报告：{work / '09_reports/final_evaluation/README.md'}")
        else:
            print("所选训练队列完成。模型已保存；外层目标中心批量评价尚未执行。")
        print(f"所有输出：{work}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断。已完成的数据准备和整轮检查点保留；再次启动选择继续。", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"\n操作停止：{error}\n未自动重试。请保留以上输出用于排查。", file=sys.stderr)
        sys.exit(1)
