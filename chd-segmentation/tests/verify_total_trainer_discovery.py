import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def child():
    from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
    from chd_segmentation.finetuning.total_trainer import ImageCHDTotalTrainer

    # 同一进程重复发现及独立进程加载均须返回原训练器类。
    for _ in range(2):
        found = recursive_find_trainer_class_by_name("ImageCHDTotalTrainer")
        assert found is ImageCHDTotalTrainer
        assert found.__module__ == "chd_segmentation.finetuning.total_trainer"
    assert "total" not in sys.modules
    print(json.dumps({"pid": os.getpid(), "trainer": found.__module__, "lookups": 2}))


def main():
    parser = argparse.ArgumentParser(description="真实nnU-Net训练器发现；双CPU进程、零训练更新、无测试文件")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if args.child:
        child()
        return
    from chd_segmentation.finetuning.total import environment, options

    config = options(ROOT / "configs" / "finetuning" / "totalsegmentator.yaml")
    env = environment(config)
    folder = Path(env["nnUNet_extTrainer"])
    assert folder == ROOT / "src/chd_segmentation/finetuning/nnunet_trainers"
    assert {p.name for p in folder.glob("*.py")} == {"ImageCHDTotalTrainer.py"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = perf_counter()
    processes = []
    try:
        for _ in range(2):
            processes.append(subprocess.Popen(
                [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), "--child"],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8"))
        for process in processes:
            output, _ = process.communicate(timeout=120)
            print(output, end="")
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args, output=output)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
    print(json.dumps({"state": "passed", "processes": 2, "lookups": 4,
                      "training_updates": 0, "gpu_used": False, "generated_files": 0,
                      "seconds": perf_counter() - start,
                      "memory_peak": "未测量", "vram_peak": "未测量"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
