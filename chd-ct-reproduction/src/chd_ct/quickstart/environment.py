"""Probe imports and tensor devices in a subprocess so broken binaries cannot crash the wizard."""

import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGES = {
    "numpy": (1, 26),
    "scipy": (1, 11),
    "torch": (2, 6),
    "nibabel": (5, 2),
    "yaml": (6, 0),
    "skimage": (0, 24),
    "networkx": (3, 2),
}


def probe():
    import re

    report = {
        "ok": True,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {},
        "errors": [],
        "warnings": [],
        "cuda": False,
    }
    if sys.version_info < (3, 11):
        report["errors"].append("Python 需要 >= 3.11")
    for name, minimum in PACKAGES.items():
        try:
            module = importlib.import_module(name)
            version = str(module.__version__)
            match = re.match(r"^(\d+)\.(\d+)", version)
            if match is None or tuple(map(int, match.groups())) < minimum:
                raise ValueError(f"{name} {version} 低于最低版本 {minimum[0]}.{minimum[1]}")
            report["packages"][name] = {"ok": True, "version": version}
        except Exception as error:
            report["packages"][name] = {"ok": False, "error": str(error)}
            report["errors"].append(f"{name}: {error}")
    if report["packages"]["torch"]["ok"]:
        import torch

        try:
            torch.set_num_threads(2)
            x = torch.tensor([1.0, 2.0], requires_grad=True)
            x.square().sum().backward()
            if not torch.isfinite(x.grad).all():
                raise RuntimeError("CPU 梯度运算出现非有限值")
            report["cpu_tensor_test"] = "passed"
        except Exception as error:
            report["errors"].append(f"CPU tensor: {error}")
        try:
            if torch.cuda.is_available():
                torch.ones(2, device="cuda").sum().item()
                torch.cuda.synchronize()
                props = torch.cuda.get_device_properties(0)
                report.update(cuda=True, gpu=props.name, gpu_memory_gib=round(props.total_memory / 2**30, 2))
            else:
                report["warnings"].append("CUDA 不可用；仍可在 CPU 上完成本次小规模测试。")
        except Exception as error:
            report["warnings"].append(f"CUDA 实际运算失败，将使用 CPU：{error}")
    report["ok"] = not report["errors"]
    return report


def check_environment(output):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "chd_ct.quickstart.environment", "--probe"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        if result.returncode:
            raise RuntimeError(result.stderr[-3000:] or f"探测程序退出码 {result.returncode}")
        report = json.loads(result.stdout)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        report = {"ok": False, "errors": [f"环境探测失败：{error}"], "cuda": False}
    report["disk_free_gib"] = round(shutil.disk_usage(Path(output)).free / 2**30, 2)
    if report["disk_free_gib"] < 1:
        report["ok"] = False
        report["errors"].append("输出磁盘剩余空间不足 1 GiB，请换一个输出目录。")
    return report


if __name__ == "__main__":
    print(json.dumps(probe(), ensure_ascii=True))
