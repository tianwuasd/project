"""Fixed server operations. No shell commands or credentials in argv."""
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from .service_state import CODE_ROOT, private_dir, read_json, redact, write_json, write_private

BUILDS = {"cpu", "cu118", "cu126", "auto"}


def python_path(workspace):
    return Path(workspace) / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def runtime_env(workspace):
    # Preserve inherited scheduler CUDA visibility; never expand GPU access.
    result = os.environ.copy()
    result.update(TOTALSEG_HOME_DIR=str(Path(workspace) / "private" / "totalseg"),
                  TOTALSEG_WEIGHTS_PATH=str(Path(workspace) / "weights"),
                  PYTHONUNBUFFERED="1", PYTHONUTF8="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    result["PATH"] = str(python_path(workspace).parent) + os.pathsep + result.get("PATH", "")
    return result


def hardware(workspace):
    gpus = []
    processes = []
    error = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if result.returncode:
            error = "nvidia-smi 查询失败"
        for line in result.stdout.splitlines() if not result.returncode else []:
            values = [v.strip() for v in line.split(",")]
            if len(values) == 7:
                gpus.append(dict(zip(("index", "uuid", "name", "driver", "used_mb", "total_mb", "utilization"), values)))
    except (OSError, subprocess.TimeoutExpired):
        error = "未检测到 NVIDIA 查询工具"
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.splitlines() if result.returncode == 0 else []:
            values = [v.strip() for v in line.split(',')]
            if len(values) == 3:
                processes.append(dict(zip(('gpu_uuid', 'pid', 'memory_mb'), values)))
    except (OSError, subprocess.TimeoutExpired):
        pass
    disk_path = Path(workspace)
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    disk = shutil.disk_usage(disk_path)
    return {"system": platform.platform(), "python": platform.python_version(),
            "cpu_count": os.cpu_count(), "disk_free_gib": round(disk.free / 2**30, 1),
            "gpus": gpus, "compute_processes": processes, "gpu_error": error,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "未设置；请按服务器分配选择")}


def choose_build(requested, host):
    if requested not in BUILDS:
        raise ValueError("不支持的 PyTorch 安装方案")
    if requested != "auto":
        return requested
    gpus = host["gpus"]
    if not gpus:
        return "cpu"
    drivers = []
    for gpu in gpus:
        match = re.match(r"(\d+)\.(\d+)", gpu["driver"])
        if not match:
            raise ValueError("无法识别驱动版本，请在面板选择安装方案")
        drivers.append(tuple(map(int, match.groups())))
    # Conservative full CUDA toolkit driver support; actual allocation is checked later.
    if min(drivers) >= (560, 35):
        return "cu126"
    if min(drivers) >= (520, 61):
        return "cu118"
    raise ValueError("驱动较旧：可选择 CPU 安装，或由管理员适配驱动；本面板不会修改驱动")


PROBE = '''
import json
from aaoca_seg.pipeline import environment
info = environment()
info['devices'] = []
info['cuda_available'] = False
try:
    import torch
    info['cuda_available'] = torch.cuda.is_available()
    info['torch_cuda'] = torch.version.cuda
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        info['devices'].append({'index': i, 'name': p.name, 'uuid': str(getattr(p, 'uuid', '未知')), 'total_gib': round(p.total_memory / 2**30, 1)})
except Exception:
    info['cuda_error'] = 'PyTorch CUDA 探测失败，请检查驱动和安装版本'
print(json.dumps(info, ensure_ascii=False))
'''


def probe(workspace):
    executable = python_path(workspace)
    if not executable.is_file():
        raise ValueError("尚未安装独立环境，请先点击安装环境")
    result = subprocess.run([str(executable), "-c", PROBE], cwd=CODE_ROOT,
                            env=runtime_env(workspace), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=90)
    if result.returncode:
        raise ValueError("环境检查失败：" + redact(result.stderr[-1800:]))
    info = json.loads(result.stdout.strip().splitlines()[-1])
    write_json(Path(workspace) / "environment.json", info)
    return info


def check_device(config, info):
    if config["device"] == "cpu":
        return
    if not config.get("gpu_confirmed"):
        raise ValueError("请先确认所选 GPU 已分配给你，并保存设置")
    index = int(config["device"].split(":")[1])
    if not info.get("cuda_available") or index not in {d["index"] for d in info.get("devices", [])}:
        raise ValueError("所选 GPU 不在当前可见且可用的设备中，请重新检查环境")
    selected = next(d for d in info["devices"] if d["index"] == index)
    if config.get("gpu_uuid") != selected["uuid"]:
        raise ValueError("GPU 映射已变化，请重新选择并确认设备")


def checked(command, workspace, timeout=None):
    result = subprocess.run(command, cwd=CODE_ROOT, env=runtime_env(workspace), timeout=timeout)
    if result.returncode:
        raise ValueError(f"安装步骤失败（退出码 {result.returncode}），查看上方日志后重试")


def install(workspace, config):
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise ValueError("自动安装使用 Python 3.11–3.13；请用该范围内的 Python 重新启动面板")
    build = choose_build(config["torch_build"], hardware(workspace))
    print(f"创建独立环境；PyTorch 2.7.1 / {build}，TotalSegmentator 2.18.0", flush=True)
    checked([sys.executable, "-m", "venv", str(Path(workspace) / "venv")], workspace)
    executable = str(python_path(workspace))
    constraints = Path(workspace) / "constraints.txt"
    write_private(constraints, "torch==2.7.1\ntorchvision==0.22.1\nTotalSegmentator==2.18.0\n")
    checked([executable, "-m", "pip", "install", "--upgrade", "pip"], workspace)
    checked([executable, "-m", "pip", "install", "--force-reinstall", "torch==2.7.1", "torchvision==0.22.1",
             "--index-url", f"https://download.pytorch.org/whl/{build}"], workspace)
    checked([executable, "-m", "pip", "install", "-c", str(constraints),
             "-r", str(CODE_ROOT / "requirements-inference.txt")], workspace)
    checked([executable, "-m", "pip", "check"], workspace)
    write_json(Path(workspace) / "installation.json", {"torch_build": build, "torch": "2.7.1", "totalsegmentator": "2.18.0"})
    info = probe(workspace)
    print("依赖安装完成。下一步验证许可并检查输入。", flush=True)
    return info


def stage_license(workspace, value):
    if not isinstance(value, str) or not re.fullmatch(r"aca_[A-Za-z0-9]{14}", value.strip()):
        raise ValueError("许可格式应为 aca_ 开头、共 18 个字符；请核对复制内容")
    write_private(Path(workspace) / "private" / "license.input", value.strip())


def apply_license(workspace):
    source = Path(workspace) / "private" / "license.input"
    if not source.is_file():
        raise ValueError("请先在面板输入许可码，或导入私有配置")
    secret = source.read_text(encoding="utf-8").strip()
    private_dir(Path(workspace) / "private" / "totalseg")
    script = '''
import sys
from totalsegmentator.config import setup_totalseg, set_license_number
setup_totalseg()
set_license_number(sys.stdin.read().strip())
'''
    # stdin is a private pipe: no license appears in argv, job JSON or stdout.
    result = subprocess.run([str(python_path(workspace)), "-c", script], input=secret,
                            cwd=CODE_ROOT, env=runtime_env(workspace), capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=60)
    if result.returncode:
        raise ValueError("官方许可验证未通过或服务器网络不可用；请检查许可与网络后重试")
    source.unlink()
    config = Path(workspace) / "private" / "totalseg" / "config.json"
    if os.name != "nt":
        config.chmod(0o600)
    print("许可已由官方验证并保存到此工作区的私有配置。", flush=True)
    return probe(workspace)
