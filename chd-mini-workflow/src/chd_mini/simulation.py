from copy import deepcopy
from importlib import import_module
from pathlib import Path
import sys
from time import perf_counter

from .contracts import TOOL_ROOT, file_sha, read_json, verify_engine, write_json


def load_environment(root):
    root = Path(root).resolve(strict=True)
    paths = [root / name for name in ("simulator.py", "common.py", "validate.py", "usage.py", "profiles/zmic44.json")]
    identity = {str(path): file_sha(path) for path in paths}
    profile = read_json(root / "profiles" / "zmic44.json")
    if profile["schema"] != 1 or profile["kind"] != "target_hardware_configuration_not_local_devices":
        raise ValueError("仿真环境配置类型不符")
    sys.path.insert(0, str(root))
    try:
        module = import_module("simulator")
        if Path(module.__file__).resolve() != root / "simulator.py":
            raise ValueError("已加载的仿真模块来自其他目录")
    finally:
        sys.path.pop(0)
    return {"module": module, "profile": profile, "identity": identity, "root": str(root)}


def check_resources(settings, work, logs):
    started = perf_counter()
    baseline = verify_engine(settings["engine_root"], settings["engine_commit"])
    environment = load_environment(settings["simulator_root"])
    simulator, profile = environment["module"], environment["profile"]
    sys.path.insert(0, str(Path(settings["engine_root"]) / "src"))
    from chd_segmentation.config import DEFAULTS
    from chd_segmentation.training.resources import capacity_key, gpu_decision

    scheduler = DEFAULTS, gpu_decision, capacity_key
    directory = Path(logs) / "simulation"
    directory.mkdir(parents=True, exist_ok=True)
    state = {"report": str(directory), "jobs": [], "cpu_slots": profile["cpu"]["logical_processors"],
             "ram_bytes": profile["ram_gib"] * 2 ** 30,
             "gpus": [{"index": i, "uuid": f"SIM-GPU-{i:02d}", "total": profile["gpu"]["vram_mib_each"] * 2 ** 20,
                       "free": profile["gpu"]["vram_mib_each"] * 2 ** 20, "busy": True,
                       "owned": False, "utilization": 0} for i in range(profile["gpu"]["count"])],
             "mappings": {profile["paths"]["code"]: settings["engine_root"],
                          profile["paths"]["work"]: str(Path(work) / "workspace"),
                          profile["paths"]["result"]: str(Path(work) / "workspace" / "results"),
                          "/data4/zhangruichen/code/chd-segmentation-assistant3": str(TOOL_ROOT)}}
    simulator.submit(state, "mini", gpus=2, cpu=4, ram_gib=4, updates=0)
    simulator.submit(state, "competitor", gpus=2, cpu=4, ram_gib=4, updates=0)
    job = state["jobs"][0]
    cases = []

    def decide(name, expected):
        selected = state["jobs"][1] if name in ("competitor_blocked", "released") else job
        actual = simulator.decision(state, selected, scheduler)
        if actual["admitted"] != expected:
            raise ValueError("仿真资源准入结果不符：" + name)
        cases.append({"name": name, "expected_admitted": expected, "actual": actual})
        return actual

    state["cpu_slots"] = 0
    decide("cpu_wait", False)
    state["cpu_slots"] = profile["cpu"]["logical_processors"]
    state["ram_bytes"] = 0
    decide("ram_wait", False)
    state["ram_bytes"] = profile["ram_gib"] * 2 ** 30
    decide("busy_wait", False)
    for gpu in state["gpus"][-2:]:
        gpu["busy"] = False
    admitted = decide("two_virtual_slots", True)
    job.update(state="running", virtual_uuids=[gpu["uuid"] for gpu in admitted["gpus"]])
    decide("competitor_blocked", False)
    job["state"] = "completed"
    decide("released", True)
    resolved = simulator.resolve_path(profile["paths"]["code"] + "/run.py", state)
    if Path(resolved) != Path(settings["engine_root"]) / "run.py":
        raise ValueError("生产项目路径映射不符")
    actual_identity = {path: file_sha(path) for path in environment["identity"]}
    if actual_identity != environment["identity"] or verify_engine(settings["engine_root"], settings["engine_commit"]) != baseline:
        raise ValueError("仿真检查期间依赖来源改变")
    result = {"scope": "virtual_resource_contract_only", "actual_device": "none", "cases": 0,
              "CT_voxels_read": 0, "training_updates": 0, "profile": profile,
              "simulator_identity": actual_identity, "production_commit": settings["engine_commit"],
              "checks": cases, "path_mappings": deepcopy(state["mappings"]),
              "seconds": perf_counter() - started, "ram_peak": "未测量", "vram_peak": "未测量",
              "coverage": "调用既有仿真资源决策与路径映射接口；真实CPU和CUDA计算另行记录"}
    write_json(directory / "resources.json", result)
    return result
