"""服务器数据定位与校验：病例划分保持不变，只重建本机文件路径。"""
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    """小型状态文件原子替换，避免中断留下半截 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def resolve_cases(data_root, cases):
    data_root = Path(data_root).expanduser().resolve()
    if not data_root.is_dir() or data_root == Path(data_root.anchor):
        raise ValueError("请提供数据集目录，不要提供整个磁盘/根目录或压缩包文件")
    # 文件可位于原始中心子目录，也可调整层级；病例文件名必须保留。
    expected = {f"{c['case_id']}_{kind}.nii.gz" for c in cases for kind in ["image", "label"]}
    found = {}
    for path in data_root.rglob("Case*.nii.gz"):
        if path.name not in expected:
            continue
        if path.name in found:
            raise ValueError(f"找到重复文件名 {path.name}：请指定更精确的数据目录")
        found[path.name] = path.resolve()
    missing = sorted(expected - found.keys())
    if missing:
        raise ValueError(f"缺少 {len(missing)} 个文件，例如：{', '.join(missing[:6])}")
    mapped = []
    for case in cases:
        item = dict(case)
        for kind in ["image", "label"]:
            path = found[f"{case['case_id']}_{kind}.nii.gz"]
            if digest(path) != case[f"{kind}_sha256"]:
                raise ValueError(f"文件校验不符：{path}。请使用原始发布文件，不自动更改划分或标签。")
            item[kind] = str(path)
        mapped.append(item)
        print(f"校验通过：{case['case_id']} ({case['status']})", flush=True)
    return mapped


def bind_workspace(code_root, work_root, data_root, gpu_memory):
    code_root, work_root, data_root = map(lambda p: Path(p).expanduser().resolve(), [code_root, work_root, data_root])
    if work_root == code_root or work_root in code_root.parents or work_root.is_relative_to(data_root) or data_root.is_relative_to(work_root):
        raise ValueError("结果目录必须独立于原始数据目录，不能直接使用项目代码根目录")
    metadata = code_root / "04_data/manifests"
    config = {"data_root": str(data_root), "work_root": str(work_root), "gpu_memory_gb": gpu_memory, "inventory_template_sha256": digest(metadata / "case_inventory.json"), "split_template_sha256": digest(metadata / "splits_v1.json")}
    config_path = work_root / "00_admin/server_config.json"
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError("已有工作区的数据路径、划分或显存规划不同；请使用新的 --work-dir，避免混合实验")
    if work_root.exists() and not config_path.exists() and any(work_root.iterdir()):
        # 启动锁是本入口唯一允许预先创建的文件。
        if any(p.name != ".run.lock" for p in work_root.iterdir()):
            raise ValueError("结果目录不是空目录，也不是本项目已有服务器工作区")
    template = read_json(metadata / "case_inventory.json")
    cases = resolve_cases(data_root, template["cases"])
    logical_splits = read_json(metadata / "splits_v1.json")
    write_json(config_path, config)
    inventory = dict(template, cases=cases, source_code_root=str(code_root), path_relocated=True)
    manifest_path = work_root / "04_data/manifests/case_inventory.json"
    write_json(manifest_path, inventory)
    write_json(work_root / "04_data/manifests/quarantine.json", [c for c in cases if c["status"] == "quarantined"])
    logical_splits["inventory_sha256"] = digest(manifest_path)
    write_json(work_root / "04_data/manifests/splits_v1.json", logical_splits)
    return config
