from copy import deepcopy
from pathlib import Path, PurePosixPath

from .records import RELEASE, read_json, signature, write_json


def portable_manifest(manifest, config):
    result = deepcopy(manifest)
    result.pop("source", None)
    result["portable_schema"] = 1
    for row in result["cases"]:
        root = config[row["dataset"] + "_root"]
        row["expected_bytes"] = {key: row["source_stat"][key]["bytes"] for key in ("image", "label")}
        for key in ("image", "label"):
            row[key] = Path(row[key]).resolve().relative_to(root).as_posix()
        del row["source_stat"]
    return result


def bound_manifest(portable, config):
    if portable["portable_schema"] != 1 or portable["author_release"] != RELEASE or portable["schema"] != 1:
        raise ValueError("输入迁移格式或作者版本不匹配")
    result = deepcopy(portable)
    del result["portable_schema"]
    seen, aliases, roles = set(), {}, {}
    for row in result["cases"]:
        if row["case_id"] in seen or row["dataset"] not in {"aaoca", "imagechd"}:
            raise ValueError("迁移病例重复或数据集未知")
        seen.add(row["case_id"])
        if row["role"] not in {"candidate_train", "candidate_validation", "predict"}:
            raise ValueError("迁移清单含未知角色")
        if row["dataset"] == "imagechd" and row["role"] != "predict":
            raise ValueError("ImageCHD在本工程仅用于预测候选输入")
        patient = (row["dataset"], row["patient_key"])
        if patient in roles and roles[patient] != row["role"]:
            raise ValueError("同一患者不可分配到不同角色")
        roles[patient] = row["role"]
        for alias in row["aliases"]:
            if alias in aliases and aliases[alias] != row["patient_key"]:
                raise ValueError("迁移患者存在未解决的别名冲突")
            aliases[alias] = row["patient_key"]
        root = config[row["dataset"] + "_root"].resolve(strict=True)
        row["source_stat"] = {}
        for key in ("image", "label"):
            relative = PurePosixPath(row[key])
            if relative.is_absolute() or ".." in relative.parts or ":" in row[key] or "\\" in row[key]:
                raise ValueError("迁移输入必须使用数据根内的相对路径")
            target = (root / relative).resolve(strict=True)
            if not target.is_relative_to(root):
                raise ValueError("迁移输入超出数据根")
            stamp = signature(target)
            if stamp["bytes"] != row["expected_bytes"][key]:
                raise ValueError("迁移文件大小与冻结登记不一致")
            row[key], row["source_stat"][key] = str(target), stamp
        del row["expected_bytes"]
    return result


def save_frozen(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError("目标已经冻结了不同输入，请使用独立任务目录")
        return True
    write_json(path, value)
    return False


def export_inputs(config, manifest, server_root=None):
    target = config["project_root"] / "private" / "server-inputs" / "manifest.json"
    reused = save_frozen(target, portable_manifest(manifest, config))
    settings = read_json(Path(__file__).resolve().parents[2] / "configs" / "zmic.example.json")
    for key in ("seed", "candidate_train", "candidate_validation", "holdout_prediction", "roi_mm",
                "offset_index_mm", "imagechd_orientation_policy", "smoke_cases"):
        settings[key] = config[key]
    config_path = target.with_name("project.server.json")
    if config_path.exists():
        read_json(config_path)
        config_reused = True
    else:
        if server_root is None or not PurePosixPath(server_root).is_absolute():
            raise ValueError("首次导出需要--server-root指定Linux服务器的绝对根目录")
        for key in ("state_root", "aaoca_root", "aaoca_pairs", "imagechd_root", "imagechd_manifest"):
            relative = PurePosixPath(settings[key]).relative_to("/srv/research")
            settings[key] = str(PurePosixPath(server_root) / relative)
        config_reused = save_frozen(config_path, settings)
    return {"inputs": str(target), "cases": len(manifest["cases"]), "reused": reused,
            "config": str(config_path), "config_reused": config_reused,
            "scope": "私有相对路径清单；不复制影像或诊断表，不得上传公开仓库"}


def bind_inputs(config, path):
    manifest = bound_manifest(read_json(path), config)
    manifest["source"] = {"portable_manifest": signature(path)}
    target = config["project_root"] / "private" / "manifest.json"
    reused = save_frozen(target, manifest)
    return {"manifest": str(target), "cases": len(manifest["cases"]), "reused": reused,
            "scope": "显式首次迁移仅核对路径与大小并登记本机状态；不验证文件内容摘要，不改变角色或诊断"}
