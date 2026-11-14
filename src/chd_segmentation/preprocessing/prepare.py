from pathlib import Path

from ..artifacts.store import digest, file_digest, operation_lock, read_json, require_output, write_json
from ..datasets.manifest import LABELS, load_manifest
from ..datasets.splits import apply_split
from .planning import resolve_plan, validate_auto_plan


def prepare(config, output):
    manifest = apply_split(load_manifest(config["manifest"], config["dataset"], config["scope"],
                                        allow_unassigned=config["split"]["path"] is not None), config)
    if config["scope"] == "formal" and config["planning"]["mode"] == "existing" and manifest.get("plan_scope") != "formal":
        raise ValueError("正式任务需要声明正式拟合计划的来源")
    rows = manifest["cases"]
    if config["scope"] == "formal" and manifest.get("split_source", {}).get("unit", "patient") == "patient":
        by_patient = {}
        for row in rows:
            if not row.get("patient"):
                raise ValueError("正式划分缺少已核实患者身份")
            roles = set(row["roles"]) - {"predict"}
            previous = by_patient.setdefault(row["patient"], set())
            previous.update(roles)
            if len(previous) > 1:
                raise ValueError("同患者出现在不同数据角色中")
    if any(len(set(row["roles"]) - {"predict"}) > 1 for row in rows) and not config["technical"]["allow_role_overlap"]:
        raise ValueError("病例角色重叠未获得技术范围声明")
    output = require_output(output, config["work_root"], [manifest["data_root"], Path(__file__).parents[3],
                            config["plan"], config["manifest"], config["pretrained"]])
    plan = resolve_plan(config, manifest)
    if len(plan["configurations"]["3d_fullres"]["spacing"]) != 3:
        raise ValueError("需要三维预处理计划")
    if not plan.get("foreground_intensity_properties_per_channel"):
        raise ValueError("缺少训练拟合的归一化统计")
    if config["scope"] == "formal" and plan.get("scope") == "technical":
        raise ValueError("正式任务不能使用技术短测计划")
    dataset_json = {"channel_names": {"0": "CT"},
                    "labels": {name: index for index, name in enumerate(LABELS[config["dataset"]])},
                    "file_ending": ".nii.gz"}
    record = {"schema": 1, "scope": config["scope"], "dataset": config["dataset"], "fold": config["fold"],
              "manifest": manifest, "plan": plan, "dataset_json": dataset_json,
              "source_identity": {"manifest": file_digest(config["manifest"]), "plan": file_digest(config["plan"])},
              "technical": config["technical"]}
    record["identity"] = digest(record)
    with operation_lock(output.parent):
        if output.exists():
            if read_json(output) != record:
                raise ValueError("已有预处理描述的身份不匹配")
        else:
            write_json(output, record)
    return record


def load_prepared(path):
    record = read_json(path)
    identity = record.pop("identity")
    if digest(record) != identity:
        raise ValueError("预处理描述内容摘要不符")
    record["identity"] = identity
    return record


def validate_prepared_config(record, config):
    for key in ("dataset", "scope", "fold", "technical"):
        if record[key] != config[key]:
            raise ValueError("当前配置与预处理描述不一致：" + key)
    for key in ("manifest", "plan"):
        if file_digest(config[key]) != record["source_identity"][key]:
            raise ValueError("当前输入内容与预处理描述不一致：" + key)
    source = record["manifest"].get("split_source")
    if source is not None:
        if config["split"]["path"] != source["path"] or config["fold"] != source["fold"] or \
                config["split"]["id_field"] != source["id_field"] or file_digest(source["path"]) != source["sha256"]:
            raise ValueError("数据划分文件或选折设置已改变，请重新准备")
        unit = config["split"]["unit"]
        if unit != "auto" and unit != source["unit"]:
            raise ValueError("划分独立性设置已改变，请重新准备")
        fmt = config["split"]["format"]
        if fmt != "auto" and fmt != source["format"]:
            raise ValueError("划分格式设置已改变，请重新准备")
    elif config["split"]["path"] is not None:
        raise ValueError("当前预处理描述未使用所选划分")
    if config.get("planning", {}).get("mode") == "auto":
        validate_auto_plan(record["plan"], config, record["manifest"])
    for row in record["manifest"]["cases"]:
        for kind, expected in row["source_stamp"].items():
            stat = Path(row[kind]).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                raise ValueError("原始文件在准备后发生变化：" + kind)
