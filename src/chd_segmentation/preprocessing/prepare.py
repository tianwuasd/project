from pathlib import Path

from ..artifacts.store import digest, file_digest, operation_lock, read_json, require_output, write_json
from ..datasets.manifest import LABELS, load_manifest
from ..datasets.splits import apply_split
from ..observability.events import event, logged, stage
from .planning import resolve_plan, validate_auto_plan


@logged("prepare")
def prepare(config, output):
    with stage(config, "manifest_and_split", manifest=config["manifest"], fold=config["fold"]):
        manifest = apply_split(load_manifest(config["manifest"], config["dataset"], config["scope"],
                                            allow_unassigned=config["split"]["path"] is not None), config)
    event(config["run_dir"], "split_ready", cases=len(manifest["cases"]),
          roles={role: sum(role in row["roles"] for row in manifest["cases"])
                 for role in ("train", "validation", "test")})
    if config["scope"] == "formal" and config["planning"]["mode"] == "existing" and manifest.get("plan_scope") != "formal":
        raise ValueError("正式任务需要声明正式拟合计划的来源")
    rows = manifest["cases"]
    if config["scope"] == "formal" and manifest.get("split_source", {}).get("unit", "patient") == "patient":
        by_patient = {}
        for row in rows:
            if not row.get("patient"):
                raise ValueError("正式划分缺少已核实患者身份")
            roles = set(row["roles"]) - {"predict"}
            if roles == {"test"}:
                roles = {row.get("test_group", "test")}
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
    event(config["run_dir"], "prepared_ready", output=str(output), identity=record["identity"],
          cases=len(rows), plan=config["plan"])
    return record


def load_prepared(path):
    record = read_json(path)
    record.setdefault('identity', digest({k: v for k, v in record.items() if k != 'identity'}))
    return record


def validate_prepared_config(record, config):
    if record['dataset'] != config['dataset']:
        raise ValueError('当前配置与预处理数据集不一致')
