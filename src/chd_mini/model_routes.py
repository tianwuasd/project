from pathlib import Path

from .attempts import STAGES
from .contracts import read_json


KINDS = ("prepare", "preprocess", "train", "predict", "evaluate")


def load_acceptance(path):
    path = Path(path).resolve(strict=True)
    settings = read_json(path)
    if settings.get("schema") != 3:
        raise ValueError("三模型技术流程需要schema 3；旧schema 2可只读检查")
    if settings["profile"] != "technical":
        raise ValueError("本轮只支持技术短流程")
    roles = settings["roles"]
    if set(roles) != {"train", "validation", "test"} or roles["test"]:
        raise ValueError("技术角色须明确训练、验证和空测试集合")
    selected = roles["train"] + roles["validation"]
    if not roles["train"] or not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected):
        raise ValueError("技术流程限1–2例且训练与验证病例互斥")
    for key in ("engine_root", "python", "work_root", "report_root"):
        item = Path(settings[key])
        settings[key] = str((path.parent / item).resolve() if not item.is_absolute() else item.resolve())
    for key in ("engine_root", "python"):
        if not Path(settings[key]).exists():
            raise FileNotFoundError(settings[key])
    work, reports, engine = (Path(settings[k]) for k in ("work_root", "report_root", "engine_root"))
    for left, right in ((work, reports), (work, engine), (reports, engine)):
        if left == right or left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("生产、工作和报告目录须独立且不互相包含")
    if settings.get("intensity_mode", "auto") == "normalized_percentile" and not settings.get("percentile_approval"):
        raise ValueError("百分位实验模式须有明确选择记录")
    settings["config_path"] = str(path)
    build_steps(settings)
    return settings


def build_steps(settings):
    steps = settings["steps"]
    if not steps:
        raise ValueError("须在最终生产接口发布后填写实际命令步骤")
    names = set()
    for step in steps:
        if not step["name"].replace("_", "").isalnum() or step["name"] in names:
            raise ValueError("步骤名称无效或重复")
        names.add(step["name"])
        if step["kind"] not in KINDS or not all(isinstance(a, str) for a in step["args"]):
            raise ValueError("步骤类型或命令参数无效")
        if step["kind"] == "train":
            if step["stage"] not in STAGES or step.get("attempt_protocol") != "chd-mini-attempt-v1":
                raise ValueError("训练入口须接入累计尝试协议")
        if step["kind"] == "predict" and step.get("checkpoint") != "best":
            raise ValueError("技术验收预测须明确选择best")
        receipt = Path(step["receipt"])
        if receipt.is_absolute() or ".." in receipt.parts:
            raise ValueError("步骤回执须位于本次报告目录内")
        if type(step["timeout_seconds"]) is not int or not 1 <= step["timeout_seconds"] <= 1200:
            raise ValueError("步骤时间上限须在1–1200秒内")
    return steps


def command_for(settings, step, work, reports):
    replacements = {"{work}": str(work), "{reports}": str(reports)}
    args = list(step["args"])
    for index, value in enumerate(args):
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        args[index] = value
    return [settings["python"], "-u", "-B", str(Path(settings["engine_root"]) / "run.py"), *args]
