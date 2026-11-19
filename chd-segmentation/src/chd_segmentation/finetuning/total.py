import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from importlib.metadata import version
from io import StringIO
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic, sleep

import psutil
import yaml

from chd_segmentation.acceleration import ResourceDemand, ResourceSession, RuntimeRequest
from chd_segmentation.artifacts.store import file_digest, operation_lock, read_json, require_output, write_json
from chd_segmentation.datasets.manifest import LABELS
from chd_segmentation.observability.events import event
from .total_inputs import RAW_EXPORT, export_case


DATASET = "Dataset901_ImageCHDTotal"
PLANS = "ImageCHDTotalPlans"
TRAINER = "ImageCHDTotalTrainer"
CONFIGURATION = "3d_fullres"
IGNORE = 8
VERSIONS = {"TotalSegmentator": "2.18.0", "nnunetv2": "2.8.1"}
CHECKPOINTS = ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth")
PUBLIC_TOTAL_CT = (291, 292, 293, 294, 295, 297, 298, 831, 832, 833, 834, 835, 836, 837)
LICENSED_301 = "Dataset301_heart_highres_1559subj"


def options(path):
    path = Path(path).resolve(strict=True)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    paths = ("prepared", "model_folder", "license_home", "work_root", "run_dir", "data_root", "dataset_dir", "result_root")
    integers = {"train_ram_bytes", "prepare_ram_bytes", "prediction_ram_bytes", "cpu_slots",
                "epochs", "batch", "gpu_count"}
    if not isinstance(config, dict) or set(config) != set(paths) | integers | {"gpu_ids", "learning_rate"}:
        raise ValueError("微调配置字段与模板不一致")
    for key in paths:
        if key == "license_home" and config[key] == "auto":
            continue
        config[key] = (path.parent / Path(config[key]).expanduser()).resolve()
    for key in integers:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} 必须是正整数")
    if max(config[key] for key in ("train_ram_bytes", "prepare_ram_bytes")) * 1.5 > 100_000_000_000:
        raise ValueError("训练或准备的内存预算加50%余量后超出100 GB上限")
    if not 1 <= config["gpu_count"] <= min(4, config["batch"]):
        raise ValueError("训练显卡数量须在1到4之间，且全局批量须覆盖每张卡")
    ids = config["gpu_ids"]
    if ids is not None:
        if not isinstance(ids, list) or len(ids) != config["gpu_count"]:
            raise ValueError("gpu_ids 需要列出与 gpu_count 相同数量的显卡标识")
        if any(type(token) not in (int, str) or not str(token).strip() for token in ids):
            raise ValueError("gpu_ids 需要使用显卡索引或 GPU UUID")
        ids = [str(token).strip() for token in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("gpu_ids 不可重复")
        config["gpu_ids"] = ids
    rate = config["learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate 必须是有限正数")
    config["learning_rate"] = float(rate)
    protected = [config["prepared"], config["model_folder"], license_directory(config)]
    require_output(config["run_dir"], config["work_root"], protected)
    require_output(config["dataset_dir"], config["data_root"], protected + [config["run_dir"]])
    return config


def license_directory(config):
    if config["license_home"] != "auto":
        return config["license_home"]
    explicit = os.environ.get("TOTALSEG_HOME_DIR")
    aaoca = os.environ.get("AAOCA_SEG_RUNTIME")
    home = Path.home()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if aaoca:
        candidates.append(Path(aaoca).expanduser() / "private" / "totalseg")
    candidates.extend((home / ".local/state/aaoca-seg-terminal/private/totalseg", home / ".totalsegmentator"))
    return next((path.resolve() for path in candidates if (path / "config.json").is_file()),
                candidates[-1].resolve())


def require_license(config):
    from totalsegmentator.config import has_valid_license_offline, is_valid_license

    directory = license_directory(config)
    license_number = read_json(directory / "config.json").get("license_number")
    if not isinstance(license_number, str) or not license_number:
        raise PermissionError("所选私有配置没有许可证号码")
    previous = os.environ.get("TOTALSEG_HOME_DIR")
    os.environ["TOTALSEG_HOME_DIR"] = str(directory)
    try:
        status, _ = has_valid_license_offline()
    finally:
        if previous is None:
            del os.environ["TOTALSEG_HOME_DIR"]
        else:
            os.environ["TOTALSEG_HOME_DIR"] = previous
    if status != "yes":
        raise PermissionError("官方离线检查未识别到心腔301许可配置")
    with redirect_stdout(StringIO()):
        accepted = is_valid_license(license_number)
    if not accepted:
        raise PermissionError("官方在线许可验证未通过；请检查授权或官方服务连接")


def frozen_cases(config):
    prepared = read_json(config["prepared"])
    if (prepared["dataset"], prepared["scope"], prepared["fold"]) != ("imagechd", "formal", 0):
        raise ValueError("TotalSegmentator 微调需要正式 ImageCHD fold0 冻结描述")
    if prepared["technical"]["roi_xyz"] is not None:
        raise ValueError("正式微调需要完整病例影像")
    require_output(config["dataset_dir"], config["data_root"], [prepared["manifest"]["data_root"]])
    require_output(config["run_dir"], config["work_root"], [prepared["manifest"]["data_root"]])
    groups = {role: [] for role in ("train", "validation", "test")}
    cases = set()
    for row in prepared["manifest"]["cases"]:
        roles = set(row["roles"]) - {"predict"}
        case = row["case_id"]
        if case in cases or len(roles) != 1 or not roles <= groups.keys():
            raise ValueError(f"ImageCHD 冻结病例角色无效 {case}")
        if not case.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"病例编号不适合用于文件名 {case}")
        if row["geometry_policy"] != "matching_affine_author_spacing":
            raise ValueError(f"ImageCHD 病例几何规则不一致 {case}")
        if set(row["unknown_classes"]) not in (set(), {5}):
            raise ValueError(f"ImageCHD 未知类别无效 {case}")
        cases.add(case)
        groups[roles.pop()].append(row)
    if not all(groups.values()):
        raise ValueError("冻结描述缺少训练、验证或测试病例")
    return prepared, groups


def environment(config):
    root = config["dataset_dir"]
    return dict(os.environ, nnUNet_raw=str(root / "raw"),
                nnUNet_preprocessed=str(root / "preprocessed"),
                nnUNet_results=str(config["run_dir"] / "checkpoints"),
                nnUNet_extTrainer=str(Path(__file__).parent), nnUNet_compile="false",
                PYTHONPATH=os.pathsep.join(filter(None, (str(Path(__file__).parents[2]), os.environ.get("PYTHONPATH")))),
                nnUNet_n_proc_DA="0", nnUNet_wandb_enabled="0",
                OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def require_environment():
    if sys.platform != "linux":
        raise RuntimeError("微调阶段需要 Linux 服务器环境")
    for package, pinned in VERSIONS.items():
        if version(package) != pinned:
            raise RuntimeError(f"请使用独立环境中的 {package}=={pinned}")


def initialization(config):
    folder = config["model_folder"]
    if folder.parent.name == LICENSED_301:
        require_license(config)
    elif not folder.parent.name.startswith(tuple(f"Dataset{task}_" for task in PUBLIC_TOTAL_CT)):
        raise ValueError("model_folder 需要指向获授权的心腔301或公开 TotalSegmentator total CT 模型")
    if not folder.name.endswith(f"__{CONFIGURATION}"):
        raise ValueError("model_folder 需要指向含 plans.json、dataset.json 和 fold_0 的三维模型目录")
    files = dict(plans=folder / "plans.json", dataset=folder / "dataset.json",
                 checkpoint=folder / "fold_0" / "checkpoint_final.pth")
    return files, {key: file_digest(path) for key, path in files.items()}


def locations(config):
    folder = config["dataset_dir"]
    return folder / "raw" / DATASET, folder / "preprocessed" / DATASET, (
        config["run_dir"] / "checkpoints" / DATASET / f"{TRAINER}__{PLANS}__{CONFIGURATION}" / "fold_0"
    )


def dataset_definition(config, groups):
    labels = {name: index for index, name in enumerate(LABELS["imagechd"])}
    labels["ignore"] = IGNORE
    return {"channel_names": {"0": "CT"}, "labels": labels, "file_ending": ".nii.gz",
            "numTraining": len(groups["train"]) + len(groups["validation"]),
            "imagechd_total_training": {key: config[key] for key in ("epochs", "learning_rate")}}


def transferred_plans(source, dataset, checkpoint, batch):
    import torch
    from nnunetv2.imageio.reader_writer_registry import recursive_find_reader_writer_by_name
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    original = read_json(source["plans"])
    manager = PlansManager(deepcopy(original))
    spec = manager.get_configuration(CONFIGURATION)
    channels = determine_num_input_channels(manager, spec, read_json(source["dataset"]))
    if channels != 1 or spec.previous_stage_name is not None:
        raise ValueError("TotalSegmentator 源计划需要单通道CT和独立三维网络")
    plan = deepcopy(original)
    plan["dataset_name"] = DATASET
    plan["plans_name"] = PLANS
    configuration = deepcopy(spec.configuration)
    configuration.pop("inherits_from", None)
    configuration.pop("next_stage", None)
    configuration["data_identifier"] = f"{PLANS}_{CONFIGURATION}"
    configuration["batch_size"] = batch
    plan["configurations"] = {CONFIGURATION: configuration}
    # 原始数据按标准 NIfTI 格式导出，使用计划声明的读取器做预检。
    recursive_find_reader_writer_by_name(plan["image_reader_writer"])
    target_manager = PlansManager(deepcopy(plan))
    target_spec = target_manager.get_configuration(CONFIGURATION)
    target = nnUNetTrainer.build_network_architecture(
        target_manager, target_spec, channels, target_manager.get_label_manager(dataset).num_segmentation_heads, True
    )
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if metadata["trainer_name"] != checkpoint.parents[1].name.split("__", maxsplit=1)[0]:
        raise ValueError("源检查点与模型目录声明的训练器不一致")
    if metadata["init_args"]["configuration"] != CONFIGURATION:
        raise ValueError("源检查点的三维配置不一致")
    weights = metadata["network_weights"]
    source_keys = {name for name in weights if ".seg_layers." not in name}
    target_state = {name: value for name, value in target.state_dict().items() if ".seg_layers." not in name}
    if source_keys != target_state.keys():
        raise ValueError(f"源和目标非分割头参数不一致 {sorted(source_keys ^ target_state.keys())}")
    for name, value in target_state.items():
        if weights[name].shape != value.shape:
            raise ValueError(f"源和目标权重形状不一致 {name}")
    return plan


def export_raw(prepared, groups, raw, run, session):
    from ..preprocessing.validation import SourceReader
    for role, rows in groups.items():
        reader = SourceReader(prepared, 4 * 2**30, with_label=role != 'test', log_dir=run)
        for row in rows:
            if session.state.get("monitor_failed") or session.state.get("paused"):
                raise RuntimeError("原始影像导出超过可用资源或资源监控已停止")
            export_case(reader, row, role, raw, run)


def source_identity(config, hashes):
    return {"prepared": file_digest(config["prepared"]), "initialization": hashes,
            "versions": VERSIONS, "dataset": DATASET,
            "raw_export": RAW_EXPORT,
            "training": {key: config[key] for key in ("epochs", "batch", "gpu_count", "learning_rate")},
            "trainer": file_digest(Path(__file__).with_name("total_trainer.py"))}


def prepare(config, configuration_path):
    prepared, groups = frozen_cases(config)
    require_environment()
    source, hashes = initialization(config)
    dataset = dataset_definition(config, groups)
    run = config["run_dir"]
    record = source_identity(config, hashes)
    receipt = run / "prepared.json"
    raw, processed, _ = locations(config)
    with operation_lock(run):
        if receipt.exists():
            if read_json(receipt) != dict(record, plans=verify_preprocessed(config, groups)):
                raise ValueError("已有微调准备记录与当前输入不一致")
            return
        plan = transferred_plans(source, dataset, source["checkpoint"], config["batch"])
        draft = run / "preparing.json"
        if draft.exists() and read_json(draft) != record:
            raise ValueError("已有准备任务使用其他数据或权重")
        if not draft.exists() and (raw.exists() or processed.exists()):
            raise ValueError("微调数据目录已存在且缺少本次准备记录")
        write_json(draft, record)
        with resource(config, "prepare", gpus=0) as session:
            export_raw(prepared, groups, raw, run, session)
            write_json(raw / "dataset.json", dataset)
            write_json(processed / "dataset.json", dataset)
            write_json(processed / f"{PLANS}.json", plan)
            write_json(processed / "splits_final.json", [{"train": case_ids(groups["train"]),
                                                          "val": case_ids(groups["validation"])}])
            write_json(processed / "unknown_classes.json", {row["case_id"]: row["unknown_classes"]
                       for group in ("train", "validation") for row in groups[group]})
            run_child(config, session, "prepare", module_command("_preprocess", configuration_path))
        write_json(receipt, dict(record, plans=verify_preprocessed(config, groups)))
        draft.unlink()


def case_ids(rows):
    return sorted(row["case_id"] for row in rows)


def verify_preprocessed(config, groups):
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    raw, processed, _ = locations(config)
    expected_split = [{"train": case_ids(groups["train"]), "val": case_ids(groups["validation"])}]
    expected_unknown = {row["case_id"]: row["unknown_classes"]
                        for role in ("train", "validation") for row in groups[role]}
    if read_json(processed / "splits_final.json") != expected_split:
        raise ValueError("微调使用的划分与冻结 fold0 不一致")
    if read_json(processed / "unknown_classes.json") != expected_unknown:
        raise ValueError("微调数据的缺失标签声明与冻结描述不一致")
    dataset = dataset_definition(config, groups)
    if any(read_json(folder / "dataset.json") != dataset for folder in (raw, processed)):
        raise ValueError("微调数据的类别定义或训练参数与配置不一致")
    fingerprint = read_json(processed / "dataset_fingerprint.json")
    if (len(fingerprint["spacings"]) != len(groups["train"]) or
            len(fingerprint["shapes_after_crop"]) != len(groups["train"])):
        raise ValueError("数据指纹与冻结训练病例数量不一致")
    case_folder = processed / f"{PLANS}_{CONFIGURATION}"
    dataset_type = infer_dataset_class(str(case_folder))
    identifiers = set(dataset_type.get_identifiers(str(case_folder)))
    if identifiers != set(expected_unknown):
        raise ValueError("nnU-Net 缓存内的病例与冻结训练和验证名单不一致")
    return file_digest(processed / f"{PLANS}.json")


def module_command(action, config_path):
    return [sys.executable, "-B", "-m", "chd_segmentation.finetuning.total", action, "--config", str(config_path)]


def memory_budget(config, stage):
    import nibabel as nib
    import numpy as np
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    prepared = read_json(config["prepared"])
    source = read_json(config["model_folder"] / "plans.json")
    manager = PlansManager(deepcopy(source))
    spacing = np.asarray(manager.get_configuration(CONFIGURATION).spacing)
    transpose = source["transpose_forward"]
    native_max, resampled_max = 0, 0
    for row in prepared["manifest"]["cases"]:
        shape = np.asarray(nib.load(row["image"]).shape)
        target = np.ceil(shape[transpose] * np.asarray(row["spacing_xyz"])[transpose] / spacing) + 2
        native_max = max(native_max, math.prod(int(n) for n in shape))
        resampled_max = max(resampled_max, math.prod(int(n) for n in target))
    key = "prediction_ram_bytes" if stage == "predict" else f"{stage}_ram_bytes"
    # 官方准备和预测包含整例数组，按源网格、重采样网格及类别缓冲预留。
    if stage == "train":
        estimate = 4 * 2**30 + math.prod(manager.get_configuration(CONFIGURATION).patch_size) * config["batch"] * 128
    else:
        estimate = 4 * 2**30 + native_max * 64 + resampled_max * (192 if stage == "predict" else 96)
    amount = math.ceil(max(config[key], estimate) * 1.5)
    if stage != "predict" and amount > 100_000_000_000:
        raise MemoryError("官方整例处理加50%余量后超过100 GB，请选择可容纳该病例的预处理方案")
    from chd_segmentation.training.host import host_capacity
    if amount > int(host_capacity()["total_bytes"] * 0.8):
        raise MemoryError("本阶段含50%余量的预算超过整机80%内存")
    return amount


def resource(config, stage, gpus):
    shares = None
    gpu_ids = None
    if gpus:
        batch = config["batch"] if stage == "train" else 1
        quotient, remainder = divmod(batch, gpus)
        shares = [quotient + (rank < remainder) for rank in range(gpus)]
        if config["gpu_ids"] is not None:
            gpu_ids = config["gpu_ids"][:gpus]
    amount = memory_budget(config, stage)
    demand = ResourceDemand(fixed_bytes=amount, cpu_slots=config["cpu_slots"],
                            gpu_count=gpus, gpu_ids=gpu_ids, local_batches=shares)
    settings = dict(device="cuda" if gpus else "cpu", min_gpus=gpus or None, share=False,
                    max_wait=None, ram_fraction=0.8, cpu_fraction=0.8, memory_limit_bytes=100_000_000_000,
                    precision="fp16" if gpus else "fp32", batch_layout=shares,
                    vram_reserve_bytes=1024**3, vram_reserve_fraction=0.1,
                    peak_margin=1.15, memory_policy="host_fraction" if stage == "predict" else "project_limit")
    event(config["run_dir"], "total_memory_plan", stage=stage, requested_bytes=amount,
          safety_margin=1.5, memory_policy=settings["memory_policy"])
    request = RuntimeRequest(str(config["run_dir"]), f"total_{stage}", config["work_root"] / "resources",
                             config["run_dir"], settings, demand)
    return ResourceSession(request, layout=dict(workers=[0] * (gpus or 1)))


def process_group_members(group_id):
    listing = subprocess.run(["ps", "-eo", "pid=,pgid="], check=True, capture_output=True, text=True)
    groups = [tuple(map(int, line.split())) for line in listing.stdout.splitlines() if line.strip()]
    selected = {pid for pid, group in groups if group == group_id}
    members = []
    for process in psutil.process_iter(("pid", "create_time", "status")):
        info = process.info
        if (info["pid"] not in selected or info["create_time"] is None or
                info["status"] in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)):
            continue
        members.append(dict(pid=info["pid"], birth=info["create_time"]))
    return members


def stop_process_group(process):
    deadline = monotonic() + 10
    for action in (signal.SIGTERM, signal.SIGKILL):
        process.poll()
        if not process_group_members(process.pid):
            break
        result = subprocess.run(["pkill", "--signal", str(int(action)), "--pgroup", str(process.pid)])
        if result.returncode not in (0, 1):
            raise subprocess.CalledProcessError(result.returncode, result.args)
        while process_group_members(process.pid) and (action == signal.SIGKILL or monotonic() < deadline):
            process.poll()
            sleep(0.2)
    process.wait()


def run_child(config, session, stage, command):
    env = environment(config)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu["uuid"] for gpu in session.allocation.gpus)
    log = config["run_dir"] / f"{stage}.log"
    with log.open("ab") as stream, subprocess.Popen(
        command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
    ) as process:
        try:
            while process.poll() is None:
                for member in process_group_members(process.pid):
                    pid = member["pid"]
                    if session.handle.members.get(pid) != member:
                        session.handle.members[pid] = member
                if session.state.get("monitor_failed") or session.state.get("paused"):
                    raise RuntimeError("微调进程超过资源预算或资源监控已停止")
                sleep(0.5)
        finally:
            stop_process_group(process)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def train(config, resume):
    import torch

    _, groups = frozen_cases(config)
    require_environment()
    source, hashes = initialization(config)
    run = config["run_dir"]
    _, _, fold_path = locations(config)
    with operation_lock(run):
        plan_digest = verify_preprocessed(config, groups)
        prepared = dict(source_identity(config, hashes), plans=plan_digest)
        if read_json(run / "prepared.json") != prepared:
            raise ValueError("微调输入已变化，请使用相符的准备记录")
        state_path = run / "training.json"
        state = {"prepared": prepared, "plans": plan_digest}
        checkpoints = [fold_path / name for name in CHECKPOINTS if (fold_path / name).is_file()]
        if resume:
            if read_json(state_path) != state or not checkpoints:
                raise ValueError("缺少与本次微调匹配的恢复检查点")
            checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            init_args = checkpoint["init_args"]
            if (checkpoint["trainer_name"] != TRAINER or init_args["fold"] != 0 or
                    init_args["configuration"] != CONFIGURATION or
                    init_args["dataset_json"] != dataset_definition(config, groups)):
                raise ValueError("恢复检查点的训练器、折号或数据配置不一致")
            if not (0 < checkpoint["current_epoch"] <= config["epochs"]) or not checkpoint["optimizer_state"]:
                raise ValueError("恢复检查点不含有效的训练状态")
            if checkpoints[0].name == CHECKPOINTS[0] and checkpoint["current_epoch"] == config["epochs"]:
                return
        elif checkpoints:
            raise ValueError("已有微调检查点，请显式选择 --resume")
        elif state_path.exists() and read_json(state_path) != state:
            raise ValueError("运行目录已有其他微调任务")
        write_json(state_path, state)
        command = [sys.executable, "-B", "-m", "nnunetv2.run.run_training", DATASET, CONFIGURATION,
                   "0", "-p", PLANS, "-tr", TRAINER, "-num_gpus", str(config["gpu_count"])]
        command.extend(["--c"] if resume else ["-pretrained_weights", str(source["checkpoint"])])
        with resource(config, "train", gpus=config["gpu_count"]) as session:
            run_child(config, session, "train", command)
        if not (fold_path / CHECKPOINTS[0]).is_file():
            raise RuntimeError("训练进程退出后缺少最终检查点")


def preprocess_child(config):
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.plan_and_preprocess_api import preprocess

    _, processed, _ = locations(config)
    fingerprint = DatasetFingerprintExtractor(DATASET, num_processes=1)
    # 指纹只记录训练病例特征，CT 归一化继续使用源计划的统计。
    training_keys = read_json(processed / "splits_final.json")[0]["train"]
    fingerprint.dataset = {key: fingerprint.dataset[key] for key in training_keys}
    fingerprint.run(overwrite_existing=True)
    preprocess([901], PLANS, configurations=[CONFIGURATION], num_processes=[1])


def prediction_directory(config, prepared, role, choice):
    path = config["result_root"] / "ImageCHD-TotalSegmentator-fold0" / f"{role}-{choice}"
    return require_output(path, config["result_root"], [prepared["manifest"]["data_root"],
                          config["dataset_dir"], config["model_folder"], config["run_dir"]])


def prediction_checkpoint(config, groups, choice):
    import torch

    _, _, fold_path = locations(config)
    checkpoint_path = fold_path / f"checkpoint_{choice}.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    init_args = checkpoint["init_args"]
    if (checkpoint["trainer_name"] != TRAINER or init_args["fold"] != 0 or
            init_args["dataset_json"] != dataset_definition(config, groups)):
        raise ValueError("预测检查点与当前冻结任务不一致")
    return checkpoint_path


def predict(config, config_path, role, choice):
    prepared, groups = frozen_cases(config)
    require_environment()
    run = config["run_dir"]
    with operation_lock(run):
        _, hashes = initialization(config)
        plan_digest = verify_preprocessed(config, groups)
        state = dict(source_identity(config, hashes), plans=plan_digest)
        if read_json(run / "training.json") != {"prepared": state, "plans": plan_digest}:
            raise ValueError("训练记录与当前冻结数据或初始化权重不一致")
        prediction_checkpoint(config, groups, choice)
        output = prediction_directory(config, prepared, role, choice)
        with operation_lock(output), resource(config, "predict", gpus=1) as session:
            command = module_command("_predict", config_path)
            run_child(config, session, "predict", command + ["--role", role, "--checkpoint", choice])


def predict_child(config, role, choice):
    import nibabel as nib
    import numpy as np
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    from chd_segmentation.prediction.metrics import case_data_identity

    prepared, groups = frozen_cases(config)
    checkpoint_path = prediction_checkpoint(config, groups, choice)
    checkpoint_sha256 = file_digest(checkpoint_path)
    output = prediction_directory(config, prepared, role, choice)
    receipt = output / "predictions.json"
    raw, _, _ = locations(config)
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                 perform_everything_on_device=False, device=torch.device("cuda", 0))
    predictor.initialize_from_trained_model_folder(str(checkpoint_path.parents[1]), use_folds=(0,),
                                                  checkpoint_name=checkpoint_path.name)
    reader = predictor.plans_manager.image_reader_writer_class()
    previous = read_json(receipt) if receipt.exists() else {"checkpoint": checkpoint_sha256, "cases": {}}
    if previous["checkpoint"] != checkpoint_sha256:
        raise ValueError("目标结果目录已有其他检查点的预测")
    cases = previous["cases"]
    staging = config["run_dir"] / "prediction_work"
    staging.mkdir(exist_ok=True)
    for row in groups[role]:
        name = row["case_id"]
        source_path = Path(row["image"])
        stat = source_path.stat()
        if [stat.st_size, stat.st_mtime_ns] != row["source_stamp"]["image"]:
            raise ValueError(f"预测源影像已变化 {name}")
        destination = output / f"{name}.nii.gz"
        case_identity = case_data_identity(prepared, row)
        if name in cases:
            saved = cases[name]
            if (saved["data_identity"] != case_identity or saved["file"] != destination.name or
                    file_digest(destination) != saved["identity"]):
                raise ValueError(f"已有预测及其记录不一致 {name}")
            continue
        source = nib.load(source_path)
        model_input = raw / ("imagesTs" if role == "test" else "imagesTr") / f"{name}_0000.nii.gz"
        data, properties = reader.read_images((str(model_input),))
        segmentation = predictor.predict_single_npy_array(data, properties)
        del data
        pending = staging / f"{name}.nii.gz"
        reader.write_seg(segmentation, str(pending), properties)
        del segmentation
        predicted = nib.load(pending)
        model_grid = nib.load(model_input)
        if predicted.shape != source.shape or not np.allclose(predicted.affine, model_grid.affine, atol=1e-4, rtol=0):
            raise ValueError(f"输出与预测输入几何不一致 {name}")
        values = np.asanyarray(predicted.dataobj)
        if not np.isin(values, range(8)).all():
            raise ValueError(f"输出标签超出 ImageCHD 类别范围 {name}")
        native = nib.Nifti1Image(values.astype(np.uint8), source.affine, source.header)
        native.set_data_dtype(np.uint8)
        # 先在结果目录保存完整文件，再原子提交到正式名称。
        output_pending = output / f"{name}.pending.nii.gz"
        nib.save(native, output_pending)
        os.replace(output_pending, destination)
        pending.unlink()
        cases[name] = dict(file=destination.name, scope=prepared["scope"],
                           roi_xyz=[[0, int(size)] for size in source.shape],
                           data_identity=case_identity, identity=file_digest(destination),
                           checkpoint_sha256=checkpoint_sha256)
        write_json(receipt, previous)
        event(config["run_dir"], "total_prediction_case", case_id=name, role=role)
    staging.rmdir()


def status(config):
    run = config["run_dir"]
    _, _, fold_path = locations(config)
    source = config["model_folder"]
    return {"prepared": (run / "prepared.json").is_file(), "run_dir": str(run),
            "source_prepared_present": config["prepared"].is_file(),
            "license_directory": str(license_directory(config)),
            "license_config_present": (license_directory(config) / "config.json").is_file(),
            "model_folder": str(source),
            "source_model_files_present": all((source / name).is_file() for name in (
                "plans.json", "dataset.json", "fold_0/checkpoint_final.pth")),
            "checkpoints": {name: str(fold_path / name) for name in CHECKPOINTS if (fold_path / name).is_file()}}


def request_stop(signum, frame):
    raise InterruptedError("收到终止请求，正在结束本任务子进程")


def main(argv=None):
    signal.signal(signal.SIGTERM, request_stop)
    parser = argparse.ArgumentParser(description="ImageCHD fold0 TotalSegmentator CT 模型微调")
    parser.add_argument("action", choices=("prepare", "train", "predict", "status", "check", "_preprocess", "_predict"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--role", choices=("validation", "test"), default="test")
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    args = parser.parse_args(argv)
    config = options(args.config)
    if args.resume and args.action != "train":
        parser.error("--resume 仅适用于 train")
    if args.action == "check":
        require_environment()
        initialization(config)
        print(json.dumps(dict(status(config), source_check="passed"), ensure_ascii=False, indent=2))
    elif args.action == "prepare":
        prepare(config, args.config.resolve())
    elif args.action == "train":
        train(config, args.resume)
    elif args.action == "predict":
        predict(config, args.config.resolve(), args.role, args.checkpoint)
    elif args.action == "_preprocess":
        preprocess_child(config)
    elif args.action == "_predict":
        predict_child(config, args.role, args.checkpoint)
    else:
        print(json.dumps(status(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
