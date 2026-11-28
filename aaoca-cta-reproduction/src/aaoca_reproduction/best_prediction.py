from pathlib import Path

from .records import event, read_json


def predict_best(config, manifest, output=None, device="cpu", check=False):
    settings = config["classification"]
    if config.get("run_purpose") == "smoke" or settings.get("smoke_input_shape"):
        raise ValueError("本入口用于正式独立预测，不接受小测配置")
    if device != "cpu" and not (device.startswith("cuda:") and device[5:].isdigit()):
        raise ValueError("device须为cpu或cuda:非负整数")
    run = Path(settings["run_directory"]).resolve(strict=True)
    best = (run / "best.pt").resolve(strict=True)
    record = read_json(run / "run.json")
    if record["model"]["task"] != settings["task"]:
        raise ValueError("训练记录与预测任务不符")
    weights = {key: record["model"][key] for key in ("task", "labels", "author_release")}
    weights.update(path=str(best), source="local_run_best_validation_loss", format="local_complete_checkpoint",
                   strategy=record["settings"]["strategy"], member_index=record["settings"]["member_index"])
    candidates = {row["case_id"]: row for row in manifest["cases"] if row["role"] == "predict"}
    selected = settings["prediction_case_ids"]
    if not selected or len(selected) != len(set(selected)) or set(selected) - candidates.keys():
        raise ValueError("预测名单必须为原冻结清单的独立保留病例且不重复")
    forbidden = set(record["train_patients"]) | set(record["validation_patients"])
    if any(candidates[key]["patient_key"] in forbidden for key in selected):
        raise ValueError("独立预测患者与本模型训练或验证患者重叠")
    requested_output = Path(output) if output else run / "prediction-best"
    if requested_output.is_symlink():
        raise ValueError("预测目录不能是软链接")
    directory = requested_output.resolve()
    course = Path(config["course_root"]).resolve()
    if directory == course or not directory.is_relative_to(course):
        raise ValueError("预测结果必须位于当前任务course_root的独立目录")
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise FileExistsError("预测目录已有结果，请保留并另行指定output")
    settings.update(device=device, gpu_count=1, ensemble_mode="single_model",
                    weights=[weights], prediction_directory=str(directory))
    settings.pop("gpu_ids", None)
    plan = {"best": str(best), "best_bytes": best.stat().st_size, "requested_cases": len(selected),
            "output": str(directory), "device": device, "check_only": check}
    if check:
        return plan
    event("classification_best_prediction_planned", **plan)
    from .prediction import predict
    result = predict(config, manifest)
    return {"output": str(directory / "predictions.json"), "metrics": result["metrics"],
            "unknown_or_inapplicable": result["unknown_or_inapplicable"],
            "skipped": len(result["localization_exclusions"]), "seconds": result["seconds"]}
