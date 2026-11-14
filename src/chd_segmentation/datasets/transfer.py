from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import yaml

from ..artifacts.store import digest, file_digest, read_json, write_json


def manifest_identity(manifest):
    return digest({"dataset": manifest["dataset"], "scope": manifest["scope"],
                   "cases": [{k: v for k, v in row.items() if k not in ("image", "label", "source_stamp")}
                             for row in manifest["cases"]]})


def write_changed(path, value):
    if Path(path).is_file() and read_json(path) == value:
        return False
    write_json(path, value)
    return True


def bind_inputs(directory, root, dataset_roots=None, pretrained=None):
    directory, root = Path(directory).resolve(strict=True), Path(root).resolve(strict=True)
    descriptor = read_json(directory / "bundle.json")
    if digest({k: v for k, v in descriptor.items() if k != "identity"}) != descriptor["identity"]:
        raise ValueError("迁移包描述内容摘要不一致")
    for name, expected in descriptor["immutable_files"].items():
        if Path(name).name != name or file_digest(directory / name) != expected:
            raise ValueError("迁移包中的冻结划分或政策摘要不一致")
    roots = {"chd1282": root / "data" / "dataset1282_CHD_CT_back", "imagechd": root / "data" / "ImageCHD_dataset"}
    roots.update(dataset_roots or {})
    ready, indexes = {}, {}
    for dataset, info in descriptor["datasets"].items():
        manifest = read_json(directory / info["manifest"])
        if manifest_identity(manifest) != info["identity"]:
            raise ValueError("迁移清单的病例事实与打包身份不同")
        data_root = Path(roots[dataset]).resolve(strict=True)
        for row in manifest["cases"]:
            for kind in ("image", "label"):
                direct = (data_root / row["preferred_files"][kind]).resolve()
                if direct.is_relative_to(data_root) and direct.is_file():
                    path = direct
                else:
                    if dataset not in indexes:
                        lookup = defaultdict(list)
                        for candidate in data_root.rglob("*.nii*"):
                            if candidate.is_file() and candidate.name.endswith((".nii", ".nii.gz")):
                                lookup[candidate.name].append(candidate.resolve())
                        indexes[dataset] = lookup
                    found = {p for alias in row["file_aliases"][kind] for p in indexes[dataset].get(alias, [])}
                    if len(found) != 1:
                        raise ValueError(f"病例 {row['case_id']} 的 {kind} 文件缺失或匹配不唯一；请核对数据根目录")
                    path = found.pop()
                if not path.is_relative_to(data_root):
                    raise ValueError("病例文件实际位置超出指定数据目录")
                expected_bytes = row.get("source_bytes", {}).get(kind)
                if expected_bytes is not None and path.stat().st_size != expected_bytes:
                    raise ValueError(f"病例 {row['case_id']} 的 {kind} 文件大小与审计记录不同")
                row[kind] = path.relative_to(data_root).as_posix()
        manifest.update(data_root=str(data_root), binding_required=False)
        ready[dataset] = manifest
    weight = Path(pretrained).resolve(strict=True) if pretrained else root / descriptor["weight_relative"]
    if not weight.is_file():
        raise FileNotFoundError("缺少既有 STU-Net 权重：" + str(weight))
    course, results = root / "course" / "chd-segmentation", root / "result" / "chd-segmentation"
    changed = []
    # 全部文件名成功解析后才写出绑定结果；元数据身份不包含可迁移路径。
    for dataset, manifest in ready.items():
        name = descriptor["datasets"][dataset]["manifest"]
        if write_changed(directory / name, manifest):
            changed.append(name)
    for experiment in descriptor["experiments"]:
        config = deepcopy(experiment["config"])
        dataset = config["dataset"]
        path = directory / experiment["filename"]
        if path.is_file():
            previous = yaml.safe_load(path.read_text(encoding="utf-8"))
            if previous.get("dataset") != dataset or previous.get("fold") != config["fold"]:
                raise ValueError("已有配置的数据集或折数与迁移包不同")
            config = previous
        config.update(manifest=str(directory / descriptor["datasets"][dataset]["manifest"]),
                      pretrained=str(weight), work_root=str(course), result_root=str(results),
                      run_dir=str(course / "runs" / (experiment["run_name"] + "_" + descriptor["identity"][:8])))
        if config.get("plan") is None:
            config["plan"] = str(course / "plans" / dataset / descriptor["identity"][:8] /
                                 f"fold_{config['fold']}" / "preprocessing_plan.json")
        config["split"]["path"] = str(directory / descriptor["datasets"][dataset]["split"])
        name = experiment["filename"]
        encoded = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            pending = path.with_name(path.name + ".pending")
            pending.write_text(encoded, encoding="utf-8", newline="\n")
            pending.replace(path)
            changed.append(name)
    receipt = {"state": "bound", "bundle_identity": descriptor["identity"], "root": str(root),
               "datasets": {d: {"root": str(ready[d]["data_root"]), "cases": len(ready[d]["cases"])} for d in ready},
               "weight": str(weight), "scope": "文件名、大小与冻结元数据；原始 SHA256 在实际读取病例时核验"}
    write_changed(directory / "binding.json", receipt)
    return {**receipt, "changed_files": changed}
