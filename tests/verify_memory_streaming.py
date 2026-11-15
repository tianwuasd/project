import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter, sleep

import psutil


def wait_memory(required):
    last = 0.
    while True:
        facts = psutil.virtual_memory()
        spare = int(facts.available - facts.total * .05)
        if spare >= required:
            return
        if perf_counter() - last >= 15:
            print(json.dumps({"stage": "waiting_local_memory", "headroom95_bytes": spare,
                              "required_additional_bytes": required}), flush=True)
            last = perf_counter()
        sleep(1)


def run(args):
    wait_memory(2 * 1024 ** 3)
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from smoke import SmokeWorkspace
    from chd_segmentation.artifacts.store import digest, read_json, write_json
    from chd_segmentation.config import DEFAULTS, load_config
    from chd_segmentation.models.stunet.preprocess import StunetTransform
    from chd_segmentation.models.stunet.streaming import stream_case
    from chd_segmentation.preprocessing.offline import preprocess
    from chd_segmentation.preprocessing.offline_store import begin_case, case_directory, case_key, recipe, save_case, owned_remove
    from chd_segmentation.preprocessing.prepare import prepare
    from chd_segmentation.training.engine import train
    from chd_segmentation.training.memory_guard import MemoryGuard

    with SmokeWorkspace(args.work_root, args.report) as workspace:
        root = workspace.root
        rows = [deepcopy(r) for r in read_json(args.manifest)["cases"] if r["case_id"] in ("ct_1003", "ct_1141")]
        if len(rows) != 2:
            raise ValueError("必须且只使用两个既定真实病例")
        for row in rows:
            row["roles"] = ["train"] if row["case_id"] == "ct_1003" else ["validation"]
            for kind in ("image", "label"):
                row[kind] = str((Path(args.data_root) / row[kind]).resolve(strict=True))
        write_json(root / "manifest.json", {"dataset": "imagechd", "scope": "technical", "data_root": args.data_root, "cases": rows})
        config = deepcopy(DEFAULTS)
        config.update(dataset="imagechd", scope="technical", manifest=str(root / "manifest.json"), pretrained=args.pretrained,
                      work_root=str(root / "course"), run_dir=str(root / "course" / "ImageCHD-STUNet"), result_root=str(root / "result"))
        config["preprocessing"].update(root=str(root / "data"), block_bytes=2 * 1024 ** 2)
        config["training"].update(patch=[32] * 3, batch=4, epochs=2, steps_per_epoch=1, validation_steps=1, save_every_attempts=1)
        config["planning"].update(target_spacing_xyz=[.25, .25, .5], samples_per_case=1000, max_samples=2000)
        config["technical"].update(roi_xyz=[[240, 272], [240, 272], [87, 119]])
        config["resources"].update(cpu_processes=2, workers=0, max_workers=2, readers=1, max_readers=1,
                                   reserve_ram_bytes=1024 ** 3, cache_bytes=1024 ** 2, ram_fraction=.95)
        write_json(root / "config.json", config)
        config = load_config(root / "config.json")
        info = {"cases": [r["case_id"] for r in rows], "roi_xyz": config["technical"]["roi_xyz"],
                "numeric_comparison": [], "optimizer_update_limit": 2, "gpu_training": False,
                "local_technical_ram_fraction": .95, "formal_ram_fraction": .8}
        workspace.results.append(info)
        began = perf_counter()
        source_path = root / "course" / "source.json"
        source = prepare(config, source_path)
        info["plan_seconds"] = perf_counter() - began
        for separate in (False, True):
            prepared = deepcopy(source)
            cfg = prepared["plan"]["configurations"]["3d_fullres"]
            for kind in ("data", "seg"):
                cfg[f"resampling_fn_{kind}_kwargs"]["force_separate_z"] = separate
            identity = digest(recipe(prepared, config))
            with MemoryGuard(config, "numerical_check") as guard:
                for row in prepared["manifest"]["cases"]:
                    key = case_key(row, identity)
                    directory = root / "comparison" / str(separate) / key[:16]
                    pending = begin_case(directory, key, identity)
                    with guard.admit(32 * 1024 ** 2, label="reference_32cube") as ready:
                        if ready is None:
                            raise MemoryError("真实数值比较等待内存余量")
                        before = StunetTransform(prepared, config["resources"]["reserve_ram_bytes"])(row)
                    mark = perf_counter()
                    after, blocks = stream_case(prepared, row, config, pending, guard)
                    delta = float(np.max(np.abs(before["image"] - after["image"])))
                    np.testing.assert_allclose(before["image"], after["image"], rtol=0, atol=1e-6)
                    np.testing.assert_array_equal(before["label"], after["label"])
                    for name in ("source_shape_xyz", "source_affine", "working_affine", "spacing_xyz", "roi_xyz"):
                        np.testing.assert_equal(before[name], after[name])
                    for name in ("shape_before_cropping", "shape_after_cropping_and_before_resampling", "bbox_used_for_cropping"):
                        np.testing.assert_equal(before["properties"][name], after["properties"][name])
                    for label, locations in before["properties"]["class_locations"].items():
                        np.testing.assert_array_equal(locations, after["properties"]["class_locations"][label])
                    info["numeric_comparison"].append({"case_id": row["case_id"], "separate_z": separate,
                        "max_abs_difference": delta, "image_bitwise_equal": bool(np.array_equal(before["image"], after["image"])),
                        "label_equal": True, "geometry_equal": True, "locations_equal": True, "seconds": perf_counter() - mark})
                    try:
                        save_case(directory, key, identity, after, blocks)
                    finally:
                        blocks.release_handles()
                    del before, after
                    owned_remove(directory)
        began = perf_counter()
        ready_path = root / "course" / "ready.json"
        prepared = preprocess(config, source_path, ready_path)
        info["preprocess_seconds"] = perf_counter() - began
        before = {str(p): p.stat().st_mtime_ns for p in Path(prepared["storage"]["root"]).rglob("*") if p.is_file() and p.name != "catalog.json"}
        repeated = preprocess(config, source_path, ready_path)
        assert repeated["identity"] == prepared["identity"]
        assert all(Path(p).stat().st_mtime_ns == value for p, value in before.items())
        progress = read_json(Path(config["run_dir"]) / "progress-preprocess.json")
        assert progress["done"] == progress["total"] == progress["reused"] == 2 and progress["new"] == 0
        info["reused_without_array_writes"] = True
        if not args.numeric_only:
            wait_memory(4 * 1024 ** 3)
            mark = perf_counter()
            first = train(config, ready_path, stop_after=1)
            second = train(config, ready_path, resume=True)
            assert first["updates"] == 1 and second["updates"] == 2 and second["restored_exact"]
            info.update(training_seconds=perf_counter() - mark, first=first, resume=second)
        log_path = Path(config["run_dir"]) / "events.jsonl"
        shutil.copyfile(log_path, workspace.report / "events.jsonl")
        write_json(workspace.report / "streaming.json", info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("manifest", "data-root", "pretrained", "work-root", "report"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--numeric-only", action="store_true")
    run(parser.parse_args())
