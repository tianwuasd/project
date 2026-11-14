import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter
from unittest import TestCase

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smoke import SmokeWorkspace, run
from chd_segmentation.artifacts.store import file_digest, write_json
from chd_segmentation.config import load_config
from chd_segmentation.datasets.manifest import load_manifest
from chd_segmentation.preprocessing.planning import plan_request
from chd_segmentation.preprocessing.prepare import prepare, validate_prepared_config


def verify(spec, root, report):
    check = TestCase()
    experiments = yaml.safe_load(Path(spec).read_text(encoding="utf-8"))["experiments"]
    if not 1 <= len(experiments) <= 2:
        raise ValueError("只允许一至两个真实病例")
    with SmokeWorkspace(root, report) as workspace:
        for experiment in experiments:
            mark = perf_counter()
            directory = workspace.root / experiment["dataset"]
            row = deepcopy(experiment["case"])
            row["roles"] = ["train", "validation"]
            manifest = {"dataset": experiment["dataset"], "scope": "technical",
                        "data_root": experiment["data_root"], "cases": [row]}
            write_json(directory / "manifest.json", manifest)
            source = {"dataset": experiment["dataset"], "scope": "technical",
                      "manifest": str(directory / "manifest.json"), "pretrained": experiment["pretrained"],
                      "work_root": str(directory), "run_dir": str(directory / "runs"),
                      "technical": {"roi_xyz": experiment["roi_xyz"], "allow_role_overlap": True},
                      "planning": {"samples_per_case": 2000, "max_samples": 2000}}
            write_json(directory / "config.json", source)
            config = load_config(directory / "config.json")
            prepared_path = directory / "prepared.json"
            first = prepare(config, prepared_path)
            check.assertEqual(first["plan"]["fit_ids"], [row["case_id"]])
            check.assertEqual(first["plan"]["scope"], "technical")
            check.assertEqual(first["plan"]["generation"]["total_samples"], 2000)
            validate_prepared_config(first, config)
            timestamp = Path(config["plan"]).stat().st_mtime_ns
            check.assertEqual(prepare(config, prepared_path), first)
            check.assertEqual(Path(config["plan"]).stat().st_mtime_ns, timestamp)
            checks = ["default_path", "training_fit_ids", "technical_scope", "sample_cap", "reuse_without_write"]
            for key, value in (("fold", 1), ("scope", "formal")):
                changed = deepcopy(config)
                changed[key] = value
                with check.assertRaises(ValueError):
                    prepare(changed, directory / f"rejected-{key}.json")
                checks.append("reject_" + key)
            changed = deepcopy(config)
            changed["planning"]["seed"] += 1
            with check.assertRaises(ValueError):
                prepare(changed, prepared_path)
            with check.assertRaises(ValueError):
                validate_prepared_config(first, changed)
            checks.append("reject_changed_settings_on_prepare_and_train")
            loaded = load_manifest(config["manifest"], config["dataset"], "technical")
            loaded["cases"][0]["roles"] = ["validation"]
            with check.assertRaises(ValueError):
                plan_request(config, loaded)
            checks.append("reject_validation_only_fitting")
            damaged = deepcopy(first["plan"])
            damaged["configurations"]["3d_fullres"]["spacing"][0] *= 2
            write_json(config["plan"], damaged)
            with check.assertRaises(ValueError):
                prepare(config, prepared_path)
            write_json(config["plan"], first["plan"])
            checks.append("reject_modified_plan")
            changed_manifest = deepcopy(manifest)
            changed_manifest["cases"][0]["unknown_classes"] = []
            write_json(config["manifest"], changed_manifest)
            with check.assertRaises(ValueError):
                prepare(config, prepared_path)
            write_json(config["manifest"], manifest)
            checks.append("reject_changed_training_metadata")
            override = deepcopy(config)
            override["plan"] = str(directory / "override" / "plan.json")
            override["planning"].update({"target_spacing_xyz": [1., 1., 1.], "clip_range": [0., 4095.],
                                         "mean": 1000., "std": 200.})
            altered = prepare(override, directory / "override-prepared.json")["plan"]
            check.assertEqual(altered["configurations"]["3d_fullres"]["spacing"], [1., 1., 1.])
            check.assertEqual(altered["foreground_intensity_properties_per_channel"]["0"]["mean"], 1000.)
            check.assertEqual(altered["foreground_intensity_properties_per_channel"]["0"]["std"], 200.)
            check.assertEqual(altered["foreground_intensity_properties_per_channel"]["0"]["percentile_00_5"], 0.)
            checks.append("manual_spacing_clip_mean_std")
            existing = deepcopy(config)
            existing["planning"]["mode"] = "existing"
            check.assertEqual(prepare(existing, directory / "existing-prepared.json")["plan"], first["plan"])
            checks.append("explicit_existing_plan")
            invalid = deepcopy(source)
            invalid["planning"]["target_spacing_xyz"] = [0, 1, 1]
            write_json(directory / "invalid.json", invalid)
            with check.assertRaises(ValueError):
                load_config(directory / "invalid.json")
            checks.append("reject_invalid_override")
            for kind in ("image", "label"):
                check.assertEqual(file_digest(row[kind]), row["sha256"][kind])
            workspace.results.append({"dataset": config["dataset"], "case_id": row["case_id"], "checks": checks,
                                      "scope": "technical", "seconds": perf_counter() - mark,
                                      "source_unchanged": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="两例真实数据的自动计划与完整 STU-Net 短测")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    root, report = Path(args.work_root), Path(args.report)
    verify(args.spec, root, report / "planning")
    run(args.spec, root, report / "pipeline")
